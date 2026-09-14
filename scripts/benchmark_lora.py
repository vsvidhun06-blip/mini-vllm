"""
Benchmark: base model vs single-LoRA vs mixed 4-adapter batch.

Run:
    python scripts/benchmark_lora.py

Needs TinyLlama in the HF cache (run `python -m src.engine.model` once). Runs on
whatever DEVICE the engine picks (GPU if present, else CPU).

WHAT WE MEASURE
---------------
The cost of the LoRA delta relative to the base forward, on an identical input:

  * base            -- no adapter active (LoRALinear's zero-overhead path).
  * single-LoRA     -- one rank-r adapter applied to every row.
  * mixed 4-adapter -- a batch whose rows are routed to 4 DIFFERENT adapters
                       (the multi-tenant serving case), exercising per-row
                       grouping + scatter.

The base GEMM is shared across all three; the only difference is the two skinny
LoRA matmuls (x@A^T then @B^T) plus, for the mixed case, the per-adapter
grouping. Expectation: < 5% overhead for a rank-16 adapter on TinyLlama -- the
delta is O(r) extra work against an O(d) base, and r << d.

Table: config | latency (ms) | overhead vs base | tokens/sec
"""
from __future__ import annotations

import time

import torch

from src.engine.device import DEVICE
from src.engine.lora import LoRAManager
from src.engine.lora_model import LoRALlamaModel, random_adapter_weights
from src.engine.model import MODEL_NAME, load_tinyllama_from_hf

BATCH = 4
SEQ = 32
RANK = 16
ALPHA = 32.0
WARMUP = 20
ITERS = 100


@torch.no_grad()
def _time_forward(model, ids, adapter_ids, iters: int) -> float:
    """Mean wall-clock seconds per forward, with CUDA sync if applicable."""
    is_cuda = next(model.model.parameters()).is_cuda if hasattr(model, "model") \
        else next(model.parameters()).is_cuda
    # Warmup.
    for _ in range(WARMUP):
        model.forward(ids, adapter_ids=adapter_ids)
    if is_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        model.forward(ids, adapter_ids=adapter_ids)
    if is_cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def main() -> None:
    model, _ = load_tinyllama_from_hf(MODEL_NAME, dtype=torch.float32)
    model.eval()

    manager = LoRAManager(max_adapters=8)
    lora_model = LoRALlamaModel(model, manager)

    # Register 4 distinct rank-16 adapters spanning every attention projection.
    for i in range(4):
        manager.load_adapter(
            f"adp{i}", rank=RANK, alpha=ALPHA,
            weights_dict=random_adapter_weights(lora_model, rank=RANK, seed=i),
        )

    g = torch.Generator().manual_seed(123)
    ids = torch.randint(0, model.config.vocab_size, (BATCH, SEQ), generator=g)

    print(f"Device: {DEVICE}")
    print(f"Workload: batch={BATCH}, seq={SEQ}, rank={RANK}, "
          f"{ITERS} timed iters (+{WARMUP} warmup)\n")

    base_s   = _time_forward(lora_model, ids, None, ITERS)
    single_s = _time_forward(lora_model, ids, "adp0", ITERS)
    mixed_s  = _time_forward(lora_model, ids, ["adp0", "adp1", "adp2", "adp3"], ITERS)

    tokens = BATCH * SEQ
    rows = [
        ("base (no LoRA)",   base_s,   0.0),
        ("single LoRA",      single_s, (single_s / base_s - 1.0) * 100.0),
        ("mixed 4-adapter",  mixed_s,  (mixed_s / base_s - 1.0) * 100.0),
    ]

    header = f"{'config':>18} | {'latency (ms)':>12} | {'overhead':>9} | {'tokens/sec':>11}"
    print(header)
    print("-" * len(header))
    for name, secs, overhead in rows:
        ov = "—" if name.startswith("base") else f"{overhead:+.1f}%"
        print(f"{name:>18} | {secs * 1e3:>12.3f} | {ov:>9} | {tokens / secs:>11.1f}")

    worst = max((single_s / base_s - 1.0), (mixed_s / base_s - 1.0)) * 100.0
    verdict = "within" if worst < 5.0 else "ABOVE"
    print(f"\nWorst-case LoRA overhead: {worst:+.1f}% -- {verdict} the <5% target "
          f"for rank-{RANK} on TinyLlama.")
    print("The base GEMM is shared; only the two skinny r-rank matmuls (and, for")
    print("the mixed case, per-adapter grouping) are extra -- O(r) over O(d), r<<d.")


if __name__ == "__main__":
    main()
