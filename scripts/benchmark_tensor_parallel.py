"""
Benchmark: tensor parallelism at world_size 1 vs 2 vs 4 (single-device simulation).

Run:
    python scripts/benchmark_tensor_parallel.py

Needs TinyLlama in the HF cache (run `python -m src.engine.model` once). Runs on
whatever DEVICE the engine picks (GPU if present, else CPU).

HONESTY UP FRONT
----------------
world_size > 1 is SIMULATED in a single process on a single device: the N rank
shards are computed sequentially and their row-parallel partials summed (the
in-process stand-in for the all-reduce). On one device this does the SAME total
FLOPs as dense, just split into N smaller matmuls plus a sum -- so it is NOT
faster, and typically a little SLOWER (more, smaller kernels + the partial sum).
The point of this benchmark is to measure that simulation overhead and exercise
the sharding/communication STRUCTURE, not to show a speedup.

  REAL speedup requires N PHYSICAL GPUs, one rank per process: each GPU does 1/N
  of every matmul in parallel and the all-reduce overlaps compute. On a single
  GPU you can overlap the rank shards across CUDA streams, but the FLOPs are
  still serialised on the one device's compute units -- still no real win.

For comparability every world_size uses the SAME greedy no-cache forward (the
path the TP simulation supports), so the numbers differ only by the sharding.

Table: world_size | TTFT (ms) | TPOT (ms) | throughput (tok/s)
"""
from __future__ import annotations

import time

import torch

from src.engine.device import DEVICE
from src.engine.model import MODEL_NAME, load_tinyllama_from_hf
from src.engine.tp_model import TensorParallelLlamaModel

PROMPT_LEN = 16
NEW_TOKENS = 16
WORLD_SIZES = (1, 2, 4)


@torch.no_grad()
def _time_generation(tp_model, ids, n_new: int) -> dict:
    """Greedy no-cache generation, timestamping each emitted token.

    Returns TTFT (ms, prompt -> first token), TPOT (ms, mean of subsequent
    inter-token gaps), and throughput (tokens/sec over the whole run).
    """
    is_cuda = next(tp_model.model.parameters()).is_cuda
    generated = ids.to(next(tp_model.model.parameters()).device)

    token_times: list[float] = []
    t0 = time.perf_counter()
    for _ in range(n_new):
        logits = tp_model.forward(generated)
        nxt = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        generated = torch.cat([generated, nxt], dim=1)
        if is_cuda:
            torch.cuda.synchronize()
        token_times.append(time.perf_counter())

    ttft = (token_times[0] - t0) * 1e3
    if len(token_times) > 1:
        gaps = [(b - a) * 1e3 for a, b in zip(token_times, token_times[1:])]
        tpot = sum(gaps) / len(gaps)
    else:
        tpot = float("nan")
    total = token_times[-1] - t0
    return {"ttft_ms": ttft, "tpot_ms": tpot, "throughput": n_new / total if total else float("nan")}


def main() -> None:
    base, _ = load_tinyllama_from_hf(MODEL_NAME, dtype=torch.float32)
    base.eval()

    g = torch.Generator().manual_seed(20240601)
    ids = torch.randint(0, base.config.vocab_size, (1, PROMPT_LEN), generator=g)

    print(f"Device: {DEVICE}")
    print(f"Prompt {PROMPT_LEN} tokens, generating {NEW_TOKENS} (greedy, no-cache)\n")

    rows = []
    for ws in WORLD_SIZES:
        # Fresh wrapper per world_size so each builds its own rank shards.
        tp = TensorParallelLlamaModel(base).replace_with_tp_layers(ws)
        # One warmup generation (compile / cache warmup) before timing.
        tp.forward(ids)
        m = _time_generation(tp, ids, NEW_TOKENS)
        rows.append((ws, m))

    header = f"{'world_size':>10} | {'TTFT (ms)':>10} | {'TPOT (ms)':>10} | {'throughput':>11}"
    print(header)
    print("-" * len(header))
    for ws, m in rows:
        print(f"{ws:>10} | {m['ttft_ms']:>10.2f} | {m['tpot_ms']:>10.2f} | "
              f"{m['throughput']:>11.1f}")

    print("\nworld_size>1 is SIMULATED on a single device: the N rank shards run")
    print("sequentially and their partials are summed, so this does the same FLOPs")
    print("as dense and is not faster (usually slightly slower). Real speedup needs")
    print("N physical GPUs (one rank per process, all-reduce overlapping compute);")
    print("CUDA streams can overlap the shards on one GPU but not its compute units.")


if __name__ == "__main__":
    main()
