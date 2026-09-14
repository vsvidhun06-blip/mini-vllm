"""
Benchmark: fp32 model vs INT8 (W8A8) quantized model.

Run:
    python scripts/benchmark_quantization.py

GPU-only, and needs TinyLlama in the HF cache (run `python -m src.engine.model`
once to fetch it). We load the model fp32, measure, then quantize it IN PLACE
and measure again -- so the two runs share one weight load (no double VRAM).

Metrics:
    * throughput -- generated tokens / second over a fixed-length run.
    * memory     -- torch.cuda.memory_allocated() resident model state.
    * TTFT       -- time-to-first-token: the prefill forward (generate 1 token).
    * TPOT       -- time-per-output-token: avg decode-step latency.

Only the attention projections are quantized here, so the memory drop is
partial (MLP + embeddings stay fp32); the table reports the real measured
numbers either way.

Table: metric | fp32 | int8 | improvement
"""
from __future__ import annotations

import torch

from src.engine.device import DEVICE
from src.engine.model import MODEL_NAME, load_tinyllama_from_hf

PROMPT = "The history of artificial intelligence began in antiquity, with myths and stories"
GEN_TOKENS = 64


def _elapsed_ms(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    out = fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end), out


def _measure(model, input_ids, eos_id):
    n_prompt = input_ids.shape[1]

    # Warmup (also triggers Triton JIT/autotune on the quantized path).
    model.generate(input_ids, max_new_tokens=4, eos_token_id=eos_id)

    # TTFT = prefill + first token.
    ttft_ms, _ = _elapsed_ms(
        lambda: model.generate(input_ids, max_new_tokens=1, eos_token_id=eos_id)
    )

    # Full run for throughput + TPOT.
    total_ms, out = _elapsed_ms(
        lambda: model.generate(input_ids, max_new_tokens=GEN_TOKENS, eos_token_id=eos_id)
    )
    n_new = out.shape[1] - n_prompt
    decode_steps = max(n_new - 1, 1)
    tpot_ms = (total_ms - ttft_ms) / decode_steps
    throughput = n_new / (total_ms / 1000.0)  # tokens/sec
    mem_mb = torch.cuda.memory_allocated() / (1024 * 1024)

    return {
        "throughput": throughput,
        "memory": mem_mb,
        "ttft": ttft_ms,
        "tpot": tpot_ms,
    }


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA not available -- INT8 quantization is GPU-only. Skipping.")
        return

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    eos_id = tokenizer.eos_token_id
    input_ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"].to(DEVICE)

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Prompt tokens: {input_ids.shape[1]} | generating {GEN_TOKENS} tokens\n")

    model, _ = load_tinyllama_from_hf(MODEL_NAME, dtype=torch.float32)
    model.eval()
    fp32 = _measure(model, input_ids, eos_id)

    model.quantize()  # in-place W8A8 of the attention projections
    int8 = _measure(model, input_ids, eos_id)

    # metric -> (fp32 val, int8 val, formatter, improvement string)
    def improve_higher(a, b):   # higher is better (throughput)
        return f"{b / a:.2f}x"

    def improve_lower(a, b):    # lower is better (memory, latency)
        return f"{a / b:.2f}x"

    rows = [
        ("throughput (tok/s)", fp32["throughput"], int8["throughput"], improve_higher),
        ("memory (MB)",        fp32["memory"],     int8["memory"],     improve_lower),
        ("TTFT (ms)",          fp32["ttft"],       int8["ttft"],       improve_lower),
        ("TPOT (ms)",          fp32["tpot"],       int8["tpot"],       improve_lower),
    ]

    header = f"{'metric':>20} | {'fp32':>12} | {'int8':>12} | {'improvement':>11}"
    print(header)
    print("-" * len(header))
    for name, a, b, imp in rows:
        print(f"{name:>20} | {a:>12.3f} | {b:>12.3f} | {imp(a, b):>11}")


if __name__ == "__main__":
    main()
