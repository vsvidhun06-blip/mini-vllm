"""
Single source of truth for the engine's device + dtype.

Why this module exists:

  Spreading `torch.device(...)` literals across model.py, kv_cache.py,
  and scheduler.py invites drift -- one file ends up on CPU while another
  thinks it's on GPU and you get a silent device-mismatch RuntimeError
  in the hot path. Importing one `DEVICE` constant everywhere makes the
  whole engine pick the same target.

DEVICE:
  CUDA if available, CPU otherwise. Evaluated at import time, so
  `CUDA_VISIBLE_DEVICES=""` (set before importing the engine) correctly
  forces the CPU path -- this is the supported fallback for hosts without
  a GPU or for the CPU-only correctness tests.

DTYPE:
  fp32, deliberately. The HF parity tests pin model behaviour at
  `atol=1e-4`, which bf16/fp16 can't hit -- they have ~3 decimal digits
  of mantissa, so the same tests would need atol ~1e-2 to pass. That's
  a 100x loosening that hides real bugs. Low-precision is a v0.3
  optimisation with its own calibration + kernel-selection story.

TF32:
  On Ampere/Ada GPUs (RTX 4060 included), PyTorch defaults to running
  fp32 matmul through tensor cores in TF32 mode -- 10-bit mantissa,
  ~8x faster, but a numerically lossy approximation of fp32. With TF32
  on, the parity tests at `atol=1e-4` will fail by 5-10x. We disable
  it explicitly so the v0.2 GPU port preserves byte-identical parity
  with the v0.1 CPU path. The throughput cost is acceptable for an
  educational, correctness-anchored project.
"""
from __future__ import annotations

import torch


DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE: torch.dtype = torch.float32


if DEVICE.type == "cuda":
    # Disable TF32 matmul + cuDNN's TF32 paths so fp32 stays "true" fp32.
    # See the module docstring for why -- short version: preserves the
    # atol=1e-4 HF parity contract.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
