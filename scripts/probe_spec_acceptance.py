"""Probe acceptance rate at different early-exit depths.

Not a permanent benchmark -- this is the v0.3 dev script that decides
what `DEFAULT_N_DRAFT_LAYERS` should be. We scan a range of depths,
measure acceptance rate on a fixed prompt, and report (depth,
acceptance_rate, theoretical_speedup) so we can pick the sweet spot.

Theoretical speedup math:
    tokens per round = 1 + alpha * K
    cost  per round  = K * (depth / total_layers) + 1   (in units of "full forwards")
    speedup vs vanilla = (1 + alpha*K) / (K * c + 1) where c = depth/total

Net speedup requires alpha > c. So depth=8 needs >36% acceptance, depth=16
needs >73%. With an untrained early-exit on TinyLlama, the curve is what
it is -- this script tells us.
"""
from __future__ import annotations

from src.engine.model import MODEL_NAME, load_tinyllama_from_hf
from src.engine.scheduler import ContinuousBatchScheduler


def probe(model, tokenizer, prompt: str, max_new: int, depth: int, k: int) -> tuple[float, int, int]:
    """Run a single-request scheduler with depth override; return (rate, total_accepted, total_drafts)."""
    # Monkeypatch DEFAULT_N_DRAFT_LAYERS so the scheduler call path uses
    # the depth we want for this trial.
    import src.engine.spec_decode as spec_mod
    original = spec_mod.DEFAULT_N_DRAFT_LAYERS
    spec_mod.DEFAULT_N_DRAFT_LAYERS = depth
    try:
        rounds: list[tuple[int, int]] = []
        prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
        sched = ContinuousBatchScheduler(
            model,
            max_batch_size=1,
            num_blocks=64,
            enable_spec_decode=True,
            spec_decode_k=k,
            spec_decode_observer=lambda a, kk: rounds.append((a, kk)),
        )
        sched.add_request(
            request_id="probe",
            prompt_ids=prompt_ids,
            max_new_tokens=max_new,
            eos_token_id=tokenizer.eos_token_id,
        )
        while sched.has_work():
            sched.step()
    finally:
        spec_mod.DEFAULT_N_DRAFT_LAYERS = original
    total_accepted = sum(a for a, _ in rounds)
    total_drafts = sum(k for _, k in rounds)
    rate = total_accepted / total_drafts if total_drafts else 0.0
    return rate, total_accepted, total_drafts


def main() -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model, _ = load_tinyllama_from_hf(MODEL_NAME)
    print(f"\nProbing acceptance rate on TinyLlama-1.1B (22 layers)")
    print(f"Prompt: 'The capital of France is'  max_new=50  K=4\n")
    print(f"{'depth':>6} {'cost_c':>8} {'alpha':>8} {'need_a':>8} {'speedup':>10}")
    print("-" * 50)
    for depth in [4, 8, 12, 16, 18, 20, 21]:
        rate, _, _ = probe(
            model, tokenizer,
            "The capital of France is",
            max_new=50, depth=depth, k=4,
        )
        c = depth / 22
        speedup = (1 + rate * 4) / (4 * c + 1)
        print(f"{depth:>6d} {c:>8.3f} {rate:>8.1%} {c:>8.1%} {speedup:>10.2f}x")


if __name__ == "__main__":
    main()
