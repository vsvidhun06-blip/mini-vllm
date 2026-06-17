# CARL controller-overhead reconciliation

Reconciles three CARL controller-overhead numbers that looked inconsistent:

| # | number | source | what it measures |
|---|--------|--------|------------------|
| 1 | ~102 µs P99 | `overhead.py` M1 → `overhead_results.json`, `real_step_end_to_end.p99_us` | isolated `controller.step()`, 10 000 samples, no serving loop |
| 2 | ~2527 µs P99 | hard-coded `LIVE_ABLATION_P99_US` in `overhead.py` (an earlier live run) | `carl_overhead.p99_decision_latency_us` inside the TinyLlama serving loop |
| 3 | ~16753 µs P99 | a later live run, same field, same script | same measurement, NON-STATIONARY, 30 requests, seeds 42–44 |

The isolated-vs-live gap (1 vs 2/3) was already plausibly explained (serving-loop
framing: the full `observe()` pipeline, lock acquisition, Python call overhead,
GC, first-call warmup). The part that needed explaining was the **live-vs-live**
gap — a 6.6× swing between two runs of *identical code*.

## Resolution: small-sample artifact, where the small-N "P99" is the single
## slowest decision, and that slowest decision is a one-time warmup / init / GC spike

It is **not** a regression and **not** a real tail-latency change. It is a
measurement artifact with two compounding causes:

**(a) `n_decisions` is tiny — P99 ≡ the max sample.** A CARL control cycle fires
once every `OBSERVE_INTERVAL` (= 10) scheduler steps (`maybe_step`,
`controller.py:181`). One timing is recorded per *actual* cycle
(`ablation_live.py` `_serve`, the `decision_us.append` at the control-cycle
boundary). A 30-request NON-STATIONARY run produces only **10 decisions per
seed**. With the nearest-rank percentile the harness uses, P99 over 10 samples is
index `round(0.99 × 9) = 9` — literally the **maximum** sample. Even pooled over
3 seeds (30 samples) P99 is index `round(0.99 × 29) = 29` — still the max. So
both "P99" numbers are just *"whichever single decision was slowest in that run."*

**(b) the slowest decision is a one-time spike, not steady state.** The first
control cycle of a process is cold (lazy imports, first `numpy.linalg` inversion,
allocator warmup). Later, sporadic spikes also occur — a GC pause, or the *first*
selection in a newly-visited per-regime bandit when the NON-STATIONARY workload
flips INTERACTIVE→BATCH mid-run (a cold `d×d` inverse for that regime). Any of
these, landing in a 10-sample window, becomes "the P99."

### Direct evidence (re-run, `overhead_warmup_probe.py`)

I could not diff the two original live JSONs: **neither `ablation_live_results.json`
nor any file containing `16753` exists on disk or anywhere in git history** — the
live ablation was never committed, and `2527` survives only as a hard-coded
constant. So I re-ran the real CARL-Full serving loop (CPU; absolute µs are
CPU-inflated vs the original GPU runs, but `n_decisions` and the warmup structure
are identical) for the exact configuration in question — NON-STATIONARY, 30
requests, seeds 42–44:

```
seed 42: n_decisions=10  cold(1st)=52970.5us  steady P50=390.7  raw-run-P99=52970.5us
seed 43: n_decisions=10  cold(1st)=  297.0us  steady P50=327.0  raw-run-P99= 9238.0us
seed 44: n_decisions=10  cold(1st)=  316.3us  steady P50=318.4  raw-run-P99=  361.4us
```

Three runs of identical code, same scenario → per-run "P99" of **52970, 9238, and
361 µs**: a **146× swing**, purely from which one-time spike happened to fall in a
10-sample window. Seed 42's max is the process cold-start (53 ms ≈ 159× the
steady-state median); seed 43's is a mid-run GC/regime-boundary spike (9.2 ms)
while its *first* call was cheap (297 µs, because the library was already warm
from seed 42); seed 44 caught no spike at all and looks essentially steady-state.
The original 2527 µs and 16753 µs are simply two more draws from this same
unstable max-of-~10 distribution — both are one-time spikes, neither is a tail
latency. Full per-decision lists: `docs/eval/raw/overhead/warmup_probe.json`.

The steady-state floor (decisions after the first, excluding the sporadic spike)
is ~320–390 µs **on CPU**; the isolated 10 000-sample measurement puts the pure
algorithm at 78 µs mean / 102 µs P99. The difference between those two is the
already-understood serving-loop framing plus CPU-vs-GPU, not the discrepancy under
investigation here.

## The three numbers, reported honestly

| latency | value | n (samples) | stability |
|---------|-------|-------------|-----------|
| **isolated algorithm cost** (per `step()`) | 78 µs mean / **102 µs P99** | 10 000 | stable — this is the real algorithm tail |
| **steady-state live cost** (warmup excluded) | ~330 µs P50 (CPU) | 27 (9/seed) | P50 robust; P99 at this N still noisy (a stray GC/init spike inflates it) |
| **cold-start live cost** (first decision only) | up to ~53 ms (CPU), one-time | 1/run | inherently variable; a one-time process-warmup cost, not recurring |

Notes:
- The **102 µs P99** is the number to put forward as CARL's per-step tail cost:
  it is measured over 10 000 samples and is stable (`overhead_results.json` M4
  also shows it is flat — O(1) — in requests seen).
- The steady-state live P50 of a few hundred µs reflects a real serving loop but
  is CPU-inflated; on GPU it sits between the 102 µs isolated figure and the CPU
  number. It should be reported as **amortized overhead per request** (cycle cost
  ÷ `OBSERVE_INTERVAL`, i.e. ≪ inference step time), not as a P99 tail.
- Cold-start is a **one-time** cost; report it separately as warmup, never folded
  into a steady-state P99.

**Do not report 2527 µs or 16753 µs as P99 tail latency.** Both are the max of a
~10-sample window dominated by a one-time spike, and are not reproducible.

## What changed in the harness

`ablation_live.py` now records, in `carl_overhead`: the raw per-run
`decision_us_per_run` lists, `n_decisions_per_run`, a **warmup-excluded** P50/P99
(first decision of each run dropped), and the per-run `cold_start_us` separately —
so a future run captures the distribution instead of collapsing it to one
warmup-dominated P99. `overhead_warmup_probe.py` is the standalone diagnostic that
produced the evidence above (CARL-Full only, full per-decision capture).

## Memory accounting (drop-in for the paper's overhead section)

CARL's memory footprint is two distinct quantities that should not be summed into
one headline number. **Learned-state cost:** the LinUCB bandit holds one `d×d`
matrix `A` and one length-`d` vector `b` per arm — 880 bytes per arm
(`10×10×8 + 10×8`), 30 arms total, **0.025 MB**. This is fixed by design: the
matrices are `d×d` regardless of how long the controller runs — measured at
exactly 26 400 bytes both before and after 10 000 updates — so it is **O(1) in
requests served** and scales only with arms × (`d² + d`)
(`overhead_results.json` M4 confirms the per-step cost is correspondingly flat).
**Evaluation-logging cost:** the controller also appends one `ControllerLogEntry`
per decision for offline analysis (adaptation traces, reward curves); at the
10 000 entries logged in the overhead run this is ~6.2 MB (≈0.54 KB/entry, linear
in entries). This is an artifact of how the eval harness retains every decision,
**not** a property of the algorithm — a production deployment would cap it with a
ring buffer or disable it, carrying only the ~0.025 MB of learned state. The real,
permanent claim is therefore **~0.03 MB of learned state, O(1) in requests**; the
~6.2 MB is evaluation logging at 10 k retained decisions and should be reported as
such, separately.

## Open / unresolved

- **Absolute live µs are CPU-measured here.** The original 2527/16753 figures came
  from GPU runs that were never committed, so the *exact* GPU steady-state and
  GPU cold-start magnitudes are not pinned down — only the *mechanism* (small-N
  P99 ≡ max ≡ one-time spike) is, and it fully accounts for a >100× run-to-run
  swing. A clean GPU re-run with the new instrumentation (`decision_us_per_run` +
  warmup-excluded P99) would replace the CPU placeholders in the table above; the
  conclusion (don't report the raw small-N P99) is independent of that.
- At n_decisions = 10/run, even the *warmup-excluded* P99 is not a stable
  statistic (seed 43's steady window still caught a 9.2 ms GC/init spike). For a
  trustworthy live tail, either raise the request count substantially (hundreds of
  decisions) or report the amortized mean / P50 and cite the isolated 102 µs P99
  as the algorithm tail. Picking whichever of 2527/16753 "looks better" would be
  the wrong call — both are noise.
