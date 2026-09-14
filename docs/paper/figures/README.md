# Figures

Placeholder directory for the CARL paper plots. The evaluation harness
(`scripts/benchmark_carl.py`) writes structured results to
`docs/carl_results.json`; the figures below are generated from that file.

Intended figures (referenced by `carl_paper.tex`):

- `fig_nonstationary.pdf` — SLO-satisfaction over time for the non-stationary
  workload (Scenario 3), one line per baseline, with vertical markers at the
  INTERACTIVE→BATCH→BURST regime boundaries. The headline plot: CARL tracks the
  regime oracle through the transitions while the static baselines and the
  independent AutoTuner sag at each shift.
- `fig_ablation.pdf` — bar chart of mean reward / SLO-satisfaction for CARL with
  each coordinated component disabled (Scenario 3 ablation).
- `fig_regime_dist.pdf` — regime-detection confusion / distribution on real
  LMSYS-Chat-1M traffic (Scenario 4).

To regenerate the underlying data:

```
python scripts/benchmark_carl.py --thompson           # synthetic, all scenarios
python scripts/benchmark_carl.py --scenario 4 --real  # real LMSYS for Scenario 4
```

Plot scripts are intentionally not committed yet (the numbers come from a
control-loop simulation; see the paper's Evaluation section for the honest
scope). Drop a small matplotlib script here that reads `carl_results.json` and
emits the PDFs above when wiring the camera-ready.
