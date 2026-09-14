# 100k stability figures -- captions

**A.** Figure A (main). Cumulative oracle-capture (%) over the 100k-request INTERACTIVE horizon (seed 42, simulation). CARL holds ~99.97% of the hindsight DynOracle throughout (slope 95% CI excludes negative), matching Static-Best and exceeding the context-free UCB1/epsilon-greedy baselines.

**B.** Figure B (main). Regret RATE -- incremental oracle-regret per 1,000-request checkpoint -- for CARL over the 100k horizon. The rate is flat and bounded (Mann-Kendall trend not significant, p=0.10); the residual reflects the irreducible per-cycle noise floor of the clipped regret against a stochastic cost model, not a growing learning cost. (Cumulative regret is linear for this reason; the RATE is the stability quantity.)

**C.** Figure C (appendix). Operational stability: SLO satisfaction (C1) and TTFT p99 (C2) per trailing-window checkpoint remain steady across the horizon.

**D.** Figure D (appendix). Per-cycle arm-switch rate is 0 across all checkpoints -- CARL settles on its arm before the first checkpoint.

**E.** Figure E (appendix). LinUCB numerical health: max design-matrix condition number (E1) and max ||theta|| (E2) across the INTERACTIVE arms. The condition number is small and numerically safe at this horizon; ||theta|| stays bounded.

**F.** Figure F (appendix). Workload validity: the calibrated arrival model (rho=0.6, single-turn, lognormal(48,24)) yields a 99.9% INTERACTIVE regime mix with a shallow queue (p99=5), confirming the experiment operates in the intended regime.

