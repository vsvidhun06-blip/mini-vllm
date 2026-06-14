"""
Contextual-bandit controllers over the CARLConfig arm set, plus the reward.

WHY A CONTEXTUAL BANDIT (and not RL, and not the AutoTuner's rules)
-------------------------------------------------------------------
The control problem is: at each decision point, observe a context x (the
normalized RuntimeState), pick one of a few discrete configs (arms), and receive
a scalar reward (utility). We want to learn which arm is best AS A FUNCTION OF
the context, online, with no training phase and a strong cold-start.

A full RL agent (value iteration over state/action trajectories) is overkill:
serving rewards are effectively immediate (a config's effect shows up within an
observe interval), so there is no long-horizon credit-assignment problem to
justify the sample cost and instability of RL. A contextual bandit is the right
tool -- it is the immediate-reward special case, and LinUCB in particular gives
us a principled explore/exploit rule with a closed-form update and provable
regret bounds (Li et al. 2010, "A Contextual-Bandit Approach to Personalized
News Article Recommendation").

It strictly generalises the AutoTuner: where the AutoTuner maps one bottleneck
to one fixed nudge, the bandit learns a context-dependent JOINT config and keeps
adapting as the reward landscape shifts.

WHAT'S HERE
-----------
  utility()              -- the scalar reward (throughput + SLO + cache).
  LinUCBBandit           -- LinUCB over a fixed arm set (the proposed method).
  ThompsonSamplingBandit -- Gaussian Thompson Sampling, same interface (ablation).
  PerRegimeBandit        -- one independent bandit per WorkloadRegime, so each
                            regime learns its own arm preferences without
                            cross-contamination. This is what the controller holds.

Only dependency is numpy (small per-arm d x d linear algebra). No torch.
"""
from __future__ import annotations

import numpy as np

from src.carl.state import WorkloadRegime


# ---------------------------------------------------------------------------
# utility() -- the reward.
# ---------------------------------------------------------------------------
#
# A single scalar the bandit maximises, trading off the four things an operator
# actually cares about. Each term is already in [0, 1] so the weights (summing to
# 1.0) make the reward itself land in [0, 1] -- a bounded reward keeps LinUCB's
# confidence radius well-scaled and Thompson's noise model sane.
#
#   throughput_norm        higher is better (already normalized to a reference).
#   1 - ttft_violation_rate  fraction of requests that MET the TTFT SLO.
#   1 - tpot_violation_rate  fraction that met the per-output-token SLO.
#   cache_hit_rate         reward for exploiting the prefix cache.
#
# Default weights bias slightly toward throughput + TTFT (0.3 each) over TPOT
# (0.2) and cache (0.2): in interactive serving, getting the first token out
# under SLO and keeping aggregate throughput up are the headline objectives.
# ---------------------------------------------------------------------------

DEFAULT_UTILITY_WEIGHTS = {
    "throughput": 0.3,
    "ttft": 0.3,
    "tpot": 0.2,
    "cache": 0.2,
}


def utility(metrics: dict, weights: dict | None = None) -> float:
    """Weighted reward in roughly [0, 1] from a metrics dict.

    Args:
        metrics: dict with keys (all optional, default to the neutral value):
            throughput_norm        float in [0, 1] (already reference-normalized)
            ttft_violation_rate    float in [0, 1]
            tpot_violation_rate    float in [0, 1]
            cache_hit_rate         float in [0, 1]
        weights: override for DEFAULT_UTILITY_WEIGHTS (keys throughput/ttft/
            tpot/cache). Does not need to sum to 1; the result is just a
            weighted sum.

    Returns:
        u = w_t * throughput_norm
          + w_ttft * (1 - ttft_violation_rate)
          + w_tpot * (1 - tpot_violation_rate)
          + w_cache * cache_hit_rate
    """
    w = weights or DEFAULT_UTILITY_WEIGHTS
    throughput = float(metrics.get("throughput_norm", 0.0))
    ttft_viol = float(metrics.get("ttft_violation_rate", 0.0))
    tpot_viol = float(metrics.get("tpot_violation_rate", 0.0))
    cache = float(metrics.get("cache_hit_rate", 0.0))
    return (
        w["throughput"] * throughput
        + w["ttft"] * (1.0 - ttft_viol)
        + w["tpot"] * (1.0 - tpot_viol)
        + w["cache"] * cache
    )


# ---------------------------------------------------------------------------
# LinUCBBandit
# ---------------------------------------------------------------------------
#
# Disjoint LinUCB (Li et al. 2010, Algorithm 1). Each arm a keeps independent
# ridge-regression sufficient statistics:
#
#     A_a = I_d + sum_t x_t x_t^T      (d x d, starts at identity for ridge reg)
#     b_a = sum_t r_t x_t              (d-vector)
#     theta_a = A_a^{-1} b_a           (the learned linear reward weights)
#
# Selection scores each arm by an upper confidence bound:
#
#     UCB_a = theta_a^T x  +  alpha * sqrt(x^T A_a^{-1} x)
#             \-- exploit --/   \------- explore --------/
#
# and picks argmax. The explore term is large for arms whose statistics in the
# CURRENT context direction are thin (high x^T A^{-1} x), so under-tried arms get
# probed; alpha tunes how aggressively. As an arm accumulates data along a
# context direction, its confidence radius there shrinks and selection converges
# to exploitation.
# ---------------------------------------------------------------------------


class LinUCBBandit:
    """Disjoint LinUCB over a fixed set of `n_arms` arms in `d`-dim context."""

    def __init__(self, n_arms: int, d: int, alpha: float = 0.5) -> None:
        if n_arms < 1:
            raise ValueError("n_arms must be >= 1")
        if d < 1:
            raise ValueError("d must be >= 1")
        self.n_arms = n_arms
        self.d = d
        self.alpha = alpha
        # One (d x d) design matrix and (d,) response vector per arm. A starts at
        # the identity (the ridge prior); b at zero.
        self.A = [np.identity(d, dtype=np.float64) for _ in range(n_arms)]
        self.b = [np.zeros(d, dtype=np.float64) for _ in range(n_arms)]
        # Per-arm selection counts -- diagnostics only (stats()/tests).
        self.counts = [0 for _ in range(n_arms)]

    def _context(self, context) -> np.ndarray:
        x = np.asarray(context, dtype=np.float64).reshape(-1)
        if x.shape[0] != self.d:
            raise ValueError(f"context dim {x.shape[0]} != bandit d {self.d}")
        return x

    def ucb_scores(self, context) -> np.ndarray:
        """UCB score for every arm at this context (exposed for tests/inspection)."""
        x = self._context(context)
        scores = np.empty(self.n_arms, dtype=np.float64)
        for a in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]
            exploit = float(theta @ x)
            explore = self.alpha * float(np.sqrt(max(0.0, x @ A_inv @ x)))
            scores[a] = exploit + explore
        return scores

    def select(self, context) -> int:
        """Return the index of the arm with the highest UCB score.

        Ties (e.g. the all-identity cold start, where every arm scores the same)
        break to the LOWEST index via np.argmax -- and since arm 0 is each
        regime's hand-tuned default (see config.config_arms), the cold-start
        choice is exactly the regime-oracle config. CARL therefore starts no
        worse than the hand-tuned baseline and only deviates once data warrants.
        """
        scores = self.ucb_scores(context)
        arm = int(np.argmax(scores))
        self.counts[arm] += 1
        return arm

    def update(self, arm: int, reward: float, context) -> None:
        """Fold one (arm, reward, context) observation into that arm's stats.

            A_a += x x^T ;  b_a += r x
        """
        if not 0 <= arm < self.n_arms:
            raise IndexError(f"arm {arm} out of range [0, {self.n_arms})")
        x = self._context(context)
        self.A[arm] += np.outer(x, x)
        self.b[arm] += float(reward) * x

    def theta(self, arm: int) -> np.ndarray:
        """Current learned weight vector for an arm (A^{-1} b)."""
        return np.linalg.inv(self.A[arm]) @ self.b[arm]


# ---------------------------------------------------------------------------
# ThompsonSamplingBandit (ablation)
# ---------------------------------------------------------------------------
#
# Bayesian linear-regression Thompson Sampling (Agrawal & Goyal 2013). It keeps
# the SAME sufficient statistics as LinUCB (A, b), so it's a drop-in alternative
# for the controller -- the ablation that answers "is the LinUCB explore rule
# doing the work, or would any reasonable bandit do?".
#
# Per arm the posterior over theta is N(theta_hat, v^2 A^{-1}) with
# theta_hat = A^{-1} b. Selection SAMPLES a theta from each arm's posterior and
# picks the arm with the highest sampled score theta^T x. Exploration is implicit
# in the posterior variance: thinly-sampled arms have wide posteriors, so their
# sampled scores occasionally come out high and they get tried. v scales the
# exploration (analogous to LinUCB's alpha).
# ---------------------------------------------------------------------------


class ThompsonSamplingBandit:
    """Gaussian Thompson Sampling over a fixed arm set; LinUCB-compatible API."""

    def __init__(self, n_arms: int, d: int, v: float = 0.5,
                 seed: int | None = None) -> None:
        if n_arms < 1:
            raise ValueError("n_arms must be >= 1")
        if d < 1:
            raise ValueError("d must be >= 1")
        self.n_arms = n_arms
        self.d = d
        self.v = v
        self.A = [np.identity(d, dtype=np.float64) for _ in range(n_arms)]
        self.b = [np.zeros(d, dtype=np.float64) for _ in range(n_arms)]
        self.counts = [0 for _ in range(n_arms)]
        # A private RNG so an ablation run is reproducible from a seed without
        # perturbing global numpy state (the benchmark seeds this).
        self._rng = np.random.default_rng(seed)

    def _context(self, context) -> np.ndarray:
        x = np.asarray(context, dtype=np.float64).reshape(-1)
        if x.shape[0] != self.d:
            raise ValueError(f"context dim {x.shape[0]} != bandit d {self.d}")
        return x

    def select(self, context) -> int:
        x = self._context(context)
        best_arm, best_score = 0, -np.inf
        for a in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[a])
            theta_hat = A_inv @ self.b[a]
            # Sample theta ~ N(theta_hat, v^2 A^{-1}). multivariate_normal wants a
            # PSD covariance; v^2 A^{-1} is PSD since A is SPD.
            cov = (self.v ** 2) * A_inv
            theta = self._rng.multivariate_normal(theta_hat, cov)
            score = float(theta @ x)
            if score > best_score:
                best_score, best_arm = score, a
        self.counts[best_arm] += 1
        return best_arm

    def update(self, arm: int, reward: float, context) -> None:
        if not 0 <= arm < self.n_arms:
            raise IndexError(f"arm {arm} out of range [0, {self.n_arms})")
        x = self._context(context)
        self.A[arm] += np.outer(x, x)
        self.b[arm] += float(reward) * x


# ---------------------------------------------------------------------------
# PerRegimeBandit
# ---------------------------------------------------------------------------
#
# One independent bandit per WorkloadRegime. This is the key structural choice:
# the optimal arm for a BURST is wrong for LONG_CONTEXT, so a single shared
# bandit would have to UN-learn between regime visits. Keeping a separate bandit
# per regime means each accumulates clean, regime-specific statistics, and a
# regime transition instantly swaps in the right learned policy instead of
# re-learning from a contaminated average.
#
# It owns the mapping from arm index -> CARLConfig (the arm sets from config.py),
# so the controller can ask for a config and later reward the arm that produced
# it. select() takes an explicit regime; update() also takes an explicit regime,
# because the controller rewards the PREVIOUS step's config, which may belong to
# a DIFFERENT regime than the one just selected (exactly what happens at a
# transition) -- relying on "last selected" here would mis-attribute reward
# across the boundary.
# ---------------------------------------------------------------------------


class PerRegimeBandit:
    """A dict of independent bandits keyed by WorkloadRegime, over CARLConfig arms."""

    def __init__(
        self,
        arms_by_regime: dict[WorkloadRegime, list],
        d: int,
        bandit_cls=LinUCBBandit,
        **bandit_kwargs,
    ) -> None:
        """
        Args:
            arms_by_regime: {regime: [CARLConfig, ...]} -- typically
                config.all_arm_sets().
            d: context dimension (state.FEATURE_DIM).
            bandit_cls: LinUCBBandit (default) or ThompsonSamplingBandit (ablation).
            **bandit_kwargs: forwarded to each per-regime bandit (e.g. alpha=, v=).
        """
        self.d = d
        self.bandit_cls = bandit_cls
        self._bandit_kwargs = dict(bandit_kwargs)
        self.arms_by_regime = {r: list(arms) for r, arms in arms_by_regime.items()}
        self.bandits = {
            r: bandit_cls(len(arms), d, **bandit_kwargs)
            for r, arms in self.arms_by_regime.items()
        }

    def reset(self) -> None:
        """Wipe all learned statistics back to the cold start (POST /carl/reset).

        Rebuilds each per-regime bandit fresh with the same arm sets and kwargs,
        so after reset CARL is back to its hand-tuned warm start (arm 0 per
        regime) with no learned history.
        """
        self.bandits = {
            r: self.bandit_cls(len(arms), self.d, **self._bandit_kwargs)
            for r, arms in self.arms_by_regime.items()
        }

    def arms(self, regime: WorkloadRegime) -> list:
        return self.arms_by_regime[regime]

    def select(self, regime: WorkloadRegime, context):
        """Select an arm for `regime`; return (arm_index, CARLConfig).

        Returning both lets the controller apply the config now and reward the
        arm index later, without re-deriving one from the other.
        """
        arm = self.bandits[regime].select(context)
        return arm, self.arms_by_regime[regime][arm]

    def update(self, regime: WorkloadRegime, arm: int, reward: float, context) -> None:
        """Reward `arm` under `regime`. Regime is explicit so cross-transition
        attribution is correct (see the class note)."""
        self.bandits[regime].update(arm, reward, context)

    def selection_counts(self) -> dict:
        """{regime_value: [count per arm]} -- diagnostics for stats()/the benchmark."""
        return {r.value: list(self.bandits[r].counts) for r in self.bandits}
