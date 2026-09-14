"""One definition of SLO accounting, so results cannot disagree with themselves.

PHASE 12. The audit that motivated this module found the SLO fields in the
committed results to be mutually inconsistent in unit, polarity and meaning:

  * `docs/eval/legacy/ablation_live_results.RECONSTRUCTED.json` reports the
    AutoTuner arm with `ttft_p99_mean: 61386.1` **and** `slo_violations: 0`,
    against a declared `slo_ttft_ms: 200.0`. A p99 of 61 seconds cannot coexist
    with zero violations of a 200 ms deadline.
  * `docs/eval/legacy/failure_cases_results.RECONSTRUCTED.json` reports
    `memory_pressure` with `ttft_p99_mean: 43232.0` and `slo_rate_mean: 0.0`,
    while `single_queue` in the same file reports `slo_rate_mean: 100.0`. One of
    those is a fraction and the other a percentage, or one is a satisfaction
    rate and the other a violation rate -- the file does not say which.
  * `slo_violations` is written by no script in the repository, so it entered
    the artifact during hand-transcription.

Both files are quarantined, but the underlying problem is that there was never
a single definition to be consistent with. This module supplies one.

DEFINITIONS (the only ones permitted in new results)
----------------------------------------------------
    satisfied(request)  <=>  ttft_ms <= ttft_target_ms
                             AND tpot_ms <= tpot_target_ms

    slo_satisfaction_rate  fraction in [0, 1] of COMPLETED requests satisfied
    slo_violation_rate     1 - slo_satisfaction_rate
    slo_violations         integer COUNT of unsatisfied completed requests

Fixed conventions, enforced by `tests/test_carl/test_slo_accounting.py`:

  * The statistical unit is the **request**, never the control cycle. Cycles
    contain different numbers of requests, so averaging per-cycle rates silently
    weights short cycles equally with long ones.
  * Rates are **fractions in [0, 1]**, never percentages. A field ending in
    `_rate` is a fraction; a field ending in `_pct` is a percentage; nothing is
    ambiguous.
  * The comparison is `<=` (meeting the deadline exactly is satisfaction).
  * Only COMPLETED requests are counted. An unfinished request has no TTFT, and
    silently treating it as satisfied is how a 61-second p99 coexists with zero
    violations.
  * `unfinished` is reported separately and never folded into either rate,
    because dropping requests is a different failure from missing a deadline.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SLOTargets:
    """Deadlines a request must meet. Milliseconds."""
    ttft_ms: float = 2000.0
    tpot_ms: float = 60.0

    def satisfied(self, ttft_ms: float | None, tpot_ms: float | None) -> bool:
        """True iff BOTH deadlines are met. A missing measurement is NOT a pass."""
        if ttft_ms is None:
            return False
        if ttft_ms > self.ttft_ms:
            return False
        # A single-token request has no TPOT; that is not a TPOT violation.
        if tpot_ms is not None and tpot_ms > self.tpot_ms:
            return False
        return True


def account(records, targets: SLOTargets, n_submitted: int | None = None) -> dict:
    """Compute the full SLO accounting block for a set of per-request records.

    Args:
        records: iterable of dicts (or objects) exposing `ttft_ms` and
            `tpot_ms`. Only completed requests should appear here.
        targets: the deadlines.
        n_submitted: total requests submitted, if known. Supplying it lets the
            block report `unfinished`, which is the field whose absence allowed
            a run that dropped requests to look like a run that met its SLO.

    Returns:
        A dict whose field names carry their own units:
          slo_satisfaction_rate  fraction [0,1] over completed requests
          slo_violation_rate     fraction [0,1], == 1 - satisfaction
          slo_violations         integer count
          slo_satisfaction_pct   percentage [0,100], for display only
          n_completed, n_submitted, unfinished
    """
    def _get(r, name):
        return r.get(name) if isinstance(r, dict) else getattr(r, name, None)

    recs = list(records)
    n = len(recs)
    ok = sum(1 for r in recs if targets.satisfied(_get(r, "ttft_ms"),
                                                  _get(r, "tpot_ms")))
    viol = n - ok
    rate = (ok / n) if n else 0.0
    out = {
        "slo_ttft_target_ms": targets.ttft_ms,
        "slo_tpot_target_ms": targets.tpot_ms,
        "n_completed": n,
        "slo_satisfaction_rate": rate,
        "slo_violation_rate": 1.0 - rate if n else 0.0,
        "slo_violations": viol,
        "slo_satisfaction_pct": rate * 100.0,
        "accounting_unit": "request",
        "rate_convention": "fraction_0_to_1",
    }
    if n_submitted is not None:
        out["n_submitted"] = int(n_submitted)
        out["unfinished"] = max(0, int(n_submitted) - n)
    return out


def check_consistency(block: dict, ttft_p99_ms: float | None = None) -> list[str]:
    """Return a list of internal contradictions in an SLO accounting block.

    Exists so a result file can be validated rather than trusted. The specific
    contradiction it was written to catch: a reported p99 far beyond the target
    alongside a claim of zero violations.
    """
    problems: list[str] = []
    n = block.get("n_completed")
    ok_rate = block.get("slo_satisfaction_rate")
    viol = block.get("slo_violations")

    if ok_rate is not None and not (0.0 <= ok_rate <= 1.0):
        problems.append(
            f"slo_satisfaction_rate={ok_rate} is outside [0,1]; a *_rate field "
            "must be a fraction (use *_pct for percentages)")
    if n and viol is not None and ok_rate is not None:
        expected = round((1.0 - ok_rate) * n)
        if abs(expected - viol) > 1:
            problems.append(
                f"slo_violations={viol} disagrees with "
                f"(1-slo_satisfaction_rate)*n_completed={expected}")
    if (ttft_p99_ms is not None and viol == 0
            and ttft_p99_ms > block.get("slo_ttft_target_ms", float("inf"))):
        problems.append(
            f"slo_violations=0 but ttft_p99={ttft_p99_ms:.0f}ms exceeds the "
            f"{block.get('slo_ttft_target_ms')}ms target: at least 1% of "
            "requests must have violated it")
    return problems
