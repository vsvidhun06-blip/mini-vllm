"""THE FROZEN SATURATED REFERENCE -- recovered, verified, then read.

WHAT THIS IS
============
H1 and H2 of the sub-critical Poisson experiment compare that experiment's
quantities with the SAME quantities at the frozen saturated operating point,
`final_hardening`. Its results JSON was not in this working tree, and both
hypotheses were recorded as unresolved. It was recovered on 2026-10-22 from an
archive, after the T4 wiring smoke had been inspected and before the full
sub-critical run:

    D:\final_hardening_FINAL.zip
        sha256 39596503e7b634307b8fe5e6e5ee81717e2c753b244187077e8faa7630e2b0ae
    member docs/eval/final_hardening_results.json
        sha256 f1c1c95e2740abc4ee9b45c65938e24f1144a04357e0d85b377496b591431ecd

The member was copied into the tree with its raw bytes, never reserialised.
So were 40 of the archive's per-seed raw files: those of the four treatments
H2 reads. They are needed because the results JSON stores each treatment's
per-seed throughput as an UNLABELLED list. Only the raw files bind each value
to its seed. Static-Utility's raw files are not needed: B is outside H2.

WHAT IS VERIFIED, AND WHY IT FAILS CLOSED
=========================================
`load_verified` either returns a record whose every number was checked, or
raises `ReferenceEvidenceError` naming everything that failed. It checks:

* the raw-byte SHA256 of the results JSON and of every raw file used;
* strict JSON (no NaN / Infinity);
* the run's identity: experiment, GPU, model, dtype, device, seeds, n, the
  held-out seed and the 7 x 4 search axes;
* the held-out grid: exactly 28 unique configurations, identical in both
  tables that hold it, every throughput finite and positive, and the stored
  min / max / spread equal to the rows';
* both static selections, and that Static-Tput is the grid's throughput argmax;
* all five treatments are present and complete, with finite measurements;
* per-seed alignment: each raw file's (treatment, seed, throughput, request
  count) against the results JSON's per-seed list in seed order;
* that the stored means and SDs, and the stored paired differences of A, B, C
  and D, reproduce from the per-seed values.

There is no fallback. In particular, a missing or mismatched artifact never
falls back to the rounded paired differences transcribed in
`subcritical_poisson.SATURATED_REFERENCE`. The caller records the failure, H1
and H2 stay unresolved, and a published run refuses to start.

Tolerance. Every recomputation uses the frozen run's own arithmetic
(`statistics.fmean` / `stdev` over the same floats). On the recovered artifact
it reproduces every stored value EXACTLY. `REL_TOL` = 1e-12 only absorbs
summation-order rounding (~1e-15 relative), and is far below any measured
difference. Raw-file and per-seed values must match exactly: both are the JSON
repr of the same float.

Nothing here writes anything.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import statistics

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

VALIDATOR_VERSION = "saturated_reference/1"

RESULTS_REL = "docs/eval/final_hardening_results.json"
RAW_REL_DIR = "docs/eval/raw/final_hardening"

EXPECTED_RESULTS_SHA256 = (
    "f1c1c95e2740abc4ee9b45c65938e24f1144a04357e0d85b377496b591431ecd")

RECOVERY = {
    "recovered_utc_date": "2026-10-22",
    "source_zip": "D:\final_hardening_FINAL.zip",
    "source_zip_sha256": (
        "39596503e7b634307b8fe5e6e5ee81717e2c753b244187077e8faa7630e2b0ae"),
    "source_member": RESULTS_REL,
    "raw_members_used": (RAW_REL_DIR + "/<treatment>_seed<NNN>.json for "
                         "Static-Tput, RuleOnly, CARL-Repaired and "
                         "CARL-Expanded, seeds 042..051 (40 of the ZIP's 50)"),
    "copied_as": "raw bytes, only where absent; never reserialised",
    "timing": ("recovered AFTER the operational T4 smoke had been inspected "
               "and BEFORE the full sub-critical run; no smoke result is "
               "scientific evidence"),
}

# The historical run's identity, as FACTS about that run. Literal on purpose:
# they are not derived from current code, which could drift.
EXPECTED_IDENTITY = {
    "experiment": "final_hardening",
    "gpu": "Tesla T4",
    "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "dtype": "torch.float16",
    "device": "cuda",
}
EXPECTED_SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
EXPECTED_REQUESTS = 200
EXPECTED_HELD_OUT_SEED = 999
EXPECTED_GRID_BATCHES = [2, 4, 8, 12, 16, 24, 32]
EXPECTED_GRID_CHUNKS = [64, 128, 256, 512]
EXPECTED_TREATMENTS = ["Static-Tput", "Static-Utility", "RuleOnly",
                       "CARL-Repaired", "CARL-Expanded"]
EXPECTED_STATIC = {"Static-Tput": (32, 512), "Static-Utility": (8, 256)}

# Comparisons whose frozen statistics are cross-checked: label -> (a, b,
# primary metric). A, C and D feed H2; B is checked, but only so that its
# frozen statistics can be reported verbatim.
FROZEN_COMPARISONS = {
    "A_adaptation_vs_throughput_baseline": ("CARL-Repaired", "Static-Tput",
                                            "throughput_tps"),
    "B_adaptation_vs_utility_baseline": ("CARL-Repaired", "Static-Utility",
                                         "run_utility_v2"),
    "C_learning_isolation": ("CARL-Repaired", "RuleOnly", "throughput_tps"),
    "D_action_space_isolation": ("CARL-Expanded", "CARL-Repaired",
                                 "throughput_tps"),
}
RAW_TREATMENTS = ("Static-Tput", "RuleOnly", "CARL-Repaired", "CARL-Expanded")

REL_TOL = 1e-12

# Raw-byte SHA256 of every per-seed raw file read, as recovered.
EXPECTED_RAW_SHA256 = {
    "Static-Tput_seed042.json":
        "1e017280e5562b92d431247ae6d6f985caef7ea00f75f894ab1352d644957e98",
    "Static-Tput_seed043.json":
        "9cf40e3e16c941a95b47296f1125510065b1ba990bd337c3e04c5671bacbca78",
    "Static-Tput_seed044.json":
        "f8b4a04486bd99d20b51feeaeb6cdefd35e5d6e289cd4fc38075112189c34052",
    "Static-Tput_seed045.json":
        "27bd69ac5245e5c032dd42cd8605cb4b12a00a6f965811084cbde3f87d671351",
    "Static-Tput_seed046.json":
        "919814e7990df960b209a3be12815c3205dfa05919854f34ae9c2198b0b0c726",
    "Static-Tput_seed047.json":
        "8818fbd589f00fbf719ddca80fafefab83fbea3ffc92ea80cdb492fd5f1f9c74",
    "Static-Tput_seed048.json":
        "373d54ac11822a463d51d05a83d720f5c96823d89e1f8c4276d917822e323539",
    "Static-Tput_seed049.json":
        "0c0d92c2ba479baff074948d25691aee63d4e14a172f21adbb8d5caf299092e7",
    "Static-Tput_seed050.json":
        "b1711da828c22f17205705078a9d47dc3d8a4410fa3157ad1b273bd428b525da",
    "Static-Tput_seed051.json":
        "c662423569beb9ef95a87270b9e69b0b38abbab0c4223c64e4425b655da7644d",
    "RuleOnly_seed042.json":
        "adc0cd932a684d75a7d150b92bd834c6a095076da4e57cf755bb890b9f28e4fa",
    "RuleOnly_seed043.json":
        "690cb10b3e519831ef2b7ef4a7abcaf4ed76bc7975e8e6d790f0dad3a015ac7a",
    "RuleOnly_seed044.json":
        "3826cee82b5d12fc1d1d542f58418b7acaafa2b06d952d40e896185a03d48381",
    "RuleOnly_seed045.json":
        "896d071d1b10bdb079926a29d428ba037ba6190dfbb5511e4011fa04abff7c61",
    "RuleOnly_seed046.json":
        "00b073b17d3f3bd992617f2d411870468df4a408ede7c895dd98983a7fe88091",
    "RuleOnly_seed047.json":
        "cedd23252dd65003b5661da006440c50767f12de9f2cb9cc81d5ef7c5bd2f60e",
    "RuleOnly_seed048.json":
        "2ec953e1ad7eec1cbf4d43377208acb7e2f7c6e260384656e9bafc013c0596be",
    "RuleOnly_seed049.json":
        "7e06e69cb4bd6f1664a1d14ff3fb4e971ed2dbe90d1cb641288b618133b82227",
    "RuleOnly_seed050.json":
        "91f0787fe29ab06936cf7b452a21b543316e48c81eed5ef90fd590ee70666678",
    "RuleOnly_seed051.json":
        "bbb79ab1c7d829ff3f92787eaeeb804064df957570ae9c982a395d5c47deee95",
    "CARL-Repaired_seed042.json":
        "3f395b674b77ed5b9e74e22baadc2c5be74c196c2dffc45662206bb42eaafe43",
    "CARL-Repaired_seed043.json":
        "2e344858b1f0847933362bf3a82bcd9e967ed8be1df44a5147ca6d88942f1ad4",
    "CARL-Repaired_seed044.json":
        "28721b201cfa7150ebdf28dc8977c26d40350b938d8f61a55868e75e79384ab7",
    "CARL-Repaired_seed045.json":
        "1fad089dbf1867c63b6dcf0ef606408e0c8795f7806a44e33f68f440e3ca548d",
    "CARL-Repaired_seed046.json":
        "89651010083ee30ef3994a9f8e5a3925dcc654c469eba681b2e83aa90585fe8d",
    "CARL-Repaired_seed047.json":
        "7bde1e4945e63727b7ff19be9c94983f2cd3b293d5282a124e5f06dd46685161",
    "CARL-Repaired_seed048.json":
        "dc88bf92c8328c8ab90b06d96e80c27731fc7827961dfb350da3721581c75469",
    "CARL-Repaired_seed049.json":
        "fd2c60d10bfc1b34584aac05e3d6aa6ddc420eb105be11dc8567a581ebe91882",
    "CARL-Repaired_seed050.json":
        "f741128a7b1f0cecab8d7e1dcec26612156c7620f2fba9b5e0f81375987a4914",
    "CARL-Repaired_seed051.json":
        "d9cfb064d29f487811dab52e796c92268995fff90094d81d604d5e8fd5c8c519",
    "CARL-Expanded_seed042.json":
        "bafa8ef2c16ebd5df022cc5afcb69337c9679316fdbcb890a50dae0ef7628d2c",
    "CARL-Expanded_seed043.json":
        "6bc6b9697c10d6e18fb1efa6e8044f3415b16ba4402c6c316640bf2172319a3b",
    "CARL-Expanded_seed044.json":
        "dd8c0e227688e120e260c36442bdff6acd6a781e3efcc34cc650c9c3108fe693",
    "CARL-Expanded_seed045.json":
        "28620f372b54c35d2ff24e12aabb2f8ea834be655ebb03724fd5fb594926a869",
    "CARL-Expanded_seed046.json":
        "177a5dc6fa359672a7eef0a28304d11b5eb35b0c4477ed824583a6be006a1392",
    "CARL-Expanded_seed047.json":
        "cc9e8640ad4ad4b5d207ff3de1a6112ea7c839f47e02cdcd70ff0299969c4c7f",
    "CARL-Expanded_seed048.json":
        "b556e4670dca575fe7de778ef7d13b5728630530e411cfe3c62d03e9b32712f1",
    "CARL-Expanded_seed049.json":
        "a4a46ecaeca95877cee4bffee48dba00751113abf987f6d26b9a2aa5adde6c1f",
    "CARL-Expanded_seed050.json":
        "7a998167a17c10168b5e1b56d506880e01bc9297ae13686fdf092b58f9ad73e9",
    "CARL-Expanded_seed051.json":
        "d01e0d74c8895a94e243231adb213a5e4631f9aef59b4c9fcd11bc24bd033a0a",
}


class ReferenceEvidenceError(RuntimeError):
    """The frozen saturated reference is missing, corrupt or inconsistent."""

    def __init__(self, problems: list, record: dict | None = None):
        self.problems = list(problems)
        self.record = record or {}
        super().__init__("saturated reference REJECTED: " + "; ".join(problems))


def raw_files_expected() -> list:
    """Repo-relative paths of every raw file `load_verified` reads."""
    return [f"{RAW_REL_DIR}/{t}_seed{s:03d}.json"
            for t in RAW_TREATMENTS for s in EXPECTED_SEEDS]


def required_files() -> list:
    """Every file the reference consists of; the source bundle must carry them."""
    return [RESULTS_REL] + raw_files_expected()


def relative_range_pct(t_min, t_max):
    """THE H1 range: 100 * (max - min) / max. One definition, used for BOTH
    operating points; the denominator is the grid maximum."""
    return (100.0 * (t_max - t_min) / t_max) if t_max > 0 else None


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _reject_constant(name):
    raise ValueError(f"non-finite JSON constant {name}")


def _strict_json(data: bytes):
    return json.loads(data.decode("utf-8"), parse_constant=_reject_constant)


def _finite_positive(x) -> bool:
    return (isinstance(x, (int, float)) and not isinstance(x, bool)
            and math.isfinite(x) and x > 0)


def _close(a, b) -> bool:
    return (isinstance(a, (int, float)) and not isinstance(a, bool)
            and isinstance(b, (int, float))
            and math.isclose(a, b, rel_tol=REL_TOL, abs_tol=0.0))


def _read(root: str, rel: str) -> bytes | None:
    path = os.path.join(root, *rel.split("/"))
    if not os.path.isfile(path):
        return None
    with open(path, "rb") as f:
        return f.read()


def raw_manifest_sha256(raw_hashes: dict) -> str:
    """sha256sum-style digest over '<sha256>  <name>\\n', sorted by name."""
    lines = "".join(f"{raw_hashes[k]}  {k}\n" for k in sorted(raw_hashes))
    return _sha256(lines.encode("utf-8"))


def _grid_rows(rows, where: str, problems: list) -> dict:
    """{(mb, cs): throughput} after checking shape, uniqueness and finiteness."""
    if not isinstance(rows, list):
        problems.append(f"{where} is not a list")
        return {}
    out = {}
    for r in rows:
        if not isinstance(r, dict):
            problems.append(f"{where} has a non-object row")
            continue
        key = (r.get("max_batch_size"), r.get("chunk_size"))
        if key in out:
            problems.append(f"{where} repeats configuration mb={key[0]} cs={key[1]}")
            continue
        t = r.get("throughput_tps")
        if not _finite_positive(t):
            problems.append(f"{where} mb={key[0]} cs={key[1]} throughput {t!r} "
                            "is not finite and positive")
            continue
        out[key] = t
    expected = {(mb, cs) for mb in EXPECTED_GRID_BATCHES
                for cs in EXPECTED_GRID_CHUNKS}
    if len(rows) != len(expected):
        problems.append(f"{where} has {len(rows)} rows, not {len(expected)}")
    if set(out) != expected:
        problems.append(f"{where} configurations differ from the 7x4 grid: "
                        f"missing {sorted(expected - set(out))}, unexpected "
                        f"{sorted(set(out) - expected, key=str)}")
    return out


def _check_identity(d: dict, problems: list) -> dict:
    prov = d.get("_provenance")
    if not isinstance(prov, dict):
        problems.append("_provenance missing")
        return {}
    for k, v in EXPECTED_IDENTITY.items():
        if prov.get(k) != v:
            problems.append(f"_provenance.{k} is {prov.get(k)!r}, expected {v!r}")
    for where, obj in (("top level", d), ("_provenance", prov)):
        if obj.get("seeds") != EXPECTED_SEEDS:
            problems.append(f"{where} seeds {obj.get('seeds')!r} != {EXPECTED_SEEDS}")
        if obj.get("requests") != EXPECTED_REQUESTS:
            problems.append(f"{where} requests {obj.get('requests')!r} != "
                            f"{EXPECTED_REQUESTS}")
        if obj.get("validation_seed") != EXPECTED_HELD_OUT_SEED:
            problems.append(f"{where} validation_seed {obj.get('validation_seed')!r}"
                            f" != {EXPECTED_HELD_OUT_SEED}")
    search = prov.get("static_search_definition") or {}
    if (search.get("max_batch_size") != EXPECTED_GRID_BATCHES
            or search.get("chunk_size") != EXPECTED_GRID_CHUNKS
            or search.get("validation_seed") != EXPECTED_HELD_OUT_SEED):
        problems.append("_provenance.static_search_definition is not the "
                        "registered 7x4 held-out search")
    return {k: prov.get(k) for k in (
        "experiment", "gpu", "model", "dtype", "device", "seeds", "requests",
        "validation_seed", "arrival_mode", "torch", "cuda", "timestamp_utc",
        "git_sha", "git_dirty")}


def _check_grid(d: dict, problems: list) -> dict:
    cal = d.get("reward_calibration") or {}
    sel = d.get("static_selection") or {}
    if cal.get("held_out_seed") != EXPECTED_HELD_OUT_SEED:
        problems.append(f"reward_calibration.held_out_seed is "
                        f"{cal.get('held_out_seed')!r}")
    if cal.get("requests_per_sweep_run") != EXPECTED_REQUESTS:
        problems.append(f"reward_calibration.requests_per_sweep_run is "
                        f"{cal.get('requests_per_sweep_run')!r}")
    obs = _grid_rows(cal.get("observations"),
                     "reward_calibration.observations", problems)
    ranked = _grid_rows(sel.get("ranked_table"),
                        "static_selection.ranked_table", problems)
    if obs and ranked and obs != ranked:
        problems.append("the two held-out grid tables disagree on throughput")
    if not obs:
        return {}
    t_min, t_max = min(obs.values()), max(obs.values())
    spread = (sel.get("objective_spread") or {}).get("throughput_tps") or {}
    if spread.get("min") != t_min or spread.get("max") != t_max:
        problems.append(f"static_selection.objective_spread.throughput_tps "
                        f"min/max {spread.get('min')!r}/{spread.get('max')!r} "
                        f"!= the grid's {t_min!r}/{t_max!r}")
    if not _close(spread.get("spread"), t_max - t_min):
        problems.append("objective_spread.throughput_tps.spread != max - min")
    for name, key in (("static_tput", "Static-Tput"),
                      ("static_utility", "Static-Utility")):
        w = (sel.get(name) or {}).get("winner") or {}
        if (w.get("max_batch_size"), w.get("chunk_size")) != EXPECTED_STATIC[key]:
            problems.append(f"static_selection.{name}.winner is "
                            f"mb={w.get('max_batch_size')} cs={w.get('chunk_size')}, "
                            f"expected {EXPECTED_STATIC[key]}")
    argmax = max(obs, key=obs.get)
    if argmax != EXPECTED_STATIC["Static-Tput"]:
        problems.append(f"the grid's throughput argmax is {argmax}, not "
                        f"Static-Tput {EXPECTED_STATIC['Static-Tput']}")
    return {
        "n_configurations": len(obs),
        "rows": [{"max_batch_size": k[0], "chunk_size": k[1],
                  "throughput_tps": obs[k]} for k in sorted(obs)],
        "throughput_min": t_min, "throughput_max": t_max,
        "throughput_range_absolute": t_max - t_min,
        "stored_objective_spread_throughput_tps": dict(spread),
    }


def _check_treatments(d: dict, problems: list) -> dict:
    configs = d.get("configs")
    if not isinstance(configs, dict):
        problems.append("configs missing")
        return {}
    if sorted(configs) != sorted(EXPECTED_TREATMENTS):
        problems.append(f"configs {sorted(configs)} != {sorted(EXPECTED_TREATMENTS)}")
    if d.get("completed_configs") != EXPECTED_TREATMENTS:
        problems.append(f"completed_configs {d.get('completed_configs')!r} is not "
                        "all five treatments")
    per_seed = {}
    for name in EXPECTED_TREATMENTS:
        c = configs.get(name)
        if not isinstance(c, dict):
            continue
        ps = c.get("_per_seed") or {}
        per_seed[name] = {}
        for metric in ("throughput_tps", "run_utility_v2"):
            vals = ps.get(metric)
            if (not isinstance(vals, list) or len(vals) != len(EXPECTED_SEEDS)
                    or not all(_finite_positive(v) for v in vals)):
                problems.append(f"configs.{name}._per_seed.{metric} is not "
                                f"{len(EXPECTED_SEEDS)} finite positive values")
                continue
            per_seed[name][metric] = vals
            stored_mean = c.get(metric + "_mean")
            if not _close(stored_mean, statistics.fmean(vals)):
                problems.append(f"configs.{name}.{metric}_mean {stored_mean!r} "
                                "!= the mean of its _per_seed values")
            if not _close(c.get(metric + "_std"), statistics.stdev(vals)):
                problems.append(f"configs.{name}.{metric}_std != the SD of its "
                                "_per_seed values")
        if name in EXPECTED_STATIC:
            ac = c.get("applied_config") or {}
            if (ac.get("max_batch_size"), ac.get("chunk_size")) != EXPECTED_STATIC[name]:
                problems.append(f"configs.{name}.applied_config is not "
                                f"{EXPECTED_STATIC[name]}")
        cs = c.get("cold_start")
        if isinstance(cs, dict):
            keys = sorted(int(k) for k in (cs.get("warmup_cycles_per_seed") or {}))
            if keys != EXPECTED_SEEDS:
                problems.append(f"configs.{name}.cold_start is keyed by {keys}, "
                                f"not {EXPECTED_SEEDS}")
    return per_seed


def _check_raw(root: str, per_seed: dict, expected_raw: dict,
               problems: list) -> dict:
    hashes = {}
    expected_names = {rel.rsplit("/", 1)[1] for rel in raw_files_expected()}
    if set(expected_raw) != expected_names:
        problems.append("the expected raw-file table does not name exactly the "
                        f"{len(expected_names)} files read")
    for t in RAW_TREATMENTS:
        for i, s in enumerate(EXPECTED_SEEDS):
            name = f"{t}_seed{s:03d}.json"
            data = _read(root, f"{RAW_REL_DIR}/{name}")
            if data is None:
                problems.append(f"raw file {name} is missing")
                continue
            digest = _sha256(data)
            hashes[name] = digest
            if digest != expected_raw.get(name):
                problems.append(f"raw file {name} sha256 {digest} != expected "
                                f"{expected_raw.get(name)}")
                continue
            try:
                r = _strict_json(data)
            except ValueError as exc:
                problems.append(f"raw file {name} is not valid JSON ({exc})")
                continue
            if not isinstance(r, dict) or r.get("config") != t or r.get("seed") != s:
                problems.append(f"raw file {name} is not labelled {t} seed {s}")
                continue
            n_req = len(r.get("requests") or [])
            if n_req != EXPECTED_REQUESTS:
                problems.append(f"raw file {name} holds {n_req} requests")
            vals = (per_seed.get(t) or {}).get("throughput_tps")
            if vals is not None and r.get("throughput_tps") != vals[i]:
                problems.append(
                    f"per-seed misalignment: {name} throughput "
                    f"{r.get('throughput_tps')!r} != configs.{t}._per_seed"
                    f".throughput_tps[{i}] {vals[i]!r}")
            if t in EXPECTED_STATIC:
                ac = r.get("applied_config") or {}
                if (ac.get("max_batch_size"), ac.get("chunk_size")) != EXPECTED_STATIC[t]:
                    problems.append(f"raw file {name} applied_config is not "
                                    f"{EXPECTED_STATIC[t]}")
    return hashes


def _check_comparisons(d: dict, per_seed: dict, problems: list) -> dict:
    comps = d.get("comparisons")
    if not isinstance(comps, dict):
        problems.append("comparisons missing")
        return {}
    frozen = {}
    for label, (a, b, metric) in FROZEN_COMPARISONS.items():
        c = comps.get(label)
        if not isinstance(c, dict):
            problems.append(f"comparisons.{label} missing")
            continue
        if (c.get("a"), c.get("b"), c.get("metric")) != (a, b, metric):
            problems.append(f"comparisons.{label} is {c.get('a')!r} - "
                            f"{c.get('b')!r} on {c.get('metric')!r}, expected "
                            f"{a} - {b} on {metric}")
            continue
        blocks = {metric: c.get(metric)}
        blocks.update(c.get("secondary") or {})
        for m, blk in blocks.items():
            av = (per_seed.get(a) or {}).get(m)
            bv = (per_seed.get(b) or {}).get(m)
            if av is None or bv is None or not isinstance(blk, dict):
                problems.append(f"comparisons.{label}.{m} cannot be checked")
                continue
            diffs = [x - y for x, y in zip(av, bv)]
            stored = blk.get("per_seed_difference")
            if (blk.get("n") != len(EXPECTED_SEEDS) or not isinstance(stored, list)
                    or len(stored) != len(diffs)
                    or not all(p == q or _close(p, q)
                               for p, q in zip(stored, diffs))):
                problems.append(f"comparisons.{label}.{m} per-seed differences "
                                f"do not reproduce from {a} and {b}")
                continue
            if not _close(blk.get("mean_difference"), statistics.fmean(diffs)):
                problems.append(f"comparisons.{label}.{m}.mean_difference does "
                                "not reproduce")
        frozen[label] = c
    return frozen


def load_verified(root: str | None = None, *,
                  expected_results_sha256: str | None = None,
                  expected_raw_sha256: dict | None = None) -> dict:
    """The verified reference record, or ReferenceEvidenceError. Reads only.

    The expected digests default to the PINNED production values; tests inject
    their own for synthetic fixtures. Nothing else about validation varies.
    """
    root = root or _REPO_ROOT
    exp_sha = (EXPECTED_RESULTS_SHA256 if expected_results_sha256 is None
               else expected_results_sha256)
    exp_raw = (EXPECTED_RAW_SHA256 if expected_raw_sha256 is None
               else expected_raw_sha256)
    base = {"validator_version": VALIDATOR_VERSION, "source_member": RESULTS_REL,
            "recovery": dict(RECOVERY), "expected_artifact_sha256": exp_sha,
            "artifact_sha256": None, "raw_manifest_sha256": None}
    data = _read(root, RESULTS_REL)
    if data is None:
        raise ReferenceEvidenceError([f"{RESULTS_REL} is missing"], base)
    digest = _sha256(data)
    base["artifact_sha256"] = digest
    if digest != exp_sha:
        raise ReferenceEvidenceError(
            [f"{RESULTS_REL} sha256 {digest} != expected {exp_sha}"], base)
    try:
        d = _strict_json(data)
    except ValueError as exc:
        raise ReferenceEvidenceError(
            [f"{RESULTS_REL} is not valid JSON ({exc})"], base) from None
    if not isinstance(d, dict):
        raise ReferenceEvidenceError([f"{RESULTS_REL} is not a JSON object"], base)
    problems: list = []
    identity = _check_identity(d, problems)
    grid = _check_grid(d, problems)
    per_seed = _check_treatments(d, problems)
    raw_hashes = _check_raw(root, per_seed, exp_raw, problems)
    frozen = _check_comparisons(d, per_seed, problems)
    base["raw_file_sha256"] = raw_hashes
    base["raw_manifest_sha256"] = raw_manifest_sha256(raw_hashes)
    if problems:
        raise ReferenceEvidenceError(problems, base)
    configs = d["configs"]
    return {
        **base,
        "status": "verified",
        "identity": identity,
        "fields_used": [
            "_provenance.{experiment,gpu,model,dtype,device,seeds,requests,"
            "validation_seed,static_search_definition}",
            "seeds, requests, validation_seed, completed_configs",
            "reward_calibration.{observations,held_out_seed,"
            "requests_per_sweep_run}",
            "static_selection.{ranked_table,objective_spread.throughput_tps,"
            "static_tput.winner,static_utility.winner}",
            "configs.<treatment>.{_per_seed,throughput_tps_mean/std,"
            "run_utility_v2_mean/std,applied_config,cold_start}",
            "comparisons.{A,B,C,D} (cross-checked; reported verbatim)",
            RAW_REL_DIR + "/<treatment>_seed<NNN>.json.{config,seed,"
            "throughput_tps,requests,applied_config}",
        ],
        "grid": grid,
        "per_seed_throughput_tps": {
            t: {str(s): v for s, v in zip(EXPECTED_SEEDS,
                                          per_seed[t]["throughput_tps"])}
            for t in EXPECTED_TREATMENTS},
        "throughput_tps_mean": {t: configs[t]["throughput_tps_mean"]
                                for t in EXPECTED_TREATMENTS},
        "frozen_comparisons_verbatim": frozen,
        "frozen_statistics_note": (
            "the frozen run's own paired statistics (z-interval, Cohen's d), "
            "copied verbatim and never recomputed or rewritten"),
        "tolerance": {"rel_tol": REL_TOL, "applies_to": (
            "recomputed means, SDs and paired differences; raw-file vs "
            "per-seed values must be identical")},
    }


def per_seed_in_order(record: dict, treatment: str) -> list:
    """A treatment's verified per-seed throughput, in EXPECTED_SEEDS order."""
    vals = record["per_seed_throughput_tps"][treatment]
    return [vals[str(s)] for s in EXPECTED_SEEDS]


def try_load(root: str | None = None, **kw) -> dict:
    """`load_verified`, but a failure becomes an explicit unverified record."""
    try:
        return load_verified(root, **kw)
    except ReferenceEvidenceError as exc:
        return {**exc.record, "status": "unverified", "problems": exc.problems}


def fingerprint_identity(record: dict) -> dict:
    """What a resume must agree on: WHICH evidence, verified by WHICH rules."""
    return {k: record.get(k) for k in (
        "validator_version", "status", "source_member", "artifact_sha256",
        "expected_artifact_sha256", "raw_manifest_sha256")}
