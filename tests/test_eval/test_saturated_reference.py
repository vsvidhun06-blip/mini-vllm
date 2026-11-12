"""The frozen saturated reference: verified before H1/H2 may read it.

Synthetic fixtures carry their OWN digests, passed explicitly to
`load_verified`; production validation is never relaxed to admit them. Only the
expected hashes differ between a fixture and the recovered artifact. Every
check on content is the production check.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import statistics
import sys

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "scripts", "eval", "repair")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import saturated_reference as sref  # noqa: E402
import source_provenance as srcprov  # noqa: E402

SEEDS = sref.EXPECTED_SEEDS


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ---------------------------------------------------------------------------
# A synthetic but structurally complete frozen artifact.
# ---------------------------------------------------------------------------

_LEVEL = {"Static-Tput": 50.0, "Static-Utility": 40.0, "RuleOnly": 45.0,
          "CARL-Repaired": 47.0, "CARL-Expanded": 46.0}


def _tput(name, i):
    return _LEVEL[name] + 0.1 * i + (0.013 * i * i if name == "CARL-Expanded" else 0.0)


def _util(name, i):
    return 0.3 + _LEVEL[name] / 1000.0 + 0.001 * i


def _paired(av, bv):
    diffs = [a - b for a, b in zip(av, bv)]
    return {"n": len(diffs), "mean_difference": statistics.fmean(diffs),
            "std_difference": statistics.stdev(diffs), "ci95": [0.0, 1.0],
            "cohens_d_paired": 1.5, "per_seed_difference": diffs}


def _artifact() -> dict:
    rows = [{"max_batch_size": mb, "chunk_size": cs,
             "throughput_tps": 10.0 + mb + cs / 100.0}
            for mb in sref.EXPECTED_GRID_BATCHES for cs in sref.EXPECTED_GRID_CHUNKS]
    tp = [r["throughput_tps"] for r in rows]
    configs, per = {}, {}
    for name in sref.EXPECTED_TREATMENTS:
        t = [_tput(name, i) for i in range(len(SEEDS))]
        u = [_util(name, i) for i in range(len(SEEDS))]
        per[name] = {"throughput_tps": t, "run_utility_v2": u}
        c = {"_per_seed": {"throughput_tps": t, "run_utility_v2": u},
             "throughput_tps_mean": statistics.fmean(t),
             "throughput_tps_std": statistics.stdev(t),
             "run_utility_v2_mean": statistics.fmean(u),
             "run_utility_v2_std": statistics.stdev(u)}
        if name in sref.EXPECTED_STATIC:
            mb, cs = sref.EXPECTED_STATIC[name]
            c["applied_config"] = {"max_batch_size": mb, "chunk_size": cs}
        else:
            c["cold_start"] = {"warmup_cycles_per_seed": {str(s): 3 for s in SEEDS}}
        configs[name] = c
    comps = {}
    for label, (a, b, metric) in sref.FROZEN_COMPARISONS.items():
        other = "run_utility_v2" if metric == "throughput_tps" else "throughput_tps"
        comps[label] = {"a": a, "b": b, "metric": metric,
                        metric: _paired(per[a][metric], per[b][metric]),
                        "secondary": {other: _paired(per[a][other], per[b][other])}}
    prov = {**sref.EXPECTED_IDENTITY, "seeds": list(SEEDS),
            "requests": sref.EXPECTED_REQUESTS,
            "validation_seed": sref.EXPECTED_HELD_OUT_SEED,
            "torch": "2.11.0+cu128", "cuda": "12.8",
            "static_search_definition": {
                "max_batch_size": list(sref.EXPECTED_GRID_BATCHES),
                "chunk_size": list(sref.EXPECTED_GRID_CHUNKS),
                "validation_seed": sref.EXPECTED_HELD_OUT_SEED}}
    return {
        "seeds": list(SEEDS), "requests": sref.EXPECTED_REQUESTS,
        "validation_seed": sref.EXPECTED_HELD_OUT_SEED,
        "reward_calibration": {"held_out_seed": 999, "requests_per_sweep_run": 200,
                               "observations": copy.deepcopy(rows)},
        "static_selection": {
            "ranked_table": list(reversed(copy.deepcopy(rows))),
            "objective_spread": {"throughput_tps": {
                "min": min(tp), "max": max(tp), "spread": max(tp) - min(tp)}},
            "static_tput": {"winner": {"max_batch_size": 32, "chunk_size": 512}},
            "static_utility": {"winner": {"max_batch_size": 8, "chunk_size": 256}}},
        "configs": configs,
        "completed_configs": list(sref.EXPECTED_TREATMENTS),
        "comparisons": comps,
        "_provenance": prov,
    }


def _raw(t: str, i: int) -> dict:
    s = SEEDS[i]
    # The raw file is the seed-labelled TRUTH, whatever the artifact says.
    r = {"config": t, "seed": s, "throughput_tps": _tput(t, i),
         "requests": [{"request_id": f"r{k}"} for k in range(sref.EXPECTED_REQUESTS)]}
    if t in sref.EXPECTED_STATIC:
        mb, cs = sref.EXPECTED_STATIC[t]
        r["applied_config"] = {"max_batch_size": mb, "chunk_size": cs}
    return r


class Fixture:
    """A tree holding the synthetic reference, and the digests it was written
    with -- the 'explicitly injected test hashes'."""

    def __init__(self, root, artifact=None, raw_edit=None):
        self.root = str(root)
        self.artifact = artifact if artifact is not None else _artifact()
        self.write(raw_edit)

    def write(self, raw_edit=None):
        data = json.dumps(self.artifact, indent=2).encode("utf-8")
        self._put(sref.RESULTS_REL, data)
        self.results_sha = _sha(data)
        self.raw_sha = {}
        for t in sref.RAW_TREATMENTS:
            for i, s in enumerate(SEEDS):
                r = _raw(t, i)
                if raw_edit:
                    raw_edit(t, s, r)
                name = f"{t}_seed{s:03d}.json"
                b = json.dumps(r).encode("utf-8")
                self._put(f"{sref.RAW_REL_DIR}/{name}", b)
                self.raw_sha[name] = _sha(b)

    def _put(self, rel, data: bytes):
        full = os.path.join(self.root, *rel.split("/"))
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "wb") as f:
            f.write(data)

    def load(self):
        return sref.load_verified(self.root, expected_results_sha256=self.results_sha,
                                  expected_raw_sha256=self.raw_sha)


def _mutated(tmp_path, fn, raw_edit=None):
    """A fixture whose CONTENT is wrong but whose digests are re-injected, so
    the content checks -- not the hash check -- must catch it."""
    d = _artifact()
    fn(d)
    return Fixture(tmp_path, d, raw_edit)


def _problems(fx) -> str:
    with pytest.raises(sref.ReferenceEvidenceError) as exc:
        fx.load()
    return " | ".join(exc.value.problems)


# ---------------------------------------------------------------------------
# The recovered artifact itself.
# ---------------------------------------------------------------------------


def test_the_recovered_artifact_verifies_with_the_pinned_hashes():
    rec = sref.load_verified()
    assert rec["status"] == "verified"
    assert rec["artifact_sha256"] == sref.EXPECTED_RESULTS_SHA256 == (
        "f1c1c95e2740abc4ee9b45c65938e24f1144a04357e0d85b377496b591431ecd")
    assert rec["recovery"]["source_zip_sha256"] == (
        "39596503e7b634307b8fe5e6e5ee81717e2c753b244187077e8faa7630e2b0ae")
    ident = rec["identity"]
    assert (ident["experiment"], ident["gpu"], ident["model"], ident["dtype"]) == (
        "final_hardening", "Tesla T4", "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "torch.float16")
    assert (ident["torch"], ident["cuda"]) == ("2.11.0+cu128", "12.8")
    assert ident["seeds"] == list(range(42, 52)) and ident["requests"] == 200
    assert ident["validation_seed"] == 999
    g = rec["grid"]
    assert g["n_configurations"] == 28
    assert g["throughput_min"] == 44.97500716097522
    assert g["throughput_max"] == 96.00104467760788
    m = rec["throughput_tps_mean"]
    assert m["Static-Tput"] == 96.15569365964514
    assert m["RuleOnly"] == 89.02456617457298
    assert m["CARL-Repaired"] == 92.78791191900132
    # Every per-seed value is in seed order, as the raw files label it.
    assert sref.per_seed_in_order(rec, "Static-Tput")[5] == 96.84613857852595


def test_the_frozen_files_are_byte_identical_to_the_recovered_members():
    with open(os.path.join(_REPO_ROOT, *sref.RESULTS_REL.split("/")), "rb") as f:
        assert _sha(f.read()) == sref.EXPECTED_RESULTS_SHA256
    for rel in sref.raw_files_expected():
        with open(os.path.join(_REPO_ROOT, *rel.split("/")), "rb") as f:
            assert _sha(f.read()) == sref.EXPECTED_RAW_SHA256[rel.rsplit("/", 1)[1]]
    assert len(sref.EXPECTED_RAW_SHA256) == 40


def test_git_never_translates_the_frozen_files_line_endings():
    """core.autocrlf=true would rewrite these LF files as CRLF on checkout."""
    with open(os.path.join(_REPO_ROOT, ".gitattributes"), encoding="utf-8") as f:
        text = f.read()
    assert "docs/eval/final_hardening_results.json -text" in text
    assert "docs/eval/raw/final_hardening/*.json -text" in text


# ---------------------------------------------------------------------------
# Synthetic fixtures: valid, then every way to be wrong.
# ---------------------------------------------------------------------------


def test_a_valid_synthetic_reference_verifies(tmp_path):
    fx = Fixture(tmp_path)
    rec = fx.load()
    assert rec["status"] == "verified"
    assert rec["artifact_sha256"] == fx.results_sha
    assert rec["raw_manifest_sha256"] == sref.raw_manifest_sha256(fx.raw_sha)
    assert rec["grid"]["throughput_max"] == 10.0 + 32 + 5.12
    assert sref.per_seed_in_order(rec, "RuleOnly") == [
        _tput("RuleOnly", i) for i in range(10)]
    assert set(rec["frozen_comparisons_verbatim"]) == set(sref.FROZEN_COMPARISONS)


def test_the_pinned_hash_rejects_a_fixture_it_was_not_given(tmp_path):
    """No injected hash -> the production digest applies, and refuses."""
    fx = Fixture(tmp_path)
    with pytest.raises(sref.ReferenceEvidenceError, match="sha256"):
        sref.load_verified(fx.root)


def test_an_artifact_hash_mismatch_is_rejected_before_parsing(tmp_path):
    fx = Fixture(tmp_path)
    with pytest.raises(sref.ReferenceEvidenceError, match="sha256") as exc:
        sref.load_verified(fx.root, expected_results_sha256="0" * 64,
                           expected_raw_sha256=fx.raw_sha)
    assert exc.value.record["artifact_sha256"] == fx.results_sha
    assert "status" not in exc.value.record


def test_a_raw_file_hash_mismatch_is_rejected(tmp_path):
    fx = Fixture(tmp_path)
    bad = dict(fx.raw_sha, **{"RuleOnly_seed044.json": "1" * 64})
    with pytest.raises(sref.ReferenceEvidenceError, match="RuleOnly_seed044"):
        sref.load_verified(fx.root, expected_results_sha256=fx.results_sha,
                           expected_raw_sha256=bad)


def test_a_missing_artifact_or_raw_file_is_rejected(tmp_path):
    with pytest.raises(sref.ReferenceEvidenceError, match="missing"):
        sref.load_verified(str(tmp_path / "empty"))
    fx = Fixture(tmp_path)
    os.remove(os.path.join(fx.root, *sref.RAW_REL_DIR.split("/"),
                           "CARL-Expanded_seed051.json"))
    assert "CARL-Expanded_seed051.json is missing" in _problems(fx)


def test_corrupt_or_non_finite_json_is_rejected(tmp_path):
    fx = Fixture(tmp_path)
    path = os.path.join(fx.root, *sref.RESULTS_REL.split("/"))
    for body in (b"{not json", b'{"x": NaN}', b"[1, 2]"):
        with open(path, "wb") as f:
            f.write(body)
        with pytest.raises(sref.ReferenceEvidenceError, match="JSON"):
            sref.load_verified(fx.root, expected_results_sha256=_sha(body),
                               expected_raw_sha256=fx.raw_sha)


@pytest.mark.parametrize("mutate, expect", [
    (lambda d: d.update(seeds=list(range(43, 53))), "seeds"),
    (lambda d: d["_provenance"].update(seeds=[42, 43]), "seeds"),
    (lambda d: d.update(requests=100), "requests"),
    (lambda d: d["_provenance"].update(requests=40), "requests"),
    (lambda d: d.update(validation_seed=42), "validation_seed"),
    (lambda d: d["_provenance"].update(gpu="NVIDIA L4"), "gpu"),
    (lambda d: d["_provenance"].update(model="other/model"), "model"),
    (lambda d: d["_provenance"].update(dtype="torch.float32"), "dtype"),
    (lambda d: d["_provenance"].update(experiment="reward_v2_live"), "experiment"),
    (lambda d: d["_provenance"]["static_search_definition"].update(
        chunk_size=[64, 128, 256]), "static_search_definition"),
])
def test_wrong_identity_seeds_n_or_hardware_is_rejected(tmp_path, mutate, expect):
    assert expect in _problems(_mutated(tmp_path, mutate))


def _dup_grid_point(d):
    obs = d["reward_calibration"]["observations"]
    obs[1] = dict(obs[0])


def _drop_grid_point(d):
    d["reward_calibration"]["observations"].pop()
    d["static_selection"]["ranked_table"].pop()


def _grid_point_not_in_axes(d):
    d["reward_calibration"]["observations"][0]["chunk_size"] = 1024


def _tables_disagree(d):
    d["static_selection"]["ranked_table"][0]["throughput_tps"] += 1e-9


def _spread_disagrees(d):
    d["static_selection"]["objective_spread"]["throughput_tps"]["min"] = 44.0


def _non_finite_grid(d):
    d["reward_calibration"]["observations"][3]["throughput_tps"] = None


def _wrong_static_winner(d):
    d["static_selection"]["static_tput"]["winner"]["max_batch_size"] = 24


@pytest.mark.parametrize("mutate, expect", [
    (_dup_grid_point, "repeats configuration"),
    (_drop_grid_point, "27 rows"),
    (_grid_point_not_in_axes, "differ from the 7x4 grid"),
    (_tables_disagree, "disagree"),
    (_spread_disagrees, "objective_spread"),
    (_non_finite_grid, "finite"),
    (_wrong_static_winner, "static_tput.winner"),
])
def test_a_wrong_or_duplicated_grid_is_rejected(tmp_path, mutate, expect):
    assert expect in _problems(_mutated(tmp_path, mutate))


def test_a_missing_treatment_or_non_finite_measurement_is_rejected(tmp_path):
    assert "configs" in _problems(_mutated(
        tmp_path / "a", lambda d: d["configs"].pop("RuleOnly")))
    assert "completed_configs" in _problems(_mutated(
        tmp_path / "b", lambda d: d.update(completed_configs=["Static-Tput"])))
    assert "finite" in _problems(_mutated(
        tmp_path / "c", lambda d: d["configs"]["CARL-Repaired"]["_per_seed"][
            "throughput_tps"].__setitem__(3, -1.0)))
    assert "10 finite" in _problems(_mutated(
        tmp_path / "d", lambda d: d["configs"]["Static-Tput"]["_per_seed"][
            "throughput_tps"].pop()))


def test_per_seed_misalignment_against_the_raw_files_is_rejected(tmp_path):
    """Swap two seeds' raw throughputs: every aggregate still agrees, so only
    the seed-labelled raw files can catch it."""
    d = _artifact()
    vals = d["configs"]["CARL-Repaired"]["_per_seed"]["throughput_tps"]

    def swap(t, s, r):
        if t == "CARL-Repaired" and s in (44, 45):
            r["throughput_tps"] = vals[3 if s == 44 else 2]
    msg = _problems(Fixture(tmp_path, d, swap))
    assert "per-seed misalignment: CARL-Repaired_seed044.json" in msg
    assert "CARL-Repaired_seed045.json" in msg


def test_a_mislabelled_or_short_raw_file_is_rejected(tmp_path):
    def relabel(t, s, r):
        if t == "Static-Tput" and s == 47:
            r["seed"] = 48
    assert "not labelled Static-Tput seed 47" in _problems(
        Fixture(tmp_path / "a", raw_edit=relabel))

    def short(t, s, r):
        if t == "RuleOnly" and s == 50:
            r["requests"] = r["requests"][:199]
    assert "199 requests" in _problems(Fixture(tmp_path / "b", raw_edit=short))


@pytest.mark.parametrize("mutate, expect", [
    (lambda d: d["configs"]["Static-Tput"].update(throughput_tps_mean=96.16),
     "throughput_tps_mean"),
    (lambda d: d["configs"]["RuleOnly"].update(
        throughput_tps_std=d["configs"]["RuleOnly"]["throughput_tps_std"] * 1.001),
     "throughput_tps_std"),
    (lambda d: d["comparisons"]["C_learning_isolation"]["throughput_tps"][
        "per_seed_difference"].__setitem__(0, 9.0), "per-seed differences"),
    (lambda d: d["comparisons"]["D_action_space_isolation"]["throughput_tps"]
     .update(mean_difference=-1.7956), "mean_difference"),
    (lambda d: d["comparisons"]["A_adaptation_vs_throughput_baseline"].update(
        b="Static-Utility"), "expected CARL-Repaired - Static-Tput"),
])
def test_stored_aggregates_must_reproduce_from_the_per_seed_values(
        tmp_path, mutate, expect):
    assert expect in _problems(_mutated(tmp_path, mutate))


def test_a_rounding_sized_discrepancy_is_still_rejected(tmp_path):
    """REL_TOL absorbs summation order only: a mean stored to 4 decimals --
    the transcription's precision -- is rejected."""
    def round_mean(d):
        c = d["configs"]["CARL-Repaired"]
        c["throughput_tps_mean"] = round(c["throughput_tps_mean"], 4) + 1e-5
    assert "throughput_tps_mean" in _problems(_mutated(tmp_path, round_mean))
    assert sref.REL_TOL == 1e-12


def test_try_load_turns_a_failure_into_an_explicit_unverified_record(tmp_path):
    rec = sref.try_load(str(tmp_path))
    assert rec["status"] == "unverified" and rec["problems"]
    assert "grid" not in rec and "per_seed_throughput_tps" not in rec
    ident = sref.fingerprint_identity(rec)
    assert ident["status"] == "unverified" and ident["artifact_sha256"] is None


def test_the_fingerprint_identity_changes_with_the_evidence(tmp_path):
    fx = Fixture(tmp_path)
    a = sref.fingerprint_identity(fx.load())
    fx.artifact["configs"]["Static-Tput"]["policy"] = "edited"
    fx.write()
    b = sref.fingerprint_identity(fx.load())
    assert a["artifact_sha256"] != b["artifact_sha256"]
    assert a["raw_manifest_sha256"] == b["raw_manifest_sha256"]


def test_the_reference_is_in_the_archive_manifest(tmp_path):
    """Every reference file is a required data file, so the archive must carry
    it and verify_tree flags an edited copy on the runtime."""
    assert set(sref.required_files()) <= set(srcprov.ARCHIVE_REQUIRED_DATA_FILES)
    tree = {e["path"]: e for e in srcprov.tree_manifest(_REPO_ROOT)["files"]}
    for rel in sref.required_files():
        assert rel in tree
    assert tree[sref.RESULTS_REL]["sha256"] == sref.EXPECTED_RESULTS_SHA256
