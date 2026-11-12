# Disclosure — sub-critical Poisson campaign history

**Recorded:** 2026-10-26. **Status:** a separate, dated correction record. It
does **not** edit either version of the sub-critical pre-registration; both are
preserved as historical evidence:

* the September version, committed locally as `b455cc2`
  (`docs/eval/PREREGISTRATION_subcritical_poisson.md`, git blob
  `4168435dc9cfaa94bf635d6944f60ba1f1af5d77`, LF sha256
  `387aa692d2bcefff9db7ce5d0c5dc7c6fdf36c276ca7198fcfa48aa51cf88486`; the CRLF
  copy executed in September hashes to
  `11d2edc6e5c268995df27a5cda5168e7c0ddcecc8692d385103127a944514ca6`);
* the October version, present in the working tree (uncommitted) and in the
  executed October source bundle, sha256
  `e3776082796c955b3476d321a404f8a685eba18b00455357fa35ca06c0c26405`.

Where this record and either pre-registration disagree about chronology, this
record governs the *interpretation*; the original wording stays as written.

---

## 1. Summary

Three **complete** campaigns of the sub-critical Poisson design exist, each
backed by its own artifacts. All three used a Tesla T4, TinyLlama-1.1B-Chat-v1.0
in fp16, evaluation seeds 42–51, 200 requests per run, held-out seed 999, the
7 × 4 `max_batch_size × chunk_size` grid, target ρ = 0.40 per phase and the
same μ/λ. Their arrival offsets and request sequences are byte-identical (§4).

| id | dates (UTC) | protocol | sessions |
|---|---|---|---|
| **S1** | 2026-09-18, results stamped 21:21:57 | September pre-registration (`b455cc2`), experiment version `subcritical_poisson/1` | 2 (one CUDA OOM, resumed) |
| **S2** | 2026-09-19, ≈15:22 → 21:09:32 | identical to S1 (same fingerprint) | 1 (uninterrupted) |
| **O** | 2026-10-23, 17:13:32 → 23:07:37 | October pre-registration (`e3776082…`), `subcritical_poisson/4` | 2 (one CUDA OOM, resumed) |

The September results were viewed and discussed on 2026-09-19, **before** the
October amendments of 2026-10-21/22 were written. The October protocol was
frozen locally (bound by hash into the run fingerprint) before the October run,
but **after** earlier results on the same seeds were available and had been
discussed. It is therefore not a protocol specified before any results were
seen. No evidence of public registration of either version was located; this is
an absence of evidence in the local records searched, not a proof that none
exists.

Nothing here labels S1/S2 as pilots or O as an independent confirmation: the
records do not support either label. At the time, S1 was described in a
contemporaneous review prompt as the "FROZEN SUB-CRITICAL POISSON RESULT", and
S2 as "this canonical run" (§5).

---

## 2. Artifacts (full hashes, exact paths)

### S1 — 2026-09-18

| item | path | sha256 |
|---|---|---|
| evidence ZIP | `D:\subcritical_poisson_FINAL.zip` | `ca9dffb9aeed7993a2858c94d2615268e15f8b614223b0d454efed8e12c521f2` |
| results member | `docs/eval/subcritical_poisson_results.json` in that ZIP | `414bb87314212e02ad4a7842dd479b9c80a939a882fcb74a1658bed47e768d48` |
| operational log | `D:\subcritical_poisson_final.log` | `6135543331ef32f938e310013551d197d0b7dea22b9d49d512ea45a824d56d1e` |
| source archive | `C:\Users\Vidhu\mini-vllm-subcritical.zip` | `1668397a43e87574a5be17d34e9aa9f0144aa26a0a5421fe68bfd56790b67bd1` |
| source sidecar (embedded) | `docs/eval/SOURCE_PROVENANCE.json` in the ZIP | `89b249dd46fd05c25d6b38382b711415bc927dd1ece43dda4d4cdd7b43760523` |
| run fingerprint | `fingerprint_sha256` in results | `4d533319e2bf99a4988e716d8beb0b6e73a2ce2eb3cd6edd05cc1522e811528d` |

### S2 — 2026-09-19

| item | path | sha256 |
|---|---|---|
| evidence ZIP | `D:\CARL_September19_recovery\subcritical_poisson_FINAL (1).zip` | `f24489384fe2b084ac484963625598a3c2ba9cd243a0b8eed6254ae995209e53` |
| results member | `docs/eval/subcritical_poisson_results.json` in that ZIP | `6dd022689466c054461ea690e4c05d09f6de4dd170bc259182980d135f6026b1` |
| operational log | `D:\CARL_September19_recovery\subcritical_poisson_final (1).log` | `4266a1ba7bf44988accc44afbc84d6e470b4df7e516829f0a680a5a51aea709f` |
| source sidecar (embedded) | `docs/eval/SOURCE_PROVENANCE.json` in the ZIP | `89b249dd46fd05c25d6b38382b711415bc927dd1ece43dda4d4cdd7b43760523` (byte-identical to S1's) |
| run fingerprint | `fingerprint_sha256` in results | `4d533319e2bf99a4988e716d8beb0b6e73a2ce2eb3cd6edd05cc1522e811528d` (identical to S1's) |

The S2 ZIP and log were recovered from Google Drive (My Drive →
`CARL_subcritical_final`) and downloaded locally on 2026-10-26, as reported by
the author. **Their hashes were computed after that download and are not
authenticated against any historical manifest.** The log's own paths
(`/content/drive/MyDrive/CARL_subcritical_final/mini-vllm/…`) are consistent
with that location.

### O — 2026-10-23

| item | path | sha256 |
|---|---|---|
| final checkpoint ZIP | `D:\checkpoint_1792796860834369199_f793587b2717.zip` (+ `.verified.json`) | `cb95067ba0bf8f183d78f3c49cfccb74f8ac2eeebd9898393fcd564aa8083417` |
| final results member | `docs/eval/subcritical_poisson_results.json` in that ZIP | `f793587b2717d9b64e3b5335eeb38aef6b70c73a0934f40c2e49db9b0c784433` |
| pre-resume checkpoint ZIP | `D:\checkpoint_1792793381218572067_d125f3c55192.zip` (+ `.verified.json`) | `fcf6c2b5579ed7cfd585446c03498c39bb13a753e63d49ca1361cd8672844ae9` |
| pre-resume results member | same member name | `d125f3c5519240e5e0dd49cf5993cfa109aec1627f2e5dc6dce39f324c66eaa8` |
| executed source | `C:\Users\Vidhu\mini-vllm-bundles\subcritical-reference-20261023-171950\mini-vllm-subcritical-reference.zip` | `d78a24ed55bd90aa346f76d15a5ee5b98cdea216322883bd71b0acd9fe75efed` |
| source sidecar | `SOURCE_PROVENANCE.json` in the same folder | `44bcd27d213d1792f8738850ac2f1d605993f74461443da11cd89ea1924eb9be` |
| run fingerprint | `fingerprint_sha256` in results | `894cc422ea7528bc479941ad5f5a4c3d5c78bb00aa4b42b101ae3bb0ac103766` |

### Saturated reference (shared by all three)

`D:\final_hardening_FINAL.zip`, sha256
`39596503e7b634307b8fe5e6e5ee81717e2c753b244187077e8faa7630e2b0ae`.
S1 and S2 used a hard-coded, rounded transcription of its comparison values;
O verified the artifact itself. The values agree.

### A misplaced file

The untracked working-tree file `docs/eval/SOURCE_PROVENANCE.json` is
byte-identical (sha256 `89b249dd…760523`) to the **September** sidecar, not the
October one. It is left untouched.

---

## 3. Four kinds of assurance, kept separate

| | S1 | S2 | O |
|---|---|---|---|
| **File integrity** | ZIP `testzip` passes; 54 members; no duplicate paths; every JSON parses | same | both checkpoint ZIPs verified earlier (`.verified.json`) |
| **Source-provenance strength** | sidecar names `b455cc2` (clean) and archive `1668397a…`; the archive's files match the sidecar's 163-entry source manifest (122 byte-exact, 41 line-ending-only). **But the run copied the manifest digest from the sidecar** (`subcritical_poisson.py:1135` at `b455cc2`) and recorded `source_archive_sha256 = null`, so it never hashed the code that executed. | identical to S1 | archive `d78a24ed…` hashed on upload (`--verify-archive`) and the manifest recomputed over the executing tree. The sidecar records `git_head b455cc2` with `git_dirty = true`: the October source is a captured working tree, not a commit. Model revision `fe8a4ea1ffedaf415f4da2f062534de366a451e6` and file hashes recorded. |
| **Measurement checks** (re-derived 2026-10-26) | 50/50 evaluation units present with correct identity, seed and 200/200 requests; raw → summary means equal; all 8 comparison/metric pairs reproduce from raw per-seed data; frozen scales reproduce; static picks are the recorded argmax | same, plus every log line (28 sweep, 50 unit) matches the artifacts | same checks; 28 sweep files each hold 200 per-request records |
| **Historical certainty** | run times from results timestamp and ZIP member times; start time not recorded | start ≈ 15:21:52, derived from `elapsed_s`; the sidecar copy is stamped 15:21:18 | session times recorded in `run_history` |

**Missing raw sweep evidence.** S1 and S2 contain **no per-sweep-point files**:
the 28 held-out rows exist only as summary rows embedded in the results JSON
(`held_out_sweep_rows` = `reward_calibration.observations`), with no
per-request data. Their per-request sweep measurements and the `utility_v2` of
each sweep row cannot be re-derived; the frozen scales and both argmax
selections can, from the recorded rows. O stores each sweep point as its own
file, with all 200 per-request records.

**Not established for S1/S2:** the model revision (only the repository name is
recorded); any GPU identifier (also absent in O).

---

## 4. What the artifacts establish (verified 2026-10-26)

* **Same workload.** Arrival offsets are byte-identical across the five
  treatments within each campaign, and across S1, S2 and O in 50/50 units.
  Request identity (`request_id`, `prompt_len` sequence) is equal in 50/50
  units. The H5 arrival statistics are therefore the same data in all three.
* **Same engine and controller code.** Between the September source archive
  and the executed October bundle, every `src/` file and `ablation_live.py`
  / `final_hardening.py` are identical up to line endings (AST-equivalent where
  the bytes differ). Only the experiment driver (`subcritical_poisson.py`) and
  `source_provenance.py` differ, and O adds `saturated_reference.py`.
* **S2 matches S1's protocol.** Identical fingerprint dict (pre-registration
  hash, source manifest, seeds, grid, μ/λ, reward, convergence-rule and arm-space
  hashes), identical embedded pre-registration and sidecar.
* **S2 was not interrupted.** `checkpoint.resumed = false`, no skipped or rerun
  units, argv without `--resume`, and a log with one header and no OOM,
  traceback or resume line.
* **Validity gates** held in every campaign: every adaptive unit withheld
  exactly 3 leading cycles and none after the first valid reward; every reward
  gate passed; convergence 0/10 for every controller.

### Environments and interruptions

| | S1 | S2 | O |
|---|---|---|---|
| host | `0380adb5e71f` | `fdbf114ae08c` | `4cc7b9406f20` (both sessions) |
| Python / torch / CUDA | 3.13.15 / 2.11.0+cu128 / 12.8 | 3.13.15 / 2.11.0+cu128 / 12.8 | 3.13.16 / 2.11.0+cu130 / 13.0 (cuDNN 92700, transformers 5.20.2) |
| engine warm-up | none | none | `engine_warmup/1`, run in both sessions |
| interruption | CUDA OOM during `PagedKVCache` construction after CARL-Expanded 6/10; resumed without warm-up (`74 units recorded complete, 28 sweep rows restored`); the first resumed unit (Expanded seed 48) shows a 1,926 ms request-0 TTFT | none | CUDA OOM while constructing Repaired seed 50; resumed with warm-up; 12 units measured in session 1 |

Cause of both OOMs: unresolved (no memory telemetry).

---

## 5. Chronology

| time (UTC) | event | source | strength |
|---|---|---|---|
| 2026-09-18 14:51 | September pre-registration committed locally (`b455cc2`); not shown as pushed (origin refs point at `a64d918`) | `git log` | strong |
| 2026-09-18 14:52 | September sidecar created (`created_utc` 14:52:08) | sidecar | strong |
| 2026-09-18 → 21:21:57 | **S1** runs, is interrupted by an OOM, resumes, completes | S1 ZIP, log | strong |
| 2026-09-19 ≤ 13:17 | S1 aggregate results, selections and realised offered load transcribed as "FROZEN SUB-CRITICAL POISSON RESULT" into a cold-review prompt | Codex history DB `C:\Users\Vidhu\.codex\thread_history_1.sqlite`: imported Claude session thread `01a11295-0124-…` (created 13:20:55) and a captured copy of ChatGPT thread `6a8f4b40-…` | strong (verbatim user prompt) |
| 2026-09-19 15:17 | **Recorded reason for the S2 repeat:** an assistant reply in ChatGPT thread `6a8f4b40-…` (turn `bd92e4fd-…`), answering a user-attached screenshot that is not preserved, states that the Colab runtime had reset, that "anything that existed only under `/content` is gone", that the JSON/raw artifacts were needed, and recommends rerunning once with every checkpoint on Google Drive. The same reply says "we do not need to change any code or science". | same DB, captured thread copy | medium (assistant text captured by a tool call; the user's screenshot is not available) |
| 2026-09-19 ≈15:22 → 21:09:32 | **S2** runs uninterrupted | S2 ZIP, log | strong |
| 2026-09-19 21:11 | S2 results discussed (selections (32, 256)/(12, 512), A −0.0520 [−0.1179, +0.0140], B −0.1580, …); the discussed figures match the S2 artifact to four decimals | same DB vs S2 results | strong |
| 2026-10-12 | a local agent session notes that "later chats discuss completed subcritical runs, but the local ledger still marks the study pending" | same DB (tool-call record in a session otherwise unrelated to this project; only the matching row was read) | medium |
| 2026-10-21 | read-only review returns NO-GO; corrections; amendments proposed (Student-t CI, H2 denominator, H5 multiplicity) | Codex session `01a11295-0214-…`; Claude session `b183f86d-…`; pre-registration corrections log | strong |
| 2026-10-22 | amendments recorded as "authorised … before any GPU run"; `/3` smoke; engine warm-up amendment (after inspecting the smoke); saturated reference recovered (21:30) — `D:\subcritical_poisson_FINAL.zip` appears in a file listing but is not opened | pre-registration; Codex session | strong |
| 2026-10-23 16:19:51 | October sidecar and bundle created | sidecar | strong |
| 2026-10-23 17:13:32 → 23:07:37 | **O** runs; OOM at 22:09:39; resumed 22:16:13 | `run_history` | strong |
| 2026-10-25 | an audit lists `D:\subcritical_poisson_FINAL.zip` as "Not used … predate[s] this run" without escalating it | Codex session `01a11295-0214-…` | strong |
| 2026-10-26 | S2 files recovered from Google Drive | author's report; file times | — |

**Prior exposure, retained.** S1 results (aggregates, selections, realised
offered load) were viewed and discussed by 2026-09-19 13:17, and S2 results by
2026-09-19 21:11. Both campaigns' logs print the n = 200 arrival-validation
statistics for seed 42 and the realised offered load per phase. The October
pre-registration's statement that "local evidence cannot show whether any
earlier session viewed the realised n = 200 means" is superseded: local
evidence shows that the September runs computed and printed them and that the
aggregates were viewed.

**Not established:**
* **Why O was run** rather than reporting S1/S2. No contemporaneous record
  giving a reason was located. None is inferred here.
* **Whether S1/S2 results influenced any October amendment.** The amendment
  texts do not mention S1/S2; the reviews that prompted them did not either.
  Absence of a mention is not evidence of no influence.
* **Whether other full or partial executions occurred.** The recorded S2
  instructions begin with `rm -rf /content/drive/MyDrive/CARL_subcritical_final`
  followed by `mkdir -p`, so any earlier contents of that folder were removed,
  and earlier Colab state under `/content` was not preserved. Additional
  attempts therefore cannot be ruled out from the available records.

---

## 6. The "before any GPU run" statements

In the October pre-registration (`e3776082…`):

| line | wording | correction of interpretation |
|---:|---|---|
| 6 | "**Registered:** before the GPU run." | True only in the sense that the document was hash-bound before **O**. S1 and S2 had already run the same seeds. |
| 394 | "*Amended 2026-10-22, before any GPU run; …*" | Incorrect: S1 and S2 had completed and been discussed. |
| 1152 | "**2026-10-21 — factual and execution corrections, before any GPU run.**" | Incorrect for the same reason. |
| 1180 | "**2026-10-22 — authorised statistical amendments, before any GPU run.**" | Incorrect for the same reason. |
| 1218–1232 | prior-exposure disclosure listing only development and test generation | Incomplete: omits S1 and S2 (§5). |

In the September pre-registration (`b455cc2`), line 6 reads "**Registered:**
before the GPU run." That holds for S1; S2 re-used the same document after S1
had completed and been discussed. Both S1 and S2 artifacts record
`pre_registration.registered_before_run = true`; for S2 that flag cannot mean
"before any result existed".

---

## 7. What changed between September and October

**Analysis changes (scientific, amendments of 2026-10-22):**

| element | September (S1, S2) | October (O) |
|---|---|---|
| primary paired CI | normal approximation, mean ± 1.96·SD/√n | Student-t, mean ± t₀.₉₇₅,₉·SD/√n; z kept as labelled secondary |
| H2 metric | "\|mean paired difference\| / mean level, comparisons A, C, D" (denominator unspecified) | denominator = comparator's mean throughput; B excluded |
| H5 rule | per-cell 95% normal interval, every phase and seed; a miss voids load-regime conclusions | 20-cell family, exact Gamma(k, kλ) bounds at α = 0.05/20; seed 999 separate |
| descriptive reporting | — | phase / regime / drain / mixed-execution reporting added |

The H5 change did not change an outcome: the arrival data are identical, and
20/20 cells fall inside the original per-cell rule (recorded in S1 and S2) and
0/20 outside the amended rule (recorded in O).

**Operational changes (October):** atomic writes; per-sweep-point evidence
files; calibration digest bound to every evaluation; validation of fresh and
resumed evidence; archive digest fixed and `--verify-archive`; a T4
requirement; runtime and model identity bound into the fingerprint; separation
of smoke and published paths; verified saturated reference in place of the
hard-coded transcription; experiment version `/1` → `/4`. **The engine warm-up
is operational in intent but changes what is measured** (it removes first-use
cost from the first measured unit), and October also ran on a different
CUDA/torch build. Their effects cannot be separated from each other or from
run-to-run variation.

---

## 8. Results across the three campaigns

Paired over seeds 42–51 (n = 10), a − b. Recorded primary interval: normal (z)
for S1/S2, Student-t for O. `d_z` = mean / SD (n − 1) of the paired differences.
Signs are +positive/−negative seeds. All values re-derived from raw per-seed
files; every recorded value reproduced. Exact two-sided Wilcoxon signed-rank p-values are recorded in all three campaigns' results and are given in the manuscript appendix; they are robustness checks, not the headline.

| | metric | S1 mean [z CI] | S2 mean [z CI] | O mean [t CI] |
|---|---|---|---|---|
| A | throughput | −0.0764 [−0.1379, −0.0148], +2/−8 | −0.0520 [−0.1179, +0.0140], +3/−7 | −0.0286 [−0.0459, −0.0113], +2/−8 |
| B | end-of-run `utility_v2` | −0.1925 [−0.2156, −0.1693], +0/−10 | −0.1580 [−0.1836, −0.1323], +0/−10 | −0.1396 [−0.1593, −0.1200], +0/−10 |
| C | throughput | −0.0427 [−0.1009, +0.0154], +3/−7 | −0.0144 [−0.0560, +0.0273], +3/−7 | −0.0325 [−0.0583, −0.0066], +2/−8 |
| C | end-of-run `utility_v2` | −0.0229 [−0.0289, −0.0170], +0/−10 | −0.0163 [−0.0217, −0.0109], +0/−10 | −0.0196 [−0.0252, −0.0139], +0/−10 |
| D | throughput | −0.0018 [−0.0357, +0.0320], +5/−5 | +0.0334 [−0.0266, +0.0935], +5/−5 | +0.0031 [−0.0162, +0.0224], +4/−6 |
| D | end-of-run `utility_v2` | +0.0009 [−0.0070, +0.0088], +4/−6 | +0.0012 [−0.0051, +0.0075], +5/−5 | −0.0092 [−0.0171, −0.0014], +1/−9 |

* A and C throughput means are negative in all three campaigns; whether the
  interval excludes zero varies. A change in whether an interval crosses zero
  is not itself evidence that the effects differ.
* D changes direction across campaigns (throughput −/+/+, utility +/+/−).
* B and C utility are negative on all ten seeds in every campaign.
* Utility magnitudes are not comparable across campaigns: each campaign froze
  its own scales and selected its own static baselines (S1 (16, 256)/(16, 512);
  S2 (32, 256)/(12, 512); O (4, 512)/(12, 512)).
* The campaigns are three measurements of the same ten workloads. They are not
  pooled, and the reused seeds are not 30 observations.

---

## 9. Consequences for reporting

* All three campaigns are reported, each with its own recorded analysis and
  protocol; retrospective common-method intervals are labelled as such.
* O may carry more methodological detail because its provenance and
  instrumentation are stronger; it is not presented as the only campaign, and
  no campaign is chosen as headline on the basis of its results.
* Unqualified "preregistered" or "confirmatory" wording for the sub-critical
  point is withdrawn. Elements specified in the September pre-registration
  (design, seeds, n, grid, ρ, μ/λ, treatments, reward, convergence rule, and
  the no-winner stance of H3) are distinguished from the October amendments
  (Student-t primary, H2 denominator, H5 family rule, descriptive reporting).
* No new GPU work is started without a demonstrated validity defect.

Documents aligned with this record: `docs/eval/CARL_REPAIR_STATUS.md`
(Phase 24), `docs/paper/CLAIM_LEDGER.md` (section E),
`docs/paper/RESULTS_DECISION.md`, and the manuscript
`D:\CARL_THREE_CAMPAIGN_MANUSCRIPT.tex`.
