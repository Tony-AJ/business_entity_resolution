# 03 — Team working strategy (5 people, 3 days, one pipeline)

Audience: everyone. Read `01_PROBLEM_ANALYSIS.md` and `02_SYSTEM_ARCHITECTURE.md` first, then
your own module document. This document says who owns what, in which order things land, and
how the pieces come back together without anyone waiting on anyone else.

## 1. Principle: one module per person, one contract for all

The pipeline is five swappable stages behind fixed DataFrame contracts (02 §4–5). Each
member owns exactly one module, its tests and its experiment versions. Nobody edits another
member's module; a needed change is a request to its owner (or a PR the owner reviews). The
lead ships a **walking skeleton** on day 1 — every stage implemented in the simplest possible
way — so that from ~17:00 IST today each member replaces one stage and measures the effect
on the same validation fold with the same metric.

## 2. Roles

| Member | Role | Owns (code) | Owns (experiments) | Never touches |
|---|---|---|---|---|
| **M1** | Lead / integration / baseline | `pipeline.py`, `evaluate.py`, `trainset.inner_split`, additions to `submission.py` and `tracking.py`, `.gitattributes`, `README.md`, `LEADERBOARD.md`, final package + `Documentation_template.md` | v001–v009 (skeleton, baseline), v100–v129 (integrated runs; every upload) | other members' modules except by their PR |
| **M2** | Normalisation + blocking | `normalize.py`, `blocking.py`, learned token maps | B1–B5, A1–A5 → v010–v039 | features, model, decision |
| **M3** | Pair features + training pairs | `features.py`, `trainset.py` (except `inner_split`) | C1–C5 → v040–v059 | blocking, model, decision |
| **M4** | Models + hard negatives | `model.py`, `trainset.mine_false_positives`, hard-negative helpers | D1–D4, HN1–HN5 → v060–v079 | features, decision |
| **M5** | Decision layer + error analysis | `decision.py`, `errors.py`, slice/error reports | E1–E5 → v080–v099 | upstream modules |

With four people: M3 also takes M4's work (features and model are the closest pair). With
six: split M2 into normalisation and blocking.

## 3. Per-member brief

Each row: inputs you receive → outputs you must produce → metrics you report → tests you own.
Details in the linked document.

**M1 — Lead.** Inputs: nothing (you go first). Outputs by 17:00 IST day 1: `trainset.inner_split`,
`evaluate.py`, `pipeline.py` with the `heuristic` matcher and a fixed `DecisionRule`,
`submission.write_pairs`, `tracking.new_experiment(number=)`, `make experiment V=`,
`.gitattributes`; `v001_baseline` notebook logged in `experiments.csv`; first upload. Then:
merge PRs (gate in 14), rerun the integrated pipeline as v10x after every merge, own the
upload decisions (15), fill the README reproduction section and the methodology document
on day 3 (16). Metrics: val `breakdown`, harder-val, per-slice, public score. Tests:
`test_pipeline.py`, `test_evaluate.py`, `test_trainset.py::test_inner_split_*`.

**M2 — Normalisation + blocking** (05, 06). Inputs: raw source frames. Outputs: normalised
frames (02 §4.1), candidate pairs (02 §4.2), the `pairs` Parquet cache for fit/tune/val/test.
Targets: pair recall ≥ 0.97, cands/S1 ≤ 40, val blocking ≤ 10 min, RAM ≤ 3 GB. Metrics:
`blocking_report` (pair recall, entity recall, ceiling F0.5, cands mean/p95/max), per-pass
and per-slice recall, seconds, peak RSS. Tests: `test_normalize.py`, `test_blocking.py`.
Order of work: B1–B2 + A1 (today), A2–A3 (today evening), B3–B4 + A4–A5 (day 2), P4 and
learned name map only if recall is short (day 2–3).

**M3 — Features + training pairs** (07). Inputs: pairs frame + normalised frames from the
cache (use v001's cache until M2's lands). Outputs: `build_features`, `feature_names`,
`sample_s1`, `label_pairs`. Metrics: importance top 20, tune AUC (diagnostic), val F0.5
through the integrated model. Tests: `test_features.py`, `test_trainset.py` (labels,
sampling). Order: C1 name groups (today), C3 address group (today, highest value), C2/C4
(day 2), C5 context + interactions (day 2).

**M4 — Models + hard negatives** (08, 09). Inputs: feature frames + labels from M3 (use the
v001 heuristic features until then: the three blocking sims). Outputs: `Matcher` with
`lgbm`/`logreg`/`heuristic` backends, saved models, `mine_false_positives`. Metrics: tune
logloss/AUC (diagnostic), val F0.5 and singleton F0.5 after the decision layer, train time.
Tests: `test_model.py`, hard-negative tests. Order: D1 LR + D3 LightGBM (today), parameter
grid (day 2 morning), HN2–HN4 (day 2), D4 only if LightGBM plateaus (day 3).

**M5 — Decision + errors** (10, 18). Inputs: scored pairs on the tune split (from
`pipeline.fit`) and on val. Outputs: `decide`, `tune`, `tag_errors`, `slice_report` content,
the error report in every notebook §6. Metrics: tune vs val F0.5 gap, singleton F0.5,
false-merge and false-singleton counts, F0.5 impact per error category. Tests:
`test_decision.py`, `test_errors.py`. Order: E1–E2 with 1-to-1 (today, on v001 scores),
E3–E4 (day 2), E5 expected-F0.5 decoding (day 2–3), error taxonomy from day 1 evening.

## 4. Day-1 sequencing (today, Friday 25 Sep, IST)

| Time | M1 | M2 | M3 | M4 | M5 |
|---|---|---|---|---|---|
| 15:00 | read 01–03, own doc; branch | same | same | same | same |
| 15:30 | inner split, evaluate, tracking `V=`, `.gitattributes` | `normalize.py` R0–R5 + tests | `features.py` skeleton, registry, `iter_chunks` | `model.py` skeleton, `heuristic` + `logreg` | `decision.py` `decide` + `tune` + tests |
| 17:00 | **v001 walking skeleton merged**; caches for fit/tune/val/test start | `blocking.py` P1 + P2 | name groups on v001 pairs | LightGBM backend | E1/E2 on v001 scored pairs |
| 19:00 | v001 val score logged; upload #1 | A1–A2 logged | C1 + C3 logged | D1/D3 on M3's features | error taxonomy v1 |
| 21:00 | integrate merged PRs → v100; upload #2 if ≥ baseline + 0.05 | A3 | C2 | D3 tuned | E2 with 1-to-1 |
| 23:00 | precision-max upload #3; day-2 plan posted | – | – | – | – |

Everyone posts (version, val F0.5, delta vs parent, KEEP/DROP) at 13:00 and 21:00 IST.

## 5. Unblocking rules

- The **cache is the interface**: `pipeline.prepare` writes normalised frames and pairs
  under `dataset/.cache/pipeline/<tag>/<country>/`. M3–M5 read whatever the latest merged
  M2 version produced; they never wait for the best blocking.
- The **heuristic backend** (`prob = nanmax of blocking sims`, 1.0 for exact pairs) makes
  the pipeline runnable before any model exists; M5 tunes rules on it from hour 3.
- **Contract change** = PR that updates 02 and every dependent document, reviewed by the
  owners of the dependent modules. Adding a column is free; renaming or removing is not.
- **Version numbers** come from your reserved range and are passed explicitly
  (`make experiment NAME=<slug> V=<n>`); `experiments.csv` merges by union.
- **One machine, 15 GB**: run the full test blocking only from the integrated branch (M1),
  cache it, and reuse it for every later model version; develop on the val fold or on a
  200k-S1 sample of it.

## 6. Integration path

1. Member finishes a version on their branch: code + tests + notebook + `metrics.json` +
   CSV row, `make lint test` green, PR opened with the val report pasted.
2. M1 checks the gate (14 §5), merges with a merge commit, reruns `pipeline.fit` +
   `run_fold(val)` on `main` as the next v1xx, logs it, and posts the delta.
3. Only v1xx versions are uploaded. A member's improvement "counts" when the integrated
   v1xx improves; if it does not, M1 and the owner investigate together (usually thresholds
   need retuning after upstream changes — M5 reruns `tune`).
4. Day 3 18:00 IST: freeze. After that only fixes that make the validator pass or restore a
   regression are merged.

## 7. Communication

- One chat thread per module plus one for integration. Version, metric delta and decision in
  every status message; links to the notebook.
- Questions about the contract go to M1; questions about a module go to its owner.
- No meeting longer than 15 minutes; the 13:00/21:00 posts replace stand-ups.

## 8. What each member reads

| Member | Read fully | Skim |
|---|---|---|
| M1 | 00–03, 11–16 | 04–10, 17, 18 |
| M2 | 01–03, 05, 06, 12–14 | 07, 11, 18 |
| M3 | 01–03, 07, 11–14 | 05, 06, 08 |
| M4 | 01–03, 08, 09, 11–14 | 07, 10 |
| M5 | 01–03, 10, 11, 18, 12–14 | 07, 08, 15 |
