# 15 — Leaderboard strategy

Audience: M1 operates every upload; everyone reads §4–§5 so nobody argues from a public score.
Local macro F0.5 on the fixed val fold decides (13 §3, §5); the public score is logged with
`make public V=vNNN SCORE=x` and used only as a consistency check. Only
`output/matching_results.tsv` is uploaded; `candidate_pairs.tsv` ships in the final zip (16).

## 1. Budget

| Day | Date (IST) | Slots | Planned use |
|---|---|---|---|
| 1 | Fri 25 Sep | 5 | 3 planned: v001 calibration, best V1, precision-max probe; 2 optional |
| 2 | Sat 26 Sep | 5 | 5, one hypothesis each, ≥ +0.003 val over the last upload |
| 3 | Sun 27 Sep, last upload 22:00 | 5 | 3 candidates, 1 safety re-upload of the best, 1 last-resort |

Five per day, fifteen in total. Unused slots lapse at 00:00 IST (assumed; the organisers do
not say they carry over, so never plan on it). A wasted slot is acceptable only when no
eligible version exists; an upload of an untested change never is, because a validator
rejection or a France-slice failure burns the slot and teaches nothing. Expect 10–20 local
versions per upload; the csv is where exploration happens.

## 2. Eligibility checklist (M1, before every upload, on the exact file that goes up)

```
- [ ] v1xx integrated version (13 §5) with a row in experiments.csv: local_f05 on the fixed val fold, decision KEEP
- [ ] harder_f_beta ≥ last uploaded version's harder_f_beta − 0.002 (evaluate.harder_fold)
- [ ] code, notebook, metrics.json committed and merged into main; csv commit column has no -dirty suffix
- [ ] run_test executed from that commit; output/*.tsv newer than the commit; no manual edit afterwards
- [ ] make validate PASS (our checker, then the organisers' validator), then both again with --check-ids:
        .venv/bin/python -m entity_resolution.submission --output-dir output --check-ids
        .venv/bin/python dataset/student_resource/utils/validate_submission.py --matching output/matching_results.tsv \
            --candidate output/candidate_pairs.tsv --test-dir dataset/student_resource/dataset/test --check-ids
      no "matches that are not candidates" warning
- [ ] test-slice sanity (11 §10; evaluate.slice_report on the test output, no labels needed):
        rows: 1,732,544 data rows in both files (wc -l prints 1,732,545 with the header)
        France present, ~15 % of rows; non-empty match rate within 10 points of US and India; cands/S1 similar
        US and India: matches per S1 within ±0.3 and singleton (empty) rate within ±5 points of the val fold
        cands_mean ≤ 40, cands_p95 ≤ max_per_s1; p_max histogram per country has the shape seen on val
- [ ] LEADERBOARD.md entry written BEFORE uploading (§6), with the configuration record and the predicted public score
- [ ] the uploaded bytes kept: cp output/matching_results.tsv experiments/v1xx_<slug>/artifacts/uploaded.tsv (local only)
```

## 3. Day-by-day plan (IST)

**Day 1, Friday 25 Sep**

| Slot | Time | Version | Purpose |
|---|---|---|---|
| 1 | ~18:00 | v001 baseline: trivial normalisation, exact block, heuristic matcher, fixed rule | calibration: first public−local offset; proves the portal accepts our format and ids |
| 2 | ~21:30 | best V1 (v101), only if val F0.5 ≥ v001 + 0.05 | first real model on the board; second offset point |
| 3 | ~23:00 | v102 = v101 with a higher `tau_abs` (precision-max variant) | how the public subset rewards precision and singletons; the only deliberate threshold-only pair |
| 4–5 | ≤ 23:30 | optional | only for a further eligible version; otherwise they lapse |

**Day 2, Saturday 26 Sep**: slots at 11:00, 14:00, 17:00, 20:00, 22:30. Each slot is one
hypothesis (one merged change integrated as a v10x, 14 §7), uploaded only if val F0.5 ≥ last
uploaded + 0.003 and harder-val does not regress. A slot with no qualifying version rolls
forward hour by hour; at 23:30 it lapses.

**Day 3, Sunday 27 Sep**: 11:00, 15:00, 18:00 candidates under the same rule; 20:30 safety
re-upload of the chosen final version (§7): we do not know whether the portal keeps the last
or the best upload, so the last upload and the zip carry the same bytes; 22:00 last-resort
slot, used only if the 20:30 upload failed or a packaging fix changed the file. Nothing after
22:00; 22:00–23:00 is packaging (16 §5).

## 4. Reading public scores

- Record every score at once: `make public V=vNNN SCORE=0.xxxx` (fills `public_f05` in the
  csv) and the LEADERBOARD.md entry, with the offset `public − local`.
- Keep the running mean and range of the offset in the LEADERBOARD.md budget section. Expect
  it to be negative and roughly constant: the test pool is ~40 % unmatched against ~26 % in
  our val fold and 15 % of test entities are French, unseen in training (01 §2, §4). Local
  F0.5 measures the model; the offset measures the distribution shift.
- After two or more uploads, an offset more than 0.02 away from the running mean stops all
  uploads until it is explained: check the France slice (match rate, cands/S1, `p_max`
  histogram against US/India), the validator warnings, the candidate cap (share of test S1
  hitting `max_per_s1` versus val), the row count, and whether the version changed the
  singleton rule. The finding goes into the entry; the next upload is the one that tests the
  explanation, at most one such diagnostic per day.
- Never tune a threshold, rule or feature to the public score: the private remainder of the
  test set decides the final ranking, and a public gain below 0.003 is subset noise, not
  evidence. The public score confirms or flags; it never selects.

## 5. Anti-overfitting rules

1. One hypothesis per upload; the LEADERBOARD.md entry names the single change against the
   previous upload.
2. No two uploads differing only in a threshold, except the deliberate day-1 precision-max
   probe (v101 vs v102), which measures the offset's sensitivity to precision, not a threshold.
3. The final upload, and the zip's `matching_results.tsv`, is the uploaded version with the
   best **local** val F0.5, not the best public score, unless the public gap is explained by a
   documented failure (France slice, candidate cap, validator warning) in the better-local one.
4. A version is uploaded once; the same bytes go up again only at the day-3 safety slot.
5. Thresholds come from `decision.tune` on the tune split (train-fold data only); the val
   fold is scored, never tuned on; the public score is logged, never tuned on.

## 6. LEADERBOARD.md entry and configuration record

Copy the template from `LEADERBOARD.md` (M1 edits that file only), newest first, written
before the upload and completed with the score after it:

```markdown
## Submission NN: YYYY-MM-DD HH:MM IST
- Version: vNNN_<slug>
- Commit: <short hash>          (merge commit on main; equal to the csv commit column)
- Change vs previous submission: <one hypothesis>
- Model: lgbm, num_leaves 63, lr 0.05, best_iteration NNNN, feature groups [blocking, name_fuzzy, ...]
- Blocking: config hash <8 hex> (dataset/.cache/pipeline/test/*/pairs_<hash>.parquet), passes P1a-c+P2+P3, k 30/20, max_per_s1 60
- Threshold / decision rule: tau_abs 0.62, tau_rel 0.25, tau_single 0.55, max_matches 11, one_to_one true
- Local F0.5 (validation fold): 0.xxxx   (harder-val 0.xxxx, singletons 0.xxxx, matched 0.xxxx)
- Public F0.5: 0.xxxx                    (offset −0.0xx; running mean −0.0xx over N uploads)
- Notes: test rows 1,732,544; France match rate 0.xx vs US 0.xx / India 0.xx; cands/S1 31.2; slots left today N
```

The configuration record (version, commit, blocking config hash, feature groups, model
params, rule) together with `metrics.json` regenerates any upload: check out the commit,
`make cache`, run the version's notebook §9 (test inference) or
`python -m entity_resolution.pipeline --config experiments/vNNN_<slug>/artifacts/config.json --run-test`
(the pipeline CLI is M1's; until it exists the notebook path is the reference).

## 7. Decision trees

Which version to upload at the next slot:

```
a KEEP v1xx exists with f_beta ≥ last uploaded + 0.003 and harder_f_beta not lower?
├─ yes → passes the eligibility checklist (§2)?
│        ├─ yes → upload; make public; compute the offset; post in chat
│        └─ no  → fix (validator, France slice, cap); rerun §2; not done within 60 min → go to "no" below
└─ no  → is there an open INVESTIGATE on the last offset (> 0.02 off the running mean)?
         ├─ yes → no upload until explained; the next upload is the diagnostic that tests the explanation (≤ 1/day)
         └─ no  → let the slot lapse; keep working locally
```

Final submission (day 3, decided by 20:00, re-uploaded at 20:30):

```
candidates = uploaded versions with a logged public score
best_local = the candidate with the highest local f_beta
its offset within ±0.02 of the running mean?
├─ yes → final = best_local
└─ no  → deviation explained (France, cap, validator) and fixed in a later upload?
         ├─ yes → final = the fixed version (highest local f_beta among normal-offset versions)
         └─ no  → final = highest-local version with a normal offset; the unexplained gap is noted in LEADERBOARD.md
re-upload final at 20:30; the same bytes go into the zip (16 §1); tag the commit final-submission
```

## 8. Fallback: the portal rejects an upload

A rejection (missing entity, unknown id, header, quoting) consumes the slot, and the fixed
file is another slot. Hence §2 runs with `--check-ids` before every upload, on the exact
file, and the file is not opened in an editor afterwards (no re-save, no line-ending change).
If a rejection happens anyway: paste the portal message into the LEADERBOARD.md entry,
reproduce it with the organisers' validator, fix in `submission.py` or `pipeline.run_test`
under a `fix:` commit, re-run `run_test`, re-validate, then use the next slot. A `SCORED`
result with an offset below −0.10 is treated as a silent format problem: check the header
bytes, `\n` line endings, unquoted ids, and that the row order covers every test S1 id.
