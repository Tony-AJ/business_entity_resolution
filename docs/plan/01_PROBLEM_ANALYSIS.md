# 01 — Problem analysis

Audience: every team member. Read this before your own module document.

Every statement is tagged: **[A]** stated in the official documents (source in brackets:
PS = `docs/PROBLEM_STATEMENT.md`, GL = `docs/GUIDELINES.md`, README = the organisers'
`dataset/student_resource/README.md`), **[B]** technical inference from the documents or the
data, **[C]** recommendation based on research and measurement. Nothing tagged [A] goes
beyond the documents; if the PDF and our summary disagree, the PDF wins.

## 1. The task

- [A] Business records arrive from three independent sources with no shared identifier.
  Source 1 is the deduplicated reference. For every Source 1 entity, output every Source 2
  and Source 3 record that describes the same real-world business. A Source 1 entity can
  match zero, one or many records (PS §Task, README).
- [A] Every record has `entity_id` (prefix `S1-`/`S2-`/`S3-` gives the source),
  `business_name`, `business_address`, `country` (README).
- [A] Ground truth (`train_ground_truth.tsv`) has `source1_entity_id` and
  `matched_entity_ids`, a comma-separated list of S2/S3 ids, empty for no match (README).
- [B] The ground truth is a many-to-one mapping from S2/S3 records to S1 entities. Measured on
  train: every matched S2/S3 id appears in exactly one list, so a pool record belongs to at
  most one S1 entity. S2 and S3 are not deduplicated against each other: an S1 entity has on
  average 3.7 matches (max 11), spread over both sources.
- [B] "Zero, one or many" means the output per entity is a **set**, not a top-1 choice.
  Singletons (5.6% of train S1) are entities with an empty list.

## 2. Sources, in numbers

| Split | S1 | S2 | S3 | Countries |
|---|---|---|---|---|
| train | 2,206,821 | 5,034,616 | 5,285,603 | US 60%, India 40% |
| test | 1,732,544 | 4,887,273 | 5,082,316 | India 47%, US 38%, France 15% |

[B] Train ground truth: 7,638,365 pairs (S2 3.69M, S3 3.94M); 123,247 singletons (5.58%);
matches per matched entity: 1: 119k, 2: 375k, 3: 531k, 4: 484k, 5: 322k, 6: 165k, 7+: 88k.
About 26% of the train pool (2.68M records) is unmatched. The test pool/S1 ratio is 5.8
against 4.7 in train, so roughly 40% of the test pool is unmatched [B]: more distractors per
entity than in our validation fold.

## 3. Noise patterns

[A] Names: abbreviations (Corp/Corporation, Pvt/Private, Ltd/Limited), legal-suffix
inconsistencies, DBA/trade names, punctuation (& vs and), word-order transpositions, typos,
transliterations. Addresses: abbreviations (Rd/Road, St/Street), transliteration variants,
missing components (no PIN, no state), landmark references ("Near SBI ATM"), municipal
numbering formats, component reordering (README, PS §Noise).

[B] Measured on 220k true pairs (60k S1 sample):

| Observation | Share |
|---|---|
| Names identical as written | 4.6% |
| Identical after accent-strip + casefold + punctuation removal | 25.6% |
| Identical after also removing legal-form tokens ("core name") | 48.4% |
| First core token identical | 75.7% |
| Zero shared name tokens | 14.7% |
| … of which pool name in an Indic script (Devanagari, Kannada, Tamil, Telugu, Malayalam, Odia, Gujarati) | 47% |
| … of which domain / handle forms (`clumora.com`, `@sakashpoint`, `#cúlturalguild`, `ORTHOPEDICHEALTHCOM`) | 35% |
| … of which full renames or heavy typos (`Ciro Builders Private Limited` ↔ `Dovaflux`, `Renet` ↔ `Rnht`) | 18% |
| Pool addresses containing a non-Latin token (usually the state: `ಕರ್ನಾಟಕ`, `महाराष्ट्र`) | 9.2% |
| Pool addresses empty | 4.4% |
| Address token Jaccard = 0 | 4–5% |
| Some shared numeric token in the address | 78% (US), 83% (India) |
| Address contains a 5-digit code (US) / 6-digit PIN (India) | ~10% / ~0% |

[B] The generator also uses leet substitutions (`Jeanl0uis`, `5tar Flxe`), token replacement by
generic words (`N+ Rare Inc` → `N+ Inc Services`), legal-form moves (`LLC Moncada Learning
Center`, `Pvt. EFS Print Ventures Ltd.`), injected accents (`Léarning`, `Àmicale`), prefix
junk (`-- Holloway`, `<< Team`, `##4805`), full-uppercase addresses, state written in full,
abbreviated or in a regional script, and address components dropped or reordered.

[B] Name ambiguity is extreme: 48% of S1 records share their core name with another S1 record
(`meridian`, `summit`, `cedar`, `family center`…). 62% of singletons have their core name
present in the pool of the same country. 21% of unmatched pool records carry the core name
of some S1 entity. Therefore the address, not the name, separates true matches from decoys.

## 4. Country as an open set

- [A] Train covers US and India; test adds France. Do not hard-code, filter or one-hot to
  {US, India}; every test entity, France included, must appear in the output (README, PS).
- [B] 100% of true pairs share the country string. Grouping records by whatever country value
  appears is therefore safe and cuts the search space by ~3× without enumerating countries.
- [C] Country enters the pipeline only as a partition key and as the "same country" fact.
  No per-country thresholds or per-country models: they cannot be fitted for France. French
  legal forms (SARL, SAS, SASU, EURL, SCI, SA) and address tokens (Rue/R., Avenue/Av,
  Boulevard/Bd, Chemin, Impasse, Allée) go into the static normalisation maps so France is
  treated like any other country. Sanity-check the France slice of the test output (match
  rate, candidates per entity, score histogram) against US/India before every upload.

## 5. Singletons

- [A] A Source 1 entity with no true matches scores 1.0 for an empty prediction and 0.0 for
  any prediction (README §Evaluation, PS §Metric).
- [B] Singletons are 5.6% of entities, so perfect singleton handling is worth up to 0.056
  macro-F0.5 and every false merge on a singleton costs a full point.
- [C] Detect singletons by the absence of a confident candidate (`p_max < τ_single`), not by
  the absence of a same-name record: 62% of singletons have a name twin in the pool.

## 6. Candidate generation and the two output files

- [A] `candidate_pairs.tsv` (`source1_entity_id`, `candidate_entity_ids`) is the exact set of
  records the matching model scored at inference: the last blocking/filtering stage, not an
  early pass. Every matched id should appear in it; the validator warns otherwise. It is not
  scored; organisers use it to analyse recall ceiling and reduction ratio (README).
- [A] `matching_results.tsv` (`source1_entity_id`, `matched_entity_ids`) is the only scored
  file and the one uploaded to the portal (README).
- [B] Blocking is the stage that turns 1.7M × 10M possible pairs into a few tens of candidates
  per entity. A true match missing from the candidates can never be recovered downstream, so
  blocking sets the recall ceiling and the organisers audit it.
- [C] Write `candidate_pairs.tsv` from the same in-memory pairs frame that was scored (the
  pipeline's `score()` input), never from an earlier pass.

## 7. Submission constraints (all [A], README §Constraints and §Output, PS §Output)

1. Tab-separated files with exactly the column names above.
2. Exactly one row per test Source 1 entity; missing entities cause rejection.
3. `matched_entity_ids` empty for singletons; comma-separated, no quoting.
4. Only S2/S3 ids that exist in the test set; no self-matches; unknown ids cause rejection.
5. No duplicate ids within a list; no duplicate `source1_entity_id` rows.
6. Files that fail validation are not evaluated; a correctly formatted upload shows `SCORED`.
7. Final model: MIT or Apache-2.0 licence, at most 8 billion parameters.
8. Check locally with `utils/validate_submission.py` (stdlib only) before uploading.

## 8. The metric

- [A] F0.5 = 1.25·P·R / (0.25·P + R), computed per Source 1 entity, then averaged over all
  Source 1 entities of the evaluation set (macro). Public leaderboard = a subset of test;
  private = the remainder; final ranking uses the private part. Predictions are submitted for
  the full test set (README §Evaluation, §Leaderboard).
- [A] Worked example: predicted {S2-00047, S2-00193, S3-00812}, truth {S2-00047, S3-00812}:
  P = 2/3, R = 1, F0.5 = 0.714.
- [B] Per-entity arithmetic that shapes the strategy (n true matches, k predicted, t correct):

| Situation | P | R | F0.5 |
|---|---|---|---|
| all n found, nothing extra | 1 | 1 | 1.000 |
| half of 4 found, nothing extra | 1 | 0.5 | 0.833 |
| 1 of 4 found, nothing extra | 1 | 0.25 | 0.625 |
| all 4 found plus 1 false | 0.8 | 1 | 0.833 |
| all 4 found plus 4 false | 0.5 | 1 | 0.556 |
| 1 prediction, wrong; entity has matches | 0 | 0 | 0.000 |
| singleton, empty prediction | – | – | 1.000 |
| singleton, any prediction | – | – | 0.000 |

- [B] A false merge costs as much as missing half of the matches; a wrong-only prediction
  costs everything. Precision counts twice as much as recall, so the decision layer should
  drop uncertain candidates rather than keep them, and prefer an empty list over one doubtful
  match.
- [B] Macro averaging weights every entity equally: an entity with 1 match matters as much
  as one with 11. Errors on small entities (1–2 matches) are proportionally the most
  expensive, and singleton errors are binary.
- [B] Our implementation (`src/entity_resolution/metrics.py`) applies exactly these rules:
  empty truth → 1.0 / 0.0; no true positives → 0.0; entities absent from the prediction
  count as empty; predictions are sets (duplicates collapse). The 0.714 example is a unit test.

## 9. External data prohibition

- [A] No external databases, APIs or services to look up business identities: no commercial
  ER APIs, no government registries, no geocoding APIs, no internet augmentation. Evidence of
  lookup means immediate disqualification; pipelines are reviewed (README §Fair Play, PS).
- [C] Architecture consequences: every dictionary (legal forms, street abbreviations, state
  names) is either hand-typed domain knowledge or **learned from the provided training pairs**;
  no gazetteers, no libpostal/OpenStreetMap-trained parsers, no pretrained address models.
  Pretrained text encoders are allowed by the licence rule only as models, and we use none
  in the MUST-have path. Everything the pipeline reads is under `dataset/`.

## 10. Licensing, parameters, reproducibility

- [A] Final model MIT or Apache-2.0 licensed, ≤ 8B parameters (README §Constraints 5).
- [A] Package: `output/` with both TSVs; `code/business_entity_resolution/` with `src/`,
  `README.md` (exact reproduction steps: data → blocking → matching → output),
  `requirements.txt`; filled `Documentation_template.md`. Anyone should be able to regenerate
  both output files from the training/test data using only that folder; top packages are
  reproduced and audited (README §Final Submission Package, GL).
- [A] Keep the version history of all submissions; source code for experiments, training and
  inference with proper function comments; 1–2 page methodology document (GL).
- [C] Our libraries are MIT/BSD/Apache/ISC only (LightGBM MIT, scikit-learn BSD, rapidfuzz
  MIT, sparse_dot_topn Apache-2.0, anyascii ISC); GPL packages (`unidecode`,
  `python-Levenshtein`) are banned. Fixed seeds (`config.SEED = 42`, split seed 42, inner
  split seed 4242), hash-based splits, pinned versions, and one notebook per version make
  every logged score reproducible from its commit.

## 11. Submissions and timeline (GL)

- [A] Window 25 Sep 00:00 to 27 Sep 23:59 IST. At most 5 uploads per day, 15 in total; the
  button is disabled afterwards. Public and private leaderboards; shortlisting uses both.
- [B] Fifteen uploads cannot explore anything. The validation fold decides; the leaderboard
  confirms (see `15_LEADERBOARD_STRATEGY.md`).

## 12. What follows from all this [C]

1. **Recall comes from blocking, precision from the decision layer.** Blocking must reach
   ≥ 0.97 pair recall on the val fold with ≤ 40 candidates per entity, using name **and**
   address channels, inside the country partition.
2. **The address is the discriminator.** Name features rank; address features decide.
   Singletons and same-name decoys are separated by address disagreement.
3. **One pool record, one owner.** Enforce the measured 1-to-1 property at decision time.
4. **Sets, not top-1.** Thresholds relative to the best candidate plus an absolute floor,
   tuned directly for macro F0.5 on an inner split of the training fold.
5. **Country-agnostic everything**, with France checked on the test output.
6. **Measure locally, upload rarely.** Fixed val fold, fixed inner split, one experiment =
   one version = one logged row.
