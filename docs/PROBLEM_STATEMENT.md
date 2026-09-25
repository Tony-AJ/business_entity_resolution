# Problem statement — Business Entity Resolution (Amazon ML Challenge 2026)

Condensed from the organisers' brief (`amazon_ml_challenge_problem_statement.pdf`,
kept outside the repo). When this summary and the PDF disagree, the PDF wins.

## Task

Business records arrive from 3 independent sources with no shared identifiers.
**Source 1 is deduplicated** and acts as the reference. For every Source 1 entity,
find all Source 2 and Source 3 records that describe the same real-world business.
A Source 1 entity can match zero, one or many records.

## Data

All files are **tab-separated** (addresses and ID lists contain commas):

```python
pd.read_csv(path, sep="\t", dtype=str, na_filter=False)
```

| File | Content |
|---|---|
| `dataset/train/train_source{1,2,3}.tsv` | training records |
| `dataset/train/train_ground_truth.tsv` | `source1_entity_id`, `matched_entity_ids` (comma list, empty = no match) |
| `dataset/test/test_source{1,2,3}.tsv` | test records, no labels |

Source columns: `entity_id` (prefix `S1-` / `S2-` / `S3-` gives the source; there is
no separate source column), `business_name`, `business_address`, `country`.

**Country is an open set.** Train covers US and India; test adds **France**, which
never appears in train. Do not hard-code, filter or one-hot to {US, India}; every test
entity, France included, must appear in the output.

### Noise to expect

- **Names**: abbreviations (Corp/Corporation, Pvt/Private, Ltd/Limited), legal-suffix
  drift, DBA/trade names, `&` vs "and", word-order swaps, typos, transliterations.
- **Addresses**: abbreviations (Rd/Road, St/Street), transliteration variants, missing
  parts (no PIN code, no state), landmark references ("Near SBI ATM"), municipal
  numbering formats, reordered components.

## Output

Two TSV files in `output/`, both with one row per test Source 1 entity:

| File | Columns | Scored |
|---|---|---|
| `matching_results.tsv` | `source1_entity_id`, `matched_entity_ids` | yes (leaderboard) |
| `candidate_pairs.tsv` | `source1_entity_id`, `candidate_entity_ids` | no (blocking audit) |

```
source1_entity_id	matched_entity_ids
S1-00001	S2-00047,S2-00193,S3-00812
S1-00002	S3-00004
S1-00003	
```

Rules (a violation gets the file rejected):

- exactly one row per test Source 1 entity, no duplicate rows;
- ID lists: comma-separated, no quoting, no duplicates, only `S2-`/`S3-` IDs that exist in
  the test set (no self-matches to Source 1); empty field for no match;
- `candidate_pairs.tsv` is the **final** candidate set the matching model scores (after
  every blocking/filter stage), and every matched ID must appear in it.

Check both files locally with the organisers' stdlib-only validator:

```bash
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

## Metric

Macro-averaged **F0.5** over Source 1 entities, singletons included:

```
F0.5 = 1.25 * P * R / (0.25 * P + R)      computed per Source 1 entity, then averaged
```

- true singleton + empty prediction → 1.0; true singleton + any prediction → 0.0;
- precision weighs twice recall: a false merge hurts more than a missed link.

Example: predict {S2-00047, S2-00193, S3-00812}, truth {S2-00047, S3-00812} →
P = 2/3, R = 1, F0.5 = 5/7 ≈ 0.714.

The public leaderboard scores a subset of test; final ranking uses the private remainder.
There is no test ground truth: hold out a validation split of train and score it with the
same formula.

## Constraints

- **No external data**: no business-lookup APIs, government registries, geocoding
  services or internet augmentation. Evidence of lookup means disqualification.
- **Final model**: MIT or Apache-2.0 licence, at most 8B parameters.
- Output format must match exactly; failed validation is not scored.

## Final submission package

```
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/business_entity_resolution/     ← this repository
│   ├── src/                             all source code
│   ├── README.md                        exact steps: data → blocking → matching → output
│   └── requirements.txt                 pinned environment
└── Documentation_template.md            methodology write-up (organisers' template)
```

The methodology document covers: approach, candidate generation / blocking strategy,
model architecture and features, anything else relevant. Top teams' packages are
re-run and audited before final rankings.

## Organisers' tips

- Blocking sets the recall ceiling: invest in candidate generation.
- String similarity features for names and addresses: Jaccard, Levenshtein, TF-IDF cosine.
- Mind country-specific address patterns.
- Tune for precision; F0.5 rewards it.
- Predicting "no match" for a real singleton is worth a full 1.0.
