# 04 — Algorithm research: technique cards and the chosen stack

Audience: everyone, before deciding what to build in your module. One card per technique,
grouped by pipeline stage (A–G, matching the plan groups in
`.claude/rules/project-rules.md` §5 and the stages of `02_SYSTEM_ARCHITECTURE.md` §1). Every
claim about our data quotes a measurement from `01_PROBLEM_ANALYSIS.md` §3 (train, 220k true
pairs); `[n]` resolves in `17_RESEARCH_REFERENCES.md`. §8 states what we build and why the
rest is not built.

## 0. How to read the cards

- **Cost** is for the test run: S1 1.73M × pool 9.97M, processed per country (India 810k ×
  4.72M, US ≈ 0.8×, France ≈ 0.1×); ≈ 24M candidate pairs after blocking; 12 cores, 15 GB
  RAM with ≤ 6 GB per stage (02 §7); the 4 GB GPU is optional and never on the MUST path.
  Figures from 02 §7 and 06 §4 are budgets; the rest are order-of-magnitude estimates to be
  calibrated on a 20k-row S1 chunk before a full run.
- **Complexity** is engineering effort for this team in three days: low / medium / high.
- **Suitability**: MUST = in the V1 path; SHOULD = in V2 or a planned ablation;
  EXPERIMENTAL = one versioned experiment once V1 is stable; OPTIONAL = only if a measured
  failure calls for it; REJECTED = not to be tried, reason given.
- Data facts used throughout (01 §3): 48.4% of true pairs have an equal core name (4.6% raw,
  25.6% after accent/case/punctuation); 14.7% share no name token, of which 47% are
  Indic-script pool names, 35% domain/handle forms and 18% renames or heavy typos; 7.3% of
  pool names are in an Indic script and 9.2% of pool addresses carry a non-Latin token (S1
  is all Latin); 48% of S1 records share their core name with another S1 record; 62% of
  singletons have a name twin in the pool; 21% of unmatched pool records carry an S1 core
  name; 100% of true pairs share the country; every pool record belongs to at most one S1
  entity (the 1-to-1 pool property); 78% (US) / 83% (India) of true pairs share a numeric
  address token; 4.4% of pool addresses are empty; 5.6% of entities are singletons; a
  matched entity has 3.7 matches on average (max 11); the test pool is ≈ 40% unmatched
  against 26% in train.

## 1. A. Normalisation (plan group B, `normalize.py`, 05)

| Technique | Why relevant here | Strengths | Weaknesses | Cost | Complexity | Suitability | Refs |
|---|---|---|---|---|---|---|---|
| Unicode NFKD, combining-mark strip, casefold | Injected accents (`Léarning`, `Àmicale`) and case noise; with the punctuation rules, raw equality 4.6% → 25.6% | Deterministic, vectorised `str` ops, script-agnostic | Nothing for script changes; NFKD splits ligatures and full-width forms, harmless here | 11.7M rows ≈ 1 min per country, < 1 GB | low | MUST (R1) | [2,23] |
| Punctuation and whitespace rules, `&` → `and`, prefix-junk and domain-form stripping | `&` vs `and`, `-- Holloway`, `<< Team`, `.com` names: domain/handle forms are 35% of the zero-overlap pairs | Cheap; feeds every downstream key | Over-stripping merges distinct names (`b+` kept as a token for that reason) | seconds | low | MUST (R2–R4) | [2] |
| Legal-form and address-abbreviation maps (`LEGAL_FORMS`, `ADDRESS_ABBREV`) | Removing legal tokens lifts core-name equality 25.6% → 48.4%; Rd/Road, St/Street dominate address noise | Hand-typed domain knowledge, no external data; canonical `legal_form` doubles as a feature | Ambiguous tokens (`co` Colorado, `pl`, `st` saint/street); French forms come from knowledge only, no train data | explode-map-join ≈ 2 min / 11.7M rows | low | MUST (R5, R7) | [2,38d] |
| Derived name keys: first token, sorted tokens, squash with leet fold | Word-order swaps (51% of pairs equal on the sorted key), `pelletierrexford.com`, `5tar Flxe`, `#high1andministries` | Exact keys for P1b/P1c; zero cost at match time | Leet fold must skip pure numbers (`39/56`); squash collides short names | seconds | low | MUST (R6) | [2,38d] |
| Token canonicalisation learned from training pairs (PMI alignment of last-two address tokens and name tokens, 05 §6) | Regional-script states (9.2% of pool addresses), `bengaluru ↔ bangalore`, `praivet → private` cannot be hand-typed exhaustively | Uses only the provided pairs (fair-play safe); learns the generator's actual substitutions | Needs ≥ 50 co-occurrences, rare tokens stay unmapped; nothing for France (no train pairs); fit on the train fold only | one pass over 7.6M train pairs < 2 min | medium | SHOULD (V2, B4) | [2,15] |
| Transliteration with `anyascii` on non-Latin rows | 7.3% of pool names and 9.2% of addresses carry Indic script; without it P2 finds nothing on that slice, which is 47% of the zero-overlap pairs | ISC licence, any script, no model data, only ~1M rows touched | Rough phonetics (`praivet limited`), so 3-gram overlap is partial; Python loop | ≈ 1M rows ≈ 30 s | low | MUST (R0) | [2,38e] |
| Number and postcode extraction (`addr_nums`, `postcode`) | 78–83% of true pairs share a numeric token; the house number separates same-street decoys | Exact, cheap, language-free | Postcodes exist in ~10% (US) / ~0% (India) of addresses; leet digits in names must not be touched | regex < 1 min | low | MUST (R8) | [2,23] |
| Address component heuristics (`region` map, `addr_last` city hint, ordinals) | State/city agreement is the decoy discriminator; the generator drops and reorders components | No external data; the same rule for every country | Positional hints fail on reordered or landmark-only addresses (`Near SBI ATM`); the static region map covers US/India/France names only | regex < 1 min | low | MUST (R8) | [2,38a] |
| CRF / statistical address parsers (usaddress, libpostal) | Clean components for US-format addresses | Accurate where trained | Models trained on external labelled addresses / OpenStreetMap (fair-play exposure, 01 §9); US-only or 1.8 GB model; Python CRF ≈ 30 min for 11.7M rows | 30 min + 2–4 GB RAM | medium | REJECTED | [38a] |
| One country-agnostic map set, country as a partition key only | Test adds France; 100% of true pairs share the country string | No country enumeration; French forms (SARL/SAS, rue/av/bd) ride in the same maps | Cannot be validated on France before the test output: check the France slice before every upload | none | low | MUST | [2] |
| Phonetic keys (Soundex / Double Metaphone via jellyfish) | Same-sounding typo variants | Cheap extra exact key | English-centric, poor on Indian names and transliterations [38e]; inflates the > 50 groups on common names | seconds | low | OPTIONAL (A4 key) | [2,3,38e] |

## 2. B. Blocking and candidate generation (plan group A, `blocking.py`, 06)

| Technique | Why relevant here | Strengths | Weaknesses | Cost | Complexity | Suitability | Refs |
|---|---|---|---|---|---|---|---|
| Partition by the country string (whatever values exist) | 100% of true pairs share the country: ≈ 3× smaller search space at zero recall cost | Free, open-set safe, France gets its own partition | None, provided no code enumerates countries | 0 | low | MUST | [2,3] |
| Exact / standard blocking on normalised keys (`name_core`, `name_sorted`, `name_squash`) | 48.4% of true pairs share `name_core`; the other keys add word swaps and domain/leet forms | Hash join, exact, deterministic, < 1 min | Recall 0.47 alone; 48% of S1 share a core name, so common keys explode (max group 1,461) and need the group cap of 50 | < 1 min, 0.5 GB | low | MUST (P1a–c) | [2,3,6] |
| Sorted neighbourhood (sort on a key, slide a window) | Cheap near-equal retrieval on one key | O(n log n), catches trailing-token differences | Blind to typos at the front; with 48% ambiguous names the window fills with same-name decoys; no address channel | sort 11.7M keys ≈ 1 min | low | OPTIONAL | [2,6] |
| Canopy clustering (inverted-index similarity with loose and tight thresholds) | The historical form of TF-IDF top-k blocking | Same idea as P2 | Subsumed by sparse top-k with `min_sim`; no top-k control | as P2 | low | REJECTED (subsumed) | [7,3] |
| Prefix filtering and all-pairs similarity joins (AllPairs, PPJoin) | Exact similarity join above a threshold with pruning | Exact; strong pruning at high thresholds | Pruning is weak at our thresholds (cos 0.20–0.30); no MIT top-k library at 10M scale; returns a threshold set, not top-k | slower than the sparse product at min_sim 0.3 | high | OPTIONAL (only if P2 exceeds 30 min) | [8,9] |
| Character n-gram TF-IDF top-k by sparse matrix products (`sparse_dot_topn`) | Typos, abbreviations, transliterations: char 3-grams of `name_core`, k=30, reach ≈ 0.85 recall | Exact top-k, deterministic, threaded C++, float32; `max_df` bounds the posting cost | Cost is Σ over grams of df_S1 × df_pool: India 10–30 min, 2.1 GB peak; blind to the 14.7% zero-overlap pairs without an address channel | India 10–30 min / 2.1 GB; val 2–3 min | low (library) | MUST (P2) | [39d,3,7] |
| Word-level TF-IDF top-k on `name_core + addr_norm` (weighted token blocking) | ≈ 90% of zero-name-overlap pairs share address tokens; the 48% ambiguous names are separated by address words | 52M nnz, 1–2 min for India; the only channel for renames and script names | Empty addresses (4.4%) degrade to name words; a bare-city address (`Delhi`) stays unreachable | 1–2 min / 1 GB | low | MUST (P3) | [33b,10,39d] |
| BM25 weighting for P3 instead of sublinear TF-IDF | Better length normalisation between 3-token and 15-token addresses | Same sparse-product machinery (pre-weight the matrix) | Gains on strings this short are usually small | as P3 | low | OPTIONAL (A3 ablation) | [10] |
| MinHash / LSH on n-gram shingles | Sub-linear Jaccard retrieval | No full product; `datasketch` is MIT | Probabilistic recall; Python-level hashing of 11.7M sets ≈ 10–20 min; signatures at 64–128 permutations cost 2–5 GB per country before the LSH tables; Jaccard ignores term weights, so common grams dominate | 10–20 min + 2–5 GB | medium | REJECTED (exact sparse top-k is cheaper) | [11a,11b,5] |
| Dense retrieval over sentence embeddings (HNSW / FAISS / chunked GEMM) | Semantic matches the lexical channels miss: transliterations, renames, reordered addresses (the 14.7% slice) | Language-agnostic; `multilingual-e5-small` is MIT, ~118M params | Encoding 11.7M rows ≈ 1 h fp16 on the GPU, 12–20 h on CPU; a float32 flat index of India is 7 GB (over budget), so fp16 chunked GEMM top-k (3.6 GB) or IVF-PQ with recall loss; embeddings blur digits and suites | 1–2 h GPU encode + 10–15 min GPU retrieval (≈ 3 h CPU) | high | EXPERIMENTAL (P5) | [12a,12b,31,38b] |
| Meta-blocking (weight and prune the block graph) | Formalises "keep candidates by evidence, drop hubs" | Large reductions on token blocking | Our passes already return top-k; the union cap ordered by max sim plus `ctx_pool_indegree` is the cheap equivalent | seconds | medium | OPTIONAL | [33,4] |
| Deep blocking (DeepBlocker: self-supervised tuple embeddings) | Reported recall gains on dirty textual data | Needs no labels | Needs word vectors and hours of training; gains over TF-IDF on structured records are small; a second encoder to ship and audit | hours GPU | high | REJECTED | [20,21] |
| Multi-pass union with a pass bitmask and a per-S1 cap | No single key reaches 0.97 (06 §1); the union of name channels and an address channel does | Recall and cost add linearly; per-pass recall is auditable in `blocking_report` | Duplicate pairs need a group-by; the cap ordering (exact first, then max sim) decides who survives | group-by on 30M int32 pairs ≈ 1 min | low | MUST | [4,5,6] |
| Reciprocal / two-sided top-k (pool → S1 neighbours too) | A pool record in many S1 lists is a decoy hub; one in none is a missed rare typo | Adds neighbours for hubs and orphans | Doubles the product; `ctx_pool_indegree` already shows hubs to the model | ×2 of P2/P3 | low | OPTIONAL (A5) | [4] |

## 3. C. Pairwise features (plan group C, `features.py`, 07)

| Technique | Why relevant here | Strengths | Weaknesses | Cost | Complexity | Suitability | Refs |
|---|---|---|---|---|---|---|---|
| Levenshtein / Indel normalised similarity (`fuzz.ratio`, `Indel`) | Typos, leet, dropped characters in names | Standard; rapidfuzz C++ | Position-sensitive: word swaps and prefix junk score low; length-biased | `cpdist` ≈ 1M pairs/s per scorer: 24M pairs ≈ 25 s each | low | MUST | [13,39c,23] |
| Jaro-Winkler | Best single comparator on short names in [13,14]; prefix weighting suits transliterations (`praivet` / `private`) | Tolerant to transpositions; cheap | Rewards shared prefixes of unrelated names (`summit a` / `summit b`) | as above | low | MUST | [14,13,38e] |
| Token Jaccard / Dice on `name_core` and `addr_norm` | Order-free overlap; `ad_jaccard` is 0 on only 4–5% of true pairs | Sparse row-wise CSR product, no Python loops | Blind to in-token typos; empty sets need the NaN policy | 24M pairs ≈ 1 min | low | MUST | [2,23] |
| Cosine TF-IDF (the P2/P3/P4 sims copied from the pairs frame) | Already computed; corpus-weighted so rare tokens count more | Free | NaN when a pass did not produce the pair; tied to the blocking config | 0 | low | MUST | [13,7] |
| Character n-gram / substring similarity (`partial_ratio`, squash ratio) | Domain and handle forms, prefix junk (35% of zero-overlap pairs) | Substring-tolerant | `partial_ratio` is high for a short name inside a long one (decoy risk) | as rapidfuzz | low | MUST | [39c,13] |
| Token-set / token-sort ratios | Word-order transpositions and inserted generic words (`N+ Inc Services`) | Order-free and typo-tolerant at once | Token-set ≈ 1 whenever one name is a subset of the other: the ambiguous-name decoy case | as rapidfuzz | low | MUST | [39c] |
| SoftTF-IDF (token TF-IDF with Jaro-Winkler token matching) | Top hybrid in the classic comparison [13] | Weighted and typo-tolerant together | No vectorised implementation; the GBDT gets the same signal from TF-IDF and JW separately | Python-level ≈ 10 min | medium | OPTIONAL | [13] |
| Numeric-token matching (`num_jaccard`, `num_first_eq`, `postcode_eq`) | House numbers and PINs are the strongest decoy separators (78–83% share a number) | Exact, language-free | Absent on ≈ 20% of pairs and on landmark addresses: NaN plus flags | seconds | low | MUST | [2,38a] |
| Address string similarity with containment (`ad_token_set`, `ad_partial`, `ad_contain`) | Dropped or reordered components: containment catches an address that is a prefix of the other | Asymmetric containment encodes "S1 fuller than pool" | Empty pool addresses (4.4%) give NaN; abbreviations must be expanded first | as rapidfuzz | low | MUST | [2,38a] |
| Postcode / region / city equality (`postcode_eq`, `region_eq`, `last_eq`) | Same-name businesses in different states are the typical decoy: 21% of unmatched pool records carry an S1 core name | Binary, high precision when both sides present | Coverage limited (postcodes rare in India); fires across scripts only with the region map | seconds | low | MUST | [2] |
| Exact-agreement indicators (`first_eq`, `sorted_eq`, `legal_eq`, `prefix4_eq`) | Trees split cleanly on binary agreement; `legal_eq` separates `Pvt Ltd` twins | Cheap, readable in error analysis | Redundant with sims for a linear model; fine for trees | seconds | low | MUST | [1,2] |
| Missingness flags (`addr_empty_r`, `legal_missing_*`) | An empty address (4.4%) means name-only evidence: lower confidence, do not guess | Makes NaN informative | Nothing beyond that | 0 | low | MUST | [2] |
| Length features (`tok_len_*`, `len_ratio_name`) | Short names are more ambiguous; the length ratio flags truncation | Cheap | Weak alone | 0 | low | MUST | [2] |
| Cross-field interactions (`name_strong_addr_weak`, `addr_strong_name_weak`) | The decoy and the rename signatures as one bit each | Helps LR; readable | Trees learn them from base features; add only if importance justifies (C5) | 0 | low | OPTIONAL (C5) | [16] |
| Group-context features (rank, gap, `ctx_n_cands`, `ctx_pool_indegree`) | With 48% ambiguous S1 names the pairwise view is blind to competition; rank and gap show the model what the decision layer sees | Cheapest precision gain; exposes decoy hubs | Must be computed inside whole S1 groups (chunking rule); distribution shifts with the candidate count (test pool 40% unmatched vs 26%) | group-by per chunk ≈ 1 min | medium | MUST (ablate in C5) | [33,16] |
| Embedding cosine (e5-small) as feature 49 | Semantic agreement independent of script and spelling | One number carries the P5 evidence | Needs the P5 embeddings (1–2 h GPU); weak on digits | after P5, seconds | low | EXPERIMENTAL | [31,38b] |

## 4. D. Matching models (plan group D, `model.py`, 08)

| Technique | Why relevant here | Strengths | Weaknesses | Cost | Complexity | Suitability | Refs |
|---|---|---|---|---|---|---|---|
| Logistic regression (supervised Fellegi-Sunter: log-odds weights over the comparison vector) | The theory of the task; a calibrated linear control for D1 and a sanity check on the features | Fits in seconds; calibrated by construction; readable weights | Linear: cannot express "name strong AND address weak → decoy" without interactions; needs `-1` imputation plus flags | 6M × 48 < 1 min | low | SHOULD (D1 control) | [1,14] |
| Unsupervised Fellegi-Sunter with EM (Splink) | The standard when labels are missing | No labels; scales on DuckDB / Spark; MIT | We have 7.6M labelled pairs; EM on 48 continuous features needs binning; heavy dependency | 10+ min | medium | REJECTED (labels exist) | [1,32] |
| Random forest | Non-linear, robust to scaling; D2 in the plan | Strong on tabular ER in the Magellan benchmarks; few knobs | sklearn RF on 6M rows: 20–60 min to fit, trees in GBs, ≈ 10 min to predict 24M rows; probabilities need calibration | 20–60 min fit | low | SHOULD (D2, one run) | [16,26a] |
| LightGBM (histogram GBDT) | Handles NaN natively; 6M × 48 in minutes; 24M predictions in 2–5 min; MIT | Best accuracy per CPU-minute on tabular data; early stopping on the tune split; gain importances drive the C-group ablations | Needs the tune split against overfit; probabilities drift under `scale_pos_weight` and hard-negative reweighting (recalibrate) | fit 5–15 min, predict 2–5 min, < 2 GB | low | MUST | [24,26a] |
| XGBoost (hist) | Same family, different regularisation; Apache-2.0 | Comparable accuracy | 1.5–2× LightGBM time; no expected gain | 10–25 min | low | OPTIONAL (D4; skip if LightGBM dominates) | [25a] |
| CatBoost | Ordered boosting, native categoricals; Apache-2.0 | Robust defaults | No categorical features in our set; 2–3× slower on CPU | 15–40 min | low | OPTIONAL (D-group tail) | [25b] |
| Learnable string similarity (affine-gap edit distance with learned costs, SVM over pair vectors) | Learns which edits are cheap (abbreviation, leet) | Adapts the comparator to the data | No maintained library; Python-speed training; superseded by a GBDT over many fixed scorers | hours | high | REJECTED (its pair-selection idea lives on in E) | [15] |
| Magellan-style matching (blocking + comparison features + RF/GBDT) | The pattern of our pipeline; benchmarks show it at parity with deep matchers on structured records | Debuggable, fast, all MIT/BSD | Ceiling set by hand-designed features | = C + D costs | medium | MUST (as design; the library is not used) | [16,22] |
| DeepMatcher (RNN / attention over attribute embeddings) | Gains reported on dirty and textual data | Learns representations | Word vectors plus hours of GPU training; ≈ 1–2k pairs/s inference → 3–7 h for 24M pairs; marginal on structured records | hours | high | REJECTED | [17,21] |
| Ditto / transformer cross-encoder (MiniLM fine-tuned on serialised pairs) | State of the art on hard EM benchmarks; reads both records jointly, strong on transliteration and renames | 22M params, Apache-2.0; fine-tune on 1M pairs ≈ 30–60 min GPU | Full pool: 24M pairs ≈ 1–2 h fp16 on the 4 GB GPU, 7–20 h on CPU; opaque; no NaN or context features | full: hours; uncertain band (≈ 1.5M pairs): ≈ 5 min GPU, ≈ 1 h CPU | high | OPTIONAL (re-rank the uncertain band only) | [18,19,21] |
| Sentence-BERT bi-encoder (`multilingual-e5-small`) | One embedding per record enables P5 and feature 49 | Linear in records, not pairs; multilingual; MIT | Cannot compare records jointly; digits and suites blur | encode 11.7M ≈ 1 h GPU / 12–20 h CPU | medium | EXPERIMENTAL | [31,29] |
| LLM matching (zero- or few-shot prompting, ≤ 8B Apache-2.0 models) | Best zero-shot generalisation in [35] | No feature engineering | Hosted APIs are external services (disqualification); a local ≤ 8B model at ≈ 30–50 pairs/s on the GPU needs ≈ 6–9 days for 24M pairs; open models trail fine-tuned PLMs in [35] | days | high | REJECTED | [35] |
| Hybrid: embedding cosine among the lexical features of one GBDT | Semantic evidence without a pair-level encoder | Keeps LightGBM's speed and calibration | Depends on P5; one more model and licence to ship | + P5 | low | EXPERIMENTAL | [31,24] |
| Name-only deep disambiguation (RNN over company names) | Company-name literature | Learns abbreviation patterns | Names alone cannot separate the 48% ambiguous S1 names: the address decides | hours | high | REJECTED | [38c,38d] |

## 5. E. Training pairs and hard-negative mining (`trainset.py`, 09)

| Technique | Why relevant here | Strengths | Weaknesses | Cost | Complexity | Suitability | Refs |
|---|---|---|---|---|---|---|---|
| Retrieval negatives: label every blocking candidate of a sampled S1 entity (the BM25-negative recipe) | The model must rank exactly the decoys the blocker produces; sampling entities, not pairs, keeps group context and the ≈ 12% positive rate | Free; inference-like distribution | Only as hard as the blocker; true pairs outside the candidate set are lost to the model (the metric still counts them) | 200k fit S1 → 5–6M pairs | low | MUST | [29,16] |
| Round-2 false-positive mining (`mine_false_positives`: rescore the fit pairs, upweight or add confident errors, refit) | Same-name decoys and singleton name twins (62%) are the errors F0.5 punishes most | Targets precision directly; one extra fit | Reweighting shifts probabilities: recalibrate before tuning thresholds; can chase label noise | rescore 6M ≈ 2 min + refit 5–15 min | low | SHOULD (V2) | [30,29,15] |
| Hard-negative contrastive fine-tuning of the bi-encoder | Would teach the P5 embeddings to separate `meridian` decoys | Improves retrieval where lexical channels fail | Needs P5 first; 1–2 h GPU per round; overfits the generator | 1–2 h GPU | high | EXPERIMENTAL (after P5) | [30,29,31] |
| Active learning / uncertainty sampling for labelling | Classic when labels are scarce | Fewest labels per unit of accuracy | 7.6M labelled pairs and no labelling budget; only the habit of reading the uncertain band survives, in error analysis | human time | medium | REJECTED as a loop; OPTIONAL as an error-analysis sampler | [34,39a] |
| Static hard-pair selection (near-miss pairs chosen by a cheap similarity) | Learnable-similarity work trained on ambiguous pairs chosen by a fixed measure | Round-2 effect without a first model | Selects by a proxy, not by the model's errors | seconds | low | OPTIONAL | [15] |
| Harder evaluation fold (`harder_fold`: drop 20% of matched pool records) | Test pool ≈ 40% unmatched vs 26% in train: more decoys per entity, more singleton-like entities | Rehearses the shift; already in `evaluate.py` | Synthetic; the real test shift is unknown | seconds | low | SHOULD (report every version) | [22] |

## 6. F. Entity-level decision (plan group E, `decision.py`, 10)

| Technique | Why relevant here | Strengths | Weaknesses | Cost | Complexity | Suitability | Refs |
|---|---|---|---|---|---|---|---|
| Global probability threshold (`tau_abs`) | E1 baseline; the floor no accepted candidate may go under | One knob, tuned on the tune split | Ignores competition: a 0.6 candidate is a match when alone and a decoy next to a 0.99 sibling | seconds per grid point | low | MUST (E1) | [1,27] |
| F-measure-optimal thresholding (grid on the tune split, seeded by the closed form) | Macro F0.5 is the only objective; [27] proves the F1-optimal threshold is F*/2 for calibrated p, and the same marginal-gain argument gives F*/(1+β²) ≈ 0.8·F* for β = 0.5, i.e. well above 0.5 | Optimises the real metric; the closed form seeds the grid | Per-entity macro F0.5 with the singleton rule is not the aggregate F of [27]: the grid decides, the formula only seeds | 500–2,000 rules × 3M tune pairs ≈ minutes | low | MUST (E2) | [27,28,36] |
| Probability calibration (Platt or isotonic on the tune split) | Expected-F0.5 decoding and the closed-form seed need calibrated p; boosting under reweighting drifts | Trivial; isotonic fixes any monotone distortion | Isotonic overfits small tune sets; calibrate on tune, never on val | seconds | low | SHOULD (V2, before expected-F decoding) | [26a,26b,26c] |
| Score gap / relative threshold (`tau_rel`: keep p ≥ p_max − gap) | Sets, not top-1: 3.7 matches per entity on average, so siblings near the best are kept and stragglers dropped | Adapts to each entity's evidence | Fails when the best candidate is itself a decoy; interacts with `tau_abs` | seconds | low | MUST (E3) | [27] |
| Set decoding with a cap (`max_matches = 11`) | The output is a set of up to 11 (train max) | Stops runaway lists on hub names | A cap is no precision device on its own | seconds | low | MUST (E5) | [2] |
| Expected-F0.5 decoding (per entity: sort by p, score every prefix set and the empty set by plug-in expected F0.5 = 1.25·E[TP] / (0.25·E[n_true] + k), keep the best) | Decision-theoretic F optimisation [28] beats a fixed threshold when probabilities are good; the empty set scores Π(1 − p_i), so singleton detection falls out of the same rule | Per-entity and principled; cumulative sums make it cheap; exact Poisson-binomial DP is feasible for ≤ 60 candidates | Needs calibration; E[n_true] must add a blocking-miss prior (≈ 0.03 × 3.7); ratio of expectations is an approximation | cumsum per group ≈ seconds | medium | SHOULD (V2, E2) | [28,27,26a] |
| 1-to-1 assignment (each pool record to its highest-probability S1) | Measured: a pool record belongs to at most one S1; 21% of unmatched pool records carry an S1 core name, so cross-entity competition is real | One-sided constraint, so greedy argmax per pool record is exact (no Hungarian); removes the same decoy from several entities at once | Steals a true match when a decoy S1 scores higher; a record dropped by its winner is not re-assigned (precision first); tune the thresholds with it on | group-by idxmax on 24M rows ≈ 10 s | low | MUST | [2,37] |
| Singleton detection (`tau_single` on p_max) | 5.6% of entities with a binary payoff; 62% have a name twin, so "no same-name record" is not a singleton test | Uses the model's confidence, not name absence | One more knob; the test's 40% unmatched pool raises the singleton-like share | seconds | low | MUST (E4) | [1,27] |
| Per-source thresholds (S2 vs S3) | S2 and S3 are generated separately; their noise and decoy rates may differ (`is_s3`) | Two more grid dimensions | Doubles the grid; overfit risk on the tune split | minutes | low | SHOULD (V2) | [27] |
| Cluster-consistency post-step (a pool twin of an accepted match inherits it) | S2 and S3 are not deduplicated against each other | Recovers a missed sibling when its twin is accepted | Propagates errors; needs pool–pool similarities we do not compute | extra pool–pool pass | medium | OPTIONAL | [37,2] |

## 7. G. Evaluation methodology

| Technique | Why relevant here | Strengths | Weaknesses | Cost | Complexity | Suitability | Refs |
|---|---|---|---|---|---|---|---|
| Macro F0.5 per S1 entity with the singleton rule on the fixed val fold (`metrics.breakdown`) | The scored metric: van Rijsbergen's E-measure with β = 0.5, precision weighted 4× | Exactly the organisers' rule, unit-tested on their 0.714 example | One 20% fold: expect ± 0.002 noise; never tune on it | seconds | low | MUST | [36,22] |
| Candidate-set report (pair recall, reduction ratio, ceiling F0.5, candidates per S1) | Blocking sets the recall ceiling and the organisers audit `candidate_pairs.tsv` | Ceiling F0.5 converts recall into metric units per entity size | Recall lost on 1-match entities costs more than on 6-match ones: report per slice | seconds | low | MUST | [22,3,4] |
| Slice and error reports (non-Latin, ambiguous names, singletons, S2/S3, country) | Every V1 → V2 change targets a slice: 7.3% Indic names, 48% ambiguous, 5.6% singletons, France | Turns a score into a work list | Small slices: report counts with rates | seconds | low | MUST | [22] |
| Efficiency reporting (seconds and peak RSS per stage) | 12 cores, 15 GB: the test run must fit in hours | Catches budget overruns before upload day | Machine-dependent | 0 | low | MUST | [22,5] |
| Cluster-level metrics (pairwise vs cluster F, generalised merge distance) | Would matter if the pool were clustered | Standard for deduplication evaluation | Our unit is the S1 entity, not a cluster | seconds | low | OPTIONAL | [37] |
| Leaderboard as confirmation only (15 uploads) | Public = a subset of test; private decides | Forces local discipline | Public-score noise on a subset | one upload | low | MUST (15) | [22] |

## 8. Chosen stack and why the alternatives were rejected

The stack follows 02 §1 stage for stage; each step is one logged experiment (project rules §2).

**Baseline (v001): exact core name + address heuristic.** P1a blocking (`name_core` equal
inside the country, groups > 50 skipped) and the `heuristic` matcher backend: accept a
candidate when the addresses share a numeric token or at least half of their tokens, else
an empty list. Expected pair recall ≈ 0.47 (48.4% core-name equality) at high precision: a
macro F0.5 far below V1, but an end-to-end run that writes both TSVs and passes
`make validate` on day one, and the reference row for every later line of `experiments.csv`.

**V1: multi-pass TF-IDF blocking + 48 features + LightGBM + tuned decision rule with 1-to-1.**
Normalisation R0–R9 (05); blocking P1a–c ∪ P2 (char 3-gram TF-IDF, top-30, cos ≥ 0.30) ∪ P3
(name + address words, top-20, cos ≥ 0.20) inside the country partition, cap 60, target pair
recall ≥ 0.97 at ≤ 40 candidates per entity (06); the 48-feature registry (07); LightGBM with
early stopping on the tune split, LR as the D1 control (08); greedy 1-to-1 by argmax
probability, then `tau_abs`, `tau_rel`, `tau_single` and `max_matches` grid-searched for
macro F0.5 on the tune split (10). Why this and nothing cleverer: the measured structure of
the data is lexical and address-driven (48.4% core-name equality, ≈ 90% of the zero-overlap
pairs share address tokens, 78–83% share a house number), so weighted sparse retrieval reaches
the recall target and a GBDT over agreement features captures the decoy pattern "name strong,
address weak" that separates the 48% ambiguous names and the 62% of singletons with a name
twin; the benchmarks [16,22] put feature-engineered learners at parity with deep matchers on
structured records; the whole run is CPU-only, ≈ 1 h for the full test set inside 6 GB per
stage, deterministic, auditable pass by pass (the organisers inspect `candidate_pairs.tsv`),
and ships only MIT/BSD/Apache/ISC code.

**V2: learned token maps, round-2 hard negatives, expected-F0.5 decoding, per-source
thresholds.** B4 token alignment from the train fold's pairs fixes regional-script states
and `bengaluru ↔ bangalore` without external data; round-2 mining refits on the model's own
confident false positives (same-name decoys, singleton name twins); calibration on the tune
split followed by per-entity expected-F0.5 prefix decoding replaces the `tau_abs`/`tau_rel`
pair wherever it wins on tune; S2- and S3-specific thresholds if the `is_s3` slices differ.
Each is one experiment, kept only if macro F0.5 on val rises; all are evaluated on the harder
fold as well, because the test pool is ≈ 40% unmatched against 26% in train.

**Experimental: `multilingual-e5-small` embedding blocking pass (P5).** The 14.7%
zero-overlap pairs are half Indic-script names; anyascii plus P3 recover part of them, and
e5-small (MIT, ~118M params, multilingual) is the cheapest encoder that could recover more.
Encode `name_addr` once per record with the `query:` / `passage:` prefixes (≈ 1 h fp16 on the
4 GB GPU), retrieve top-10 per S1 by chunked fp16 GEMM inside the country, union it as bit
64, and judge it by recall on the non-Latin and zero-overlap slices against the candidates it
adds and the ceiling F0.5; then try the cosine as feature 49. Kept only if ceiling F0.5 rises
and the encode fits the timeline; if shipped, its licence and parameter count go into the
documentation.

**Optional: MiniLM cross-encoder on the uncertain band.** If error tagging shows F0.5 lost on
pairs with 0.2 ≤ p ≤ 0.8 (expected 5–10% of pairs), fine-tune a MiniLM cross-encoder on
serialised `name | address` pairs from the fit set and re-score only that band (≈ 5 min on
the GPU); its score enters a second-stage LightGBM or replaces p inside the band, tuned like
any other rule. Never on the full pool.

**Rejected**, with the replacement that covers the same need:

| Alternative | Why rejected | Replacement |
|---|---|---|
| Full-pool cross-encoder (Ditto-style) | 24M pairs: 1–2 h GPU, 7–20 h CPU; opaque; no NaN or context features | LightGBM on 48 features; cross-encoder only on the uncertain band |
| Deep blocking (DeepBlocker) | hours of training for small gains over TF-IDF on structured records; a second encoder to audit | multi-pass TF-IDF; e5-small P5 as the single embedding experiment |
| libpostal, usaddress | models trained on external address data (fair-play, 01 §9); RAM and CPU cost; US-only | hand-typed maps plus the token map learned from train pairs |
| GPL / Artistic libraries (unidecode, text-unidecode, python-Levenshtein) | licence policy (17 §2) | anyascii (ISC), rapidfuzz (MIT) |
| LLM matching | hosted APIs are external services; local ≤ 8B models need days for 24M pairs and trail fine-tuned PLMs | none needed |
| MinHash / LSH | probabilistic recall, Python hashing time, 2–5 GB signatures per country; no gain over exact sparse top-k | `sparse_dot_topn` |
| float32 FAISS / HNSW index over the pool | 7 GB for India plus the graph exceeds the 6 GB stage budget | fp16 chunked GEMM top-k, only if P5 runs |
| Unsupervised Fellegi-Sunter / EM (Splink [32]), Dedupe active learning [39a] | we have 7.6M labelled pairs; Python-level loops; extra dependencies | supervised LightGBM on retrieval negatives |
| recordlinkage toolkit [39b] | pins pandas < 3 | own vectorised feature registry |
| DeepMatcher, name-only RNNs | hours of training, marginal on structured records, names alone cannot separate 48% ambiguous S1 names | address-driven features in the GBDT |
| Per-country models or thresholds | France has no training data; the metric is macro over all entities | country as a partition key only; France slice checked before upload |
| Sorted neighbourhood, canopies, phonetic keys as extra passes | no address channel; subsumed by P2; English-centric | P1–P3 union; revisit only if a slice stays under target |
