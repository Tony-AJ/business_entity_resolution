# 05 — Normalisation strategy (module `normalize.py`, owner M2, plan group B)

What you build: `normalise_records(df, cfg) -> DataFrame` with the columns of
`02_SYSTEM_ARCHITECTURE.md` §4.1, plus the learned region map. Input: a source frame
(`entity_id`, `business_name`, `business_address`, `country`, all `str`). Output: the
normalised frame, same row order, every string column non-null (`""` for missing).
Everything vectorised (`Series.str.*` on Arrow strings); the only Python-level map is the
sorted-token join and `anyascii` on the ~7–9% non-Latin rows.

## 1. Why these rules (measured on true pairs, see 01 §3)

Raw-name equality is 4.6%; accent/case/punctuation normalisation lifts it to 25.6%;
removing legal-form tokens to 48%. 14.7% of pairs share no name token: half are Indic-script
names (transliteration), a third are domain/handle forms (squash key), the rest are renames
that only the address can link. Addresses differ mostly by case, abbreviation, component
order, dropped components and a regional-script state token.

## 2. Ordered rules

Apply in this order; each step feeds the next.

```
R0  transliterate      rows with any char outside [\x00-\x7FÀ-ɏḀ-ỿ] → anyascii(text); set non_latin
R1  unicode            NFKD, drop combining marks (Léarning → Learning, Àmicale → Amicale), casefold
R2  ampersand          "&" → " and "; "+" kept as a token ("b+ retail" ↔ "b+ retail")
R3  domain forms       strip leading "@#-<>[]*." runs; strip "www."; strip trailing ".com|.net|.org|.in|.co|.io|.fr|.biz|.info"
R4  punctuation        [^a-z0-9+] → " "; collapse whitespace; strip                                → name_norm / addr_norm base
R5  legal forms        map each token through LEGAL_FORMS; tokens that map are removed from name_core and
                       appended (canonical, in order) to legal_form
R6  derived name keys  name_first, name_sorted, name_squash (see §4)
R7  address abbrev     token map ADDRESS_ABBREV (only when expand_abbrev) after R4
R8  address parts      addr_nums, postcode, region (REGION_ABBREV or learned map), addr_last, addr_tokens
R9  name_addr          name_core + " " + addr_norm
```

Never drop a record: an empty name or address yields `""` in every derived column and
`addr_tokens = 0`.

## 3. Token maps (static, hand-typed domain knowledge, no external data)

`LEGAL_FORMS` (token → canonical): `inc, incorporated → inc`; `corp, corporation → corp`;
`co, company → co`; `llc, l l c → llc`; `llp → llp`; `ltd, limited → ltd`; `pvt, private → pvt`;
`plc → plc`; `lp → lp`; `pc → pc`; `sa → sa`; `sas, sasu → sas`; `sarl, eurl → sarl`; `sci → sci`;
`gmbh → gmbh`; `pte → pte`; `opc → opc`; `m s, ms → ""` (the Indian "M/s" prefix, dropped).
Multi-token forms are matched after R4 on the joined string (`"l l c"`, `"m s"`).
Note: `co` is ambiguous (Colorado is `co` in addresses) — the map applies to names only.

`ADDRESS_ABBREV` (token → canonical, addresses only): `st → street`, `rd → road`,
`ave, av → avenue`, `blvd, bd → boulevard`, `dr → drive`, `ln → lane`, `ct → court`,
`cir → circle`, `hwy → highway`, `pl → place`, `ste → suite`, `apt → apartment`,
`fl → floor`, `bldg → building`, `n, s, e, w → north, south, east, west`, `ne, nw, se, sw`,
`nr → near`, `opp → opposite`, `mkt → market`, `rly → railway`, `r → rue`, `imp → impasse`,
`che → chemin`, `all → allee`, `pl → place` (French collides with US `pl`; both map to `place`,
harmless), `no, num → number`, `bis` kept. Ordinals: `1st, 2nd, 3rd, 4th … → 1, 2, 3, 4`.

`REGION_ABBREV` (static seed): the 50 US states + DC (`tx ↔ texas`), Indian states and union
territories (`ka ↔ karnataka`, `mh ↔ maharashtra`, `tn ↔ tamil nadu`, `up ↔ uttar pradesh`,
`od, or ↔ odisha, orissa`, `telangana`, `andhra pradesh`…), French regions (`ile de france`,
`hauts de france`, `nouvelle aquitaine`, `pays de la loire`…) and departments seen in the data
(`nord`, `loire atlantique`). The canonical form is the full lowercase name without accents.

## 4. Derived name keys

- `name_first`: first token of `name_core` (`""` if none).
- `name_sorted`: `" ".join(sorted(tokens))` — catches word-order swaps (51% of pairs equal).
- `name_squash`: `re.sub("[^a-z0-9]", "", name_core)` then leet fold
  `0→o, 1→l, 3→e, 4→a, 5→s, 7→t, 8→b` **only when the token also contains letters** (never
  fold pure numbers such as `39/56`). `pelletierrexford.com` → `pelletierrexford`,
  `#high1andministries` → `highlandministries`, `5tar flxe llc` → `starflxe`.

## 5. Transliteration

- `anyascii` (ISC licence) maps any script to ASCII; applied only to rows flagged
  `non_latin` (7% of pool names, 9% of pool addresses, 0% of S1). Devanagari
  `प्राइवेट लिमिटेड` → `praivet limited`-like output; char 3-grams then overlap with
  `private limited` partially, and `name_addr` word retrieval plus the address carry the rest.
- Do not use `unidecode` (GPL) or `text-unidecode` (Artistic/GPL).
- B3 experiment: measure P2 recall on the `non_latin` slice with and without R0; expect a
  large gain on names, smaller on addresses (states are the usual non-Latin token, handled
  by the region map below).

## 6. Learned token alignment (B4, `fit_region_map` and `fit_token_map`)

No gazetteer is allowed, but the training pairs are. For a true pair the two addresses
usually differ in one component; aligning them yields a dictionary of equivalent tokens
(`tx ↔ texas`, `ಕರ್ನಾಟಕ ↔ karnataka`, `bengaluru ↔ bangalore`, `mh ↔ maharashtra`).

```
fit_token_map(pairs, s1n, pooln, min_count=50, min_pmi=3.0):
  L = last-two-token set of S1 addr_norm, R = same for pool addr_norm (before region mapping)
  for each true pair: for a in L(s1), for b in R(pool), if a != b: count[a,b] += 1
  count_a, count_b marginals over all records
  pmi(a,b) = log( count[a,b] * N / (count_a * count_b) )
  keep (a,b) with count >= min_count and pmi >= min_pmi and b is the argmax partner of a
  canonical = the longer, all-Latin, more frequent member; map both to it
```

Fit on the **train fold's pairs only** (`fold.pairs` of `load_fold("train")`), save to
`experiments/vNNN/artifacts/token_map.json`, and pass it as `NormaliseConfig.region_map`.
The same procedure on name tokens (`fit_token_map(..., column="name_core")`) learns
`praivet → private`, `limited` transliterations and common abbreviations; use it for the
`name_translit` variant in B3/B4 experiments if `anyascii` alone under-delivers.

## 7. Country-aware handling

Country never selects code paths. Everything is one function applied to all rows; the maps
simply contain tokens from every country. The only country use is the blocking partition.
Check on the France slice of test that `legal_form` is non-empty for a sensible share
(SARL/SAS/EURL/SCI are common) and that `region` is populated for the region names above.

## 8. Implementation notes

- Compile all regexes once at module import; use `Series.str.replace(pat, repl, regex=True)`.
- Token maps: `Series.str.split()` → `explode` → `map(dict).fillna(token)` → `groupby(level).agg(" ".join)`
  is the vectorised route for 12M rows; benchmark against a single regex alternation
  (`\b(st|rd|ave)\b`) which is usually faster for < 100 keys.
- `anyascii` is a Python loop: filter rows first (`str.contains(NON_LATIN_RE)`), expect ~1M
  rows in the full train pool, ≈ 30 s.
- Cache the normalised frames per split/country under `dataset/.cache/pipeline/<tag>/<country>/`.
- Deterministic: no randomness anywhere in this module.

## 9. Tests (`tests/test_normalize.py`)

| Test | Assertion |
|---|---|
| `test_basic_norm_accents_ampersand_punct` | `"Café & Co."` → `name_norm == "cafe and co"` |
| `test_legal_forms_removed_and_canonical` | `"Sharma Traders Pvt. Ltd."` → core `sharma traders`, legal `pvt ltd`; `"ACME Corporation"` and `"Acme Corp"` share `name_core` and `legal_form` |
| `test_legal_form_position_independent` | `"LLC Moncada Learning Center"` → core `moncada learning center`, legal `llc` |
| `test_squash_domain_and_leet` | `"pelletierrexford.com"`, `"Pelletier Rexford Corp"` → same `name_squash`; `"#high1andministries"` → `highlandministries` |
| `test_squash_keeps_pure_numbers` | `"39/56 Asia"` → `3956asia` (no leet fold on `3956`) |
| `test_sorted_key` | `"Bustamante Sanchez"` and `"Sanchez Bustamante"` share `name_sorted` |
| `test_address_abbrev_and_numbers` | `"1795 Westchester Dr, High Point, NC"` → `addr_norm` contains `drive`, `addr_nums == "1795"`, `region == "north carolina"`, `addr_last == "high point"` |
| `test_postcode_extraction` | `"MG Road, Pune 411001"` → `postcode == "411001"`; `"… TX 75701"` → `75701`; none → `""` |
| `test_transliterate_only_non_latin` | Latin rows unchanged byte-for-byte; `non_latin` True only for script rows; Devanagari row becomes ASCII |
| `test_region_map_override` | a learned map `{"ka": "karnataka"}` is applied when passed in the config |
| `test_empty_fields` | empty name/address → all derived columns `""`, `addr_tokens == 0`, no NaN anywhere |
| `test_schema` | `list(out.columns) == NORM_COLUMNS`, dtypes as specified, `len(out) == len(df)`, order preserved |
| `test_fit_token_map_learns_alignment` | synthetic pairs with `tx`/`texas` in 60 rows → map contains `tx → texas`; a 3-row coincidence is not learned |

## 10. Experiments (B group; log with `group="B1"…"B5"`)

| Version | Change | Measure |
|---|---|---|
| B1 | R0–R4 only, exact `name_norm` block | P1 recall (expect ≈ 0.26) and cands/S1 |
| B2 | + legal forms (R5) | P1 recall ≈ 0.48; group sizes (cap 50) |
| B3 | + transliteration (R0) and `name_squash` | recall on `non_latin` and domain slices |
| B4 | + learned token map (region and name) | P3 recall, address feature agreement on true pairs |
| B5 | + address abbreviations, ordinals, numbers | `addr_jaccard` distribution on true vs decoy pairs |

Success metric for this module: blocking pair recall and candidate count after each rule
(via `blocking_report`), plus the agreement rates in §1 recomputed on the val fold. Failure
cases to inspect: true pairs still with zero token overlap after B4 (`error_samples(kind=
"missed")` filtered to `pass == 0`), and any rule that increases exact-group sizes above 50.

## 11. Integration

`blocking.py` reads only the column names of §4.1; `features.py` reads `name_norm`,
`name_core`, `legal_form`, `addr_norm`, `addr_nums`, `postcode`, `region`, `addr_last`,
`addr_tokens`, `non_latin`. Add columns freely; never rename or drop one without a PR that
updates 02, 06 and 07.
