# 17 — Research references, libraries and licences

Audience: everyone. Documents 04, 08, 09 and 10 cite the numbers below; never renumber an
entry. Add a reference by appending a new number (or a letter suffix to an existing one),
never by inserting. Every entry was checked live on 2026-09-25 (DOI resolution, arXiv, PyPI,
Hugging Face); §4 lists what is still second-hand. Do not add a citation, DOI or version that
has not been resolved the same way.

## 1. References

[1] Fellegi, I. P. & Sunter, A. B. "A Theory for Record Linkage". JASA 64(328):1183–1210, 1969. doi:10.1080/01621459.1969.10501049
[2] Christen, P. "Data Matching: Concepts and Techniques for Record Linkage, Entity Resolution, and Duplicate Detection". Springer, 2012. doi:10.1007/978-3-642-31164-2
[3] Christen, P. "A Survey of Indexing Techniques for Scalable Record Linkage and Deduplication". IEEE TKDE 24(9):1537–1555, 2012. doi:10.1109/TKDE.2011.127
[4] Papadakis, G., Skoutas, D., Thanos, E., Palpanas, T. "Blocking and Filtering Techniques for Entity Resolution: A Survey". ACM Computing Surveys 53(2), art. 31, 2020. doi:10.1145/3377455, arXiv:1905.06167
[5] Papadakis, G., Svirsky, J., Gal, A., Palpanas, T. "Comparative Analysis of Approximate Blocking Techniques for Entity Resolution". PVLDB 9(9):684–695, 2016. doi:10.14778/2947618.2947624
[6] Hernández, M. A. & Stolfo, S. J. "The Merge/Purge Problem for Large Databases". SIGMOD 1995, pp. 127–138. doi:10.1145/223784.223807
[7] McCallum, A., Nigam, K., Ungar, L. H. "Efficient Clustering of High-Dimensional Data Sets with Application to Reference Matching". KDD 2000, pp. 169–178. doi:10.1145/347090.347123
[8] Bayardo, R. J., Ma, Y., Srikant, R. "Scaling Up All Pairs Similarity Search". WWW 2007, pp. 131–140. doi:10.1145/1242572.1242591
[9] Xiao, C., Wang, W., Lin, X., Yu, J. X. "Efficient Similarity Joins for Near Duplicate Detection". WWW 2008, pp. 131–140. doi:10.1145/1367497.1367516
[10] Robertson, S. & Zaragoza, H. "The Probabilistic Relevance Framework: BM25 and Beyond". Foundations and Trends in IR 3(4):333–389, 2009. doi:10.1561/1500000019
[11a] Broder, A. Z. "On the resemblance and containment of documents". Compression and Complexity of SEQUENCES 1997, pp. 21–29. doi:10.1109/SEQUEN.1997.666900
[11b] Indyk, P. & Motwani, R. "Approximate Nearest Neighbors: Towards Removing the Curse of Dimensionality". STOC 1998, pp. 604–613. doi:10.1145/276698.276876
[12a] Malkov, Y. A. & Yashunin, D. A. "Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs". IEEE TPAMI 42(4):824–836, 2020. doi:10.1109/TPAMI.2018.2889473, arXiv:1603.09320
[12b] Johnson, J., Douze, M., Jégou, H. "Billion-Scale Similarity Search with GPUs". IEEE Trans. Big Data 7(3):535–547, 2021. doi:10.1109/TBDATA.2019.2921572, arXiv:1702.08734
[13] Cohen, W. W., Ravikumar, P., Fienberg, S. E. "A Comparison of String Distance Metrics for Name-Matching Tasks". IIWeb-03 workshop (IJCAI), 2003. dblp conf/ijcai/CohenRF03
[14] Winkler, W. E. "String Comparator Metrics and Enhanced Decision Rules in the Fellegi-Sunter Model of Record Linkage". Proc. Section on Survey Research Methods, ASA, 1990, pp. 354–359. ERIC ED325505
[15] Bilenko, M. & Mooney, R. J. "Adaptive Duplicate Detection Using Learnable String Similarity Measures". KDD 2003, pp. 39–48. doi:10.1145/956750.956759
[16] Konda, P. et al. "Magellan: Toward Building Entity Matching Management Systems". PVLDB 9(12):1197–1208, 2016. doi:10.14778/2994509.2994535
[17] Mudgal, S. et al. "Deep Learning for Entity Matching: A Design Space Exploration". SIGMOD 2018, pp. 19–34. doi:10.1145/3183713.3196926
[18] Li, Y., Li, J., Suhara, Y., Doan, A., Tan, W.-C. "Deep Entity Matching with Pre-Trained Language Models". PVLDB 14(1):50–60, 2020. doi:10.14778/3421424.3421431, arXiv:2004.00584
[19] Brunner, U. & Stockinger, K. "Entity Matching with Transformer Architectures – A Step Forward in Data Integration". EDBT 2020, pp. 463–473. doi:10.5441/002/edbt.2020.58
[20] Thirumuruganathan, S. et al. "Deep Learning for Blocking in Entity Matching: A Design Space Exploration". PVLDB 14(11):2459–2472, 2021. doi:10.14778/3476249.3476294
[21] Barlaug, N. & Gulla, J. A. "Neural Networks for Entity Matching: A Survey". ACM TKDD 15(3), art. 52, 2021. doi:10.1145/3442200
[22] Köpcke, H., Thor, A., Rahm, E. "Evaluation of entity resolution approaches on real-world match problems". PVLDB 3(1–2):484–493, 2010. doi:10.14778/1920841.1920904
[23] Elmagarmid, A. K., Ipeirotis, P. G., Verykios, V. S. "Duplicate Record Detection: A Survey". IEEE TKDE 19(1):1–16, 2007. doi:10.1109/TKDE.2007.250581
[24] Ke, G. et al. "LightGBM: A Highly Efficient Gradient Boosting Decision Tree". NeurIPS 30, 2017.
[25a] Chen, T. & Guestrin, C. "XGBoost: A Scalable Tree Boosting System". KDD 2016, pp. 785–794. doi:10.1145/2939672.2939785
[25b] Prokhorenkova, L. et al. "CatBoost: unbiased boosting with categorical features". NeurIPS 31, 2018. arXiv:1706.09516
[26a] Niculescu-Mizil, A. & Caruana, R. "Predicting Good Probabilities with Supervised Learning". ICML 2005, pp. 625–632. doi:10.1145/1102351.1102430
[26b] Platt, J. "Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods". Advances in Large Margin Classifiers, MIT Press, 1999, pp. 61–74.
[26c] Zadrozny, B. & Elkan, C. "Transforming Classifier Scores into Accurate Multiclass Probability Estimates". KDD 2002, pp. 694–699. doi:10.1145/775047.775151
[27] Lipton, Z. C., Elkan, C., Naryanaswamy, B. "Optimal Thresholding of Classifiers to Maximize F1 Measure". ECML-PKDD 2014, LNCS 8725, pp. 225–239. doi:10.1007/978-3-662-44851-9_15, arXiv:1402.1892
[28] Ye, N., Chai, K. M. A., Lee, W. S., Chieu, H. L. "Optimizing F-Measures: A Tale of Two Approaches". ICML 2012. arXiv:1206.4625
[29] Karpukhin, V. et al. "Dense Passage Retrieval for Open-Domain Question Answering". EMNLP 2020, pp. 6769–6781. doi:10.18653/v1/2020.emnlp-main.550, arXiv:2004.04906
[30] Robinson, J., Chuang, C.-Y., Sra, S., Jegelka, S. "Contrastive Learning with Hard Negative Samples". ICLR 2021. arXiv:2010.04592
[31] Reimers, N. & Gurevych, I. "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks". EMNLP-IJCNLP 2019, pp. 3980–3990. doi:10.18653/v1/D19-1410, arXiv:1908.10084
[32] Linacre, R., Lindsay, S., Manassis, T., Slade, Z., Hepworth, K. et al. "Splink: Free software for probabilistic record linkage at scale". International Journal of Population Data Science 7(3), 2022. doi:10.23889/ijpds.v7i3.1794
[33] Papadakis, G., Koutrika, G., Palpanas, T., Nejdl, W. "Meta-Blocking: Taking Entity Resolution to the Next Level". IEEE TKDE 26(8):1946–1960, 2014. doi:10.1109/TKDE.2013.54
[33b] Papadakis, G., Ioannou, E., Palpanas, T., Niederée, C., Nejdl, W. "A Blocking Framework for Entity Resolution in Highly Heterogeneous Information Spaces". IEEE TKDE 25(12):2665–2682, 2013. doi:10.1109/TKDE.2012.150
[34] Sarawagi, S. & Bhamidipaty, A. "Interactive Deduplication using Active Learning". KDD 2002, pp. 269–278. doi:10.1145/775047.775087
[35] Peeters, R., Steiner, A., Bizer, C. "Entity Matching using Large Language Models". EDBT 2025, pp. 529–541. arXiv:2310.11244 (DOI unverified)
[36] van Rijsbergen, C. J. "Information Retrieval", 2nd ed. Butterworths, 1979. ISBN 0-408-70929-4 (E-measure / F-measure origin, ch. 7)
[37] Hassanzadeh, O., Chiang, F., Lee, H. C., Miller, R. J. "Framework for Evaluating Clustering Algorithms in Duplicate Detection". PVLDB 2(1):1282–1293, 2009. doi:10.14778/1687627.1687771
[38a] Comber, S. & Arribas-Bel, D. "Machine learning innovations in address matching: A practical comparison of word2vec and CRFs". Transactions in GIS 23(2):334–348, 2019. doi:10.1111/tgis.12522
[38b] Lin, Y., Kang, M., Wu, Y., Du, Q., Liu, T. "A deep learning architecture for semantic address matching". IJGIS 34(3):559–576, 2020. doi:10.1080/13658816.2019.1681431
[38c] Basile, A. et al. "Disambiguation of company names via deep recurrent networks". Expert Systems with Applications 238:122035, 2024. doi:10.1016/j.eswa.2023.122035, arXiv:2303.05391
[38d] Gschwind, T., Miksovic, C., Minder, J., Mirylenka, K., Scotton, P. "Fast Record Linkage for Company Entities". arXiv:1907.08667, 2019 (IEEE BigData 2019 venue unverified)
[38e] Christen, P. "A Comparison of Personal Name Matching: Techniques and Practical Issues". ICDM Workshops 2006, pp. 290–294. doi:10.1109/ICDMW.2006.2
[39a] Gregg, F. & Eder, D. "Dedupe" (software, MIT). https://github.com/dedupeio/dedupe
[39b] de Bruin, J. "Python Record Linkage Toolkit" (software, BSD-3). doi:10.5281/zenodo.3559042. Note: pins pandas<3, incompatible with our environment; cite only.
[39c] Bachmann, M. "RapidFuzz" (software, MIT). https://github.com/rapidfuzz/RapidFuzz
[39d] ING Bank. "sparse_dot_topn" (software, Apache-2.0). https://github.com/ing-bank/sparse_dot_topn

## 2. Libraries and models (PyPI / Hugging Face as of 2026-09-25)

Policy. Only MIT, BSD, Apache-2.0 and ISC code goes into the solution (`requirements.txt`,
`src/`, notebooks), including transitive dependencies: run `pip-licenses` before pinning a
new package. GPL and Artistic packages are banned even for a one-off script that touches the
submission. `usaddress` and `libpostal` are rejected on fair-play grounds, not licence: they
ship models trained on external labelled address data (01 §9), which the organisers may read
as an external lookup. `recordlinkage` pins pandas < 3 and cannot be installed next to our
Arrow-string pandas 3 environment: cite only. A pretrained encoder counts as a model: if one
ships, its licence (MIT / Apache-2.0) and parameter count (≤ 8B) go into
`Documentation_template.md`. Bold rows are non-permissive or otherwise excluded.

### 2.1 Python packages

| Package | Version | Licence | Role / notes |
|---|---|---|---|
| rapidfuzz | 3.14.6 | MIT | Levenshtein, Indel, Jaro-Winkler, token ratios; `process.cpdist` pairwise with C++ threads (07 §2) |
| jellyfish | 1.2.1 | MIT | phonetic codes (Soundex, Metaphone); optional blocking key |
| sparse_dot_topn | 1.2.0 | Apache-2.0 | top-k sparse matrix products for P2/P3/P4 (06); numpy 2 OK |
| scikit-learn | 1.9.1 | BSD-3 | `TfidfVectorizer`, LR baseline, calibration, diagnostics |
| scipy | 1.18.1 | BSD-3 | CSR matrices, sparse row-wise Jaccard |
| lightgbm | 4.7.0 | MIT | the matcher (08) |
| xgboost | 3.4.1 | Apache-2.0 | D4 comparison, optional |
| catboost | 1.2.10 | Apache-2.0 | D-group tail, optional |
| polars | 1.44.2 | MIT | optional fast group-bys over Parquet caches |
| duckdb | 1.5.5 | MIT | optional SQL over Parquet caches |
| pynndescent | 0.6.0 | BSD-2 | ANN alternative; numba dependency risk (pin numba and numpy together) |
| hnswlib | 0.8.0 | Apache-2.0 | sdist only, stale; prefer faiss-cpu |
| faiss-cpu | 1.15.1 | MIT | brute-force or IVF-PQ top-k for the embedding pass (P5) |
| datasketch | 2.0.0 | MIT | MinHash / LSH; pin: 2.0 changed the default MinHash (signatures differ from 1.x) |
| sentence-transformers | 6.1.0 | Apache-2.0 | bi-encoders and cross-encoders; needs torch ≥ 2.2 |
| anyascii | 0.3.3 | ISC | transliteration of non-Latin rows (05 §5), chosen |
| **unidecode** | 1.4.0 | **GPL-2.0-or-later** | **banned** |
| **text-unidecode** | 1.3 | **Artistic / GPL** | **banned** |
| indic-transliteration | 2.3.82 | MIT | scheme-aware Indic transliteration; alternative to anyascii on the Indic slice if B3 under-delivers |
| **python-Levenshtein → Levenshtein** | 0.27.5 | **GPL-2.0-or-later** | **banned; rapidfuzz covers every scorer** |
| pyjarowinkler | 3.0.0 | Apache-2.0 | redundant with rapidfuzz |
| **usaddress** | 0.5.16 | MIT | **rejected**: ships a CRF model trained on external labelled US addresses (fair-play exposure), US-only |
| **libpostal / pypostal** | 1.1.11 | MIT | **rejected**: needs ≈ 1.8 GB model data trained on OpenStreetMap / OpenAddresses (fair-play risk, RAM) |
| regex | 2026.9.10 | Apache-2.0 / CNRI | Unicode script properties if `re` is not enough for the non-Latin detector |
| unicodedata2 | 18.0.0 | Apache-2.0 | current Unicode database for NFKD |
| optuna | 5.0.0 | MIT | hyper-parameter search in the D group, optional |

### 2.2 Pretrained encoders (all MIT or Apache-2.0, all far below 8B parameters)

| Model | Params | Licence | Notes |
|---|---|---|---|
| sentence-transformers/all-MiniLM-L6-v2 | 22.7M | Apache-2.0 | English-only; base for an optional cross-encoder on the uncertain band |
| paraphrase-multilingual-MiniLM-L12-v2 | 117.65M | Apache-2.0 | multilingual bi-encoder alternative |
| intfloat/multilingual-e5-small | ~118M (secondary source) | MIT | P5 candidate; needs `query:` / `passage:` prefixes |
| intfloat/multilingual-e5-base | ~278M (secondary source) | MIT | larger e5; CPU encode of 11.7M rows is out of budget |
| BAAI/bge-small-en-v1.5 | 33.4M | MIT | English only |
| BAAI/bge-m3 | ~568M | MIT | fp16 only on a 4 GB GPU |
| sentence-transformers/LaBSE | 470.9M | Apache-2.0 | language-agnostic; heavy |
| Alibaba-NLP/gte-multilingual-base | 305M | Apache-2.0 | `trust_remote_code` |
| nomic-embed-text-v1.5 | ~137M | Apache-2.0 | `trust_remote_code`, task prefixes |

## 3. Where each reference is used

| Reference | Used in | For |
|---|---|---|
| [1], [14] | 04 §4, §6; 08 (D1); 10 | Fellegi-Sunter decision theory; LR as supervised Fellegi-Sunter weights; Winkler's comparator and enhanced decision rules |
| [2], [3], [23] | 04 all groups; 05; 06; 07 | textbook definitions of normalisation, indexing and comparison functions used by the cards |
| [4], [5], [6] | 04 §2; 06 | blocking taxonomy, multi-pass union, recall vs reduction-ratio trade-off, sorted neighbourhood |
| [7] | 04 §2 | canopies as the historical form of TF-IDF top-k retrieval |
| [8], [9] | 04 §2 | prefix filtering and similarity joins as an optional P2 accelerator |
| [10] | 04 §2; 06 (A3) | BM25 weighting as a P3 ablation |
| [11a], [11b] | 04 §2 | MinHash / LSH (rejected at our scale) |
| [12a], [12b], [31] | 04 §2, §4; 06 (P5) | ANN indexes and sentence encoders for the experimental embedding pass |
| [13], [38e], [39c] | 04 §3; 07 | choice of string comparators (Jaro-Winkler, TF-IDF hybrids, phonetic codes); rapidfuzz scorers |
| [15] | 04 §4, §5 | learnable similarity (rejected) and static hard-pair selection |
| [16], [22] | 04 §4, §7, §8; 08 | Magellan-style feature-engineered matching (chosen design) and the benchmark evidence for it |
| [17], [18], [19], [20], [21], [35] | 04 §2, §4; 08 | DeepMatcher, Ditto / cross-encoders, DeepBlocker, neural-EM survey, LLM matching (rejected or optional band re-ranking) |
| [24], [25a], [25b] | 04 §4; 08 | LightGBM (chosen), XGBoost and CatBoost (optional comparisons) |
| [26a], [26b], [26c] | 04 §6; 10 | probability calibration before expected-F0.5 decoding |
| [27], [28] | 04 §6; 10 | F-optimal thresholds and decision-theoretic (expected-F) decoding |
| [29], [30], [34] | 04 §5; 09 | retrieval negatives, hard-negative mining, active learning (rejected as a loop) |
| [32], [39a], [39b] | 04 §4, §5; 17 §2 | Splink, Dedupe, recordlinkage: cited as alternatives, not used |
| [33], [33b] | 04 §2, §3; 06; 07 | meta-blocking (candidate cap, `ctx_pool_indegree`), token blocking (P3) |
| [36], [37] | 04 §6, §7; 01 §8; `metrics.py` | F-measure definition; cluster-level evaluation and consistency post-step |
| [38a], [38b], [38c], [38d] | 04 §1, §2, §4 | address parsing (CRF rejected), semantic address matching, company-name models, company record linkage |
| [39d] | 04 §2; 06; 02 §7 | `sparse_dot_topn` top-k products behind P2/P3/P4 |

## 4. Verification notes

- Verified by resolving the DOI on 2026-09-25: [1]–[10], [11a], [11b], [12a], [12b],
  [15]–[23], [25a], [26a], [26c], [27], [29], [31], [32], [33], [33b], [34], [37], [38a],
  [38b], [38c], [38e], [39b] (Zenodo).
- Verified on arXiv (no DOI, or DOI not used): [25b], [28], [30], [35], [38d]. The arXiv ids
  given next to a DOI ([4], [12a], [12b], [18], [27], [29], [31], [38c]) match the DOI record.
- Verified through another authoritative page: [13] dblp key; [14] ERIC ED325505; [24]
  NeurIPS 2017 proceedings (no DOI issued); [26b] MIT Press chapter (no DOI); [36] ISBN;
  [39a], [39c], [39d] GitHub repositories and their licence files.
- Package versions and licences: PyPI metadata on 2026-09-25. Model licences and parameter
  counts: Hugging Face model cards on the same day, except where marked.
- Still secondary-source, to fix before the final documentation: the parameter counts of
  `multilingual-e5-small` (~118M) and `-base` (~278M) come from third-party summaries, not
  the model card or paper; confirm with `sum(p.numel() for p in model.parameters())` before
  shipping either. [35]: the EDBT 2025 DOI could not be resolved; cite arXiv:2310.11244 with
  the OpenProceedings page numbers. [38d]: the IEEE BigData 2019 venue is quoted from the
  arXiv comment field only; cite as arXiv.
