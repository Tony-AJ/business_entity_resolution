"""Business Entity Resolution: Amazon ML Challenge 2026.

Match Source 2 / Source 3 business records to the deduplicated Source 1 entities
they describe, optimising macro F0.5 (precision-heavy).

Modules:
    config      paths, file schema, metric constants
    data        TSV loaders for source files and ground truth, Parquet-cached
    split       fixed validation split shared by every experiment
    metrics     macro F0.5 per Source 1 entity, singletons included
    submission  write + validate matching_results.tsv / candidate_pairs.tsv
    tracking    experiment registry: vNNN folders, experiments.csv, timings
    snapshot    feature snapshots (M4): build matcher inputs once, evaluate models fast
"""
