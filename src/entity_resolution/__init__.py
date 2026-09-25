"""Business Entity Resolution: Amazon ML Challenge 2026.

Match Source 2 / Source 3 business records to the deduplicated Source 1 entities
they describe, optimising macro F0.5 (precision-heavy).

Modules:
    config      paths, file schema, metric constants
    data        TSV loaders for source files and ground truth
    metrics     macro F0.5 per Source 1 entity, singletons included
"""
