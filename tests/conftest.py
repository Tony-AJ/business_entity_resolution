"""Tiny synthetic challenge files, written per test under tmp_path.

Mirrors the real layout (dataset/{train,test}/...) and plants the traps loaders
must survive: commas in addresses, a business literally named "NA", an empty
address, a singleton, and a country (France) that only exists in test.
"""
from __future__ import annotations

from pathlib import Path

import pytest

HEADER = ["entity_id", "business_name", "business_address", "country"]

TRAIN = {
    1: [["S1-00001", "Acme Corp", "12 Main St, Springfield, IL", "US"],
        ["S1-00002", "Sharma Traders Pvt Ltd", "Near SBI ATM, MG Road, Pune", "India"],
        ["S1-00003", "NA", "", "US"]],
    2: [["S2-00001", "ACME Corporation", "12 Main Street, Springfield", "US"],
        ["S2-00002", "Sharma Traders Private Limited", "MG Rd, Pune 411001", "India"]],
    3: [["S3-00001", "Acme Corp.", "12 Main St", "US"],
        ["S3-00002", "Globex LLC", "1 Elm Rd, Austin, TX", "US"]],
}
TRUTH = [["S1-00001", "S2-00001,S3-00001"],
         ["S1-00002", "S2-00002"],
         ["S1-00003", ""]]
TEST = {
    1: [["S1-00010", "Boulangerie Dupont SARL", "5 Rue de la Paix, Paris", "France"],
        ["S1-00011", "Patel & Sons", "Station Road, Surat", "India"]],
    2: [["S2-00010", "Boulangerie Dupont", "5 rue de la Paix, 75002 Paris", "France"]],
    3: [["S3-00010", "Patel and Sons", "Stn Rd, Surat", "India"],
        ["S3-00011", "Initech", "4 Oak Ave, Dallas", "US"]],
}


def write_tsv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["\t".join(r) for r in [header, *rows]]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture
def dataset_dir(tmp_path: Path) -> Path:
    root = tmp_path / "dataset"
    for split, sources in (("train", TRAIN), ("test", TEST)):
        for s, rows in sources.items():
            write_tsv(root / split / f"{split}_source{s}.tsv", HEADER, rows)
    write_tsv(root / "train" / "train_ground_truth.tsv",
              ["source1_entity_id", "matched_entity_ids"], TRUTH)
    return root
