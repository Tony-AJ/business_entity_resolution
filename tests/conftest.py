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


# The pool writes "Center" / "Services" into six true matches; "Holdings" is part of two S1
# names and is copied as is, and it marks four decoys ("Acme Holdings": another business, no
# true match; three fall in the train fold). Ids from 9 on put entities on every side at
# frac=0.5 (val: 1, 4, 7; inner tune: 0, 6; fit: 2, 3, 5). The test split is the shared one.
FILLER_NAMES = [("Acme", "Acme Center"), ("Globex", "Globex Services Center"),
                ("Initech", "Center Initech"), ("Umbrella", "Umbrella Center"),
                ("Soylent", "Soylent Center"), ("Vandelay", "Vandelay Center"),
                ("Hooli Holdings", "Hooli Holdings"), ("Stark Holdings", "Stark Holdings")]


@pytest.fixture
def filler_dir(tmp_path: Path) -> Path:
    """Challenge files whose pool names carry filler words (8 US entities, 8 true pairs)."""
    root = tmp_path / "fill"
    s1 = [[f"S1-2{i + 9:04d}", n, f"{i + 1} Main St, Springfield", "US"]
          for i, (n, _) in enumerate(FILLER_NAMES)]
    pool = [[f"S{2 + i % 2}-2{i + 9:04d}", p, f"{i + 1} Main Street, Springfield", "US"]
            for i, (_, p) in enumerate(FILLER_NAMES)]
    decoys = [[f"S{2 + i % 2}-29{i:03d}", f"{n} Holdings", f"{90 + i} Oak Ave, Dallas", "US"]
              for i, n in enumerate(["Globex", "Acme", "Initech", "Umbrella"])]
    for s, rows in ((1, s1), (2, pool[0::2] + decoys[0::2]), (3, pool[1::2] + decoys[1::2])):
        write_tsv(root / "train" / f"train_source{s}.tsv", HEADER, rows)
    for s, rows in TEST.items():
        write_tsv(root / "test" / f"test_source{s}.tsv", HEADER, rows)
    write_tsv(root / "train" / "train_ground_truth.tsv",
              ["source1_entity_id", "matched_entity_ids"], [[a[0], b[0]] for a, b in zip(
                  s1, pool, strict=True)])
    return root
