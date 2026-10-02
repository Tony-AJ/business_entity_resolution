"""Smoke test of the final pipeline (final.py) on a generated dataset, on the CPU.

The real run takes ~3.5 h on a GPU; this one runs every step of both stages on 300 train and
180 test entities in under a minute: normalisation, token map, mock fold, blocking, stage 1
on disk chunks, the stage-1 passes, decoy features, stage 2, calibration, the decoder grid,
test inference, both output files and both checks.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import HEADER, write_tsv
from entity_resolution import config as C
from entity_resolution.data import read_id_lists
from entity_resolution.features import DEFAULT_GROUPS
from entity_resolution.final import (
    BLOCKING_KEY,
    FinalConfig,
    compare_to_submitted,
    main,
    sha256,
    stage1_pipeline,
    stage_a,
    stage_b,
)
from entity_resolution.model import MatcherParams
from entity_resolution.normalize import NormaliseConfig
from entity_resolution.twostage import TwoStageConfig
from test_pipeline import tiny_cfg

WORDS = ("acme globex initech umbrella hooli vandelay soylent stark wayne wonka tyrell cyber "
         "aperture oscorp monarch nakatomi duff krusty gekko sirius cobalt amber birch cedar "
         "delta ember falcon granite harbor indigo jasper kestrel lotus maple nova onyx pebble "
         "quartz raven sable tundra umber velvet willow zephyr").split()
LEGAL = {"US": ["Inc", "LLC", "Corp"], "India": ["Pvt Ltd", "LLP", "Private Limited"],
         "France": ["SARL", "SAS", "SA"]}
STREETS = "Main Oak Elm Park Lake Hill Market Station Church Mill River Cedar".split()
STREET_TYPE = {"US": ("St", "Street"), "India": ("Rd", "Road"), "France": ("R.", "Rue")}
CITY = {"US": ("Springfield", "Austin"), "India": ("Pune", "Surat"), "France": ("Paris", "Lyon")}


def _split(rng: np.random.Generator, tag: str, countries: tuple[str, ...], n: int) -> tuple:
    """S1 records, S2 / S3 records and the truth of one split: 0-3 noisy true records per
    entity, and for half of them an unowned decoy (the name + "Holdings", a nearby number)."""
    s1, pool, truth = [], {2: [], 3: []}, []
    for country in countries:
        for _ in range(n):
            k = len(s1)
            w1, w2 = (str(w).title() for w in rng.choice(WORDS, 2, replace=False))
            num, street = int(rng.integers(10, 999)), str(rng.choice(STREETS))
            short, full = STREET_TYPE[country]
            city = str(rng.choice(CITY[country]))
            name = f"{w1} {w2} {rng.choice(LEGAL[country])}"
            sid = f"S1-{tag}{k:05d}"
            s1.append([sid, name, f"{num} {street} {full}, {city}", country])
            matched = []
            for j in range(int(rng.integers(0, 4))):
                src, pid = 2 + j % 2, f"S{2 + j % 2}-{tag}{k:05d}{j}"
                vname = name.upper() if rng.random() < 0.3 else f"{w1} {w2}"
                vaddr = f"{num} {street} {short}" + (f", {city}" if rng.random() < 0.7 else "")
                pool[src].append([pid, vname, vaddr, country])
                matched.append(pid)
            if rng.random() < 0.5:
                src = 2 + k % 2
                pool[src].append([f"S{src}-{tag}{k:05d}9", f"{w1} {w2} Holdings",
                                  f"{num + int(rng.integers(1, 9))} {street} {short}, {city}",
                                  country])
            truth.append([sid, ",".join(matched)])
    return s1, pool, truth


@pytest.fixture
def generated_dir(tmp_path: Path) -> Path:
    """Challenge files of 300 train entities (US, India) and 180 test ones (+ France)."""
    rng = np.random.default_rng(0)
    root = tmp_path / "gen"
    for split, countries, n in (("train", ("US", "India"), 150),
                                ("test", ("France", "India", "US"), 60)):
        s1, pool, truth = _split(rng, split[:2], countries, n)
        write_tsv(root / split / f"{split}_source1.tsv", HEADER, s1)
        for src in (2, 3):
            write_tsv(root / split / f"{split}_source{src}.tsv", HEADER, pool[src])
        if split == "train":
            write_tsv(root / split / "train_ground_truth.tsv",
                      [C.S1_ID, C.MATCHED_IDS], truth)
    return root


def _small(tmp_path: Path, dataset_dir: Path) -> FinalConfig:
    """The final configuration sized for the generated files, on the CPU."""
    cpu = MatcherParams(backend="xgb", device="cpu", num_leaves=7, learning_rate=0.3,
                        n_estimators=30, early_stopping=5, num_threads=2)
    pipeline = replace(tiny_cfg(tmp_path, dataset_dir), normalise=NormaliseConfig(rules=3),
                       feature_groups=(*DEFAULT_GROUPS, "frequency", "idf", "token_freq",
                                       "ctx_idf", "address_extra"))
    return FinalConfig(pipeline=pipeline, n_stage1=10_000, stage1_model=cpu,
                       stage2=TwoStageConfig(floor=0.01, max_cands=16, rivals=True,
                                             train_roles=("fit", "tune"), model=cpu),
                       decode_w=(2.0, 4.0), misses=(0.0, 0.4))


def test_defaults_are_the_submission() -> None:
    """The default configuration is v110's blocking and rules v3, on the GPU."""
    fc = FinalConfig()
    assert fc.pipeline.blocking.key() == BLOCKING_KEY == stage1_pipeline().blocking.key()
    assert fc.pipeline.normalise.rules == 3 and fc.n_stage1 == 110_000
    assert fc.stage1_dir().name == "v110_db1f33c4_f0.01_k16_a1_c0_r1"
    assert fc.with_device("cpu").stage2.model.device == "cpu"
    assert fc.stage2.model.device == "cuda"


def test_final_pipeline_end_to_end(generated_dir: Path, tmp_path: Path) -> None:
    """Both stages write two valid files, one row per test S1; stage B is deterministic."""
    fc = _small(tmp_path, generated_dir)
    a = stage_a(fc, tmp_path / "models")
    assert a["stage1_entities"] > 0 and (tmp_path / "models" / "stage1" / "model").exists()
    assert (fc.stage1_dir() / "test" / "test_France.parquet").exists()   # unseen in train
    report = stage_b(fc, tmp_path / "models", tmp_path / "out")
    assert report["checks"]["our_checker"] == "PASS"
    assert report["mock_val"]["f_beta"] > 0.5 and report["mock_val"]["w"] in fc.decode_w
    _, matches = read_id_lists(tmp_path / "out" / C.MATCHING_FILE)
    _, cands = read_id_lists(tmp_path / "out" / C.CANDIDATE_FILE)
    test_s1 = pd.read_csv(generated_dir / "test" / "test_source1.tsv", sep="\t", dtype=str)
    assert [s for s, _ in matches] == test_s1[C.ENTITY_ID].tolist()   # every test S1, in order
    assert all(set(m) <= set(c) for (_, m), (_, c) in zip(matches, cands, strict=True))
    assert sum(map(len, (m for _, m in matches))) == report["test"]["matched_pairs"]
    assert report["test"]["s1"] == 180 and (tmp_path / "models" / "isotonic.json").exists()
    again = stage_b(fc, tmp_path / "models", tmp_path / "out2")
    for name in (C.MATCHING_FILE, C.CANDIDATE_FILE):
        assert sha256(tmp_path / "out" / name) == sha256(tmp_path / "out2" / name)
    assert not again["checks"]["vs_upload"][C.MATCHING_FILE]["identical_to_upload"]


def test_verify_compares_with_the_upload(tmp_path: Path) -> None:
    """--verify exits 1 for files that are not the uploaded ones (or are missing)."""
    for name in (C.MATCHING_FILE, C.CANDIDATE_FILE):
        (tmp_path / name).write_text("source1_entity_id\tmatched_entity_ids\n")
    assert not any(r["identical_to_upload"] for r in compare_to_submitted(tmp_path).values())
    assert main(["--verify", str(tmp_path)]) == 1
    assert main(["--verify", str(tmp_path / "missing")]) == 1
