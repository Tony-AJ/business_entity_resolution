"""Synthetic datasets and small configs for the snapshot tests (tests/test_snapshot.py).

``make_dataset`` writes a train split of a few hundred S1 entities with typo, case,
word-order and address-order variants plus same-name decoys at another address: enough
signal for LightGBM to train on, so snapshot evaluation can be checked against the
pipeline with the default ``lgbm`` backend. ``tiny_blocking`` / ``tiny_cfg`` size the
pipeline for such tiny vocabularies. Kept out of the test module so each stays short; not
a test module itself (no ``test_`` prefix), and it never reads the real dataset/.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from entity_resolution import config as C
from entity_resolution.blocking import BlockingConfig, TopKSpec
from entity_resolution.decision import Grid
from entity_resolution.model import MatcherParams
from entity_resolution.pipeline import PipelineConfig

HEADER = ["entity_id", "business_name", "business_address", "country"]
WORDS = ["acme", "globex", "initech", "umbrella", "stark", "wayne", "wonka", "tyrell",
         "cyberdyne", "soylent", "hooli", "vandelay", "sterling", "dunder", "prestige",
         "monarch", "zenith", "apex", "summit", "harbor", "pioneer", "liberty", "eagle",
         "falcon", "maple", "cedar", "river", "sunrise", "golden", "silver", "royal", "metro"]
KINDS = ["traders", "foods", "motors", "textiles", "systems", "labs", "builders", "pharma"]
STREETS = ["main", "oak", "elm", "park", "lake", "hill", "station", "market", "church"]
CITIES = {"US": ["springfield", "austin", "denver", "boston"],
          "India": ["pune", "surat", "jaipur", "nagpur"]}
SUFFIX = {"US": ["Inc", "LLC", "Corp"], "India": ["Pvt Ltd", "Private Limited"]}


def tiny_blocking() -> BlockingConfig:
    """TF-IDF passes sized for tiny vocabularies (min_df=1, max_df=1.0), one thread."""
    return BlockingConfig(
        name_char=TopKSpec("name_core", top_k=5, min_sim=0.1, max_df=1.0, min_df=1),
        name_addr_word=TopKSpec("name_addr", "word", (1, 2), 5, 0.05, 1.0, min_df=1),
        n_threads=1)


def tiny_cfg(tmp_path: Path, dataset_dir: Path) -> PipelineConfig:
    """test_pipeline.tiny_cfg: heuristic matcher, small grid, cache under tmp_path."""
    grid = Grid(tau_abs=(0.3, 0.7, 0.2), tau_rel=(0.0, 0.7), single_delta=(0.0, 0.1),
                max_matches=(11,))
    return PipelineConfig(blocking=tiny_blocking(), model=MatcherParams(backend="heuristic"),
                          grid=grid, n_fit_s1=10, n_stop_s1=10, n_tune_s1=None,
                          chunk_rows=1000, dataset_dir=dataset_dir,
                          cache_dir=tmp_path / "cache")


def write_tsv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    """One challenge TSV file (UTF-8, tab-separated, header first)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join("\t".join(r) for r in [header, *rows]) + "\n", encoding="utf-8")


def make_dataset(root: Path, n_s1: int = 260, seed: int = 0) -> Path:
    """A train split of ``n_s1`` S1 entities: 0-3 noisy copies each, same-name decoys at
    another address, unmatched pool records; plus a tiny test split for the loaders."""
    rng = np.random.default_rng(seed)
    rows: dict[int, list[list[str]]] = {1: [], 2: [], 3: []}
    truth: list[list[str]] = []

    def pool(name: str, addr: str, country: str) -> str:
        """Append a record to Source 2 or 3 at random and return its id."""
        src = int(rng.integers(2, 4))
        eid = f"S{src}-{len(rows[src]) + 1:05d}"
        rows[src].append([eid, name, addr, country])
        return eid

    def typo(word: str) -> str:
        """Swap two neighbouring letters."""
        i = int(rng.integers(0, len(word) - 1))
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]

    for i in range(1, n_s1 + 1):
        country = "US" if rng.random() < 0.6 else "India"
        w1, w2 = (str(w) for w in rng.choice(WORDS, 2, replace=False))
        kind, city = str(rng.choice(KINDS)), str(rng.choice(CITIES[country]))
        street, suffix = str(rng.choice(STREETS)), str(rng.choice(SUFFIX[country]))
        num = int(rng.integers(1, 999))
        road = "St" if country == "US" else "Road"
        core = f"{w1.title()} {w2.title()} {kind.title()}"
        sid = f"S1-{i:05d}"
        rows[1].append([sid, f"{core} {suffix}", f"{num} {street.title()} {road}, "
                                                 f"{city.title()}", country])
        matched = []
        for _ in range(int(rng.choice(4, p=[0.2, 0.4, 0.25, 0.15]))):
            name = [core, f"{core} {suffix}".upper(), f"{typo(w1).title()} {w2.title()} "
                    f"{kind.title()} {suffix}", f"{w2.title()} {w1.title()} {kind.title()}"
                    ][int(rng.integers(0, 4))]
            addr = [f"{num} {street} {'Street' if country == 'US' else 'Rd'}",
                    f"{num} {street.title()} {road}, {city.title()}",
                    f"{city.title()}, {num} {street.title()} {road}"][int(rng.integers(0, 3))]
            matched.append(pool(name, addr, country))
        if rng.random() < 0.35:   # same name, another address: a decoy
            other = str(rng.choice([s for s in STREETS if s != street]))
            pool(core, f"{int(rng.integers(1, 999))} {other.title()} {road}, "
                       f"{city.title()}", country)
        truth.append([sid, ",".join(matched)])
    for _ in range(n_s1 // 3):    # unrelated records nobody owns
        country = "US" if rng.random() < 0.6 else "India"
        w1, w2 = (str(w) for w in rng.choice(WORDS, 2, replace=False))
        pool(f"{w1.title()} {w2.title()} {str(rng.choice(KINDS)).title()}",
             f"{int(rng.integers(1, 999))} {str(rng.choice(STREETS)).title()} Lane", country)
    for s in (1, 2, 3):
        write_tsv(root / "train" / f"train_source{s}.tsv", HEADER, rows[s])
        write_tsv(root / "test" / f"test_source{s}.tsv", HEADER, [rows[s][0]])
    write_tsv(root / "train" / C.GROUND_TRUTH_FILE, ["source1_entity_id",
                                                     "matched_entity_ids"], truth)
    return root
