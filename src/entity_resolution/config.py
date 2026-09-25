"""Central configuration: paths, file schema and metric constants.

Single source of truth for everything the pipeline stages share. Paths follow the
layout the organisers' tooling expects (dataset/ and output/ at the repo root);
entry points accept explicit path arguments to run from anywhere else.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STUDENT_RESOURCE = ROOT / "dataset" / "student_resource"  # organisers' zip, unzipped
DATASET_CANDIDATES = (ROOT / "dataset", STUDENT_RESOURCE / "dataset")


def _find_dataset() -> Path:
    """First known layout holding a train/ folder: flat dataset/ or the unzipped zip."""
    return next((p for p in DATASET_CANDIDATES if (p / "train").is_dir()), DATASET_CANDIDATES[0])


DATASET = _find_dataset()
OFFICIAL_VALIDATOR = STUDENT_RESOURCE / "utils" / "validate_submission.py"
OUTPUT = ROOT / "output"
EXPERIMENTS = ROOT / "experiments"          # one vNNN_<slug>/ folder per experiment
EXPERIMENTS_CSV = EXPERIMENTS / "experiments.csv"
LEADERBOARD = ROOT / "LEADERBOARD.md"

SEED = 42

# ------------------------------------------------------------- file schema ----
SEP = "\t"            # every challenge file is TSV: addresses contain commas
ID_LIST_SEP = ","     # inside matched / candidate ID lists
SPLITS = ("train", "test")
SOURCES = (1, 2, 3)   # Source 1 is the deduplicated reference
SOURCE_PREFIX = {1: "S1-", 2: "S2-", 3: "S3-"}

ENTITY_ID = "entity_id"
NAME = "business_name"
ADDRESS = "business_address"
COUNTRY = "country"   # open set: test adds France, unseen in train
SOURCE_COLUMNS = (ENTITY_ID, NAME, ADDRESS, COUNTRY)

S1_ID = "source1_entity_id"
MATCHED_IDS = "matched_entity_ids"
CANDIDATE_IDS = "candidate_entity_ids"

GROUND_TRUTH_FILE = "train_ground_truth.tsv"
MATCHING_FILE = "matching_results.tsv"
CANDIDATE_FILE = "candidate_pairs.tsv"

# ------------------------------------------------------------------ metric ----
BETA = 0.5            # F_beta with beta < 1 weights precision over recall
