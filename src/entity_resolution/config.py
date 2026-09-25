"""Central configuration: paths, file schema and metric constants.

Single source of truth for everything the pipeline stages share. Paths follow the
layout the organisers' tooling expects (dataset/ and output/ at the repo root);
entry points accept explicit path arguments to run from anywhere else.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "dataset"
OUTPUT = ROOT / "output"

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
