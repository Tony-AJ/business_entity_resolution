"""Learned token evidence (evidence.py): near-duplicate pairs, shrunk log-odds, lookups."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from entity_resolution import config as C
from entity_resolution.evidence import (
    EvidenceConfig,
    TokenEvidence,
    evidence_table,
    fit_token_evidence,
    near_duplicates,
)


def _records(prefix: str, names: list[str]) -> pd.DataFrame:
    """Records of one country holding these sorted word sets (name_sorted)."""
    return pd.DataFrame({C.ENTITY_ID: [f"{prefix}-{i}" for i in range(len(names))],
                         C.COUNTRY: "US", "name_sorted": names}).astype("str")


def _case() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(s1n, pooln, truth): "center" joins three true matches, "holdings" three other
    businesses, the true match of S1-3 drops "group"; "stark" is two S1 names, so it
    anchors nothing (its match "center stark" teaches nothing)."""
    s1 = _records("S1", ["acme", "globex", "initech", "group hooli", "stark", "stark"])
    pool = _records("S2", ["acme center", "center globex", "center initech", "acme holdings",
                           "globex holdings", "holdings initech", "hooli", "stark zork",
                           "center stark"])
    truth = pd.DataFrame({C.S1_ID: ["S1-0", "S1-1", "S1-2", "S1-3", "S1-4"],
                          C.ENTITY_ID: ["S2-0", "S2-1", "S2-2", "S2-6", "S2-8"]}).astype("str")
    return s1, pool, truth


def test_near_duplicates_label_one_word_differences() -> None:
    """Pool = S1 + a word and S1 = pool + a word, anchored on unique S1 names, labelled."""
    pairs = near_duplicates(*_case(), EvidenceConfig(sample_share=1.0))
    got = sorted(zip(pairs["side"].astype(str), pairs["word"], pairs["match"], strict=True))
    assert got == [("pool", "center", 1)] * 3 + [("pool", "holdings", 0)] * 3 + [
        ("s1", "group", 1)]
    assert len(near_duplicates(*_case(), EvidenceConfig(sample_share=0.0))) == 0


def test_evidence_table_shrinks_toward_zero_and_needs_support() -> None:
    """log((n1 + k p0) / (n0 + k (1 - p0))) - logit(p0); rare words and a side without
    decoys get no value."""
    s1, pool, truth = _case()
    ev = fit_token_evidence(s1, pool, truth, EvidenceConfig(sample_share=1.0, min_support=3,
                                                            prior=1.0))
    assert ev.pool == pytest.approx({"center": np.log(7.0), "holdings": -np.log(7.0)}, abs=1e-4)
    assert ev.s1 == {} and ev.info["s1"] == {"pairs": 1, "matches": 1}   # no contrast
    shrunk = fit_token_evidence(s1, pool, truth, EvidenceConfig(sample_share=1.0,
                                                                min_support=3, prior=20.0))
    assert shrunk.pool["center"] == pytest.approx(np.log(13 / 10), abs=1e-4)
    assert fit_token_evidence(s1, pool, truth, EvidenceConfig(sample_share=1.0,
                                                              min_support=4)).pool == {}
    empty = evidence_table(near_duplicates(s1.iloc[:0], pool, truth))
    assert empty.pool == {} and empty.s1 == {}


def test_token_evidence_lookup_and_record() -> None:
    """Unknown words read NaN; the table survives a JSON round trip."""
    ev = TokenEvidence({"center": 2.0, "holdings": -3.0}, {"group": 1.5}, 1.0, {"pool": {}})
    words = pa.array(["holdings", "zork", "center"])
    np.testing.assert_array_equal(ev.values("pool", words), [-3.0, np.nan, 2.0])
    assert np.isnan(ev.values("s1", words)).all()
    again = TokenEvidence.from_record(json.loads(json.dumps(ev.record())))
    assert again == ev
    assert EvidenceConfig().learn is False    # opt-in
