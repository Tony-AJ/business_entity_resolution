"""Challenge metric: macro F_beta (beta = 0.5) over Source 1 entities.

Each Source 1 entity is scored on its own, then the scores are averaged with
singletons included:

    no true matches, empty prediction   -> 1.0
    no true matches, any prediction     -> 0.0
    true matches, empty prediction      -> 0.0
    otherwise                           -> F_beta of that entity's precision and recall

Entities missing from the prediction count as empty predictions.
"""
from __future__ import annotations

import argparse
from collections.abc import Collection, Mapping
from pathlib import Path
from statistics import fmean

import numpy as np

from .config import BETA
from .data import read_id_lists

IdLists = Mapping[str, Collection[str]]


def entity_fbeta(pred: Collection[str], truth: Collection[str], beta: float = BETA) -> float:
    pred, truth = set(pred), set(truth)
    if not truth:
        return 0.0 if pred else 1.0
    tp = len(pred & truth)
    if tp == 0:
        return 0.0
    precision, recall = tp / len(pred), tp / len(truth)
    b2 = beta * beta
    return (1 + b2) * precision * recall / (b2 * precision + recall)


def macro_fbeta(pred: IdLists, truth: IdLists, beta: float = BETA) -> float:
    """Mean per-entity F_beta over every Source 1 entity in ``truth``."""
    if not truth:
        raise ValueError("truth is empty")
    return fmean(entity_fbeta(pred.get(k, ()), v, beta) for k, v in truth.items())


def breakdown(pred: IdLists, truth: IdLists, beta: float = BETA) -> dict[str, float | int]:
    """Macro F_beta plus what drives it: singletons vs matched entities, pair P/R.

    Predictions for entities missing from ``truth`` are ignored.
    """
    if not truth:
        raise ValueError("truth is empty")
    scores = {k: entity_fbeta(pred.get(k, ()), v, beta) for k, v in truth.items()}
    singles = [s for k, s in scores.items() if not truth[k]]
    matched = [s for k, s in scores.items() if truth[k]]
    tp = n_pred = n_true = 0
    for k, v in truth.items():
        p, t = set(pred.get(k, ())), set(v)
        tp += len(p & t)
        n_pred += len(p)
        n_true += len(t)
    nan = float("nan")
    return {
        "f_beta": fmean(scores.values()),
        "f_beta_singletons": fmean(singles) if singles else nan,
        "f_beta_matched": fmean(matched) if matched else nan,
        "pair_precision": tp / n_pred if n_pred else nan,
        "pair_recall": tp / n_true if n_true else nan,
        "entities": len(scores),
        "singletons": len(singles),
    }


def candidate_report(candidates: IdLists, truth: IdLists, pool_size: int | None = None,
                     beta: float = BETA) -> dict[str, float | int]:
    """Blocking quality on labelled data: what the candidate set keeps and what it costs.

    pair_recall        true pairs kept among the candidates / all true pairs
    entity_recall      matched entities with at least one true match kept
    ceiling_f_beta     macro F_beta of a perfect matcher that only sees the candidates;
                       the best score any model can reach on top of this blocking
    candidates_*       candidates per Source 1 entity (mean, p95, max)
    reduction_ratio    1 - candidate pairs / all possible pairs (needs ``pool_size``,
                       the number of Source 2 + 3 records)

    Candidates for entities missing from ``truth`` are ignored.
    """
    if not truth:
        raise ValueError("truth is empty")
    counts, ceiling = [], []
    kept = n_true = hit_entities = matched_entities = 0
    for k, t in truth.items():
        t, c = set(t), set(candidates.get(k, ()))
        hit = t & c
        counts.append(len(c))
        kept += len(hit)
        n_true += len(t)
        if t:
            matched_entities += 1
            hit_entities += bool(hit)
        ceiling.append(entity_fbeta(hit, t, beta))  # a perfect matcher predicts exactly `hit`
    n = np.asarray(counts)
    nan = float("nan")
    return {
        "pair_recall": kept / n_true if n_true else nan,
        "entity_recall": hit_entities / matched_entities if matched_entities else nan,
        "ceiling_f_beta": fmean(ceiling),
        "candidates_mean": float(n.mean()),
        "candidates_p95": float(np.percentile(n, 95)),
        "candidates_max": int(n.max()),
        "candidate_pairs": int(n.sum()),
        "reduction_ratio": 1 - n.sum() / (len(n) * pool_size) if pool_size else nan,
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Score a matching file against ground truth.")
    ap.add_argument("--pred", type=Path, required=True, help="matching_results-format TSV")
    ap.add_argument("--truth", type=Path, required=True, help="ground truth, same format")
    args = ap.parse_args(argv)
    pred = dict(read_id_lists(args.pred)[1])
    truth = {k: set(v) for k, v in read_id_lists(args.truth)[1]}
    for name, value in breakdown(pred, truth).items():
        print(f"{name:>18}  {value:.4f}" if isinstance(value, float) else f"{name:>18}  {value}")


if __name__ == "__main__":
    main()
