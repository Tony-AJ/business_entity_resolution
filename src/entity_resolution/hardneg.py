"""Hard-negative mining for the matcher (09, M4, plan groups HN2-HN4, versions v070-v079).

Why: the negatives that cost F0.5 are not random pool records (never candidates) but
same-name decoys and the name twins of singletons (62% of singletons have one, 01 §3), and a
false merge on a singleton costs the entity's whole point. Blocking already mines the first
kind (HN1: every candidate of the sampled fit S1 is a training row); this module decides how
much each fit row counts:

    HardNegConfig         one experiment's settings (09 §3 / §4), ``record()`` for metrics.json
    singleton_mask        rows whose S1 entity has no true pair (the "no match" evidence)
    mine_by_rule          label-0 rows of one 09 §3 kind, read from the feature columns
    is_easy_negative      label-0 rows no model can get wrong (09 §4), never a singleton's
    easy_keep             order-free hash sample of the easy negatives (HN2)
    mine_false_positives  scored pairs with prob >= threshold that are not true pairs
    mine_near_misses      true pairs with prob < threshold
    fit_weights           one weight per fit row from all of the above, plus ``hn_counts``
    weight_fn             ``fit_weights`` shaped as ``snapshot.evaluate_params``'s weight_fn
    round2                09 §7 on a snapshot: fit, mine the fit side in sample, refit, keep
                          the refit only when tune F0.5 rises by ``ROUND2_MARGIN``

Weights replace 09 §4's duplication (``Matcher.fit(weight=)`` exists now): a weight of k is
k copies in LightGBM's gradient and hessian sums without the memory, and a weight of 0 drops
the row (``snapshot`` leaves it out of the training set, so HN2 fits faster). Everything
reads the fit side only (09 §5): the snapshot's fit rows, their labels and fit-side truth;
round 2 scores the fit rows with the model trained on them, and the tune side is used only
to choose between the two models, as ``decision.tune`` uses it. ``label``, ``prob`` and the
mined kinds never enter ``X``. The extra S1 sample of HN3b / HN5 (``n_extra_s1``) is not
built: it needs a second blocked fit side in the snapshot.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from . import config as C
from .data import isin
from .decision import DEFAULT_GRID, Grid, tune
from .evaluate import pair_in
from .split import hash_unit
from .trainset import SAMPLE_SEED

if TYPE_CHECKING:   # snapshot imports the pipeline; only the round-2 loop needs it
    from .model import Matcher, MatcherParams, SeedEnsemble
    from .snapshot import Snapshot

HN_KINDS = ("same_name_diff_addr", "same_addr_diff_name", "top_cosine_nonmatch",
            "singleton_decoy")
LABEL = "label"
# 09 §3 rule thresholds (similarities are in [0, 1]; NaN never satisfies a comparison)
SAME_NAME_ADDR_MAX = 0.2    # same_name_diff_addr: ad_jaccard below this
SAME_ADDR_MIN = 0.9         # same_addr_diff_name: ad_token_set at least this ...
SAME_ADDR_NAME_MAX = 0.5    # ... and core_token_set below this
# 09 §4 easy negative: names and addresses both far apart
EASY_NAME_MAX = 0.5         # sim_name_char below this
EASY_ADDR_MAX = 0.1         # ad_jaccard below this
ROUND2_MARGIN = 0.002       # 09 §7: the refit is kept only for a clear tune gain


@dataclass(frozen=True)
class HardNegConfig:
    """Settings of one hard-negative experiment (09 §3, §4, §7); the default changes nothing.

    ``easy_neg_keep``: share of easy negatives kept (HN2: 0.2). ``singleton_weight``: weight of
    every candidate of a singleton S1 (HN3: 2, 3). ``rule_kinds`` / ``rule_weight``: label-0
    rows of those 09 §3 kinds weigh ``rule_weight`` (on the base sample; the extra sample is
    not built). ``round2``: refit once on the model's own fit-side false positives (weight +
    ``fp_weight - 1``) and, with ``keep_near_misses``, missed true pairs (+
    ``near_miss_weight - 1``), both at ``round2_threshold`` (HN4; 09 §10 says 0.8 / 1 when
    the false positives look like label noise). ``seed`` keys the easy-negative hash sample.
    """

    easy_neg_keep: float = 1.0
    singleton_weight: float = 1.0
    rule_kinds: tuple[str, ...] = ()
    rule_weight: float = 1.0
    round2: bool = False
    round2_threshold: float = 0.5
    fp_weight: float = 2.0
    keep_near_misses: bool = True
    near_miss_weight: float = 2.0
    seed: int = SAMPLE_SEED

    def __post_init__(self) -> None:
        """Reject settings that would silently mean something else."""
        if not 0.0 <= self.easy_neg_keep <= 1.0:
            raise ValueError(f"easy_neg_keep must be in [0, 1], got {self.easy_neg_keep}")
        for name in ("singleton_weight", "rule_weight", "fp_weight", "near_miss_weight"):
            if getattr(self, name) < 1.0:
                raise ValueError(f"{name} must be >= 1 (1 = unchanged), got {getattr(self, name)}")
        if not 0.0 < self.round2_threshold < 1.0:
            raise ValueError(f"round2_threshold must be in (0, 1), got {self.round2_threshold}")
        unknown = [k for k in self.rule_kinds if k not in HN_KINDS]
        if unknown:
            raise ValueError(f"unknown rule kinds {unknown}; known: {list(HN_KINDS)}")
        object.__setattr__(self, "rule_kinds", tuple(self.rule_kinds))

    def record(self) -> dict:
        """JSON-ready settings for metrics.json (``hard_negatives``)."""
        return asdict(self)


def _col(X: pd.DataFrame, name: str) -> np.ndarray:
    """One feature as float32 (NaN kept); a clear error when the snapshot lacks it.

    Features are stored as float32, so rules compare in float32 against ``_f32`` thresholds:
    widened to float64, a stored 0.9 reads 0.8999999762 and would fail ``>= 0.9``.
    """
    if name not in X.columns:
        raise ValueError(f"hard-negative rules need the feature {name!r}, absent from X")
    return X[name].to_numpy(dtype=np.float32, na_value=np.nan)


def _f32(threshold: float) -> np.float32:
    """A rule threshold rounded as the features are."""
    return np.float32(threshold)


def _labels(meta: pd.DataFrame) -> np.ndarray:
    """The 0/1 ``label`` column of ``meta`` as int8."""
    return meta[LABEL].to_numpy(dtype=np.int8)


def singleton_mask(meta: pd.DataFrame, truth: pd.DataFrame | None = None) -> np.ndarray:
    """Per row: its S1 entity has no true pair (09 §4).

    With ``truth`` (the side's true pairs) this is the metric's singleton; without it, no
    candidate of the entity is labelled 1, which also counts entities whose true pairs were
    all lost by blocking (about 1%): they look like singletons to the model anyway.
    """
    s1 = meta[C.S1_ID]
    if truth is not None:
        return ~isin(s1, pd.Index(truth[C.S1_ID].unique()))
    matched = s1[_labels(meta) == 1].unique()
    return ~isin(s1, pd.Index(matched))


def mine_by_rule(meta: pd.DataFrame, X: pd.DataFrame, kind: str,
                 singleton: np.ndarray | None = None) -> np.ndarray:
    """Label-0 rows of one 09 §3 ``kind`` (bool mask over ``meta`` / ``X`` rows).

    ``same_name_diff_addr``: equal core names (``core_eq`` when the features have it, else
    ``core_ratio == 1``) and ``ad_jaccard < 0.2``; ``same_addr_diff_name``: ``ad_token_set
    >= 0.9`` and ``core_token_set < 0.5``; ``top_cosine_nonmatch``: ``ctx_rank_name == 1``;
    ``singleton_decoy``: every candidate of a singleton (``singleton`` from
    ``singleton_mask``, recomputed from labels when None). Label-1 rows are never selected.
    """
    if kind not in HN_KINDS:
        raise ValueError(f"unknown kind {kind!r}; known: {list(HN_KINDS)}")
    negative = _labels(meta) == 0
    if kind == "same_name_diff_addr":
        same = (_col(X, "core_eq") == 1) if "core_eq" in X.columns else (
            _col(X, "core_ratio") >= 1.0)
        hit = same & (_col(X, "ad_jaccard") < _f32(SAME_NAME_ADDR_MAX))
    elif kind == "same_addr_diff_name":
        hit = ((_col(X, "ad_token_set") >= _f32(SAME_ADDR_MIN))
               & (_col(X, "core_token_set") < _f32(SAME_ADDR_NAME_MAX)))
    elif kind == "top_cosine_nonmatch":
        hit = _col(X, "ctx_rank_name") == 1
    else:
        hit = singleton_mask(meta) if singleton is None else np.asarray(singleton, dtype=bool)
    return negative & hit


def is_easy_negative(X: pd.DataFrame, y: np.ndarray, singleton: np.ndarray) -> np.ndarray:
    """Label-0 rows with ``sim_name_char < 0.5`` and ``ad_jaccard < 0.1``, never a singleton's.

    NaN compares False, so a pair a char-gram pass did not propose (no ``sim_name_char``) or
    with an empty address is never easy: only pairs both similarities call distant are.
    """
    far = ((_col(X, "sim_name_char") < _f32(EASY_NAME_MAX))
           & (_col(X, "ad_jaccard") < _f32(EASY_ADDR_MAX)))
    return (np.asarray(y) == 0) & far & ~np.asarray(singleton, dtype=bool)


def easy_keep(meta: pd.DataFrame, share: float, seed: int = SAMPLE_SEED) -> np.ndarray:
    """Per row: kept by the ``share`` hash sample of ``s1 id | pool id`` (order-free)."""
    if share >= 1.0:
        return np.ones(len(meta), dtype=bool)
    keys = meta[C.S1_ID].astype(str) + "|" + meta[C.ENTITY_ID].astype(str)
    return hash_unit(keys.reset_index(drop=True), seed) < share


def _mine(scored: pd.DataFrame, keep: np.ndarray, label: int) -> pd.DataFrame:
    """Rows of ``scored`` where ``keep``, index kept, plus ``label`` int8."""
    return scored[keep].assign(**{LABEL: np.int8(label)})


def mine_false_positives(scored: pd.DataFrame, truth_pairs: pd.DataFrame,
                         threshold: float = 0.5) -> pd.DataFrame:
    """Rows of ``scored`` (SCORED_COLUMNS) with ``prob >= threshold`` that are not true pairs,
    ``label`` 0, index kept so the caller can select the matching feature rows."""
    true = pair_in(scored[[C.S1_ID, C.ENTITY_ID]], truth_pairs)
    return _mine(scored, (scored["prob"].to_numpy() >= threshold) & ~true, 0)


def mine_near_misses(scored: pd.DataFrame, truth_pairs: pd.DataFrame,
                     threshold: float = 0.5) -> pd.DataFrame:
    """The mirror of ``mine_false_positives``: true pairs with ``prob < threshold``, label 1."""
    true = pair_in(scored[[C.S1_ID, C.ENTITY_ID]], truth_pairs)
    return _mine(scored, (scored["prob"].to_numpy() < threshold) & true, 1)


def fit_weights(meta: pd.DataFrame, X: pd.DataFrame, cfg: HardNegConfig,
                truth: pd.DataFrame | None = None, fp: np.ndarray | None = None,
                fn: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """``(weight, hn_counts)``: one float64 weight per fit row and the 09 §6 counts.

    Base weight 1; easy negatives outside the ``easy_neg_keep`` hash sample 0 (dropped);
    rows of ``cfg.rule_kinds`` at least ``rule_weight``; singleton candidates at least
    ``singleton_weight`` (09 §4: never dropped). Round 2 then adds ``fp_weight - 1`` to each
    row of ``fp`` and ``near_miss_weight - 1`` to each of ``fn`` (bool masks), as the extra
    copies of 09 §7 are added to the round-1 set: a dropped easy false positive comes back
    at ``fp_weight - 1``. Deterministic: depends on ids, features, labels and ``cfg.seed``.
    """
    n = len(meta)
    if len(X) != n:
        raise ValueError(f"meta has {n} rows, X has {len(X)}")
    y = _labels(meta)
    single = singleton_mask(meta, truth)
    weight = np.ones(n, dtype=np.float64)
    counts: dict = {"rows": n, "positives": int(y.sum()),
                    "singleton_decoys": int((single & (y == 0)).sum())}
    if cfg.easy_neg_keep < 1.0:
        easy = is_easy_negative(X, y, single)
        drop = easy & ~easy_keep(meta, cfg.easy_neg_keep, cfg.seed)
        weight[drop] = 0.0
        counts.update(easy=int(easy.sum()), easy_dropped=int(drop.sum()))
    for kind in cfg.rule_kinds:
        hit = mine_by_rule(meta, X, kind, single)
        weight[hit] = np.maximum(weight[hit], cfg.rule_weight)
        counts[kind] = int(hit.sum())
    if cfg.singleton_weight > 1.0:
        weight[single] = np.maximum(weight[single], cfg.singleton_weight)
    for name, mask, extra in (("fps", fp, cfg.fp_weight - 1.0),
                              ("fns", fn, cfg.near_miss_weight - 1.0)):
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != (n,):
                raise ValueError(f"{name} mask has shape {mask.shape}, expected ({n},)")
            weight[mask] += extra
            counts[name] = int(mask.sum())
    counts.update(kept_rows=int((weight > 0).sum()), weight_sum=float(weight.sum()))
    return weight, counts


def weight_fn(cfg: HardNegConfig, truth: pd.DataFrame | None = None,
              fp: np.ndarray | None = None, fn: np.ndarray | None = None,
              counts: dict | None = None):
    """``fit_weights`` as ``evaluate_params``'s ``weight_fn(meta, X)``; ``counts`` (a dict)
    receives ``hn_counts`` when it runs. ``truth`` = ``snap.truth("fit")``."""
    def fn_(meta: pd.DataFrame, X: pd.DataFrame) -> np.ndarray:
        """Weights of the snapshot's fit rows."""
        weight, c = fit_weights(meta, X, cfg, truth, fp, fn)
        if counts is not None:
            counts.update(c)
        return weight
    return fn_


def _tune_f_beta(snap: Snapshot, model: Matcher | SeedEnsemble, grid: Grid,
                 batch_rows: int) -> float:
    """Best macro F0.5 of ``decision.tune`` on the snapshot's tune side (09 §7 ``tune_f05``)."""
    from .snapshot import score_side
    scored, _ = score_side(snap, "tune", model, batch_rows)
    _, table = tune(scored, snap.s1("tune")[C.ENTITY_ID], snap.truth("tune"), grid)
    return float(table["f_beta"].max())


def round2(snap: Snapshot, params: MatcherParams, cfg: HardNegConfig, *,
           grid: Grid | None = None, seeds: Sequence[int] | None = None,
           batch_rows: int | None = None) -> tuple[Matcher | SeedEnsemble, dict]:
    """09 §7 on a snapshot: ``(chosen model, info)``; evaluate it with
    ``evaluate_params(snap, model=model, grid=grid)``.

    Round 1 fits with ``cfg``'s base weights (easy cap, rules, singletons). Its in-sample
    probabilities on the fit rows give the false positives (label 0, ``prob >=
    round2_threshold``) and, with ``keep_near_misses``, the near misses (label 1, below it);
    round 2 refits from scratch with those rows weighted up (early stopping re-derived on the
    same stop rows). Both are scored on tune with the rule re-tuned; round 2 is kept only if
    its tune F0.5 is at least round 1's + ``ROUND2_MARGIN``. ``info``: ``round2_kept``,
    ``tune_f_beta_round1`` / ``_round2``, ``hn_counts`` of the kept model (round 2's counts
    carry ``fps`` / ``fns``) and both fit timings. Val is not read.
    """
    from .snapshot import BATCH_ROWS, fit_snapshot, score_side
    batch_rows = BATCH_ROWS if batch_rows is None else batch_rows
    grid = DEFAULT_GRID if grid is None else grid
    truth = snap.truth("fit")
    counts1: dict = {}
    m1, t1 = fit_snapshot(snap, params, weight_fn=weight_fn(cfg, truth, counts=counts1),
                          seeds=seeds, batch_rows=batch_rows)
    scored, y = score_side(snap, "fit", m1, batch_rows)   # in sample, 09 §5.2
    prob = scored["prob"].to_numpy()
    fp = (y == 0) & (prob >= cfg.round2_threshold)
    fn = ((y == 1) & (prob < cfg.round2_threshold)) if cfg.keep_near_misses else None
    del scored, prob
    counts2: dict = {}
    m2, t2 = fit_snapshot(snap, params, weight_fn=weight_fn(cfg, truth, fp, fn, counts2),
                          seeds=seeds, batch_rows=batch_rows)
    f1 = _tune_f_beta(snap, m1, grid, batch_rows)
    f2 = _tune_f_beta(snap, m2, grid, batch_rows)
    kept = f2 >= f1 + ROUND2_MARGIN
    info = {"round2_kept": bool(kept), "tune_f_beta_round1": f1, "tune_f_beta_round2": f2,
            "hn_counts": counts2 if kept else counts1,
            "hn_counts_round2": counts2, "fit_round1": t1, "fit_round2": t2}
    return (m2 if kept else m1), info
