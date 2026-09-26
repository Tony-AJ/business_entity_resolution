"""Decision layer (plan group E, 10_DECISION_LAYER): scored candidate pairs -> match sets.

The matcher gives every candidate pair a probability; ``decide`` turns the candidates of each
Source 1 entity into a set (zero, one or many pool ids) under a ``DecisionRule``, in order:

    1. pool-side 1-to-1: a pool record stays only with the S1 entity scoring it highest
    2. thresholds per S1: prob >= tau_abs, prob >= tau_rel * p_max, p_max >= tau_single
    3. cap: at most max_matches pairs per S1, by prob rank

Tie-breaks are fixed for the module: prob descending, then ``source1_entity_id``, then
``entity_id`` ascending, stable. ``prob`` is cast to float64 once, and ``decide`` and ``tune``
share the same sorted arrays (``_rank``) and comparisons, so a rule keeps exactly the pairs it
was scored on. ``tune`` scores every rule of a ``Grid`` by macro F0.5 on the tune split from
count arrays (``evaluate.entity_f05_from_counts`` and an exact ``math.fsum``, bitwise equal to
``evaluate.macro_f05_from_counts``), then refines ``tau_abs`` around the best rule.
"""
from __future__ import annotations

import itertools
import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C
from .data import isin
from .evaluate import entity_f05_from_counts, positions
from .trainset import label_pairs

SCORED_COLUMNS = [C.S1_ID, C.ENTITY_ID, "prob"]
MATCH_COLUMNS = [C.S1_ID, C.ENTITY_ID]
RULE_FIELDS = ["tau_abs", "tau_rel", "tau_single", "max_matches", "one_to_one"]
TABLE_COLUMNS = [*RULE_FIELDS, "f_beta", "n_pred", "pair_precision", "pair_recall",
                 "match_rate", "stage"]
DECIMALS = 10  # grid values are rounded: 0.30 + 3 * 0.02 becomes the float 0.36, no noise


@dataclass(frozen=True)
class DecisionRule:
    """How the scored candidates of a Source 1 entity become its match set (10 §2)."""

    tau_abs: float = 0.5      # absolute floor: keep a pair only if prob >= tau_abs
    tau_rel: float = 0.0      # relative floor: prob >= tau_rel * p_max of its S1 entity
    tau_single: float = 0.5   # singleton floor: nothing is kept unless p_max >= tau_single
    max_matches: int = 11     # cap per S1 entity (train maximum is 11 matches)
    one_to_one: bool = True   # a pool record is kept only for its highest-prob S1 entity


@dataclass(frozen=True)
class Grid:
    """The rules ``tune`` tries: the product of the axes, then a finer ``tau_abs`` sweep."""

    tau_abs: tuple[float, float, float] = (0.30, 0.90, 0.02)   # start, stop inclusive, step
    tau_rel: tuple[float, ...] = (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
    single_delta: tuple[float, ...] = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30)  # tau_single - tau_abs
    max_matches: tuple[int, ...] = (4, 6, 11)
    one_to_one: tuple[bool, ...] = (True,)
    refine_span: float = 0.03     # refine stage: best tau_abs ± refine_span ...
    refine_step: float = 0.005    # ... in refine_step steps, the other fields as the best rule
    tie_tol: float = 1e-4         # rules within tie_tol of the best F0.5 are ties (see _pick)

    def __post_init__(self) -> None:
        """Reject grids that cannot be enumerated or that only create ties."""
        start, stop, step = self.tau_abs
        if not step > 0 or stop < start:
            raise ValueError(f"tau_abs must be (start, stop >= start, step > 0), "
                             f"got {self.tau_abs}")
        if any(d < 0 for d in self.single_delta):
            raise ValueError(f"single_delta must be >= 0, got {self.single_delta}: a singleton "
                             "floor below tau_abs is a no-op that only creates ties")
        if not (self.tau_rel and self.single_delta and self.max_matches and self.one_to_one):
            raise ValueError("every grid axis needs at least one value")
        if not self.refine_step > 0 or self.refine_span < 0 or self.tie_tol < 0:
            raise ValueError("refine_step must be > 0; refine_span and tie_tol >= 0")

    def tau_abs_values(self) -> list[float]:
        """``start, start + step, ..., stop`` (stop inclusive): 31 values by default."""
        start, stop, step = self.tau_abs
        count = math.floor((stop - start) / step + 1e-9) + 1   # 1e-9: 30.000000000000004 -> 30
        return [round(start + i * step, DECIMALS) for i in range(count)]

    def rules(self, one_to_one: bool) -> list[DecisionRule]:
        """Every grid rule for one ``one_to_one`` flag: 31 · 7 · 6 · 3 = 3,906 by default."""
        return [DecisionRule(ta, tr, round(ta + d, DECIMALS), mm, one_to_one)
                for ta, tr, d, mm in itertools.product(self.tau_abs_values(), self.tau_rel,
                                                       self.single_delta, self.max_matches)]

    def refine_rules(self, best: DecisionRule) -> list[DecisionRule]:
        """``best`` with tau_abs ± refine_span in refine_step steps (13 by default), keeping
        its ``tau_single - tau_abs`` gap and every other field."""
        k = round(self.refine_span / self.refine_step)
        gap = best.tau_single - best.tau_abs
        out = []
        for j in range(-k, k + 1):
            ta = round(best.tau_abs + j * self.refine_step, DECIMALS)
            out.append(DecisionRule(ta, best.tau_rel, round(ta + gap, DECIMALS),
                                    best.max_matches, best.one_to_one))
        return out


DEFAULT_GRID = Grid()  # frozen, so one shared default instance is safe


# --------------------------------------------------------------- ranking ----
@dataclass(frozen=True)
class _Ranked:
    """Scored pairs after the sort and the 1-to-1 step, as aligned arrays in decision order."""

    prob: np.ndarray       # float64, descending (NaN last, never kept)
    p_max: np.ndarray      # float64: best surviving prob of the row's S1 entity
    rank: np.ndarray       # 0 for the entity's best surviving row, 1 for the next, ...
    s1: np.ndarray         # S1 code per row; codes follow id order
    pool: np.ndarray       # pool code per row; codes follow id order
    rows: np.ndarray       # position of each surviving row in the input frame
    s1_ids: pd.Index       # sorted distinct S1 ids: code -> id
    pool_ids: pd.Index     # sorted distinct pool ids: code -> id


def _check_scored(scored: pd.DataFrame) -> None:
    """The schema assertion of 02 §3, as an exception that survives ``python -O``."""
    if list(scored.columns[:3]) != SCORED_COLUMNS:
        raise ValueError(f"scored must start with {SCORED_COLUMNS}, got {list(scored.columns)}")


def _rank(scored: pd.DataFrame, one_to_one: bool) -> _Ranked:
    """Sort, 1-to-1, p_max and rank: everything about a rule that is not a threshold.

    Ids become codes of ``pd.factorize(sort=True)``, so ordering codes orders ids and the
    whole ranking is integer and float sorts (seconds on 15M rows, no string group-bys).
    """
    _check_scored(scored)
    prob = scored["prob"].to_numpy(dtype=np.float64, na_value=np.nan)  # cast once (10 §2)
    s1, s1_ids = pd.factorize(scored[C.S1_ID], sort=True, use_na_sentinel=False)
    pool, pool_ids = pd.factorize(scored[C.ENTITY_ID], sort=True, use_na_sentinel=False)
    key = s1 * max(len(pool_ids), 1) + pool       # one int64 per pair, ordered like the ids
    by_key = np.argsort(key)
    if (np.diff(key[by_key]) == 0).any():
        raise ValueError("scored holds duplicate (source1_entity_id, entity_id) pairs")
    order = by_key[np.argsort(-prob[by_key], kind="stable")]  # prob desc, S1 id, pool id asc
    del key, by_key
    if one_to_one:   # step 1: a pool record stays with its first row: highest prob, then S1 id
        order = order[~pd.Series(pool[order]).duplicated().to_numpy()]
    prob, s1 = prob[order], s1[order]
    first = ~pd.Series(s1).duplicated().to_numpy()   # an entity's first row holds its p_max
    best = np.full(len(s1_ids), np.nan)
    best[s1[first]] = prob[first]
    rank = pd.Series(s1).groupby(s1, sort=False).cumcount().to_numpy()  # rows are prob desc
    return _Ranked(prob, best[s1], rank, s1, pool[order], order, s1_ids, pool_ids)


def _keep(prob: np.ndarray, p_max: np.ndarray, rank: np.ndarray,
          rule: DecisionRule) -> np.ndarray:
    """Steps 2-3 as one row mask. Every condition is monotone in ``prob`` inside an entity,
    so the kept rows are a prefix of its ranked list and the cap is ``rank < max_matches``."""
    return ((prob >= rule.tau_abs) & (prob >= rule.tau_rel * p_max)
            & (p_max >= rule.tau_single) & (rank < rule.max_matches))


def decide(scored: pd.DataFrame, rule: DecisionRule) -> pd.DataFrame:
    """The match set of every S1 entity under ``rule`` (10 §2-§3).

    ``scored`` starts with ``SCORED_COLUMNS``; duplicate pairs raise ``ValueError``. Returns
    ``MATCH_COLUMNS``, one row per kept pair, sorted by S1 id then pool id, fresh index;
    entities with nothing kept have no rows (``evaluate.pairs_to_lists`` and
    ``submission.write_pairs`` add their empty lists).
    """
    r = _rank(scored, rule.one_to_one)
    keep = _keep(r.prob, r.p_max, r.rank, rule)
    s1, pool = r.s1[keep], r.pool[keep]
    order = np.lexsort((pool, s1))                 # codes follow id order: sorted by ids
    return pd.DataFrame({C.S1_ID: r.s1_ids.take(s1[order]),
                         C.ENTITY_ID: r.pool_ids.take(pool[order])})


def one_to_one_filter(scored: pd.DataFrame) -> pd.DataFrame:
    """The rows of ``scored`` that survive the pool-side 1-to-1 (step 1 of ``decide``).

    A pool record stays only with its highest-prob S1 entity, ties broken exactly as in
    ``decide``. Filtering a whole partition and then running ``decide`` or ``tune`` on a
    subset of its entities gives that subset the matches ``decide`` gives it on the whole
    partition: steps 2-3 only read an entity's own surviving rows. Row order is kept.
    """
    return scored.iloc[np.sort(_rank(scored, one_to_one=True).rows)]


# ---------------------------------------------------------------- tuning ----
@dataclass(frozen=True)
class _Arrays:
    """``_Ranked`` of one ``one_to_one`` flag, S1 codes mapped to positions in ``s1_ids``."""

    neg_prob: np.ndarray   # -prob, ascending: prob >= t on searchsorted(neg_prob, -t, "right") rows
    prob: np.ndarray
    p_max: np.ndarray
    rank: np.ndarray
    s1: np.ndarray         # position of the row's entity in s1_ids
    is_true: np.ndarray    # the row is a truth pair
    pmax_ent: np.ndarray   # p_max per entity of s1_ids, -inf without candidates
    scorer: _Scorer


SPLIT = 134217729.0  # 2**27 + 1, Veltkamp's constant: splits a double into two 26-bit halves


class _Scorer:
    """Macro F0.5 and pair totals of one rule from a histogram of entity count triples.

    An entity's F0.5 depends only on its (n_true, n_pred, tp) triple, encoded as one integer,
    so a rule costs one ``np.bincount`` over the entities plus float work on the few distinct
    triples, instead of the formula and ``math.fsum`` on every entity (~10x faster on 100k
    entities). The mean stays bitwise equal to ``evaluate.macro_f05_from_counts``: the values
    come from ``entity_f05_from_counts``, and a value v held by c entities enters ``math.fsum``
    as hi·c + lo·c, where v = hi + lo is its Veltkamp split (26 significant bits each) and
    c < 2**27, so both products are exact and fsum rounds exactly the per-entity sum, once.
    """

    def __init__(self, n_true: np.ndarray, width: int) -> None:
        """``n_true`` per entity; ``width`` exceeds every n_pred (so every tp) to be encoded."""
        self.n, self.width, self.square = len(n_true), width, width * width
        self.empty = n_true.astype(np.int64) * self.square      # code of an empty prediction
        self.dense = (int(n_true.max(initial=0)) + 1) * self.square <= 16 * self.n + (1 << 20)

    def code(self, tp: np.ndarray, n_pred: np.ndarray) -> np.ndarray:
        """One integer per entity for its (n_true, n_pred, tp) triple."""
        return self.empty + n_pred * self.width + tp

    def __call__(self, code: np.ndarray) -> tuple[float, int, int, int]:
        """(macro F0.5, tp, n_pred, entities predicting something), all entities summed."""
        if self.dense:                              # small code space: a plain histogram
            counts = np.bincount(code)
            triple = np.flatnonzero(counts)
            count = counts[triple]
        else:                                       # a huge candidate list somewhere: sort
            code = np.sort(code)
            start = np.flatnonzero(np.r_[True, code[1:] != code[:-1]])
            triple, count = code[start], np.diff(np.r_[start, len(code)])
        n_true, rest = np.divmod(triple, self.square)
        n_pred, tp = np.divmod(rest, self.width)
        f = entity_f05_from_counts(tp, n_pred, n_true)
        t = SPLIT * f
        hi = t - (t - f)                            # f == hi + (f - hi) exactly
        c = count.astype(np.float64)
        f_beta = math.fsum(np.concatenate([hi * c, (f - hi) * c]).tolist()) / self.n
        return (f_beta, int((tp * count).sum()), int((n_pred * count).sum()),
                int(count[n_pred > 0].sum()))


class _Tuning:
    """What every rule evaluation shares, prepared once per ``tune`` / ``evaluate_rules``."""

    def __init__(self, scored: pd.DataFrame, s1_ids: Iterable[str],
                 truth_pairs: pd.DataFrame) -> None:
        """Check the inputs; count every truth pair of ``s1_ids``; label the scored rows."""
        _check_scored(scored)
        self.ids = pd.Index(s1_ids)
        self.n = len(self.ids)
        if self.n == 0:
            raise ValueError("s1_ids is empty")
        if not self.ids.is_unique:
            raise ValueError("s1_ids holds duplicate ids")
        truth = truth_pairs[MATCH_COLUMNS]
        truth = truth[isin(truth[C.S1_ID], self.ids)].drop_duplicates()  # misses included
        self.n_true = np.bincount(positions(truth[C.S1_ID], self.ids), minlength=self.n)
        self.is_true = label_pairs(scored[MATCH_COLUMNS], truth)["label"].to_numpy() == 1
        self.scored = scored
        self._arrays: dict[bool, _Arrays] = {}

    def arrays(self, one_to_one: bool) -> _Arrays:
        """Ranked arrays for one flag, computed on first use."""
        if one_to_one not in self._arrays:
            r = _rank(self.scored, one_to_one)
            code_pos = positions(r.s1_ids, self.ids)      # S1 code -> position in s1_ids
            if (code_pos < 0).any():
                raise ValueError(f"scored holds {int((code_pos < 0).sum()):,} S1 ids that are "
                                 "not in s1_ids")
            s1 = code_pos[r.s1]
            pmax_ent = np.full(self.n, -np.inf)
            pmax_ent[s1] = r.p_max                        # equal for every row of an entity
            width = int(np.bincount(s1, minlength=self.n).max()) + 1   # n_pred <= rows
            self._arrays[one_to_one] = _Arrays(-r.prob, r.prob, r.p_max, r.rank, s1,
                                               self.is_true[r.rows], pmax_ent,
                                               _Scorer(self.n_true, width))
        return self._arrays[one_to_one]


def _evaluate(data: _Tuning, rules: Sequence[DecisionRule], stage: str) -> pd.DataFrame:
    """Score ``rules`` on counts; one ``TABLE_COLUMNS`` row per rule, in the input order.

    Rules sharing (one_to_one, tau_rel, max_matches) share one row mask; ``prob >= tau_abs``
    is a prefix of the prob-sorted arrays, so their counts grow segment by segment over the
    ``tau_abs`` values; ``p_max >= tau_single`` is a switch per entity. Rows with prob below
    every ``tau_abs`` are never touched (they can never be kept, and p_max and rank of the
    other rows were computed before). Counts equal ``bincount`` of ``_keep`` on the same
    arrays, so every row scores exactly what ``decide`` would keep; ``_Scorer`` turns them
    into macro F0.5 with one histogram per rule. Cost: O(rows) per rule group plus O(entities)
    per rule; the default grid takes ~3 s on 100k entities and 3M pairs, preparation included.
    """
    groups: dict[tuple, dict[float, list[tuple[float, int]]]] = defaultdict(
        lambda: defaultdict(list))
    for i, rule in enumerate(rules):
        key = (rule.one_to_one, rule.tau_rel, rule.max_matches)
        groups[key][rule.tau_abs].append((rule.tau_single, i))
    n, n_true = data.n, data.n_true
    n_true_sum = int(n_true.sum())
    rows: list[tuple] = [()] * len(rules)
    for (one_to_one, tau_rel, max_matches), by_abs in groups.items():
        a = data.arrays(one_to_one)
        cut = np.searchsorted(a.neg_prob, -min(by_abs), side="right")  # prob >= lowest tau_abs
        base = (a.prob[:cut] >= tau_rel * a.p_max[:cut]) & (a.rank[:cut] < max_matches)
        hit = base & a.is_true[:cut]
        n_pred_c, tp_c, done = np.zeros(n, np.int64), np.zeros(n, np.int64), 0
        for tau_abs in sorted(by_abs, reverse=True):          # the prefix only grows
            end = np.searchsorted(a.neg_prob, -tau_abs, side="right")
            n_pred_c += np.bincount(a.s1[done:end][base[done:end]], minlength=n)
            tp_c += np.bincount(a.s1[done:end][hit[done:end]], minlength=n)
            done = end
            code = a.scorer.code(tp_c, n_pred_c)
            for tau_single, i in by_abs[tau_abs]:
                # singleton floor: an entity-wide switch between its counts and nothing
                on = a.pmax_ent >= tau_single
                f_beta, tp_s, pred_s, matched = a.scorer(np.where(on, code, a.scorer.empty))
                rule = rules[i]
                rows[i] = (rule.tau_abs, rule.tau_rel, rule.tau_single, rule.max_matches,
                           rule.one_to_one, f_beta, pred_s, tp_s / max(pred_s, 1),
                           tp_s / n_true_sum if n_true_sum else float("nan"), matched / n,
                           stage)
    return pd.DataFrame(rows, columns=TABLE_COLUMNS)


def evaluate_rules(scored: pd.DataFrame, s1_ids: Iterable[str], truth_pairs: pd.DataFrame,
                   rules: Iterable[DecisionRule], stage: str = "grid") -> pd.DataFrame:
    """Macro F0.5 and pair statistics of each rule on a scored sample (10 §4).

    Scored over ALL ``s1_ids``: ``n_true`` counts every truth pair of those entities, those
    blocking missed included, and entities without candidates predict nothing, exactly as
    ``metrics.macro_fbeta`` scores ``pairs_to_lists(decide(scored, rule), s1_ids)``. Every S1
    id of ``scored`` must be in ``s1_ids``. Columns ``TABLE_COLUMNS``, one row per rule, in order.
    """
    return _evaluate(_Tuning(scored, s1_ids, truth_pairs), list(rules), stage)


def _pick(table: pd.DataFrame, tie_tol: float) -> DecisionRule:
    """The best rule of ``table``. Rules within ``tie_tol`` of the best F0.5 are ties, and ties
    go to the most conservative rule: highest tau_abs, then tau_rel, then tau_single, then the
    lowest max_matches (then one_to_one on), because test holds more decoys than tune (10 §9)."""
    near = table[table["f_beta"] >= table["f_beta"].max() - tie_tol]
    best = near.sort_values(RULE_FIELDS, ascending=[False, False, False, True, False],
                            kind="stable").iloc[0]
    return DecisionRule(float(best["tau_abs"]), float(best["tau_rel"]), float(best["tau_single"]),
                        int(best["max_matches"]), bool(best["one_to_one"]))


def tune(scored: pd.DataFrame, s1_ids: Iterable[str], truth_pairs: pd.DataFrame,
         grid: Grid = DEFAULT_GRID) -> tuple[DecisionRule, pd.DataFrame]:
    """Best ``DecisionRule`` on the tune sample by macro F0.5, plus every rule tried.

    Stage "grid" scores ``grid.rules(flag)`` for each ``one_to_one`` flag (3,906 by default);
    stage "refine" sweeps tau_abs around the best of them (13 more rules); the answer is
    ``_pick`` over both. Use the tune side of ``trainset.inner_split`` only, never val (10 §5).
    """
    data = _Tuning(scored, s1_ids, truth_pairs)
    table = _evaluate(data, [r for flag in grid.one_to_one for r in grid.rules(flag)], "grid")
    refine = _evaluate(data, grid.refine_rules(_pick(table, grid.tie_tol)), "refine")
    table = pd.concat([table, refine], ignore_index=True)
    return _pick(table, grid.tie_tol), table
