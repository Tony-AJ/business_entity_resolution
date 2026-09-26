"""Normalisation of business names and addresses (plan group B, 05_NORMALISATION_STRATEGY).

``normalise_records`` turns a source frame (``entity_id``, ``business_name``,
``business_address``, ``country``) into the columns every later stage reads
(``NORM_COLUMNS``, 02 §4.1). The rules undo the noise measured on true pairs (01 §3):
accents, case and punctuation; legal-form drift (``Pvt. Ltd.`` / ``Private Limited``);
Indic-script names and states (transliterated with ``anyascii``, ISC licence); leet and
domain forms (``gl0ba``, ``allh0spitalityproducts.com``); honorific prefixes (``Mr``,
``Smt``); street-type abbreviations (``St``/``Street``/``Saint``, ``R.``/``Rue``); state
names, codes and native-script forms; old/new city names.

Rules v4 (French forms measured on the test pool; they change no US or India record):
the pool's ``N°`` / ``Nº`` number marker leaves addresses before folding, and ``Compagnie``
is a legal form like ``Cie``.

Rules v5 (they change US and India records too; measured on train true pairs): the pool's
``&`` written ``et`` (France) or as a standalone ``+`` (every country) reads as ``and``, and
``Frs`` as ``Freres``.

Everything is vectorised on Arrow strings. Regexes run in pyarrow (RE2 syntax, so no
look-arounds); token maps run once per *distinct* token through a dictionary encoding;
the only Python loops are ``anyascii`` on rows that still hold non-ASCII characters and
the per-distinct-token maps. Country never selects a code path: the maps simply hold
tokens from every country, France included.
"""
from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
from anyascii import anyascii

from . import config as C

NORM_COLUMNS = [C.ENTITY_ID, C.COUNTRY, "non_latin", "name_norm", "name_core", "legal_form",
                "name_first", "name_sorted", "name_squash", "addr_norm", "addr_nums", "postcode",
                "region", "addr_last", "addr_tokens", "name_addr"]
EXTRA_COLUMNS = ["domain_form", "addr_non_latin"]  # added after NORM_COLUMNS (05 §11)
# Bump when a rule changes the output: the pipeline's normalisation cache key includes it.
# 4: French number marker and "compagnie" (US and India output byte-identical to 3).
# 5: "et" / "+" -> "and" and "frs" -> "freres" in names (US and India records change too).
RULES_VERSION = 5

# Letters of non-Latin scripts (Greek to Indic to CJK): the rows anyascii must transliterate.
NON_LATIN_RE = r"[\x{0370}-\x{1DBF}\x{2C00}-\x{2DFF}\x{3000}-\x{D7FF}]"
# The French number marker "N° 32" / "Nº 32" (v4): written by the pool only (6.5% of French
# pool addresses, never in S1, US or India), so it is dropped before it becomes an "n" token.
NUMBER_MARKER_RE = r"(?i)\bn\s*[°º]"

from .token_maps import (  # noqa: E402  (re-exported: tests and features import them from here)
    ADDRESS_TOKENS,
    HONORIFIC_RE,
    LEET,
    LEGAL_FORMS,
    NAME_TOKENS,
    REGION_ABBREV,
    TRANSLIT_LEGAL,
)


@dataclass(frozen=True)
class NormaliseConfig:
    """Switches for the normalisation rules (05 §2); defaults are the V1 pipeline."""

    transliterate: bool = True      # anyascii on non-ASCII rows (R0)
    strip_legal: bool = True        # legal forms out of name_core (R5)
    expand_abbrev: bool = True      # address token map (R7)
    region_map: Mapping[str, str] | None = None  # None -> static REGION_ABBREV
    chunk_rows: int = 1_000_000     # rows normalised at a time (bounds peak memory)
    # learned transliterated-token -> Latin-token map for non-Latin names (fit_token_map);
    # applied by apply_token_map after the cached static normalisation, never hashed in
    learn_token_map: bool = True
    token_map_min_count: int = 3
    token_map_min_share: float = 0.5


DEFAULT = NormaliseConfig()


# --------------------------------------------------------------- primitives ----
def _arrow(s: pd.Series) -> pa.Array:
    """The Series as one non-null Arrow string array ("" for missing).

    A Series read from Parquet or concatenated holds several Arrow chunks; the list and
    dictionary kernels below need a single contiguous array.
    """
    arr = pa.array(s.fillna("").astype("str"), type=pa.string())
    return arr.combine_chunks() if isinstance(arr, pa.ChunkedArray) else arr


def _series(arr: pa.Array | pa.ChunkedArray, index: pd.Index) -> pd.Series:
    """Arrow strings back to a pandas "str" Series on ``index``."""
    out = arr.to_pandas(types_mapper=pd.ArrowDtype)
    out.index = index  # positional: never reindex (a later chunk's labels start past 0)
    return out.astype("str")


def _collapse(arr: pa.Array) -> pa.Array:
    """Collapse whitespace runs to one space and strip both ends."""
    return pc.utf8_trim_whitespace(pc.replace_substring_regex(arr, r"\s+", " "))


def map_tokens(arr: pa.Array, fn: Callable[[str], str]) -> pa.Array:
    """Apply ``fn`` to every whitespace token, calling it once per distinct token.

    Tokens mapped to "" disappear; a token may map to several words. The token list is
    flattened, dictionary-encoded (the ~1-2M distinct tokens of a split are all Python
    ever sees), mapped, and joined back with single spaces.
    """
    lists = pc.utf8_split_whitespace(arr)
    flat = pc.list_flatten(lists)
    if len(flat) == 0:
        return arr
    enc = pc.dictionary_encode(flat)
    uniq = enc.dictionary.to_pylist()
    mapped = pa.array([fn(t) for t in uniq], type=pa.string()).take(enc.indices)
    rebuilt = pa.ListArray.from_arrays(lists.offsets, mapped)
    return _collapse(pc.binary_join(rebuilt, " "))


def _dict_fn(mapping: Mapping[str, str]) -> Callable[[str], str]:
    """Token function for a plain dict map (unknown tokens unchanged)."""
    return lambda t: mapping.get(t, t)


def _leet(token: str) -> str:
    """Fold leet digits to letters in a token holding both (``gl0ba`` -> ``globa``)."""
    if token.isalpha() or token.isdigit():
        return token
    return token.translate(LEET)


def _join_initials_py(text: str) -> str:
    """Join runs of single letters: ``l l c`` -> ``llc``, ``p c`` -> ``pc``."""
    return _INITIALS.sub(lambda m: m.group(0).replace(" ", ""), text)


_INITIALS = re.compile(r"\b[a-z](?: [a-z])+\b")


def fold(s: pd.Series, transliterate: bool = True,
         number_marker: bool = False) -> tuple[pa.Array, np.ndarray]:
    """Lowercase ASCII text plus the non-Latin-script flag (rules R0-R1).

    Rows holding any non-ASCII character go through ``anyascii`` on the raw text, which
    both transliterates Indic scripts (``प्राइवेट`` -> ``praivet``) and strips accents
    (``Léarning`` -> ``Learning``). Stripping combining marks first would delete the
    Indic vowel signs, so it is only the fallback when transliteration is switched off.
    ``number_marker`` (addresses, v4) drops the whole ``N°`` / ``Nº`` marker first.
    """
    arr = _arrow(s)
    non_latin = pc.match_substring_regex(arr, NON_LATIN_RE).to_numpy(zero_copy_only=False)
    if number_marker:  # "N° 32 R Pierre" -> " 32 R Pierre", as S1 writes it
        arr = pc.replace_substring_regex(arr, NUMBER_MARKER_RE, " ")
    arr = pc.replace_substring_regex(arr, "[°º]", " ")  # any other ° or º separates tokens
    if transliterate:
        rest = pc.match_substring_regex(arr, r"[^\x00-\x7F]").to_numpy(zero_copy_only=False)
        if rest.any():
            vals = np.asarray(arr.to_pylist(), dtype=object)
            vals[rest] = [anyascii(v) for v in vals[rest]]
            arr = pa.array(vals.tolist(), type=pa.string())
    else:
        arr = pc.replace_substring_regex(pc.utf8_normalize(arr, "NFKD"), r"\p{Mn}", "")
    return pc.utf8_lower(arr), non_latin


def basic_norm(s: pd.Series) -> pd.Series:
    """Lowercase ASCII, ``&`` -> ``and``, punctuation -> space, collapsed (R1-R4)."""
    arr, _ = fold(s)
    return _series(_punct(arr), s.index)


def _punct(arr: pa.Array) -> pa.Array:
    """Apostrophes vanish, ``&`` becomes ``and``, other punctuation becomes a space."""
    arr = pc.replace_substring_regex(arr, r"['`]", "")          # orelee's -> orelees
    arr = pc.replace_substring_regex(arr, "&", " and ")
    return _collapse(pc.replace_substring_regex(arr, r"[^a-z0-9+]+", " "))


def _join_initials(arr: pa.Array) -> pa.Array:
    """Join runs of single-letter tokens on the rows that have one (Python on a subset)."""
    has = pc.match_substring_regex(arr, r"\b[a-z] [a-z]\b").to_numpy(zero_copy_only=False)
    if not has.any():
        return arr
    vals = np.asarray(arr.to_pylist(), dtype=object)
    vals[has] = [_join_initials_py(v) for v in vals[has]]
    return pa.array(vals.tolist(), type=pa.string())


def _name_tokens(arr: pa.Array) -> pa.Array:
    """Name spellings of one word (v5): the French "et" and a standalone "+" are the "&" that
    S1 writes (``Aero et Cie`` / ``ARC + CIE`` -> ``and``), ``frs`` is ``freres``."""
    arr = pc.replace_substring_regex(arr, r" et( |$)", r" and\1")  # after a word: "ET Ltd" stays
    return map_tokens(arr, _dict_fn(NAME_TOKENS))  # "+" -> "and", "frs" -> "freres"


# ------------------------------------------------------------------- names ----
def normalise_names(names: pd.Series, cfg: NormaliseConfig = DEFAULT,
                    token_map: Mapping[str, str] | None = None) -> pd.DataFrame:
    """Name columns of 02 §4.1 (R0-R6) plus ``domain_form``.

    name_norm    folded, punctuation-free name, dotted initials joined
    name_core    name_norm without legal forms and leading honorifics, leet folded
    legal_form   canonical legal forms in order of appearance ("pvt ltd")
    name_first / name_sorted / name_squash   blocking keys (first token, sorted distinct
                 tokens, letters and digits only)

    ``token_map`` (learned by ``fit_token_map``) rewrites tokens of non-Latin rows.
    """
    arr, non_latin = fold(names, cfg.transliterate)
    raw_domain = pc.match_substring_regex(
        arr, r"^[@#]|[a-z0-9]\.(?:com|net|org|in|co|io|fr|biz|info)\b|^www\.")
    arr = pc.replace_substring_regex(arr, r"^www\.", "")
    arr = pc.replace_substring_regex(
        arr, r"([a-z0-9])\.(?:com|net|org|co\.in|in|co|io|fr|biz|info|us)\b", r"\1")
    norm = _join_initials(_punct(arr))
    # a long single glued token ending in "com" is a domain written without its dot
    # ("orthopedichealthcom"); short words keep it ("intercom", "telecom")
    norm = pc.replace_substring_regex(norm, r"^([a-z0-9]{7,})com$", r"\1")
    norm = _name_tokens(norm)  # v5: "et" / "+" -> "and", "frs" -> "freres"
    out = _name_columns(norm, non_latin, cfg, token_map)
    out.insert(1, "domain_form", raw_domain.to_numpy(zero_copy_only=False))
    out.index = names.index
    return out


def _name_columns(norm: pa.Array, non_latin: np.ndarray, cfg: NormaliseConfig,
                  token_map: Mapping[str, str] | None) -> pd.DataFrame:
    """Everything derived from ``name_norm``: the learned map, legal forms, keys (R5-R6)."""
    if token_map:  # learned transliteration fixes, on the rows written in a non-Latin script
        norm = _by_script(norm, non_latin, lambda a, m: map_tokens(a, _dict_fn(m)), {}, token_map)
    latin_legal = {**LEGAL_FORMS}
    translit_legal = {**LEGAL_FORMS, **TRANSLIT_LEGAL}
    if cfg.strip_legal:
        core = _by_script(norm, non_latin, lambda a, m: map_tokens(
            a, lambda t, m=m: "" if t in m else _leet(t)), latin_legal, translit_legal)
        legal = _by_script(norm, non_latin, lambda a, m: map_tokens(
            a, lambda t, m=m: m.get(t, "")), latin_legal, translit_legal)
    else:
        core = map_tokens(norm, _leet)
        legal = pa.array([""] * len(norm), type=pa.string())
    core = _collapse(pc.replace_substring_regex(core, HONORIFIC_RE, ""))
    # "Dover & Co" loses "co" and keeps a dangling "and": drop it at either end
    core = _collapse(pc.replace_substring_regex(core, r"^(?:and )+|(?: and)+$|^and$", ""))
    idx = pd.RangeIndex(len(norm))
    core_s = _series(core, idx)
    tokens = core_s.str.split()
    return pd.DataFrame({
        "non_latin": non_latin,
        "name_norm": _series(norm, idx),
        "name_core": core_s,
        "legal_form": _series(legal, idx),
        "name_first": core_s.str.split(" ", n=1).str[0].fillna("").astype("str"),
        # sorted distinct tokens: word swaps and doubled words ("colonial colonial") agree
        "name_sorted": tokens.map(lambda t: " ".join(sorted(set(t)))).astype("str"),
        "name_squash": core_s.str.replace(r"[^a-z0-9]", "", regex=True),
    }, index=idx)


def _by_script(arr: pa.Array, non_latin: np.ndarray, fn, latin_map, translit_map) -> pa.Array:
    """``fn(arr, map)`` with the transliteration-aware map on non-Latin rows only."""
    if not non_latin.any():
        return fn(arr, latin_map)
    out = np.asarray(fn(arr, latin_map).to_pylist(), dtype=object)
    sub = pa.array(np.asarray(arr.to_pylist(), dtype=object)[non_latin].tolist(),
                   type=pa.string())
    out[non_latin] = fn(sub, translit_map).to_pylist()
    return pa.array(out.tolist(), type=pa.string())


# --------------------------------------------------------------- addresses ----
def normalise_addresses(addr: pd.Series, cfg: NormaliseConfig = DEFAULT) -> pd.DataFrame:
    """Address columns of 02 §4.1 (R7-R8) plus ``addr_non_latin``.

    The region is found per comma-separated component (so "Tamil Nadu", "TN" and the
    Tamil-script name all become ``tn`` wherever the component sits) and replaced by its
    code inside ``addr_norm``; then ordinals lose their suffix, digits split from letters
    (``25233b`` -> ``25233 b``), leading zeros go, and street-type tokens are canonicalised.
    """
    regions = dict(cfg.region_map) if cfg.region_map is not None else REGION_ABBREV
    arr, non_latin = fold(addr, cfg.transliterate, number_marker=True)
    arr = pc.replace_substring_regex(arr, r"<null>|\bn/a\b", " ")
    # --- region per comma component
    comps = pc.split_pattern(arr, ",")
    flat = pc.list_flatten(comps)
    parent = pc.list_parent_indices(comps).to_numpy()
    flat_norm = _punct(flat)
    # a component is a region when its words (numbers aside: "New York 14610") are one
    flat_words = _collapse(pc.replace_substring_regex(flat_norm, r"\b\d+\b", " "))
    flat_nums = _collapse(pc.replace_substring_regex(flat_norm, r"\b[a-z+]\w*\b", " "))
    if not regions:  # an empty override: no component is a region
        regions = {"\x00": "\x00"}
    keys = pa.array(list(regions), type=pa.string())
    codes = pa.array(list(regions.values()), type=pa.string())
    hit = pc.index_in(flat_words, value_set=keys)
    is_region = pc.is_valid(hit).to_numpy(zero_copy_only=False)
    code_flat = pc.take(codes, pc.fill_null(hit, 0))
    region_text = pc.binary_join_element_wise(code_flat, flat_nums, " ")
    flat_out = pc.if_else(pa.array(is_region), region_text, flat_norm)
    region = np.full(len(arr), "", dtype=object)
    first = pd.Series(np.flatnonzero(is_region)).groupby(parent[is_region]).first()
    region[first.index.to_numpy()] = pc.take(code_flat, pa.array(first.to_numpy())).to_pylist()
    rebuilt = pa.ListArray.from_arrays(comps.offsets, flat_out)
    text = pc.binary_join(rebuilt, " ")
    # --- tokens
    text = pc.replace_substring_regex(text, r"\b(\d+)(?:st|nd|rd|th)\b", r"\1")  # 8th -> 8
    text = pc.replace_substring_regex(text, r"(\d)([a-z])", r"\1 \2")
    text = pc.replace_substring_regex(text, r"([a-z])(\d)", r"\1 \2")
    text = pc.replace_substring_regex(text, r"\b0+(\d)", r"\1")                  # 0033 -> 33
    text = _collapse(text)
    if cfg.expand_abbrev:
        text = map_tokens(text, _dict_fn(ADDRESS_TOKENS))

    idx = addr.index
    norm = _series(text, idx)
    nums = norm.str.findall(r"\b\d+\b")
    words = norm.str.replace(r"\b\d+\b", " ", regex=True).str.split()
    postcode = norm.str.extract(r"\b(\d{6})\b", expand=False).fillna("")
    zip5 = norm.str.extract(r"\b(\d{5})$", expand=False).fillna("")
    region_s = pd.Series(region, index=idx).astype("str")
    out = pd.DataFrame({
        "addr_non_latin": non_latin,
        "addr_norm": norm,
        "addr_nums": nums.map(" ".join).astype("str"),
        # a 6-digit PIN anywhere, else a 5-digit code that ends the address
        "postcode": postcode.where(postcode != "", zip5),
        "region": region_s,
        "addr_last": words.map(lambda w: " ".join(w[-2:])).astype("str"),
        "addr_tokens": norm.str.count(r"\S+").astype("int16"),
    }, index=idx)
    out["postcode"] = out["postcode"].astype("str")
    return out


# ----------------------------------------------------------------- records ----
def normalise_records(df: pd.DataFrame, cfg: NormaliseConfig = DEFAULT) -> pd.DataFrame:
    """Normalised frame for a source frame: ``NORM_COLUMNS`` then ``EXTRA_COLUMNS``.

    Same row count and order as ``df`` (index reset); every string column non-null.
    Works in chunks of ``cfg.chunk_rows`` rows to bound the transient Arrow memory.
    """
    df = df.reset_index(drop=True)
    parts = []
    for start in range(0, max(len(df), 1), cfg.chunk_rows):
        part = df.iloc[start:start + cfg.chunk_rows]
        names = normalise_names(part[C.NAME], cfg)
        addrs = normalise_addresses(part[C.ADDRESS], cfg)
        out = pd.concat([part[[C.ENTITY_ID, C.COUNTRY]].astype("str"), names, addrs], axis=1)
        name_addr = (out["name_core"] + " " + out["addr_norm"]).str.strip()
        out["name_addr"] = name_addr.astype("str")
        parts.append(out[NORM_COLUMNS + EXTRA_COLUMNS])
    return pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]


# ------------------------------------------------------- learned token map ----
def fit_token_map(truth_pairs: pd.DataFrame, s1n: pd.DataFrame, pooln: pd.DataFrame,
                  min_count: int = 3, min_share: float = 0.5) -> dict[str, str]:
    """Transliterated name token -> Latin token, learned from true pairs (05 §6, B4).

    For every true pair whose pool name was written in a non-Latin script and has as many
    ``name_norm`` tokens as its Source 1 partner, tokens are aligned by position
    (``शक्ति श्याम ट्रेडिंग`` -> ``sakti syam treding`` against ``shakti shyam trading``).
    A pool token is kept when it met the same Latin token at least ``min_count`` times and
    in at least ``min_share`` of its aligned occurrences. Fit on the train fold's pairs only.
    """
    right = pooln.loc[pooln["non_latin"].to_numpy(), [C.ENTITY_ID, "name_norm"]]
    p = truth_pairs[[C.S1_ID, C.ENTITY_ID]].merge(right, on=C.ENTITY_ID)
    left = s1n[[C.ENTITY_ID, "name_norm"]].rename(columns={C.ENTITY_ID: C.S1_ID,
                                                          "name_norm": "l"})
    p = p.merge(left, on=C.S1_ID)
    lt, rt = p["l"].str.split(), p["name_norm"].str.split()
    same = (lt.str.len() == rt.str.len()).to_numpy()
    if not same.any():
        return {}
    df = pd.DataFrame({"r": rt[same].explode().to_numpy(), "l": lt[same].explode().to_numpy()})
    total = df["r"].value_counts()  # every aligned occurrence, identical ones included
    df = df[df["r"] != df["l"]]
    counts = df.value_counts()                           # (r, l) -> n, largest first
    best = counts[~counts.index.get_level_values(0).duplicated()]
    r = best.index.get_level_values(0)
    share = best.to_numpy() / total.reindex(r).to_numpy()
    keep = (best.to_numpy() >= min_count) & (share >= min_share)
    return dict(zip(r[keep], best.index.get_level_values(1)[keep], strict=True))


def apply_token_map(norm: pd.DataFrame, token_map: Mapping[str, str],
                    cfg: NormaliseConfig = DEFAULT) -> pd.DataFrame:
    """Recompute the name columns of the non-Latin rows of ``norm`` with ``token_map``.

    Starts from the cached ``name_norm`` (already transliterated), so the raw names are not
    needed. Latin rows are untouched, which keeps this cheap (~7 % of pool rows).
    """
    rows = np.flatnonzero(norm["non_latin"].to_numpy())
    if len(rows) == 0 or not token_map:
        return norm
    out = norm.copy()
    sub = pa.array(norm["name_norm"].iloc[rows].tolist(), type=pa.string())
    fixed = _name_columns(sub, np.ones(len(rows), dtype=bool), cfg, token_map)
    for col in ["name_norm", "name_core", "legal_form", "name_first", "name_sorted",
                "name_squash"]:
        out.loc[out.index[rows], col] = fixed[col].to_numpy()
    name_addr = (out["name_core"].iloc[rows] + " " + out["addr_norm"].iloc[rows]).str.strip()
    out.loc[out.index[rows], "name_addr"] = name_addr.to_numpy()
    return out
