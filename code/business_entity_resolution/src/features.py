"""Batched, highly vectorized lexical and structural features for candidate pairs."""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from rapidfuzz import fuzz, process
from sklearn.feature_extraction.text import HashingVectorizer

from .data import Record
from .preprocessing import (
    core_name,
    extract_acronym,
    extract_address_numbers,
    extract_postal_code,
    normalize_address,
    normalize_name,
)

FEATURE_NAMES = [
    # Exact (4)
    "name_exact", "core_exact", "sorted_core_exact", "address_exact",
    # Country (4)
    "country_match", "country_missing", "country_both_missing", "country_conflict",
    # RapidFuzz (7)
    "name_ratio", "core_ratio", "address_ratio",
    "name_token_sort_ratio", "name_token_set_ratio", "address_token_sort_ratio", "address_token_set_ratio",
    # Partial (2)
    "name_partial_ratio", "core_partial_ratio",
    # Containment & Acronym (3)
    "name_token_containment", "address_token_containment", "acronym_match",
    # Numbers & Postals (4)
    "address_number_jaccard", "first_number_match", "postal_match", "postal_conflict",
    # Trigrams (2)
    "name_char_trigram_cosine", "address_char_trigram_cosine",
    # Token scores (4)
    "name_token_jaccard", "name_token_overlap", "address_token_jaccard", "address_token_overlap",
    # Length & Counts (4)
    "name_length_ratio", "address_length_ratio", "name_token_count_diff", "core_token_count_diff",
    # Retrieval (3)
    "candidate_rank", "blocking_similarity", "target_source",
    # Suffix (2)
    "legal_suffix_agree", "legal_suffix_conflict",
    # Missingness (4)
    "name_missing_left", "name_missing_right", "address_missing_left", "address_missing_right",
]

_CHAR_TRIGRAMS = HashingVectorizer(
    analyzer="char_wb", ngram_range=(3, 3), n_features=2 ** 14,
    alternate_sign=False, norm=None, dtype=np.float32,
)


def _tokens(value: str) -> set[str]:
    return set(value.split())


def _token_scores(left: Sequence[str], right: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    jaccard = np.zeros(len(left), dtype=np.float32)
    overlap = np.zeros(len(left), dtype=np.float32)
    for index, (a, b) in enumerate(zip(left, right)):
        ta, tb = _tokens(a), _tokens(b)
        union = ta | tb
        jaccard[index] = len(ta & tb) / len(union) if union else 1.0
        overlap[index] = float(len(ta & tb))
    return jaccard, overlap


def _containment_scores(left: Sequence[str], right: Sequence[str]) -> np.ndarray:
    containment = np.zeros(len(left), dtype=np.float32)
    for index, (a, b) in enumerate(zip(left, right)):
        ta, tb = _tokens(a), _tokens(b)
        m = min(len(ta), len(tb))
        containment[index] = len(ta & tb) / m if m > 0 else 0.0
    return containment


_HASH_CACHE = {}

def _hashed_cosine(left: Sequence[str], right: Sequence[str]) -> np.ndarray:
    global _HASH_CACHE
    if len(_HASH_CACHE) > 200_000:
        _HASH_CACHE.clear()

    def _get(strings: Sequence[str]) -> tuple[list, np.ndarray]:
        missing_idx = []
        missing_str = []
        mats = [None] * len(strings)
        norms = np.zeros(len(strings), dtype=np.float32)
        for i, s in enumerate(strings):
            cached = _HASH_CACHE.get(s)
            if cached is not None:
                mats[i], norms[i] = cached
            else:
                missing_idx.append(i)
                missing_str.append(s)
        if missing_str:
            new_mats = _CHAR_TRIGRAMS.transform(missing_str)
            new_norms = np.sqrt(np.asarray(new_mats.multiply(new_mats).sum(axis=1)).ravel())
            for i, m, n, s in zip(missing_idx, new_mats, new_norms, missing_str):
                _HASH_CACHE[s] = (m, n)
                mats[i] = m
                norms[i] = n
        return mats, norms

    l_mats, l_norms = _get(left)
    r_mats, r_norms = _get(right)
    
    numerator = np.zeros(len(left), dtype=np.float32)
    for i, (lm, rm) in enumerate(zip(l_mats, r_mats)):
        numerator[i] = lm.multiply(rm).sum()
        
    denom = l_norms * r_norms
    return np.divide(numerator, denom, out=np.zeros(len(left), dtype=np.float32), where=denom > 0)


def _number_scores(left: Sequence[str], right: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    jaccard = np.full(len(left), 0.5, dtype=np.float32)
    first_match = np.full(len(left), 0.5, dtype=np.float32)
    for index, (a, b) in enumerate(zip(left, right)):
        lv, rv = a.split(), b.split()
        if lv and rv:
            union = set(lv) | set(rv)
            jaccard[index] = len(set(lv) & set(rv)) / len(union)
            first_match[index] = float(lv[0] == rv[0])
    return jaccard, first_match


def _postal_scores(left: Sequence[str], right: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    match = np.zeros(len(left), dtype=np.float32)
    conflict = np.zeros(len(left), dtype=np.float32)
    for i, (a, b) in enumerate(zip(left, right)):
        p1, p2 = extract_postal_code(a), extract_postal_code(b)
        if p1 and p2:
            if p1 == p2:
                match[i] = 1.0
            else:
                conflict[i] = 1.0
    return match, conflict


def _acronym_matches(left: Sequence[str], right: Sequence[str]) -> np.ndarray:
    matches = np.zeros(len(left), dtype=np.float32)
    for i, (a, b) in enumerate(zip(left, right)):
        acr_a, acr_b = extract_acronym(a), extract_acronym(b)
        if (acr_a and acr_a == b) or (acr_b and acr_b == a) or (acr_a and acr_b and acr_a == acr_b):
            matches[i] = 1.0
    return matches


def feature_batch(rows: Sequence[tuple]) -> np.ndarray:
    """Vectorize a batch of DB rows using multithreaded RapidFuzz and NumPy."""
    if not rows:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)

    columns = list(zip(*rows))
    s_country, s_name, s_core, s_sorted, s_address, s_numbers = columns[2:8]
    t_country, t_name, t_core, t_sorted, t_address, t_numbers = columns[8:14]
    evidence, similarity, rank = columns[14:17]

    name_ratio = process.cpdist(s_name, t_name, scorer=fuzz.ratio, dtype=np.uint8, workers=-1).astype(np.float32) / 100.0
    core_ratio = process.cpdist(s_core, t_core, scorer=fuzz.ratio, dtype=np.uint8, workers=-1).astype(np.float32) / 100.0
    address_ratio = process.cpdist(s_address, t_address, scorer=fuzz.ratio, dtype=np.uint8, workers=-1).astype(np.float32) / 100.0
    name_token_sort = process.cpdist(s_name, t_name, scorer=fuzz.token_sort_ratio, dtype=np.uint8, workers=-1).astype(np.float32) / 100.0
    name_token_set = process.cpdist(s_name, t_name, scorer=fuzz.token_set_ratio, dtype=np.uint8, workers=-1).astype(np.float32) / 100.0
    address_token_sort = process.cpdist(s_address, t_address, scorer=fuzz.token_sort_ratio, dtype=np.uint8, workers=-1).astype(np.float32) / 100.0
    address_token_set = process.cpdist(s_address, t_address, scorer=fuzz.token_set_ratio, dtype=np.uint8, workers=-1).astype(np.float32) / 100.0
    name_partial = process.cpdist(s_name, t_name, scorer=fuzz.partial_ratio, dtype=np.uint8, workers=-1).astype(np.float32) / 100.0
    core_partial = process.cpdist(s_core, t_core, scorer=fuzz.partial_ratio, dtype=np.uint8, workers=-1).astype(np.float32) / 100.0

    name_jaccard, name_overlap = _token_scores(s_core, t_core)
    address_jaccard, address_overlap = _token_scores(s_address, t_address)
    name_containment = _containment_scores(s_core, t_core)
    address_containment = _containment_scores(s_address, t_address)
    acronym_match = _acronym_matches(s_core, t_core)

    number_jaccard, first_number_match = _number_scores(s_numbers, t_numbers)
    postal_match, postal_conflict = _postal_scores(s_address, t_address)

    s_name_len = np.asarray([len(v) for v in s_name], dtype=np.float32)
    t_name_len = np.asarray([len(v) for v in t_name], dtype=np.float32)
    s_addr_len = np.asarray([len(v) for v in s_address], dtype=np.float32)
    t_addr_len = np.asarray([len(v) for v in t_address], dtype=np.float32)

    s_suffix = [n[len(c):].strip() if c and n.startswith(c) else (n.replace(c, "", 1).strip() if c else "") for n, c in zip(s_name, s_core)]
    t_suffix = [n[len(c):].strip() if c and n.startswith(c) else (n.replace(c, "", 1).strip() if c else "") for n, c in zip(t_name, t_core)]

    return np.column_stack((
        # Exact
        np.asarray([bool(a) and a == b for a, b in zip(s_name, t_name)], dtype=np.float32),
        np.asarray([bool(a) and a == b for a, b in zip(s_core, t_core)], dtype=np.float32),
        np.asarray([bool(a) and a == b for a, b in zip(s_sorted, t_sorted)], dtype=np.float32),
        np.asarray([bool(a) and a == b for a, b in zip(s_address, t_address)], dtype=np.float32),
        # Country
        np.asarray([bool(a) and a == b for a, b in zip(s_country, t_country)], dtype=np.float32),
        np.asarray([not a or not b for a, b in zip(s_country, t_country)], dtype=np.float32),
        np.asarray([not a and not b for a, b in zip(s_country, t_country)], dtype=np.float32),
        np.asarray([bool(a) and bool(b) and a != b for a, b in zip(s_country, t_country)], dtype=np.float32),
        # RapidFuzz
        name_ratio, core_ratio, address_ratio,
        name_token_sort, name_token_set, address_token_sort, address_token_set,
        name_partial, core_partial,
        # Containment & Acronym
        name_containment, address_containment, acronym_match,
        # Numbers & Postals
        number_jaccard, first_number_match, postal_match, postal_conflict,
        # Trigrams
        _hashed_cosine(s_name, t_name), _hashed_cosine(s_address, t_address),
        # Token scores
        name_jaccard, name_overlap, address_jaccard, address_overlap,
        # Length & Counts
        np.minimum(s_name_len, t_name_len) / np.maximum(np.maximum(s_name_len, t_name_len), 1.0),
        np.minimum(s_addr_len, t_addr_len) / np.maximum(np.maximum(s_addr_len, t_addr_len), 1.0),
        np.abs(np.asarray([len(a.split()) for a in s_name], dtype=np.float32) - np.asarray([len(b.split()) for b in t_name], dtype=np.float32)),
        np.abs(np.asarray([len(a.split()) for a in s_core], dtype=np.float32) - np.asarray([len(b.split()) for b in t_core], dtype=np.float32)),
        # Retrieval
        np.asarray(rank, dtype=np.float32),
        np.asarray(similarity, dtype=np.float32),
        np.asarray([3.0 if tid.startswith("S3-") else 2.0 for tid in columns[1]], dtype=np.float32),
        # Suffix
        np.asarray([bool(a) and a == b for a, b in zip(s_suffix, t_suffix)], dtype=np.float32),
        np.asarray([bool(a) and bool(b) and a != b for a, b in zip(s_suffix, t_suffix)], dtype=np.float32),
        # Missingness
        np.asarray([not a for a in s_name], dtype=np.float32),
        np.asarray([not a for a in t_name], dtype=np.float32),
        np.asarray([not a for a in s_address], dtype=np.float32),
        np.asarray([not a for a in t_address], dtype=np.float32),
    )).astype(np.float32, copy=False)