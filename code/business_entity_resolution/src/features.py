"""Batched lexical features for the bounded candidate set.

No embedding model is used.  The former SentenceTransformer path could silently
produce an all-zero feature without an offline model cache, so it was removed.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from rapidfuzz import fuzz, process
from sklearn.feature_extraction.text import HashingVectorizer

from .data import Record
from .preprocessing import extract_address_numbers, core_name, normalize_address, normalize_name

FEATURE_NAMES = [
    "name_exact", "core_exact", "sorted_core_exact", "address_exact", "country_match", "country_missing",
    "name_ratio", "core_ratio", "address_ratio", "name_token_sort_ratio", "name_token_set_ratio",
    "address_token_sort_ratio", "address_token_set_ratio", "name_char_trigram_cosine", "address_char_trigram_cosine",
    "name_token_jaccard", "name_token_overlap", "address_token_jaccard", "address_token_overlap",
    "address_number_jaccard", "first_number_match", "name_length_ratio", "address_length_ratio",
    "candidate_rank", "blocking_similarity",
    # Missingness, source, and structural features
    "name_missing_left", "name_missing_right", "address_missing_left", "address_missing_right",
    "target_source", "legal_suffix_agree", "legal_suffix_conflict",
    "name_token_count_diff", "core_token_count_diff",
    "country_both_missing", "country_conflict",
]

_CHAR_TRIGRAMS = HashingVectorizer(
    analyzer="char_wb", ngram_range=(3, 3), n_features=2 ** 14,
    alternate_sign=False, norm=None, dtype=np.float32,
)


def _tokens(value: str) -> set[str]:
    return set(value.split())


def _token_scores(left: Sequence[str], right: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    jaccard, overlap = np.zeros(len(left), dtype=np.float32), np.zeros(len(left), dtype=np.float32)
    for index, (a, b) in enumerate(zip(left, right)):
        a_tokens, b_tokens = _tokens(a), _tokens(b)
        union = a_tokens | b_tokens
        jaccard[index] = len(a_tokens & b_tokens) / len(union) if union else 1.0
        overlap[index] = len(a_tokens & b_tokens)
    return jaccard, overlap


def _hashed_cosine(left: Sequence[str], right: Sequence[str]) -> np.ndarray:
    """Fit-free character 3-gram cosine for already-bounded candidate pairs."""
    left_matrix = _CHAR_TRIGRAMS.transform(left)
    right_matrix = _CHAR_TRIGRAMS.transform(right)
    numerator = np.asarray(left_matrix.multiply(right_matrix).sum(axis=1)).ravel()
    left_norm = np.sqrt(np.asarray(left_matrix.multiply(left_matrix).sum(axis=1)).ravel())
    right_norm = np.sqrt(np.asarray(right_matrix.multiply(right_matrix).sum(axis=1)).ravel())
    return np.divide(numerator, left_norm * right_norm, out=np.zeros(len(left), dtype=np.float32), where=(left_norm * right_norm) > 0)


def _number_scores(left: Sequence[str], right: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    jaccard, first_match = np.full(len(left), 0.5, dtype=np.float32), np.full(len(left), 0.5, dtype=np.float32)
    for index, (a, b) in enumerate(zip(left, right)):
        left_values, right_values = a.split(), b.split()
        if left_values and right_values:
            union = set(left_values) | set(right_values)
            jaccard[index] = len(set(left_values) & set(right_values)) / len(union)
            first_match[index] = float(left_values[0] == right_values[0])
    return jaccard, first_match


def feature_batch(rows: Sequence[tuple]) -> np.ndarray:
    """Vectorize one DB batch; edit distances use RapidFuzz's C++ cpdist."""
    if not rows:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
    # (sid, tid, s_country, s_name, s_core, s_sorted, s_address, s_numbers,
    #  t_country, t_name, t_core, t_sorted, t_address, t_numbers, evidence, similarity, rank)
    columns = list(zip(*rows))
    s_country, s_name, s_core, s_sorted, s_address, s_numbers = columns[2:8]
    t_country, t_name, t_core, t_sorted, t_address, t_numbers = columns[8:14]
    evidence, similarity, rank = columns[14:17]
    name_ratio = process.cpdist(s_name, t_name, scorer=fuzz.ratio, dtype=np.uint8, workers=-1).astype(np.float32) / 100
    core_ratio = process.cpdist(s_core, t_core, scorer=fuzz.ratio, dtype=np.uint8, workers=-1).astype(np.float32) / 100
    address_ratio = process.cpdist(s_address, t_address, scorer=fuzz.ratio, dtype=np.uint8, workers=-1).astype(np.float32) / 100
    name_token_sort = process.cpdist(s_name, t_name, scorer=fuzz.token_sort_ratio, dtype=np.uint8, workers=-1).astype(np.float32) / 100
    name_token_set = process.cpdist(s_name, t_name, scorer=fuzz.token_set_ratio, dtype=np.uint8, workers=-1).astype(np.float32) / 100
    address_token_sort = process.cpdist(s_address, t_address, scorer=fuzz.token_sort_ratio, dtype=np.uint8, workers=-1).astype(np.float32) / 100
    address_token_set = process.cpdist(s_address, t_address, scorer=fuzz.token_set_ratio, dtype=np.uint8, workers=-1).astype(np.float32) / 100
    name_jaccard, name_overlap = _token_scores(s_core, t_core)
    address_jaccard, address_overlap = _token_scores(s_address, t_address)
    number_jaccard, first_number_match = _number_scores(s_numbers, t_numbers)
    s_name_len, t_name_len = np.asarray([len(value) for value in s_name]), np.asarray([len(value) for value in t_name])
    s_addr_len, t_addr_len = np.asarray([len(value) for value in s_address]), np.asarray([len(value) for value in t_address])
    # Legal suffix: difference between normalized name and core name
    s_suffix = [n[len(c):].strip() if n.startswith(c) else n.replace(c, '', 1).strip()
                for n, c in zip(s_name, s_core)]
    t_suffix = [n[len(c):].strip() if n.startswith(c) else n.replace(c, '', 1).strip()
                for n, c in zip(t_name, t_core)]
    return np.column_stack((
        np.asarray([bool(a) and a == b for a, b in zip(s_name, t_name)], dtype=np.float32),
        np.asarray([bool(a) and a == b for a, b in zip(s_core, t_core)], dtype=np.float32),
        np.asarray([bool(a) and a == b for a, b in zip(s_sorted, t_sorted)], dtype=np.float32),
        np.asarray([bool(a) and a == b for a, b in zip(s_address, t_address)], dtype=np.float32),
        np.asarray([bool(a) and a == b for a, b in zip(s_country, t_country)], dtype=np.float32),
        np.asarray([not a or not b for a, b in zip(s_country, t_country)], dtype=np.float32),
        name_ratio, core_ratio, address_ratio, name_token_sort, name_token_set, address_token_sort, address_token_set,
        _hashed_cosine(s_name, t_name), _hashed_cosine(s_address, t_address),
        name_jaccard, name_overlap, address_jaccard, address_overlap, number_jaccard, first_number_match,
        np.minimum(s_name_len, t_name_len) / np.maximum(np.maximum(s_name_len, t_name_len), 1),
        np.minimum(s_addr_len, t_addr_len) / np.maximum(np.maximum(s_addr_len, t_addr_len), 1),
        np.asarray(rank, dtype=np.float32), np.asarray(similarity, dtype=np.float32),
        # ── Missingness features ──
        np.asarray([not a for a in s_name], dtype=np.float32),
        np.asarray([not a for a in t_name], dtype=np.float32),
        np.asarray([not a for a in s_address], dtype=np.float32),
        np.asarray([not a for a in t_address], dtype=np.float32),
        # ── Target source (2=S2, 3=S3) ──
        np.asarray([3.0 if tid.startswith('S3-') else 2.0 for tid in columns[1]], dtype=np.float32),
        # ── Legal suffix agreement / conflict ──
        np.asarray([bool(a) and a == b for a, b in zip(s_suffix, t_suffix)], dtype=np.float32),
        np.asarray([bool(a) and bool(b) and a != b for a, b in zip(s_suffix, t_suffix)], dtype=np.float32),
        # ── Token count differences ──
        np.abs(np.asarray([len(a.split()) for a in s_name], dtype=np.float32) -
               np.asarray([len(a.split()) for a in t_name], dtype=np.float32)),
        np.abs(np.asarray([len(a.split()) for a in s_core], dtype=np.float32) -
               np.asarray([len(a.split()) for a in t_core], dtype=np.float32)),
        # ── Refined country features ──
        np.asarray([not a and not b for a, b in zip(s_country, t_country)], dtype=np.float32),
        np.asarray([bool(a) and bool(b) and a != b for a, b in zip(s_country, t_country)], dtype=np.float32),
    )).astype(np.float32, copy=False)


def pair_features(left: Record, right: Record, retrieval: Mapping[str, float]) -> dict[str, float]:
    """Compatibility helper for small diagnostics and unit tests."""
    s_name, t_name = normalize_name(left.get("business_name", "")), normalize_name(right.get("business_name", ""))
    s_core, t_core = core_name(left.get("business_name", "")), core_name(right.get("business_name", ""))
    s_address, t_address = normalize_address(left.get("business_address", "")), normalize_address(right.get("business_address", ""))
    s_sorted, t_sorted = " ".join(sorted(s_core.split())), " ".join(sorted(t_core.split()))
    row = ("", "", str(left.get("country", "")).casefold(), s_name, s_core, s_sorted, s_address, extract_address_numbers(s_address),
           str(right.get("country", "")).casefold(), t_name, t_core, t_sorted, t_address, extract_address_numbers(t_address),
           0.0, float(retrieval.get("similarity", 0.0)), float(retrieval.get("rank", 0.0)))
    return dict(zip(FEATURE_NAMES, feature_batch([row])[0].tolist()))


def feature_matrix(candidates: Mapping[tuple[str, str], dict], source1: Sequence[Record], source2: Sequence[Record], source3: Sequence[Record]):
    """Small-data compatibility API. Production uses feature_batch over SQLite rows."""
    left = {record["entity_id"]: record for record in source1}
    right = {record["entity_id"]: record for record in (*source2, *source3)}
    rows, pairs = [], []
    for (source_id, target_id), retrieval in candidates.items():
        if source_id not in left or target_id not in right:
            continue
        s, t = left[source_id], right[target_id]
        s_name, t_name = normalize_name(s.get("business_name", "")), normalize_name(t.get("business_name", ""))
        s_core, t_core = core_name(s.get("business_name", "")), core_name(t.get("business_name", ""))
        s_address, t_address = normalize_address(s.get("business_address", "")), normalize_address(t.get("business_address", ""))
        rows.append((source_id, target_id, str(s.get("country", "")).casefold(), s_name, s_core, " ".join(sorted(s_core.split())), s_address, extract_address_numbers(s_address),
                     str(t.get("country", "")).casefold(), t_name, t_core, " ".join(sorted(t_core.split())), t_address, extract_address_numbers(t_address),
                     0.0, retrieval.get("similarity", 0.0), retrieval.get("rank", 0.0)))
        pairs.append((source_id, target_id))
    return feature_batch(rows), pairs, list(FEATURE_NAMES)
