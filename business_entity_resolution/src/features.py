"""Batched lexical features for the bounded candidate set.

No embedding model is used.  The former SentenceTransformer path could silently
produce an all-zero feature without an offline model cache, so it was removed.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from rapidfuzz import fuzz, process

from .data import Record
from .preprocessing import extract_address_numbers, core_name, normalize_address, normalize_name

FEATURE_NAMES = [
    "name_exact", "core_exact", "sorted_core_exact", "address_exact", "country_match",
    "name_ratio", "core_ratio", "address_ratio", "name_token_jaccard", "name_token_overlap",
    "address_token_jaccard", "address_token_overlap", "address_number_match", "name_length_ratio",
    "address_length_ratio", "candidate_rank", "blocking_similarity",
]


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
    name_jaccard, name_overlap = _token_scores(s_core, t_core)
    address_jaccard, address_overlap = _token_scores(s_address, t_address)
    s_name_len, t_name_len = np.asarray([len(value) for value in s_name]), np.asarray([len(value) for value in t_name])
    s_addr_len, t_addr_len = np.asarray([len(value) for value in s_address]), np.asarray([len(value) for value in t_address])
    return np.column_stack((
        np.asarray([bool(a) and a == b for a, b in zip(s_name, t_name)], dtype=np.float32),
        np.asarray([bool(a) and a == b for a, b in zip(s_core, t_core)], dtype=np.float32),
        np.asarray([bool(a) and a == b for a, b in zip(s_sorted, t_sorted)], dtype=np.float32),
        np.asarray([bool(a) and a == b for a, b in zip(s_address, t_address)], dtype=np.float32),
        np.asarray([bool(a) and a == b for a, b in zip(s_country, t_country)], dtype=np.float32),
        name_ratio, core_ratio, address_ratio, name_jaccard, name_overlap, address_jaccard, address_overlap,
        np.asarray([0.5 if not a or not b else float(a == b) for a, b in zip(s_numbers, t_numbers)], dtype=np.float32),
        np.minimum(s_name_len, t_name_len) / np.maximum(np.maximum(s_name_len, t_name_len), 1),
        np.minimum(s_addr_len, t_addr_len) / np.maximum(np.maximum(s_addr_len, t_addr_len), 1),
        np.asarray(rank, dtype=np.float32), np.asarray(similarity, dtype=np.float32),
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
