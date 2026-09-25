"""Pairwise lexical, structural, and semantic similarity features."""
from __future__ import annotations

import math
import re
import warnings
from difflib import SequenceMatcher
from typing import Dict, Mapping, Sequence, Tuple

from .data import Record
from .preprocessing import (
    normalize_name,
    preprocess_record,
    preprocess_records,
)

_SPACE = re.compile(r"\s+")
_EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_embedding_model = None
_embedding_model_attempted = False
_embedding_cache: dict[tuple[str, str], object] = {}


def _raw_text(value: object) -> str:
    return _SPACE.sub(" ", str(value or "").casefold()).strip()


def _levenshtein_ratio(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return 1.0 - previous[-1] / max(len(a), len(b))


def _jaro_winkler(a: str, b: str) -> float:
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    radius = max(0, max(len(a), len(b)) // 2 - 1)
    am, bm = [False] * len(a), [False] * len(b)
    matches = 0
    for i, char in enumerate(a):
        for j in range(max(0, i - radius), min(i + radius + 1, len(b))):
            if not bm[j] and char == b[j]:
                am[i] = bm[j] = True
                matches += 1
                break
    if not matches:
        return 0.0
    a_match = [a[i] for i in range(len(a)) if am[i]]
    b_match = [b[i] for i in range(len(b)) if bm[i]]
    transpositions = sum(x != y for x, y in zip(a_match, b_match)) / 2
    jaro = (matches / len(a) + matches / len(b) + (matches - transpositions) / matches) / 3
    prefix = 0
    for x, y in zip(a[:4], b[:4]):
        if x != y:
            break
        prefix += 1
    return jaro + prefix * 0.1 * (1 - jaro)


def _lcs_ratio(a: str, b: str) -> float:
    """Normalized longest common subsequence length using linear memory."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    if len(b) > len(a):
        a, b = b, a
    previous = [0] * (len(b) + 1)
    for ca in a:
        current = [0]
        for j, cb in enumerate(b, 1):
            current.append(previous[j - 1] + 1 if ca == cb else max(previous[j], current[-1]))
        previous = current
    return previous[-1] / max(len(a), len(b))


def _substring_ratio(a: str, b: str) -> float:
    """Normalized longest common contiguous substring length."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    if len(b) > len(a):
        a, b = b, a
    previous = [0] * (len(b) + 1)
    longest = 0
    for ca in a:
        current = [0]
        for j, cb in enumerate(b, 1):
            length = previous[j - 1] + 1 if ca == cb else 0
            current.append(length)
            longest = max(longest, length)
        previous = current
    return longest / max(len(a), len(b))


def _token_features(a: str, b: str, prefix: str, features: dict) -> None:
    ta, tb = set(a.split()), set(b.split())
    union = ta | tb
    features[prefix + "_jaccard"] = len(ta & tb) / len(union) if union else 1.0
    features[prefix + "_overlap"] = float(len(ta & tb))


def _embedding_key(record: Mapping[str, object]) -> tuple[str, str]:
    text = normalize_name(record.get("business_name", ""))
    entity_id = str(record.get("entity_id", ""))
    return (entity_id, text) if entity_id else (text, text)


def _get_embedding_model():
    global _embedding_model, _embedding_model_attempted
    if _embedding_model_attempted:
        return _embedding_model
    _embedding_model_attempted = True
    try:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer(_EMBEDDING_MODEL_NAME)
    except Exception as exc:
        warnings.warn(
            f"Semantic embeddings unavailable ({exc}); name_embedding_cosine will be 0.0.",
            RuntimeWarning,
            stacklevel=2,
        )
        _embedding_model = None
    return _embedding_model


def _cache_embeddings(records: Sequence[Mapping[str, object]]) -> None:
    missing: dict[tuple[str, str], str] = {}
    for record in records:
        key = _embedding_key(record)
        if key not in _embedding_cache:
            missing[key] = key[1]
    for key, text in tuple(missing.items()):
        if not text:
            _embedding_cache[key] = None
            del missing[key]
    if not missing:
        return
    model = _get_embedding_model()
    if model is None:
        for key in missing:
            _embedding_cache[key] = None
        return
    try:
        vectors = model.encode(list(missing.values()), convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        for key, vector in zip(missing, vectors):
            _embedding_cache[key] = vector
    except Exception as exc:
        warnings.warn(
            f"Semantic embedding failed ({exc}); affected cosine features will be 0.0.",
            RuntimeWarning,
            stacklevel=2,
        )
        for key in missing:
            _embedding_cache[key] = None


def _embedding_similarity(left: Mapping[str, object], right: Mapping[str, object]) -> float:
    _cache_embeddings((left, right))
    a = _embedding_cache.get(_embedding_key(left))
    b = _embedding_cache.get(_embedding_key(right))
    if a is None or b is None:
        return 0.0
    try:
        similarity = float(a @ b)
        return max(-1.0, min(1.0, similarity)) if math.isfinite(similarity) else 0.0
    except (TypeError, ValueError):
        return 0.0


def pair_features(left: Record, right: Record, retrieval: Mapping[str, float]) -> Dict[str, float]:
    derived = {
        "business_name_normalized", "business_name_core",
        "business_address_normalized", "business_address_numbers",
    }
    left_fields = left if derived.issubset(left) else preprocess_record(left)
    right_fields = right if derived.issubset(right) else preprocess_record(right)
    name_a, name_b = left_fields["business_name_normalized"], right_fields["business_name_normalized"]
    core_a, core_b = left_fields["business_name_core"], right_fields["business_name_core"]
    address_a, address_b = left_fields["business_address_normalized"], right_fields["business_address_normalized"]
    raw_name_a, raw_name_b = _raw_text(left.get("business_name", "")), _raw_text(right.get("business_name", ""))
    raw_address_a, raw_address_b = _raw_text(left.get("business_address", "")), _raw_text(right.get("business_address", ""))
    source_name_a, source_name_b = str(left.get("business_name", "") or ""), str(right.get("business_name", "") or "")
    source_address_a, source_address_b = str(left.get("business_address", "") or ""), str(right.get("business_address", "") or "")
    numbers_a = left_fields["business_address_numbers"]
    numbers_b = right_fields["business_address_numbers"]

    features: Dict[str, float] = {
        "name_exact": float(bool(name_a) and name_a == name_b),
        "name_exact_raw": float(bool(source_name_a) and source_name_a == source_name_b),
        "name_exact_core": float(bool(core_a) and core_a == core_b),
        "address_exact": float(bool(address_a) and address_a == address_b),
        "address_exact_raw": float(bool(source_address_a) and source_address_a == source_address_b),
        "name_levenshtein": _levenshtein_ratio(name_a, name_b),
        "name_levenshtein_raw": _levenshtein_ratio(raw_name_a, raw_name_b),
        "name_levenshtein_core": _levenshtein_ratio(core_a, core_b),
        "address_levenshtein": _levenshtein_ratio(address_a, address_b),
        "name_jaro_winkler": _jaro_winkler(name_a, name_b),
        "name_jaro_winkler_raw": _jaro_winkler(raw_name_a, raw_name_b),
        "name_jaro_winkler_core": _jaro_winkler(core_a, core_b),
        "name_sequence": SequenceMatcher(None, name_a, name_b).ratio(),
        "address_sequence": SequenceMatcher(None, address_a, address_b).ratio(),
        "name_lcs_ratio": _lcs_ratio(name_a, name_b),
        "name_substring_ratio": _substring_ratio(name_a, name_b),
        "address_lcs_ratio": _lcs_ratio(address_a, address_b),
        "address_substring_ratio": _substring_ratio(address_a, address_b),
        "name_length_ratio": min(len(name_a), len(name_b)) / max(len(name_a), len(name_b), 1),
        "address_length_ratio": min(len(address_a), len(address_b)) / max(len(address_a), len(address_b), 1),
        "address_number_match": 0.5 if not numbers_a or not numbers_b else float(numbers_a == numbers_b),
        "name_embedding_cosine": _embedding_similarity(left, right),
        "country_match": float(bool(_raw_text(left.get("country", ""))) and _raw_text(left.get("country", "")) == _raw_text(right.get("country", ""))),
        "candidate_rank": _finite_float(retrieval.get("rank", 0.0)),
        "blocking_similarity": _finite_float(retrieval.get("similarity", 0.0)),
    }
    _token_features(raw_name_a, raw_name_b, "name_raw", features)
    _token_features(name_a, name_b, "name", features)
    _token_features(core_a, core_b, "name_core", features)
    _token_features(raw_address_a, raw_address_b, "address_raw", features)
    _token_features(address_a, address_b, "address", features)
    return {key: _finite_float(value) for key, value in features.items()}


def _finite_float(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    return number if math.isfinite(number) else 0.0


_FEATURE_NAMES = [
    "name_exact", "name_exact_raw", "name_exact_core", "address_exact", "address_exact_raw",
    "name_levenshtein", "name_levenshtein_raw", "name_levenshtein_core", "address_levenshtein",
    "name_jaro_winkler", "name_jaro_winkler_raw", "name_jaro_winkler_core",
    "name_raw_jaccard", "name_raw_overlap", "name_jaccard", "name_overlap",
    "name_core_jaccard", "name_core_overlap", "address_raw_jaccard", "address_raw_overlap",
    "address_jaccard", "address_overlap", "name_sequence", "address_sequence",
    "name_lcs_ratio", "name_substring_ratio", "address_lcs_ratio", "address_substring_ratio",
    "name_length_ratio", "address_length_ratio", "address_number_match", "name_embedding_cosine",
    "country_match", "candidate_rank", "blocking_similarity",
]


def feature_matrix(candidates: Mapping[Tuple[str, str], dict], source1: Sequence[Record], source2: Sequence[Record], source3: Sequence[Record]):
    import numpy as np
    left = {record["entity_id"]: record for record in preprocess_records(source1)}
    right = {
        record["entity_id"]: record
        for record in preprocess_records((*source2, *source3))
    }
    _cache_embeddings(tuple(left.values()) + tuple(right.values()))
    rows, pairs = [], []
    for pair, retrieval in candidates.items():
        if pair[0] not in left or pair[1] not in right:
            continue
        features = pair_features(left[pair[0]], right[pair[1]], retrieval)
        rows.append([features[name] for name in _FEATURE_NAMES])
        pairs.append(pair)
    matrix = np.asarray(rows, dtype=np.float32).reshape((-1, len(_FEATURE_NAMES)))
    return matrix, pairs, list(_FEATURE_NAMES)
