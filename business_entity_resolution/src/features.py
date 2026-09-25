"""Pairwise, language-agnostic string similarity features."""
from __future__ import annotations

from difflib import SequenceMatcher
from typing import Dict, Mapping, Sequence, Tuple

from .data import Record
from .preprocessing import normalize_address, normalize_name


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
    m = matches
    jaro = (m / len(a) + m / len(b) + (m - transpositions) / m) / 3
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


def _token_features(a: str, b: str, prefix: str, features: dict) -> None:
    ta, tb = set(a.split()), set(b.split())
    union = ta | tb
    features[prefix + "_jaccard"] = len(ta & tb) / len(union) if union else 1.0
    features[prefix + "_overlap"] = float(len(ta & tb))


def pair_features(left: Record, right: Record, retrieval: Mapping[str, float]) -> Dict[str, float]:
    name_a, name_b = normalize_name(left.get("business_name", "")), normalize_name(right.get("business_name", ""))
    addr_a, addr_b = normalize_address(left.get("business_address", "")), normalize_address(right.get("business_address", ""))
    features: Dict[str, float] = {}
    features["name_exact"] = float(bool(name_a) and name_a == name_b)
    features["address_exact"] = float(bool(addr_a) and addr_a == addr_b)
    features["name_levenshtein"] = _levenshtein_ratio(name_a, name_b)
    features["address_levenshtein"] = _levenshtein_ratio(addr_a, addr_b)
    features["name_jaro_winkler"] = _jaro_winkler(name_a, name_b)
    _token_features(name_a, name_b, "name", features)
    _token_features(addr_a, addr_b, "address", features)
    features["name_sequence"] = SequenceMatcher(None, name_a, name_b).ratio()
    features["address_sequence"] = SequenceMatcher(None, addr_a, addr_b).ratio()
    features["name_lcs_ratio"] = _lcs_ratio(name_a, name_b)
    features["address_lcs_ratio"] = _lcs_ratio(addr_a, addr_b)
    features["name_length_ratio"] = min(len(name_a), len(name_b)) / max(len(name_a), len(name_b), 1)
    features["address_length_ratio"] = min(len(addr_a), len(addr_b)) / max(len(addr_a), len(addr_b), 1)
    features["country_match"] = float(left.get("country", "").strip().casefold() == right.get("country", "").strip().casefold())
    features["candidate_rank"] = float(retrieval.get("rank", 0))
    features["blocking_similarity"] = float(retrieval.get("similarity", 0.0))
    return features


def feature_matrix(candidates: Mapping[Tuple[str, str], dict], source1: Sequence[Record], source2: Sequence[Record], source3: Sequence[Record]):
    import numpy as np
    from .data import index_by_id
    left, right = index_by_id(source1), {**index_by_id(source2), **index_by_id(source3)}
    names = ["name_exact", "address_exact", "name_levenshtein", "address_levenshtein", "name_jaro_winkler", "name_jaccard", "name_overlap", "address_jaccard", "address_overlap", "name_sequence", "address_sequence", "name_lcs_ratio", "address_lcs_ratio", "name_length_ratio", "address_length_ratio", "country_match", "candidate_rank", "blocking_similarity"]
    rows, pairs = [], []
    for pair, retrieval in candidates.items():
        if pair[0] not in left or pair[1] not in right:
            continue
        feat = pair_features(left[pair[0]], right[pair[1]], retrieval)
        rows.append([feat[name] for name in names])
        pairs.append(pair)
    return np.asarray(rows, dtype=np.float32).reshape((-1, len(names))), pairs, names

