"""Batched lexical, structured-address, retrieval and reciprocal-rank features."""
from __future__ import annotations

from typing import Sequence

import numpy as np
from rapidfuzz import fuzz, process
from scipy import sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer

from .preprocessing import extract_acronym, extract_address_numbers, extract_postal_code, normalize_address, normalize_name
from .semantic_retrieval import SemanticRetriever, entity_text
from .variation import VariationModel

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
    # Structured address (8)
    "street_number_match", "street_number_conflict",
    "street_ratio", "street_token_set_ratio",
    "city_ratio", "city_token_set_ratio",
    "city_exact", "postal_missing_either",
    # Reciprocal / cross-field ranking (6)
    "name_rank_score", "address_rank_score", "name_address_rank_agreement",
    "reverse_similarity_rank_score", "mutual_best", "source2_indicator",
    "source3_indicator",
    # Target-set relational evidence
    "opposite_source_bridge",
    "same_source_competition",
    "bridge_gap",
    # Training-derived lexical variation (zero-valued during untuned baseline).
    "learned_name_alias", "learned_name_explained",
    "learned_address_alias", "learned_address_explained",
    # Semantic retrieval / reranking.
    "semantic_similarity", "semantic_rank_score",
    "semantic_lexical_alignment", "cross_encoder_score",
]

_CHAR_TRIGRAMS = HashingVectorizer(
    analyzer="char_wb", ngram_range=(3, 3), n_features=2 ** 14,
    alternate_sign=False, norm=None, dtype=np.float32,
)
_HASH_CACHE: dict[str, tuple[object, np.float32]] = {}
_HASH_CACHE_LIMIT = 50_000


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


def _hashed_cosine(left: Sequence[str], right: Sequence[str]) -> np.ndarray:
    global _HASH_CACHE
    while len(_HASH_CACHE) > _HASH_CACHE_LIMIT:
        _HASH_CACHE.pop(next(iter(_HASH_CACHE)))

    def _get(strings: Sequence[str]) -> tuple[list, np.ndarray]:
        missing_idx: list[int] = []
        missing_str: list[str] = []
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
    left_matrix = sp.vstack(l_mats, format="csr")
    right_matrix = sp.vstack(r_mats, format="csr")
    numerator = np.asarray(left_matrix.multiply(right_matrix).sum(axis=1)).ravel().astype(np.float32, copy=False)
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


def _set_jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta and not tb:
        return 1.0
    union = ta | tb
    return len(ta & tb) / len(union) if union else 0.0


def _group_relational_scores(
    source_ids: Sequence[str],
    target_ids: Sequence[str],
    target_cores: Sequence[str],
    target_addresses: Sequence[str],
    retrieval_scores: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cheap cross-source corroboration inside each S1 candidate set.

    We inspect at most three strong opposite-source and same-source anchors per
    candidate. This provides multi-source consistency without another fuzzy pass.
    """
    n = len(source_ids)
    opposite = np.zeros(n, dtype=np.float32)
    competition = np.zeros(n, dtype=np.float32)
    groups: dict[str, list[int]] = {}
    for i, sid in enumerate(source_ids):
        groups.setdefault(sid, []).append(i)

    for indices in groups.values():
        ranked = sorted(indices, key=lambda i: (-float(retrieval_scores[i]), target_ids[i]))
        s2 = [i for i in ranked if target_ids[i].startswith("S2-")]
        s3 = [i for i in ranked if target_ids[i].startswith("S3-")]
        buckets = {"S2": s2, "S3": s3}
        for i in indices:
            source = "S3" if target_ids[i].startswith("S3-") else "S2"
            opposite_pool = buckets["S2" if source == "S3" else "S3"][:3]
            same_pool = [j for j in buckets[source][:4] if j != i][:3]
            if opposite_pool:
                opposite[i] = max(
                    0.60 * _set_jaccard(target_cores[i], target_cores[j])
                    + 0.40 * _set_jaccard(target_addresses[i], target_addresses[j])
                    for j in opposite_pool
                )
            if same_pool:
                competition[i] = max(
                    0.60 * _set_jaccard(target_cores[i], target_cores[j])
                    + 0.40 * _set_jaccard(target_addresses[i], target_addresses[j])
                    for j in same_pool
                )

    return opposite, competition, opposite - competition


def _safe_optional_columns(columns: list[tuple], n: int) -> tuple[list[str], ...]:
    """Read structured/rank columns while preserving legacy test compatibility."""
    # Current candidate rows have 19 base columns followed by 13 structured/rank
    # fields. Older tests may still pass the historical 30/31-column shape.
    if len(columns) >= 32:
        s_num = columns[19]
        s_street = columns[20]
        s_city = columns[21]
        s_postal = columns[22]
        t_num = columns[23]
        t_street = columns[24]
        t_city = columns[25]
        t_postal = columns[26]
        name_rank = columns[27]
        address_rank = columns[28]
        reverse_rank = columns[29]
        mutual_best = columns[30]
        rank_agreement = columns[31]
        return (
            list(s_num), list(s_street), list(s_city), list(s_postal),
            list(t_num), list(t_street), list(t_city), list(t_postal),
            list(name_rank), list(address_rank), list(reverse_rank),
            list(mutual_best), list(rank_agreement),
        )

    # Legacy rich rows without semantic columns.
    if len(columns) >= 30:
        s_num = columns[17]
        s_street = columns[18]
        s_city = columns[19]
        s_postal = columns[20]
        t_num = columns[21]
        t_street = columns[22]
        t_city = columns[23]
        t_postal = columns[24]
        name_rank = columns[25]
        address_rank = columns[26]
        reverse_rank = columns[27]
        mutual_best = columns[28]
        rank_agreement = columns[29]
        return (
            list(s_num), list(s_street), list(s_city), list(s_postal),
            list(t_num), list(t_street), list(t_city), list(t_postal),
            list(name_rank), list(address_rank), list(reverse_rank),
            list(mutual_best), list(rank_agreement),
        )

    blanks = [""] * n
    zeros = [0.0] * n
    neutral_rank = [1.0] * n
    neutral_agreement = [1.0] * n
    return (
        blanks.copy(), blanks.copy(), blanks.copy(), blanks.copy(),
        blanks.copy(), blanks.copy(), blanks.copy(), blanks.copy(),
        neutral_rank.copy(), neutral_rank.copy(), neutral_rank.copy(),
        zeros.copy(), neutral_agreement.copy(),
    )


def feature_batch(
    rows: Sequence[tuple],
    variation_model: VariationModel | None = None,
    semantic_retriever: SemanticRetriever | None = None,
) -> np.ndarray:
    """Vectorize a batch of candidate rows.

    ``rows`` may be the legacy 17-column candidate representation used by older
    tests or the current richer representation produced by ``BlockingStore``.
    """
    if not rows:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)

    columns = list(zip(*rows))
    s_country, s_name, s_core, s_sorted, s_address, s_numbers = columns[2:8]
    t_country, t_name, t_core, t_sorted, t_address, t_numbers = columns[8:14]
    evidence, similarity, rank = columns[14:17]
    semantic_similarity = (
        np.asarray(columns[17], dtype=np.float32)
        if len(columns) >= 19 else np.zeros(len(rows), dtype=np.float32)
    )
    semantic_rank = (
        np.asarray(columns[18], dtype=np.float32)
        if len(columns) >= 19 else np.zeros(len(rows), dtype=np.float32)
    )
    (
        s_num, s_street, s_city, s_postal,
        t_num, t_street, t_city, t_postal,
        name_rank, address_rank, reverse_rank,
        mutual_best, rank_agreement,
    ) = _safe_optional_columns(columns, len(rows))

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

    street_ratio = process.cpdist(s_street, t_street, scorer=fuzz.ratio, dtype=np.uint8, workers=-1).astype(np.float32) / 100.0
    street_token_set = process.cpdist(s_street, t_street, scorer=fuzz.token_set_ratio, dtype=np.uint8, workers=-1).astype(np.float32) / 100.0
    city_ratio = process.cpdist(s_city, t_city, scorer=fuzz.ratio, dtype=np.uint8, workers=-1).astype(np.float32) / 100.0
    city_token_set = process.cpdist(s_city, t_city, scorer=fuzz.token_set_ratio, dtype=np.uint8, workers=-1).astype(np.float32) / 100.0
    street_number_match = np.asarray([bool(a) and bool(b) and a == b for a, b in zip(s_num, t_num)], dtype=np.float32)
    street_number_conflict = np.asarray([bool(a) and bool(b) and a != b for a, b in zip(s_num, t_num)], dtype=np.float32)
    city_exact = np.asarray([bool(a) and bool(b) and a == b for a, b in zip(s_city, t_city)], dtype=np.float32)
    postal_missing_either = np.asarray([not a or not b for a, b in zip(s_postal, t_postal)], dtype=np.float32)

    s_name_len = np.asarray([len(v) for v in s_name], dtype=np.float32)
    t_name_len = np.asarray([len(v) for v in t_name], dtype=np.float32)
    s_addr_len = np.asarray([len(v) for v in s_address], dtype=np.float32)
    t_addr_len = np.asarray([len(v) for v in t_address], dtype=np.float32)

    s_suffix = [n[len(c):].strip() if c and n.startswith(c) else (n.replace(c, "", 1).strip() if c else "") for n, c in zip(s_name, s_core)]
    t_suffix = [n[len(c):].strip() if c and n.startswith(c) else (n.replace(c, "", 1).strip() if c else "") for n, c in zip(t_name, t_core)]

    name_rank_score = 1.0 / np.maximum(np.asarray(name_rank, dtype=np.float32), 1.0)
    address_rank_score = 1.0 / np.maximum(np.asarray(address_rank, dtype=np.float32), 1.0)
    reverse_rank_score = 1.0 / np.maximum(np.asarray(reverse_rank, dtype=np.float32), 1.0)
    opposite_bridge, same_source_competition, bridge_gap = _group_relational_scores(
        columns[0], columns[1], t_core, t_address, similarity
    )

    if variation_model is None:
        learned_variation = np.zeros((len(rows), 4), dtype=np.float32)
    else:
        learned_variation = variation_model.score_batch(
            s_core, t_core, s_address, t_address
        )

    semantic_rank_score = np.divide(
        1.0,
        1.0 + semantic_rank,
        out=np.zeros(len(rows), dtype=np.float32),
        where=semantic_rank > 0,
    )
    semantic_lexical_alignment = semantic_similarity * np.maximum(name_ratio, address_ratio)

    if semantic_retriever is None or not np.any(semantic_rank > 0):
        cross_encoder_score = np.zeros(len(rows), dtype=np.float32)
    else:
        rerank_rows = []
        for sc, sa, scountry, tc, ta, tcountry, srank in zip(
            s_name, s_address, s_country, t_name, t_address, t_country, semantic_rank
        ):
            rerank_rows.append(
                (
                    entity_text(str(sc), str(sa), str(scountry)),
                    entity_text(str(tc), str(ta), str(tcountry)),
                    int(srank),
                )
            )
        cross_encoder_score = semantic_retriever.rerank(rerank_rows)

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
        # Structured address
        street_number_match, street_number_conflict,
        street_ratio, street_token_set,
        city_ratio, city_token_set, city_exact, postal_missing_either,
        # Reciprocal / cross-field ranking
        name_rank_score, address_rank_score,
        np.asarray(rank_agreement, dtype=np.float32),
        reverse_rank_score,
        np.asarray(mutual_best, dtype=np.float32),
        np.asarray([float(tid.startswith("S2-")) for tid in columns[1]], dtype=np.float32),
        np.asarray([float(tid.startswith("S3-")) for tid in columns[1]], dtype=np.float32),
        opposite_bridge,
        same_source_competition,
        bridge_gap,
        learned_variation[:, 0], learned_variation[:, 1],
        learned_variation[:, 2], learned_variation[:, 3],
        # Semantic retrieval / reranking
        semantic_similarity, semantic_rank_score,
        semantic_lexical_alignment, cross_encoder_score,
    )).astype(np.float32, copy=False)
