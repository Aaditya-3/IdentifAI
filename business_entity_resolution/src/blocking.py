"""Union high-recall name and address retrieval strategies."""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Mapping, Sequence, Set, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .data import Labels, Record
from .preprocessing import core_name, normalize_address

Pair = Tuple[str, str]
logger = logging.getLogger(__name__)


def _strategy_scores(
    source1: Sequence[Record],
    targets: Sequence[Record],
    query_texts: Sequence[str],
    target_texts: Sequence[str],
    analyzer: str,
    ngram_range: Tuple[int, int],
) -> List[Dict[int, float]]:
    corpus = [text or " " for text in (*query_texts, *target_texts)]
    try:
        vectorizer = TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=ngram_range,
            min_df=1,
            sublinear_tf=True,
            strip_accents="unicode",
            dtype=np.float32,
        )
        matrix = normalize(vectorizer.fit_transform(corpus), norm="l2", copy=False)
    except ValueError:
        return [{} for _ in source1]
    query_matrix = matrix[:len(source1)]
    target_transpose = matrix[len(source1):].T.tocsr()
    scores_by_query: List[Dict[int, float]] = []
    for qi in range(len(source1)):
        row = (query_matrix[qi] @ target_transpose).tocsr()
        scores_by_query.append({int(j): float(score) for j, score in zip(row.indices, row.data)})
    return scores_by_query


def _retrieval_order(
    scores: Mapping[int, float],
    targets: Sequence[Record],
    targets_by_country: Mapping[str, List[int]],
    country: str,
    top_k: int,
) -> List[int]:
    global_order = sorted(scores, key=lambda j: (-scores[j], targets[j]["entity_id"]))[:top_k]
    local_order = sorted(
        (j for j in targets_by_country.get(country, ()) if j in scores),
        key=lambda j: (-scores[j], targets[j]["entity_id"]),
    )[:top_k]
    return list(dict.fromkeys(local_order + global_order))


def generate_candidates(source1: Sequence[Record], targets: Sequence[Record], top_k: int = 20) -> Dict[Pair, dict]:
    """Generate candidates from core-name character and address word TF-IDF."""
    if top_k < 1 or not source1 or not targets:
        return {}

    name_scores = _strategy_scores(
        source1,
        targets,
        [core_name(row.get("business_name", "")) for row in source1],
        [core_name(row.get("business_name", "")) for row in targets],
        analyzer="char_wb",
        ngram_range=(2, 4),
    )
    address_scores = _strategy_scores(
        source1,
        targets,
        [normalize_address(row.get("business_address", "")) for row in source1],
        [normalize_address(row.get("business_address", "")) for row in targets],
        analyzer="word",
        ngram_range=(1, 2),
    )

    targets_by_country: Dict[str, List[int]] = defaultdict(list)
    for index, record in enumerate(targets):
        targets_by_country[(record.get("country", "") or "").strip().casefold()].append(index)

    candidates: Dict[Pair, dict] = {}
    for qi, query in enumerate(source1):
        country = (query.get("country", "") or "").strip().casefold()
        for strategy, scores_by_query, rank_offset in (
            ("name", name_scores, 0),
            ("address", address_scores, 2 * top_k),
        ):
            scores = scores_by_query[qi]
            ordered = _retrieval_order(scores, targets, targets_by_country, country, top_k)
            for rank, target_index in enumerate(ordered, 1):
                target_id = targets[target_index]["entity_id"]
                pair = (query["entity_id"], target_id)
                similarity = scores[target_index]
                metadata = candidates.setdefault(pair, {
                    "rank": rank + rank_offset,
                    "similarity": similarity,
                    "name_similarity": 0.0,
                    "address_similarity": 0.0,
                })
                metadata["rank"] = min(metadata["rank"], rank + rank_offset)
                metadata["similarity"] = max(metadata["similarity"], similarity)
                metadata[f"{strategy}_similarity"] = max(metadata[f"{strategy}_similarity"], similarity)
    return candidates


def blocking_recall_ceiling(candidates: Mapping[Pair, object], truth: Labels) -> float | None:
    """Return retrieved true links / all labeled true links, or None if none exist."""
    total = sum(len(target_ids) for target_ids in truth.values())
    if total == 0:
        return None
    retrieved = sum(
        1
        for source_id, target_ids in truth.items()
        for target_id in target_ids
        if (source_id, target_id) in candidates
    )
    return retrieved / total


def log_blocking_recall(candidates: Mapping[Pair, object], truth: Labels) -> float | None:
    recall = blocking_recall_ceiling(candidates, truth)
    total = sum(len(target_ids) for target_ids in truth.values())
    if recall is None:
        logger.info("Blocking Recall Ceiling: not defined (no labeled true matches)")
    else:
        retrieved = round(recall * total)
        logger.info("Blocking Recall Ceiling: %.2f%% (%d/%d true matches retrieved)", 100 * recall, retrieved, total)
    return recall


def generate_all_candidates(
    source1: Sequence[Record],
    source2: Sequence[Record],
    source3: Sequence[Record],
    top_k: int = 20,
    truth: Labels | None = None,
) -> Dict[Pair, dict]:
    """Retrieve from S2 and S3, then optionally report the labeled recall ceiling."""
    combined: Dict[Pair, dict] = {}
    for targets in (source2, source3):
        for pair, metadata in generate_candidates(source1, targets, top_k).items():
            previous = combined.get(pair)
            if previous is None:
                combined[pair] = dict(metadata)
                continue
            previous["rank"] = min(previous["rank"], metadata["rank"])
            previous["similarity"] = max(previous["similarity"], metadata["similarity"])
            previous["name_similarity"] = max(previous["name_similarity"], metadata["name_similarity"])
            previous["address_similarity"] = max(previous["address_similarity"], metadata["address_similarity"])
    if truth is not None:
        log_blocking_recall(combined, truth)
    return combined
