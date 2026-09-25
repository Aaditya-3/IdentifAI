"""Sparse character n-gram retrieval for high-recall candidate generation."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .data import Record
from .preprocessing import composite_text

Pair = Tuple[str, str]


def generate_candidates(source1: Sequence[Record], targets: Sequence[Record], top_k: int = 20) -> Dict[Pair, dict]:
    """Return pair metadata, taking same-country top K first plus global fallback."""
    if top_k < 1 or not source1 or not targets:
        return {}
    corpus = [composite_text(r) or " " for r in (*source1, *targets)]
    try:
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1, sublinear_tf=True, dtype=np.float32)
        matrix = normalize(vectorizer.fit_transform(corpus), norm="l2", copy=False)
    except ValueError:  # e.g. all records contain only punctuation/empty text
        return {}
    query_matrix = matrix[:len(source1)]
    target_matrix = matrix[len(source1):]
    target_transpose = target_matrix.T.tocsr()
    targets_by_country: Dict[str, List[int]] = defaultdict(list)
    for index, record in enumerate(targets):
        targets_by_country[record.get("country", "").strip().casefold()].append(index)
    out: Dict[Pair, dict] = {}
    for qi, query in enumerate(source1):
        row = (query_matrix[qi] @ target_transpose).tocsr()
        scores = {int(j): float(score) for j, score in zip(row.indices, row.data)}
        global_order = sorted(scores, key=lambda j: (-scores[j], targets[j]["entity_id"]))[:top_k]
        country = query.get("country", "").strip().casefold()
        local_order = sorted((j for j in targets_by_country.get(country, ()) if j in scores),
                             key=lambda j: (-scores[j], targets[j]["entity_id"]))[:top_k]
        # Same-country candidates rank ahead of global candidates; global retrieval
        # remains available for country spelling variation and unseen countries.
        ordered = list(dict.fromkeys(local_order + global_order))
        for rank, tj in enumerate(ordered, 1):
            pair = (query["entity_id"], targets[tj]["entity_id"])
            out[pair] = {"rank": rank, "similarity": scores[tj]}
    return out


def generate_all_candidates(source1: Sequence[Record], source2: Sequence[Record], source3: Sequence[Record], top_k: int = 20) -> Dict[Pair, dict]:
    """Retrieve independently from S2 and S3; rank is relative to each source."""
    return {**generate_candidates(source1, source2, top_k), **generate_candidates(source1, source3, top_k)}

