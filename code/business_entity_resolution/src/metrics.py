"""Exact competition macro F0.5 metric, scoring true singletons explicitly."""
from __future__ import annotations

from typing import Iterable, Mapping, Set


def macro_f0_5(
    y_true: Mapping[str, Set[str]],
    y_pred: Mapping[str, Set[str]],
    source1_ids: Iterable[str] | None = None,
) -> float:
    """Compute the unweighted macro F0.5 score across all Source 1 entities."""
    ids = list(source1_ids) if source1_ids is not None else list(y_true)
    if not ids:
        return 0.0
    scores = []
    for sid in ids:
        true_ids = y_true.get(sid, set()) or set()
        pred_ids = y_pred.get(sid, set()) or set()
        if not true_ids:
            scores.append(1.0 if not pred_ids else 0.0)
            continue
        if not pred_ids:
            scores.append(0.0)
            continue
        tp = len(true_ids & pred_ids)
        precision = tp / len(pred_ids)
        recall = tp / len(true_ids)
        denom = 0.25 * precision + recall
        scores.append((1.25 * precision * recall) / denom if denom > 0 else 0.0)
    return sum(scores) / len(scores)