"""Competition macro F0.5 metric, including true singletons."""
from __future__ import annotations

from typing import Mapping, Set


def macro_f0_5(y_true: Mapping[str, Set[str]], y_pred: Mapping[str, Set[str]]) -> float:
    if not y_true:
        return 0.0
    scores = []
    for sid, true_ids in y_true.items():
        pred_ids = y_pred.get(sid, set())
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
        scores.append(1.25 * precision * recall / denom if denom else 0.0)
    return sum(scores) / len(scores)

