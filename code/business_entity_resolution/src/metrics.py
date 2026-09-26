"""Single source of truth for the competition's macro F0.5 metric."""
from __future__ import annotations

from typing import Iterable, Mapping, Set


def f05_from_counts(tp: int, predicted: int, truth: int) -> float:
    """Return one Source-1 entity's F0.5 score, including singleton semantics."""
    if truth == 0:
        return 1.0 if predicted == 0 else 0.0
    if predicted == 0 or tp == 0:
        return 0.0
    precision = tp / predicted
    recall = tp / truth
    denominator = 0.25 * precision + recall
    return (1.25 * precision * recall) / denominator if denominator > 0 else 0.0


def macro_f0_5(
    y_true: Mapping[str, Set[str]],
    y_pred: Mapping[str, Set[str]],
    source1_ids: Iterable[str] | None = None,
) -> float:
    """Compute unweighted macro F0.5 over Source-1 entities."""
    ids = list(source1_ids) if source1_ids is not None else list(y_true)
    if not ids:
        return 0.0

    scores = []
    for sid in ids:
        true_ids = y_true.get(sid, set()) or set()
        pred_ids = y_pred.get(sid, set()) or set()
        scores.append(
            f05_from_counts(
                len(true_ids & pred_ids),
                len(pred_ids),
                len(true_ids),
            )
        )
    return sum(scores) / len(scores)


__all__ = ["f05_from_counts", "macro_f0_5"]
