"""Precision-heavy, entity-level decisioning for business ER.

The competition metric is macro F0.5 over Source-1 entities, so threshold
selection must be group-aware.  The exact optimizer below performs an
O(N log N) probability sweep instead of evaluating every unique threshold
against every entity, which was unnecessarily expensive in the earlier version.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class DecisionPolicy:
    high_threshold: float = 0.5
    low_threshold: float = 0.5
    ambiguous_margin: float = 0.0
    second_match_delta: float = 0.0
    max_matches: int = 64
    min_absolute_score: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def f05_from_counts(tp: int, predicted: int, truth: int) -> float:
    if truth == 0:
        return 1.0 if predicted == 0 else 0.0
    if predicted == 0 or tp == 0:
        return 0.0
    precision = tp / predicted
    recall = tp / truth
    denominator = 0.25 * precision + recall
    return (1.25 * precision * recall) / denominator if denominator else 0.0


def optimize_grouped_threshold(
    source1_ids: Sequence[str],
    labels: np.ndarray,
    probabilities: np.ndarray,
    truth_counts: Mapping[str, int],
) -> tuple[float, float]:
    """Exact macro-F0.5 optimum over observed probabilities in O(N log N).

    All candidates are sorted once by score.  As the threshold moves downward,
    only the Source-1 groups touched by the newly included probability are
    recomputed.  This is exact for the grouped macro objective and avoids the
    previous O(number_of_thresholds × number_of_entities × candidates) loop.
    """
    probabilities = np.asarray(probabilities, dtype=np.float32)
    labels = np.asarray(labels, dtype=bool)
    source_ids = np.asarray(source1_ids, dtype=object)

    if len(source_ids) != len(probabilities) or len(labels) != len(probabilities):
        raise ValueError("source1_ids, labels and probabilities must have equal length")

    entities = sorted(set(str(sid) for sid in source_ids))
    if not entities:
        return 1.0, 0.0

    entity_index = {sid: idx for idx, sid in enumerate(entities)}
    s_indices = np.asarray([entity_index[str(sid)] for sid in source_ids], dtype=np.int32)
    truth_arr = np.asarray([int(truth_counts.get(sid, 0)) for sid in entities], dtype=np.int32)
    pred_counts = np.zeros(len(entities), dtype=np.int32)
    tp_counts = np.zeros(len(entities), dtype=np.int32)

    # Each entity starts with an empty prediction set.
    current_scores = np.asarray(
        [f05_from_counts(0, 0, int(truth)) for truth in truth_arr], dtype=np.float64
    )
    total_score = float(np.sum(current_scores, dtype=np.float64))
    best_score = total_score / len(entities)
    best_threshold = np.float32(np.nextafter(np.float32(1.0), np.float32(2.0)))

    if probabilities.size == 0:
        return float(best_threshold), float(best_score)

    order = np.argsort(-probabilities, kind="stable")
    sorted_probs = probabilities[order]
    sorted_groups = s_indices[order]
    sorted_labels = labels[order]

    i = 0
    n = len(order)
    while i < n:
        threshold = float(sorted_probs[i])
        j = i
        touched: list[int] = []
        touched_set: set[int] = set()

        while j < n and float(sorted_probs[j]) == threshold:
            group = int(sorted_groups[j])
            if group not in touched_set:
                # Remove the old contribution before updating this group.
                total_score -= float(current_scores[group])
                touched.append(group)
                touched_set.add(group)
            pred_counts[group] += 1
            if sorted_labels[j]:
                tp_counts[group] += 1
            j += 1

        for group in touched:
            current_scores[group] = f05_from_counts(
                int(tp_counts[group]),
                int(pred_counts[group]),
                int(truth_arr[group]),
            )
            total_score += float(current_scores[group])

        macro = total_score / len(entities)
        if macro > best_score or (macro == best_score and threshold > float(best_threshold)):
            best_score = float(macro)
            best_threshold = np.float32(threshold)

        i = j

    return float(best_threshold), float(best_score)


def choose_hysteresis_policy(
    base_threshold: float,
    *,
    ambiguous_margin: float = 0.03,
    second_match_delta: float = 0.04,
    max_matches: int = 64,
    min_absolute_score: float = 0.0,
) -> DecisionPolicy:
    """Construct a policy from already-validated parameters."""
    high = float(np.clip(base_threshold, 0.0, 1.0))
    low = float(np.clip(high - max(0.0, ambiguous_margin), 0.0, high))
    return DecisionPolicy(
        high_threshold=high,
        low_threshold=low,
        ambiguous_margin=float(max(0.0, ambiguous_margin)),
        second_match_delta=float(max(0.0, second_match_delta)),
        max_matches=max(1, int(max_matches)),
        min_absolute_score=float(np.clip(min_absolute_score, 0.0, 1.0)),
    )


def apply_entity_policy(
    candidates: Sequence[Mapping[str, float | int | str]],
    policy: DecisionPolicy,
) -> set[str]:
    """Apply conservative entity-level hysteresis to one Source-1 group."""
    if not candidates:
        return set()

    ordered = sorted(
        candidates,
        key=lambda row: (-float(row.get("probability", 0.0)), str(row.get("target_id", ""))),
    )
    accepted: list[Mapping[str, float | int | str]] = []

    for position, row in enumerate(ordered):
        if len(accepted) >= policy.max_matches:
            break

        probability = float(row.get("probability", 0.0))
        if probability < policy.low_threshold or probability < policy.min_absolute_score:
            break

        next_probability = (
            float(ordered[position + 1].get("probability", 0.0))
            if position + 1 < len(ordered)
            else 0.0
        )
        margin = probability - next_probability
        name_score = float(row.get("name_score", 0.0))
        address_score = float(row.get("address_score", 0.0))
        reciprocal = float(row.get("mutual_best", 0.0)) > 0.5
        bridge = float(row.get("opposite_source_bridge", 0.0))
        alias = float(row.get("learned_name_alias", 0.0))
        address_alias = float(row.get("learned_address_alias", 0.0))

        strong_pair = (
            (name_score >= 0.92 and address_score >= 0.82)
            or (name_score >= 0.95 and address_score >= 0.70)
            or (address_score >= 0.95 and name_score >= 0.70)
            or reciprocal
            or bridge >= 0.80
            or alias >= 0.80
            or address_alias >= 0.80
        )

        if probability < policy.high_threshold:
            if margin < policy.ambiguous_margin and not strong_pair:
                continue

        if accepted and probability < policy.high_threshold + policy.second_match_delta and not strong_pair:
            continue

        accepted.append(row)

    return {str(row["target_id"]) for row in accepted}


def policy_to_json_dict(policy: DecisionPolicy) -> dict[str, object]:
    return policy.to_dict()
