"""Submission and validation-error writers with candidate-subset enforcement."""
from __future__ import annotations

import csv
import sqlite3
from pathlib import Path
from typing import Iterable, Mapping, Sequence, Set, Tuple

from .data import Record

Pair = Tuple[str, str]


def _clean_id(value: object) -> str:
    if value is None:
        return ""
    entity_id = str(value).strip()
    return "" if entity_id.casefold() in {"", "none", "nan"} else entity_id


def _ids(values: Iterable[object] | None) -> Set[str]:
    if values is None:
        return set()
    if isinstance(values, str):
        values = values.split(",")
    return {entity_id for value in values if (entity_id := _clean_id(value))}


def _source_order(source1: Sequence[Record]) -> list[str]:
    return list(dict.fromkeys(entity_id for row in source1 if (entity_id := _clean_id(row.get("entity_id")))))


def group_candidates(source1: Sequence[Record], candidate_pairs: Iterable[Pair]) -> dict[str, Set[str]]:
    grouped = {source_id: set() for source_id in _source_order(source1)}
    for pair in candidate_pairs:
        source_id, target_id = _clean_id(pair[0]), _clean_id(pair[1])
        if source_id in grouped and target_id:
            grouped[source_id].add(target_id)
    return grouped


def enforce_candidate_subset(
    predictions: Mapping[str, Iterable[object]],
    candidates: Mapping[str, Iterable[object]],
) -> dict[str, Set[str]]:
    candidate_sets = {source_id: _ids(target_ids) for source_id, target_ids in candidates.items()}
    normalized = {source_id: _ids(target_ids) for source_id, target_ids in predictions.items()}
    for source_id, target_ids in normalized.items():
        if source_id not in candidate_sets:
            if target_ids:
                raise AssertionError(f"Predictions contain unknown Source-1 ID {source_id}")
            continue
        unexpected = target_ids - candidate_sets[source_id]
        if unexpected:
            raise AssertionError(
                f"Predictions for {source_id} are not a subset of candidate pairs: {sorted(unexpected)}"
            )
    return normalized


def _write_grouped_tsv(
    path: Path,
    key_name: str,
    value_name: str,
    grouped: Mapping[str, Iterable[object]],
    order: Sequence[str],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
        writer.writerow([key_name, value_name])
        for source_id in order:
            target_ids = sorted(_ids(grouped.get(source_id)))
            writer.writerow([source_id, ",".join(target_ids) if target_ids else ""])


def write_submission(
    output_dir: str | Path,
    source1: Sequence[Record],
    candidate_pairs: Iterable[Pair],
    predictions: Mapping[str, Iterable[object]],
) -> Tuple[Path, Path]:
    """Write candidates and matches, including an empty row for every singleton."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    order = _source_order(source1)
    candidate_ids = group_candidates(source1, candidate_pairs)
    prediction_ids = enforce_candidate_subset(predictions, candidate_ids)
    candidate_path = output_dir / "candidate_pairs.tsv"
    matching_path = output_dir / "matching_results.tsv"
    _write_grouped_tsv(candidate_path, "source1_entity_id", "candidate_entity_ids", candidate_ids, order)
    _write_grouped_tsv(matching_path, "source1_entity_id", "matched_entity_ids", prediction_ids, order)
    return candidate_path, matching_path


def write_submission_from_database(output_dir: str | Path, database: str | Path, threshold: float) -> Tuple[Path, Path]:
    """Stream a submission from the canonical candidate store.

    The ordered left join writes exactly one row per Source 1 entity without
    materializing candidate lists in Python.  Predictions are selected only from
    ``final_candidates``, which makes the subset invariant structural.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = output_dir / "candidate_pairs.tsv"
    matching_path = output_dir / "matching_results.tsv"
    connection = sqlite3.connect(database)
    cursor = connection.execute("""
        SELECT s.entity_id, c.target_id, COALESCE(c.probability, 0.0)
        FROM source1 s LEFT JOIN final_candidates c ON c.source1_id=s.entity_id
        ORDER BY s.entity_id, c.target_id
    """)
    with candidate_path.open("w", encoding="utf-8", newline="") as candidates, matching_path.open("w", encoding="utf-8", newline="") as matches:
        candidates.write("source1_entity_id\tcandidate_entity_ids\n")
        matches.write("source1_entity_id\tmatched_entity_ids\n")
        current_id, candidate_ids, matched_ids = None, [], []
        for source_id, target_id, probability in cursor:
            if source_id != current_id:
                if current_id is not None:
                    candidates.write(f"{current_id}\t{','.join(candidate_ids)}\n")
                    matches.write(f"{current_id}\t{','.join(matched_ids)}\n")
                current_id, candidate_ids, matched_ids = source_id, [], []
            if target_id:
                candidate_ids.append(target_id)
                if probability >= threshold:
                    matched_ids.append(target_id)
        if current_id is not None:
            candidates.write(f"{current_id}\t{','.join(candidate_ids)}\n")
            matches.write(f"{current_id}\t{','.join(matched_ids)}\n")
    connection.close()
    return candidate_path, matching_path


def write_error_csv(
    path: str | Path,
    truth: Mapping[str, Set[str]],
    predictions: Mapping[str, Set[str]],
    candidate_pairs: Iterable[Pair],
    pairs: Sequence[Pair],
    probabilities: Sequence[float],
    threshold: float,
) -> Path:
    """Export pair-level false positives and false negatives for validation."""
    if len(pairs) != len(probabilities):
        raise ValueError("Probabilities and candidate pairs must have the same length")
    candidate_set = {(_clean_id(sid), _clean_id(tid)) for sid, tid in candidate_pairs}
    probability_by_pair = {
        (_clean_id(sid), _clean_id(tid)): float(probability)
        for (sid, tid), probability in zip(pairs, probabilities)
    }
    errors = []
    for source_id, true_values in truth.items():
        true_ids = _ids(true_values)
        predicted_ids = _ids(predictions.get(source_id))
        for target_id in sorted(predicted_ids - true_ids):
            pair = (_clean_id(source_id), target_id)
            errors.append(("false_positive", pair, pair in candidate_set))
        for target_id in sorted(true_ids - predicted_ids):
            pair = (_clean_id(source_id), target_id)
            errors.append(("false_negative", pair, pair in candidate_set))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow([
            "error_type", "source1_entity_id", "candidate_entity_id", "probability",
            "candidate_retrieved", "threshold", "error_stage",
        ])
        for error_type, (source_id, target_id), retrieved in errors:
            probability = probability_by_pair.get((source_id, target_id))
            if not retrieved:
                stage = "blocking_miss"
            elif error_type == "false_negative":
                stage = "below_threshold"
            else:
                stage = "model_false_positive"
            writer.writerow([
                error_type,
                source_id,
                target_id,
                "" if probability is None else probability,
                str(retrieved).lower(),
                threshold,
                stage,
            ])
    return path
