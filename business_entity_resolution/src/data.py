"""TSV loading and singleton-aware train/validation splitting."""
from __future__ import annotations

import csv
import random
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple

Record = Dict[str, str]
Labels = Dict[str, Set[str]]


def load_records(path: str | Path) -> List[Record]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        required = {"entity_id", "business_name", "business_address", "country"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain tab-separated columns: {', '.join(sorted(required))}")
        records = []
        for row in reader:
            records.append({key: (row.get(key) or "").strip() for key in required})
    return records


def load_ground_truth(path: str | Path) -> Labels:
    labels: Labels = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        required = {"source1_entity_id", "matched_entity_ids"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain source1_entity_id and matched_entity_ids columns")
        for row in reader:
            sid = (row.get("source1_entity_id") or "").strip()
            if sid:
                labels[sid] = {value.strip() for value in (row.get("matched_entity_ids") or "").split(",") if value.strip()}
    return labels


def split_source1(records: Sequence[Record], labels: Mapping[str, Set[str]], validation_fraction: float = 0.2, seed: int = 42) -> Tuple[List[Record], List[Record]]:
    """Stratify by singleton status and coarse match-count bucket."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    buckets: Dict[Tuple[bool, int], List[Record]] = {}
    for record in records:
        count = len(labels.get(record["entity_id"], set()))
        key = (count == 0, 0 if count == 0 else min(count, 3))
        buckets.setdefault(key, []).append(record)
    rng = random.Random(seed)
    train: List[Record] = []
    valid: List[Record] = []
    for bucket in buckets.values():
        bucket = list(bucket)
        rng.shuffle(bucket)
        n_valid = int(round(len(bucket) * validation_fraction))
        if len(bucket) > 1:
            n_valid = max(1, min(len(bucket) - 1, n_valid))
        elif validation_fraction >= 0.5:
            n_valid = 1
        valid.extend(bucket[:n_valid])
        train.extend(bucket[n_valid:])
    return train, valid


def split_country_holdout(
    records: Sequence[Record],
    train_country: str | None = None,
    validation_country: str | None = None,
) -> Tuple[List[Record], List[Record]]:
    """Split Source 1 by country for zero-shot validation.

    If countries are omitted, use the two most represented countries, with the
    larger group for training. Other countries are excluded from both partitions.
    """
    country_by_record = {
        index: (record.get("country", "") or "").strip().casefold()
        for index, record in enumerate(records)
    }
    counts = Counter(country for country in country_by_record.values() if country)
    if len(counts) < 2:
        raise ValueError("Country holdout requires Source 1 records from at least two countries")

    requested_train = (train_country or "").strip().casefold()
    requested_valid = (validation_country or "").strip().casefold()
    if requested_train and requested_valid and requested_train == requested_valid:
        raise ValueError("Training and validation countries must be different")
    if requested_train and requested_train not in counts:
        raise ValueError(f"No Source 1 records found for training country: {train_country}")
    if requested_valid and requested_valid not in counts:
        raise ValueError(f"No Source 1 records found for validation country: {validation_country}")

    ranked = sorted(counts, key=lambda country: (-counts[country], country))
    chosen_train = requested_train or next((c for c in ranked if c != requested_valid), "")
    chosen_valid = requested_valid or next((c for c in ranked if c != chosen_train), "")
    if not chosen_train or not chosen_valid:
        raise ValueError("Could not select distinct training and validation countries")

    train = [record for index, record in enumerate(records) if country_by_record[index] == chosen_train]
    valid = [record for index, record in enumerate(records) if country_by_record[index] == chosen_valid]
    if not train or not valid:
        raise ValueError("Country holdout produced an empty training or validation partition")
    return train, valid


def index_by_id(records: Iterable[Record]) -> Dict[str, Record]:
    return {record["entity_id"]: record for record in records}
