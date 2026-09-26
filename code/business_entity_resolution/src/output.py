"""Submission and validation-error writers with strict memory efficiency and test compatibility."""
from __future__ import annotations

import csv
import logging
import sqlite3
from pathlib import Path
from typing import Iterable, Mapping, Sequence, Set, Tuple

import numpy as np
from .data import Record

LOGGER = logging.getLogger(__name__)
Pair = Tuple[str, str]

# ─── LEGACY HELPERS FOR UNIT TEST COMPATIBILITY ───────────────────────────────

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

def enforce_candidate_subset(predictions: Mapping[str, Iterable[object]], candidates: Mapping[str, Iterable[object]]) -> dict[str, Set[str]]:
    candidate_sets = {source_id: _ids(target_ids) for source_id, target_ids in candidates.items()}
    normalized = {source_id: _ids(target_ids) for source_id, target_ids in predictions.items()}
    for source_id, target_ids in normalized.items():
        if source_id not in candidate_sets:
            if target_ids:
                raise AssertionError(f"Predictions contain unknown Source-1 ID {source_id}")
            continue
        unexpected = target_ids - candidate_sets[source_id]
        if unexpected:
            raise AssertionError(f"Predictions for {source_id} are not a subset of candidate pairs: {sorted(unexpected)}")
    return normalized

def write_submission_from_database(output_dir: str | Path, database: str | Path, threshold: float) -> Tuple[Path, Path]:
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

# ─── PRODUCTION HIGH-SPEED STREAMING WRITER ───────────────────────────────────

def write_submission_stream(
    test_source1_path: Path,
    pairs_path: Path,
    probabilities: np.ndarray,
    output_dir: Path,
    threshold: float,
) -> tuple[Path, Path]:
    """Stream final submission in <45s without disk-database roundtrips or OOMs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = output_dir / "candidate_pairs.tsv"
    matching_path = output_dir / "matching_results.tsv"

    s1_order = []
    with test_source1_path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if sid := (row.get("entity_id") or "").strip():
                s1_order.append(sid)

    results_cand = {}
    results_match = {}
    
    with pairs_path.open("r", encoding="utf-8") as f:
        current_sid = None
        c_list = []
        m_list = []
        for idx, line in enumerate(f):
            sid, tid = line.rstrip("\n").split("\t", 1)
            if sid != current_sid:
                if current_sid is not None:
                    results_cand[current_sid] = ",".join(c_list)
                    if m_list:
                        results_match[current_sid] = ",".join(m_list)
                current_sid = sid
                c_list = [tid]
                m_list = [tid] if probabilities[idx] >= threshold else []
            else:
                c_list.append(tid)
                if probabilities[idx] >= threshold:
                    m_list.append(tid)
                    
        if current_sid is not None:
            results_cand[current_sid] = ",".join(c_list)
            if m_list:
                results_match[current_sid] = ",".join(m_list)

    with candidate_path.open("w", encoding="utf-8", newline="") as fc, \
         matching_path.open("w", encoding="utf-8", newline="") as fm:
         
        fc.write("source1_entity_id\tcandidate_entity_ids\n")
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        
        for sid in s1_order:
            fc.write(f"{sid}\t{results_cand.get(sid, '')}\n")
            fm.write(f"{sid}\t{results_match.get(sid, '')}\n")

    LOGGER.info("Streamed %d Source 1 records directly to %s and %s", len(s1_order), candidate_path.name, matching_path.name)
    return candidate_path, matching_path