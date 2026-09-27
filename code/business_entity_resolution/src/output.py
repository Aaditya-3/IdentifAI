"""Submission and validation-error writers with strict memory efficiency and test compatibility."""
from __future__ import annotations

import csv
import logging
import sqlite3
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence, Set, Tuple

import numpy as np
from .data import Record
from .features import FEATURE_NAMES, feature_batch
from .decision import DecisionPolicy, apply_entity_policy

LOGGER = logging.getLogger(__name__)
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
    """Legacy DB reader preserved exclusively so the older unit tests don't crash."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = output_dir / "candidate_pairs.tsv"
    matching_path = output_dir / "matching_results.tsv"
    
    connection = sqlite3.connect(database)
    
    # Check if probability column exists to prevent test crashes
    has_prob = any(row[1] == "probability" for row in connection.execute("PRAGMA table_info(final_candidates)"))
    prob_col = "c.probability" if has_prob else "0.0"
    
    cursor = connection.execute(f"""
        SELECT s.entity_id, c.target_id, COALESCE({prob_col}, 0.0)
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


def write_submission_from_store_streaming(
    store,
    model,
    output_dir: Path,
    threshold: float,
    batch_size: int = 25_000,
    threshold_by_country: Mapping[str, float] | None = None,
    unseen_country_threshold: float | None = None,
    variation_model=None,
    decision_policies_by_country: Mapping[str, DecisionPolicy] | None = None,
    unseen_decision_policy: DecisionPolicy | None = None,
) -> tuple[Path, Path]:
    """Run bounded test inference and write the exact required submission files.

    ``candidate_pairs.tsv`` is the exact final-candidate set in
    ``final_candidates``.  Matching decisions are made entity-by-entity after all
    candidates for an S1 have been scored, allowing the validated hysteresis
    policy to use probability margins and pair evidence without changing the
    candidate set itself.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = output_dir / "candidate_pairs.tsv"
    matching_path = output_dir / "matching_results.tsv"

    s1_cursor = iter(
        store.connection.execute(
            "SELECT entity_id, country FROM source1 ORDER BY entity_id"
        )
    )
    current = next(s1_cursor, None)
    current_sid = current[0] if current else None
    current_country = (current[1] or "").casefold() if current else ""

    current_candidates: list[str] = []
    current_rows_for_policy: list[dict[str, object]] = []
    current_seen: set[str] = set()
    written = 0
    consumed_pairs = 0

    idx_name = FEATURE_NAMES.index("name_ratio")
    idx_address = FEATURE_NAMES.index("address_ratio")
    idx_mutual = FEATURE_NAMES.index("mutual_best")
    idx_bridge = FEATURE_NAMES.index("opposite_source_bridge")
    idx_alias = FEATURE_NAMES.index("learned_name_alias")
    idx_address_alias = FEATURE_NAMES.index("learned_address_alias")
    idx_semantic = FEATURE_NAMES.index("semantic_similarity")
    idx_rerank = FEATURE_NAMES.index("cross_encoder_score")

    def _policy_for_country(country: str) -> DecisionPolicy:
        normalized = (country or "").casefold()
        if decision_policies_by_country:
            policy = decision_policies_by_country.get(normalized)
            if policy is not None:
                return policy
        if unseen_decision_policy is not None:
            return unseen_decision_policy
        active_threshold = threshold
        if threshold_by_country is not None:
            active_threshold = float(
                threshold_by_country.get(
                    normalized,
                    unseen_country_threshold
                    if unseen_country_threshold is not None
                    else threshold,
                )
            )
        return DecisionPolicy(
            high_threshold=active_threshold,
            low_threshold=active_threshold,
            ambiguous_margin=0.0,
            second_match_delta=0.0,
            max_matches=max(1, int(store.top_k)),
            min_absolute_score=0.0,
        )

    def _write_current(fc, fm) -> None:
        nonlocal written, current_candidates, current_rows_for_policy
        if current_sid is None:
            return

        policy = _policy_for_country(current_country)
        selected = apply_entity_policy(current_rows_for_policy, policy)
        ordered_matches = [
            row["target_id"]
            for row in sorted(
                current_rows_for_policy,
                key=lambda row: (
                    -float(row["probability"]),
                    str(row["target_id"]),
                ),
            )
            if str(row["target_id"]) in selected
        ]

        fc.write(f"{current_sid}\t{','.join(current_candidates)}\n")
        fm.write(f"{current_sid}\t{','.join(ordered_matches)}\n")
        written += 1
        current_candidates = []
        current_rows_for_policy = []
        current_seen.clear()

    def _advance_to(fc, fm, sid: str) -> None:
        nonlocal current_sid, current_country
        while current_sid is not None and current_sid < sid:
            _write_current(fc, fm)
            current = next(s1_cursor, None)
            current_sid = current[0] if current else None
            current_country = (current[1] or "").casefold() if current else ""
        if current_sid != sid:
            raise AssertionError(
                f"Candidate stream contains unknown/unordered Source-1 ID: {sid}"
            )

    cursor = store.feature_rows()
    with candidate_path.open("w", encoding="utf-8", newline="") as fc, \
         matching_path.open("w", encoding="utf-8", newline="") as fm:
        fc.write("source1_entity_id\tcandidate_entity_ids\n")
        fm.write("source1_entity_id\tmatched_entity_ids\n")

        while True:
            batch = []
            try:
                for _ in range(batch_size):
                    batch.append(next(cursor))
            except StopIteration:
                pass
            if not batch:
                break

            feature_matrix = feature_batch(
                [row[:-1] for row in batch],
                variation_model=variation_model,
                semantic_retriever=store.semantic_retriever,
            )
            probs = model.predict_proba(feature_matrix)
            if len(probs) != len(batch):
                raise AssertionError(
                    "Model probability count does not match feature batch size"
                )

            for row, prob, feature_vector in zip(batch, probs, feature_matrix):
                sid, tid = row[0], row[1]
                _advance_to(fc, fm, sid)

                if tid in current_seen:
                    raise AssertionError(
                        f"Duplicate candidate target {tid} for Source-1 {sid}"
                    )
                if tid.startswith("S1-"):
                    raise AssertionError(f"Invalid target Source-1 ID {tid}")

                current_seen.add(tid)
                current_candidates.append(tid)
                current_rows_for_policy.append(
                    {
                        "target_id": tid,
                        "probability": float(prob),
                        "name_score": float(feature_vector[idx_name]),
                        "address_score": float(feature_vector[idx_address]),
                        "mutual_best": float(feature_vector[idx_mutual]),
                        "opposite_source_bridge": float(feature_vector[idx_bridge]),
                        "learned_name_alias": float(feature_vector[idx_alias]),
                        "learned_address_alias": float(feature_vector[idx_address_alias]),
                        "semantic_score": float(feature_vector[idx_semantic]),
                        "cross_encoder_score": float(feature_vector[idx_rerank]),
                    }
                )
                consumed_pairs += 1

        while current_sid is not None:
            _write_current(fc, fm)
            current = next(s1_cursor, None)
            current_sid = current[0] if current else None
            current_country = (current[1] or "").casefold() if current else ""

    expected_entities = store.connection.execute(
        "SELECT COUNT(*) FROM source1"
    ).fetchone()[0]
    if written != expected_entities:
        raise AssertionError(
            f"Wrote {written} Source-1 rows, expected {expected_entities}"
        )

    LOGGER.info(
        "Streamed %d candidates and %d Source-1 rows directly to submission files",
        consumed_pairs,
        written,
    )
    return candidate_path, matching_path

def write_submission_stream(
    test_source1_path: Path,
    pairs_path: Path,
    probabilities: np.ndarray,
    output_dir: Path,
    threshold: float,
) -> tuple[Path, Path]:
    """Stream final submission without storing entire dataset in memory.
    
    Enforces the candidate-subset invariant: every predicted match must
    appear in the candidate set. Asserts no duplicate S1 IDs, no duplicate
    target IDs per S1, and every S1 appears exactly once.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = output_dir / "candidate_pairs.tsv"
    matching_path = output_dir / "matching_results.tsv"

    s1_order = []
    with test_source1_path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if sid := (row.get("entity_id") or "").strip():
                s1_order.append(sid)
    
    # Assert no duplicate S1 IDs
    assert len(s1_order) == len(set(s1_order)), \
        f"Duplicate S1 IDs found: {len(s1_order)} total vs {len(set(s1_order))} unique"
    
    # Sort to match the SQLite ORDER BY source1_id used when creating pairs_path
    s1_order.sort()

    s1_written = set()
    violations = 0

    with candidate_path.open("w", encoding="utf-8", newline="") as fc, \
         matching_path.open("w", encoding="utf-8", newline="") as fm:
         
        fc.write("source1_entity_id\tcandidate_entity_ids\n")
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        
        pairs_iter = pairs_path.open("r", encoding="utf-8")
        current_pair_line = pairs_iter.readline()
        prob_idx = 0
        
        for sid in s1_order:
            c_list = []
            m_list = []
            seen_tids = set()
            
            while current_pair_line:
                pair_sid, pair_tid = current_pair_line.rstrip("\n").split("\t", 1)
                if pair_sid < sid:
                    # Should not happen if everything is consistent, but skip if it does
                    current_pair_line = pairs_iter.readline()
                    prob_idx += 1
                    continue
                elif pair_sid == sid:
                    # Assert no duplicate target IDs within one S1
                    assert pair_tid not in seen_tids, \
                        f"Duplicate target {pair_tid} for S1 {sid}"
                    seen_tids.add(pair_tid)
                    
                    # Assert target is S2 or S3 (not S1)
                    assert not pair_tid.startswith("S1-"), \
                        f"Target {pair_tid} is an S1 entity, not S2/S3"
                    
                    c_list.append(pair_tid)
                    if probabilities[prob_idx] >= threshold:
                        m_list.append(pair_tid)
                    current_pair_line = pairs_iter.readline()
                    prob_idx += 1
                else:
                    # pair_sid > sid, which means no more targets for this sid
                    break
            
            # Candidate-subset invariant: every match must be a candidate
            for mid in m_list:
                if mid not in seen_tids:
                    violations += 1
                    
            fc.write(f"{sid}\t{','.join(c_list)}\n")
            fm.write(f"{sid}\t{','.join(m_list)}\n")
            
            assert sid not in s1_written, f"S1 {sid} written twice"
            s1_written.add(sid)
            
        pairs_iter.close()

    if violations > 0:
        raise AssertionError(f"{violations} predicted matches were not in candidate set")

    # Assert every S1 appeared
    assert len(s1_written) == len(s1_order), \
        f"Only {len(s1_written)}/{len(s1_order)} S1 entities were written"

    LOGGER.info("Streamed %d Source 1 records directly to %s and %s", len(s1_order), candidate_path.name, matching_path.name)
    return candidate_path, matching_path

def validate_output_against_store(
    store,
    candidate_path: str | Path,
    matching_path: str | Path,
) -> None:
    """Validate submission files in a single streaming pass.

    The official challenge validator remains the final submission gate. This local
    check deliberately avoids loading millions of S1/candidate rows into memory.
    """
    candidate_path = Path(candidate_path)
    matching_path = Path(matching_path)
    expected_cursor = iter(store.connection.execute("SELECT entity_id FROM source1 ORDER BY entity_id"))
    expected_count = store.connection.execute("SELECT COUNT(*) FROM source1").fetchone()[0]

    def _next_line(handle, expected_header: str, path: Path, first: bool = False):
        line = handle.readline()
        if first:
            if line.rstrip("\n") != expected_header:
                raise AssertionError(f"{path} has invalid header")
            return None
        if not line:
            return None
        s1, sep, values = line.rstrip("\n").partition("\t")
        if not sep or not s1:
            raise AssertionError(f"Malformed row in {path}: {line.rstrip()!r}")
        ids = [value for value in values.split(",") if value]
        if len(ids) != len(set(ids)):
            raise AssertionError(f"Duplicate target IDs in {path} for {s1}")
        return s1, ids

    candidate_total = 0
    match_total = 0
    previous_sid = None

    with candidate_path.open("r", encoding="utf-8", newline="") as fc, matching_path.open("r", encoding="utf-8", newline="") as fm:
        _next_line(fc, "source1_entity_id\tcandidate_entity_ids", candidate_path, first=True)
        _next_line(fm, "source1_entity_id\tmatched_entity_ids", matching_path, first=True)

        for row_index in range(expected_count):
            expected = next(expected_cursor, None)
            if expected is None:
                raise AssertionError("Output contains more Source-1 rows than the store")
            expected_sid = expected[0]
            candidate = _next_line(fc, "", candidate_path)
            match = _next_line(fm, "", matching_path)
            if candidate is None or match is None:
                raise AssertionError(f"Output ended early at expected Source-1 row {row_index + 1}")
            candidate_sid, candidate_ids = candidate
            match_sid, match_ids = match
            if candidate_sid != expected_sid or match_sid != expected_sid:
                raise AssertionError(
                    f"Source-1 row order mismatch at row {row_index + 1}: "
                    f"expected {expected_sid}, got {candidate_sid}/{match_sid}"
                )
            if previous_sid is not None and expected_sid <= previous_sid:
                raise AssertionError("Source-1 IDs are not strictly increasing")
            previous_sid = expected_sid
            candidate_set = set(candidate_ids)
            for target_id in candidate_ids:
                if not target_id.startswith(("S2-", "S3-")):
                    raise AssertionError(f"Illegal candidate target {target_id} for {expected_sid}")
            for target_id in match_ids:
                if not target_id.startswith(("S2-", "S3-")):
                    raise AssertionError(f"Illegal match target {target_id} for {expected_sid}")
            unexpected = set(match_ids) - candidate_set
            if unexpected:
                raise AssertionError(
                    f"Matches outside candidate set for {expected_sid}: {sorted(unexpected)[:5]}"
                )
            candidate_total += len(candidate_ids)
            match_total += len(match_ids)

        if _next_line(fc, "", candidate_path) is not None:
            raise AssertionError("candidate_pairs.tsv contains extra Source-1 rows")
        if _next_line(fm, "", matching_path) is not None:
            raise AssertionError("matching_results.tsv contains extra Source-1 rows")

    target_count = store.connection.execute("SELECT COUNT(*) FROM targets").fetchone()[0]
    if target_count <= 0:
        raise AssertionError("Target store is empty; cannot produce a valid submission")

    LOGGER.info(
        "Submission contract validation PASS: %d S1 rows, %d candidates, %d matches, %d targets",
        expected_count,
        candidate_total,
        match_total,
        target_count,
    )

