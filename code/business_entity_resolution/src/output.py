"""Submission and validation-error writers with strict memory efficiency."""
from __future__ import annotations

import csv
import logging
import sqlite3
from pathlib import Path
from typing import Sequence
import numpy as np

LOGGER = logging.getLogger(__name__)


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

    # 1. Read S1 ordered entities (zero memory bloat)
    s1_order = []
    with test_source1_path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if sid := (row.get("entity_id") or "").strip():
                s1_order.append(sid)

    # 2. Iterate lazily over sorted pairs_path, collapsing results into comma-separated strings
    # This prevents storing 50 million lists of strings in RAM.
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

    # 3. Stream direct output
    with candidate_path.open("w", encoding="utf-8", newline="") as fc, \
         matching_path.open("w", encoding="utf-8", newline="") as fm:
         
        fc.write("source1_entity_id\tcandidate_entity_ids\n")
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        
        for sid in s1_order:
            fc.write(f"{sid}\t{results_cand.get(sid, '')}\n")
            fm.write(f"{sid}\t{results_match.get(sid, '')}\n")

    LOGGER.info("Streamed %d Source 1 records directly to %s and %s", len(s1_order), candidate_path.name, matching_path.name)
    return candidate_path, matching_path