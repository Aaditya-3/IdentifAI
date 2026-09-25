"""Memory-bounded, conservative exact-key fallback for very large TSV inputs.

This script intentionally uses only the supplied files.  It is used when a
high-recall learned candidate generator cannot be completed within the bounded
blocking experiments.  Candidate pairs are partitioned on disk, so results can
be emitted without holding all target records or all pairs in memory.
"""
from __future__ import annotations

import argparse
import csv
import shutil
from collections import defaultdict
from pathlib import Path

import jellyfish

try:
    from .src.preprocessing import core_name, normalize_address, normalize_name
except ImportError:
    from src.preprocessing import core_name, normalize_address, normalize_name


MAX_SOURCE1_PER_SIGNATURE = 5
PARTITIONS = 128


def _add(index: dict[str, list[str] | None], key: str, source_id: str) -> None:
    if not key:
        return
    values = index.get(key)
    if values is None:
        if key not in index:
            index[key] = [source_id]
    elif len(values) < MAX_SOURCE1_PER_SIGNATURE:
        values.append(source_id)
    else:
        # Ambiguous signatures are deliberately not candidates in this
        # precision-oriented fallback.
        index[key] = None


def _values(index: dict[str, list[str] | None], key: str) -> list[str]:
    return index.get(key) or []


def _keys(row: dict[str, str]) -> tuple[str, str, str]:
    return (
        normalize_name(row.get("business_name", "")),
        core_name(row.get("business_name", "")),
        normalize_address(row.get("business_address", "")),
    )


def _phonetic(core: str) -> str:
    token = next((value for value in core.split() if len(value) >= 3), "")
    return jellyfish.soundex(token) if token else ""


def _prediction_flag(s1: tuple[str, str, str], target: tuple[str, str, str]) -> bool:
    name, core, address = s1
    target_name, target_core, target_address = target
    # Full normalized-name agreement is strong.  A suffix-stripped match is
    # accepted only when corroborated by an address or a sufficiently specific
    # (long) core name; this avoids merging common short names.
    return (
        bool(name and name == target_name)
        or bool(core and core == target_core and (address == target_address or len(core) >= 16))
        or bool(address and address == target_address and core and target_core and len(core) >= 10)
    )


def run(test_dir: Path, output_dir: Path, scratch_dir: Path) -> None:
    if scratch_dir.exists():
        shutil.rmtree(scratch_dir)
    scratch_dir.mkdir(parents=True)
    writers = [
        (scratch_dir / f"pairs-{index:03d}.tsv").open("w", encoding="utf-8", newline="")
        for index in range(PARTITIONS)
    ]
    name_index: dict[str, list[str] | None] = {}
    core_index: dict[str, list[str] | None] = {}
    address_index: dict[str, list[str] | None] = {}
    prefix_index: dict[str, list[str] | None] = {}
    phonetic_index: dict[str, list[str] | None] = {}
    source_keys: dict[str, tuple[str, str, str]] = {}

    with (test_dir / "test_source1.tsv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            source_id = row["entity_id"].strip()
            keys = _keys(row)
            source_keys[source_id] = keys
            _add(name_index, keys[0], source_id)
            _add(core_index, keys[1], source_id)
            _add(address_index, keys[2], source_id)
            country = row.get("country", "").strip().casefold()
            _add(prefix_index, f"{country}:{keys[1][:5]}", source_id)
            _add(phonetic_index, f"{country}:{_phonetic(keys[1])}", source_id)

    try:
        for source in (2, 3):
            with (test_dir / f"test_source{source}.tsv").open(encoding="utf-8-sig", newline="") as stream:
                for row in csv.DictReader(stream, delimiter="\t"):
                    target_id = row["entity_id"].strip()
                    target_keys = _keys(row)
                    candidates = set(_values(name_index, target_keys[0]))
                    candidates.update(_values(core_index, target_keys[1]))
                    candidates.update(_values(address_index, target_keys[2]))
                    country = row.get("country", "").strip().casefold()
                    candidates.update(_values(prefix_index, f"{country}:{target_keys[1][:5]}"))
                    candidates.update(_values(phonetic_index, f"{country}:{_phonetic(target_keys[1])}"))
                    for source_id in candidates:
                        flag = "1" if _prediction_flag(source_keys[source_id], target_keys) else "0"
                        writers[hash(source_id) % PARTITIONS].write(f"{source_id}\t{target_id}\t{flag}\n")
    finally:
        for writer in writers:
            writer.close()

    output_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = output_dir / "candidate_pairs.tsv"
    matching_path = output_dir / "matching_results.tsv"
    seen: set[str] = set()
    with candidates_path.open("w", encoding="utf-8", newline="") as candidate_out, matching_path.open("w", encoding="utf-8", newline="") as matching_out:
        candidate_out.write("source1_entity_id\tcandidate_entity_ids\n")
        matching_out.write("source1_entity_id\tmatched_entity_ids\n")
        for partition in range(PARTITIONS):
            grouped: dict[str, dict[str, bool]] = defaultdict(dict)
            with (scratch_dir / f"pairs-{partition:03d}.tsv").open(encoding="utf-8", newline="") as stream:
                for source_id, target_id, flag in csv.reader(stream, delimiter="\t"):
                    grouped[source_id][target_id] = grouped[source_id].get(target_id, False) or flag == "1"
            for source_id in sorted(grouped):
                values = grouped[source_id]
                candidate_ids = sorted(values)
                match_ids = [target_id for target_id in candidate_ids if values[target_id]]
                candidate_out.write(f"{source_id}\t{','.join(candidate_ids)}\n")
                matching_out.write(f"{source_id}\t{','.join(match_ids)}\n")
                seen.add(source_id)
        for source_id in source_keys:
            if source_id not in seen:
                candidate_out.write(f"{source_id}\t\n")
                matching_out.write(f"{source_id}\t\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scratch-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.test_dir, args.output_dir, args.scratch_dir)


if __name__ == "__main__":
    main()
