"""Disk-backed, bounded candidate retrieval for large entity-resolution data."""
from __future__ import annotations

import csv
import hashlib
import logging
import sqlite3
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import jellyfish
import numpy as np
from datasketch import MinHash

from .preprocessing import core_name, extract_address_numbers, normalize_address, normalize_name

LOGGER = logging.getLogger(__name__)


class BlockingStore:
    """A reproducible on-disk inverted index and bounded candidate store.

    SQLite is used as an embedded disk index: no source-target Cartesian product,
    corpus-wide TF-IDF fit, or all-target in-memory index is created.  Every key
    retains a deterministic bounded sample of its bucket instead of nulling an
    overloaded signature, and other keys remain available for the same record.
    """

    KEY_CAP = 160
    TOP_K = 30
    PRE_SCORE_K = 90
    MINHASH_PERMUTATIONS = 32
    MINHASH_BANDS = 8
    TOKEN_DF_LIMIT = 5_000

    def __init__(self, database: str | Path, top_k: int = TOP_K, key_cap: int = KEY_CAP):
        self.path = Path(database)
        self.top_k = max(1, int(top_k))
        self.key_cap = max(8, int(key_cap))
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute("PRAGMA temp_store=FILE")
        self.connection.execute("PRAGMA cache_size=-200000")
        self._rare_token_cache: dict[str, bool] = {}

    def close(self) -> None:
        self.connection.close()

    def reset(self) -> None:
        self.connection.executescript("""
            DROP TABLE IF EXISTS source1;
            DROP TABLE IF EXISTS targets;
            DROP TABLE IF EXISTS token_df;
            DROP TABLE IF EXISTS block_keys;
            DROP TABLE IF EXISTS bounded_keys;
            DROP TABLE IF EXISTS target_keys;
            DROP TABLE IF EXISTS raw_candidates;
            DROP TABLE IF EXISTS shortlist;
            DROP TABLE IF EXISTS final_candidates;
            DROP TABLE IF EXISTS truth;
        """)
        self.connection.executescript("""
            CREATE TABLE source1 (
                entity_id TEXT PRIMARY KEY, country TEXT NOT NULL, name TEXT NOT NULL,
                core TEXT NOT NULL, sorted_core TEXT NOT NULL, address TEXT NOT NULL,
                numbers TEXT NOT NULL, split INTEGER NOT NULL
            );
            CREATE TABLE targets (
                entity_id TEXT PRIMARY KEY, country TEXT NOT NULL, name TEXT NOT NULL,
                core TEXT NOT NULL, sorted_core TEXT NOT NULL, address TEXT NOT NULL,
                numbers TEXT NOT NULL
            );
            CREATE TABLE token_df (token TEXT PRIMARY KEY, frequency INTEGER NOT NULL);
            CREATE TABLE block_keys (key TEXT NOT NULL, entity_id TEXT NOT NULL, weight REAL NOT NULL);
            CREATE TABLE raw_candidates (
                source1_id TEXT NOT NULL, target_id TEXT NOT NULL, evidence REAL NOT NULL,
                PRIMARY KEY (source1_id, target_id)
            ) WITHOUT ROWID;
            CREATE TABLE truth (source1_id TEXT NOT NULL, target_id TEXT NOT NULL,
                PRIMARY KEY (source1_id, target_id)) WITHOUT ROWID;
        """)
        self.connection.commit()

    @staticmethod
    def _records(path: str | Path) -> Iterator[dict[str, str]]:
        with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
            yield from csv.DictReader(stream, delimiter="\t")

    @staticmethod
    def _derived(row: Mapping[str, str]) -> tuple[str, str, str, str, str, str]:
        country = (row.get("country") or "").strip().casefold()
        name = normalize_name(row.get("business_name", ""))
        core = core_name(row.get("business_name", ""))
        sorted_core = " ".join(sorted(core.split()))
        address = normalize_address(row.get("business_address", ""))
        return country, name, core, sorted_core, address, extract_address_numbers(address)

    @staticmethod
    def _stable_split(entity_id: str) -> int:
        return int.from_bytes(hashlib.blake2b(entity_id.encode("utf-8"), digest_size=2).digest(), "big") % 5

    @staticmethod
    def _grams(value: str) -> set[str]:
        padded = f"  {value}  "
        return {padded[index:index + 3] for index in range(max(0, len(padded) - 2))}

    def _lsh_keys(self, country: str, core: str) -> list[str]:
        grams = self._grams(core)
        if len(grams) < 3:
            return []
        signature = MinHash(num_perm=self.MINHASH_PERMUTATIONS, seed=17)
        for gram in grams:
            signature.update(gram.encode("utf-8"))
        values = signature.hashvalues
        band_size = self.MINHASH_PERMUTATIONS // self.MINHASH_BANDS
        return [
            f"lsh:{country}:{band}:{hashlib.blake2b(values[band * band_size:(band + 1) * band_size].tobytes(), digest_size=8).hexdigest()}"
            for band in range(self.MINHASH_BANDS)
        ]

    @staticmethod
    def _soundex(token: str) -> str:
        try:
            return jellyfish.soundex(token)
        except (TypeError, ValueError):
            return ""

    def _keys(self, country: str, name: str, core: str, sorted_core: str, address: str,
              numbers: str, rare_tokens: Iterable[str] = ()) -> list[tuple[str, float]]:
        prefix = core[:5]
        keys = [
            (f"full:{country}:{name}", 12.0) if name else None,
            (f"core:{country}:{core}", 10.0) if core else None,
            (f"sorted:{country}:{sorted_core}", 9.0) if sorted_core else None,
            (f"address:{country}:{address}", 7.0) if address else None,
            (f"prefix:{country}:{prefix}", 2.0) if len(prefix) == 5 else None,
        ]
        keys.extend((f"number:{country}:{number}", 3.0) for number in set(numbers.split()) if len(number) >= 3)
        keys.extend((f"phonetic:{country}:{self._soundex(token)}", 2.0)
                    for token in rare_tokens if len(token) >= 3 and self._soundex(token))
        keys.extend((key, 1.5) for key in self._lsh_keys(country, core))
        return [key for key in keys if key is not None]

    def _rare_tokens(self, core: str) -> Iterator[str]:
        """Read document-frequency decisions from disk with a bounded cache."""
        for token in set(core.split()):
            if len(token) < 3:
                continue
            rare = self._rare_token_cache.get(token)
            if rare is None:
                row = self.connection.execute(
                    "SELECT frequency <= ? FROM token_df WHERE token=?", (self.TOKEN_DF_LIMIT, token)
                ).fetchone()
                rare = bool(row and row[0])
                if len(self._rare_token_cache) >= 100_000:
                    self._rare_token_cache.clear()
                self._rare_token_cache[token] = rare
            if rare:
                yield token

    def build_source_index(self, source1_path: str | Path) -> int:
        """Count document frequencies, then persist Source 1 records and keys."""
        token_rows, token_count = [], 0
        for row in self._records(source1_path):
            token_rows.extend((token,) for token in set(core_name(row.get("business_name", "")).split()) if len(token) >= 3)
            if len(token_rows) >= 50_000:
                self.connection.executemany(
                    "INSERT INTO token_df(token, frequency) VALUES (?, 1) ON CONFLICT(token) DO UPDATE SET frequency=frequency+1",
                    token_rows,
                )
                token_count += len(token_rows); token_rows.clear(); self.connection.commit()
        if token_rows:
            self.connection.executemany(
                "INSERT INTO token_df(token, frequency) VALUES (?, 1) ON CONFLICT(token) DO UPDATE SET frequency=frequency+1",
                token_rows,
            )
            token_count += len(token_rows)
        self.connection.commit()
        retained = self.connection.execute("SELECT COUNT(*) FROM token_df WHERE frequency <= ?", (self.TOKEN_DF_LIMIT,)).fetchone()[0]
        total_tokens = self.connection.execute("SELECT COUNT(*) FROM token_df").fetchone()[0]
        LOGGER.info("Blocking token DF pruning retained %d/%d name tokens (limit=%d)", retained, total_tokens, self.TOKEN_DF_LIMIT)

        records, keys, count = [], [], 0
        for row in self._records(source1_path):
            entity_id = (row.get("entity_id") or "").strip()
            if not entity_id:
                continue
            country, name, core, sorted_core, address, numbers = self._derived(row)
            records.append((entity_id, country, name, core, sorted_core, address, numbers, self._stable_split(entity_id)))
            rare_tokens = self._rare_tokens(core)
            keys.extend((key, entity_id, weight) for key, weight in self._keys(country, name, core, sorted_core, address, numbers, rare_tokens))
            count += 1
            if count % 20_000 == 0:
                self.connection.executemany("INSERT INTO source1 VALUES (?, ?, ?, ?, ?, ?, ?, ?)", records)
                self.connection.executemany("INSERT INTO block_keys VALUES (?, ?, ?)", keys)
                records.clear(); keys.clear(); self.connection.commit()
        if records:
            self.connection.executemany("INSERT INTO source1 VALUES (?, ?, ?, ?, ?, ?, ?, ?)", records)
            self.connection.executemany("INSERT INTO block_keys VALUES (?, ?, ?)", keys)
        self.connection.execute("CREATE INDEX block_keys_key ON block_keys(key, entity_id)")
        self.connection.commit()
        self._bound_keys()
        return count

    def _bound_keys(self) -> None:
        """Retain a deterministic slice of overloaded buckets; never delete a key."""
        self.connection.execute("""
            CREATE TABLE bounded_keys AS
            SELECT key, entity_id, weight FROM (
                SELECT key, entity_id, weight,
                    ROW_NUMBER() OVER (PARTITION BY key ORDER BY entity_id) AS position
                FROM block_keys
            ) WHERE position <= ?
        """, (self.key_cap,))
        self.connection.execute("CREATE INDEX bounded_keys_key ON bounded_keys(key)")
        self.connection.execute("DROP TABLE block_keys")
        self.connection.commit()

    def add_truth(self, truth_path: str | Path) -> int:
        rows, count = [], 0
        with Path(truth_path).open("r", encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                source_id = (row.get("source1_entity_id") or "").strip()
                rows.extend((source_id, target_id.strip()) for target_id in (row.get("matched_entity_ids") or "").split(",") if target_id.strip())
                if len(rows) >= 50_000:
                    self.connection.executemany("INSERT OR IGNORE INTO truth VALUES (?, ?)", rows)
                    count += len(rows); rows.clear()
        if rows:
            self.connection.executemany("INSERT OR IGNORE INTO truth VALUES (?, ?)", rows); count += len(rows)
        self.connection.commit()
        return count

    def add_targets_and_retrieve(self, target_paths: Sequence[str | Path]) -> None:
        """Stream each target file, join its keys in SQLite, and then discard them."""
        for path in target_paths:
            self.connection.execute("CREATE TABLE target_keys (target_id TEXT NOT NULL, key TEXT NOT NULL, weight REAL NOT NULL)")
            target_rows, key_rows, count = [], [], 0
            for row in self._records(path):
                entity_id = (row.get("entity_id") or "").strip()
                if not entity_id:
                    continue
                country, name, core, sorted_core, address, numbers = self._derived(row)
                target_rows.append((entity_id, country, name, core, sorted_core, address, numbers))
                rare_tokens = self._rare_tokens(core)
                key_rows.extend((entity_id, key, weight) for key, weight in self._keys(country, name, core, sorted_core, address, numbers, rare_tokens))
                count += 1
                if count % 20_000 == 0:
                    self.connection.executemany("INSERT OR REPLACE INTO targets VALUES (?, ?, ?, ?, ?, ?, ?)", target_rows)
                    self.connection.executemany("INSERT INTO target_keys VALUES (?, ?, ?)", key_rows)
                    target_rows.clear(); key_rows.clear(); self.connection.commit()
            if target_rows:
                self.connection.executemany("INSERT OR REPLACE INTO targets VALUES (?, ?, ?, ?, ?, ?, ?)", target_rows)
                self.connection.executemany("INSERT INTO target_keys VALUES (?, ?, ?)", key_rows)
            self.connection.execute("CREATE INDEX target_keys_key ON target_keys(key)")
            self.connection.execute("""
                INSERT INTO raw_candidates(source1_id, target_id, evidence)
                SELECT b.entity_id, t.target_id, SUM(b.weight * t.weight)
                FROM target_keys t JOIN bounded_keys b ON b.key = t.key
                GROUP BY b.entity_id, t.target_id
                ON CONFLICT(source1_id, target_id) DO UPDATE SET evidence=evidence + excluded.evidence
            """)
            self.connection.execute("DROP TABLE target_keys")
            self.connection.commit()
            LOGGER.info("Retrieved blocking candidates from %s (%d targets)", Path(path).name, count)

    def finalize_candidates(self) -> int:
        """Use C-accelerated RapidFuzz scores after an evidence-only short list."""
        from rapidfuzz import fuzz, process
        self.connection.execute("""
            CREATE TABLE shortlist AS
            SELECT source1_id, target_id, evidence FROM (
                SELECT source1_id, target_id, evidence,
                    ROW_NUMBER() OVER (PARTITION BY source1_id ORDER BY evidence DESC, target_id) AS position
                FROM raw_candidates
            ) WHERE position <= ?
        """, (self.top_k * 3,))
        self.connection.execute("ALTER TABLE shortlist ADD COLUMN similarity REAL")
        reader = self.connection.execute("""
            SELECT c.source1_id, c.target_id, s.core, s.address, t.core, t.address, c.evidence
            FROM shortlist c JOIN source1 s ON s.entity_id=c.source1_id JOIN targets t ON t.entity_id=c.target_id
        """)
        update_rows = []
        while True:
            rows = reader.fetchmany(25_000)
            if not rows:
                break
            name_scores = process.cpdist([row[2] for row in rows], [row[4] for row in rows], scorer=fuzz.ratio, dtype=np.uint8, workers=-1)
            address_scores = process.cpdist([row[3] for row in rows], [row[5] for row in rows], scorer=fuzz.ratio, dtype=np.uint8, workers=-1)
            update_rows.extend((float(evidence) + float(name) / 20.0 + float(address) / 100.0, sid, tid)
                               for (sid, tid, *_unused, evidence), name, address in zip(rows, name_scores, address_scores))
            if len(update_rows) >= 100_000:
                self.connection.executemany("UPDATE shortlist SET similarity=? WHERE source1_id=? AND target_id=?", update_rows)
                update_rows.clear(); self.connection.commit()
        if update_rows:
            self.connection.executemany("UPDATE shortlist SET similarity=? WHERE source1_id=? AND target_id=?", update_rows)
        self.connection.execute("""
            CREATE TABLE final_candidates AS
            SELECT source1_id, target_id, evidence, similarity,
                   ROW_NUMBER() OVER (PARTITION BY source1_id ORDER BY similarity DESC, target_id) AS rank
            FROM shortlist
        """)
        self.connection.execute("DELETE FROM final_candidates WHERE rank > ?", (self.top_k,))
        self.connection.execute("CREATE UNIQUE INDEX final_pair ON final_candidates(source1_id, target_id)")
        self.connection.execute("DROP TABLE shortlist")
        self.connection.execute("DROP TABLE raw_candidates")
        self.connection.commit()
        return self.connection.execute("SELECT COUNT(*) FROM final_candidates").fetchone()[0]

    def recall_ceiling(self, where: str = "", parameters: Sequence[object] = ()) -> tuple[float, int, int]:
        predicate = f"WHERE {where}" if where else ""
        total = self.connection.execute(
            f"SELECT COUNT(*) FROM truth t JOIN source1 s ON s.entity_id=t.source1_id {predicate}", parameters
        ).fetchone()[0]
        retrieved = self.connection.execute(
            f"SELECT COUNT(*) FROM truth t JOIN final_candidates c ON c.source1_id=t.source1_id AND c.target_id=t.target_id JOIN source1 s ON s.entity_id=t.source1_id {predicate}", parameters
        ).fetchone()[0]
        return (retrieved / total if total else 1.0), retrieved, total

    def candidate_summary(self, where: str = "", parameters: Sequence[object] = ()) -> tuple[int, float]:
        predicate = f"WHERE {where}" if where else ""
        entity_count = self.connection.execute(f"SELECT COUNT(*) FROM source1 s {predicate}", parameters).fetchone()[0]
        candidate_count = self.connection.execute(
            f"SELECT COUNT(*) FROM final_candidates c JOIN source1 s ON s.entity_id=c.source1_id {predicate}", parameters
        ).fetchone()[0]
        return candidate_count, candidate_count / entity_count if entity_count else 0.0

    def feature_rows(self, where: str = "", parameters: Sequence[object] = ()) -> Iterator[tuple]:
        predicate = f"WHERE {where}" if where else ""
        cursor = self.connection.execute(f"""
            SELECT c.source1_id, c.target_id, s.country, s.name, s.core, s.sorted_core, s.address, s.numbers,
                   t.country, t.name, t.core, t.sorted_core, t.address, t.numbers,
                   c.evidence, c.similarity, c.rank,
                   CASE WHEN truth.target_id IS NULL THEN 0 ELSE 1 END
            FROM final_candidates c
            JOIN source1 s ON s.entity_id=c.source1_id
            JOIN targets t ON t.entity_id=c.target_id
            LEFT JOIN truth ON truth.source1_id=c.source1_id AND truth.target_id=c.target_id
            {predicate}
            ORDER BY c.source1_id, c.rank, c.target_id
        """, parameters)
        yield from cursor

    def feature_count(self, where: str = "", parameters: Sequence[object] = ()) -> int:
        predicate = f"WHERE {where}" if where else ""
        return self.connection.execute(
            f"SELECT COUNT(*) FROM final_candidates c JOIN source1 s ON s.entity_id=c.source1_id {predicate}", parameters
        ).fetchone()[0]

    def update_probabilities(self, pairs: Sequence[tuple[str, str]], probabilities: Sequence[float]) -> None:
        if len(pairs) != len(probabilities):
            raise ValueError("pairs and probabilities must have the same length")
        self.connection.execute("ALTER TABLE final_candidates ADD COLUMN probability REAL") if not self._has_probability() else None
        self.connection.executemany(
            "UPDATE final_candidates SET probability=? WHERE source1_id=? AND target_id=?",
            ((float(probability), source_id, target_id) for (source_id, target_id), probability in zip(pairs, probabilities)),
        )
        self.connection.commit()

    def _has_probability(self) -> bool:
        return any(row[1] == "probability" for row in self.connection.execute("PRAGMA table_info(final_candidates)"))
