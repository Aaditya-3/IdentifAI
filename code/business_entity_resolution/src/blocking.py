"""Disk-backed, high-recall bounded candidate retrieval for large entity-resolution data."""
from __future__ import annotations

import csv
import hashlib
import logging
import sqlite3
import time
import zlib
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import jellyfish
import numpy as np
from rapidfuzz import fuzz, process

from .preprocessing import (
    core_name,
    extract_acronym,
    extract_address_numbers,
    normalize_address,
    normalize_name,
)

LOGGER = logging.getLogger(__name__)


class BlockingStore:
    KEY_CAP = 160
    TOP_K = 30
    MINHASH_PERMUTATIONS = 32
    MINHASH_BANDS = 16  # r=2 rows/band -> 99.0% recall on 0.5 Jaccard overlap
    TOKEN_DF_LIMIT = 5_000
    
    _MINHASH_PRIME = np.uint64(4294967291)
    _MINHASH_A = np.asarray([2746317214, 478163328, 107420370, 3184935164, 1181241944, 1051802513, 958682847, 599310826, 3163119786, 440213416, 2906402158, 3181143732, 3831882065, 2342331445, 373399427, 2536146026, 1812140442, 136505588, 127978095, 402418011, 939042956, 999270937, 2170484434, 2585650757, 113971124, 2410529191, 854001194, 3075280818, 2791232394, 3012167821, 2340505847, 1801823909], dtype=np.uint64)
    _MINHASH_B = np.asarray([946785248, 1929338154, 2530876844, 1194819984, 3476477323, 3733616459, 27911967, 3259052811, 3460967357, 685731524, 2998485882, 1815115025, 1461364854, 1193448329, 667779376, 924765563, 4111198819, 3279182318, 1445662585, 438989805, 398340369, 1631775357, 415393687, 1541804686, 3639960595, 1477278577, 2592983555, 1136108454, 3466589567, 186618211, 3134174160, 1973214822], dtype=np.uint64)

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
        self._common_token_set: set[str] | None = None
        self.diagnostics: dict = {}

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
            DROP TABLE IF EXISTS scored_candidates;
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
            CREATE TABLE truth (
                source1_id TEXT NOT NULL, target_id TEXT NOT NULL,
                PRIMARY KEY (source1_id, target_id)
            ) WITHOUT ROWID;
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

    def _lsh_keys(self, text: str, prefix: str = "lsh") -> list[str]:
        grams = self._grams(text)
        if len(grams) < 3:
            return []
        hashes = np.fromiter((zlib.crc32(gram.encode("utf-8")) for gram in grams), dtype=np.uint64, count=len(grams))
        values = np.min((hashes[:, None] * self._MINHASH_A + self._MINHASH_B) % self._MINHASH_PRIME, axis=0)
        band_size = self.MINHASH_PERMUTATIONS // self.MINHASH_BANDS
        return [
            f"{prefix}:{band}:{hashlib.blake2b(values[band * band_size:(band + 1) * band_size].tobytes(), digest_size=8).hexdigest()}"
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
        acronym = extract_acronym(core)
        keys = [
            (f"full:{name}", 15.0) if name else None,
            (f"core:{core}", 12.0) if core else None,
            (f"sorted:{sorted_core}", 10.0) if sorted_core else None,
            (f"acronym:{acronym}", 8.0) if len(acronym) >= 2 else None,
            (f"address:{address}", 7.0) if address else None,
            (f"full_composite:{name}|{address}", 25.0) if name and address else None,
            (f"prefix:{prefix}", 2.0) if len(prefix) == 5 else None,
        ]
        rare_tokens = tuple(rare_tokens)
        
        # De-weight common street numbers; prioritize 5-6 digit postal codes
        for number in set(numbers.split()):
            weight = 4.0 if len(number) >= 5 else 0.5
            keys.append((f"number:{number}", weight))

        keys.extend((f"token:{token}", 6.0) for token in rare_tokens)
        keys.extend((f"phonetic:{self._soundex(token)}", 2.5) for token in rare_tokens if self._soundex(token))
        keys.extend((key, 2.0) for key in self._lsh_keys(core))
        keys.extend((key, 1.0) for key in self._lsh_keys(address, prefix="addr_lsh"))
        return [key for key in keys if key is not None]

    def _rare_tokens(self, core: str) -> Iterator[str]:
        for token in set(core.split()):
            if len(token) < 3:
                continue
            if self._common_token_set is not None:
                if token not in self._common_token_set:
                    yield token
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
        self.connection.commit()

        self._common_token_set = {
            row[0] for row in self.connection.execute(
                "SELECT token FROM token_df WHERE frequency > ?", (self.TOKEN_DF_LIMIT,)
            )
        }
        self._rare_token_cache.clear()

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
        self.connection.execute("CREATE INDEX IF NOT EXISTS tmp_bk_eid ON block_keys(entity_id)")
        self.connection.execute("""
            CREATE TABLE bounded_keys AS
            WITH entity_key_counts AS (
                SELECT entity_id, COUNT(*) AS key_count FROM block_keys GROUP BY entity_id
            )
            SELECT key, entity_id, weight FROM (
                SELECT bk.key, bk.entity_id, bk.weight,
                    ROW_NUMBER() OVER (
                        PARTITION BY bk.key
                        ORDER BY ekc.key_count ASC, bk.entity_id
                    ) AS position
                FROM block_keys bk
                JOIN entity_key_counts ekc ON ekc.entity_id = bk.entity_id
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
        self.connection.execute("CREATE TABLE all_target_keys (target_id TEXT NOT NULL, key TEXT NOT NULL, weight REAL NOT NULL)")
        valid_keys = {row[0] for row in self.connection.execute("SELECT DISTINCT key FROM bounded_keys")}
        for path in target_paths:
            target_rows, key_rows, count = [], [], 0
            for row in self._records(path):
                entity_id = (row.get("entity_id") or "").strip()
                if not entity_id:
                    continue
                country, name, core, sorted_core, address, numbers = self._derived(row)
                target_rows.append((entity_id, country, name, core, sorted_core, address, numbers))
                rare_tokens = self._rare_tokens(core)
                key_rows.extend((entity_id, key, weight) for key, weight in self._keys(country, name, core, sorted_core, address, numbers, rare_tokens) if key in valid_keys)
                count += 1
                if count % 20_000 == 0:
                    self.connection.executemany("INSERT OR REPLACE INTO targets VALUES (?, ?, ?, ?, ?, ?, ?)", target_rows)
                    self.connection.executemany("INSERT INTO all_target_keys VALUES (?, ?, ?)", key_rows)
                    target_rows.clear(); key_rows.clear(); self.connection.commit()
            if target_rows:
                self.connection.executemany("INSERT OR REPLACE INTO targets VALUES (?, ?, ?, ?, ?, ?, ?)", target_rows)
                self.connection.executemany("INSERT INTO all_target_keys VALUES (?, ?, ?)", key_rows)
            self.connection.commit()
            LOGGER.info("Retrieved target blocks from %s (%d targets)", Path(path).name, count)
            
        self.connection.execute("CREATE INDEX atk_key ON all_target_keys(key)")
        self.connection.execute("""
            CREATE TABLE bounded_target_keys AS
            WITH key_counts AS (
                SELECT key, COUNT(*) AS key_count FROM all_target_keys GROUP BY key
            )
            SELECT key, target_id, weight FROM (
                SELECT t.key, t.target_id, t.weight,
                    ROW_NUMBER() OVER (
                        PARTITION BY t.key
                        ORDER BY c.key_count ASC, t.target_id
                    ) AS position
                FROM all_target_keys t
                JOIN key_counts c ON c.key = t.key
            ) WHERE position <= ?
        """, (self.key_cap,))
        self.connection.execute("CREATE INDEX btk_key ON bounded_target_keys(key)")
        self.connection.execute("""
            INSERT INTO raw_candidates(source1_id, target_id, evidence)
            SELECT b.entity_id, t.target_id, SUM(b.weight * t.weight)
            FROM bounded_target_keys t JOIN bounded_keys b ON b.key = t.key
            GROUP BY b.entity_id, t.target_id
            ON CONFLICT(source1_id, target_id) DO UPDATE SET evidence=evidence + excluded.evidence
        """)
        self.connection.execute("DROP TABLE all_target_keys")
        self.connection.execute("DROP TABLE bounded_target_keys")
        self.connection.commit()

    def finalize_candidates(self) -> int:
        diagnostics: dict = {}
        has_truth = self._has_truth()

        if has_truth:
            raw_r, raw_hit, raw_tot = self._recall_on_table("raw_candidates")
            diagnostics["raw_blocking"] = {"recall": raw_r, "retrieved": raw_hit, "total": raw_tot}
        raw_stats = self._candidate_stats("raw_candidates")
        diagnostics["raw_stats"] = raw_stats

        self.connection.execute("""
            CREATE TABLE shortlist AS
            WITH raw_filtered AS (
                SELECT source1_id, target_id, evidence FROM (
                    SELECT source1_id, target_id, evidence,
                        ROW_NUMBER() OVER (
                            PARTITION BY source1_id ORDER BY evidence DESC, target_id
                        ) AS global_pos,
                        ROW_NUMBER() OVER (
                            PARTITION BY source1_id,
                                         CASE WHEN target_id LIKE 'S2-%' THEN 2 ELSE 3 END
                            ORDER BY evidence DESC, target_id
                        ) AS source_pos
                    FROM raw_candidates
                ) WHERE global_pos <= ? OR source_pos <= ?
            )
            SELECT source1_id, target_id,
                   COALESCE((evidence - MIN(evidence) OVER (PARTITION BY source1_id)) / 
                            NULLIF(MAX(evidence) OVER (PARTITION BY source1_id) - MIN(evidence) OVER (PARTITION BY source1_id), 0), 1.0) AS evidence
            FROM raw_filtered
        """, (self.top_k * 4, self.top_k))

        if has_truth:
            sl_r, sl_hit, sl_tot = self._recall_on_table("shortlist")
            diagnostics["shortlist"] = {"recall": sl_r, "retrieved": sl_hit, "total": sl_tot}
        diagnostics["shortlist_stats"] = self._candidate_stats("shortlist")

        self.connection.execute("""
            CREATE TABLE scored_candidates (
                source1_id TEXT NOT NULL, target_id TEXT NOT NULL,
                evidence REAL NOT NULL, similarity REAL NOT NULL
            );
        """)
        reader = self.connection.execute("""
            SELECT c.source1_id, c.target_id, s.core, s.address, t.core, t.address, c.evidence
            FROM shortlist c JOIN source1 s ON s.entity_id=c.source1_id JOIN targets t ON t.entity_id=c.target_id
        """)
        scored_batch = []
        while True:
            rows = reader.fetchmany(50_000)
            if not rows:
                break
            s_cores = [r[2] for r in rows]
            t_cores = [r[4] for r in rows]
            s_addrs = [r[3] for r in rows]
            t_addrs = [r[5] for r in rows]

            name_ratio = process.cpdist(s_cores, t_cores, scorer=fuzz.ratio, dtype=np.uint8, workers=-1)
            name_set = process.cpdist(s_cores, t_cores, scorer=fuzz.token_set_ratio, dtype=np.uint8, workers=-1)
            name_scores = np.maximum(name_ratio, name_set)
            addr_scores = process.cpdist(s_addrs, t_addrs, scorer=fuzz.token_set_ratio, dtype=np.uint8, workers=-1)

            for (sid, tid, *_, ev), n_s, a_s in zip(rows, name_scores, addr_scores):
                sim = float(ev) + (float(n_s) / 20.0) + (float(a_s) / 100.0)
                scored_batch.append((sid, tid, float(ev), sim))

            if len(scored_batch) >= 100_000:
                self.connection.executemany("INSERT INTO scored_candidates VALUES (?, ?, ?, ?)", scored_batch)
                scored_batch.clear(); self.connection.commit()
        if scored_batch:
            self.connection.executemany("INSERT INTO scored_candidates VALUES (?, ?, ?, ?)", scored_batch)
            self.connection.commit()

        self.connection.execute("""
            CREATE TABLE final_candidates (
                source1_id TEXT NOT NULL, target_id TEXT NOT NULL,
                evidence REAL NOT NULL, similarity REAL NOT NULL, rank INTEGER NOT NULL,
                PRIMARY KEY (source1_id, target_id)
            ) WITHOUT ROWID;
        """)
        self.connection.execute("""
            INSERT INTO final_candidates
            SELECT source1_id, target_id, evidence, similarity, rank FROM (
                SELECT source1_id, target_id, evidence, similarity,
                       ROW_NUMBER() OVER (PARTITION BY source1_id ORDER BY similarity DESC, target_id) AS rank
                FROM scored_candidates
            ) WHERE rank <= ?
        """, (self.top_k,))
        self.connection.execute("DROP TABLE scored_candidates")
        self.connection.execute("DROP TABLE shortlist")
        self.connection.execute("DROP TABLE raw_candidates")
        self.connection.commit()

        if has_truth:
            fin_r, fin_hit, fin_tot = self._recall_on_table("final_candidates")
            diagnostics["final"] = {"recall": fin_r, "retrieved": fin_hit, "total": fin_tot}
        fin_stats = self._candidate_stats("final_candidates")
        diagnostics["final_stats"] = fin_stats

        self.diagnostics = diagnostics
        LOGGER.info("Final blocking recall ceiling: %.4f%% (%d pairs, avg %.2f/entity)",
                    100 * diagnostics.get("final", {}).get("recall", 0),
                    fin_stats["total_pairs"], fin_stats["avg_candidates"])
        return fin_stats["total_pairs"]

    # ─── Legacy method restored to prevent old tests from crashing ───
    def update_probabilities(self, pairs: Sequence[tuple[str, str]], probabilities: Sequence[float]) -> None:
        if not any(row[1] == "probability" for row in self.connection.execute("PRAGMA table_info(final_candidates)")):
            self.connection.execute("ALTER TABLE final_candidates ADD COLUMN probability REAL")
        self.connection.executemany(
            "UPDATE final_candidates SET probability=? WHERE source1_id=? AND target_id=?",
            ((float(p), sid, tid) for (sid, tid), p in zip(pairs, probabilities))
        )
        self.connection.commit()

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
            f"SELECT COUNT(*) FROM final_candidates c JOIN source1 s ON s.entity_id=c.source1_id LEFT JOIN truth ON truth.source1_id=c.source1_id AND truth.target_id=c.target_id {predicate}", parameters
        ).fetchone()[0]

    def _has_truth(self) -> bool:
        return self.connection.execute("SELECT COUNT(*) FROM truth").fetchone()[0] > 0

    def _recall_on_table(self, table: str, where: str = "", parameters: Sequence[object] = ()) -> tuple[float, int, int]:
        predicate = f"WHERE {where}" if where else ""
        total = self.connection.execute(
            f"SELECT COUNT(*) FROM truth t JOIN source1 s ON s.entity_id=t.source1_id {predicate}", parameters
        ).fetchone()[0]
        retrieved = self.connection.execute(
            f"SELECT COUNT(*) FROM truth t JOIN {table} c ON c.source1_id=t.source1_id AND c.target_id=t.target_id "
            f"JOIN source1 s ON s.entity_id=t.source1_id {predicate}", parameters
        ).fetchone()[0]
        return (retrieved / total if total else 1.0), retrieved, total

    def _candidate_stats(self, table: str) -> dict:
        counts = [r[0] for r in self.connection.execute(f"SELECT COUNT(*) FROM {table} GROUP BY source1_id")]
        if not counts:
            return {"total_pairs": 0, "avg_candidates": 0.0, "median_candidates": 0, "max_candidates": 0}
        n = len(counts)
        counts.sort()
        return {
            "total_pairs": sum(counts),
            "avg_candidates": sum(counts) / n,
            "median_candidates": counts[n // 2],
            "max_candidates": counts[-1],
        }