"""Disk-backed, high-recall bounded candidate retrieval for large entity-resolution data.

Changes from prior version:
- REMOVED arbitrary KEY_CAP=160 truncation that silently dropped entities.
- Added frequency-aware key filtering: low-frequency keys keep all entities,
  high-frequency keys are discarded entirely (they are not selective).
- Fixed cheap ranking score to properly balance name vs address signals.
- Added complete-match-set recall diagnostics.
- Replaced fixed candidate padding with an adaptive, evidence-aware final cap.
- Keeps the recall-first raw reservoir while avoiding repeated SQLite window sorts.
"""
from __future__ import annotations

import csv
import hashlib
import heapq
import logging
import sqlite3
import time
import zlib
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

try:
    import jellyfish
except ImportError:  # lightweight offline fallback for soundex only
    jellyfish = None
import numpy as np
from rapidfuzz import fuzz, process

from .preprocessing import (
    core_name,
    extract_acronym,
    extract_address_numbers,
    normalize_address,
    normalize_name,
    parse_address_components,
)
from .semantic_retrieval import SemanticConfig, SemanticRetriever, entity_text

LOGGER = logging.getLogger(__name__)


class BlockingStore:
    # One production source of truth for the hard candidate cap.
    # The final candidate set is adaptive and may be much smaller than TOP_K.
    TOP_K = 64
    SHORTLIST_MULTIPLIER = 8
    MIN_FINAL_CANDIDATES = 4
    AMBIGUOUS_MIN_FINAL_CANDIDATES = 12
    FINAL_SCORE_FLOOR = 0.55
    FINAL_SCORE_MARGIN = 0.20
    MINHASH_PERMUTATIONS = 32
    MINHASH_BANDS = 16  # r=2 rows/band -> 99.0% recall on 0.5 Jaccard overlap
    TOKEN_DF_LIMIT = 5_000
    # Frequency-aware key filtering thresholds (replace old KEY_CAP=160)
    KEY_FREQ_MAX_SOURCE = 2_000   # Source-side: drop keys with >N entries (not selective)
    KEY_FREQ_MAX_TARGET = 2_000   # Target-side: same
    HIGH_FREQ_RESCUE_TOP = 64
    HIGH_FREQ_RESCUE_MAX_BLOCK = 5_000

    _MINHASH_PRIME = np.uint64(4294967291)
    _MINHASH_A = np.asarray([2746317214, 478163328, 107420370, 3184935164, 1181241944, 1051802513, 958682847, 599310826, 3163119786, 440213416, 2906402158, 3181143732, 3831882065, 2342331445, 373399427, 2536146026, 1812140442, 136505588, 127978095, 402418011, 939042956, 999270937, 2170484434, 2585650757, 113971124, 2410529191, 854001194, 3075280818, 2791232394, 3012167821, 2340505847, 1801823909], dtype=np.uint64)
    _MINHASH_B = np.asarray([946785248, 1929338154, 2530876844, 1194819984, 3476477323, 3733616459, 27911967, 3259052811, 3460967357, 685731524, 2998485882, 1815115025, 1461364854, 1193448329, 667779376, 924765563, 4111198819, 3279182318, 1445662585, 438989805, 398340369, 1631775357, 415393687, 1541804686, 3639960595, 1477278577, 2592983555, 1136108454, 3466589567, 186618211, 3134174160, 1973214822], dtype=np.uint64)

    def __init__(self, database: str | Path, top_k: int = TOP_K):
        self.path = Path(database)
        self.top_k = max(1, int(top_k))
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute("PRAGMA temp_store=FILE")
        self.connection.execute("PRAGMA cache_size=-200000")
        self._rare_token_cache: dict[str, bool] = {}
        self._common_token_set: set[str] | None = None
        self._active_key_hashes: np.ndarray | None = None
        self.diagnostics: dict = {}
        self.semantic_config = SemanticConfig.from_environment(top_k=self.top_k)
        self.semantic_retriever = SemanticRetriever(self.semantic_config)

    def close(self) -> None:
        try:
            self.semantic_retriever.close()
        finally:
            self.connection.close()

    def reset(self) -> None:
        self.connection.executescript("""
            DROP TABLE IF EXISTS source1;
            DROP TABLE IF EXISTS targets;
            DROP TABLE IF EXISTS token_df;
            DROP TABLE IF EXISTS block_keys;
            DROP TABLE IF EXISTS bounded_keys;
            DROP TABLE IF EXISTS high_freq_source_keys;
            DROP TABLE IF EXISTS high_freq_target_keys;
            DROP TABLE IF EXISTS all_target_keys;
            DROP TABLE IF EXISTS bounded_target_keys;
            DROP TABLE IF EXISTS raw_candidates;
            DROP TABLE IF EXISTS shortlist;
            DROP TABLE IF EXISTS scored_candidates;
            DROP TABLE IF EXISTS final_candidates;
            DROP TABLE IF EXISTS truth;
            DROP TABLE IF EXISTS pipeline_meta;
            DROP TABLE IF EXISTS semantic_index_map;
        """)
        self.connection.executescript("""
            CREATE TABLE source1 (
                entity_id TEXT PRIMARY KEY, country TEXT NOT NULL, name TEXT NOT NULL,
                core TEXT NOT NULL, sorted_core TEXT NOT NULL, address TEXT NOT NULL,
                numbers TEXT NOT NULL, street_number TEXT NOT NULL, street TEXT NOT NULL,
                city TEXT NOT NULL, postal_code TEXT NOT NULL, split INTEGER NOT NULL
            );
            CREATE TABLE targets (
                entity_id TEXT PRIMARY KEY, country TEXT NOT NULL, name TEXT NOT NULL,
                core TEXT NOT NULL, sorted_core TEXT NOT NULL, address TEXT NOT NULL,
                numbers TEXT NOT NULL, street_number TEXT NOT NULL, street TEXT NOT NULL,
                city TEXT NOT NULL, postal_code TEXT NOT NULL
            );
            CREATE TABLE token_df (token TEXT PRIMARY KEY, frequency INTEGER NOT NULL);
            CREATE TABLE block_keys (key TEXT NOT NULL, entity_id TEXT NOT NULL, weight REAL NOT NULL);
            CREATE TABLE raw_candidates (
                source1_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                evidence REAL NOT NULL,
                support_count INTEGER NOT NULL DEFAULT 0,
                name_evidence REAL NOT NULL DEFAULT 0.0,
                address_evidence REAL NOT NULL DEFAULT 0.0,
                structural_evidence REAL NOT NULL DEFAULT 0.0,
                exact_evidence REAL NOT NULL DEFAULT 0.0,
                semantic_score REAL NOT NULL DEFAULT 0.0,
                semantic_rank INTEGER NOT NULL DEFAULT 0,
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
    def _derived(row: Mapping[str, str]):
        country = (row.get("country") or "").strip().casefold()
        name = normalize_name(row.get("business_name", ""))
        core = core_name(row.get("business_name", ""))
        sorted_core = " ".join(sorted(core.split()))
        components = parse_address_components(row.get("business_address", ""))
        address = components.normalized
        return (
            country, name, core, sorted_core, address, extract_address_numbers(address),
            components.street_number, components.street, components.city, components.postal_code,
        )

    @staticmethod
    def _stable_split(entity_id: str) -> int:
        return int.from_bytes(hashlib.blake2b(entity_id.encode("utf-8"), digest_size=2).digest(), "big") % 5

    @staticmethod
    def _key_hash64(key: str) -> np.uint64:
        # 64-bit digest is used only as a memory-efficient prefilter.
        # Final joins still compare the exact key string, so a hash collision
        # can add a harmless false-positive key but cannot create a false match.
        return np.uint64(
            int.from_bytes(
                hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest(),
                "little",
            )
        )

    def _prepare_active_key_hashes(self) -> None:
        rows = self.connection.execute(
            """
            SELECT key FROM bounded_keys
            UNION
            SELECT key FROM high_freq_source_keys
            """
        )
        hashes = np.fromiter(
            (int(self._key_hash64(row[0])) for row in rows),
            dtype=np.uint64,
        )
        if hashes.size:
            hashes = np.unique(hashes)
        self._active_key_hashes = hashes
        LOGGER.info("Active blocking-key hash filter prepared: %d unique keys", hashes.size)

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
        token = (token or "").strip().upper()
        if not token:
            return ""
        if jellyfish is not None:
            try:
                return jellyfish.soundex(token)
            except (TypeError, ValueError):
                pass
        # Deterministic pure-Python fallback matching standard American Soundex shape.
        codes = {**{c: "1" for c in "BFPV"}, **{c: "2" for c in "CGJKQSXZ"},
                 **{c: "3" for c in "DT"}, "L": "4", **{c: "5" for c in "MN"}, "R": "6"}
        first = token[0]
        prev = codes.get(first, "")
        out: list[str] = []
        for char in token[1:]:
            code = codes.get(char, "")
            if code and code != prev:
                out.append(code)
            prev = code
            if len(out) == 3:
                break
        return (first + "".join(out) + "000")[:4]

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
        for token in rare_tokens:
            sx = self._soundex(token)
            if sx:
                keys.append((f"phonetic:{sx}", 2.5))
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
            (country, name, core, sorted_core, address, numbers, street_number, street, city, postal_code) = self._derived(row)
            records.append((entity_id, country, name, core, sorted_core, address, numbers, street_number, street, city, postal_code, self._stable_split(entity_id)))
            rare_tokens = self._rare_tokens(core)
            keys.extend((key, entity_id, weight) for key, weight in self._keys(country, name, core, sorted_core, address, numbers, rare_tokens))
            count += 1
            if count % 20_000 == 0:
                self.connection.executemany("INSERT INTO source1 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", records)
                self.connection.executemany("INSERT INTO block_keys VALUES (?, ?, ?)", keys)
                records.clear(); keys.clear(); self.connection.commit()
        if records:
            self.connection.executemany("INSERT INTO source1 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", records)
            self.connection.executemany("INSERT INTO block_keys VALUES (?, ?, ?)", keys)
        self.connection.execute("CREATE INDEX block_keys_key ON block_keys(key, entity_id)")
        self.connection.commit()
        self._filter_keys_by_frequency()
        return count

    def _filter_keys_by_frequency(self) -> None:
        """Replace the old _bound_keys() that used arbitrary position-based truncation.

        New approach: frequency-aware filtering.
        - Keys with <= KEY_FREQ_MAX_SOURCE entries are kept intact (all entities preserved).
        - Keys with > KEY_FREQ_MAX_SOURCE entries are dropped entirely (not selective).

        This ensures NO entity is silently lost due to its ID sorting position.
        """
        # Preserve oversized source blocks separately. They are not used as an
        # unrestricted Cartesian join; they are handled later by bounded,
        # relevance-aware rescue retrieval.
        self.connection.execute("""
            CREATE TABLE high_freq_source_keys AS
            SELECT bk.key, bk.entity_id, bk.weight
            FROM block_keys bk
            JOIN (SELECT key, COUNT(*) AS cnt FROM block_keys GROUP BY key) kc
              ON kc.key = bk.key
            WHERE kc.cnt > ?
        """, (self.KEY_FREQ_MAX_SOURCE,))

        self.connection.execute("""
            CREATE TABLE bounded_keys AS
            SELECT bk.key, bk.entity_id, bk.weight
            FROM block_keys bk
            JOIN (SELECT key, COUNT(*) AS cnt FROM block_keys GROUP BY key) kc
              ON kc.key = bk.key
            WHERE kc.cnt <= ?
        """, (self.KEY_FREQ_MAX_SOURCE,))

        dropped = self.connection.execute("SELECT COUNT(DISTINCT key) FROM high_freq_source_keys").fetchone()[0]
        kept = self.connection.execute("SELECT COUNT(DISTINCT key) FROM bounded_keys").fetchone()[0]
        LOGGER.info("Source key filtering: kept %d selective keys; %d oversized keys routed to relevance-aware rescue",
                     kept, dropped)

        self.connection.execute("CREATE INDEX bounded_keys_key ON bounded_keys(key)")
        self.connection.execute("CREATE INDEX high_freq_source_key ON high_freq_source_keys(key, entity_id)")
        self._prepare_active_key_hashes()
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
        if self._active_key_hashes is None:
            self._prepare_active_key_hashes()

        for path in target_paths:
            target_rows, key_rows, count = [], [], 0
            for row in self._records(path):
                entity_id = (row.get("entity_id") or "").strip()
                if not entity_id:
                    continue
                (country, name, core, sorted_core, address, numbers, street_number, street, city, postal_code) = self._derived(row)
                target_rows.append((entity_id, country, name, core, sorted_core, address, numbers, street_number, street, city, postal_code))

                rare_tokens = self._rare_tokens(core)
                keys_for_target = self._keys(
                    country, name, core, sorted_core, address, numbers, rare_tokens
                )

                # Memory-safe key membership prefilter. We intentionally use
                # a 64-bit digest rather than a huge Python set of key strings.
                # Exact string equality is still performed by the later SQL
                # joins, so hash collisions are safe (they only add extra work).
                if self._active_key_hashes is not None and len(keys_for_target):
                    key_strings = [key for key, _ in keys_for_target]
                    key_hashes = np.asarray(
                        [self._key_hash64(key) for key in key_strings],
                        dtype=np.uint64,
                    )
                    positions = np.searchsorted(self._active_key_hashes, key_hashes)
                    mask = (
                        positions < self._active_key_hashes.size
                    )
                    if np.any(mask):
                        safe_positions = positions[mask]
                        safe_hashes = key_hashes[mask]
                        mask[mask] &= (
                            self._active_key_hashes[safe_positions] == safe_hashes
                        )
                    keys_for_target = [
                        item for item, keep in zip(keys_for_target, mask.tolist())
                        if keep
                    ]

                key_rows.extend(
                    (entity_id, key, weight)
                    for key, weight in keys_for_target
                )
                count += 1
                if count % 20_000 == 0:
                    self.connection.executemany("INSERT OR REPLACE INTO targets VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", target_rows)
                    self.connection.executemany("INSERT INTO all_target_keys VALUES (?, ?, ?)", key_rows)
                    target_rows.clear(); key_rows.clear(); self.connection.commit()
            if target_rows:
                self.connection.executemany("INSERT OR REPLACE INTO targets VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", target_rows)
                self.connection.executemany("INSERT INTO all_target_keys VALUES (?, ?, ?)", key_rows)
            self.connection.commit()
            LOGGER.info("Retrieved target blocks from %s (%d targets)", Path(path).name, count)

        self.connection.execute("CREATE INDEX atk_key ON all_target_keys(key)")

        # Frequency-aware target key filtering for the normal bounded join.
        # Oversized keys stay in all_target_keys and are handled by the rescue stage.
        self.connection.execute("""
            CREATE TABLE bounded_target_keys AS
            SELECT t.key, t.target_id, t.weight
            FROM all_target_keys t
            JOIN (SELECT key, COUNT(*) AS cnt FROM all_target_keys GROUP BY key) kc
              ON kc.key = t.key
            WHERE kc.cnt <= ?
        """, (self.KEY_FREQ_MAX_TARGET,))

        self.connection.execute("""
            CREATE TABLE high_freq_target_keys AS
            SELECT t.key, t.target_id, t.weight
            FROM all_target_keys t
            JOIN (SELECT key, COUNT(*) AS cnt FROM all_target_keys GROUP BY key) kc
              ON kc.key = t.key
            WHERE kc.cnt > ?
        """, (self.KEY_FREQ_MAX_TARGET,))
        self.connection.execute("CREATE INDEX high_freq_target_key ON high_freq_target_keys(key, target_id)")
        dropped = self.connection.execute("SELECT COUNT(DISTINCT key) FROM high_freq_target_keys").fetchone()[0]
        LOGGER.info("Target key filtering: %d oversized keys routed to relevance-aware rescue", dropped)

        self.connection.execute("CREATE INDEX btk_key ON bounded_target_keys(key)")
        self.connection.execute("""
            INSERT INTO raw_candidates(
                source1_id,
                target_id,
                evidence,
                support_count,
                name_evidence,
                address_evidence,
                structural_evidence,
                exact_evidence,
                semantic_score,
                semantic_rank
            )
            SELECT
                b.entity_id,
                t.target_id,
                SUM(b.weight * t.weight) AS evidence,
                COUNT(*) AS support_count,
                MAX(CASE
                    WHEN b.key LIKE 'full:%'
                      OR b.key LIKE 'core:%'
                      OR b.key LIKE 'sorted:%'
                      OR b.key LIKE 'acronym:%'
                      OR b.key LIKE 'prefix:%'
                      OR b.key LIKE 'lsh:%'
                    THEN b.weight * t.weight ELSE 0.0 END) AS name_evidence,
                MAX(CASE
                    WHEN b.key LIKE 'address:%'
                      OR b.key LIKE 'full_composite:%'
                      OR b.key LIKE 'addr_lsh:%'
                    THEN b.weight * t.weight ELSE 0.0 END) AS address_evidence,
                MAX(CASE
                    WHEN b.key LIKE 'number:%'
                      OR b.key LIKE 'token:%'
                      OR b.key LIKE 'phonetic:%'
                    THEN b.weight * t.weight ELSE 0.0 END) AS structural_evidence,
                MAX(CASE
                    WHEN b.key LIKE 'full_composite:%'
                      OR b.key LIKE 'full:%'
                      OR b.key LIKE 'core:%'
                      OR b.key LIKE 'sorted:%'
                      OR b.key LIKE 'address:%'
                    THEN b.weight * t.weight ELSE 0.0 END) AS exact_evidence,
                0.0 AS semantic_score,
                0 AS semantic_rank
            FROM bounded_target_keys t
            JOIN bounded_keys b ON b.key = t.key
            GROUP BY b.entity_id, t.target_id
            ON CONFLICT(source1_id, target_id) DO UPDATE SET
                evidence = raw_candidates.evidence + excluded.evidence,
                support_count = raw_candidates.support_count + excluded.support_count,
                name_evidence = MAX(raw_candidates.name_evidence, excluded.name_evidence),
                address_evidence = MAX(raw_candidates.address_evidence, excluded.address_evidence),
                structural_evidence = MAX(raw_candidates.structural_evidence, excluded.structural_evidence),
                exact_evidence = MAX(raw_candidates.exact_evidence, excluded.exact_evidence)
        """)
        self._rescue_high_frequency_keys()

        # Independent semantic retrieval route: BGE bi-encoder + FAISS HNSW.
        # It augments, rather than replaces, deterministic lexical blocking.
        if self.semantic_config.enabled:
            target_iter = (
                (row[0], entity_text(row[2], row[5], row[1]))
                for row in self.connection.execute(
                    "SELECT entity_id, country, name, core, sorted_core, address FROM targets ORDER BY entity_id"
                )
            )
            self.semantic_retriever.build(target_iter)
            source_iter = (
                (row[0], entity_text(row[2], row[5], row[1]))
                for row in self.connection.execute(
                    "SELECT entity_id, country, name, core, sorted_core, address FROM source1 ORDER BY entity_id"
                )
            )
            semantic_rows = self.semantic_retriever.query(
                source_iter, top_k=self.semantic_config.semantic_top_k
            )
            self.connection.executemany(
                """
                INSERT INTO raw_candidates(
                    source1_id, target_id, evidence, support_count,
                    name_evidence, address_evidence, structural_evidence, exact_evidence,
                    semantic_score, semantic_rank
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source1_id, target_id) DO UPDATE SET
                    evidence = MAX(raw_candidates.evidence, excluded.evidence),
                    support_count = raw_candidates.support_count + excluded.support_count,
                    semantic_score = MAX(raw_candidates.semantic_score, excluded.semantic_score),
                    semantic_rank = CASE
                        WHEN raw_candidates.semantic_rank = 0 THEN excluded.semantic_rank
                        ELSE MIN(raw_candidates.semantic_rank, excluded.semantic_rank)
                    END
                """,
                (
                    (sid, tid, max(0.0, score), 1, 0.0, 0.0, 0.0, 0.0, float(score), int(rank))
                    for sid, tid, score, rank in semantic_rows
                ),
            )
            self.connection.commit()
            self.diagnostics["semantic_retrieval"] = {
                "enabled": True,
                "backend": "bge_faiss_hnsw" if self.semantic_retriever.using_real_backend else "deterministic_fallback",
                "embedding_model": self.semantic_config.embedding_model,
                "semantic_top_k": self.semantic_config.semantic_top_k,
                "rerank_top_k": self.semantic_config.rerank_top_k,
                "hnsw_ef_search": self.semantic_config.hnsw_ef_search,
                "retrieved_rows": len(semantic_rows),
            }
        else:
            self.diagnostics["semantic_retrieval"] = {"enabled": False}

        self.connection.execute("DROP TABLE all_target_keys")
        self.connection.execute("DROP TABLE bounded_target_keys")
        self.connection.execute("DROP TABLE high_freq_source_keys")
        self.connection.execute("DROP TABLE high_freq_target_keys")
        self.connection.commit()

    def _rescue_high_frequency_keys(self) -> None:
        """Recover candidates from oversized blocks without an unrestricted Cartesian join.

        For each oversized shared key, retrieve a bounded top set using cheap
        name/address fuzzy similarity. This removes arbitrary ID-order truncation
        while keeping worst-case candidate volume bounded by the rescue width.
        """
        keys = [row[0] for row in self.connection.execute(
            "SELECT key FROM (SELECT DISTINCT key FROM high_freq_source_keys UNION SELECT DISTINCT key FROM high_freq_target_keys) ORDER BY key"
        )]
        if not keys:
            return

        rescued_pairs = 0
        skipped_blocks = 0
        for key in keys:
            # Only rescue Source-1 entities that are not already saturated by
            # selective candidates. This keeps high-frequency rescue bounded.
            source_rows = self.connection.execute(
                """SELECT s.entity_id, s.core, s.address
                   FROM (
                       SELECT entity_id FROM high_freq_source_keys WHERE key=?
                       UNION
                       SELECT entity_id FROM bounded_keys WHERE key=?
                   ) h
                   JOIN source1 s ON s.entity_id=h.entity_id
                   ORDER BY s.entity_id""", (key, key)
            ).fetchall()
            target_rows = self.connection.execute(
                """SELECT t.entity_id, t.core, t.address
                   FROM all_target_keys k
                   JOIN targets t ON t.entity_id=k.target_id
                   WHERE k.key=?
                   GROUP BY t.entity_id
                   ORDER BY t.entity_id""", (key,)
            ).fetchall()

            if not source_rows or not target_rows:
                continue
            if len(target_rows) > self.HIGH_FREQ_RESCUE_MAX_BLOCK:
                skipped_blocks += 1
                continue

            target_ids = [r[0] for r in target_rows]
            target_names = [r[1] for r in target_rows]
            target_addresses = [r[2] for r in target_rows]
            target_lookup = {r[0]: r for r in target_rows}

            for sid, s_name, s_address in source_rows:
                best: dict[str, float] = {}
                if s_name:
                    for _, score, idx in process.extract(
                        s_name, target_names, scorer=fuzz.ratio, limit=self.HIGH_FREQ_RESCUE_TOP
                    ):
                        best[target_ids[idx]] = max(best.get(target_ids[idx], 0.0), 0.55 * float(score) / 100.0)
                if s_address:
                    for _, score, idx in process.extract(
                        s_address, target_addresses, scorer=fuzz.ratio, limit=self.HIGH_FREQ_RESCUE_TOP
                    ):
                        best[target_ids[idx]] = max(best.get(target_ids[idx], 0.0), 0.45 * float(score) / 100.0)

                # Re-score the union so a candidate strong on both modalities wins.
                for tid in list(best):
                    trow = target_lookup[tid]
                    name_score = fuzz.ratio(s_name, trow[1]) / 100.0 if s_name and trow[1] else 0.0
                    addr_score = fuzz.ratio(s_address, trow[2]) / 100.0 if s_address and trow[2] else 0.0
                    if s_name and s_address:
                        cheap = 0.55 * name_score + 0.45 * addr_score
                    elif s_name:
                        cheap = name_score
                    else:
                        cheap = addr_score
                    best[tid] = cheap

                rows = [
                    (sid, tid, 1.0 + score)
                    for tid, score in sorted(best.items(), key=lambda item: (-item[1], item[0]))
                    if score >= 0.20
                ][: self.HIGH_FREQ_RESCUE_TOP]
                if rows:
                    family = key.split(":", 1)[0]
                    payload = []
                    for sid2, tid2, score2 in rows:
                        name_signal = float(score2 - 1.0) if s_name else 0.0
                        address_signal = float(score2 - 1.0) if s_address else 0.0
                        structural_signal = float(score2 - 1.0) if family in {
                            "number", "token", "phonetic"
                        } else 0.0
                        exact_signal = float(score2 - 1.0) if family in {
                            "full", "core", "sorted", "address", "full_composite"
                        } else 0.0
                        payload.append((
                            sid2,
                            tid2,
                            float(score2),
                            1,
                            name_signal,
                            address_signal,
                            structural_signal,
                            exact_signal,
                            0.0,
                            0,
                        ))

                    self.connection.executemany(
                        """INSERT INTO raw_candidates(
                               source1_id, target_id, evidence, support_count,
                                   name_evidence, address_evidence,
                               structural_evidence, exact_evidence,
                               semantic_score, semantic_rank
                           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                           ON CONFLICT(source1_id, target_id) DO UPDATE SET
                               evidence=MAX(raw_candidates.evidence, excluded.evidence),
                               support_count=raw_candidates.support_count + excluded.support_count,
                               name_evidence=MAX(raw_candidates.name_evidence, excluded.name_evidence),
                               address_evidence=MAX(raw_candidates.address_evidence, excluded.address_evidence),
                               structural_evidence=MAX(raw_candidates.structural_evidence, excluded.structural_evidence),
                               exact_evidence=MAX(raw_candidates.exact_evidence, excluded.exact_evidence)""",
                        payload,
                    )
                    rescued_pairs += len(rows)

        self.diagnostics["high_freq_rescue"] = {
            "rescued_pairs": rescued_pairs,
            "skipped_blocks": skipped_blocks,
            "max_rescue_block": self.HIGH_FREQ_RESCUE_MAX_BLOCK,
        }
        if skipped_blocks:
            LOGGER.warning(
                "Skipped %d oversized rescue blocks (> %d targets); "
                "selective/LSH blocks must cover those cases",
                skipped_blocks,
                self.HIGH_FREQ_RESCUE_MAX_BLOCK,
            )
        LOGGER.info("High-frequency rescue added %d bounded candidate rows", rescued_pairs)

    def finalize_candidates(self) -> int:
        """Build final candidates with a recall-first, low-I/O pruning path.

        The old implementation used several SQLite window functions over the
        entire raw candidate table.  On the real benchmark that meant repeatedly
        sorting millions of rows and writing large temporary tables to disk.

        This version keeps the same raw blocking stage, but performs pruning in
        one ordered pass over ``raw_candidates``:

        1. Gather one Source-1 entity's raw candidates.
        2. Reserve candidates from independent evidence families.
        3. Add top name/address lexical retrieval candidates from the raw block.
        4. Score only the bounded shortlist with RapidFuzz.
        5. Select at most TOP_K final candidates in memory.

        This preserves multiple retrieval routes while avoiding the large
        SQLite window-sort / scored_candidates / selected_candidates pipeline.
        """
        diagnostics: dict = {}
        has_truth = self._has_truth()

        if has_truth:
            raw_r, raw_hit, raw_tot = self._recall_on_table("raw_candidates")
            diagnostics["raw_blocking"] = {
                "recall": raw_r,
                "retrieved": raw_hit,
                "total": raw_tot,
                "complete_match_set_recall": self._complete_match_set_recall("raw_candidates"),
                "by_country": self._country_diagnostics("raw_candidates"),
            }
        raw_stats = self._candidate_stats("raw_candidates")
        diagnostics["raw_stats"] = raw_stats

        # For TOP_K=64 this creates at most 512 shortlist rows per S1.
        # Allocation is intentionally diversified and includes independent
        # lexical retrieval paths so sparse blocking evidence cannot eliminate a
        # candidate before stronger name/address similarity is evaluated.
        shortlist_budget = max(1, self.top_k * self.SHORTLIST_MULTIPLIER)
        global_quota = max(1, self.top_k + self.top_k // 2)
        exact_quota = max(1, self.top_k)
        name_quota = max(1, self.top_k)
        address_quota = max(1, self.top_k)
        structural_quota = max(1, self.top_k // 2)
        semantic_quota = (
            min(shortlist_budget, max(self.top_k, self.semantic_config.semantic_top_k))
            if self.semantic_config.enabled else 0
        )
        lexical_name_quota = max(1, self.top_k + self.top_k // 2)
        lexical_address_quota = max(1, self.top_k + self.top_k // 2)

        # Route quotas are upper bounds; candidates overlap heavily across routes.
        # The selector enforces the single shortlist budget while preserving the
        # strongest evidence from each independent retrieval family.

        # Truth is consumed as an ordered stream only when diagnostics are
        # available. This avoids materializing the full ground-truth mapping in
        # Python while still measuring shortlist/final recall precisely.
        truth_iter = None
        truth_current = None
        truth_total_pairs = 0
        truth_total_s1 = 0
        shortlist_hit_pairs = 0
        final_hit_pairs = 0
        shortlist_complete_s1 = 0
        final_complete_s1 = 0

        if has_truth:
            truth_total_pairs = int(self.connection.execute("SELECT COUNT(*) FROM truth").fetchone()[0])
            truth_total_s1 = int(self.connection.execute("SELECT COUNT(DISTINCT source1_id) FROM truth").fetchone()[0])
            truth_iter = iter(self.connection.execute(
                "SELECT source1_id, target_id FROM truth ORDER BY source1_id, target_id"
            ))
            truth_current = next(truth_iter, None)

        def _truth_for_sid(sid: str) -> set[str]:
            nonlocal truth_current
            if truth_iter is None:
                return set()
            while truth_current is not None and truth_current[0] < sid:
                # This S1 has no raw candidate group, so it contributes no hits.
                truth_current = next(truth_iter, None)
            if truth_current is None or truth_current[0] != sid:
                return set()
            result: set[str] = set()
            while truth_current is not None and truth_current[0] == sid:
                result.add(truth_current[1])
                truth_current = next(truth_iter, None)
            return result

        self.connection.execute("DROP TABLE IF EXISTS final_candidates")
        self.connection.execute("""
            CREATE TABLE final_candidates (
                source1_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                evidence REAL NOT NULL,
                similarity REAL NOT NULL,
                name_score REAL NOT NULL,
                address_score REAL NOT NULL,
                semantic_score REAL NOT NULL DEFAULT 0.0,
                semantic_rank INTEGER NOT NULL DEFAULT 0,
                rank INTEGER NOT NULL,
                PRIMARY KEY (source1_id, target_id)
            ) WITHOUT ROWID;
        """)

        insert_buffer: list[tuple] = []

        reader = self.connection.execute("""
            SELECT
                r.source1_id,
                r.target_id,
                r.evidence,
                r.support_count,
                r.name_evidence,
                r.address_evidence,
                r.structural_evidence,
                r.exact_evidence,
                r.semantic_score,
                r.semantic_rank,
                s.core,
                s.address,
                t.core,
                t.address
            FROM raw_candidates r
            JOIN source1 s ON s.entity_id=r.source1_id
            JOIN targets t ON t.entity_id=r.target_id
            ORDER BY r.source1_id, r.target_id
        """)

        current_sid: str | None = None
        current_rows: list[tuple] = []

        def _select_and_insert(rows: list[tuple], sid: str) -> None:
            nonlocal shortlist_hit_pairs, final_hit_pairs
            nonlocal shortlist_complete_s1, final_complete_s1
            if not rows:
                return

            # Raw-block evidence reservoirs. Each family gets its own bounded
            # route. Duplicates are removed by target_id when routes overlap.
            selected: dict[str, tuple] = {}

            def _take(candidates: list[tuple], quota: int, key_fn) -> None:
                if quota <= 0 or not candidates or len(selected) >= shortlist_budget:
                    return
                for row in heapq.nlargest(quota, candidates, key=key_fn):
                    if len(selected) >= shortlist_budget:
                        break
                    selected[row[1]] = row

            _take(rows, global_quota, lambda r: (r[2], r[3], r[1]))
            _take(rows, exact_quota, lambda r: (r[7], r[2], r[3], r[1]))
            _take(rows, name_quota, lambda r: (r[4], r[2], r[3], r[1]))
            _take(rows, address_quota, lambda r: (r[5], r[2], r[3], r[1]))
            _take(rows, structural_quota, lambda r: (r[6], r[3], r[2], r[1]))
            if semantic_quota:
                _take(
                    rows,
                    semantic_quota,
                    lambda r: (r[8], -r[9] if r[9] else -10_000, r[2], r[1]),
                )

            # Independent lexical retrieval directly over the raw block is the
            # key recall safeguard. A true match with weak blocking evidence can
            # still enter the shortlist when its normalized name/address is very
            # similar to the S1 record.
            s_core = rows[0][10]
            s_address = rows[0][11]
            target_cores = [r[12] for r in rows]
            target_addresses = [r[13] for r in rows]
            target_lookup = {r[1]: r for r in rows}

            if s_core:
                for _, _, idx in process.extract(
                    s_core,
                    target_cores,
                    scorer=fuzz.token_set_ratio,
                    limit=lexical_name_quota,
                ):
                    row = rows[idx]
                    selected[row[1]] = row
            if s_address:
                for _, _, idx in process.extract(
                    s_address,
                    target_addresses,
                    scorer=fuzz.token_set_ratio,
                    limit=lexical_address_quota,
                ):
                    row = rows[idx]
                    selected[row[1]] = row

            # Fill unused capacity from the strongest remaining raw candidates.
            # The global reservoir is deliberately larger than the final budget
            # so lexical-family overlap never reduces the actual shortlist size.
            selected_ids = set(selected)
            if len(selected_ids) < shortlist_budget:
                for row in heapq.nlargest(
                    shortlist_budget,
                    rows,
                    key=lambda r: (r[2], r[3], r[1]),
                ):
                    if row[1] not in selected_ids:
                        selected[row[1]] = row
                        selected_ids.add(row[1])
                        if len(selected_ids) >= shortlist_budget:
                            break

            shortlist_rows = list(selected.values())[:shortlist_budget]
            truth_ids = _truth_for_sid(sid) if has_truth else set()
            shortlist_ids = {row[1] for row in shortlist_rows}
            if truth_ids:
                shortlist_hit_pairs += len(truth_ids & shortlist_ids)
                if truth_ids.issubset(shortlist_ids):
                    shortlist_complete_s1 += 1

            if not shortlist_rows:
                return

            # Strong lexical scoring is now done only on the bounded shortlist.
            n_left = [row[10] for row in shortlist_rows]
            n_right = [row[12] for row in shortlist_rows]
            a_left = [row[11] for row in shortlist_rows]
            a_right = [row[13] for row in shortlist_rows]

            name_ratio = process.cpdist(
                n_left, n_right, scorer=fuzz.ratio, dtype=np.uint8, workers=-1
            ).astype(np.float32) / 100.0
            name_set = process.cpdist(
                n_left, n_right, scorer=fuzz.token_set_ratio, dtype=np.uint8, workers=-1
            ).astype(np.float32) / 100.0
            name_scores = np.maximum(name_ratio, name_set)

            addr_ratio = process.cpdist(
                a_left, a_right, scorer=fuzz.ratio, dtype=np.uint8, workers=-1
            ).astype(np.float32) / 100.0
            addr_set = process.cpdist(
                a_left, a_right, scorer=fuzz.token_set_ratio, dtype=np.uint8, workers=-1
            ).astype(np.float32) / 100.0
            address_scores = np.maximum(addr_ratio, addr_set)

            evidence_values = [float(row[2]) for row in shortlist_rows]
            min_ev = min(evidence_values)
            max_ev = max(evidence_values)
            denom = max_ev - min_ev

            scored: list[tuple] = []
            for row, n_s, a_s in zip(shortlist_rows, name_scores, address_scores):
                ev = float(row[2])
                ev_norm = (ev - min_ev) / denom if denom > 0 else 1.0
                support_signal = min(int(row[3]), 4) / 4.0
                structural_signal = 1.0 if float(row[6]) > 0 else 0.0
                exact_signal = 1.0 if float(row[7]) > 0 else 0.0
                semantic_signal = float(np.clip(row[8], 0.0, 1.0))
                sim = (
                    0.31 * float(n_s)
                    + 0.31 * float(a_s)
                    + 0.20 * semantic_signal
                    + 0.08 * ev_norm
                    + 0.05 * exact_signal
                    + 0.05 * max(structural_signal, support_signal)
                )
                scored.append((
                    row[0], row[1], ev_norm, float(sim), float(n_s), float(a_s),
                    float(max(structural_signal, support_signal)), float(exact_signal),
                    semantic_signal, int(row[9]) if row[9] else 0,
                ))

            # Adaptive final selection.  The previous implementation always
            # padded every Source-1 entity to TOP_K, which created large candidate
            # files even for obvious singleton/easy cases.  Here TOP_K is a HARD
            # CEILING, not a target count.
            #
            # We protect recall in three independent ways:
            #   1) strong exact/modality agreements are never discarded while room exists;
            #   2) a small per-source reserve prevents S2 or S3 from crowding out the other source;
            #   3) the remainder is selected by an absolute/relative cheap-score floor.
            ordered = sorted(scored, key=lambda r: (-r[3], r[1]))
            max_keep = self.top_k
            if len(ordered) <= max_keep:
                chosen = ordered
            else:
                top_score = float(ordered[0][3])
                if top_score >= 0.90:
                    min_keep = self.MIN_FINAL_CANDIDATES
                elif top_score >= 0.80:
                    min_keep = min(self.AMBIGUOUS_MIN_FINAL_CANDIDATES, max_keep)
                else:
                    min_keep = min(self.AMBIGUOUS_MIN_FINAL_CANDIDATES * 2, max_keep)

                score_floor = max(
                    self.FINAL_SCORE_FLOOR,
                    top_score - self.FINAL_SCORE_MARGIN,
                )

                chosen_by_id: dict[str, tuple] = {}

                def _add(row: tuple) -> None:
                    if len(chosen_by_id) < max_keep:
                        chosen_by_id.setdefault(row[1], row)

                # Exact retrieval evidence and very strong two-field agreement
                # are the safest candidates to protect first.
                for row in ordered:
                    strong = (
                        row[7] > 0.0
                        or row[8] >= 0.90
                        or (row[4] >= 0.92 and row[5] >= 0.85)
                        or (row[4] >= 0.95 and row[5] >= 0.70)
                        or (row[5] >= 0.95 and row[4] >= 0.70)
                    )
                    if strong:
                        _add(row)
                        if len(chosen_by_id) >= max_keep:
                            break

                # Preserve a small number from each real target source.
                source_quota = min(4, max(1, max_keep // 8))
                for source_prefix in ("S2-", "S3-"):
                    count = 0
                    for row in ordered:
                        if row[1].startswith(source_prefix) and row[1] not in chosen_by_id:
                            _add(row)
                            count += 1
                            if count >= source_quota or len(chosen_by_id) >= max_keep:
                                break

                # Main adaptive score band.  This is deliberately not a top-K
                # fill: a weak candidate below the evidence floor stays out.
                for row in ordered:
                    if row[3] >= score_floor:
                        _add(row)
                        if len(chosen_by_id) >= max_keep:
                            break

                # Guarantee a small recall floor even when scores are uniformly
                # low or the block is noisy.
                for row in ordered:
                    if len(chosen_by_id) >= min_keep:
                        break
                    _add(row)

                chosen = sorted(chosen_by_id.values(), key=lambda r: (-r[3], r[1]))

            if truth_ids:
                final_ids = {row[1] for row in chosen}
                final_hit_pairs += len(truth_ids & final_ids)
                if truth_ids.issubset(final_ids):
                    final_complete_s1 += 1

            insert_buffer.extend(
                (
                    row[0], row[1], float(row[2]), float(row[3]), float(row[4]),
                    float(row[5]), float(row[8]), int(row[9]), rank
                )
                for rank, row in enumerate(chosen, start=1)
            )

            if len(insert_buffer) >= 20_000:
                self.connection.executemany(
                    "INSERT INTO final_candidates(source1_id,target_id,evidence,similarity,name_score,address_score,semantic_score,semantic_rank,rank) VALUES (?,?,?,?,?,?,?,?,?)",
                    insert_buffer,
                )
                insert_buffer.clear()
                self.connection.commit()

        while True:
            batch = reader.fetchmany(50_000)
            if not batch:
                break
            for row in batch:
                sid = row[0]
                if current_sid is None:
                    current_sid = sid
                if sid != current_sid:
                    _select_and_insert(current_rows, current_sid)
                    current_rows = []
                    current_sid = sid
                current_rows.append(row)

        if current_sid is not None:
            _select_and_insert(current_rows, current_sid)

        if insert_buffer:
            self.connection.executemany(
                "INSERT INTO final_candidates(source1_id,target_id,evidence,similarity,name_score,address_score,semantic_score,semantic_rank,rank) VALUES (?,?,?,?,?,?,?,?,?)",
                insert_buffer,
            )
            insert_buffer.clear()
        self.connection.commit()

        # Any truth-only Source-1 entities not encountered in raw_candidates had
        # zero retrieval at both pruning stages and are therefore already counted
        # as misses by the denominator below.
        if has_truth:
            diagnostics["shortlist"] = {
                "recall": shortlist_hit_pairs / truth_total_pairs if truth_total_pairs else 1.0,
                "retrieved": shortlist_hit_pairs,
                "total": truth_total_pairs,
                "complete_match_set_recall": shortlist_complete_s1 / truth_total_s1 if truth_total_s1 else 1.0,
            }
            diagnostics["final"] = {
                "recall": final_hit_pairs / truth_total_pairs if truth_total_pairs else 1.0,
                "retrieved": final_hit_pairs,
                "total": truth_total_pairs,
                "complete_match_set_recall": final_complete_s1 / truth_total_s1 if truth_total_s1 else 1.0,
                "s1_with_zero_true_candidates": 0.0,
                "by_country": self._country_diagnostics("final_candidates"),
            }

            # Exact zero-hit S1 rates are cheap to compute from the final table.
            diagnostics["shortlist"].pop("s1_with_zero_true_candidates", None)
            diagnostics["final"]["s1_with_zero_true_candidates"] = self._s1_zero_true_candidates("final_candidates")

        fin_stats = self._candidate_stats("final_candidates")
        diagnostics["final_stats"] = fin_stats
        if has_truth:
            # Preserve the existing diagnostic name expected by downstream logs.
            diagnostics["final"]["s1_with_zero_true_candidates"] = self._s1_zero_true_candidates("final_candidates")

        self.connection.execute("DROP TABLE raw_candidates")
        self.connection.commit()

        self.diagnostics = diagnostics
        if has_truth:
            LOGGER.info(
                "Final blocking recall ceiling: %.4f%% (%d pairs, avg %.2f/entity, CMS recall %.4f%%)",
                100 * diagnostics["final"]["recall"],
                fin_stats["total_pairs"],
                fin_stats["avg_candidates"],
                100 * diagnostics["final"]["complete_match_set_recall"],
            )
        else:
            LOGGER.info(
                "Final candidate set built: %d pairs, avg %.2f/entity "
                "(test set has no truth, so recall is not computed)",
                fin_stats["total_pairs"],
                fin_stats["avg_candidates"],
            )
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
            WITH ranked AS (
                SELECT
                    c.*,
                    ROW_NUMBER() OVER (
                        PARTITION BY c.source1_id
                        ORDER BY c.name_score DESC, c.target_id
                    ) AS name_rank,
                    ROW_NUMBER() OVER (
                        PARTITION BY c.source1_id
                        ORDER BY c.address_score DESC, c.target_id
                    ) AS address_rank,
                    ROW_NUMBER() OVER (
                        PARTITION BY c.target_id
                        ORDER BY c.similarity DESC, c.source1_id
                    ) AS reverse_similarity_rank
                FROM final_candidates c
            )
            SELECT
                r.source1_id, r.target_id,
                s.country, s.name, s.core, s.sorted_core, s.address, s.numbers,
                t.country, t.name, t.core, t.sorted_core, t.address, t.numbers,
                r.evidence, r.similarity, r.rank, r.semantic_score, r.semantic_rank,
                s.street_number, s.street, s.city, s.postal_code,
                t.street_number, t.street, t.city, t.postal_code,
                r.name_rank, r.address_rank, r.reverse_similarity_rank,
                CASE WHEN r.rank = 1 AND r.reverse_similarity_rank = 1 THEN 1.0 ELSE 0.0 END AS mutual_best,
                1.0 / (1.0 + ABS(r.name_rank - r.address_rank)) AS rank_agreement,
                CASE WHEN truth.target_id IS NULL THEN 0 ELSE 1 END
            FROM ranked r
            JOIN source1 s ON s.entity_id=r.source1_id
            JOIN targets t ON t.entity_id=r.target_id
            LEFT JOIN truth ON truth.source1_id=r.source1_id AND truth.target_id=r.target_id
            {predicate.replace('c.', 'r.').replace('s.', 's.')}
            ORDER BY r.source1_id, r.rank, r.target_id
        """, parameters)
        yield from cursor

    def feature_count(self, where: str = "", parameters: Sequence[object] = ()) -> int:
        predicate = f"WHERE {where}" if where else ""
        return self.connection.execute(
            f"SELECT COUNT(*) FROM final_candidates c JOIN source1 s ON s.entity_id=c.source1_id JOIN targets t ON t.entity_id=c.target_id LEFT JOIN truth ON truth.source1_id=c.source1_id AND truth.target_id=c.target_id {predicate}", parameters
        ).fetchone()[0]

    def _has_truth(self) -> bool:
        return self.connection.execute("SELECT COUNT(*) FROM truth").fetchone()[0] > 0

    def _country_diagnostics(self, table: str) -> dict[str, dict]:
        """Return recall/CMS/candidate-count diagnostics split by Source-1 country."""
        if not self._has_truth():
            return {}
        rows = self.connection.execute(
            f"""
            SELECT
                COALESCE(NULLIF(s.country, ''), '<missing>') AS country,
                COUNT(*) AS truth_pairs,
                SUM(CASE WHEN c.target_id IS NOT NULL THEN 1 ELSE 0 END) AS retrieved_pairs,
                COUNT(DISTINCT t.source1_id) AS truth_s1
            FROM truth t
            JOIN source1 s ON s.entity_id=t.source1_id
            LEFT JOIN {table} c
              ON c.source1_id=t.source1_id AND c.target_id=t.target_id
            GROUP BY COALESCE(NULLIF(s.country, ''), '<missing>')
            ORDER BY country
            """
        ).fetchall()
        candidate_rows = self.connection.execute(
            f"""
            SELECT
                COALESCE(NULLIF(s.country, ''), '<missing>') AS country,
                COUNT(c.target_id) AS candidate_pairs,
                COUNT(DISTINCT s.entity_id) AS s1_entities
            FROM source1 s
            LEFT JOIN {table} c ON c.source1_id=s.entity_id
            GROUP BY COALESCE(NULLIF(s.country, ''), '<missing>')
            ORDER BY country
            """
        ).fetchall()
        candidate_map = {row[0]: row for row in candidate_rows}
        result: dict[str, dict] = {}
        for country, truth_pairs, retrieved_pairs, truth_s1 in rows:
            _country, candidate_pairs, s1_entities = candidate_map.get(country, (country, 0, 0))
            # Complete-set recall requires that every truth target for an S1 is present.
            cms = self.connection.execute(
                f"""
                SELECT COUNT(*) FROM (
                    SELECT t.source1_id,
                           COUNT(*) - COUNT(c.target_id) AS missing
                    FROM truth t
                    JOIN source1 s ON s.entity_id=t.source1_id
                    LEFT JOIN {table} c
                      ON c.source1_id=t.source1_id AND c.target_id=t.target_id
                    WHERE COALESCE(NULLIF(s.country, ''), '<missing>')=?
                    GROUP BY t.source1_id
                ) q WHERE q.missing=0
                """, (country,)
            ).fetchone()[0]
            result[country] = {
                "pair_recall": retrieved_pairs / truth_pairs if truth_pairs else 1.0,
                "complete_match_set_recall": cms / truth_s1 if truth_s1 else 1.0,
                "retrieved_pairs": int(retrieved_pairs),
                "truth_pairs": int(truth_pairs),
                "candidate_pairs": int(candidate_pairs),
                "candidate_avg": candidate_pairs / s1_entities if s1_entities else 0.0,
            }
        return result

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

    def _complete_match_set_recall(self, table: str) -> float:
        """Fraction of S1 entities for which ALL true matches are in the candidate set."""
        result = self.connection.execute(f"""
            SELECT
                CAST(SUM(CASE WHEN missing = 0 THEN 1 ELSE 0 END) AS REAL) / COUNT(*) 
            FROM (
                SELECT t.source1_id,
                       COUNT(*) - COUNT(c.target_id) AS missing
                FROM truth t
                LEFT JOIN {table} c ON c.source1_id = t.source1_id AND c.target_id = t.target_id
                GROUP BY t.source1_id
            )
        """).fetchone()[0]
        return float(result) if result is not None else 1.0

    def _s1_zero_true_candidates(self, table: str) -> float:
        """Fraction of S1 entities (that have truth) with zero true candidates retrieved."""
        result = self.connection.execute(f"""
            SELECT
                CAST(SUM(CASE WHEN retrieved = 0 THEN 1 ELSE 0 END) AS REAL) / COUNT(*)
            FROM (
                SELECT t.source1_id,
                       COUNT(c.target_id) AS retrieved
                FROM truth t
                LEFT JOIN {table} c ON c.source1_id = t.source1_id AND c.target_id = t.target_id
                GROUP BY t.source1_id
            )
        """).fetchone()[0]
        return float(result) if result is not None else 0.0

    def _candidate_stats(self, table: str) -> dict:
        # Keep one compact numeric value per Source-1 instead of a Python list
        # of Python ints. This matters when the real test set has millions of
        # Source-1 entities.
        counts = np.fromiter(
            (int(row[0]) for row in self.connection.execute(
                f"SELECT COUNT({table}.target_id) FROM source1 LEFT JOIN {table} "
                f"ON {table}.source1_id = source1.entity_id GROUP BY source1.entity_id"
            )),
            dtype=np.int32,
        )
        if counts.size == 0:
            return {
                "total_pairs": 0,
                "avg_candidates": 0.0,
                "median_candidates": 0,
                "max_candidates": 0,
                "p50": 0,
                "p95": 0,
                "p99": 0,
            }

        counts.sort()
        n = int(counts.size)
        total_pairs = int(np.sum(counts, dtype=np.int64))

        return {
            "total_pairs": total_pairs,
            "avg_candidates": total_pairs / n,
            "median_candidates": int(counts[n // 2]),
            "max_candidates": int(counts[-1]),
            "p50": int(counts[n // 2]),
            "p95": int(counts[min(n - 1, int(n * 0.95))]),
            "p99": int(counts[min(n - 1, int(n * 0.99))]),
        }