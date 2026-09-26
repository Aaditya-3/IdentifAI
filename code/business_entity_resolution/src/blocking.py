"""Disk-backed, high-recall bounded candidate retrieval for large entity-resolution data.

Changes from prior version:
- REMOVED arbitrary KEY_CAP=160 truncation that silently dropped entities.
- Added frequency-aware key filtering: low-frequency keys keep all entities,
  high-frequency keys are discarded entirely (they are not selective).
- Fixed cheap ranking score to properly balance name vs address signals.
- Added complete-match-set recall diagnostics.
- Widened shortlist window before scoring.
"""
from __future__ import annotations

import csv
import hashlib
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
)

LOGGER = logging.getLogger(__name__)


class BlockingStore:
    TOP_K = 64
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
        valid_keys.update(row[0] for row in self.connection.execute("SELECT DISTINCT key FROM high_freq_source_keys"))
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
            INSERT INTO raw_candidates(source1_id, target_id, evidence)
            SELECT b.entity_id, t.target_id, SUM(b.weight * t.weight)
            FROM bounded_target_keys t JOIN bounded_keys b ON b.key = t.key
            GROUP BY b.entity_id, t.target_id
            ON CONFLICT(source1_id, target_id) DO UPDATE SET evidence=evidence + excluded.evidence
        """)
        self._rescue_high_frequency_keys()
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
                   LEFT JOIN (SELECT source1_id, COUNT(*) AS cnt FROM raw_candidates GROUP BY source1_id) rc
                     ON rc.source1_id=s.entity_id
                   WHERE COALESCE(rc.cnt, 0) < ?
                   ORDER BY s.entity_id""", (key, key, self.top_k)
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
                    self.connection.executemany(
                        """INSERT INTO raw_candidates(source1_id, target_id, evidence) VALUES (?, ?, ?)
                           ON CONFLICT(source1_id, target_id) DO UPDATE SET evidence=MAX(raw_candidates.evidence, excluded.evidence)""",
                        rows,
                    )
                    rescued_pairs += len(rows)

        if skipped_blocks:
            LOGGER.warning("Skipped %d oversized rescue blocks (> %d targets); selective/LSH blocks must cover those cases",
                           skipped_blocks, self.HIGH_FREQ_RESCUE_MAX_BLOCK)
        LOGGER.info("High-frequency rescue added %d bounded candidate rows", rescued_pairs)

    def finalize_candidates(self) -> int:
        diagnostics: dict = {}
        has_truth = self._has_truth()

        if has_truth:
            raw_r, raw_hit, raw_tot = self._recall_on_table("raw_candidates")
            diagnostics["raw_blocking"] = {"recall": raw_r, "retrieved": raw_hit, "total": raw_tot}
            cms_r = self._complete_match_set_recall("raw_candidates")
            diagnostics["raw_blocking"]["complete_match_set_recall"] = cms_r
        raw_stats = self._candidate_stats("raw_candidates")
        diagnostics["raw_stats"] = raw_stats

        # Wider shortlist: top_k * 8 globally, top_k * 2 per source type
        shortlist_global = self.top_k * 8
        shortlist_per_source = self.top_k * 2
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
        """, (shortlist_global, shortlist_per_source))

        if has_truth:
            sl_r, sl_hit, sl_tot = self._recall_on_table("shortlist")
            diagnostics["shortlist"] = {"recall": sl_r, "retrieved": sl_hit, "total": sl_tot}
            diagnostics["shortlist"]["complete_match_set_recall"] = self._complete_match_set_recall("shortlist")
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
            name_scores = np.maximum(name_ratio, name_set).astype(np.float32) / 100.0
            addr_ratio = process.cpdist(s_addrs, t_addrs, scorer=fuzz.ratio, dtype=np.uint8, workers=-1)
            addr_set = process.cpdist(s_addrs, t_addrs, scorer=fuzz.token_set_ratio, dtype=np.uint8, workers=-1)
            addr_scores = np.maximum(addr_ratio, addr_set).astype(np.float32) / 100.0

            for (sid, tid, *_, ev), n_s, a_s in zip(rows, name_scores, addr_scores):
                # Balanced cheap ranking: name and address contribute equally,
                # evidence provides a boost.  No single signal dominates.
                # name_s and addr_s are in [0, 1]; ev is normalized evidence in [0, 1].
                sim = 0.40 * float(n_s) + 0.35 * float(a_s) + 0.25 * float(ev)
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
            diagnostics["final"]["complete_match_set_recall"] = self._complete_match_set_recall("final_candidates")
            diagnostics["final"]["s1_with_zero_true_candidates"] = self._s1_zero_true_candidates("final_candidates")
        fin_stats = self._candidate_stats("final_candidates")
        diagnostics["final_stats"] = fin_stats

        self.diagnostics = diagnostics
        LOGGER.info("Final blocking recall ceiling: %.4f%% (%d pairs, avg %.2f/entity, CMS recall %.4f%%)",
                    100 * diagnostics.get("final", {}).get("recall", 0),
                    fin_stats["total_pairs"], fin_stats["avg_candidates"],
                    100 * diagnostics.get("final", {}).get("complete_match_set_recall", 0))
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
            f"SELECT COUNT(*) FROM final_candidates c JOIN source1 s ON s.entity_id=c.source1_id JOIN targets t ON t.entity_id=c.target_id LEFT JOIN truth ON truth.source1_id=c.source1_id AND truth.target_id=c.target_id {predicate}", parameters
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
        counts = [r[0] for r in self.connection.execute(f"SELECT COUNT(*) FROM {table} GROUP BY source1_id")]
        if not counts:
            return {"total_pairs": 0, "avg_candidates": 0.0, "median_candidates": 0, "max_candidates": 0,
                    "p50": 0, "p95": 0, "p99": 0}
        n = len(counts)
        counts.sort()
        return {
            "total_pairs": sum(counts),
            "avg_candidates": sum(counts) / n,
            "median_candidates": counts[n // 2],
            "max_candidates": counts[-1],
            "p50": counts[n // 2],
            "p95": counts[min(n - 1, int(n * 0.95))],
            "p99": counts[min(n - 1, int(n * 0.99))],
        }