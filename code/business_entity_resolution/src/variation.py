"""Training-derived variation signals for noisy business records.

The module learns token correspondences only from known positive pairs in the
training fold.  It does not use external data and is intentionally bounded:
only the strongest target-token mappings are retained for each source token.

The model is used during post-baseline tuning/production, never as a hidden
source of labels.  Validation folds must construct it from their training side
only.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np


def _tokenize(value: str) -> tuple[str, ...]:
    return tuple(token for token in str(value or "").split() if len(token) >= 3)


@dataclass(frozen=True)
class _TokenMap:
    targets: Mapping[str, tuple[tuple[str, float], ...]]

    def score(self, left: str, right: str) -> tuple[float, float]:
        """Return directional alias score and fraction of tokens explained."""
        left_tokens = _tokenize(left)
        right_set = set(_tokenize(right))
        if not left_tokens or not right_set:
            return 0.0, 0.0

        total = 0.0
        explained = 0
        for token in left_tokens:
            options = self.targets.get(token)
            if not options:
                continue
            best = 0.0
            for target_token, probability in options:
                if target_token in right_set:
                    best = max(best, float(probability))
            if best > 0.0:
                explained += 1
                total += best
        return total / len(left_tokens), explained / len(left_tokens)


@dataclass(frozen=True)
class VariationModel:
    """Bounded lexical-variation model learned from positive training pairs."""

    name_forward: _TokenMap
    name_reverse: _TokenMap
    address_forward: _TokenMap
    address_reverse: _TokenMap
    positive_pairs: int

    MAX_MAPPINGS_PER_TOKEN = 12
    MIN_MAPPING_SUPPORT = 2

    @classmethod
    def empty(cls) -> "VariationModel":
        empty = _TokenMap({})
        return cls(empty, empty, empty, empty, 0)

    @classmethod
    def from_store(
        cls,
        store,
        where: str = "1=1",
        parameters: Sequence[object] = (),
        max_positive_pairs: int | None = None,
    ) -> "VariationModel":
        """Build variation maps from the truth pairs satisfying ``where``.

        ``where`` is evaluated against the Source-1 row using alias ``s``.  The
        mapping is built exclusively from positive pairs in that training view.
        """
        name_forward: dict[str, Counter[str]] = defaultdict(Counter)
        name_reverse: dict[str, Counter[str]] = defaultdict(Counter)
        addr_forward: dict[str, Counter[str]] = defaultdict(Counter)
        addr_reverse: dict[str, Counter[str]] = defaultdict(Counter)

        sql = f"""
            SELECT s.core, s.address, t.core, t.address
            FROM truth tr
            JOIN source1 s ON s.entity_id = tr.source1_id
            JOIN targets t ON t.entity_id = tr.target_id
            WHERE {where}
        """
        cursor = store.connection.execute(sql, parameters)
        positive_pairs = 0

        for s_core, s_addr, t_core, t_addr in cursor:
            positive_pairs += 1
            if max_positive_pairs is not None and positive_pairs > max_positive_pairs:
                break

            s_tokens = set(_tokenize(s_core))
            t_tokens = set(_tokenize(t_core))
            for st in s_tokens:
                counter = name_forward[st]
                for tt in t_tokens:
                    counter[tt] += 1
            for tt in t_tokens:
                counter = name_reverse[tt]
                for st in s_tokens:
                    counter[st] += 1

            s_tokens = set(_tokenize(s_addr))
            t_tokens = set(_tokenize(t_addr))
            for st in s_tokens:
                counter = addr_forward[st]
                for tt in t_tokens:
                    counter[tt] += 1
            for tt in t_tokens:
                counter = addr_reverse[tt]
                for st in s_tokens:
                    counter[st] += 1

        def finalize(source: Mapping[str, Counter[str]]) -> _TokenMap:
            table: dict[str, tuple[tuple[str, float], ...]] = {}
            for token, counter in source.items():
                filtered = [(target, count) for target, count in counter.items() if count >= cls.MIN_MAPPING_SUPPORT]
                if not filtered:
                    continue
                total = sum(count for _, count in filtered)
                top = sorted(filtered, key=lambda item: (-item[1], item[0]))[: cls.MAX_MAPPINGS_PER_TOKEN]
                table[token] = tuple((target, count / total) for target, count in top)
            return _TokenMap(table)

        return cls(
            finalize(name_forward),
            finalize(name_reverse),
            finalize(addr_forward),
            finalize(addr_reverse),
            positive_pairs,
        )

    def score_batch(
        self,
        source_cores: Sequence[str],
        target_cores: Sequence[str],
        source_addresses: Sequence[str],
        target_addresses: Sequence[str],
    ) -> np.ndarray:
        """Return four learned-variation features per pair."""
        n = len(source_cores)
        out = np.zeros((n, 4), dtype=np.float32)
        if n == 0 or self.positive_pairs == 0:
            return out

        for i, (sc, tc, sa, ta) in enumerate(
            zip(source_cores, target_cores, source_addresses, target_addresses)
        ):
            nf, ne = self.name_forward.score(sc, tc)
            nr, _ = self.name_reverse.score(tc, sc)
            af, ae = self.address_forward.score(sa, ta)
            ar, _ = self.address_reverse.score(ta, sa)
            out[i, 0] = 0.5 * (nf + nr)
            out[i, 1] = 0.5 * (ne + self._reverse_explained(self.name_reverse, tc, sc))
            out[i, 2] = 0.5 * (af + ar)
            out[i, 3] = 0.5 * (ae + self._reverse_explained(self.address_reverse, ta, sa))
        return out

    @staticmethod
    def _reverse_explained(token_map: _TokenMap, left: str, right: str) -> float:
        return token_map.score(left, right)[1]


__all__ = ["VariationModel"]
