"""BGE semantic retrieval, FAISS ANN indexing, and bounded cross-encoder reranking.

The semantic layer is deliberately retrieval-first:
- a bi-encoder creates one compact embedding per target entity;
- FAISS HNSW retrieves only a bounded semantic neighborhood per Source-1 entity;
- a cross-encoder is evaluated only on the strongest semantic candidates;
- lexical blocking remains active, so semantic retrieval can add recall without
  replacing the deterministic blockers.

The production path uses Sentence Transformers + FAISS when those packages are
installed. A deterministic HashingVectorizer fallback exists solely so unit and
offline smoke tests remain runnable; production requirements install the real
BGE/FAISS stack.
"""
from __future__ import annotations

import hashlib
import logging
import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SemanticConfig:
    enabled: bool = True
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-base"
    semantic_top_k: int = 48
    encode_batch_size: int = 128
    rerank_top_k: int = 4
    rerank_batch_size: int = 32
    hnsw_m: int = 32
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 96
    cache_size: int = 20_000
    device: str = "auto"
    allow_fallback: bool = False

    @classmethod
    def from_environment(cls, *, top_k: int | None = None) -> "SemanticConfig":
        def _bool(name: str, default: bool) -> bool:
            value = os.getenv(name)
            if value is None:
                return default
            return value.strip().casefold() not in {"0", "false", "no", "off"}

        semantic_top_k = int(os.getenv("IDENTIFAI_SEMANTIC_TOP_K", "48"))
        if top_k is not None:
            semantic_top_k = max(semantic_top_k, min(64, int(top_k)))

        return cls(
            enabled=_bool("IDENTIFAI_SEMANTIC_ENABLED", True),
            embedding_model=os.getenv("IDENTIFAI_BGE_MODEL", cls.embedding_model),
            reranker_model=os.getenv("IDENTIFAI_RERANKER_MODEL", cls.reranker_model),
            semantic_top_k=max(1, semantic_top_k),
            encode_batch_size=max(1, int(os.getenv("IDENTIFAI_BGE_BATCH_SIZE", str(cls.encode_batch_size)))),
            rerank_top_k=max(0, int(os.getenv("IDENTIFAI_RERANK_TOP_K", str(cls.rerank_top_k)))),
            rerank_batch_size=max(1, int(os.getenv("IDENTIFAI_RERANK_BATCH_SIZE", str(cls.rerank_batch_size)))),
            hnsw_m=max(4, int(os.getenv("IDENTIFAI_HNSW_M", str(cls.hnsw_m)))),
            hnsw_ef_construction=max(8, int(os.getenv("IDENTIFAI_HNSW_EF_CONSTRUCTION", str(cls.hnsw_ef_construction)))),
            hnsw_ef_search=max(8, int(os.getenv("IDENTIFAI_HNSW_EF_SEARCH", str(cls.hnsw_ef_search)))),
            cache_size=max(0, int(os.getenv("IDENTIFAI_RERANK_CACHE", str(cls.cache_size)))),
            device=os.getenv("IDENTIFAI_DEVICE", cls.device),
            allow_fallback=_bool("IDENTIFAI_SEMANTIC_FALLBACK", False),
        )

    def signature(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "embedding_model": self.embedding_model,
            "reranker_model": self.reranker_model,
            "semantic_top_k": self.semantic_top_k,
            "encode_batch_size": self.encode_batch_size,
            "rerank_top_k": self.rerank_top_k,
            "rerank_batch_size": self.rerank_batch_size,
            "hnsw_m": self.hnsw_m,
            "hnsw_ef_construction": self.hnsw_ef_construction,
            "hnsw_ef_search": self.hnsw_ef_search,
            "cache_size": self.cache_size,
            "device": self.device,
            "allow_fallback": self.allow_fallback,
        }


def entity_text(name: str, address: str, country: str) -> str:
    """Build one consistent representation used for embedding and reranking."""
    parts: list[str] = []
    if name:
        parts.append(f"business name: {name}")
    if address:
        parts.append(f"business address: {address}")
    if country:
        parts.append(f"country: {country}")
    return "; ".join(parts)


def _device_name(config: SemanticConfig) -> str | None:
    if config.device.casefold() == "auto":
        return None
    return config.device


class SemanticRetriever:
    """Reusable BGE + FAISS retrieval component."""

    def __init__(self, config: SemanticConfig):
        self.config = config
        self._encoder = None
        self._index = None
        self._target_ids: list[str] = []
        self._fallback_matrix: np.ndarray | None = None
        self._fallback_vectorizer = HashingVectorizer(
            analyzer="char",
            ngram_range=(2, 5),
            n_features=2**16,
            alternate_sign=False,
            norm="l2",
            lowercase=True,
        )
        self._using_real_backend = False
        self._ready = False

    @property
    def using_real_backend(self) -> bool:
        return self._using_real_backend

    def _load_encoder(self):
        if self._encoder is not None:
            return self._encoder
        if not self.config.enabled:
            return None
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            if not self.config.allow_fallback:
                raise RuntimeError(
                    "Semantic retrieval requires sentence-transformers. "
                    "Install requirements.txt or disable semantic retrieval explicitly with "
                    "IDENTIFAI_SEMANTIC_ENABLED=0."
                ) from exc
            LOGGER.warning(
                "Sentence Transformers is unavailable; using explicit deterministic fallback "
                "because IDENTIFAI_SEMANTIC_FALLBACK is enabled."
            )
            self._encoder = False
            return None
        self._encoder = SentenceTransformer(
            self.config.embedding_model,
            device=_device_name(self.config),
        )
        self._using_real_backend = True
        LOGGER.info("Loaded semantic encoder: %s", self.config.embedding_model)
        return self._encoder

    @staticmethod
    def _normalize(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        return np.divide(values, norms, out=np.zeros_like(values), where=norms > 1e-12)

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        encoder = self._load_encoder()
        if encoder is None or encoder is False:
            matrix = self._fallback_vectorizer.transform(texts).astype(np.float32)
            return matrix.toarray()
        encoded = encoder.encode(
            list(texts),
            batch_size=self.config.encode_batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return self._normalize(np.asarray(encoded, dtype=np.float32))

    def _create_index(self, dimension: int):
        try:
            import faiss
        except ImportError as exc:
            if not self.config.allow_fallback:
                raise RuntimeError(
                    "Semantic retrieval requires faiss-cpu for the bounded ANN index. "
                    "Install requirements.txt or disable semantic retrieval explicitly with "
                    "IDENTIFAI_SEMANTIC_ENABLED=0."
                ) from exc
            LOGGER.warning(
                "FAISS is unavailable; using explicit exact NumPy fallback "
                "because IDENTIFAI_SEMANTIC_FALLBACK is enabled."
            )
            return None

        index = faiss.IndexHNSWFlat(
            int(dimension),
            int(self.config.hnsw_m),
            faiss.METRIC_INNER_PRODUCT,
        )
        index.hnsw.efConstruction = int(self.config.hnsw_ef_construction)
        index.hnsw.efSearch = int(self.config.hnsw_ef_search)
        return index

    def build(self, targets: Iterable[tuple[str, str]]) -> None:
        """Build the target semantic index from ``(entity_id, entity_text)`` rows."""
        if not self.config.enabled:
            self._ready = True
            return

        batch_ids: list[str] = []
        batch_texts: list[str] = []
        all_fallback: list[np.ndarray] = []
        count = 0

        def _flush() -> None:
            nonlocal count
            if not batch_texts:
                return
            embeddings = self._encode(batch_texts)
            if self._index is None and self._fallback_matrix is None:
                self._index = self._create_index(embeddings.shape[1])
            if self._index is not None:
                self._index.add(np.ascontiguousarray(embeddings, dtype=np.float32))
            else:
                all_fallback.append(embeddings)
            self._target_ids.extend(batch_ids)
            count += len(batch_ids)
            batch_ids.clear()
            batch_texts.clear()

        for target_id, text in targets:
            if not target_id:
                continue
            batch_ids.append(str(target_id))
            batch_texts.append(text)
            if len(batch_texts) >= self.config.encode_batch_size:
                _flush()
        _flush()

        if self._index is None:
            self._fallback_matrix = (
                np.vstack(all_fallback).astype(np.float32, copy=False)
                if all_fallback
                else np.empty((0, 0), dtype=np.float32)
            )
        self._ready = True
        LOGGER.info(
            "Semantic target index ready: %d targets (%s backend)",
            count,
            "BGE+FAISS-HNSW" if self._index is not None and self._using_real_backend else "deterministic fallback",
        )

    def query(self, sources: Iterable[tuple[str, str]], top_k: int | None = None) -> list[tuple[str, str, float, int]]:
        if not self.config.enabled or not self._ready:
            return []
        if not self._target_ids:
            return []
        k = max(1, int(top_k or self.config.semantic_top_k))
        results: list[tuple[str, str, float, int]] = []
        source_ids: list[str] = []
        texts: list[str] = []

        def _flush() -> None:
            if not texts:
                return
            query_vectors = self._encode(texts)
            if self._index is not None:
                distances, indices = self._index.search(
                    np.ascontiguousarray(query_vectors, dtype=np.float32),
                    min(k, len(self._target_ids)),
                )
            else:
                if self._fallback_matrix is None or len(self._target_ids) == 0:
                    distances = np.empty((len(texts), 0), dtype=np.float32)
                    indices = np.empty((len(texts), 0), dtype=np.int64)
                else:
                    scores = query_vectors @ self._fallback_matrix.T
                    local_k = min(k, scores.shape[1])
                    idx = np.argpartition(-scores, local_k - 1, axis=1)[:, :local_k]
                    local_scores = np.take_along_axis(scores, idx, axis=1)
                    order = np.argsort(-local_scores, axis=1, kind="stable")
                    indices = np.take_along_axis(idx, order, axis=1)
                    distances = np.take_along_axis(local_scores, order, axis=1)

            for row_index, source_id in enumerate(source_ids):
                seen: set[int] = set()
                rank = 0
                for distance, target_index in zip(distances[row_index], indices[row_index]):
                    target_index = int(target_index)
                    if target_index < 0 or target_index >= len(self._target_ids):
                        continue
                    if target_index in seen:
                        continue
                    seen.add(target_index)
                    rank += 1
                    results.append(
                        (
                            source_id,
                            self._target_ids[target_index],
                            float(distance),
                            rank,
                        )
                    )

            source_ids.clear()
            texts.clear()

        for source_id, text in sources:
            source_ids.append(str(source_id))
            texts.append(text)
            if len(texts) >= self.config.encode_batch_size:
                _flush()
        _flush()
        return results

    def rerank(self, pairs: Sequence[tuple[str, str, int]]) -> np.ndarray:
        """Return bounded cross-encoder scores for ``(query, document, rank)`` rows."""
        output = np.zeros(len(pairs), dtype=np.float32)
        if not pairs or not self.config.enabled or self.config.rerank_top_k <= 0:
            return output

        eligible_indices = [
            i for i, (_, _, semantic_rank) in enumerate(pairs)
            if 1 <= int(semantic_rank) <= self.config.rerank_top_k
        ]
        if not eligible_indices:
            return output

        try:
            import torch
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            if not self.config.allow_fallback:
                raise RuntimeError(
                    "Cross-encoder reranking requires sentence-transformers and torch. "
                    "Install requirements.txt or set IDENTIFAI_RERANK_TOP_K=0 explicitly."
                ) from exc
            return output

        if not hasattr(self, "_reranker"):
            self._reranker = CrossEncoder(
                self.config.reranker_model,
                activation_fn=torch.nn.Sigmoid(),
                device=_device_name(self.config),
            )
            self._rerank_cache: OrderedDict[str, float] = OrderedDict()
            LOGGER.info("Loaded cross-encoder reranker: %s", self.config.reranker_model)

        to_score: list[tuple[int, str, str]] = []
        for index in eligible_indices:
            query, document, _rank = pairs[index]
            key = hashlib.blake2b(
                f"{query}\0{document}".encode("utf-8"), digest_size=16
            ).hexdigest()
            cached = self._rerank_cache.get(key)
            if cached is None:
                to_score.append((index, query, document))
            else:
                output[index] = float(cached)
                self._rerank_cache.move_to_end(key)

        if to_score:
            scores = self._reranker.predict(
                [(query, document) for _, query, document in to_score],
                batch_size=self.config.rerank_batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            for (index, query, document), score in zip(to_score, np.asarray(scores).reshape(-1)):
                value = float(np.clip(score, 0.0, 1.0))
                output[index] = value
                key = hashlib.blake2b(
                    f"{query}\0{document}".encode("utf-8"), digest_size=16
                ).hexdigest()
                self._rerank_cache[key] = value
                self._rerank_cache.move_to_end(key)
                while len(self._rerank_cache) > self.config.cache_size:
                    self._rerank_cache.popitem(last=False)

        return output

    def close(self) -> None:
        self._index = None
        self._fallback_matrix = None
        self._target_ids.clear()
        self._encoder = None
        if hasattr(self, "_reranker"):
            self._reranker = None
        self._ready = False


__all__ = ["SemanticConfig", "SemanticRetriever", "entity_text"]
