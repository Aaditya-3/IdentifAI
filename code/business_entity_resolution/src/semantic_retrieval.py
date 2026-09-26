"""Optional semantic retrieval for business entity resolution.

The semantic route is deliberately *fail-open*:

* lexical blocking remains the primary, deterministic retrieval path;
* BGE + FAISS is used when the required packages and model weights are available;
* model weights are loaded locally by default, so an offline environment never waits
  for a network download;
* missing/broken semantic dependencies, model load failures, FAISS failures, and
  cross-encoder failures degrade to lexical-only matching instead of crashing the
  training or prediction pipeline;
* cross-encoder reranking is disabled by default because applying a transformer
  cross-encoder to millions of candidate pairs is not a bounded operation. It can be
  explicitly enabled once the model is locally available and runtime has been measured.

A strict mode is still available through ``IDENTIFAI_SEMANTIC_FALLBACK=0`` for
debugging/deployment validation when silent degradation is undesirable.
"""
from __future__ import annotations

import hashlib
import logging
import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SemanticConfig:
    enabled: bool = True
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-base"
    semantic_top_k: int = 48
    encode_batch_size: int = 128
    # Cross-encoder scoring is intentionally opt-in on the full benchmark.
    rerank_top_k: int = 0
    rerank_batch_size: int = 32
    hnsw_m: int = 32
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 96
    cache_size: int = 20_000
    device: str = "auto"
    # Fail-open is the production default: lexical matching remains usable.
    allow_fallback: bool = True
    # Network downloads are opt-in. Local/cached model weights are preferred.
    allow_download: bool = False

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
            encode_batch_size=max(
                1,
                int(os.getenv("IDENTIFAI_BGE_BATCH_SIZE", str(cls.encode_batch_size))),
            ),
            rerank_top_k=max(
                0,
                int(os.getenv("IDENTIFAI_RERANK_TOP_K", str(cls.rerank_top_k))),
            ),
            rerank_batch_size=max(
                1,
                int(
                    os.getenv(
                        "IDENTIFAI_RERANK_BATCH_SIZE",
                        str(cls.rerank_batch_size),
                    )
                ),
            ),
            hnsw_m=max(
                4,
                int(os.getenv("IDENTIFAI_HNSW_M", str(cls.hnsw_m))),
            ),
            hnsw_ef_construction=max(
                8,
                int(
                    os.getenv(
                        "IDENTIFAI_HNSW_EF_CONSTRUCTION",
                        str(cls.hnsw_ef_construction),
                    )
                ),
            ),
            hnsw_ef_search=max(
                8,
                int(
                    os.getenv(
                        "IDENTIFAI_HNSW_EF_SEARCH",
                        str(cls.hnsw_ef_search),
                    )
                ),
            ),
            cache_size=max(
                0,
                int(os.getenv("IDENTIFAI_RERANK_CACHE", str(cls.cache_size))),
            ),
            device=os.getenv("IDENTIFAI_DEVICE", cls.device),
            allow_fallback=_bool("IDENTIFAI_SEMANTIC_FALLBACK", True),
            allow_download=_bool("IDENTIFAI_SEMANTIC_ALLOW_DOWNLOAD", False),
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
            "allow_download": self.allow_download,
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


def _construct_with_local_policy(factory, model_name: str, config: SemanticConfig):
    """Instantiate a Sentence-Transformers model without violating download policy.

    When downloads are disabled we must *never* retry by removing
    ``local_files_only``: doing so could silently start a Hugging Face network
    request on an offline training machine. Older/fake constructors that do not
    support the keyword therefore cause the semantic route to fail open instead.
    """
    kwargs = {"device": _device_name(config)}
    if config.allow_download:
        try:
            return factory(model_name, **kwargs)
        except TypeError as exc:
            # Older/fake implementations may reject the device keyword.
            if "device" not in str(exc):
                raise
            kwargs.pop("device", None)
            return factory(model_name, **kwargs)

    kwargs["local_files_only"] = True
    try:
        return factory(model_name, **kwargs)
    except TypeError as exc:
        if "local_files_only" not in str(exc):
            raise
        raise RuntimeError(
            "The installed Sentence-Transformers API does not support "
            "local_files_only; refusing to fall back to a network model download. "
            "Upgrade sentence-transformers or provide a compatible local model."
        ) from exc


def _score_to_probability(values: np.ndarray) -> np.ndarray:
    """Normalize CrossEncoder outputs to [0, 1] without assuming logits vs probabilities."""
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return values
    if float(np.min(values)) < 0.0 or float(np.max(values)) > 1.0:
        clipped = np.clip(values, -50.0, 50.0)
        values = 1.0 / (1.0 + np.exp(-clipped))
    return np.clip(values, 0.0, 1.0).astype(np.float32, copy=False)


class SemanticRetriever:
    """Reusable BGE + FAISS retrieval component with fail-open behavior."""

    def __init__(self, config: SemanticConfig):
        self.config = config
        self._encoder = None
        self._reranker = None
        self._rerank_cache: OrderedDict[str, float] = OrderedDict()
        self._index = None
        self._target_ids: list[str] = []
        self._using_real_backend = False
        self._backend = "disabled" if not config.enabled else "uninitialized"
        self._failure_reason = ""
        self._ready = not config.enabled
        self._reranker_disabled = False

    @property
    def using_real_backend(self) -> bool:
        return self._using_real_backend

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def failure_reason(self) -> str:
        return self._failure_reason

    def _degrade(self, reason: str, exc: Exception | None = None) -> None:
        """Disable only the semantic route; never damage the lexical pipeline."""
        self._encoder = False
        self._reranker = None
        self._index = None
        self._target_ids.clear()
        self._using_real_backend = False
        self._backend = "disabled_fallback"
        self._failure_reason = str(reason)
        self._ready = True
        if exc is None:
            LOGGER.warning("Semantic retrieval disabled: %s", reason)
        else:
            LOGGER.warning(
                "Semantic retrieval disabled: %s (%s: %s)",
                reason,
                type(exc).__name__,
                exc,
            )

    def _handle_required_failure(self, message: str, exc: Exception) -> None:
        if not self.config.allow_fallback:
            raise RuntimeError(message) from exc
        self._degrade(message, exc)

    def _load_encoder(self):
        if self._encoder is False:
            return None
        if self._encoder is not None:
            return self._encoder
        if not self.config.enabled:
            return None

        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:  # Import can fail for binary/dependency reasons too.
            self._handle_required_failure(
                "Semantic retrieval could not import sentence-transformers; "
                "the lexical pipeline will continue without semantic retrieval.",
                exc,
            )
            return None

        try:
            encoder = _construct_with_local_policy(
                SentenceTransformer,
                self.config.embedding_model,
                self.config,
            )
        except Exception as exc:
            mode = (
                "network download is disabled"
                if not self.config.allow_download
                else "model download/load failed"
            )
            self._handle_required_failure(
                f"Semantic embedding model could not be loaded ({mode}). "
                f"Model={self.config.embedding_model!r}. "
                "Provide cached/local model weights or enable "
                "IDENTIFAI_SEMANTIC_ALLOW_DOWNLOAD=1.",
                exc,
            )
            return None

        self._encoder = encoder
        self._using_real_backend = True
        self._backend = "bge_pending_faiss"
        LOGGER.info(
            "Loaded semantic encoder: %s%s",
            self.config.embedding_model,
            " (local-only)" if not self.config.allow_download else "",
        )
        return encoder

    @staticmethod
    def _normalize(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError(f"Expected a 2-D embedding matrix, got shape {values.shape}")
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        return np.divide(values, norms, out=np.zeros_like(values), where=norms > 1e-12)

    def _encode(self, texts: Sequence[str]) -> np.ndarray | None:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        encoder = self._load_encoder()
        if encoder is None or encoder is False:
            return None
        try:
            encoded = encoder.encode(
                list(texts),
                batch_size=self.config.encode_batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            matrix = self._normalize(np.asarray(encoded, dtype=np.float32))
            if matrix.shape[0] != len(texts):
                raise ValueError(
                    f"Encoder returned {matrix.shape[0]} rows for {len(texts)} texts"
                )
            return matrix
        except Exception as exc:
            self._handle_required_failure(
                "Semantic embedding failed during inference; "
                "the lexical pipeline will continue without semantic retrieval.",
                exc,
            )
            return None

    def _create_index(self, dimension: int):
        try:
            import faiss
        except Exception as exc:
            self._handle_required_failure(
                "FAISS could not be imported; semantic retrieval will be disabled.",
                exc,
            )
            return None

        try:
            index = faiss.IndexHNSWFlat(
                int(dimension),
                int(self.config.hnsw_m),
                faiss.METRIC_INNER_PRODUCT,
            )
            index.hnsw.efConstruction = int(self.config.hnsw_ef_construction)
            index.hnsw.efSearch = int(self.config.hnsw_ef_search)
            return index
        except Exception as exc:
            self._handle_required_failure(
                "FAISS HNSW index construction failed; semantic retrieval will be disabled.",
                exc,
            )
            return None

    def build(self, targets: Iterable[tuple[str, str]]) -> None:
        """Build the target semantic index from ``(entity_id, entity_text)`` rows."""
        if not self.config.enabled:
            self._ready = True
            self._backend = "disabled"
            return

        # We intentionally do not maintain an exact NumPy/HashingVectorizer
        # fallback over the complete target table. On a million-row benchmark that
        # would turn the fallback into an O(N*d) memory/time trap.
        batch_ids: list[str] = []
        batch_texts: list[str] = []

        def _disable_if_needed() -> bool:
            return self._backend == "disabled_fallback"

        def _flush() -> bool:
            if not batch_texts:
                return True
            embeddings = self._encode(batch_texts)
            if embeddings is None or _disable_if_needed():
                return False
            if self._index is None:
                self._index = self._create_index(embeddings.shape[1])
            if self._index is None or self._backend == "disabled_fallback":
                return False
            try:
                self._index.add(np.ascontiguousarray(embeddings, dtype=np.float32))
                self._target_ids.extend(batch_ids)
                batch_ids.clear()
                batch_texts.clear()
                return True
            except Exception as exc:
                self._handle_required_failure(
                    "FAISS rejected target embeddings during index construction; "
                    "semantic retrieval will be disabled.",
                    exc,
                )
                return False

        target_iter = iter(targets)
        try:
            for target_id, text in target_iter:
                if not target_id:
                    continue
                batch_ids.append(str(target_id))
                batch_texts.append(str(text))
                if len(batch_texts) >= self.config.encode_batch_size:
                    if not _flush():
                        batch_ids.clear()
                        batch_texts.clear()
                        break
            else:
                _flush()
        finally:
            # A caller may pass a generator backed by a SQLite cursor. Close it
            # explicitly, including the failure/early-exit path, so the caller can
            # safely drop temporary SQLite tables afterwards.
            close = getattr(target_iter, "close", None)
            if callable(close):
                close()
            batch_ids.clear()
            batch_texts.clear()

        if self._backend != "disabled_fallback":
            if self._index is None or not self._target_ids:
                # Empty target tables should not crash prediction.
                self._degrade(
                    "Semantic target index is empty; continuing with lexical retrieval only."
                )
            else:
                self._using_real_backend = True
                self._backend = "bge_faiss_hnsw"
                self._ready = True
                LOGGER.info(
                    "Semantic target index ready: %d targets (BGE+FAISS-HNSW)",
                    len(self._target_ids),
                )
        else:
            self._ready = True

    def query(
        self,
        sources: Iterable[tuple[str, str]],
        top_k: int | None = None,
    ) -> list[tuple[str, str, float, int]]:
        if (
            not self.config.enabled
            or not self._ready
            or self._backend != "bge_faiss_hnsw"
            or self._index is None
            or not self._target_ids
        ):
            return []

        k = max(1, int(top_k or self.config.semantic_top_k))
        k = min(k, len(self._target_ids))

        results: list[tuple[str, str, float, int]] = []
        source_ids: list[str] = []
        texts: list[str] = []

        def _flush() -> bool:
            if not texts:
                return True
            query_vectors = self._encode(texts)
            if query_vectors is None:
                return False
            try:
                distances, indices = self._index.search(
                    np.ascontiguousarray(query_vectors, dtype=np.float32),
                    k,
                )
            except Exception as exc:
                self._handle_required_failure(
                    "FAISS search failed; semantic retrieval will be disabled.",
                    exc,
                )
                return False

            for row_index, source_id in enumerate(source_ids):
                seen: set[int] = set()
                rank = 0
                for distance, target_index in zip(
                    distances[row_index],
                    indices[row_index],
                ):
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
            return True

        source_iter = iter(sources)
        try:
            for source_id, text in source_iter:
                source_ids.append(str(source_id))
                texts.append(str(text))
                if len(texts) >= self.config.encode_batch_size:
                    if not _flush():
                        break
            if source_ids:
                _flush()
        finally:
            # See the matching comment in build(): closing an upstream generator
            # is necessary when it wraps a database-backed cursor.
            close = getattr(source_iter, "close", None)
            if callable(close):
                close()
        return results

    def _load_reranker(self):
        if self._reranker_disabled:
            return None
        if self._reranker is not None:
            return self._reranker
        try:
            from sentence_transformers import CrossEncoder
        except Exception as exc:
            if self.config.allow_fallback:
                self._reranker_disabled = True
                LOGGER.warning(
                    "CrossEncoder is unavailable; reranking is disabled (%s: %s)",
                    type(exc).__name__,
                    exc,
                )
                return None
            raise RuntimeError("CrossEncoder could not be imported") from exc

        try:
            reranker = _construct_with_local_policy(
                CrossEncoder,
                self.config.reranker_model,
                self.config,
            )
        except Exception as exc:
            if self.config.allow_fallback:
                self._reranker_disabled = True
                LOGGER.warning(
                    "CrossEncoder model could not be loaded; reranking is disabled "
                    "(%s: %s)",
                    type(exc).__name__,
                    exc,
                )
                return None
            raise RuntimeError(
                f"CrossEncoder model {self.config.reranker_model!r} could not be loaded"
            ) from exc

        self._reranker = reranker
        LOGGER.info(
            "Loaded cross-encoder reranker: %s%s",
            self.config.reranker_model,
            " (local-only)" if not self.config.allow_download else "",
        )
        return reranker

    def rerank(self, pairs: Sequence[tuple[str, str, int]]) -> np.ndarray:
        """Return bounded cross-encoder scores for semantic candidates."""
        output = np.zeros(len(pairs), dtype=np.float32)
        if (
            not pairs
            or not self.config.enabled
            or self.config.rerank_top_k <= 0
            or self._backend != "bge_faiss_hnsw"
        ):
            return output

        eligible_indices = [
            i
            for i, (_, _, semantic_rank) in enumerate(pairs)
            if 1 <= int(semantic_rank) <= self.config.rerank_top_k
        ]
        if not eligible_indices:
            return output

        reranker = self._load_reranker()
        if reranker is None:
            return output

        to_score: list[tuple[int, str, str]] = []
        for index in eligible_indices:
            query, document, _rank = pairs[index]
            key = hashlib.blake2b(
                f"{self.config.reranker_model}\0{query}\0{document}".encode("utf-8"),
                digest_size=16,
            ).hexdigest()
            cached = self._rerank_cache.get(key)
            if cached is None:
                to_score.append((index, query, document))
            else:
                output[index] = float(cached)
                self._rerank_cache.move_to_end(key)

        if to_score:
            try:
                scores = reranker.predict(
                    [(query, document) for _, query, document in to_score],
                    batch_size=self.config.rerank_batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
                scores = _score_to_probability(np.asarray(scores))
            except Exception as exc:
                if self.config.allow_fallback:
                    self._reranker_disabled = True
                    LOGGER.warning(
                        "CrossEncoder inference failed; reranking is disabled "
                        "for the remainder of this run (%s: %s)",
                        type(exc).__name__,
                        exc,
                    )
                    return output
                raise RuntimeError("CrossEncoder inference failed") from exc

            for (index, query, document), score in zip(to_score, scores):
                value = float(score)
                output[index] = value
                key = hashlib.blake2b(
                    f"{self.config.reranker_model}\0{query}\0{document}".encode("utf-8"),
                    digest_size=16,
                ).hexdigest()
                if self.config.cache_size > 0:
                    self._rerank_cache[key] = value
                    self._rerank_cache.move_to_end(key)
                    while len(self._rerank_cache) > self.config.cache_size:
                        self._rerank_cache.popitem(last=False)

        return output

    def close(self) -> None:
        self._index = None
        self._target_ids.clear()
        self._encoder = None
        self._reranker = None
        self._rerank_cache.clear()
        self._ready = False
        self._using_real_backend = False


__all__ = ["SemanticConfig", "SemanticRetriever", "entity_text"]
