"""Optional, memory-bounded semantic retrieval for business entity resolution.

The semantic route is deliberately fail-open and resource-aware:

* deterministic lexical blocking remains the primary retrieval path;
* BGE embeddings are used only when the optional packages and local/cached model
  weights are available;
* network model downloads are disabled by default;
* the production FAISS index is IVF+PQ, not HNSWFlat, so the full 10M+ target
  table does not require tens of gigabytes of RAM;
* target IDs are stored in a fixed-width disk-backed memmap instead of a Python
  list when the dataset is large;
* semantic query results are streamed in batches, so millions of results are
  never accumulated in one Python list;
* model/index/resource failures disable only the semantic route and leave lexical
  training/prediction fully operational;
* the optional CrossEncoder is disabled by default and is bounded by an explicit
  top-k, because reranking every candidate is not a scalable operation.
"""
from __future__ import annotations

import hashlib
import logging
import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np

LOGGER = logging.getLogger(__name__)

_GIB = 1024 ** 3


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().casefold() not in {"0", "false", "no", "off"}


def _available_memory_bytes() -> int | None:
    """Best-effort available-RAM probe, including Linux container limits."""
    try:
        if os.name == "nt":
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_uint32),
                    ("dwMemoryLoad", ctypes.c_uint32),
                    ("ullTotalPhys", ctypes.c_uint64),
                    ("ullAvailPhys", ctypes.c_uint64),
                    ("ullTotalPageFile", ctypes.c_uint64),
                    ("ullAvailPageFile", ctypes.c_uint64),
                    ("ullTotalVirtual", ctypes.c_uint64),
                    ("ullAvailVirtual", ctypes.c_uint64),
                    ("sullAvailExtendedVirtual", ctypes.c_uint64),
                ]

            status = MemoryStatus()
            status.dwLength = ctypes.sizeof(MemoryStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return int(status.ullAvailPhys)

        # cgroup v2 is the common Linux container memory controller. Respecting
        # it is essential: /proc/meminfo may expose host RAM even when the process
        # itself is limited to a much smaller container budget.
        cgroup_v2 = Path("/sys/fs/cgroup")
        max_file = cgroup_v2 / "memory.max"
        current_file = cgroup_v2 / "memory.current"
        if max_file.exists() and current_file.exists():
            raw_max = max_file.read_text(encoding="utf-8").strip()
            if raw_max != "max":
                limit = int(raw_max)
                current = int(current_file.read_text(encoding="utf-8").strip())
                return max(0, limit - current)

        # cgroup v1 fallback.
        v1_limit = Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")
        v1_usage = Path("/sys/fs/cgroup/memory/memory.usage_in_bytes")
        if v1_limit.exists() and v1_usage.exists():
            limit = int(v1_limit.read_text(encoding="utf-8").strip())
            usage = int(v1_usage.read_text(encoding="utf-8").strip())
            # A value near uint64 max means there is effectively no cgroup limit.
            if limit < (1 << 60):
                return max(0, limit - usage)

        meminfo = Path("/proc/meminfo")
        if meminfo.exists():
            values: dict[str, int] = {}
            for line in meminfo.read_text(encoding="utf-8").splitlines():
                key, _, raw = line.partition(":")
                if not raw:
                    continue
                pieces = raw.strip().split()
                if pieces and pieces[0].isdigit():
                    values[key] = int(pieces[0]) * 1024
            if "MemAvailable" in values:
                return int(values["MemAvailable"])
    except Exception:
        return None
    return None


@dataclass(frozen=True)
class SemanticConfig:
    enabled: bool = True
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-base"
    semantic_top_k: int = 32
    encode_batch_size: int = 128
    # Cross-encoder scoring is opt-in. 0 means disabled.
    rerank_top_k: int = 0
    rerank_batch_size: int = 32
    # Memory-bounded ANN: IVF + product quantization.
    index_kind: str = "ivfpq"
    ivf_nlist: int = 0  # 0 = derive from target count.
    ivf_nprobe: int = 16
    pq_m: int = 32
    pq_nbits: int = 8
    train_sample_size: int = 100_000
    # Resource guard. Semantic retrieval must never be allowed to consume the
    # majority of RAM required by the rest of the ER pipeline.
    min_free_gb: float = 4.0
    max_available_fraction: float = 0.70
    model_memory_gb: float = 1.5
    # Fixed-width on-disk target-id store is enabled for large indexes.
    id_memmap_threshold: int = 500_000
    device: str = "auto"
    allow_fallback: bool = True
    allow_download: bool = False
    cache_size: int = 20_000
    # Only entities with a small lexical candidate set are queried semantically.
    # Semantic retrieval is a rescue route, not a second full Cartesian block.
    semantic_trigger_candidates: int = 24
    semantic_max_queries: int = 0  # 0 = unlimited
    embedding_dimension_hint: int = 768

    @classmethod
    def from_environment(cls, *, top_k: int | None = None) -> "SemanticConfig":
        semantic_top_k = int(os.getenv("IDENTIFAI_SEMANTIC_TOP_K", "48"))
        if top_k is not None:
            semantic_top_k = max(semantic_top_k, min(64, int(top_k)))

        return cls(
            enabled=_bool_env("IDENTIFAI_SEMANTIC_ENABLED", True),
            embedding_model=os.getenv("IDENTIFAI_BGE_MODEL", cls.embedding_model),
            reranker_model=os.getenv("IDENTIFAI_RERANKER_MODEL", cls.reranker_model),
            semantic_top_k=max(1, semantic_top_k),
            encode_batch_size=max(1, int(os.getenv("IDENTIFAI_BGE_BATCH_SIZE", str(cls.encode_batch_size)))),
            rerank_top_k=max(0, int(os.getenv("IDENTIFAI_RERANK_TOP_K", str(cls.rerank_top_k)))),
            rerank_batch_size=max(1, int(os.getenv("IDENTIFAI_RERANK_BATCH_SIZE", str(cls.rerank_batch_size)))),
            index_kind=os.getenv("IDENTIFAI_SEMANTIC_INDEX", cls.index_kind).casefold(),
            ivf_nlist=max(0, int(os.getenv("IDENTIFAI_IVF_NLIST", str(cls.ivf_nlist)))),
            ivf_nprobe=max(1, int(os.getenv("IDENTIFAI_IVF_NPROBE", str(cls.ivf_nprobe)))),
            pq_m=max(1, int(os.getenv("IDENTIFAI_PQ_M", str(cls.pq_m)))),
            pq_nbits=min(8, max(4, int(os.getenv("IDENTIFAI_PQ_NBITS", str(cls.pq_nbits))))),
            train_sample_size=max(10_000, int(os.getenv("IDENTIFAI_SEMANTIC_TRAIN_SAMPLE", str(cls.train_sample_size)))),
            min_free_gb=max(0.5, float(os.getenv("IDENTIFAI_SEMANTIC_MIN_FREE_GB", str(cls.min_free_gb)))),
            max_available_fraction=min(0.90, max(0.25, float(os.getenv("IDENTIFAI_SEMANTIC_MAX_RAM_FRACTION", str(cls.max_available_fraction))))),
            model_memory_gb=max(0.25, float(os.getenv("IDENTIFAI_SEMANTIC_MODEL_GB", str(cls.model_memory_gb)))),
            id_memmap_threshold=max(1, int(os.getenv("IDENTIFAI_SEMANTIC_ID_MEMMAP_THRESHOLD", str(cls.id_memmap_threshold)))),
            device=os.getenv("IDENTIFAI_DEVICE", cls.device),
            allow_fallback=_bool_env("IDENTIFAI_SEMANTIC_FALLBACK", True),
            allow_download=_bool_env("IDENTIFAI_SEMANTIC_ALLOW_DOWNLOAD", False),
            cache_size=max(0, int(os.getenv("IDENTIFAI_RERANK_CACHE", str(cls.cache_size)))),
            semantic_trigger_candidates=max(0, int(os.getenv("IDENTIFAI_SEMANTIC_TRIGGER_CANDIDATES", str(cls.semantic_trigger_candidates)))),
            semantic_max_queries=max(0, int(os.getenv("IDENTIFAI_SEMANTIC_MAX_QUERIES", str(cls.semantic_max_queries)))),
            embedding_dimension_hint=max(1, int(os.getenv("IDENTIFAI_BGE_DIM_HINT", str(cls.embedding_dimension_hint)))),
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
            "index_kind": self.index_kind,
            "ivf_nlist": self.ivf_nlist,
            "ivf_nprobe": self.ivf_nprobe,
            "pq_m": self.pq_m,
            "pq_nbits": self.pq_nbits,
            "train_sample_size": self.train_sample_size,
            "min_free_gb": self.min_free_gb,
            "max_available_fraction": self.max_available_fraction,
            "model_memory_gb": self.model_memory_gb,
            "id_memmap_threshold": self.id_memmap_threshold,
            "device": self.device,
            "allow_fallback": self.allow_fallback,
            "allow_download": self.allow_download,
            "cache_size": self.cache_size,
            "semantic_trigger_candidates": self.semantic_trigger_candidates,
            "semantic_max_queries": self.semantic_max_queries,
            "embedding_dimension_hint": self.embedding_dimension_hint,
        }


def entity_text(name: str, address: str, country: str) -> str:
    parts: list[str] = []
    if name:
        parts.append(f"business name: {name}")
    if address:
        parts.append(f"business address: {address}")
    if country:
        parts.append(f"country: {country}")
    return "; ".join(parts)


def _device_name(config: SemanticConfig) -> str | None:
    return None if config.device.casefold() == "auto" else config.device


def _construct_with_local_policy(factory, model_name: str, config: SemanticConfig):
    """Instantiate a model without ever turning a local-only failure into a download."""
    kwargs: dict[str, object] = {"device": _device_name(config)}
    if config.allow_download:
        try:
            return factory(model_name, **kwargs)
        except TypeError as exc:
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
            "Installed sentence-transformers does not support local_files_only; "
            "refusing to retry without it because that could start a network download."
        ) from exc


def _score_to_probability(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return values
    if float(np.min(values)) < 0.0 or float(np.max(values)) > 1.0:
        values = 1.0 / (1.0 + np.exp(-np.clip(values, -50.0, 50.0)))
    return np.clip(values, 0.0, 1.0).astype(np.float32, copy=False)


class SemanticRetriever:
    """Memory-bounded BGE + FAISS retriever with fail-open behavior."""

    def __init__(self, config: SemanticConfig, id_storage_path: str | Path | None = None):
        self.config = config
        self._encoder = None
        self._reranker = None
        self._rerank_cache: OrderedDict[str, float] = OrderedDict()
        self._index = None
        self._target_ids: list[str] = []
        self._target_ids_memmap: np.memmap | None = None
        self._target_id_width = 0
        self._target_count = 0
        self._id_storage_path = Path(id_storage_path) if id_storage_path else None
        self._using_real_backend = False
        self._backend = "disabled" if not config.enabled else "uninitialized"
        self._failure_reason = ""
        self._ready = not config.enabled
        self._reranker_disabled = False
        self._dimension = 0
        self._nlist = 0
        self._pq_m = 0
        self._faiss = None

    @property
    def using_real_backend(self) -> bool:
        return self._using_real_backend

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def failure_reason(self) -> str:
        return self._failure_reason

    @property
    def index_stats(self) -> dict[str, int]:
        return {
            "dimension": int(self._dimension),
            "nlist": int(self._nlist),
            "nprobe": int(self.config.ivf_nprobe),
            "pq_m": int(self._pq_m),
            "pq_nbits": int(self.config.pq_nbits),
            "indexed_targets": int(self._index.ntotal) if self._index is not None else 0,
        }

    def _degrade(self, reason: str, exc: Exception | None = None) -> None:
        self._encoder = False
        self._reranker = None
        self._index = None
        self._target_ids.clear()
        self._close_id_store(remove_file=True)
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

    def preflight(self, target_count: int, target_id_width: int = 64) -> bool:
        """Reject semantic indexing before model loading when RAM is insufficient."""
        if not self.config.enabled:
            self._ready = True
            self._backend = "disabled"
            return False
        target_count = int(target_count)
        if target_count <= 0:
            self._degrade("Semantic target table is empty.")
            return False

        available = _available_memory_bytes()
        if available is None:
            return True

        nlist = self._derive_nlist(target_count)
        code_bytes = max(1, self.config.pq_m * self.config.pq_nbits // 8)
        # IVF-PQ stores code + 64-bit vector id. We also budget for the training
        # sample, model weights, the encoder's temporary activation buffers, and
        # the fixed-width ID memmap page cache.
        index_bytes = target_count * (code_bytes + 8)
        sample_rows = min(
            self.config.train_sample_size,
            max(50_000, 39 * nlist),
            target_count,
        )
        sample_bytes = sample_rows * self.config.embedding_dimension_hint * 4
        id_width = max(8, int(target_id_width))
        id_bytes = target_count * id_width
        estimated = index_bytes + sample_bytes + id_bytes + int(self.config.model_memory_gb * _GIB)
        floor = int(self.config.min_free_gb * _GIB)
        if available < floor:
            self._degrade(
                "Semantic IVF-PQ preflight disabled semantic retrieval because available "
                f"RAM ({available / _GIB:.2f} GiB) is below the configured safety floor "
                f"({self.config.min_free_gb:.2f} GiB)."
            )
            return False
        allowed = int(available * self.config.max_available_fraction)

        if estimated > allowed:
            self._degrade(
                "Semantic IVF-PQ preflight rejected the index because the estimated "
                f"working set ({estimated / _GIB:.2f} GiB) exceeds the safe RAM budget "
                f"({allowed / _GIB:.2f} GiB)."
            )
            return False
        return True

    def _load_encoder(self):
        if self._encoder is False:
            return None
        if self._encoder is not None:
            return self._encoder
        if not self.config.enabled:
            return None
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:
            self._handle_required_failure(
                "sentence-transformers could not be imported; continuing lexically.",
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
            mode = "network download is disabled" if not self.config.allow_download else "model download/load failed"
            self._handle_required_failure(
                f"Semantic embedding model could not be loaded ({mode}). "
                f"Model={self.config.embedding_model!r}.",
                exc,
            )
            return None
        self._encoder = encoder
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
            if self._dimension == 0:
                self._dimension = int(matrix.shape[1])
            return matrix
        except Exception as exc:
            self._handle_required_failure(
                "Semantic embedding failed during inference; continuing lexically.",
                exc,
            )
            return None

    def _derive_nlist(self, target_count: int) -> int:
        if self.config.ivf_nlist > 0:
            return min(int(self.config.ivf_nlist), max(1, target_count))
        # 1024 is deliberately conservative for a CPU benchmark with millions of
        # vectors. We need enough training samples for the IVF k-means stage.
        return min(1024, max(16, int(np.sqrt(max(1, target_count)))))

    @staticmethod
    def _choose_pq_m(dimension: int, requested: int) -> int:
        upper = max(1, min(int(requested), int(dimension), 64))
        divisors = [value for value in range(1, upper + 1) if dimension % value == 0]
        return max(divisors) if divisors else 1

    def _load_faiss(self):
        if self._faiss is not None:
            return self._faiss
        try:
            import faiss
        except Exception as exc:
            self._handle_required_failure(
                "FAISS could not be imported; continuing lexically.",
                exc,
            )
            return None
        self._faiss = faiss
        return faiss

    def _create_index(self, dimension: int, target_count: int, training_vectors: np.ndarray):
        faiss = self._load_faiss()
        if faiss is None:
            return None

        self._dimension = int(dimension)
        self._nlist = self._derive_nlist(target_count)
        self._pq_m = self._choose_pq_m(dimension, self.config.pq_m)

        try:
            if target_count < 2_048 or self.config.index_kind == "flat":
                index = faiss.IndexFlatIP(int(dimension))
                return index

            if self.config.index_kind != "ivfpq":
                raise ValueError(
                    f"Unsupported semantic index kind {self.config.index_kind!r}; expected ivfpq or flat"
                )

            quantizer = faiss.IndexFlatIP(int(dimension))
            index = faiss.IndexIVFPQ(
                quantizer,
                int(dimension),
                int(self._nlist),
                int(self._pq_m),
                int(self.config.pq_nbits),
                faiss.METRIC_INNER_PRODUCT,
            )
            # Do not let precomputed tables silently consume GBs of additional RAM.
            if hasattr(index, "use_precomputed_table"):
                index.use_precomputed_table = 0
            if hasattr(index, "nprobe"):
                index.nprobe = min(int(self.config.ivf_nprobe), int(self._nlist))
            if not index.is_trained:
                index.train(np.ascontiguousarray(training_vectors, dtype=np.float32))
            return index
        except Exception as exc:
            self._handle_required_failure(
                "FAISS memory-bounded semantic index construction/training failed; continuing lexically.",
                exc,
            )
            return None

    def _close_id_store(self, remove_file: bool) -> None:
        try:
            if self._target_ids_memmap is not None:
                self._target_ids_memmap.flush()
        except Exception:
            pass
        self._target_ids_memmap = None
        if remove_file and self._id_storage_path is not None:
            try:
                self._id_storage_path.unlink(missing_ok=True)
            except Exception:
                pass

    def _init_id_store(self, target_count: int, width: int) -> None:
        self._target_count = int(target_count)
        self._target_id_width = int(width)
        if (
            self._id_storage_path is not None
            and target_count >= self.config.id_memmap_threshold
        ):
            self._id_storage_path.parent.mkdir(parents=True, exist_ok=True)
            self._target_ids_memmap = np.memmap(
                self._id_storage_path,
                dtype=f"S{self._target_id_width}",
                mode="w+",
                shape=(target_count,),
            )
            self._target_ids = []
        else:
            self._target_ids = []
            self._target_ids_memmap = None

    def _store_ids(self, start: int, values: Sequence[str]) -> None:
        if self._target_ids_memmap is not None:
            encoded = np.asarray([value.encode("utf-8") for value in values], dtype=f"S{self._target_id_width}")
            self._target_ids_memmap[start : start + len(encoded)] = encoded
        else:
            self._target_ids.extend(values)

    def _target_ids_for_indices(self, indices: Sequence[int]) -> list[str]:
        if not indices:
            return []
        if self._target_ids_memmap is None:
            return [self._target_ids[int(index)] for index in indices]
        values = self._target_ids_memmap[np.asarray(indices, dtype=np.int64)]
        return [bytes(value).rstrip(b"\x00").decode("utf-8", errors="replace") for value in values]

    def _target_id_at(self, index: int) -> str:
        return self._target_ids_for_indices([index])[0]

    def build(
        self,
        targets: Iterable[tuple[str, str]],
        *,
        target_count: int | None = None,
        target_id_width: int | None = None,
    ) -> None:
        """Build the target ANN index from a streaming ``(entity_id, text)`` source."""
        if not self.config.enabled:
            self._ready = True
            self._backend = "disabled"
            return

        if target_count is not None and not self.preflight(int(target_count), target_id_width or 64):
            return

        encoder = self._load_encoder()
        if encoder is None or self._backend == "disabled_fallback":
            return
        faiss = self._load_faiss()
        if faiss is None or self._backend == "disabled_fallback":
            return

        batch_ids: list[str] = []
        batch_texts: list[str] = []
        sample_ids: list[str] = []
        sample_texts: list[str] = []
        sample_vectors: np.ndarray | None = None
        inserted = 0
        expected_count = int(target_count) if target_count is not None else None
        id_width = int(target_id_width or 64)
        if expected_count is not None:
            self._init_id_store(expected_count, max(8, id_width))

        target_iter = iter(targets)

        def _encode_sample() -> bool:
            nonlocal sample_vectors
            if sample_vectors is not None:
                return True
            if not sample_texts:
                return False
            matrix = self._encode(sample_texts)
            if matrix is None:
                return False
            sample_vectors = matrix
            return True

        def _ensure_index() -> bool:
            if self._index is not None:
                return True
            if not _encode_sample():
                return False
            index = self._create_index(
                int(sample_vectors.shape[1]),
                int(expected_count or max(len(sample_ids), len(sample_texts))),
                sample_vectors,
            )
            if index is None:
                return False
            self._index = index
            return True

        try:
            # The first bounded sample supplies IVF/PQ training vectors. The rest
            # of the target stream is encoded and added incrementally.
            sample_limit = min(
                self.config.train_sample_size,
                expected_count if expected_count is not None else self.config.train_sample_size,
            )
            while len(sample_texts) < sample_limit:
                try:
                    target_id, text = next(target_iter)
                except StopIteration:
                    break
                if not target_id:
                    continue
                encoded_id = str(target_id)
                if len(encoded_id.encode("utf-8")) + 1 > id_width:
                    raise ValueError("Target entity_id exceeds configured semantic ID width")
                sample_ids.append(encoded_id)
                sample_texts.append(str(text))

            if not sample_texts or not _ensure_index():
                self._degrade("Semantic target sample is empty or could not initialize the ANN index.")
                return

            # Add the sample vectors first; they have already been encoded once for
            # IVF/PQ training, avoiding a second expensive BGE pass over the sample.
            self._index.add(np.ascontiguousarray(sample_vectors, dtype=np.float32))
            self._store_ids(0, sample_ids)
            inserted = len(sample_ids)
            sample_ids.clear()
            sample_texts.clear()
            sample_vectors = None

            def flush_batch() -> bool:
                nonlocal inserted
                if not batch_texts:
                    return True
                matrix = self._encode(batch_texts)
                if matrix is None or self._backend == "disabled_fallback":
                    return False
                self._index.add(np.ascontiguousarray(matrix, dtype=np.float32))
                self._store_ids(inserted, batch_ids)
                inserted += len(batch_ids)
                batch_ids.clear()
                batch_texts.clear()
                return True

            for target_id, text in target_iter:
                if not target_id:
                    continue
                clean_id = str(target_id)
                if len(clean_id.encode("utf-8")) + 1 > id_width:
                    raise ValueError("Target entity_id exceeds configured semantic ID width")
                batch_ids.append(clean_id)
                batch_texts.append(str(text))
                if len(batch_texts) >= self.config.encode_batch_size:
                    if not flush_batch():
                        return
            if not flush_batch():
                return

            if expected_count is not None and inserted != expected_count:
                raise ValueError(
                    f"Semantic target count mismatch: indexed {inserted}, expected {expected_count}"
                )

            self._using_real_backend = True
            self._backend = "bge_faiss_ivfpq" if self._nlist > 0 and self._index.__class__.__name__ != "IndexFlatIP" else "bge_faiss_flat"
            self._ready = True
            LOGGER.info(
                "Semantic target index ready: %d targets (%s, d=%d, nlist=%d, pq_m=%d)",
                inserted,
                self._backend,
                self._dimension,
                self._nlist,
                self._pq_m,
            )
        except Exception as exc:
            self._handle_required_failure(
                "Semantic target index construction failed; continuing with lexical retrieval.",
                exc,
            )
        finally:
            close = getattr(target_iter, "close", None)
            if callable(close):
                close()
            batch_ids.clear()
            batch_texts.clear()
            sample_ids.clear()
            sample_texts.clear()

    def iter_query(
        self,
        sources: Iterable[tuple[str, str]],
        top_k: int | None = None,
    ) -> Iterator[tuple[str, str, float, int]]:
        if (
            not self.config.enabled
            or not self._ready
            or not self._using_real_backend
            or self._index is None
            or self._target_count == 0 and not self._target_ids
        ):
            return

        indexed_count = int(self._index.ntotal)
        if indexed_count <= 0:
            return
        k = min(max(1, int(top_k or self.config.semantic_top_k)), indexed_count)
        source_ids: list[str] = []
        texts: list[str] = []

        try:
            for source_id, text in sources:
                source_ids.append(str(source_id))
                texts.append(str(text))
                if len(texts) < self.config.encode_batch_size:
                    continue

                query_vectors = self._encode(texts)
                if query_vectors is None:
                    return
                try:
                    distances, indices = self._index.search(
                        np.ascontiguousarray(query_vectors, dtype=np.float32),
                        k,
                    )
                except Exception as exc:
                    self._handle_required_failure(
                        "FAISS semantic search failed; continuing lexically.",
                        exc,
                    )
                    return

                for row_index, source_id in enumerate(source_ids):
                    valid_pairs: list[tuple[int, float]] = []
                    seen: set[int] = set()
                    for distance, target_index in zip(distances[row_index], indices[row_index]):
                        target_index = int(target_index)
                        if target_index < 0 or target_index >= indexed_count or target_index in seen:
                            continue
                        seen.add(target_index)
                        valid_pairs.append((target_index, float(distance)))
                    target_names = self._target_ids_for_indices([index for index, _ in valid_pairs])
                    for rank, ((_, distance), target_name) in enumerate(zip(valid_pairs, target_names), start=1):
                        yield (source_id, target_name, distance, rank)
                source_ids.clear()
                texts.clear()

            if texts:
                query_vectors = self._encode(texts)
                if query_vectors is None:
                    return
                distances, indices = self._index.search(
                    np.ascontiguousarray(query_vectors, dtype=np.float32),
                    k,
                )
                for row_index, source_id in enumerate(source_ids):
                    valid_pairs: list[tuple[int, float]] = []
                    seen: set[int] = set()
                    for distance, target_index in zip(distances[row_index], indices[row_index]):
                        target_index = int(target_index)
                        if target_index < 0 or target_index >= indexed_count or target_index in seen:
                            continue
                        seen.add(target_index)
                        valid_pairs.append((target_index, float(distance)))
                    target_names = self._target_ids_for_indices([index for index, _ in valid_pairs])
                    for rank, ((_, distance), target_name) in enumerate(zip(valid_pairs, target_names), start=1):
                        yield (source_id, target_name, distance, rank)
        finally:
            close = getattr(sources, "close", None)
            if callable(close):
                close()

    def query(
        self,
        sources: Iterable[tuple[str, str]],
        top_k: int | None = None,
    ) -> list[tuple[str, str, float, int]]:
        """Compatibility wrapper; production blocking uses ``iter_query``."""
        return list(self.iter_query(sources, top_k=top_k))

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
                LOGGER.warning("CrossEncoder unavailable; reranking disabled (%s: %s)", type(exc).__name__, exc)
                return None
            raise RuntimeError("CrossEncoder could not be imported") from exc
        try:
            self._reranker = _construct_with_local_policy(
                CrossEncoder,
                self.config.reranker_model,
                self.config,
            )
        except Exception as exc:
            if self.config.allow_fallback:
                self._reranker_disabled = True
                LOGGER.warning("CrossEncoder load failed; reranking disabled (%s: %s)", type(exc).__name__, exc)
                return None
            raise RuntimeError("CrossEncoder model could not be loaded") from exc
        return self._reranker

    def rerank(self, pairs: Sequence[tuple[str, str, int]]) -> np.ndarray:
        output = np.zeros(len(pairs), dtype=np.float32)
        if (
            not pairs
            or not self.config.enabled
            or self.config.rerank_top_k <= 0
            or not self._using_real_backend
        ):
            return output

        eligible = [i for i, (_, _, rank) in enumerate(pairs) if 1 <= int(rank) <= self.config.rerank_top_k]
        if not eligible:
            return output
        reranker = self._load_reranker()
        if reranker is None:
            return output

        pending: list[tuple[int, str, str, str]] = []
        for index in eligible:
            query, document, _ = pairs[index]
            key = hashlib.blake2b(
                f"{self.config.reranker_model}\0{query}\0{document}".encode("utf-8"),
                digest_size=16,
            ).hexdigest()
            cached = self._rerank_cache.get(key)
            if cached is None:
                pending.append((index, query, document, key))
            else:
                output[index] = float(cached)
                self._rerank_cache.move_to_end(key)

        if pending:
            try:
                values = reranker.predict(
                    [(q, d) for _, q, d, _ in pending],
                    batch_size=self.config.rerank_batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
                scores = _score_to_probability(np.asarray(values))
            except Exception as exc:
                if self.config.allow_fallback:
                    self._reranker_disabled = True
                    LOGGER.warning("CrossEncoder inference failed; reranking disabled (%s: %s)", type(exc).__name__, exc)
                    return output
                raise RuntimeError("CrossEncoder inference failed") from exc

            for (index, _query, _doc, key), score in zip(pending, scores):
                value = float(score)
                output[index] = value
                if self.config.cache_size > 0:
                    self._rerank_cache[key] = value
                    self._rerank_cache.move_to_end(key)
                    while len(self._rerank_cache) > self.config.cache_size:
                        self._rerank_cache.popitem(last=False)

        return output

    def close(self) -> None:
        self._index = None
        self._target_ids.clear()
        self._close_id_store(remove_file=True)
        self._encoder = None
        self._reranker = None
        self._rerank_cache.clear()
        self._ready = False
        self._using_real_backend = False
        self._faiss = None


__all__ = ["SemanticConfig", "SemanticRetriever", "entity_text"]
