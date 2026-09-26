Business Entity Resolution Pipeline — Methodology

Objective

Resolve each Source-1 entity to zero, one, or multiple Source-2/Source-3 entities while preserving the submission contract. The optimization metric is macro F0.5 over Source-1 entities.

Data handling

The pipeline reads only the supplied challenge TSV files. It does not call external entity databases, geocoders, registries, business APIs, or web enrichment services. Git-LFS pointer files are rejected before processing.

Preprocessing

Names are Unicode-normalized, case-folded, and canonicalized for common legal-form variants. Addresses are normalized with conservative aliases, structured components, house-number extraction, postal-code extraction, and landmark filler removal. The normalization rules are deterministic and data-independent.

Candidate generation

Candidate generation combines deterministic lexical blocking with an independent semantic retrieval route. Lexical blocking uses exact and near-exact keys, rare-token filtering, phonetic keys, address structure, and MinHash/LSH. Oversized blocks are handled by a relevance-aware rescue stage instead of arbitrary ID truncation.

The semantic route uses a Sentence-Transformers BGE bi-encoder, normalized embeddings, and a FAISS HNSW inner-product index. For each Source-1 entity it retrieves a bounded semantic neighborhood and unions those candidates with the lexical candidates.

The final candidate set has a configurable hard TOP_K ceiling. Those exact final candidates are the only pairs materialized into the feature matrix and the only pairs eligible for prediction.

Features

The matcher uses lexical, structured-address, country, retrieval-rank, reciprocal-rank, cross-source corroboration, learned variation, semantic similarity, semantic rank, semantic/lexical alignment, and bounded cross-encoder reranking features.

Matching model

The model layer supports LightGBM, a linear logistic model, and a calibrated probability ensemble. The tuning stage measures model variants on the leakage-safe validation splits and optimizes the grouped macro-F0.5 threshold on the same Source-1 grouping used by the metric.

Inference uses the tuned threshold through a single entity-level policy. No post-tuning margin or threshold offset is inserted unless it was explicitly measured during tuning.

Cache correctness

SQLite candidate stores and feature matrices are invalidated when dataset signatures, feature counts, semantic configuration, or relevant source-module hashes change. Materialized matrices carry sidecar metadata so a stale feature layout cannot be silently reused.

Reproducibility

All behavior is controlled by code and configuration, not entity-specific IDs or hand-written exceptions. Semantic model names, ANN settings, batch sizes, reranking depth, and device selection can be overridden through environment variables.

Runtime requirements

The production semantic path requires sentence-transformers and faiss-cpu from requirements.txt. For deterministic unit tests only, semantic retrieval is disabled in the test configuration.

Evaluation artifacts

Real-data validation measurements are written to scratch/validation_report.json and tuning results to scratch/tuned_policy.json. Repository documentation intentionally does not copy those measurements so that synthetic smoke-test results cannot be mistaken for challenge results.