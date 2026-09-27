Business Entity Resolution Pipeline

This is the production entrypoint for the ML Challenge entity-resolution workflow.

1. Setup

Create a clean environment and install the core pipeline:

python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

python -m pip install -r code/business_entity_resolution/requirements.txt

The semantic layer is optional at installation time. To install BGE + FAISS support:

python -m pip install -r code/business_entity_resolution/requirements-semantic.txt

The pipeline is deliberately fail-open: if semantic dependencies or cached model weights are unavailable, lexical blocking and the ML matcher continue instead of crashing.

2. Data layout

The CLI expects a materialized challenge dataset directory containing:

train_source1.tsv
train_source2.tsv
train_source3.tsv
train_ground_truth.tsv   # train only

Test prediction additionally needs:

test_source1.tsv
test_source2.tsv
test_source3.tsv

Git-LFS pointer files are rejected explicitly. Pass the real materialized dataset with --data_dir / --train_dir / --test_dir.

3. Recommended workflow

Run the measured baseline first:

python run.py --mode validate --data_dir student_resource/dataset/train

Then tune:

python run.py --mode tune --data_dir student_resource/dataset/train

Then produce the submission:

python run.py   --mode predict   --train_dir student_resource/dataset/train   --test_dir student_resource/dataset/test   --check-submission

Prediction is intentionally blocked until a completed baseline and tuned policy exist. Use --allow-baseline-predict only for debugging.

4. Semantic retrieval safety

BGE retrieval is enabled by default, but model weights are local-only by default:

IDENTIFAI_SEMANTIC_ENABLED=1
IDENTIFAI_SEMANTIC_ALLOW_DOWNLOAD=0

This means an offline machine never stalls on a Hugging Face download. If a compatible local/cached BGE model is unavailable, the semantic route is disabled and lexical blocking remains active.

To permit model downloads in an environment where network access is explicitly available:

IDENTIFAI_SEMANTIC_ALLOW_DOWNLOAD=1

You can also point the model setting directly at a local model directory:

IDENTIFAI_BGE_MODEL=C:\models\bge-base-en-v1.5

Strict semantic mode is available only when diagnosing deployment problems:

IDENTIFAI_SEMANTIC_FALLBACK=0

The default is fail-open.

5. Cross-encoder reranking

The cross-encoder path remains implemented, but it is disabled by default:

IDENTIFAI_RERANK_TOP_K=0

This is intentional. On a benchmark with millions of entities, running a transformer cross-encoder over even a handful of pairs per entity can dominate runtime. Enable it only after the local reranker weights are available and its runtime/accuracy trade-off has been measured:

IDENTIFAI_RERANK_TOP_K=1
IDENTIFAI_RERANK_MODEL=C:\models\bge-reranker-base

A reranker import, model-load, or inference failure also degrades safely to a zero reranker feature instead of aborting the run.

6. Outputs

Prediction writes:

output/matching_results.tsv
output/candidate_pairs.tsv

The candidate file is the exact final candidate set produced by blocking. Every final match is a member of that candidate set.

The output writer emits exactly one row per Source-1 entity, including an empty match list when no target is selected.

7. Submission validation

--check-submission runs the bundled standard-library validator without loading the entire S2/S3 universe into memory.

For the additional, memory-heavy target-ID existence check:

python run.py ... --check-submission --check-submission-ids

Use the latter only as a diagnostic on large test data.

8. Caches

Blocking and materialized feature caches are invalidated when relevant dataset, feature, model configuration, semantic configuration, or source-code signatures change.

Generated files belong under scratch/ and output/; do not commit them.

9. Tests

From the repository root:

python -m pytest -q

The test bootstrap also makes the package importable when pytest is launched from code/business_entity_resolution.

Unit tests never download transformer models.