IdentifAI — Business Entity Resolution Pipeline

This is the final packaged pipeline for the entity-resolution challenge. It combines deterministic preprocessing and lexical blocking with BGE semantic retrieval, FAISS HNSW ANN search, bounded cross-encoder reranking, leakage-safe feature construction, LightGBM/linear probability modeling, measured threshold tuning, and entity-level output validation.

Package layout

run.py
code/business_entity_resolution/
  run.py
  requirements.txt
  README.md
  src/
  tests/
  utils/validate_submission.py

Challenge datasets are intentionally not bundled. Put the materialized TSV files in a directory and pass that directory explicitly. Git-LFS pointer files are rejected.

Install

python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows
# .venv\Scripts\activate
pip install -r code/business_entity_resolution/requirements.txt

The production semantic path requires the Sentence-Transformers and FAISS packages listed in requirements.txt. The first model load may need the configured model files to already be available in the local model cache or filesystem.

Dataset contract

For training/validation, the supplied directory must contain:

train_source1.tsv
train_source2.tsv
train_source3.tsv
train_ground_truth.tsv

For prediction, the test directory must contain:

test_source1.tsv
test_source2.tsv
test_source3.tsv

Final workflow

Run the three stages in order:

python run.py --mode validate --data_dir /path/to/train --scratch_dir scratch
python run.py --mode tune --data_dir /path/to/train --scratch_dir scratch
python run.py --mode predict \
  --train_dir /path/to/train \
  --test_dir /path/to/test \
  --output_dir output \
  --scratch_dir scratch \
  --check-submission

predict is intentionally blocked until a completed validation report and tuned policy exist. The --allow-baseline-predict flag is only for debugging and should not be used for the final submission.

Semantic configuration

The defaults are configurable through environment variables; there are no entity-specific hardcoded exceptions. Useful overrides include:

IDENTIFAI_SEMANTIC_ENABLED
IDENTIFAI_BGE_MODEL
IDENTIFAI_RERANKER_MODEL
IDENTIFAI_SEMANTIC_TOP_K
IDENTIFAI_BGE_BATCH_SIZE
IDENTIFAI_RERANK_TOP_K
IDENTIFAI_RERANK_BATCH_SIZE
IDENTIFAI_HNSW_M
IDENTIFAI_HNSW_EF_CONSTRUCTION
IDENTIFAI_HNSW_EF_SEARCH
IDENTIFAI_RERANK_CACHE
IDENTIFAI_DEVICE

Semantic retrieval is enabled by default. If its production dependencies are missing, the pipeline fails clearly rather than silently producing a degraded final run. Tests explicitly disable semantic retrieval so they do not download models.

Outputs

The final prediction writes exactly:

output/matching_results.tsv
output/candidate_pairs.tsv

The candidate file contains the exact final candidate set supplied to the matching model. Final matches are a subset of that file. The bundled validator checks row coverage, duplicates, ID validity, and candidate-subset consistency.

Validation and tuning artifacts

validate writes scratch/validation_report.json. tune writes scratch/tuned_policy.json and scratch/decision_policy.json. Feature matrices are accompanied by metadata so a cache built with a different feature schema or semantic configuration cannot be reused silently.

Correctness principles

The implementation avoids data-specific entity IDs, hand-written match exceptions, synthetic measurements in repository documentation, and unmeasured post-tuning offsets. All changes to candidate generation, features, models, and decisioning are represented as reusable logic or configuration.