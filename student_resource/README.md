Business Entity Resolution Pipeline

Production-style, offline entity-resolution pipeline for the Amazon/IdentifAI business-record challenge.

What it does

The pipeline uses only the supplied challenge data:

deterministic name/address normalization and structured address parsing

frequency-aware inverted-index blocking

MinHash/character-n-gram LSH

phonetic, postal, street-number and acronym blocking keys

bounded high-frequency rescue

adaptive candidate selection with a hard maximum rather than fixed-K padding

65 pair features, including structured-address, reciprocal-rank and cross-source signals

training-derived lexical variation features, learned only from the training fold

LightGBM plus an optional standardized logistic model ensemble

exact grouped macro-F0.5 threshold optimization

country-aware/unseen-country decision policies

streaming inference and strict submission-contract validation

No external business registries, APIs, geocoders, or internet entity lookups are used.

Environment

Python 3.10+ is recommended.

From the repository root:

python -m pip install -r code/business_entity_resolution/requirements.txt

Windows activation example:

python -m venv .venv
.venv\Scripts\activate
python -m pip install -r code\business_entity_resolution\requirements.txt

The final submission package must not include .venv/.

Test suite

Run from the repository root:

python -m pytest -q

The tests are written so the package can be validated from the repository root.

Required real-data workflow

The challenge requires a real validation run before model/threshold tuning is retained.

1. Baseline validation

This does not perform model-weight or threshold search. It creates the measured baseline:

python run.py --mode validate

or explicitly:

python run.py --mode validate   --data_dir student_resource/dataset/train   --scratch_dir scratch

Expected artifact:

scratch/validation_report.json
scratch/decision_policy.json

2. Measured tuning

Only run this after the baseline report is complete:

python run.py --mode tune

This evaluates the class-weighting/linear-ensemble/variation alternatives on the real held-out training data and writes:

scratch/tuned_policy.json
scratch/decision_policy.json

The selected policy is the one with the strongest robust measured macro-F0.5 across the in-distribution, country-holdout and reverse-country views used by the pipeline.

3. Final prediction

Prediction requires the tuned policy by default:

python run.py --mode predict

This produces:

output/matching_results.tsv
output/candidate_pairs.tsv

For debugging only, a baseline prediction can be forced with:

python run.py --mode predict --allow-baseline-predict

Do not use the baseline override for the final submission.

4. Official submission validation

Run:

python run.py --mode predict --check-submission

This invokes the supplied stdlib validator with --check-ids.

The final package must contain:

output/
  matching_results.tsv
  candidate_pairs.tsv

code/
  business_entity_resolution/
    src/
    README.md
    requirements.txt

Documentation.md

Output contract

matching_results.tsv:

source1_entity_id    matched_entity_ids

candidate_pairs.tsv:

source1_entity_id    candidate_entity_ids

Both are tab-separated. The ID lists are comma-separated.

The pipeline guarantees:

exactly one row for every Source-1 test entity

empty lists for entities with no selected matches/candidates

candidate IDs are S2/S3 only

no duplicate IDs inside an ID list

every final match is contained in the exact candidate set passed to the model

The official validator remains the final submission gate.

Design notes

The candidate set is the final set stored in final_candidates; that exact set is both written to candidate_pairs.tsv and passed to the downstream feature/model stage.

The hard candidate maximum is a ceiling, not a target. The adaptive selector can retain substantially fewer candidates for easy entities.

The validation/tuning pipeline is intentionally separated so a threshold or weighting choice is not silently introduced before a real baseline exists.

Fair-play constraints

The implementation does not perform external entity lookup or data enrichment. All learned variation maps, models and decision policies are derived from the supplied training data and held-out validation design.