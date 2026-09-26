Business Entity Resolution Pipeline

Production-oriented, offline entity-resolution pipeline for the ML Challenge 2026 Business Entity Resolution task.

The challenge scores macro F0.5 per Source-1 entity, with precision weighted more heavily than recall. Source-1 may have zero, one, or many matches, so singleton false positives are important.

Package contract

A completed submission package must contain:

output/
├── matching_results.tsv
└── candidate_pairs.tsv

code/business_entity_resolution/
├── src/
├── README.md
└── requirements.txt

Documentation.md

The two TSVs are generated only by a completed prediction run. Do not create stand-in or cached submission files by hand.

Dataset layout

The repository root must contain the materialized challenge data:

student_resource/
├── dataset/
│   ├── train/
│   │   ├── train_source1.tsv
│   │   ├── train_source2.tsv
│   │   ├── train_source3.tsv
│   │   └── train_ground_truth.tsv
│   └── test/
│       ├── test_source1.tsv
│       ├── test_source2.tsv
│       └── test_source3.tsv
└── utils/
    └── validate_submission.py

The pipeline explicitly rejects Git-LFS pointer stubs instead of processing them as data.

Environment

Use Python 3.10+.

pip install -r code/business_entity_resolution/requirements.txt

No virtual environment is included in the submission archive.

Test suite

Run from the repository root:

python -m pytest -q

The tests configure their own package path, so the command is reproducible from the clean repository root.

Required first real-data run

Do not tune thresholds, class weighting, ensemble rules, or pseudo-labeling before the first complete real-data validation report exists.

Run:

python run.py --mode validate

This creates:

scratch/train.sqlite
scratch/validation_report.json
scratch/decision_policy.json

The first run is an untuned baseline: unweighted LightGBM with a fixed 0.5 threshold. The report records:

country holdout F0.5 / precision / recall

reverse-country holdout F0.5 / precision / recall

in-distribution F0.5 / precision / recall

raw / final candidate recall

complete-match-set recall

candidate counts and per-country blocking diagnostics

stage runtime

feature count and model configuration

Use that real report as the evidence baseline for subsequent tuning.

Full prediction run

After a completed validation run:

python run.py --mode predict

The command reuses the validated training candidate store when its input/code signatures still match, trains the production model on all training data, builds the test candidate store, streams inference, and writes:

output/matching_results.tsv
output/candidate_pairs.tsv

The writer enforces the required candidate-subset relationship and checks that every test Source-1 entity receives exactly one row, including empty rows for no-match cases.

Official submission validation

Before packaging, run the official validator exactly as supplied by the challenge:

python student_resource/utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir student_resource/dataset/test \
  --check-ids

The --check-ids mode loads Source-2/3 IDs and can require substantial memory on the full test set. It is still the required final diagnostic before packaging.

You can also ask the pipeline to run the same validator after prediction:

python run.py --mode predict --check-submission

Candidate generation

Candidate generation uses:

normalized name, legal-suffix-stripped core name and sorted core

acronyms and rare tokens

extracted address numbers and postal codes

phonetic token keys

character-ngram MinHash-LSH on names and addresses

frequency-aware key filtering rather than arbitrary ID-order truncation

bounded rescue retrieval for oversized blocks

recall-first shortlist construction

adaptive final candidate counts with a hard maximum rather than fixed padding

candidate_pairs.tsv is generated from the final candidate table that is actually passed into the matching model. It is not an earlier raw blocking dump.

Features

The feature layer combines:

exact and fuzzy name/address similarities

token-set and token-sort similarities

legal suffix agreement/conflict

acronym agreement

number and postal-code consistency

conservative structured address components (street number, street, city when explicitly delimited, postal code)

country agreement/conflict/missingness

character-trigram similarity

candidate rank and blocking evidence

cross-field name/address rank agreement

reciprocal candidate rank and a mutual-best indicator

missingness indicators

No external business registry, geocoder, API, web lookup or other data augmentation is used.

Reproducibility

SQLite candidate stores include signatures for input files, blocking configuration, feature/preprocessing/model code and the cache schema. A stale candidate database is rebuilt automatically instead of being silently reused.