# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** IdentifAI  
**Submission Date:** 2026-09-26

## 1. Executive Summary

This submission uses a fully local, reproducible blocking-plus-classifier pipeline. It normalizes each supplied record, retrieves bounded lexical candidates from a SQLite inverted index, scores only those pairs with lexical features, and uses LightGBM with a precision-oriented macro-F0.5 threshold.

No external data, API, geocoder, pretrained identity model, or entity-lookup service is used. LightGBM, RapidFuzz, scikit-learn hashing, and the supplied TSV files are the complete inference inputs.

## 2. Methodology

### 2.1 Problem Analysis

The supplied training data contains 2,206,821 Source 1 records, 5,034,616 Source 2 records, 5,285,603 Source 3 records, and 7,638,365 labelled links. Source 1 has US and India labels in training; test additionally contains France, so country is treated as an open, noisy string field rather than a closed categorical partition. The test files contain 1,732,544/4,887,273/5,082,316 Source 1/2/3 rows respectively.

Expected record variation includes legal suffixes, punctuation, word-order changes, incomplete addresses, and country spelling differences. About 5.58% of training Source 1 records are labelled singletons, making false merges especially costly under macro F0.5.

### 2.2 Solution Strategy

**Approach Type:** Blocking + LightGBM pair classifier.  
**Core Innovation:** Country-free, DF-gated lexical retrieval retains candidates despite dirty or unseen country labels, while a model learns whether country agreement is useful after retrieval.

Normalization is a pure function of input text. It strips Latin accents, normalizes legal suffixes and street forms, and includes a small fixed, hand-curated Indian place-name alias table: Bombay/Mumbai, Bangalore/Bengaluru, Calcutta/Kolkata, Madras/Chennai, Pondicherry/Puducherry, and Trivandrum/Thiruvananthapuram. This is static code, not an external lookup. French legal and street forms—including SASU, SCI, SNC, EI, EIRL, avenue, boulevard, place, chemin, residence, and impasse—are normalized for the France-only test segment.

## 3. Candidate Generation (Blocking)

All retrieval keys deliberately omit country. Keys include normalized/full and core names, sorted core names, normalized addresses, exact full name+address composites, address numbers, DF-gated raw name tokens, phonetic forms of those tokens, prefixes, and deterministic character-3-gram MinHash LSH bands. The full name+address composite is never subject to the key bucket cap; every other overloaded key is deterministically capped. Candidate pairs are then ranked by blocking evidence plus RapidFuzz name/address similarity and bounded to the configured top-K per Source 1 entity.

Every predicted ID is structurally drawn from this final candidate table. The output writer emits one row for every Source 1 ID, de-duplicates IDs, and writes empty cells for singleton predictions.

## 4. Matching Model

**Model type:** Local LightGBM binary classifier (classical gradient boosting).

**Features used:**

- Exact normalized name/core/sorted-core/address comparisons.
- Country agreement plus an explicit country-missing indicator.
- RapidFuzz edit, token-sort, and token-set similarities for both names and addresses.
- Fit-free hashed character-3-gram cosine similarity for both names and addresses, calculated only over bounded candidate pairs.
- Name/address token Jaccard and overlap, address-number set Jaccard, first-number agreement, field-length ratios, candidate rank, and blocking similarity.

The country-held-out split selects among unweighted, `class_weight="balanced"`, and a moderate `scale_pos_weight` configuration. Thresholds are searched from 0.30 to 0.995 (0.001 granularity above 0.90). The final prediction threshold is the more conservative of the country-OOD and deterministic in-distribution thresholds; both are logged by validation.

## 5. Results & Error Analysis

The revised full-data validation and prediction run is in progress as this document is being prepared. No blocking recall, F0.5, threshold, candidate count, or submission validation result is claimed here until `scratch/validation_report.json` and `validate_submission.py --check-ids` complete. This avoids carrying forward the obsolete fallback measurements from the prior engineering log.

Validation writes country-holdout and in-distribution macro-F0.5, the complete class-weight ablation, the final candidate recall ceiling, candidate counts, and a country-holdout error CSV to `scratch/`.

## 6. Reproduction

The complete runnable package is under `code/business_entity_resolution/`. From the repository root:

```powershell
python -m pip install -r code\business_entity_resolution\requirements.txt
python run.py --mode validate --data_dir student_resource\dataset\train --scratch_dir scratch
python run.py --mode predict --train_dir student_resource\dataset\train --test_dir student_resource\dataset\test --output_dir output --scratch_dir scratch
python student_resource\utils\validate_submission.py --matching output\matching_results.tsv --candidate output\candidate_pairs.tsv --test-dir student_resource\dataset\test --check-ids
```

`output/matching_results.tsv` is always a strict per-entity subset of `output/candidate_pairs.tsv` by construction.
