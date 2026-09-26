Business Entity Resolution Challenge — Methodology

Objective

The pipeline resolves each Source-1 record to zero, one, or many Source-2/Source-3 records using only the supplied challenge data. The competition metric is macro F0.5 per Source-1 entity, with singleton entities included in the macro average.

Preprocessing

Business names are Unicode-normalized, case-folded and legal-suffix canonicalized. The normalization covers common US/India forms and French legal suffixes such as SASU, SCI, SNC and EIRL.

Addresses are normalized with deterministic street aliases and address aliases. Structured components include street number, street body, city when an explicit delimiter supports a conservative boundary, and postal code. Landmark filler terms such as “near”, “opposite”, “beside” and “next to” are removed from address text before comparison. No geocoder or external address database is used.

Blocking and candidate generation

Blocking uses multiple independent retrieval paths:

exact normalized full/core/sorted names

acronyms

exact and tokenized addresses

postal codes and street numbers

rare name tokens

phonetic/Soundex keys

character n-gram MinHash-LSH

bounded retrieval from oversized frequency buckets

Keys above the configured frequency ceiling are discarded from the unrestricted join rather than truncated by entity-ID position. Oversized buckets are handled only through bounded relevance-aware rescue.

The final candidate set has a hard maximum but is not padded to that maximum. Weak candidates can be removed by the validated adaptive selection policy.

candidate_pairs.tsv is generated from the same final candidate table that feeds feature computation and model inference.

Features

The pair representation contains 65 features covering:

exact and fuzzy name/address similarity

token-set, token-sort and partial similarity

character-trigram cosine similarity

token overlap and containment

postal-code and street-number agreement/conflict

structured street/city comparisons

legal-suffix consistency

acronym relationships

missingness

country match/conflict

retrieval evidence and candidate rank

name-vs-address ranking agreement

reciprocal candidate rank and mutual-best signal

cross-source corroboration

training-derived lexical-variation scores

The training-derived variation model is fit only from positive training pairs inside the appropriate training fold, avoiding validation-label leakage.

Models and tuning

The first real validation run is intentionally an untuned baseline using the LightGBM pair classifier and fixed threshold 0.5. This establishes a reproducible evidence baseline before any weighting, ensemble or decision-policy choice is retained.

The post-baseline tuning stage evaluates:

unweighted LightGBM

balanced LightGBM

moderate scale_pos_weight

a standardized logistic regression model

probability ensemble variants

training-derived lexical variation

Thresholds are optimized against grouped macro F0.5 rather than ordinary pairwise accuracy.

The selected production policy is evaluated across:

in-distribution Source-1 splits

country holdout

reverse-country holdout

and stores country-specific and unseen-country decision policies.

Decisioning

The final decision is made per Source-1 entity rather than independently for every pair.

The policy supports:

high-confidence acceptance

lower-confidence rejection

an ambiguity band

a stricter requirement for additional matches

a hard maximum number of matches

minimum absolute score

The decision layer operates only on candidates already present in candidate_pairs.tsv; it cannot introduce an out-of-candidate match.

Validation and reproducibility

Run the pipeline in this order:

python run.py --mode validate
python run.py --mode tune
python run.py --mode predict --check-submission

The baseline validation produces scratch/validation_report.json. The tuning stage produces scratch/tuned_policy.json and scratch/decision_policy.json. Prediction writes:

output/matching_results.tsv
output/candidate_pairs.tsv

The official challenge validator is run with --check-ids before a submission package is considered ready.

Fair play

The implementation uses only the supplied challenge records and labels. It performs no external business lookup, registry search, geocoding, commercial API lookup, or web-based entity enrichment.