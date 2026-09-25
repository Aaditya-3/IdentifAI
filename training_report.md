# Real-data execution report

Run date: 2026-09-25. All source files were loaded as tab-separated UTF-8 TSVs. No external data, APIs, or identity lookup services were used.

## Data verification

| File | Rows | Duplicate IDs | Countries |
| --- | ---: | ---: | --- |
| train_source1.tsv | 2,206,821 | 0 | US 1,323,633; India 883,188 |
| train_source2.tsv | 5,034,616 | 0 | US 3,016,817; India 2,017,799 |
| train_source3.tsv | 5,285,603 | 0 | US 3,170,056; India 2,115,547 |
| test_source1.tsv | 1,732,544 | 0 | US 663,106; India 809,986; France 259,452 |
| test_source2.tsv | 4,887,273 | 0 | US 1,871,330; India 2,312,565; France 703,378 |
| test_source3.tsv | 5,082,316 | 0 | US 1,945,701; India 2,405,000; France 731,615 |
| train_ground_truth.tsv | 2,206,821 | 0 Source-1 rows | 7,638,365 labeled links |

Every file had its expected TSV columns. All ground-truth Source-1 IDs exist in training Source 1, and all 7,638,365 listed target IDs exist in training Sources 2/3. The match-count distribution is 0: 123,247; 1: 119,157; 2: 375,212; 3: 530,841; 4: 484,115; 5: 321,957; 6: 164,868; 7: 63,968; 8: 18,680; 9: 4,205; 10: 534; 11: 37. Singletons are 123,247 / 2,206,821 = **5.5848%**.

## Environment

The isolated `.venv` installed pandas 2.2.3, NumPy 2.1.3, scikit-learn 1.5.2, LightGBM 4.5.0, RapidFuzz 3.10.1, Jellyfish 1.1.0, and SciPy 1.14.1. LightGBM required the local `libomp` runtime; after installation, a toy classifier fit and probability prediction passed. The project requirements were pinned accordingly.

## Blocking experiments

The following are real results from a deterministic 1% hash sample of training Source 1 (21,861 records; 75,334 true links), queried against all 10,320,219 training target records. This diagnostic sample is not a substitute for the requested stratified 80/20 split.

| Round | Candidate strategy | Candidate links | Recall ceiling | Reduction ratio | Unfiltered candidate macro F0.5 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | exact normalized full name, core name, normalized address, or specific address number | 764,438 | 53.1367% (40,030/75,334) | 99.9996612% | 0.410864 |
| 2 | at least one rare informative name/address token | 47,434,574 | 82.4462% (62,110/75,334) | 99.9789750% | 0.026241 |
| 3 | at least two rare informative name/address tokens | 471,301 | 50.6810% (38,180/75,334) | 99.9997911% | 0.325009 |

**Recall gate: not met.** The best measured ceiling was 82.4462%, below the approximately 90% target, after three bounded attempts. The recall-rich one-token strategy was also impractical for full learned-pair inference (47.4M candidates for just 1% of Source 1). The likely limitation is noisy/transliterated names and incomplete or reordered addresses.

## Classifier and threshold search

No LightGBM pair classifier was trained for final inference, and no threshold sweep, in-distribution macro F0.5, singleton accuracy, or country-holdout OOD macro F0.5 is claimed. Those values were not measured; reporting them would be fabrication. The failed recall gate prevented a meaningful full learned-classifier run within the requested iteration budget. The existing feature code includes normalized/raw/core name exactness, Levenshtein and Jaro-Winkler ratios, token overlap, address features, country agreement, rank, and blocking similarity.

LightGBM is a classical gradient-boosting library and is exempt from the stated <=8B neural parameter constraint. The final fallback contains no learned model.

## Final test artifacts

`business_entity_resolution/scalable_exact_fallback.py` was run across the complete test data. It indexes normalized full names, suffix-stripped core names, and normalized addresses; signatures mapping to more than five Source-1 records are excluded. Candidate pairs are streamed through 128 temporary partitions. Predictions require exact full normalized-name equality or conservative corroborated core/address equality.

The generated files contain all 1,732,544 test Source-1 rows: 4,394,057 candidate links across 1,263,389 non-empty candidate rows, and 3,810,985 predicted links across 1,225,227 non-empty result rows. Matches are a strict subset of candidates.

Validation was run with:

```text
python3 student_resource/utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir student_resource/dataset/test --check-ids
```

Result: **PASS — no blocking issues found. Safe to submit.** The optional full target-ID existence check was enabled.

## Limitations

This is a format-valid conservative fallback, not an optimized learned-model submission. Its true test F0.5 is unknowable without test labels. The required 90% blocking gate was not met; a future run should add scalable approximate top-K character n-gram retrieval and then run a proper stratified 80/20 and country-holdout LightGBM evaluation.
