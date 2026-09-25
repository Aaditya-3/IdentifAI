# IdentifAI entity resolution

This is the single supported pipeline. It uses a disk-backed SQLite inverted
index, bounded MinHash LSH and lexical blocking, RapidFuzz features, and a
LightGBM classifier. It uses only the supplied TSV files.

Install the pinned dependencies:

```powershell
python -m pip install -r code\business_entity_resolution\requirements.txt
```

Run full real-data validation. It builds a persistent scratch database and
memory-mapped feature matrices, so choose a drive with substantial free space:

```powershell
python run.py --mode validate --data_dir student_resource\dataset\train --scratch_dir scratch
```

`validate` reports both country-held-out and deterministic 80/20 in-distribution
macro F0.5, along with the measured blocking recall ceiling and average final
candidates per Source-1 entity. It writes `validation_report.json` and raw-string
false-positive/false-negative files to the scratch directory.

Generate the complete test submission:

```powershell
python run.py --mode predict --train_dir student_resource\dataset\train --test_dir student_resource\dataset\test --output_dir output --scratch_dir scratch
```

The output writer streams every test Source-1 ID and writes empty cells for
singletons. Predictions are read only from the final bounded candidate table,
which makes every output match a candidate by construction. Validate it with:

```powershell
python student_resource\utils\validate_submission.py --matching output\matching_results.tsv --candidate output\candidate_pairs.tsv --test-dir student_resource\dataset\test --check-ids
```

No pretrained component is included. The former optional SentenceTransformer
feature was removed because its model weights were neither pinned nor packaged
for offline grading. LightGBM, RapidFuzz, Datasketch, and Jellyfish run locally
and do not perform external identity lookup.
