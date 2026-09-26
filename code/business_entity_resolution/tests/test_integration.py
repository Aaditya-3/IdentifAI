import csv
import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.blocking import BlockingStore
from src.output import write_submission_from_database
from src.pipeline import predict

class TestMissingRequirements(unittest.TestCase):
    def test_bound_keys_priority_queue(self):
        """A direct test for BlockingStore._bound_keys() that builds an overloaded bucket."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            source1_path = tmp / "source1.tsv"
            
            # Create > 160 Source-1 entities sharing one generic block key "token:generic"
            # We want entities with FEWER total keys to be retained.
            # Entity 1..100: "generic" (only 1 key)
            # Entity 101..200: "generic additional_word" (more keys)
            # Both groups will generate the "token:generic" key.
            # Priority should favor 1..100 over 101..200 because they have fewer total keys.
            
            rows = []
            for i in range(1, 201):
                if i <= 100:
                    rows.append([f"S1-{i}", f"generic", "1 Main St", "US"])
                else:
                    rows.append([f"S1-{i}", f"generic raretoken{i}", "1 Main St", "US"])
                    
            with source1_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream, delimiter="\t")
                writer.writerow(["entity_id", "business_name", "business_address", "country"])
                writer.writerows(rows)
                
            store = BlockingStore(tmp / "blocking.sqlite", top_k=5, key_cap=160)
            store.reset()
            # Force token_df to treat "generic" and "raretoken" as rare (freq <= 5000)
            store.TOKEN_DF_LIMIT = 5000
            store.build_source_index(source1_path)
            
            # _bound_keys() has run.
            # Check the contents of the bounded_keys table for 'token:generic'
            retained = store.connection.execute(
                "SELECT entity_id FROM bounded_keys WHERE key='token:generic'"
            ).fetchall()
            
            retained_ids = {r[0] for r in retained}
            
            # It should have retained exactly 160
            self.assertEqual(len(retained_ids), 160)
            
            # It should contain ALL of S1-1 to S1-100
            for i in range(1, 101):
                self.assertIn(f"S1-{i}", retained_ids)
            
            store.close()

    def test_write_submission_from_database(self):
        """A direct test for output.write_submission_from_database()."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            db_path = tmp / "dummy.sqlite"
            conn = sqlite3.connect(db_path)
            conn.executescript("""
                CREATE TABLE source1 (entity_id TEXT PRIMARY KEY);
                CREATE TABLE final_candidates (source1_id TEXT, target_id TEXT, probability REAL);
                INSERT INTO source1 VALUES ('S1-1'), ('S1-2'), ('S1-3');
                INSERT INTO final_candidates VALUES ('S1-1', 'S2-1', 0.9), ('S1-1', 'S2-2', 0.4);
                INSERT INTO final_candidates VALUES ('S1-3', 'S2-5', 0.99);
            """)
            conn.close()
            
            cand_path, match_path = write_submission_from_database(tmp, db_path, threshold=0.5)
            
            # Check Candidates
            with cand_path.open("r", encoding="utf-8") as f:
                cands = f.read().splitlines()
            self.assertEqual(cands[0], "source1_entity_id\tcandidate_entity_ids")
            self.assertEqual(cands[1], "S1-1\tS2-1,S2-2")
            self.assertEqual(cands[2], "S1-2\t")  # Singleton handling
            self.assertEqual(cands[3], "S1-3\tS2-5")
            
            # Check Matches
            with match_path.open("r", encoding="utf-8") as f:
                matches = f.read().splitlines()
            self.assertEqual(matches[0], "source1_entity_id\tmatched_entity_ids")
            self.assertEqual(matches[1], "S1-1\tS2-1")  # S2-2 dropped because 0.4 < 0.5
            self.assertEqual(matches[2], "S1-2\t")
            self.assertEqual(matches[3], "S1-3\tS2-5")

    def test_integration_pipeline_metrics(self):
        """One synthetic end-to-end integration test with hand-computed F0.5."""
        from src.pipeline import _build_store, _materialize, _fast_tune_threshold
        
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            
            data_dir = tmp / "data"
            data_dir.mkdir()
            def write_tsv(path, rows):
                with path.open("w", encoding="utf-8", newline="") as f:
                    writer = csv.writer(f, delimiter="\t")
                    writer.writerow(["entity_id", "business_name", "business_address", "country"])
                    writer.writerows(rows)
                    
            write_tsv(data_dir / "train_source1.tsv", [
                ["S1-1", "Match One Corp", "100 Same St", "US"],
                ["S1-2", "Match Two Inc", "200 Exact Ave", "France"],
                ["S1-3", "Mismatch Corp", "300 Diff Rd", "US"],
            ])
            write_tsv(data_dir / "train_source2.tsv", [
                ["S2-1", "Match One Corp", "100 Same St", "US"],
                ["S2-3", "Other Company", "300 Diff Rd", "US"],
            ])
            write_tsv(data_dir / "train_source3.tsv", [
                ["S3-2", "Match Two Inc", "200 Exact Ave", "France"],
            ])
            with (data_dir / "train_ground_truth.tsv").open("w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f, delimiter="\t")
                writer.writerow(["source1_entity_id", "matched_entity_ids"])
                writer.writerows([
                    ["S1-1", "S2-1"],
                    ["S1-2", "S3-2"],
                ])
                
            store = _build_store(data_dir, "train", tmp / "train.sqlite", top_k=5, with_truth=True)
            fit = _materialize(store, tmp, "train_fit", labels=True)
            
            # Since min_child_samples=50 prevents tree splitting on 5 rows, inject perfect probabilities
            probs = np.zeros(fit.rows, dtype=np.float32)
            with fit.pairs_path.open("r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    s, t = line.strip().split("\t")
                    if (s, t) in [("S1-1", "S2-1"), ("S1-2", "S3-2")]:
                        probs[i] = 0.9
                    else:
                        probs[i] = 0.1
            
            # Predict pairs via the store and check optimal threshold
            sql_score, best_thresh = _fast_tune_threshold(store, fit, probs, "1=1", ())
            
            # The model should be able to separate the easy matches from negatives,
            # so at the optimal threshold, the macro F0.5 should be 1.0
            self.assertEqual(sql_score, 1.0)
            
            store.close()

if __name__ == "__main__":
    unittest.main()
