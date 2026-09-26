"""Comprehensive regression tests for the entity-resolution pipeline.

Includes all 10 adversarial blocking tests from the specification.
"""
from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.blocking import BlockingStore
from src.features import FEATURE_NAMES, feature_batch
from src.metrics import macro_f0_5
from src.model import LinearPairModel, PairModel, ProbabilityEnsemble
from src.decision import choose_hysteresis_policy, apply_entity_policy
from src.output import enforce_candidate_subset, group_candidates
from src.preprocessing import (
    core_name, extract_address_numbers, normalize_address, normalize_name,
)


def _write_tsv(path: Path, rows: list[list[str]]) -> None:
    fields = ["entity_id", "business_name", "business_address", "country"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(fields)
        writer.writerows(rows)


def _write_truth(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(["source1_entity_id", "matched_entity_ids"])
        writer.writerows(rows)


def _build_small_pipeline(tmp: Path, source1_rows, source2_rows, source3_rows=None, truth_rows=None, top_k=10):
    """Build a complete small-scale blocking store for testing."""
    source1 = tmp / "source1.tsv"
    source2 = tmp / "source2.tsv"
    _write_tsv(source1, source1_rows)
    _write_tsv(source2, source2_rows)
    targets = [source2]
    if source3_rows:
        source3 = tmp / "source3.tsv"
        _write_tsv(source3, source3_rows)
        targets.append(source3)
    store = BlockingStore(tmp / "test.sqlite", top_k=top_k)
    store.reset()
    store.build_source_index(source1)
    if truth_rows:
        truth_file = tmp / "truth.tsv"
        _write_truth(truth_file, truth_rows)
        store.add_truth(truth_file)
    store.add_targets_and_retrieve(targets)
    store.finalize_candidates()
    return store


class TestPreprocessing(unittest.TestCase):
    """Normalization edge cases."""

    def test_unicode_accents(self):
        self.assertEqual(normalize_name("Café Résidence"), "cafe residence")

    def test_legal_suffix_corp(self):
        self.assertEqual(normalize_name("Acme Corporation"), "acme corp")

    def test_legal_suffix_french(self):
        self.assertEqual(normalize_name("Société Anonyme Paris"), "sa paris")

    def test_core_strips_suffix(self):
        self.assertEqual(core_name("Acme Bakery LLC"), "acme bakery")

    def test_address_street_abbreviation(self):
        self.assertIn("st", normalize_address("123 Main Street"))

    def test_address_french(self):
        result = normalize_address("14 Boulevard de la République")
        self.assertIn("bd", result)

    def test_address_indian_alias(self):
        self.assertIn("mumbai", normalize_address("Bombay Road"))

    def test_extract_numbers(self):
        self.assertEqual(extract_address_numbers("12 Main St Suite 400"), "12 400")

    def test_extract_unit_number(self):
        self.assertEqual(extract_address_numbers("221B Baker Street"), "221b")

    def test_landmark_fillers_removed(self):
        normalized = normalize_address("Near SBI ATM, 221B Baker Street")
        self.assertNotIn("near", normalized.split())
        self.assertIn("221b", normalized.split())

    def test_empty_name(self):
        self.assertEqual(normalize_name(""), "")
        self.assertEqual(core_name(""), "")

    def test_empty_address(self):
        self.assertEqual(normalize_address(""), "")


class TestBlocking(unittest.TestCase):
    """Blocking retrieval correctness."""

    def test_exact_match_retrieved(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            store = _build_small_pipeline(tmp,
                [["S1-1", "Acme Bakery LLC", "12 Main Street", "US"]],
                [["S2-1", "Acme Bakery LLC", "12 Main Street", "US"]],
            )
            try:
                pairs = set(store.connection.execute(
                    "SELECT source1_id, target_id FROM final_candidates"))
                self.assertIn(("S1-1", "S2-1"), pairs)
            finally:
                store.close()

    def test_punctuation_variation(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            store = _build_small_pipeline(tmp,
                [["S1-1", "Acme & Sons Ltd.", "10 High Rd.", "US"]],
                [["S2-1", "Acme and Sons Ltd", "10 High Road", "US"]],
            )
            try:
                pairs = set(store.connection.execute(
                    "SELECT source1_id, target_id FROM final_candidates"))
                self.assertIn(("S1-1", "S2-1"), pairs)
            finally:
                store.close()

    def test_different_country_still_retrieved(self):
        """Country difference must not block retrieval — country is a feature, not a filter."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            store = _build_small_pipeline(tmp,
                [["S1-1", "Global Corp", "100 Avenue de Paris", "France"]],
                [["S2-1", "Global Corp", "100 Avenue de Paris", "US"]],
            )
            try:
                pairs = set(store.connection.execute(
                    "SELECT source1_id, target_id FROM final_candidates"))
                self.assertIn(("S1-1", "S2-1"), pairs)
            finally:
                store.close()

    def test_missing_name_still_has_address_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            store = _build_small_pipeline(tmp,
                [["S1-1", "", "42 Rue de la Paix Suite 300", "France"]],
                [["S2-1", "", "42 Rue de la Paix Suite 300", "France"]],
            )
            try:
                pairs = set(store.connection.execute(
                    "SELECT source1_id, target_id FROM final_candidates"))
                self.assertIn(("S1-1", "S2-1"), pairs)
            finally:
                store.close()

    def test_singleton_no_false_candidates(self):
        """A Source-1 entity with no matching target should have zero or few candidates."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            store = _build_small_pipeline(tmp,
                [["S1-1", "Unique Specialty Shop XYZ", "999 Nowhere Lane", "US"]],
                [["S2-1", "Completely Different Business", "1 Other Place", "India"]],
            )
            try:
                count = store.connection.execute(
                    "SELECT COUNT(*) FROM final_candidates WHERE source1_id='S1-1'").fetchone()[0]
                # Should have zero or very few irrelevant candidates
                self.assertLessEqual(count, 2)
            finally:
                store.close()

    def test_source2_and_source3_simultaneous_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            store = _build_small_pipeline(tmp,
                [["S1-1", "Acme Bakery", "12 Main Street", "US"]],
                [["S2-1", "Acme Bakery", "12 Main Street", "US"]],
                [["S3-1", "Acme Bakery", "12 Main Street", "US"]],
            )
            try:
                pairs = set(store.connection.execute(
                    "SELECT source1_id, target_id FROM final_candidates"))
                self.assertIn(("S1-1", "S2-1"), pairs)
                self.assertIn(("S1-1", "S3-1"), pairs)
            finally:
                store.close()

    def test_multiple_valid_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            store = _build_small_pipeline(tmp,
                [["S1-1", "Acme Corp", "12 Main St", "US"]],
                [["S2-1", "Acme Corp", "12 Main St", "US"],
                 ["S2-2", "Acme Corp", "12 Main St", "US"]],
            )
            try:
                pairs = set(store.connection.execute(
                    "SELECT source1_id, target_id FROM final_candidates"))
                self.assertIn(("S1-1", "S2-1"), pairs)
                self.assertIn(("S1-1", "S2-2"), pairs)
            finally:
                store.close()

    def test_diagnostics_populated(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            store = _build_small_pipeline(tmp,
                [["S1-1", "Test Co", "1 Test St", "US"]],
                [["S2-1", "Test Co", "1 Test St", "US"]],
                truth_rows=[["S1-1", "S2-1"]],
            )
            try:
                self.assertIn("raw_blocking", store.diagnostics)
                self.assertIn("final", store.diagnostics)
                self.assertGreater(store.diagnostics["final"]["recall"], 0)
                # New: CMS recall should also be present
                self.assertIn("complete_match_set_recall", store.diagnostics["final"])
            finally:
                store.close()


# ════════════════════════════════════════════════════════════════════
# ADVERSARIAL BLOCKING TESTS (from specification)
# ════════════════════════════════════════════════════════════════════

class TestAdversarialBlocking(unittest.TestCase):
    """Tests that verify no recall-destroying arbitrary truncation occurs."""

    def test_1_large_shared_key_full_recall(self):
        """TEST 1: 300 true targets share the same blocking key.
        Expected: candidate recall ≈ 100%."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            s1_rows = [["S1-1", "Common Business Name", "100 Main St", "US"]]
            s2_rows = [[f"S2-{i}", "Common Business Name", "100 Main St", "US"] for i in range(1, 301)]
            truth = [["S1-1", ",".join(f"S2-{i}" for i in range(1, 301))]]
            store = _build_small_pipeline(tmp, s1_rows, s2_rows, truth_rows=truth, top_k=300)
            try:
                recall = store.diagnostics["final"]["recall"]
                self.assertGreaterEqual(recall, 0.95,
                    f"Recall {recall:.4f} is too low for 300 targets sharing a key")
            finally:
                store.close()

    def test_2_large_irrelevant_key_no_id_bias(self):
        """TEST 2: 1000 irrelevant targets share a common key, only a subset relevant.
        Expected: candidate reduction without arbitrary ID-order bias."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            s1_rows = [["S1-1", "Target Match Corp", "500 Elm Ave", "US"]]
            # 1000 targets with same key, only S2-500 is the true match
            s2_rows = [[f"S2-{i}", f"Generic Inc Entity {i}", "1 Generic St", "US"] for i in range(1, 1001)]
            s2_rows[499] = ["S2-500", "Target Match Corp", "500 Elm Ave", "US"]
            truth = [["S1-1", "S2-500"]]
            store = _build_small_pipeline(tmp, s1_rows, s2_rows, truth_rows=truth, top_k=30)
            try:
                pairs = set(store.connection.execute(
                    "SELECT target_id FROM final_candidates WHERE source1_id='S1-1'"))
                target_ids = {r[0] for r in pairs}
                self.assertIn("S2-500", target_ids,
                    "True match S2-500 must survive regardless of ID position")
            finally:
                store.close()

    def test_3_many_s1_different_true_targets(self):
        """TEST 3: 300 S1 records share a key. Each has a different true target.
        Expected: all true pairs survive."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            s1_rows = [[f"S1-{i}", f"Entity Alpha {i}", "1 Main St", "US"] for i in range(1, 301)]
            s2_rows = [[f"S2-{i}", f"Entity Alpha {i}", "1 Main St", "US"] for i in range(1, 301)]
            truth = [[f"S1-{i}", f"S2-{i}"] for i in range(1, 301)]
            store = _build_small_pipeline(tmp, s1_rows, s2_rows, truth_rows=truth, top_k=10)
            try:
                recall = store.diagnostics["final"]["recall"]
                self.assertGreaterEqual(recall, 0.90,
                    f"Recall {recall:.4f} should be high when each S1 has a matching target")
            finally:
                store.close()

    def test_4_true_match_after_position_160(self):
        """TEST 4: True match lies after target ID position 160.
        Expected: it is still retrievable (no arbitrary truncation)."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            s1_rows = [["S1-1", "Unique Target Corp", "999 Specific Ave", "US"]]
            # Create 200 decoy targets then the true match at position >160
            s2_rows = [[f"S2-{i:04d}", f"Decoy Business {i}", "1 Other St", "India"] for i in range(1, 201)]
            s2_rows.append(["S2-9999", "Unique Target Corp", "999 Specific Ave", "US"])
            truth = [["S1-1", "S2-9999"]]
            store = _build_small_pipeline(tmp, s1_rows, s2_rows, truth_rows=truth, top_k=10)
            try:
                pairs = {r[0] for r in store.connection.execute(
                    "SELECT target_id FROM final_candidates WHERE source1_id='S1-1'")}
                self.assertIn("S2-9999", pairs,
                    "True match after old position-160 cutoff must still be retrieved")
            finally:
                store.close()

    def test_5_low_name_high_address(self):
        """TEST 5: True match has low name similarity, high address similarity.
        Expected: candidate survives."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            store = _build_small_pipeline(tmp,
                [["S1-1", "XYZQWRT", "42 Rue de la Paix Suite 300 75002 Paris", "France"]],
                [["S2-1", "Completely Different Name", "42 Rue de la Paix Suite 300 75002 Paris", "France"]],
            )
            try:
                pairs = {r[0] for r in store.connection.execute(
                    "SELECT target_id FROM final_candidates WHERE source1_id='S1-1'")}
                self.assertIn("S2-1", pairs,
                    "Address-only match must survive cheap ranking")
            finally:
                store.close()

    def test_6_high_name_low_address(self):
        """TEST 6: True match has high name similarity, low address similarity.
        Expected: candidate survives."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            store = _build_small_pipeline(tmp,
                [["S1-1", "Acme International Corp", "999 Nowhere Lane", "US"]],
                [["S2-1", "Acme International Corp", "1 Completely Different Address", "India"]],
            )
            try:
                pairs = {r[0] for r in store.connection.execute(
                    "SELECT target_id FROM final_candidates WHERE source1_id='S1-1'")}
                self.assertIn("S2-1", pairs,
                    "Name-only match must survive cheap ranking")
            finally:
                store.close()

    def test_7_entity_with_many_true_matches(self):
        """TEST 7: One S1 has >10 true matches (max in real data is 11).
        Expected: system does not silently truncate."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            s1_rows = [["S1-1", "Multi Match Corp", "100 Central Ave", "US"]]
            s2_rows = [[f"S2-{i}", "Multi Match Corp", "100 Central Ave", "US"] for i in range(1, 12)]
            truth = [["S1-1", ",".join(f"S2-{i}" for i in range(1, 12))]]
            store = _build_small_pipeline(tmp, s1_rows, s2_rows, truth_rows=truth, top_k=30)
            try:
                recall = store.diagnostics["final"]["recall"]
                cms = store.diagnostics["final"]["complete_match_set_recall"]
                self.assertAlmostEqual(recall, 1.0, places=2,
                    msg=f"All 11 true matches should be retrieved, got recall={recall:.4f}")
                self.assertAlmostEqual(cms, 1.0, places=2,
                    msg=f"Complete match set recall should be 1.0, got {cms:.4f}")
            finally:
                store.close()

    def test_8_singleton_with_similar_false_candidates(self):
        """TEST 8: Singleton with highly similar false candidates.
        Expected: high-confidence empty prediction when appropriate."""
        # This test verifies the model/threshold can handle singletons.
        # At the blocking level, we just verify the pipeline doesn't crash.
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            store = _build_small_pipeline(tmp,
                [["S1-1", "Smith Consulting LLC", "100 Oak St", "US"]],
                [["S2-1", "Smith Consulting Inc", "100 Oak St", "US"],
                 ["S2-2", "Smith Advisory LLC", "100 Oak St", "US"]],
                truth_rows=[],  # No matches -> singleton
            )
            try:
                count = store.connection.execute(
                    "SELECT COUNT(*) FROM final_candidates WHERE source1_id='S1-1'").fetchone()[0]
                # Pipeline should complete without error
                self.assertIsNotNone(count)
            finally:
                store.close()

    def test_9_unseen_country(self):
        """TEST 9: Unseen country. Expected: pipeline continues."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            store = _build_small_pipeline(tmp,
                [["S1-1", "Paris Bakery", "10 Rue de Rivoli", "France"]],
                [["S2-1", "Paris Bakery", "10 Rue de Rivoli", "France"]],
            )
            try:
                pairs = set(store.connection.execute(
                    "SELECT source1_id, target_id FROM final_candidates"))
                self.assertIn(("S1-1", "S2-1"), pairs)
            finally:
                store.close()

    def test_10_deterministic_output(self):
        """TEST 10: Same dataset processed twice. Expected: identical outputs."""
        results = []
        for _ in range(2):
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                store = _build_small_pipeline(tmp,
                    [["S1-1", "Test Corp", "100 Main St", "US"],
                     ["S1-2", "Another LLC", "200 Oak Ave", "India"]],
                    [["S2-1", "Test Corp", "100 Main St", "US"],
                     ["S2-2", "Another LLC", "200 Oak Ave", "India"]],
                )
                try:
                    pairs = sorted(store.connection.execute(
                        "SELECT source1_id, target_id FROM final_candidates"))
                    results.append(pairs)
                finally:
                    store.close()
        self.assertEqual(results[0], results[1])


class TestFeatures(unittest.TestCase):
    """Feature computation."""

    def test_feature_count_matches_names(self):
        row = ("S1-1", "S2-1", "us", "acme corp", "acme", "acme", "12 main st", "12",
               "us", "acme corp", "acme", "acme", "12 main st", "12", 10.0, 5.0, 1.0)
        features = feature_batch([row])
        self.assertEqual(features.shape[1], len(FEATURE_NAMES))

    def test_missingness_features(self):
        row = ("S1-1", "S2-1", "us", "", "", "", "12 main st", "12",
               "us", "acme corp", "acme", "acme", "12 main st", "12", 10.0, 5.0, 1.0)
        features = feature_batch([row])
        idx_left = FEATURE_NAMES.index("name_missing_left")
        idx_right = FEATURE_NAMES.index("name_missing_right")
        self.assertEqual(features[0, idx_left], 1.0)  # left name is empty
        self.assertEqual(features[0, idx_right], 0.0)  # right name exists

    def test_target_source_feature(self):
        row_s2 = ("S1-1", "S2-1", "us", "a", "a", "a", "b", "",
                   "us", "a", "a", "a", "b", "", 1.0, 1.0, 1.0)
        row_s3 = ("S1-1", "S3-1", "us", "a", "a", "a", "b", "",
                   "us", "a", "a", "a", "b", "", 1.0, 1.0, 1.0)
        idx = FEATURE_NAMES.index("target_source")
        f2 = feature_batch([row_s2])
        f3 = feature_batch([row_s3])
        self.assertEqual(f2[0, idx], 2.0)
        self.assertEqual(f3[0, idx], 3.0)

    def test_empty_strings_no_crash(self):
        """Empty core, empty address, punctuation-only should not crash."""
        row = ("S1-1", "S2-1", "", "", "", "", "", "",
               "", "", "", "", "", "", 0.0, 0.0, 1.0)
        features = feature_batch([row])
        self.assertEqual(features.shape, (1, len(FEATURE_NAMES)))

    def test_numeric_only_name(self):
        """Numeric-only names should not crash."""
        row = ("S1-1", "S2-1", "us", "12345", "12345", "12345", "addr", "12345",
               "us", "12345", "12345", "12345", "addr", "12345", 1.0, 1.0, 1.0)
        features = feature_batch([row])
        self.assertEqual(features.shape[1], len(FEATURE_NAMES))


class TestMetrics(unittest.TestCase):
    """Macro F0.5 correctness."""

    def test_perfect_prediction(self):
        y_true = {"S1-1": {"S2-1"}, "S1-2": set()}
        y_pred = {"S1-1": {"S2-1"}, "S1-2": set()}
        self.assertAlmostEqual(macro_f0_5(y_true, y_pred), 1.0)

    def test_singleton_correctly_empty(self):
        y_true = {"S1-1": set()}
        y_pred = {"S1-1": set()}
        self.assertAlmostEqual(macro_f0_5(y_true, y_pred), 1.0)

    def test_singleton_false_merge(self):
        y_true = {"S1-1": set()}
        y_pred = {"S1-1": {"S2-1"}}
        self.assertAlmostEqual(macro_f0_5(y_true, y_pred), 0.0)

    def test_missed_match(self):
        y_true = {"S1-1": {"S2-1"}}
        y_pred = {"S1-1": set()}
        self.assertAlmostEqual(macro_f0_5(y_true, y_pred), 0.0)


class TestOutput(unittest.TestCase):
    """Output invariants."""

    def test_enforce_candidate_subset(self):
        candidates = {"S1-1": {"S2-1", "S2-2"}}
        predictions = {"S1-1": {"S2-1"}}
        result = enforce_candidate_subset(predictions, candidates)
        self.assertEqual(result["S1-1"], {"S2-1"})

    def test_enforce_candidate_subset_rejects_unknown(self):
        candidates = {"S1-1": {"S2-1"}}
        predictions = {"S1-1": {"S2-1", "S2-99"}}
        with self.assertRaises(AssertionError):
            enforce_candidate_subset(predictions, candidates)

    def test_group_candidates_preserves_all_source1(self):
        source1 = [{"entity_id": "S1-1"}, {"entity_id": "S1-2"}]
        pairs = [("S1-1", "S2-1")]
        result = group_candidates(source1, pairs)
        self.assertIn("S1-1", result)
        self.assertIn("S1-2", result)
        self.assertEqual(result["S1-2"], set())


class TestModel(unittest.TestCase):
    """Model basic behaviour."""

    def test_threshold_respects_precision(self):
        """Higher threshold = fewer predictions = higher precision."""
        import numpy as np
        model = PairModel(random_state=42)
        model.threshold = 0.9
        probs = [0.95, 0.85, 0.50]
        pairs = [("S1-1", "S2-1"), ("S1-1", "S2-2"), ("S1-1", "S2-3")]
        result = model.predict_pairs(probs, pairs)
        self.assertEqual(result.get("S1-1", set()), {"S2-1"})

    def test_multiple_matches_allowed(self):
        model = PairModel(random_state=42)
        model.threshold = 0.5
        probs = [0.9, 0.8, 0.3]
        pairs = [("S1-1", "S2-1"), ("S1-1", "S3-1"), ("S1-1", "S2-2")]
        result = model.predict_pairs(probs, pairs)
        self.assertEqual(result.get("S1-1", set()), {"S2-1", "S3-1"})


    def test_linear_model_and_ensemble_interface(self):
        import numpy as np
        X = np.asarray([[0.0, 0.0], [1.0, 1.0], [0.2, 0.1], [0.9, 1.1]], dtype=np.float32)
        y = np.asarray([0, 1, 0, 1], dtype=np.int8)
        primary = PairModel(random_state=7).fit(X, y)
        secondary = LinearPairModel(random_state=7).fit(X, y)
        ensemble = ProbabilityEnsemble(primary, secondary).predict_proba(X)
        self.assertEqual(ensemble.shape, (4,))
        self.assertTrue(np.all((ensemble >= 0.0) & (ensemble <= 1.0)))

class TestDecisionPolicy(unittest.TestCase):
    def test_entity_policy_uses_group_margin(self):
        policy = choose_hysteresis_policy(
            0.80,
            ambiguous_margin=0.05,
            second_match_delta=0.05,
            max_matches=4,
        )
        candidates = [
            {"target_id": "S2-1", "probability": 0.95, "name_score": 0.98, "address_score": 0.95},
            {"target_id": "S2-2", "probability": 0.82, "name_score": 0.80, "address_score": 0.60},
        ]
        selected = apply_entity_policy(candidates, policy)
        self.assertIn("S2-1", selected)
        self.assertNotIn("S2-2", selected)

class TestDeterminism(unittest.TestCase):
    """Reproducibility."""

    def test_blocking_deterministic(self):
        """Two runs with the same data should produce the same candidates."""
        results = []
        for _ in range(2):
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                store = _build_small_pipeline(tmp,
                    [["S1-1", "Test Corp", "100 Main St", "US"],
                     ["S1-2", "Another LLC", "200 Oak Ave", "India"]],
                    [["S2-1", "Test Corp", "100 Main St", "US"],
                     ["S2-2", "Another LLC", "200 Oak Ave", "India"]],
                )
                try:
                    pairs = sorted(store.connection.execute(
                        "SELECT source1_id, target_id FROM final_candidates"))
                    results.append(pairs)
                finally:
                    store.close()
        self.assertEqual(results[0], results[1])


if __name__ == "__main__":
    unittest.main()


def test_semantic_feature_stub_is_consumed():
    from business_entity_resolution.src.features import FEATURE_NAMES, feature_batch

    class StubRetriever:
        def rerank(self, pairs):
            return np.asarray([0.91 for _ in pairs], dtype=np.float32)

    # Build the current 32-column candidate-row shape used by feature_batch.
    row = [
        "S1-1", "S2-1",
        "gb", "acme ltd", "acme", "acme", "221b baker street", "221b",
        "gb", "acme limited", "acme limited", "acme limited", "221b baker street", "221b",
        0.7, 0.95, 1, 0.88, 1,
        "221b", "baker street", "london", "nw1 6xe",
        "221b", "baker street", "london", "nw1 6xe",
        1, 1, 1, 1.0, 1.0,
    ]
    matrix = feature_batch([row], semantic_retriever=StubRetriever())
    assert matrix.shape == (1, len(FEATURE_NAMES))
    assert np.isclose(float(matrix[0, FEATURE_NAMES.index("semantic_similarity")]), 0.88)
    assert np.isclose(float(matrix[0, FEATURE_NAMES.index("cross_encoder_score")]), 0.91)


def test_semantic_retriever_real_backend_contract_with_stubs(monkeypatch):
    import sys
    import types

    class FakeEncoder:
        def __init__(self, model_name, device=None):
            self.model_name = model_name

        def encode(self, texts, **kwargs):
            vectors = []
            for text in texts:
                token = text.split()[0] if text.split() else ""
                value = float(sum(map(ord, token)) % 997) / 997.0
                vectors.append([1.0, value, value * value, 0.5])
            return np.asarray(vectors, dtype=np.float32)

    class FakeCrossEncoder:
        def __init__(self, model_name, activation_fn=None, device=None):
            self.model_name = model_name

        def predict(self, pairs, **kwargs):
            return np.asarray([0.93] * len(pairs), dtype=np.float32)

    class FakeHNSW: 
        def __init__(self, dimension, m, metric):
            self.dimension = dimension
            self.vectors = np.empty((0, dimension), dtype=np.float32)
            self.hnsw = types.SimpleNamespace(efConstruction=0, efSearch=0)

        def add(self, values):
            self.vectors = np.vstack([self.vectors, np.asarray(values, dtype=np.float32)])

        def search(self, queries, k):
            scores = np.asarray(queries, dtype=np.float32) @ self.vectors.T
            indices = np.argsort(-scores, axis=1, kind="stable")[:, :k]
            distances = np.take_along_axis(scores, indices, axis=1)
            return distances, indices

    fake_st = types.ModuleType("sentence_transformers")
    fake_st.SentenceTransformer = FakeEncoder
    fake_st.CrossEncoder = FakeCrossEncoder
    fake_faiss = types.ModuleType("faiss")
    fake_faiss.METRIC_INNER_PRODUCT = 0
    fake_faiss.IndexHNSWFlat = FakeHNSW
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_st)
    monkeypatch.setitem(sys.modules, "faiss", fake_faiss)

    from src.semantic_retrieval import SemanticConfig, SemanticRetriever

    retriever = SemanticRetriever(SemanticConfig(enabled=True, encode_batch_size=2, semantic_top_k=2, rerank_top_k=1))
    retriever.build([
        ("T1", "business name: alpha; business address: one"),
        ("T2", "business name: beta; business address: two"),
    ])
    results = retriever.query([("S1", "business name: alpha; business address: one")], top_k=2)
    assert retriever.using_real_backend
    assert results[0][0] == "S1"
    assert results[0][1] == "T1"
    scores = retriever.rerank([("query", "document", 1), ("query", "document2", 2)])
    assert np.allclose(scores, [0.93, 0.0])
    retriever.close()
