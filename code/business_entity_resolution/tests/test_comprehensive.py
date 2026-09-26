"""Comprehensive regression tests for the entity-resolution pipeline."""
from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from src.blocking import BlockingStore
from src.features import FEATURE_NAMES, feature_batch
from src.metrics import macro_f0_5
from src.model import PairModel
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
            finally:
                store.close()


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
