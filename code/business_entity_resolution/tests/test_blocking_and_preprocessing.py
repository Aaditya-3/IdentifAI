"""Small regression tests for country-agnostic retrieval and normalization."""
from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from src.blocking import BlockingStore
from src.preprocessing import normalize_address


class BlockingAndPreprocessingTests(unittest.TestCase):
    def test_country_format_difference_does_not_hide_exact_pair(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source1, source2 = root / "source1.tsv", root / "source2.tsv"
            fields = ["entity_id", "business_name", "business_address", "country"]
            for path, rows in (
                (source1, [["S1-1", "Acme Bakery LLC", "12 Main Street", "US"]]),
                (source2, [["S2-1", "Acme Bakery LLC", "12 Main Street", "United States"]]),
            ):
                with path.open("w", encoding="utf-8", newline="") as stream:
                    writer = csv.writer(stream, delimiter="\t")
                    writer.writerow(fields)
                    writer.writerows(rows)
            store = BlockingStore(root / "blocking.sqlite", top_k=5)
            try:
                store.reset()
                store.build_source_index(source1)
                store.add_targets_and_retrieve((source2,))
                store.finalize_candidates()
                pairs = set(store.connection.execute("SELECT source1_id, target_id FROM final_candidates"))
                self.assertIn(("S1-1", "S2-1"), pairs)
            finally:
                store.close()

    def test_french_and_indian_address_normalization(self) -> None:
        self.assertEqual(normalize_address("Résidence, Boulevard des Lilas"), "res bd des lilas")
        self.assertEqual(normalize_address("5 Bombay Avenue"), "5 mumbai av")


if __name__ == "__main__":
    unittest.main()
