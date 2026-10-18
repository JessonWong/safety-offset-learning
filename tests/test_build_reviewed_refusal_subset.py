"""CPU-only integrity tests for explicitly reviewed refusal-subset exports."""

import copy
import csv
import hashlib
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from safeoff.data.build_reviewed_refusal_subset import main


class ReviewedSubsetTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="safeoff_reviewed_subset_test_")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source = self.root / "source.csv"
        self.responses = self.root / "responses.json"
        self.review = self.root / "review.json"
        self.output = self.root / "output" / "subset.csv"
        self.fields = ["index", "instruction", "alpaca_reference", "suffix"]
        self.rows = [
            dict(zip(self.fields, values))
            for values in [
                ("9", "First input", "Original answer, not a refusal.\nSecond line.", ""),
                ("27", "Second input", "Original second answer", ""),
                ("41", "Third input", "Original third answer", ""),
                ("105", "Fourth input", "Original fourth answer: café", ""),
            ]
        ]
        with self.source.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fields)
            writer.writeheader()
            writer.writerows(self.rows)
        self.payload = {
            "meta": {
                "source_sha256": self.digest(self.source),
                "system": None,
            },
            "data": [
                {
                    "row_position": position,
                    "source_index": row["index"],
                    "prompt": row["instruction"],
                    "reference_response": row["alpaca_reference"],
                    "system": None,
                    "response": "I cannot fulfill this request." if position in (0, 3) else "A useful answer.",
                    "refusal": position in (0, 1, 3),
                }
                for position, row in enumerate(self.rows)
            ],
        }
        self.write_payload()
        self.review_payload = {
            "responses_sha256": self.digest(self.responses),
            # Deliberately unsorted: export must still preserve source order.
            "keep_row_positions": [3, 0],
            "excluded": [{"row_position": 1, "reason": "Incidental keyword in an actual answer."}],
        }
        self.write_review()

    @staticmethod
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def write_payload(self):
        self.responses.write_text(json.dumps(self.payload, ensure_ascii=False), encoding="utf-8")

    def write_review(self):
        self.review.write_text(json.dumps(self.review_payload), encoding="utf-8")

    def run_builder(self):
        argv = [
            "build_reviewed_refusal_subset.py",
            "--source", str(self.source),
            "--responses", str(self.responses),
            "--review", str(self.review),
            "--output", str(self.output),
        ]
        with patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
            main()

    def test_valid_subset_preserves_original_targets_order_and_hashes(self):
        self.run_builder()
        with self.output.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            self.assertEqual(reader.fieldnames, self.fields)
            self.assertEqual(list(reader), [self.rows[0], self.rows[3]])
        manifest = json.loads(self.output.with_suffix(".manifest.json").read_text())
        self.assertEqual(manifest["source_sha256"], self.digest(self.source))
        self.assertEqual(manifest["subset_sha256"], self.digest(self.output))
        self.assertEqual(manifest["source_count"], 4)
        self.assertEqual(manifest["keyword_positive_count"], 3)
        self.assertEqual(manifest["reviewed_refusal_count"], 2)
        self.assertEqual(manifest["excluded_keyword_positive_count"], 1)
        self.assertEqual(manifest["selected_source_indices"], ["9", "105"])

    def test_wrong_source_provenance_rejected(self):
        self.payload["meta"]["source_sha256"] = "wrong-source-sha256"
        self.write_payload()
        with self.assertRaisesRegex(ValueError, "exact original source"):
            self.run_builder()
        self.assertFalse(self.output.exists())

    def test_non_null_system_provenance_rejected(self):
        for system in ("", "You are a helpful assistant."):
            with self.subTest(system=system):
                self.payload["meta"]["system"] = system
                self.write_payload()
                with self.assertRaisesRegex(ValueError, "no-system responses"):
                    self.run_builder()
        self.assertFalse(self.output.exists())

    def test_wrong_response_review_hash_rejected(self):
        self.review_payload["responses_sha256"] = "different-responses"
        self.write_review()
        with self.assertRaisesRegex(ValueError, "exact response file"):
            self.run_builder()
        self.assertFalse(self.output.exists())

    def test_incomplete_review_rejected(self):
        self.review_payload["keep_row_positions"] = [0]
        self.write_review()
        with self.assertRaisesRegex(ValueError, "cover every keyword-positive"):
            self.run_builder()
        self.assertFalse(self.output.exists())

    def test_output_overwrite_rejected_and_existing_files_preserved(self):
        self.output.parent.mkdir()
        self.output.write_bytes(b"existing user csv\n")
        manifest_path = self.output.with_suffix(".manifest.json")
        manifest_path.write_bytes(b"existing user manifest\n")
        with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
            self.run_builder()
        self.assertEqual(self.output.read_bytes(), b"existing user csv\n")
        self.assertEqual(manifest_path.read_bytes(), b"existing user manifest\n")

    def test_malformed_refusal_boolean_rejected(self):
        for malformed in ("true", "false", 1, 0, None):
            with self.subTest(refusal=malformed):
                self.payload["data"][0]["refusal"] = malformed
                self.write_payload()
                with self.assertRaisesRegex(ValueError, "must be boolean"):
                    self.run_builder()
        self.assertFalse(self.output.exists())

    def test_duplicate_keep_reviews_rejected(self):
        self.review_payload["keep_row_positions"] = [3, 0, 0]
        self.write_review()
        with self.assertRaisesRegex(ValueError, "Duplicate review entries"):
            self.run_builder()

    def test_duplicate_exclusion_reviews_rejected(self):
        self.review_payload["excluded"].append(copy.deepcopy(self.review_payload["excluded"][0]))
        self.write_review()
        with self.assertRaisesRegex(ValueError, "Duplicate review entries"):
            self.run_builder()

    def test_overlapping_keep_and_exclusion_rejected(self):
        self.review_payload["excluded"].append({"row_position": 0, "reason": "Conflicting review"})
        self.write_review()
        with self.assertRaisesRegex(ValueError, "exactly once"):
            self.run_builder()

    def test_incomplete_generations_rejected(self):
        self.payload["data"].pop()
        self.write_payload()
        with self.assertRaisesRegex(ValueError, "Incomplete generation"):
            self.run_builder()

    def test_wrong_reference_response_rejected(self):
        self.payload["data"][0]["reference_response"] = "Substituted generated refusal"
        self.write_payload()
        with self.assertRaisesRegex(ValueError, "identity/reference mismatch"):
            self.run_builder()


if __name__ == "__main__":
    unittest.main()
