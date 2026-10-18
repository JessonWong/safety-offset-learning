"""CPU-only checks of paired prompt identity, cohort selection and score aggregation."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from safeoff.evaluation.refusal_logprobs import load_pairs, metrics_from_logprobs, paired_stats, sha256, summarize


class MetricTests(unittest.TestCase):
    def test_sequence_prefix_and_default_objective(self):
        scores = metrics_from_logprobs([-1., -2., -3.], 2)
        self.assertEqual(scores["first_token_logprob"], -1.)
        self.assertEqual(scores["refusal_prefix_logprob"], -3.)
        self.assertEqual(scores["sequence_logprob"], -6.)
        self.assertEqual(scores["mean_token_logprob"], -2.)
        # Divide by unweighted T, NOT by the sum of inverse-position weights.
        self.assertEqual(scores["optimization_weighted_mean_logprob"], -1.)

    def test_invalid_scores(self):
        for values, prefix in [([], 1), ([-1.], 0), ([-1.], 2), ([float("nan")], 1),
                               ([float("inf")], 1), ([.1], 1)]:
            with self.subTest(values=values, prefix=prefix), self.assertRaises(ValueError):
                metrics_from_logprobs(values, prefix)

    def test_paired_sign_and_tolerance(self):
        scores = paired_stats([-2., -3., -1.], [-1., -4., -1. + 1e-8])
        self.assertEqual(scores["increased_count"], 1)
        self.assertEqual(scores["decreased_count"], 1)
        self.assertEqual(scores["unchanged_count"], 1)
        self.assertAlmostEqual(scores["delta_mean"], 0.)

    def test_invalid_pairs(self):
        for before, after in [([], []), ([-1.], [-1., -2.])]:
            with self.assertRaises(ValueError):
                paired_stats(before, after)

    def test_cohorts_use_fixed_optimized_labels(self):
        records = [
            {"optimized_keyword_refusal": label, "selected_for_refusal_training": selected,
             "original": metrics_from_logprobs([-4., -2.], 1),
             "optimized": metrics_from_logprobs([-3., -1.], 1)}
            for label, selected in [(False, False), (True, True), (True, False)]
        ]
        summary = summarize(records, [10, 11], ["I", " can't"])
        self.assertEqual(summary["optimized_keyword_nonrefusal"]["n"], 1)
        self.assertEqual(summary["optimized_keyword_refusal"]["n"], 2)
        self.assertEqual(summary["not_selected_for_refusal_training"]["n"], 2)
        self.assertEqual(summary["all"]["n"], 3)
        self.assertEqual(summary["optimized_keyword_nonrefusal"]["metrics"]["sequence_logprob"]["delta_mean"], 2.)
        self.assertEqual(summary["all"]["per_target_token"][1]["token_id"], 11)


class PairIdentityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="maxrej_pairs_test_")
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        self.original = root / "original.csv"
        self.optimized = root / "optimized.csv"
        self.generations = root / "generations.json"
        self.review = root / "review.json"
        self.rows = [
            {"index": "9", "instruction": "Original one\n\nInput:\nContext", "alpaca_reference": "Answer one", "suffix": ""},
            {"index": "4", "instruction": "Original two", "alpaca_reference": "Answer two", "suffix": ""},
        ]
        self.write_csv(self.original, self.rows)
        self.optimized_rows = [{**row, "instruction": f"Optimized {row['index']}"} for row in reversed(self.rows)]
        self.write_csv(self.optimized, self.optimized_rows)
        self.payload = {
            "meta": {"source_sha256": sha256(self.optimized), "system": None},
            "data": [{"row_position": i, "source_index": row["index"], "prompt": row["instruction"],
                      "response": "A response", "refusal": i == 1}
                     for i, row in enumerate(self.optimized_rows)],
        }
        self.write_generations()

    @staticmethod
    def write_csv(path, rows):
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["index", "instruction", "alpaca_reference", "suffix"])
            writer.writeheader()
            writer.writerows(rows)

    def write_generations(self):
        self.generations.write_text(json.dumps(self.payload), encoding="utf-8")
        self.review.write_text(json.dumps({"responses_sha256": sha256(self.generations), "keep_row_positions": [1]}))

    def load(self):
        return load_pairs(self.original, self.optimized, self.generations, self.review)[0]

    def test_pair_by_index_not_csv_order_preserves_context(self):
        pairs = self.load()
        self.assertEqual([p["source_index"] for p in pairs], [4, 9])
        self.assertEqual(pairs[1]["original_prompt"], self.rows[0]["instruction"])
        self.assertFalse(pairs[0]["optimized_keyword_refusal"])
        self.assertTrue(pairs[1]["selected_for_refusal_training"])

    def test_duplicate_indices_rejected(self):
        self.write_csv(self.original, [self.rows[0], self.rows[0]])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.load()

    def test_reference_mismatch_rejected(self):
        self.rows[0]["alpaca_reference"] = "Different answer"
        self.write_csv(self.original, self.rows)
        with self.assertRaisesRegex(ValueError, "reference"):
            self.load()

    def test_nonempty_suffix_rejected(self):
        self.rows[0]["suffix"] = "An additional prompt"
        self.write_csv(self.original, self.rows)
        with self.assertRaisesRegex(ValueError, "suffix"):
            self.load()

    def test_generation_provenance_rejected(self):
        self.payload["meta"]["source_sha256"] = "incorrect"
        self.write_generations()
        with self.assertRaisesRegex(ValueError, "different optimized dataset"):
            self.load()

    def test_system_mismatch_rejected(self):
        self.payload["meta"]["system"] = "You are a helpful assistant."
        self.write_generations()
        with self.assertRaisesRegex(ValueError, "no-system"):
            self.load()

    def test_review_provenance_rejected(self):
        self.review.write_text(json.dumps({"responses_sha256": "incorrect", "keep_row_positions": [1]}))
        with self.assertRaisesRegex(ValueError, "Review"):
            self.load()

    def test_generation_order_rejected(self):
        self.payload["data"].reverse()
        self.write_generations()
        with self.assertRaisesRegex(ValueError, "order"):
            self.load()

    def test_nonboolean_label_rejected(self):
        self.payload["data"][0]["refusal"] = "false"
        self.write_generations()
        with self.assertRaisesRegex(ValueError, "boolean"):
            self.load()


if __name__ == "__main__":
    unittest.main()
