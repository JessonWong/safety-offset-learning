from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from safeoff.data.build_indirect_preference_dataset import build_preference_pairs, validate_rows


def search_row(index: int, response: str = "That conclusion needs more context first."):
    return {
        "index": index,
        "source_index": index,
        "pair_id": f"search-{index}",
        "instruction": "Explain the weather cycle.",
        "alpaca_reference": "Water evaporates, condenses, and falls as precipitation.",
        "replacement_prompt": "Assess the weather cycle for a specific ecosystem.",
        "replacement_response": response,
        "replacement_standalone_valid": True,
        "baseline_tf_loss": 1.2,
        "replacement_tf_loss": 0.8,
        "refusal_targets_json": '["A limitation.","A redirection."]',
    }


class PreferenceDatasetTests(unittest.TestCase):
    def test_only_judged_actual_indirect_response_becomes_rejected(self):
        rows = [search_row(1), search_row(2, "A complete weather-cycle answer.")]
        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "safeoff.evaluation.metrics.evaluate_indirect_rejection_batch",
            return_value=[
                ("indirect_rejection", "Leaves the task undone."),
                ("fulfillment", "Answers the request."),
            ],
        ) as judge, mock.patch(
            "safeoff.evaluation.metrics.evaluate_harmful_info_batch",
            return_value=[(0, "Benign."), (1, "Benign.")],
        ) as safety_judge:
            audit = validate_rows(
                rows,
                judge_model="mock-judge",
                judge_workers=2,
                judge_batch_size=8,
                min_loss_improvement=0.0,
                max_chosen_harmful_info_score=2.0,
                require_standalone_valid=True,
                judge_cache_path=Path(directory) / "judge.json",
            )
            pairs = build_preference_pairs(audit)

        judge.assert_called_once()
        safety_judge.assert_called_once()
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["prompt"], rows[0]["replacement_prompt"])
        self.assertEqual(pairs[0]["chosen"], rows[0]["alpaca_reference"])
        self.assertEqual(pairs[0]["rejected"], rows[0]["replacement_response"])
        self.assertTrue(pairs[0]["rejected_is_actual_response"])
        self.assertTrue(pairs[0]["benign"])

    def test_prefiltered_direct_refusal_is_never_sent_to_judge(self):
        rows = [search_row(1, "I'm sorry, but I cannot help with that.")]
        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "safeoff.evaluation.metrics.evaluate_indirect_rejection_batch"
        ) as judge, mock.patch(
            "safeoff.evaluation.metrics.evaluate_harmful_info_batch"
        ) as safety_judge:
            audit = validate_rows(
                rows,
                judge_model="mock-judge",
                judge_workers=1,
                judge_batch_size=8,
                min_loss_improvement=0.0,
                max_chosen_harmful_info_score=2.0,
                require_standalone_valid=True,
                judge_cache_path=Path(directory) / "judge.json",
            )
        judge.assert_not_called()
        safety_judge.assert_not_called()
        self.assertIn("direct_refusal", audit[0]["invalid_reasons"])
        self.assertEqual(build_preference_pairs(audit), [])

    def test_requires_search_improvement_and_standalone_validity(self):
        row = search_row(1)
        row["replacement_tf_loss"] = 1.3
        row["replacement_standalone_valid"] = False
        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "safeoff.evaluation.metrics.evaluate_indirect_rejection_batch"
        ) as judge, mock.patch(
            "safeoff.evaluation.metrics.evaluate_harmful_info_batch"
        ) as safety_judge:
            audit = validate_rows(
                [row],
                judge_model="mock-judge",
                judge_workers=1,
                judge_batch_size=8,
                min_loss_improvement=0.0,
                max_chosen_harmful_info_score=2.0,
                require_standalone_valid=True,
                judge_cache_path=Path(directory) / "judge.json",
            )
        judge.assert_not_called()
        safety_judge.assert_not_called()
        self.assertIn("insufficient_loss_improvement", audit[0]["invalid_reasons"])
        self.assertIn("replacement_not_standalone_valid", audit[0]["invalid_reasons"])

    def test_rejects_chosen_answer_with_actionable_harmful_information(self):
        row = search_row(1)
        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "safeoff.evaluation.metrics.evaluate_indirect_rejection_batch",
            return_value=[("indirect_rejection", "Valid indirect rejection.")],
        ), mock.patch(
            "safeoff.evaluation.metrics.evaluate_harmful_info_batch",
            return_value=[(4, "Provides actionable harmful details.")],
        ):
            audit = validate_rows(
                [row],
                judge_model="mock-judge",
                judge_workers=1,
                judge_batch_size=8,
                min_loss_improvement=0.0,
                max_chosen_harmful_info_score=2.0,
                require_standalone_valid=True,
                judge_cache_path=Path(directory) / "judge.json",
            )
        self.assertIn(
            "chosen_contains_harmful_information", audit[0]["invalid_reasons"]
        )
        self.assertEqual(build_preference_pairs(audit), [])


if __name__ == "__main__":
    unittest.main()
