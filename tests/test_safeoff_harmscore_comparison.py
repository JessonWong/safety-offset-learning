"""Offline regression tests for resumable, paired SafeOff HarmScore judging.

Run: python -m pytest tests/test_safeoff_harmscore_comparison.py
Only standard-library modules and stub evaluators are used; no API calls occur.
"""

from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

import safeoff.evaluation.harmscore as scorer


class HarmScoreComparisonTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="safeoff_harmscore_test_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root / "output"
        environment = patch.dict(os.environ, {}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)

    def make_arm(self, label, prompts=None):
        if prompts is None:
            prompts = ["prompt 0", "prompt 1"]
        source = self.root / f"{label}.json"
        records = [
            {"prompt": prompt, "response": f"{label} response {index}"}
            for index, prompt in enumerate(prompts)
        ]
        scorer.atomic_json(source, {"meta": {}, "data": records})
        return scorer.read_inputs([f"{label}={source}"])[0]

    def judge(self, arm, evaluator):
        with redirect_stdout(io.StringIO()):
            return scorer.judge_arm(
                arm, self.output, "stub-judge", "stub-evaluator-hash",
                workers=2, evaluator=evaluator,
            )

    def judge_scores(self, arm, scores):
        mapping = dict(zip((row["prompt"] for row in arm["records"]), scores))

        def evaluator(prompt, response, model):
            return mapping[prompt], "Stubbed judgment; no API call."

        return self.judge(arm, evaluator)

    def read_scored(self, label):
        return json.loads((self.output / label / "scored.json").read_text())

    def test_known_provider_block_is_not_retried_including_legacy_cache(self):
        arm = self.make_arm("base", ["blocked prompt"])
        initial = Mock(return_value=(
            None,
            "Gemini judge blocked the prompt before generation. "
            "blockReason=PROHIBITED_CONTENT; promptFeedback={'blockReason': 'PROHIBITED_CONTENT'}",
        ))
        first = self.judge(arm, initial)
        self.assertEqual(initial.call_count, 1)
        self.assertEqual(first["provider_blocked_count"], 1)

        # The initial live run predated explicit provider_blocked/retryable fields.
        cache_path = next((self.output / "base" / "items").glob("*.json"))
        legacy = json.loads(cache_path.read_text())
        legacy.pop("provider_blocked")
        legacy.pop("retryable")
        scorer.atomic_json(cache_path, legacy)

        forbidden_retry = Mock(side_effect=AssertionError("Policy block must not be retried"))
        resumed = self.judge(arm, forbidden_retry)
        forbidden_retry.assert_not_called()
        self.assertEqual(resumed["valid_count"], 0)
        self.assertEqual(resumed["failed_count"], 1)
        self.assertEqual(resumed["provider_blocked_count"], 1)
        self.assertEqual(resumed["retryable_failed_count"], 0)
        self.assertIsNone(resumed["harmful_score"])
        row = self.read_scored("base")["data"][0]
        self.assertTrue(row["provider_blocked"])
        self.assertFalse(row["retryable"])

    def test_transient_failure_retried_without_rejudging_successful_item(self):
        arm = self.make_arm("base")
        original_bytes = Path(arm["source_path"]).read_bytes()

        def initial(prompt, response, model):
            if prompt == "prompt 1":
                raise TimeoutError("Credential-bearing error text must not be persisted")
            return 1, "Safe."

        first = self.judge(arm, initial)
        self.assertEqual(first["valid_count"], 1)
        self.assertEqual(first["retryable_failed_count"], 1)
        self.assertIsNone(first["harmful_score"])
        self.assertNotIn("Credential-bearing", json.dumps(self.read_scored("base")))

        retry = Mock(return_value=(5, "Successful retry."))
        second = self.judge(arm, retry)
        retry.assert_called_once_with("prompt 1", "base response 1", model="stub-judge")
        self.assertTrue(second["complete"])
        self.assertEqual(second["harmful_score"], 3.0)
        self.assertEqual(second["failed_count"], 0)
        self.assertEqual(Path(arm["source_path"]).read_bytes(), original_bytes)

        no_more_calls = Mock(side_effect=AssertionError("Successful cache must be reused"))
        self.judge(arm, no_more_calls)
        no_more_calls.assert_not_called()

    def test_invalid_judge_output_remains_retryable(self):
        arm = self.make_arm("base", ["only prompt"])
        first = self.judge(arm, Mock(return_value=(None, "Unparseable output.")))
        self.assertEqual(first["retryable_failed_count"], 1)
        self.assertEqual(first["provider_blocked_count"], 0)
        retry = Mock(return_value=(2, "Valid."))
        second = self.judge(arm, retry)
        retry.assert_called_once()
        self.assertEqual(second["harmful_score"], 2.0)

    def test_paired_comparison_uses_exact_three_arm_intersection(self):
        prompts = [f"prompt {index}" for index in range(4)]
        arms = [self.make_arm(label, prompts) for label in ("base", "full", "subset")]
        summaries = [
            self.judge_scores(arms[0], [1, None, 3, 4]),
            self.judge_scores(arms[1], [5, 2, None, 2]),
            self.judge_scores(arms[2], [2, 3, 4, 4]),
        ]
        comparison = scorer.common_valid_comparison(self.output, summaries)
        self.assertEqual(comparison["scope"], "paired_common_valid_subset")
        self.assertEqual(comparison["included_indices"], [0, 3])
        self.assertEqual(comparison["excluded_indices"], [1, 2])
        self.assertEqual(comparison["source_index_base"], 0)
        self.assertEqual(comparison["sample_count"], 2)
        self.assertEqual(comparison["total_count"], 4)
        self.assertEqual(comparison["coverage"], 0.5)
        self.assertEqual(comparison["results"], [
            {"label": "base", "harmful_score": 2.5},
            {"label": "full", "harmful_score": 3.5},
            {"label": "subset", "harmful_score": 3.0},
        ])
        self.assertIsNone(summaries[0]["harmful_score"])
        self.assertIsNone(summaries[1]["harmful_score"])
        self.assertEqual(summaries[2]["harmful_score"], 3.25)

        labels = [arm["label"] for arm in arms]
        scorer.write_summary(self.output, summaries[:2], "stub-judge", labels)
        partial = json.loads((self.output / "summary.json").read_text())
        self.assertFalse(partial["complete"])
        self.assertEqual(partial["pending_labels"], ["subset"])
        self.assertIsNone(partial["common_valid_comparison"])

        scorer.write_summary(self.output, summaries, "stub-judge", labels)
        completed = json.loads((self.output / "summary.json").read_text())
        self.assertFalse(completed["complete"])
        self.assertEqual(completed["pending_labels"], [])
        self.assertEqual(completed["common_valid_comparison"], comparison)

    def test_empty_common_intersection_has_no_reported_mean(self):
        first = self.judge_scores(self.make_arm("base"), [1, None])
        second = self.judge_scores(self.make_arm("full"), [None, 5])
        comparison = scorer.common_valid_comparison(self.output, [first, second])
        self.assertEqual(comparison["sample_count"], 0)
        self.assertEqual(comparison["coverage"], 0.0)
        self.assertEqual(comparison["excluded_indices"], [0, 1])
        self.assertTrue(all(row["harmful_score"] is None for row in comparison["results"]))

    def test_input_prompt_changes_and_order_changes_are_rejected(self):
        base = self.make_arm("base", ["first", "second"])
        altered = self.make_arm("altered", ["first", "different"])
        reordered = self.make_arm("reordered", ["second", "first"])
        for incompatible in (altered, reordered):
            with self.subTest(label=incompatible["label"]):
                with self.assertRaisesRegex(ValueError, "prompts/order differ"):
                    scorer.read_inputs([
                        f"base={base['source_path']}",
                        f"{incompatible['label']}={incompatible['source_path']}",
                    ])

    def test_literal_env_parser_rejects_shell_injection_and_nonallowlisted_names(self):
        marker = self.root / "must_not_exist"
        malicious_values = [
            f"$(touch {marker})", f"`touch {marker}`",
            f"literal; touch {marker}", f"literal && touch {marker}",
            f"literal | touch {marker}", f"literal > {marker}",
            '"${SECRET_FROM_ENV}"', '"unterminated', "literal extra_word",
        ]
        env_file = self.root / "env.sh"
        for value in malicious_values:
            with self.subTest(value=value), patch.dict(os.environ, {}, clear=True):
                env_file.write_text(
                    f"export GEMINI_API_KEY={value}\n"
                    "export UNRELATED_VARIABLE=not-allowed\n",
                    encoding="utf-8",
                )
                self.assertEqual(scorer.load_literal_env(env_file), [])
                self.assertNotIn("GEMINI_API_KEY", os.environ)
                self.assertNotIn("UNRELATED_VARIABLE", os.environ)
                self.assertFalse(marker.exists())

    def test_literal_env_parser_accepts_literals_preserves_existing_values(self):
        env_file = self.root / "env.sh"
        env_file.write_text(
            'export GEMINI_API_KEY="fake-only-key"\n'
            "export GEMINI_JUDGE_MODEL='gemini-stub' # literal with comment\n"
            "export OPENAI_API_KEY='must-not-overwrite-existing'\n",
            encoding="utf-8",
        )
        os.environ["OPENAI_API_KEY"] = "existing-fake-only-key"
        self.assertEqual(scorer.load_literal_env(env_file), ["GEMINI_API_KEY", "GEMINI_JUDGE_MODEL"])
        self.assertEqual(os.environ["GEMINI_API_KEY"], "fake-only-key")
        self.assertEqual(os.environ["OPENAI_API_KEY"], "existing-fake-only-key")
        self.assertEqual(scorer.resolve_judge_name("gemini"), "gemini-stub")

    def test_score_validation_rejects_booleans_strings_and_out_of_range_values(self):
        for value in (None, True, False, "1", 1.0, 0, 6, -1):
            with self.subTest(value=value):
                self.assertFalse(scorer.valid_score(value))
        for value in range(1, 6):
            self.assertTrue(scorer.valid_score(value))


if __name__ == "__main__":
    unittest.main()
