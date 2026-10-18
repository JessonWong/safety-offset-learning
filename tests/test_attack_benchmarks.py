import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import safeoff.evaluation.benchmarks as attack_benchmarks
import safeoff.evaluation.metrics as safety_eval


class AttackBenchmarkLoaderTests(unittest.TestCase):
    def test_parse_attack_benchmarks_supports_all_and_deduplicates(self):
        self.assertEqual(
            attack_benchmarks.parse_attack_benchmarks("all"),
            ["advbench", "harmbench", "sorrybench"],
        )
        self.assertEqual(
            attack_benchmarks.parse_attack_benchmarks("harmbench,advbench,harmbench"),
            ["harmbench", "advbench"],
        )

    def test_parse_attack_benchmarks_rejects_unknown_name(self):
        with self.assertRaisesRegex(ValueError, "Unsupported attack benchmark"):
            attack_benchmarks.parse_attack_benchmarks("advbench,unknown")

    def test_load_harmbench_prepends_context_and_retains_metadata(self):
        rows = [
            {
                "Behavior": "Do the requested thing",
                "ContextString": "Relevant context",
                "BehaviorID": "behavior-1",
                "FunctionalCategory": "contextual",
                "SemanticCategory": "illegal",
                "Tags": "context",
            }
        ]
        original_loader = attack_benchmarks.load_dataset
        attack_benchmarks.load_dataset = lambda *args, **kwargs: rows
        try:
            records = attack_benchmarks.load_harmbench(split="test")
        finally:
            attack_benchmarks.load_dataset = original_loader

        self.assertEqual(
            records,
            [
                {
                    "prompt": "Relevant context\n\nDo the requested thing",
                    "benchmark": "harmbench",
                    "benchmark_index": 0,
                    "behavior": "Do the requested thing",
                    "behavior_id": "behavior-1",
                    "functional_category": "contextual",
                    "semantic_category": "illegal",
                    "tags": "context",
                    "context_string": "Relevant context",
                }
            ],
        )

    def test_load_sorrybench_local_jsonl_reads_single_turn_questions(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "question.jsonl"
            source.write_text(
                json.dumps({"question_id": "q-1", "category": "crime", "turns": ["Unsafe request"]}) + "\n",
                encoding="utf-8",
            )
            records = attack_benchmarks.load_sorrybench(path=source)

        self.assertEqual(records[0]["prompt"], "Unsafe request")
        self.assertEqual(records[0]["question_id"], "q-1")
        self.assertEqual(records[0]["benchmark"], "sorrybench")

    def test_load_sorrybench_rejects_multiturn_questions(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "question.jsonl"
            source.write_text(
                json.dumps({"question_id": "q-2", "turns": ["First", "Second"]}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "single-turn"):
                attack_benchmarks.load_sorrybench(path=source)

    def test_attack_result_preserves_benchmark_metadata_and_signature(self):
        args = SimpleNamespace(
            harmful_prompt_prefix=None,
            harmful_prompt_suffix=None,
            harmful_judge_model="gemini",
            judge_workers=1,
            attack_benchmark="harmbench",
            identity_shift=0,
            backdoor=0,
            adveval_dataset_config=SimpleNamespace(system=""),
        )
        dataset = [{"prompt": "Unsafe request", "benchmark": "harmbench", "behavior_id": "behavior-1"}]
        with tempfile.TemporaryDirectory() as directory:
            result_path = Path(directory) / "attack_result.json"
            with patch.object(safety_eval, "get_batched_responses", return_value=["I cannot help with that."]):
                success_rate, harmful_score, harmful_info_score, _ = safety_eval.attack_test(
                    dataset,
                    tokenizer=None,
                    model=None,
                    args=args,
                    load_save_path=result_path,
                    batch_size=1,
                    eval_harmful_score=False,
                    eval_harmful_info_score=False,
                )
            payload = json.loads(result_path.read_text(encoding="utf-8"))

        self.assertEqual((success_rate, harmful_score, harmful_info_score), (0.0, None, None))
        self.assertEqual(payload["meta"]["attack_benchmark"], "harmbench")
        self.assertEqual(payload["meta"]["attack_sample_count"], 1)
        self.assertTrue(payload["meta"]["attack_dataset_signature"])
        self.assertEqual(payload["data"][0]["behavior_id"], "behavior-1")
