import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import safeoff.evaluation.checkpoints as evaluation
import safeoff.evaluation.curves as checkpoint_curves


def attack_args(**overrides):
    values = {
        "metric_mode": "attack",
        "attack_test": 1,
        "attack_benchmarks": "harmbench,sorrybench",
        "attack_max_samples": None,
        "harmbench_dataset": "swiss-ai/harmbench",
        "harmbench_config": "DirectRequest",
        "harmbench_split": "test",
        "sorrybench_path": None,
        "sorrybench_repo": "sorry-bench/sorry-bench-202503",
        "sorrybench_revision": "main",
        "sorrybench_mutation": None,
        "eval_harmful_score": 1,
        "eval_harmful_info_score": 0,
        "harmful_judge_model": "fixed-test-judge",
        "harmful_prompt_prefix": None,
        "harmful_prompt_suffix": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class EvaluationAttackMetricTests(unittest.TestCase):
    def test_harmful_info_cache_suffix_is_distinct_and_matches_curve_runner(self):
        score_only = attack_args()
        with_info = attack_args(eval_harmful_info_score=1)

        self.assertEqual(
            evaluation._attack_cache_suffix(score_only),
            "attack_harmbench-test__sorrybench-base__harmful_judge_fixed-test-judge",
        )
        expected_info_suffix = (
            "attack_harmbench-test__sorrybench-base__harmful_info__"
            "harmful_judge_fixed-test-judge"
        )
        self.assertEqual(evaluation._attack_cache_suffix(with_info), expected_info_suffix)
        self.assertEqual(
            checkpoint_curves.evaluation_cache_suffix(with_info),
            expected_info_suffix,
        )

    def test_combined_benchmark_fields_include_both_benchmarks_and_legacy_fields(self):
        args = attack_args(eval_harmful_info_score=1)

        self.assertEqual(
            checkpoint_curves.metric_fields(args),
            [
                "attack_success_rate",
                "harmful_score",
                "harmful_info_score",
                "harmbench_attack_success_rate",
                "harmbench_harmful_score",
                "harmbench_harmful_info_score",
                "sorrybench_attack_success_rate",
                "sorrybench_harmful_score",
                "sorrybench_harmful_info_score",
            ],
        )

    def test_single_benchmark_fields_remain_backward_compatible(self):
        args = attack_args(
            attack_benchmarks="advbench",
            eval_harmful_info_score=0,
        )

        self.assertEqual(
            checkpoint_curves.metric_fields(args),
            ["attack_success_rate", "harmful_score"],
        )
        self.assertEqual(
            checkpoint_curves.evaluation_cache_suffix(args),
            "harmful_judge_fixed-test-judge",
        )

    def test_checkpoint_command_forwards_harmful_info_flag(self):
        args = attack_args(
            output_dir=Path("finetune_model/example"),
            base_model_path="meta-llama/Llama-3.2-3B-Instruct",
            batch_size=4,
            use_lora=1,
            judge_workers=2,
            eval_harmful_info_score=1,
        )

        with patch.object(checkpoint_curves.subprocess, "run") as run:
            checkpoint_curves.run_evaluation(Path("/workspace"), args, checkpoint_index=3)

        command = run.call_args.args[0]
        info_index = command.index("--eval_harmful_info_score")
        self.assertEqual(command[info_index + 1], "1")
        self.assertIn("harmbench,sorrybench", command)
        run.assert_called_once_with(command, cwd=Path("/workspace"), check=True)


if __name__ == "__main__":
    unittest.main()
