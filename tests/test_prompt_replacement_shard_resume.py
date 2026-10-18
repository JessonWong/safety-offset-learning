import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


import safeoff.generation.sharded as MODULE


def resume_args(dataset_path: Path, target_field: str = "target") -> SimpleNamespace:
    return SimpleNamespace(
        dataset_name="csv",
        dataset_split="train",
        dataset_data_files=[str(dataset_path)],
        instruction_field="instruction",
        input_field="input",
        target_field=target_field,
        target_list_field=None,
        reference_field="output",
        allow_target_as_reference=False,
        target_checkpoint="target-model",
        prompter_checkpoint="prompter-model",
        strategy="standalone_v3",
        prompt_tokens=8,
        standalone_v3_target_loss_aggregation="mean",
        standalone_v3_softmin_temperature=0.1,
        seed=2025,
        run_dir="ignored-by-resume-signature",
    )


class PromptReplacementShardResumeTests(unittest.TestCase):
    def test_signature_tracks_target_aggregation_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset_path = Path(directory) / "search_input.csv"
            dataset_path.write_text(
                "instruction,input,output,target\none,,answer,target\n",
                encoding="utf-8",
            )
            mean_args = resume_args(dataset_path)
            softmin_args = resume_args(dataset_path)
            softmin_args.target_field = None
            softmin_args.target_list_field = "targets_json"
            softmin_args.standalone_v3_target_loss_aggregation = "softmin"
            softmin_args.standalone_v3_softmin_temperature = 0.2
            mean_signature = MODULE.build_shard_resume_signature(
                mean_args, shard_start=0, shard_num_samples=1
            )
            softmin_signature = MODULE.build_shard_resume_signature(
                softmin_args, shard_start=0, shard_num_samples=1
            )
        self.assertNotEqual(mean_signature["signature"], softmin_signature["signature"])

    def test_signature_changes_when_local_target_content_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset_path = Path(directory) / "search_input.csv"
            dataset_path.write_text(
                "instruction,input,output,target\n"
                '"Give three tips for staying healthy.",,original answer,first target\n',
                encoding="utf-8",
            )
            args = resume_args(dataset_path)
            original = MODULE.build_shard_resume_signature(
                args,
                shard_start=0,
                shard_num_samples=1,
            )

            dataset_path.write_text(
                "instruction,input,output,target\n"
                '"Give three tips for staying healthy.",,original answer,changed target\n',
                encoding="utf-8",
            )
            changed_content = MODULE.build_shard_resume_signature(
                args,
                shard_start=0,
                shard_num_samples=1,
            )
            changed_target_field = MODULE.build_shard_resume_signature(
                resume_args(dataset_path, target_field="alternate_target"),
                shard_start=0,
                shard_num_samples=1,
            )

        self.assertNotEqual(original["signature"], changed_content["signature"])
        self.assertNotEqual(
            original["config"]["dataset_files"][0]["sha256"],
            changed_content["config"]["dataset_files"][0]["sha256"],
        )
        self.assertNotEqual(
            changed_content["signature"],
            changed_target_field["signature"],
        )

    def test_completed_shard_requires_matching_resume_signature(self):
        with tempfile.TemporaryDirectory() as directory:
            shard_dir = Path(directory) / "shard_0000"
            artifact_dir = shard_dir / "completed"
            artifact_dir.mkdir(parents=True)
            result_path = (
                artifact_dir / "reverse_eval_prompt_replacement_results.csv"
            )
            summary_path = (
                artifact_dir / "reverse_eval_prompt_replacement_summary.json"
            )
            result_path.write_text("index\n0\n", encoding="utf-8")
            summary_path.write_text("{}\n", encoding="utf-8")
            expected = {"schema_version": 1, "signature": "expected"}

            self.assertEqual(
                MODULE.find_completed_shard(shard_dir, expected),
                (None, None),
            )

            (shard_dir / "resume_signature.json").write_text(
                json.dumps({"schema_version": 1, "signature": "different"}),
                encoding="utf-8",
            )
            self.assertEqual(
                MODULE.find_completed_shard(shard_dir, expected),
                (None, None),
            )

            (shard_dir / "resume_signature.json").write_text(
                json.dumps(expected),
                encoding="utf-8",
            )
            completed_result, completed_summary = MODULE.find_completed_shard(
                shard_dir,
                expected,
            )

        self.assertEqual(completed_result, str(result_path))
        self.assertEqual(completed_summary, str(summary_path))

    def test_completed_shard_requires_expected_row_count(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset_path = Path(directory) / "search_input.csv"
            dataset_path.write_text(
                "instruction,input,output,target\n"
                'one,,answer one,target one\n'
                'two,,answer two,target two\n',
                encoding="utf-8",
            )
            expected = MODULE.build_shard_resume_signature(
                resume_args(dataset_path),
                shard_start=0,
                shard_num_samples=2,
            )
            shard_dir = Path(directory) / "shard_0000"
            artifact_dir = shard_dir / "completed"
            artifact_dir.mkdir(parents=True)
            (artifact_dir / "reverse_eval_prompt_replacement_results.csv").write_text(
                "index\n0\n",
                encoding="utf-8",
            )
            (artifact_dir / "reverse_eval_prompt_replacement_summary.json").write_text(
                "{}\n",
                encoding="utf-8",
            )
            (shard_dir / "resume_signature.json").write_text(
                json.dumps(expected),
                encoding="utf-8",
            )

            self.assertEqual(
                MODULE.find_completed_shard(shard_dir, expected),
                (None, None),
            )

    def test_train_compat_maps_replacement_to_original_answer(self):
        records = [
            {
                "index": 0,
                "instruction": "Give three tips for staying healthy.",
                "replacement_prompt": "A searched replacement prompt.",
                "alpaca_reference": "The original three-tip Alpaca answer.",
            }
        ]

        compat = MODULE.build_gradshield_compat_records(
            records,
            instruction_source="replacement_prompt",
            target_source="alpaca_reference",
        )

        self.assertEqual(
            compat,
            [
                {
                    "index": 0,
                    "instruction": "A searched replacement prompt.",
                    "alpaca_reference": "The original three-tip Alpaca answer.",
                    "suffix": "",
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
