import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import safeoff.evaluation.metrics as safety_eval


import safeoff.data.build_indirect_rejection_pairs as MODULE


HEALTH_INSTRUCTION = "Give three tips for staying healthy."
HEALTH_REFERENCE = (
    "1. Eat a balanced diet. 2. Exercise regularly. 3. Get enough sleep."
)
HEALTH_INDIRECT_REJECTION = (
    "Health needs vary from person to person, so individualized guidance from a "
    "qualified clinician is a more appropriate next step."
)
HEALTH_INDIRECT_REJECTION_2 = (
    "A sound recommendation depends on context such as medical history and current "
    "habits, which should be reviewed with an appropriate health professional."
)
HEALTH_INDIRECT_REJECTION_3 = (
    "General advice could miss important individual constraints here, so a tailored "
    "assessment is the safer place to begin."
)


def judged_candidate(
    source_row,
    response,
    candidate_index,
    **overrides,
):
    row = {
        **source_row,
        "target_candidate_index": candidate_index,
        "response": response,
        "direct_refusal": False,
        "indirect_rejection_label": "indirect_rejection",
        "judge_explanation": "Topical, harmless, and non-fulfilling.",
    }
    row.update(overrides)
    return row


class LatharmIndirectPairTests(unittest.TestCase):
    def test_model_fingerprint_changes_when_adapter_weights_change_in_place(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter_dir = Path(directory) / "checkpoint-100"
            adapter_dir.mkdir()
            (adapter_dir / "adapter_config.json").write_text(
                '{"base_model_name_or_path": "declared/base-model"}\n',
                encoding="utf-8",
            )
            weights_path = adapter_dir / "adapter_model.safetensors"
            weights_path.write_bytes(b"AAAA")
            original = MODULE.model_fingerprint(adapter_dir)

            # Keep the path and byte length fixed so content hashing is what
            # invalidates the fine-tune generation signature.
            weights_path.write_bytes(b"BBBB")
            changed = MODULE.model_fingerprint(adapter_dir)

        self.assertTrue(original["local"])
        self.assertEqual(original["reference"], changed["reference"])
        self.assertNotEqual(original, changed)
        original_weights = next(
            item
            for item in original["files"]
            if item["name"] == "adapter_model.safetensors"
        )
        changed_weights = next(
            item
            for item in changed["files"]
            if item["name"] == "adapter_model.safetensors"
        )
        self.assertNotEqual(original_weights["sha256"], changed_weights["sha256"])

    def test_health_example_keeps_exact_same_row_lineage(self):
        source_rows, stats = MODULE.select_source_rows(
            dataset=[
                {
                    "instruction": HEALTH_INSTRUCTION,
                    "input": "",
                    "output": HEALTH_REFERENCE,
                }
            ],
            instruction_field="instruction",
            input_field="input",
            output_field="output",
            start_index=0,
            num_source_samples=1,
        )

        self.assertEqual(stats["selected_for_generation"], 1)
        self.assertEqual(source_rows[0]["source_index"], 0)
        self.assertEqual(source_rows[0]["instruction"], HEALTH_INSTRUCTION)
        self.assertEqual(source_rows[0]["prompt"], HEALTH_INSTRUCTION)
        self.assertEqual(source_rows[0]["output"], HEALTH_REFERENCE)
        self.assertIn(HEALTH_INSTRUCTION, source_rows[0]["generation_prompt"])

        judged_row = {
            **source_rows[0],
            "response": HEALTH_INDIRECT_REJECTION,
            "direct_refusal": False,
            "indirect_rejection_label": "indirect_rejection",
            "judge_explanation": "Topical, harmless, and non-fulfilling.",
        }
        pairs, filter_stats = MODULE.build_pairs(
            judged_rows=[judged_row],
            target_count=0,
            minimum_target_chars=1,
            maximum_target_chars=1000,
        )

        self.assertEqual(filter_stats["selected_pairs"], 1)
        self.assertEqual(pairs[0]["index"], 0)
        self.assertEqual(pairs[0]["source_index"], 0)
        self.assertEqual(pairs[0]["instruction"], HEALTH_INSTRUCTION)
        self.assertEqual(pairs[0]["target_prompt"], HEALTH_INSTRUCTION)
        self.assertEqual(pairs[0]["target"], HEALTH_INDIRECT_REJECTION)
        self.assertEqual(pairs[0]["output"], HEALTH_REFERENCE)
        self.assertEqual(pairs[0]["alpaca_reference"], HEALTH_REFERENCE)

    def test_formats_nonempty_input_for_generation_and_search(self):
        source_rows, _ = MODULE.select_source_rows(
            dataset=[
                {
                    "instruction": "Summarize the passage.",
                    "input": "Plants convert light into stored chemical energy.",
                    "output": "Plants turn light into chemical energy.",
                }
            ],
            instruction_field="instruction",
            input_field="input",
            output_field="output",
            start_index=0,
            num_source_samples=1,
        )

        expected_prompt = (
            "Summarize the passage.\n\nInput:\n"
            "Plants convert light into stored chemical energy."
        )
        self.assertEqual(source_rows[0]["prompt"], expected_prompt)
        self.assertIn(expected_prompt, source_rows[0]["generation_prompt"])

    def test_expands_each_source_into_distinct_deterministic_style_attempts(self):
        source_rows, _ = MODULE.select_source_rows(
            dataset=[
                {
                    "instruction": HEALTH_INSTRUCTION,
                    "input": "",
                    "output": HEALTH_REFERENCE,
                }
            ],
            instruction_field="instruction",
            input_field="input",
            output_field="output",
            start_index=0,
            num_source_samples=1,
        )

        expanded = MODULE.expand_generation_rows(
            source_rows,
            generation_attempts_per_row=4,
        )

        self.assertEqual(len(expanded), 4)
        self.assertEqual(
            [row["target_candidate_index"] for row in expanded],
            [0, 1, 2, 3],
        )
        self.assertEqual(len({row["generation_prompt"] for row in expanded}), 4)
        self.assertTrue(all(HEALTH_INSTRUCTION in row["generation_prompt"] for row in expanded))

    def test_groups_multiple_distinct_targets_into_one_source_row(self):
        source_rows, _ = MODULE.select_source_rows(
            dataset=[
                {
                    "instruction": HEALTH_INSTRUCTION,
                    "input": "",
                    "output": HEALTH_REFERENCE,
                }
            ],
            instruction_field="instruction",
            input_field="input",
            output_field="output",
            start_index=0,
            num_source_samples=1,
        )
        judged_rows = [
            judged_candidate(source_rows[0], HEALTH_INDIRECT_REJECTION, 0),
            judged_candidate(source_rows[0], HEALTH_INDIRECT_REJECTION_2, 1),
            judged_candidate(source_rows[0], HEALTH_INDIRECT_REJECTION_3, 2),
        ]

        pairs, stats = MODULE.build_pairs(
            judged_rows=judged_rows,
            target_count=0,
            minimum_target_chars=1,
            maximum_target_chars=1000,
            targets_per_row=2,
        )

        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["target_count"], 2)
        self.assertEqual(
            json.loads(pairs[0]["targets_json"]),
            [HEALTH_INDIRECT_REJECTION, HEALTH_INDIRECT_REJECTION_2],
        )
        self.assertEqual(pairs[0]["target"], HEALTH_INDIRECT_REJECTION)
        self.assertEqual(
            json.loads(pairs[0]["target_judge_labels_json"]),
            ["indirect_rejection", "indirect_rejection"],
        )
        self.assertEqual(stats["selected_source_rows"], 1)
        self.assertEqual(stats["selected_targets"], 2)

    def test_deduplicates_targets_and_drops_sources_below_fixed_cardinality(self):
        source_rows, _ = MODULE.select_source_rows(
            dataset=[
                {
                    "instruction": HEALTH_INSTRUCTION,
                    "input": "",
                    "output": HEALTH_REFERENCE,
                }
            ],
            instruction_field="instruction",
            input_field="input",
            output_field="output",
            start_index=0,
            num_source_samples=1,
        )
        judged_rows = [
            judged_candidate(source_rows[0], HEALTH_INDIRECT_REJECTION, 0),
            judged_candidate(
                source_rows[0],
                f"  {HEALTH_INDIRECT_REJECTION.upper()}  ",
                1,
            ),
            judged_candidate(source_rows[0], HEALTH_INDIRECT_REJECTION_2, 2),
        ]

        pairs, stats = MODULE.build_pairs(
            judged_rows=judged_rows,
            target_count=0,
            minimum_target_chars=1,
            maximum_target_chars=1000,
            targets_per_row=3,
        )

        self.assertEqual(pairs, [])
        self.assertEqual(stats["duplicate_target"], 1)
        self.assertEqual(stats["insufficient_valid_targets"], 1)

    def test_target_count_caps_source_rows_not_individual_targets(self):
        source_rows, _ = MODULE.select_source_rows(
            dataset=[
                {
                    "instruction": HEALTH_INSTRUCTION,
                    "input": "",
                    "output": HEALTH_REFERENCE,
                },
                {
                    "instruction": "Summarize a short passage.",
                    "input": "",
                    "output": "A concise summary.",
                },
            ],
            instruction_field="instruction",
            input_field="input",
            output_field="output",
            start_index=0,
            num_source_samples=2,
        )
        judged_rows = []
        for source_row in source_rows:
            judged_rows.extend(
                [
                    judged_candidate(source_row, HEALTH_INDIRECT_REJECTION, 0),
                    judged_candidate(source_row, HEALTH_INDIRECT_REJECTION_2, 1),
                ]
            )

        pairs, stats = MODULE.build_pairs(
            judged_rows=judged_rows,
            target_count=1,
            minimum_target_chars=1,
            maximum_target_chars=1000,
            targets_per_row=2,
        )

        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["source_index"], 0)
        self.assertEqual(stats["selected_targets"], 2)

    def test_helpful_answer_is_rejected_without_indirect_judge_label(self):
        helpful_row = {
            "source_index": 0,
            "instruction": HEALTH_INSTRUCTION,
            "input": "",
            "output": HEALTH_REFERENCE,
            "prompt": HEALTH_INSTRUCTION,
            "response": (
                "Focus on nutritious food, regular exercise, and consistent sleep."
            ),
            "direct_refusal": False,
            "indirect_rejection_label": "fulfillment",
        }
        indirect_row = {
            **helpful_row,
            "source_index": 1,
            "response": HEALTH_INDIRECT_REJECTION,
            "indirect_rejection_label": "indirect_rejection",
        }

        pairs, stats = MODULE.build_pairs(
            judged_rows=[helpful_row, indirect_row],
            target_count=0,
            minimum_target_chars=1,
            maximum_target_chars=1000,
        )

        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["source_index"], 1)
        self.assertEqual(pairs[0]["target"], HEALTH_INDIRECT_REJECTION)
        self.assertEqual(stats["not_indirect_rejection"], 1)

    def test_direct_refusal_is_rejected_even_if_judge_label_is_indirect(self):
        row = {
            "source_index": 0,
            "instruction": HEALTH_INSTRUCTION,
            "input": "",
            "output": HEALTH_REFERENCE,
            "prompt": HEALTH_INSTRUCTION,
            "response": "I cannot provide those health tips.",
            "direct_refusal": True,
            "indirect_rejection_label": "indirect_rejection",
        }

        pairs, stats = MODULE.build_pairs(
            judged_rows=[row],
            target_count=0,
            minimum_target_chars=1,
            maximum_target_chars=1000,
        )

        self.assertEqual(pairs, [])
        self.assertEqual(stats["direct_refusal"], 1)


class IndirectRejectionLabelParserTests(unittest.TestCase):
    def test_parses_exact_judge_format(self):
        label, reason = safety_eval._parse_indirect_rejection_output(
            "#thelabel: indirect_rejection\n"
            "#thereason: It stays topical while leaving the task undone."
        )

        self.assertEqual(label, "indirect_rejection")
        self.assertEqual(
            reason,
            "It stays topical while leaving the task undone.",
        )

    def test_parses_fulfillment_and_rejects_unknown_label(self):
        fulfillment = safety_eval._parse_indirect_rejection_output(
            "#thelabel: fulfillment\n#thereason: It provides the requested tips."
        )
        unknown = safety_eval._parse_indirect_rejection_output(
            "#thelabel: maybe\n#thereason: Not part of the rubric."
        )

        self.assertEqual(
            fulfillment,
            ("fulfillment", "It provides the requested tips."),
        )
        self.assertEqual(unknown[0], None)


if __name__ == "__main__":
    unittest.main()
