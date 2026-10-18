import importlib.util
import json
import unittest
from pathlib import Path


import safeoff.data.pair_benign_data_with_indirect_targets as MODULE


HEALTH_INSTRUCTION = "Give three tips for staying healthy."
HEALTH_REFERENCE = (
    "1. Eat a balanced diet. 2. Exercise regularly. 3. Get enough sleep."
)
HEALTH_INDIRECT_REJECTION = (
    "Health needs vary from person to person, so individualized guidance from a "
    "qualified clinician is a more appropriate next step."
)
HEALTH_INDIRECT_REJECTION_2 = (
    "A reliable recommendation depends on personal context that should first be "
    "reviewed with a qualified health professional."
)


def health_target(**overrides):
    row = {
        "pair_id": "alpaca-indirect-0-example",
        # Target rows enter this helper through csv.DictReader in production.
        "source_index": "0",
        "instruction": HEALTH_INSTRUCTION,
        "input": "",
        "output": HEALTH_REFERENCE,
        "target_prompt": HEALTH_INSTRUCTION,
        "target": HEALTH_INDIRECT_REJECTION,
        "indirect_rejection": "True",
        "direct_refusal": "False",
        "indirect_rejection_label": "indirect_rejection",
    }
    row.update(overrides)
    return row


def multi_health_target(targets=None, **overrides):
    targets = targets or [HEALTH_INDIRECT_REJECTION, HEALTH_INDIRECT_REJECTION_2]
    row = health_target(
        target=targets[0],
        targets_json=json.dumps(targets),
        target_count=str(len(targets)),
        target_judge_labels_json=json.dumps(
            ["indirect_rejection"] * len(targets)
        ),
    )
    row.update(overrides)
    return row


class PairBenignTargetsTests(unittest.TestCase):
    def test_health_example_preserves_same_row_original_supervision(self):
        benign = [
            {
                "instruction": HEALTH_INSTRUCTION,
                "input": "",
                "output": HEALTH_REFERENCE,
            }
        ]

        rows = MODULE.combine_rows(benign, [health_target()])

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["index"], 0)
        self.assertEqual(rows[0]["source_index"], 0)
        self.assertEqual(rows[0]["instruction"], HEALTH_INSTRUCTION)
        self.assertEqual(rows[0]["target_prompt"], HEALTH_INSTRUCTION)
        self.assertEqual(rows[0]["target"], HEALTH_INDIRECT_REJECTION)
        self.assertEqual(rows[0]["target_response"], HEALTH_INDIRECT_REJECTION)
        self.assertEqual(rows[0]["output"], HEALTH_REFERENCE)
        self.assertEqual(rows[0]["alpaca_reference"], HEALTH_REFERENCE)
        self.assertNotEqual(rows[0]["target"], rows[0]["alpaca_reference"])

    def test_formats_nonempty_input_identically_to_target_prompt(self):
        instruction = "Summarize the passage."
        input_text = "Plants convert light into stored chemical energy."
        original = "Plants turn light into chemical energy."
        expected_prompt = f"{instruction}\n\nInput:\n{input_text}"
        benign = [
            {
                "instruction": instruction,
                "input": input_text,
                "output": original,
            }
        ]
        target = health_target(
            instruction=instruction,
            input=input_text,
            output=original,
            target_prompt=expected_prompt,
            target=(
                "The passage would benefit from subject-specific context before "
                "being reduced to a single summary."
            ),
        )

        rows = MODULE.combine_rows(benign, [target])

        self.assertEqual(rows[0]["input"], input_text)
        self.assertEqual(rows[0]["target_prompt"], expected_prompt)
        self.assertEqual(rows[0]["output"], original)
        self.assertEqual(rows[0]["alpaca_reference"], original)

    def test_preserves_canonical_multi_target_list_and_legacy_first_target(self):
        benign = [
            {
                "instruction": HEALTH_INSTRUCTION,
                "input": "",
                "output": HEALTH_REFERENCE,
            }
        ]

        rows = MODULE.combine_rows(
            benign,
            [multi_health_target()],
            expected_target_count=2,
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["target_count"], 2)
        self.assertEqual(rows[0]["target"], HEALTH_INDIRECT_REJECTION)
        self.assertEqual(rows[0]["target_response"], HEALTH_INDIRECT_REJECTION)
        self.assertEqual(
            json.loads(rows[0]["targets_json"]),
            [HEALTH_INDIRECT_REJECTION, HEALTH_INDIRECT_REJECTION_2],
        )
        self.assertEqual(rows[0]["targets_json"], rows[0]["target_responses_json"])

    def test_rejects_invalid_or_duplicate_multi_target_json(self):
        benign = [
            {
                "instruction": HEALTH_INSTRUCTION,
                "input": "",
                "output": HEALTH_REFERENCE,
            }
        ]

        with self.assertRaisesRegex(ValueError, "invalid JSON"):
            MODULE.combine_rows(
                benign,
                [multi_health_target(targets_json="not-json")],
            )
        with self.assertRaisesRegex(ValueError, "duplicate entries"):
            MODULE.combine_rows(
                benign,
                [
                    multi_health_target(
                        targets=[
                            HEALTH_INDIRECT_REJECTION,
                            f"  {HEALTH_INDIRECT_REJECTION.upper()}  ",
                        ]
                    )
                ],
            )
        with self.assertRaisesRegex(ValueError, "does not equal the first"):
            MODULE.combine_rows(
                benign,
                [multi_health_target(target="A stale scalar target.")],
            )

    def test_rejects_declared_or_expected_target_count_mismatch(self):
        benign = [
            {
                "instruction": HEALTH_INSTRUCTION,
                "input": "",
                "output": HEALTH_REFERENCE,
            }
        ]

        with self.assertRaisesRegex(ValueError, "declares target_count"):
            MODULE.combine_rows(
                benign,
                [multi_health_target(target_count="3")],
            )
        with self.assertRaisesRegex(ValueError, "expected 3"):
            MODULE.combine_rows(
                benign,
                [multi_health_target()],
                expected_target_count=3,
            )

    def test_rejects_variable_target_cardinality_across_source_rows(self):
        benign = [
            {
                "instruction": HEALTH_INSTRUCTION,
                "input": "",
                "output": HEALTH_REFERENCE,
            },
            {
                "instruction": "Summarize a passage.",
                "input": "",
                "output": "A concise summary.",
            },
        ]
        second_target = multi_health_target(
            source_index="1",
            instruction="Summarize a passage.",
            output="A concise summary.",
            target_prompt="Summarize a passage.",
            targets=[HEALTH_INDIRECT_REJECTION],
        )

        with self.assertRaisesRegex(ValueError, "fixed cardinality"):
            MODULE.combine_rows(
                benign,
                [multi_health_target(), second_target],
            )

    def test_rejects_if_any_indirect_target_equals_original_answer(self):
        benign = [
            {
                "instruction": HEALTH_INSTRUCTION,
                "input": "",
                "output": HEALTH_REFERENCE,
            }
        ]

        with self.assertRaisesRegex(ValueError, "target 1 equals"):
            MODULE.combine_rows(
                benign,
                [
                    multi_health_target(
                        targets=[HEALTH_INDIRECT_REJECTION, HEALTH_REFERENCE]
                    )
                ],
            )

    def test_requires_per_target_indirect_labels_for_multi_target_rows(self):
        benign = [
            {
                "instruction": HEALTH_INSTRUCTION,
                "input": "",
                "output": HEALTH_REFERENCE,
            }
        ]

        with self.assertRaisesRegex(ValueError, "missing 'target_judge_labels_json'"):
            MODULE.combine_rows(
                benign,
                [multi_health_target(target_judge_labels_json="")],
            )
        with self.assertRaisesRegex(ValueError, "Not every target"):
            MODULE.combine_rows(
                benign,
                [
                    multi_health_target(
                        target_judge_labels_json=json.dumps(
                            ["indirect_rejection", "fulfillment"]
                        )
                    )
                ],
            )

    def test_rejects_unrelated_target_instruction(self):
        benign = [
            {
                "instruction": HEALTH_INSTRUCTION,
                "input": "",
                "output": HEALTH_REFERENCE,
            }
        ]
        unrelated = health_target(
            instruction="Develop a guide for intercepting email.",
            target_prompt="Develop a guide for intercepting email.",
        )

        with self.assertRaisesRegex(ValueError, "instruction differs"):
            MODULE.combine_rows(benign, [unrelated])

    def test_rejects_changed_original_response(self):
        benign = [
            {
                "instruction": HEALTH_INSTRUCTION,
                "input": "",
                "output": HEALTH_REFERENCE,
            }
        ]

        with self.assertRaisesRegex(ValueError, "original response differs"):
            MODULE.combine_rows(
                benign,
                [health_target(output="An answer from a different Alpaca row.")],
            )

    def test_rejects_target_without_indirect_judge_verdict(self):
        benign = [
            {
                "instruction": HEALTH_INSTRUCTION,
                "input": "",
                "output": HEALTH_REFERENCE,
            }
        ]

        with self.assertRaisesRegex(ValueError, "not judged"):
            MODULE.combine_rows(
                benign,
                [health_target(indirect_rejection_label="fulfillment")],
            )


if __name__ == "__main__":
    unittest.main()
