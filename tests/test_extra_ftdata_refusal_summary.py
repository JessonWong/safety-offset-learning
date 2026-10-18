#!/usr/bin/env python3
"""CPU-only tests for the extra-baseline paired refusal summarizer."""

from __future__ import annotations

import copy
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import safeoff.analysis.dataset_refusal_summary as summary


class RefusalSummaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data_root = self.root / "data"
        self.result_root = self.root / "results"
        self.data_root.mkdir()
        self.rows = [
            {"index": str(index), "instruction": f"Prompt {index}", "alpaca_reference": f"Answer {index}", "suffix": ""}
            for index in range(3)
        ]
        self.before_paths = {}
        self.after_paths = {}
        for name, _ in summary.DATASETS:
            source = self.data_root / f"{name}.csv"
            with source.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(self.rows[0]))
                writer.writeheader()
                writer.writerows(self.rows)
            adapter = self.root / "adapters" / name / "checkpoint-250"
            adapter.mkdir(parents=True)
            config = adapter / "adapter_config.json"
            config.write_text(json.dumps({"base_model_name_or_path": summary.BASE_MODEL}), encoding="utf-8")
            weights = adapter / "adapter_model.safetensors"
            weights.write_bytes(b"test fixture weights; no model loading")
            meta = {
                "source": str(source.resolve()), "source_sha256": summary.sha256(source),
                "model": summary.BASE_MODEL, "dtype": "bfloat16", "batch_size": 16,
                "max_new_tokens": 200, "do_sample": False, "max_input_tokens": 1000,
                "num_prompts": 3, "refusal_detector": "safeoff.evaluation.metrics.contains_keyword",
                "chat_template_sha256": "a" * 64, "model_commit": "b" * 40,
                "condition": "helpful_system", "system": summary.HELPFUL_SYSTEM,
            }
            data = [
                {
                    "row_position": index, "source_index": row["index"],
                    "prompt": row["instruction"], "reference_response": row["alpaca_reference"],
                    "system": summary.HELPFUL_SYSTEM, "response": "A response", "refusal": index == 0,
                }
                for index, row in enumerate(self.rows)
            ]
            before = self.result_root / "raw" / name / "before_base" / "helpful_system.json"
            after = self.result_root / "raw" / name / "after_checkpoint_250" / "helpful_system.json"
            self.before_paths[name] = before
            self.after_paths[name] = after
            self.save(before, {"meta": meta, "data": data})
            after_meta = {
                **meta, "adapter_path": str(adapter),
                "adapter_config_sha256": summary.sha256(config),
                "adapter_weights_sha256": summary.sha256(weights),
            }
            after_data = copy.deepcopy(data)
            for record in after_data:
                record["refusal"] = False
            self.save(after, {"meta": after_meta, "data": after_data})

    @staticmethod
    def save(path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def mutate(self, path, change):
        payload = json.loads(path.read_text(encoding="utf-8"))
        change(payload)
        self.save(path, payload)

    def build(self):
        return summary.build_summary(self.result_root, self.data_root, expected_rows=3)

    def test_complete_counts_and_true_zero(self):
        result = self.build()
        self.assertTrue(result["complete"])
        self.assertIn("safeoff.evaluation.metrics.contains_keyword", result["refusal_definition"])
        for item in result["datasets"].values():
            self.assertEqual(item["before"]["refusal_count"], 1)
            self.assertEqual(item["before"]["refusal_rate"], 1 / 3)
            self.assertEqual(item["after"]["refusal_count"], 0)
            self.assertEqual(item["after"]["refusal_rate"], 0)

    def test_missing_after_is_null_not_zero(self):
        self.after_paths["simple_rewrite"].unlink()
        result = self.build()
        phase = result["datasets"]["simple_rewrite"]["after"]
        self.assertFalse(result["complete"])
        self.assertEqual(phase["status"], "missing")
        self.assertIsNone(phase["refusal_count"])
        self.assertIsNone(phase["refusal_rate"])
        self.assertIn(r"Simple Rewrite & 33.3\% & --", summary.latex_text(result))
        csv_rows = list(csv.DictReader(io.StringIO(summary.csv_text(result))))
        self.assertEqual(csv_rows[0]["after_refusal_rate"], "")

    def test_partial_after_exposes_progress_but_not_final_rate(self):
        self.mutate(self.after_paths["overcautious"], lambda payload: payload.update(data=payload["data"][:2]))
        result = self.build()
        phase = result["datasets"]["overcautious"]["after"]
        self.assertEqual(phase["status"], "partial")
        self.assertEqual(phase["num_generated"], 2)
        self.assertEqual(phase["observed_refusal_count"], 0)
        self.assertIsNone(phase["refusal_rate"])
        self.assertFalse(result["complete"])

    def test_missing_before_is_also_null(self):
        self.before_paths["subliminal"].unlink()
        result = self.build()
        self.assertIsNone(result["datasets"]["subliminal"]["before"]["refusal_rate"])
        self.assertFalse(result["complete"])

    def test_source_hash_tampering_rejected(self):
        self.mutate(self.before_paths["simple_rewrite"], lambda payload: payload["meta"].update(source_sha256="0" * 64))
        with self.assertRaisesRegex(ValueError, "source_sha256"):
            self.build()

    def test_row_identity_and_types_rejected(self):
        path = self.before_paths["simple_rewrite"]
        original = json.loads(path.read_text(encoding="utf-8"))
        for field, value in (
            ("row_position", True), ("source_index", "wrong"), ("prompt", "wrong"),
            ("reference_response", "wrong"), ("system", "wrong"),
            ("response", None), ("refusal", 1), ("refusal", "false"),
        ):
            with self.subTest(field=field, value=value):
                modified = copy.deepcopy(original)
                modified["data"][0][field] = value
                self.save(path, modified)
                with self.assertRaises(ValueError):
                    self.build()
        self.save(path, original)

    def test_extra_generation_rejected(self):
        self.mutate(self.before_paths["simple_rewrite"], lambda payload: payload["data"].append(payload["data"][-1]))
        with self.assertRaisesRegex(ValueError, "generation count"):
            self.build()

    def test_before_with_adapter_rejected(self):
        self.mutate(self.before_paths["simple_rewrite"], lambda payload: payload["meta"].update(adapter_path=None))
        with self.assertRaisesRegex(ValueError, "without adapter"):
            self.build()

    def test_after_without_adapter_rejected(self):
        self.mutate(self.after_paths["simple_rewrite"], lambda payload: payload["meta"].pop("adapter_weights_sha256"))
        with self.assertRaisesRegex(ValueError, "lacks adapter"):
            self.build()

    def test_actual_adapter_weights_tampering_rejected(self):
        weights = self.root / "adapters/simple_rewrite/checkpoint-250/adapter_model.safetensors"
        weights.write_bytes(b"changed after evaluation")
        with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
            self.build()

    def test_adapter_wrong_base_rejected_even_if_hash_matches(self):
        config = self.root / "adapters/simple_rewrite/checkpoint-250/adapter_config.json"
        config.write_text(json.dumps({"base_model_name_or_path": "different-model"}), encoding="utf-8")
        self.mutate(self.after_paths["simple_rewrite"], lambda payload: payload["meta"].update(adapter_config_sha256=summary.sha256(config)))
        with self.assertRaisesRegex(ValueError, "base model differs"):
            self.build()

    def test_protocol_drift_rejected(self):
        path = self.after_paths["simple_rewrite"]
        original = json.loads(path.read_text(encoding="utf-8"))
        for field, value in (
            ("batch_size", 8), ("model_commit", "c" * 40), ("chat_template_sha256", "d" * 64),
            ("dtype", "float16"), ("max_new_tokens", 300), ("do_sample", True),
            ("system", None), ("model", "different"),
        ):
            with self.subTest(field=field):
                modified = copy.deepcopy(original)
                modified["meta"][field] = value
                self.save(path, modified)
                with self.assertRaises(ValueError):
                    self.build()
        self.save(path, original)

    def test_production_requires_1000_source_rows(self):
        with self.assertRaisesRegex(ValueError, "Expected 1000 source rows"):
            summary.build_summary(self.result_root, self.data_root)

    def test_locked_atomic_outputs_and_three_latex_columns(self):
        with patch.object(summary.fcntl, "flock", wraps=summary.fcntl.flock) as locked:
            result = summary.write_summary(self.result_root, self.data_root, expected_rows=3)
        locked.assert_called_once()
        self.assertEqual(locked.call_args.args[1], summary.fcntl.LOCK_EX)
        self.assertEqual(json.loads((self.result_root / "summary.json").read_text()), result)
        self.assertTrue((self.result_root / "summary.csv").is_file())
        latex = (self.result_root / "table.tex").read_text()
        self.assertIn(r"\begin{tabular}{lcc}", latex)
        self.assertIn("own 1,000 fine-tuning prompts", latex)
        self.assertIn("step 250", latex)
        self.assertIn("keyword-based proxy", latex)
        self.assertIn("200 generated tokens", latex)
        self.assertEqual(list(self.result_root.glob(".summary.json.*")), [])
        self.assertEqual(list(self.result_root.glob(".table.tex.*")), [])

    def test_atomic_failure_keeps_prior_output(self):
        target = self.root / "prior.json"
        target.write_text("original", encoding="utf-8")
        with patch.object(summary.os, "replace", side_effect=OSError("simulated replacement error")):
            with self.assertRaisesRegex(OSError, "simulated"):
                summary.atomic_write(target, "new content")
        self.assertEqual(target.read_text(), "original")
        self.assertEqual(list(self.root.glob(".prior.json.*")), [])


if __name__ == "__main__":
    unittest.main()
