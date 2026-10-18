"""CPU-only tests for selection integrity and cache validation."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from safeoff.evaluation.system_ablation import adapter_provenance, export_keyword_subset, read_source, validate_cached, write_json


class AblationDataTests(unittest.TestCase):
    @unittest.skipUnless((Path(__file__).resolve().parents[1] / "data/8b_newtarget.csv").is_file(),
                         "local research dataset is not included in the source release")
    def test_original_source_has_empty_suffix(self):
        fields, rows = read_source(Path(__file__).resolve().parents[1] / "data/8b_newtarget.csv")
        self.assertEqual(len(rows), 1000)
        self.assertIn("alpaca_reference", fields)
        self.assertTrue(all(not row["suffix"] for row in rows))

    def test_read_source_preserves_quoted_and_multiline_answers(self):
        fields = ["index", "instruction", "alpaca_reference", "suffix"]
        rows = [{"index": "7", "instruction": "Say hello, then goodbye.",
                 "alpaca_reference": "Hello!\nGoodbye!", "suffix": ""}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            loaded_fields, loaded_rows = read_source(path)
        self.assertEqual(loaded_fields, fields)
        self.assertEqual(loaded_rows, rows)

    def test_export_retains_original_answers_and_indices(self):
        fields = ["index", "instruction", "alpaca_reference", "suffix"]
        rows = [dict(zip(fields, values)) for values in [
            ("9", "First input", "Original answer, not refusal", ""),
            ("27", "Second input", "Another answer", ""),
        ]]
        records = [{"refusal": True}, {"refusal": False}]
        with tempfile.TemporaryDirectory(prefix="safeoff_ablation_test_") as directory:
            path = Path(directory) / "subset.csv"
            self.assertEqual(export_keyword_subset(path, fields, rows, records), 1)
            _, selected = read_source(path)
            self.assertEqual(selected, [rows[0]])

    def test_incomplete_selection_rejected(self):
        with self.assertRaises(ValueError):
            export_keyword_subset(None, [], [{"instruction": "input"}], [])

    def test_cache_provenance_and_order(self):
        meta = {"system": None, "source_sha256": "a"}
        rows = [{"instruction": "input"}]
        records = [{"row_position": 0, "prompt": "input"}]
        self.assertEqual(validate_cached({"meta": meta, "data": records}, meta, rows), records)
        with self.assertRaises(ValueError):
            validate_cached({"meta": {"system": ""}, "data": records}, meta, rows)
        with self.assertRaises(ValueError):
            validate_cached({"meta": meta, "data": records}, meta, [{"instruction": "different"}])

    def test_json_null_system_round_trip(self):
        with tempfile.TemporaryDirectory(prefix="safeoff_ablation_test_") as directory:
            path = Path(directory) / "result.json"
            write_json(path, {"system": None})
            self.assertIsNone(json.loads(path.read_text())["system"])

    @unittest.skipUnless(all((Path(__file__).resolve().parents[1] / "data" / f"{name}.csv").is_file()
                            for name in ("simple_rewrite", "overcautious", "subliminal")),
                         "local research datasets are not included in the source release")
    def test_extra_baseline_datasets_have_all_1000_prompts(self):
        root = Path(__file__).resolve().parents[1]
        for name in ("simple_rewrite", "overcautious", "subliminal"):
            with self.subTest(dataset=name):
                _, rows = read_source(root / "data" / f"{name}.csv")
                self.assertEqual(len(rows), 1000)
                self.assertEqual({int(row["index"]) for row in rows}, set(range(1000)))

    def test_no_adapter_retains_base_metadata(self):
        self.assertEqual(adapter_provenance(None, "base"), {})

    def test_adapter_missing_files_rejected(self):
        with tempfile.TemporaryDirectory(prefix="refusal_adapter_test_") as directory:
            with self.assertRaises(FileNotFoundError):
                adapter_provenance(Path(directory), "base")

    def test_adapter_wrong_base_rejected(self):
        with tempfile.TemporaryDirectory(prefix="refusal_adapter_test_") as directory:
            path = Path(directory)
            (path / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "other"}))
            (path / "adapter_model.safetensors").write_bytes(b"dummy weights for hashing only")
            with self.assertRaisesRegex(ValueError, "different base model"):
                adapter_provenance(path, "base")

    def test_adapter_hash_invalidates_cache_if_weights_change(self):
        with tempfile.TemporaryDirectory(prefix="refusal_adapter_test_") as directory:
            path = Path(directory)
            (path / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "base"}))
            (path / "adapter_model.safetensors").write_bytes(b"first weights for hashing only")
            original = adapter_provenance(path, "base")
            self.assertEqual(original["adapter_path"], str(path.resolve()))
            self.assertEqual(len(original["adapter_weights_sha256"]), 64)
            (path / "adapter_model.safetensors").write_bytes(b"changed weights for hashing only")
            updated = adapter_provenance(path, "base")
            self.assertNotEqual(original["adapter_weights_sha256"], updated["adapter_weights_sha256"])
            with self.assertRaisesRegex(ValueError, "provenance"):
                validate_cached({"meta": original, "data": []}, updated, [])


if __name__ == "__main__":
    unittest.main()
