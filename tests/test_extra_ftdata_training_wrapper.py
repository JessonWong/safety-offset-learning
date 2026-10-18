"""CPU-only checks of the actual Slurm wrapper's preflight and overwrite guards."""

import ast
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


WRAPPER = Path(__file__).resolve().parents[1] / "experiments/slurm/extra_baselines.sh"
BLOCKS = [part.split("\nPY\n")[0] for part in WRAPPER.read_text().split("<<'PY'\n")[1:]]


class TrainingPreflightTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="extra_ftdata_training_test_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "source.csv"
        self.output = self.root / "new_training"
        self.result = self.root / "results"
        self.raw = self.result / "raw/simple_rewrite/before_base/helpful_system.json"
        self.raw.parent.mkdir(parents=True)
        self.rows = [{"index": str(i), "instruction": f"Prompt {i}", "alpaca_reference": f"Answer {i}", "suffix": ""}
                     for i in range(1000)]
        with self.source.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)
        self.payload = {"meta": {
            "source_sha256": hashlib.sha256(self.source.read_bytes()).hexdigest(),
            "model": "NousResearch/Meta-Llama-3.1-8B-Instruct", "dtype": "bfloat16",
            "batch_size": 16, "max_new_tokens": 200, "do_sample": False,
            "max_input_tokens": 1000, "system": "You are a helpful assistant.",
            "model_commit": "a" * 40, "chat_template_sha256": "b" * 64,
        }, "data": [{"row_position": i, "source_index": row["index"], "prompt": row["instruction"],
                     "reference_response": row["alpaca_reference"], "response": "Answer", "refusal": False}
                    for i, row in enumerate(self.rows)]}
        for name in ("src/safeoff/training/sft.py", "src/safeoff/data/config.py", "src/safeoff/core/model_utils.py", "src/safeoff/data/loaders.py",
                     "src/safeoff/evaluation/system_ablation.py", "experiments/slurm/extra_baselines.sh"):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("test source-code fingerprint fixture\n")

    def run_preflight(self):
        self.raw.write_text(json.dumps(self.payload))
        return subprocess.run([sys.executable, "-c", BLOCKS[0], str(self.source), str(self.output),
                               str(self.result), "simple_rewrite", "--max_steps", "250"],
                              cwd=self.root, capture_output=True, text=True)

    def test_valid_manifest_and_argument_offsets(self):
        result = self.run_preflight()
        self.assertEqual(result.returncode, 0, result.stderr)
        manifest = json.loads((self.result / "training/simple_rewrite/protocol.json").read_text())
        self.assertEqual(manifest["training_cli"], ["--max_steps", "250"])
        self.assertEqual(manifest["num_training_rows"], 1000)
        self.assertEqual(manifest["status"], "prepared")
        self.assertEqual(len(manifest["source_code_sha256"]), 6)
        self.assertFalse(self.output.exists(), "Preflight must not launch training")

    def test_nonempty_model_directory_is_preserved(self):
        self.output.mkdir()
        sentinel = self.output / "user_checkpoint"
        sentinel.write_text("preserve me")
        result = self.run_preflight()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Refusing to overwrite", result.stderr)
        self.assertEqual(sentinel.read_text(), "preserve me")

    def test_existing_manifest_requires_inspection(self):
        self.assertEqual(self.run_preflight().returncode, 0)
        result = self.run_preflight()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Existing training manifest", result.stderr)

    def test_wrong_source_hash_rejected(self):
        self.payload["meta"]["source_sha256"] = "different"
        result = self.run_preflight()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must match", result.stderr)

    def test_incomplete_baseline_rejected(self):
        self.payload["data"].pop()
        self.assertNotEqual(self.run_preflight().returncode, 0)

    def test_wrong_prompt_pair_rejected(self):
        self.payload["data"][0]["prompt"] = "Different prompt"
        result = self.run_preflight()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("row mismatch", result.stderr)

    def test_system_mismatch_rejected(self):
        self.payload["meta"]["system"] = None
        self.assertNotEqual(self.run_preflight().returncode, 0)

    def test_all_embedded_python_parses(self):
        self.assertEqual(len(BLOCKS), 3)
        for block in BLOCKS:
            ast.parse(block)


if __name__ == "__main__":
    unittest.main()
