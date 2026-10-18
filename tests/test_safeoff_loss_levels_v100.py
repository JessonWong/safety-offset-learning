import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import safeoff.experiments.loss_levels_v100 as module


class V100ExecutionTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.run = Path(folder.name)
        self.args = SimpleNamespace(run=self.run, round=0, workers=6)
        self.protocol = {"training_precision": "float16", "gradient_checkpointing": True,
                         "eval_batch_size": 4}
        module.write_json(self.run / "protocol.json", self.protocol)

    def test_requires_explicit_v100_runtime_and_32gb_card(self):
        options = module.v100_training_options(self.protocol)
        self.assertTrue(options["fp16"])
        self.assertFalse(options["bf16"])
        with self.assertRaises(ValueError):
            module.v100_training_options({})
        module.check_v100_device("Tesla V100-SXM2-32GB", 32 * 1024**3)
        for name, memory in [("Tesla V100-SXM2-16GB", 16 * 1024**3), ("H100", 80 * 1024**3)]:
            with self.assertRaises(ValueError):
                module.check_v100_device(name, memory)

    def test_failed_probe_never_starts_full_training(self):
        module.write_json(self.run / "probe/status.json", {"status": "failed", "error": "out of memory"})
        with patch.object(module, "run_serial_stage", side_effect=SystemExit(1)) as stage:
            with self.assertRaises(SystemExit):
                module.v100_suite(self.args)
        self.assertEqual(stage.call_count, 1)
        status = json.loads((self.run / "status.json").read_text())
        self.assertEqual(status["stage"], "v100_probe")
        self.assertEqual(status["details"]["error"], "out of memory")

    def test_verified_probe_gates_all_seven_arms_in_one_worker(self):
        module.write_json(self.run / "probe/models/level_3/verified.json",
                          {"global_step": 5, "changed_tensors": 2})
        module.write_json(self.run / "probe/generations/level_3.json", {"data": [{}] * 4})
        module.write_json(self.run / "probe/runtime/level_3.json", {"precision": "float16"})
        with patch.object(module, "run_serial_stage") as stage:
            module.v100_suite(self.args)
        self.assertEqual(stage.call_count, 9)
        self.assertEqual(stage.call_args_list[0].args[0].run, self.run / "probe")
        self.assertEqual([call.args[3] for call in stage.call_args_list[1:8]], module.ARMS)
        self.assertEqual(stage.call_args_list[-1].args[1], "score_plot")
        self.assertTrue((self.run / "v100_probe_passed.json").exists())


if __name__ == "__main__":
    unittest.main()
