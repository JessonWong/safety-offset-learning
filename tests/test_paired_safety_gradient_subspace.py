import importlib.util
import unittest
from pathlib import Path

import torch


import safeoff.analysis.safety_subspace as MODULE


class SafetyGradientSubspaceTest(unittest.TestCase):
    def test_projection_coordinates_match_known_basis(self):
        rows = torch.tensor(
            [[2.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32
        )
        subspace = MODULE.build_subspace(rows)
        vector = torch.tensor([3.0, 4.0, 5.0])
        coordinates = MODULE.projection_coordinates(subspace, vector, 2)
        self.assertAlmostEqual(float(coordinates.square().sum()), 25.0, places=5)
        summary = MODULE.projection_summary(subspace, vector, [1, 2])
        self.assertAlmostEqual(summary["2"]["captured_energy"], 25.0 / 50.0, places=5)

    def test_did_records_subtract_matched_controls(self):
        rows = []
        values = {
            "safeoff": (2.0, 1.8, 1.5),
            "original": (1.0, 0.9, 0.7),
            "llm_lat_rejected": (3.0, 2.5, 2.0),
            "llm_lat_chosen": (0.5, 0.4, 0.3),
        }
        for arm, triple in values.items():
            for step, scale in zip((1, 5, 10), (1.0, 2.0, 3.0)):
                rows.append(
                    {
                        "arm": arm,
                        "step": step,
                        "actual_safety_change": triple[0] * scale,
                        "full_taylor_prediction": triple[1] * scale,
                        "subspace_taylor_prediction": triple[2] * scale,
                    }
                )
        did = MODULE.build_did_records(rows, primary_k=8)
        self.assertEqual(len(did), 6)
        first = did[0]
        self.assertEqual(first["comparison"], "safeoff_minus_original")
        self.assertAlmostEqual(first["actual_did_safety_change"], 1.0)
        self.assertAlmostEqual(first["full_taylor_did_prediction"], 0.9)
        self.assertAlmostEqual(first["subspace_taylor_did_prediction"], 0.8)

    def test_llm_lat_indices_are_reproducible_and_unique(self):
        first = MODULE.llm_lat_indices(1000, seed=30, count=100)
        second = MODULE.llm_lat_indices(1000, seed=30, count=100)
        self.assertEqual(first, second)
        self.assertEqual(len(first), len(set(first)))


if __name__ == "__main__":
    unittest.main()
