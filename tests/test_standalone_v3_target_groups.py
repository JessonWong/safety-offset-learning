from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch



from safeoff.generation.search.optimizer import aggregate_target_losses  # noqa: E402
from safeoff.generation.replacement import get_example_targets  # noqa: E402


class TargetLossAggregationTests(unittest.TestCase):
    def test_single_target_is_identity(self):
        losses = torch.tensor([[0.25, 1.5]])
        for aggregation in ("mean", "softmin"):
            actual = aggregate_target_losses(
                losses,
                aggregation=aggregation,
                temperature=0.2,
            )
            self.assertTrue(torch.allclose(actual, losses[0]))

    def test_normalized_softmin_matches_formula(self):
        losses = torch.tensor([[0.2], [0.8], [1.1]])
        temperature = 0.3
        expected = -temperature * (
            torch.logsumexp(-losses / temperature, dim=0)
            - torch.log(torch.tensor(float(losses.shape[0])))
        )
        actual = aggregate_target_losses(
            losses,
            aggregation="softmin",
            temperature=temperature,
        )
        self.assertTrue(torch.allclose(actual, expected))
        self.assertGreaterEqual(actual.item(), losses.min().item())

    def test_small_temperature_approaches_minimum(self):
        losses = torch.tensor([[0.2], [0.8], [1.1]])
        actual = aggregate_target_losses(
            losses,
            aggregation="softmin",
            temperature=1e-4,
        )
        self.assertAlmostEqual(actual.item(), 0.2, places=3)

    def test_invalid_options_fail_closed(self):
        with self.assertRaises(ValueError):
            aggregate_target_losses(torch.ones(2, 1), aggregation="unknown")
        with self.assertRaises(ValueError):
            aggregate_target_losses(
                torch.ones(2, 1), aggregation="softmin", temperature=0
            )


class PerExampleTargetParsingTests(unittest.TestCase):
    def args(self):
        return SimpleNamespace(
            target_list_field="targets_json",
            target_field=None,
            refusal_target="fallback",
        )

    def test_parses_canonical_json_list(self):
        targets = ["A cautious boundary.", "A scoped redirection."]
        actual = get_example_targets(
            {"targets_json": json.dumps(targets)},
            self.args(),
        )
        self.assertEqual(actual, targets)

    def test_rejects_duplicates_and_invalid_json(self):
        with self.assertRaises(ValueError):
            get_example_targets(
                {"targets_json": json.dumps(["same", " same "])},
                self.args(),
            )
        with self.assertRaises(ValueError):
            get_example_targets({"targets_json": "not-json"}, self.args())


if __name__ == "__main__":
    unittest.main()
