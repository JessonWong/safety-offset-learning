"""Numerical and masking checks for final-checkpoint effective v rollback."""
import unittest
import copy

import torch
from torch import nn

from safeoff.analysis.rollback import DeltaHookManager, build_directions, difference, effective_rank, encode_response, frobenius, matched_norm, prepare_evaluations, summarize_teacher, teacher_batch, generation_length_metadata


class TinyProjectionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(5, 4, bias=False, dtype=torch.float64)
        self.v_proj = nn.Linear(5, 4, bias=False, dtype=torch.float64)

    def forward(self, inputs):
        return self.q_proj(inputs) + self.v_proj(inputs)


class TinyTokenizer:
    """A prefix-preserving chat template, including an assistant end token."""
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = "".join(f"<{message['role']}>{message['content']}" for message in messages)
        return text + ("<assistant>" if add_generation_prompt else "<end>")

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(character) for character in text]}


class EarlyVprojRollbackTest(unittest.TestCase):
    def setUp(self):
        self.rng = torch.Generator().manual_seed(321)

    def pair(self, rank=2, inputs=5, outputs=4):
        return (torch.randn(rank, inputs, generator=self.rng, dtype=torch.float64),
                torch.randn(outputs, rank, generator=self.rng, dtype=torch.float64))

    def test_late_direction_is_effective_matrix_difference(self):
        final, early = self.pair(3), self.pair(2)
        a, b = difference(final, early)
        torch.testing.assert_close(b @ a, final[1] @ final[0] - early[1] @ early[0], rtol=1e-12, atol=1e-12)

    def test_hook_equals_dense_final_minus_early_with_q_untouched(self):
        model = TinyProjectionModel()
        q, v, early = self.pair(), self.pair(), self.pair()
        inputs = torch.randn(2, 3, 5, generator=self.rng, dtype=torch.float64)
        original = {key: value.clone() for key, value in model.state_dict().items()}
        hooks = DeltaHookManager(model, ["q_proj", "v_proj"])
        try:
            for strength in (0.0, 0.25, 0.5, 1.0):
                hooks.set_state({"q_proj": q, "v_proj": v}, {"v_proj": early}, strength)
                expected_weight = original["q_proj.weight"] + q[1] @ q[0]
                expected_weight += original["v_proj.weight"] + v[1] @ v[0] - strength * early[1] @ early[0]
                torch.testing.assert_close(model(inputs), inputs @ expected_weight.T, rtol=1e-12, atol=1e-12)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, original[key], rtol=0, atol=0)
        finally:
            hooks.close()

    def test_lambda_zero_and_reset_restore_exact_baseline(self):
        model = TinyProjectionModel()
        inputs = torch.randn(3, 5, generator=self.rng, dtype=torch.float64)
        base = model(inputs).clone()
        final = {"q_proj": self.pair(), "v_proj": self.pair()}
        hooks = DeltaHookManager(model, final)
        hooks.set_state(final)
        baseline = model(inputs).clone()
        hooks.set_state(final, {"v_proj": self.pair()}, 0.0)
        torch.testing.assert_close(model(inputs), baseline, rtol=0, atol=0)
        hooks.set_state(final, {"v_proj": self.pair()}, 1.0)
        self.assertFalse(torch.equal(model(inputs), baseline))
        hooks.set_state(final)
        torch.testing.assert_close(model(inputs), baseline, rtol=0, atol=0)
        hooks.set_state()
        torch.testing.assert_close(model(inputs), base, rtol=0, atol=0)
        hooks.close()
        torch.testing.assert_close(model(inputs), base, rtol=0, atol=0)

    def test_bfloat16_baseline_matches_peft_eval_linear(self):
        from peft.tuners.lora.layer import Linear
        base = nn.Linear(5, 4, bias=False, dtype=torch.bfloat16)
        reference = Linear(copy.deepcopy(base), adapter_name="default", r=2,
                           lora_alpha=8, lora_dropout=0.5)
        a, b = self.pair()
        reference.lora_A["default"].to(dtype=torch.float32)
        reference.lora_B["default"].to(dtype=torch.float32)
        with torch.no_grad():
            reference.lora_A["default"].weight.copy_(a.float())
            reference.lora_B["default"].weight.copy_(b.float())
        reference.eval()
        model = nn.Sequential(base).eval()
        inputs = torch.randn(2, 3, 5, generator=self.rng).to(torch.bfloat16)
        hooks = DeltaHookManager(model, ["0"])
        try:
            hooks.set_state({"0": (a.float(), b.float() * 4)})
            torch.testing.assert_close(model(inputs), reference(inputs), rtol=0, atol=0)
        finally:
            hooks.close()

    def test_all_controls_norm_match_and_random_rank_match(self):
        final, early, clean, cross = ({"layer.v_proj": self.pair()} for _ in range(4))
        directions, metadata = build_directions(final, early, clean, cross, seed=777)
        reference = frobenius(early["layer.v_proj"])
        early_matrix = early["layer.v_proj"][1] @ early["layer.v_proj"][0]
        for condition, modules in directions.items():
            pair = modules["layer.v_proj"]
            self.assertAlmostEqual(frobenius(pair), reference, places=10)
            self.assertAlmostEqual(metadata[condition]["layer.v_proj"]["matched_norm"], reference, places=10)
            if condition.startswith("random"):
                self.assertEqual(effective_rank(pair), effective_rank(early["layer.v_proj"]))
        torch.testing.assert_close(directions["own_early"]["layer.v_proj"][1] @ directions["own_early"]["layer.v_proj"][0], early_matrix, rtol=0, atol=0)
        repeated, _ = build_directions(final, early, clean, cross, seed=777)
        for condition in directions:
            for actual, expected in zip(directions[condition]["layer.v_proj"], repeated[condition]["layer.v_proj"]):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_zero_control_rejected(self):
        candidate = self.pair()
        with self.assertRaises(ValueError):
            matched_norm((candidate[0], candidate[1] * 0), self.pair())

    def test_right_padding_and_response_mask(self):
        examples = [{"input_ids": [4, 5, 6, 7], "prompt_length": 2},
                    {"input_ids": [2, 3], "prompt_length": 1}]
        batch, labels = teacher_batch(examples, 0, "cpu")
        self.assertEqual(batch["input_ids"].tolist(), [[4, 5, 6, 7], [2, 3, 0, 0]])
        self.assertEqual(batch["attention_mask"].tolist(), [[1, 1, 1, 1], [1, 1, 0, 0]])
        self.assertEqual(labels.tolist(), [[-100, -100, 6, 7], [-100, 3, -100, -100]])
        self.assertEqual((labels[:, 1:] != -100).sum(dim=1).tolist(), [2, 1])

    def test_overlength_harmful_pair_excluded_together(self):
        tokenizer = TinyTokenizer()
        harmful = {"id": "harm", "group": "harmful", "prompt": "x", "safe_response": "safe", "harmful_response": "y" * 100}
        benign = {"id": "benign", "group": "benign", "prompt": "x", "target": "ok"}
        short, reason = encode_response(tokenizer, harmful, "safe", 1000)
        self.assertIsNone(reason)
        examples, skipped = prepare_evaluations(tokenizer, [harmful, benign], len(short["input_ids"]) + 1)
        self.assertEqual([row["id"] for row in examples], ["benign"])
        self.assertTrue(skipped[0]["entire_pair_excluded"])

    def test_safety_margin_sign_and_mean_token_definition(self):
        rows = [
            {"id": "a", "group": "harmful", "response_kind": "safe", "nll_mean": 1.0, "sequence_logprob": -10.0},
            {"id": "a", "group": "harmful", "response_kind": "harmful", "nll_mean": 3.0, "sequence_logprob": -15.0},
            {"id": "b", "group": "benign", "response_kind": "target", "nll_mean": 0.5, "sequence_logprob": -2.0},
        ]
        summary, paired = summarize_teacher(rows)
        self.assertEqual(summary["harmful"]["safety_margin"]["mean"], 2.0)
        self.assertEqual(summary["harmful"]["safe_sequence_logprob"]["mean"], -10.0)
        self.assertEqual(summary["benign"]["target_ce"]["mean"], 0.5)
        self.assertEqual(paired[0]["safety_margin"], 2.0)

    def test_generated_lengths_ignore_padding_after_first_eos(self):
        self.assertEqual(generation_length_metadata([9, 8, 2, 2, 2], [1, 2], 5),
                         {"generated_token_count": 3, "ended_with_eos": True, "hit_token_limit": False})
        self.assertEqual(generation_length_metadata([9, 8, 7], [1, 2], 3),
                         {"generated_token_count": 3, "ended_with_eos": False, "hit_token_limit": True})


if __name__ == "__main__":
    unittest.main()
