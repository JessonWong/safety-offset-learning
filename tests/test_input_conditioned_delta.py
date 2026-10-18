import unittest

import numpy as np
import torch

from safeoff.analysis.input_conditioned_delta import MODEL_NAMES, collect_base_grams, cosine_from_gram, lowrank_outputs, output_gram, pair_summary, paired_advantage, pool_prompt_inputs


class InputConditionedDeltaTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(314)

    def test_effective_lowrank_matches_dense_and_gauge_transform(self):
        inputs = torch.randn(4, 2, 13)
        a, b = torch.randn(3, 13), 4 * torch.randn(7, 3)
        actual = lowrank_outputs(inputs, a, b)
        torch.testing.assert_close(actual, inputs @ (b @ a).T)
        gauge = torch.eye(3) + .1 * torch.randn(3, 3)
        transformed = lowrank_outputs(inputs, gauge @ a, b @ torch.linalg.inv(gauge))
        torch.testing.assert_close(actual, transformed)
        gram = output_gram(torch.stack([actual, -actual], dim=-2))
        self.assertEqual(gram.dtype, torch.float64)
        np.testing.assert_allclose(cosine_from_gram(gram.numpy(), 0, 1), -1.)

    def test_padding_mean_commutes_with_delta_and_batching(self):
        x = torch.randn(2, 5, 7)
        mask = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 0, 0, 0]])
        x[~mask.bool()] = float("nan")
        pooled = pool_prompt_inputs(x, mask)
        torch.testing.assert_close(pooled[0, 0], x[0, 4])
        torch.testing.assert_close(pooled[1, 0], x[1, 1])
        torch.testing.assert_close(pooled[0, 1], x[0, 2:].mean(dim=0))
        for i in range(2):
            unpadded = x[i, mask[i].bool()][None]
            torch.testing.assert_close(pooled[i:i+1], pool_prompt_inputs(unpadded, torch.ones(unpadded.shape[:2])))
        a, b = torch.randn(2, 7), torch.randn(3, 2)
        after_pool = lowrank_outputs(pooled, a, b)
        before_pool = pool_prompt_inputs(lowrank_outputs(x, a, b), mask)
        torch.testing.assert_close(after_pool, before_pool)
        with self.assertRaises(ValueError):
            pool_prompt_inputs(x[:1], torch.zeros(1, 5))

    def test_energy_aggregation_is_not_average_cosine(self):
        outputs = torch.tensor([[[1., 0.], [1., 0.]], [[10., 0.], [-10., 0.]]])
        grams = output_gram(outputs).numpy()
        summary = pair_summary(grams, 0, 1)
        self.assertEqual(summary["mean_example_cosine"], 0.)
        self.assertAlmostEqual(summary["energy_cosine"], -99 / 101)
        # Concatenating layer outputs gives the same Gram as summing layer Grams.
        extra = torch.randn(2, 2, 3)
        torch.testing.assert_close(output_gram(torch.cat([outputs, extra], dim=-1)),
                                   output_gram(outputs) + output_gram(extra))
        zeros = pair_summary(np.zeros((2, 2, 2)), 0, 1)
        self.assertIsNone(zeros["mean_example_cosine"])
        self.assertIsNone(zeros["energy_cosine"])

    def test_bootstrap_pairs_prompts_and_preserves_sign(self):
        values = np.array([[[1., 0.], [-1., 0.], [0., 1.], [1., 0.], [0., 1.]],
                           [[3., 0.], [-2., 0.], [0., 1.], [2., 0.], [0., 1.]]])
        grams = values @ values.transpose(0, 2, 1)
        draws = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
        result = paired_advantage(grams, draws)
        self.assertEqual(result["mean_cosine_advantage"], 2.)
        self.assertEqual(result["mean_cosine_advantage_ci_low"], 2.)
        self.assertEqual(result["mean_cosine_advantage_ci_high"], 2.)
        with self.assertRaises(ValueError):
            paired_advantage(grams, draws[:, :1])

    def test_real_llama_hooks_use_normalized_common_inputs_and_remove_hooks(self):
        from transformers import LlamaConfig, LlamaForCausalLM

        model = LlamaForCausalLM(LlamaConfig(
            vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, attention_dropout=0.,
        )).eval()
        adapters = {25: {name: {layer: (torch.randn(2, 16), torch.randn(8, 2))
                               for layer in range(2)} for name in MODEL_NAMES}}
        batch = dict(input_ids=torch.tensor([[0, 0, 2, 3], [1, 3, 2, 4]]),
                     attention_mask=torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]]))
        raw = {}
        handles = [model.model.layers[layer].input_layernorm.register_forward_hook(
            lambda module, args, output, layer=layer: raw.__setitem__(layer, output.detach()))
            for layer in range(2)]
        try:
            measured = collect_base_grams(model, batch, adapters, [25], [0, 1])
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(measured.shape, (2, 1, 2, 2, 5, 5))
        for layer in range(2):
            pooled = pool_prompt_inputs(raw[layer], batch["attention_mask"])
            expected = output_gram(torch.stack([
                pooled @ (adapters[25][name][layer][1] @ adapters[25][name][layer][0]).T
                for name in MODEL_NAMES], dim=-2)).numpy()
            np.testing.assert_allclose(measured[:, 0, :, layer], expected, rtol=3e-5, atol=3e-5)
            self.assertFalse(model.model.layers[layer].self_attn.v_proj._forward_pre_hooks)


if __name__ == "__main__":
    unittest.main()
