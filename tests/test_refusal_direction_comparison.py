import functools
import copy
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from safeoff.analysis.refusal_direction import ROOT, activation_metrics, input_projection_energy, projection_metrics, unit_direction


class RefusalComparisonTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(17)

    def test_low_rank_matches_dense_and_is_gauge_invariant(self):
        a, b = torch.randn(3, 13, dtype=torch.float64), torch.randn(7, 3, dtype=torch.float64)
        direction = unit_direction(torch.randn(13))
        expected = ((b @ a) @ direction).square().sum().item()
        self.assertAlmostEqual(input_projection_energy(a, b, direction), expected, places=10)
        gauge = torch.randn(3, 3, dtype=torch.float64) + 3 * torch.eye(3)
        self.assertAlmostEqual(input_projection_energy(gauge @ a, b @ torch.linalg.inv(gauge), direction), expected, places=10)
        self.assertLessEqual(expected, (b @ a).square().sum().item() + 1e-10)

    def test_projection_limits_and_undefined_zero_update(self):
        direction = torch.tensor([1., 0., 0.], dtype=torch.float64)
        a = torch.tensor([[2., 0., 0.]], dtype=torch.float64)
        b = torch.tensor([[1.], [3.]], dtype=torch.float64)
        energy = input_projection_energy(a, b, direction)
        result = projection_metrics(energy, (b @ a).square().sum().item(), torch.ones(10), 3)
        self.assertAlmostEqual(result['input_alignment'], 1.)
        self.assertAlmostEqual(result['isotropic_enrichment'], 3.)
        self.assertEqual(input_projection_energy(a, b, torch.tensor([0., 1., 0.], dtype=torch.float64)), 0.)
        self.assertIsNone(projection_metrics(0., 0., torch.zeros(10), 3)['input_alignment'])
        with self.assertRaises(ValueError):
            unit_direction(torch.zeros(3))
        with self.assertRaises(ValueError):
            input_projection_energy(a, b, torch.ones(2))

    def test_activation_sign_and_zero_cosine(self):
        base = torch.zeros(3, 2)
        tuned = torch.tensor([[2., 0.], [-1., 0.], [0., 0.]])
        result, projections, cosines = activation_metrics(base, tuned, torch.tensor([1., 0.]))
        self.assertEqual(result['n_nonzero_shift'], 2)
        self.assertAlmostEqual(result['mean_example_cosine'], 0.)
        self.assertAlmostEqual(result['cosine_of_mean_shift'], 1.)
        self.assertAlmostEqual(result['mean_delta_projection'], 1 / 3)
        self.assertTrue(torch.isnan(cosines[2]))
        reversed_result, _, _ = activation_metrics(base, tuned, torch.tensor([-1., 0.]))
        self.assertAlmostEqual(reversed_result['mean_delta_projection'], -1 / 3)

    @unittest.skipUnless((ROOT / "third_party/refusal_direction/pipeline").is_dir(),
                         "optional refusal_direction checkout is not installed")
    def test_real_transformer_hooks_and_adapter_switching(self):
        sys.path.insert(0, str(ROOT / 'third_party/refusal_direction'))
        from transformers import LlamaConfig, LlamaForCausalLM
        from peft import LoraConfig, get_peft_model
        from safeoff.analysis.refusal_direction import capture_activations

        model = LlamaForCausalLM(LlamaConfig(vocab_size=48, hidden_size=16, intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2)).eval()
        model.config.use_cache = False
        model.forward = functools.partial(model.forward, logits_to_keep=1)

        class Batch(dict):
            def to(self, device):
                return {key: value.to(device) for key, value in self.items()}

        def tokenize(instructions):
            ids = torch.tensor([[1, int(i), 3, 4] for i in instructions])
            return Batch(input_ids=ids, attention_mask=torch.ones_like(ids))

        base = SimpleNamespace(model=model, model_block_modules=model.model.layers,
                               tokenize_instructions_fn=tokenize, refusal_toks=[40])
        with torch.inference_mode():
            original, _ = capture_activations(base, ['5', '6', '7'], -2, 2)
            single, _ = capture_activations(base, ['5', '6', '7'], -2, 1)
            torch.testing.assert_close(original, single, atol=1e-6, rtol=1e-5)
            self.assertEqual(tuple(original.shape), (3, 2, 16))
            wrapped = get_peft_model(model, LoraConfig(r=2, lora_alpha=4, target_modules=['q_proj', 'v_proj']), adapter_name='changed')
            for name, parameter in wrapped.named_parameters():
                if 'lora_B' in name:
                    parameter.fill_(0.1)
            base.model = wrapped.eval()
            changed, _ = capture_activations(base, ['5', '6', '7'], -2, 2)
            self.assertFalse(torch.allclose(original[:, 1], changed[:, 1]))
            torch.testing.assert_close(original[:, 0], changed[:, 0])
            with wrapped.disable_adapter():
                disabled, _ = capture_activations(base, ['5', '6', '7'], -2, 2)
            torch.testing.assert_close(original, disabled)
        self.assertTrue(all(not layer._forward_pre_hooks for layer in model.model.layers))

    @unittest.skipUnless((ROOT / "third_party/refusal_direction/pipeline").is_dir(),
                         "optional refusal_direction checkout is not installed")
    def test_sequential_checkpoint_loading_matches_independent_model(self):
        """Check the production multi-adapter path against separately loaded models."""
        sys.path.insert(0, str(ROOT / 'third_party/refusal_direction'))
        from transformers import LlamaConfig, LlamaForCausalLM
        from peft import LoraConfig, PeftModel, get_peft_model
        from safeoff.analysis.refusal_direction import capture_activations

        original = LlamaForCausalLM(LlamaConfig(vocab_size=48, hidden_size=16, intermediate_size=32,
            num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2)).eval()
        original.config.use_cache = False

        class Batch(dict):
            def to(self, device):
                return {key: value.to(device) for key, value in self.items()}

        def tokenize(instructions):
            ids = torch.tensor([[1, int(i), 3, 4] for i in instructions])
            return Batch(input_ids=ids, attention_mask=torch.ones_like(ids))

        def capture(model, blocks):
            obj = SimpleNamespace(model=model.eval(), model_block_modules=blocks,
                                  tokenize_instructions_fn=tokenize, refusal_toks=[40])
            return capture_activations(obj, ['5', '6', '7'], -2, 2)

        with tempfile.TemporaryDirectory() as tmp, torch.no_grad():
            checkpoints = {}
            for name, value in [('first', 0.15), ('second', -0.25)]:
                trained = get_peft_model(copy.deepcopy(original), LoraConfig(
                    r=2, lora_alpha=4, lora_dropout=0.2, target_modules=['q_proj', 'v_proj']))
                for parameter_name, parameter in trained.named_parameters():
                    if 'lora_B' in parameter_name:
                        parameter.fill_(value)
                checkpoints[name] = Path(tmp) / name
                trained.save_pretrained(checkpoints[name])

            shared_base = copy.deepcopy(original)
            shared_blocks = shared_base.model.layers
            shared = PeftModel.from_pretrained(shared_base, checkpoints['first'], adapter_name='first')
            shared.load_adapter(checkpoints['second'], adapter_name='second')
            for name in ['second', 'first', 'second']:
                shared.set_adapter(name)
                shared.eval().requires_grad_(False)
                actual = capture(shared, shared_blocks)
                separate_base = copy.deepcopy(original)
                separate_blocks = separate_base.model.layers
                separate = PeftModel.from_pretrained(separate_base, checkpoints[name], adapter_name=name)
                expected = capture(separate, separate_blocks)
                for observed, reference in zip(actual, expected):
                    torch.testing.assert_close(observed, reference, rtol=0, atol=0)
                self.assertTrue(all(module.active_adapters == [name]
                                    for module in shared.modules() if hasattr(module, 'lora_A')))
            with shared.disable_adapter():
                disabled = capture(shared, shared_blocks)
            baseline = capture(original, original.model.layers)
            for observed, reference in zip(disabled, baseline):
                torch.testing.assert_close(observed, reference, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
