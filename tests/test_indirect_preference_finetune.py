import importlib.util
import json
import math
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch


import safeoff.training.preference as MODULE


class TinyChatTokenizer:
    name_or_path = "tiny-chat-tokenizer"
    vocab_size = 512
    pad_token_id = 0
    special_tokens_map = {"pad_token": "<pad>", "eos_token": "<eos>"}
    chat_template = "tiny-template-v1"

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=False,
    ):
        assert not tokenize
        text = "".join(
            f"<{message['role']}>{message['content']}</{message['role']}>"
            for message in messages
        )
        if add_generation_prompt:
            text += "<assistant>"
        elif messages and messages[-1]["role"] == "assistant":
            # Make the full conversation begin with exactly the same generation
            # prefix as the prompt-only conversation.
            assistant = messages[-1]["content"]
            prefix_messages = messages[:-1]
            text = "".join(
                f"<{message['role']}>{message['content']}</{message['role']}>"
                for message in prefix_messages
            )
            text += f"<assistant>{assistant}<eos>"
        return text

    def __call__(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return {"input_ids": [ord(character) + 1 for character in text]}


class UniformLogitModel(torch.nn.Module):
    def __init__(self, vocab_size=512):
        super().__init__()
        self.offset = torch.nn.Parameter(torch.zeros(()))
        self.vocab_size = vocab_size

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        del attention_mask, use_cache
        logits = self.offset + torch.zeros(
            (*input_ids.shape, self.vocab_size),
            device=input_ids.device,
        )
        return SimpleNamespace(logits=logits)


def valid_pair(**updates):
    pair = {
        "pair_id": "pair-1",
        "prompt": "Give three tips for staying healthy.",
        "chosen": "Eat well, exercise regularly, and get enough sleep.",
        "rejected": (
            "Individual health needs vary, so a clinician can offer more "
            "appropriate personalized guidance."
        ),
        "rejected_is_actual_response": True,
        "rejected_label": "indirect_rejection",
        "direct_refusal": False,
        "same_row": True,
        "benign": True,
        "chosen_harmful_info_score": 1.0,
        "chosen_harmful_info_explanation": "The answer is benign and non-actionable.",
    }
    pair.update(updates)
    return pair


class PreferencePairValidationTests(unittest.TestCase):
    def test_accepts_only_canonical_actual_response_pair(self):
        records = MODULE.validate_preference_records([valid_pair()])

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["pair_id"], "pair-1")
        self.assertTrue(records[0]["rejected_is_actual_response"])
        self.assertEqual(records[0]["rejected_label"], "indirect_rejection")

    def test_rejects_synthetic_target_or_wrong_judge_label(self):
        with self.assertRaisesRegex(ValueError, "rejected_is_actual_response"):
            MODULE.validate_preference_records(
                [valid_pair(rejected_is_actual_response=False)]
            )
        with self.assertRaisesRegex(ValueError, "rejected_label"):
            MODULE.validate_preference_records(
                [valid_pair(rejected_label="fulfillment")]
            )

    def test_recomputes_direct_refusal_detector(self):
        with self.assertRaisesRegex(ValueError, "direct-refusal keyword"):
            MODULE.validate_preference_records(
                [valid_pair(rejected="I cannot provide those health tips.")]
            )

    def test_rejects_chosen_answer_above_harmful_info_threshold(self):
        with self.assertRaisesRegex(ValueError, "exceeds allowed maximum"):
            MODULE.validate_preference_records(
                [valid_pair(chosen_harmful_info_score=3.0)],
                max_chosen_harmful_info_score=2.0,
            )

    def test_requires_numeric_chosen_harmful_info_score_and_explanation(self):
        with self.assertRaisesRegex(ValueError, "must be numeric"):
            MODULE.validate_preference_records(
                [valid_pair(chosen_harmful_info_score="")]
            )
        with self.assertRaisesRegex(ValueError, "chosen_harmful_info_explanation"):
            MODULE.validate_preference_records(
                [valid_pair(chosen_harmful_info_explanation="")]
            )

    def test_rejects_duplicate_pair_ids(self):
        with self.assertRaisesRegex(ValueError, "duplicate pair_id"):
            MODULE.validate_preference_records([valid_pair(), valid_pair()])

    def test_csv_boolean_fields_are_parsed_strictly(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pairs.json"
            pair = valid_pair(
                rejected_is_actual_response="true",
                direct_refusal="false",
                same_row="1",
                benign="true",
            )
            path.write_text(json.dumps([pair]), encoding="utf-8")

            records = MODULE.load_preference_records(path)

        self.assertTrue(records[0]["same_row"])
        self.assertFalse(records[0]["direct_refusal"])


class PreferenceTokenizationTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = TinyChatTokenizer()
        self.record = MODULE.validate_preference_records([valid_pair()])[0]

    def test_prompt_is_masked_and_both_completions_remain(self):
        tokenized = MODULE.tokenize_preference_record(
            self.record,
            self.tokenizer,
            "You are helpful.",
            max_sequence_length=1000,
        )

        chosen_mask = tokenized["chosen_completion_mask"]
        rejected_mask = tokenized["rejected_completion_mask"]
        self.assertIn(1, chosen_mask)
        self.assertIn(1, rejected_mask)
        first_chosen = chosen_mask.index(1)
        first_rejected = rejected_mask.index(1)
        self.assertEqual(first_chosen, first_rejected)
        self.assertTrue(all(value == 0 for value in chosen_mask[:first_chosen]))
        self.assertTrue(all(value == 1 for value in chosen_mask[first_chosen:]))

    def test_collator_orders_chosen_then_rejected_and_masks_padding(self):
        first = MODULE.tokenize_preference_record(
            self.record,
            self.tokenizer,
            None,
            max_sequence_length=1000,
        )
        second_record = dict(self.record)
        second_record["pair_id"] = "pair-2"
        second_record["chosen"] = "A short answer."
        second = MODULE.tokenize_preference_record(
            second_record,
            self.tokenizer,
            None,
            max_sequence_length=1000,
        )
        collator = MODULE.PreferenceDataCollator(self.tokenizer.pad_token_id)

        batch = collator([first, second])

        self.assertEqual(batch["input_ids"].shape[0], 4)
        chosen_first_length = len(first["chosen_input_ids"])
        rejected_first_length = len(first["rejected_input_ids"])
        self.assertTrue(
            torch.equal(
                batch["input_ids"][0, :chosen_first_length],
                torch.tensor(first["chosen_input_ids"]),
            )
        )
        self.assertTrue(
            torch.equal(
                batch["input_ids"][2, :rejected_first_length],
                torch.tensor(first["rejected_input_ids"]),
            )
        )
        self.assertTrue(
            torch.all(batch["completion_mask"][batch["attention_mask"] == 0] == 0)
        )

    def test_overlength_pair_fails_by_default(self):
        with self.assertRaisesRegex(ValueError, "overlength"):
            MODULE.build_tokenized_dataset(
                [self.record],
                self.tokenizer,
                None,
                max_sequence_length=10,
                fail_on_overlength=True,
                split_name="train",
            )


class PreferenceLossTests(unittest.TestCase):
    def test_completion_logps_only_include_masked_tokens(self):
        logits = torch.zeros((1, 4, 3), dtype=torch.float32)
        input_ids = torch.tensor([[0, 1, 2, 1]], dtype=torch.long)
        completion_mask = torch.tensor([[False, False, True, True]])

        logps, counts = MODULE.completion_sequence_logps(
            logits, input_ids, completion_mask
        )

        self.assertEqual(counts.item(), 2)
        self.assertAlmostEqual(logps.item(), -2.0 * math.log(3.0), places=5)

    def test_equal_policy_and_reference_has_log_two_loss(self):
        policy_chosen = torch.tensor([-3.0], requires_grad=True)
        policy_rejected = torch.tensor([-5.0], requires_grad=True)
        reference_chosen = torch.tensor([-3.0])
        reference_rejected = torch.tensor([-5.0])

        loss, metrics = MODULE.reference_corrected_dpo_loss(
            policy_chosen,
            policy_rejected,
            reference_chosen,
            reference_rejected,
            beta=0.1,
        )

        self.assertAlmostEqual(loss.item(), math.log(2.0), places=6)
        self.assertAlmostEqual(metrics["reward_margin"].item(), 0.0, places=6)

    def test_gradient_increases_chosen_and_decreases_rejected_logp(self):
        policy_chosen = torch.tensor([0.0], requires_grad=True)
        policy_rejected = torch.tensor([0.0], requires_grad=True)
        zeros = torch.tensor([0.0])
        loss, _ = MODULE.reference_corrected_dpo_loss(
            policy_chosen,
            policy_rejected,
            zeros,
            zeros,
            beta=1.0,
        )

        loss.backward()

        self.assertLess(policy_chosen.grad.item(), 0.0)
        self.assertGreater(policy_rejected.grad.item(), 0.0)

    def test_trainer_eval_outputs_one_preference_logit_per_pair(self):
        tokenizer = TinyChatTokenizer()
        records = MODULE.validate_preference_records(
            [valid_pair(), valid_pair(pair_id="pair-2")]
        )
        dataset = MODULE.build_tokenized_dataset(
            records,
            tokenizer,
            None,
            max_sequence_length=1000,
            fail_on_overlength=True,
            split_name="train",
        )
        references = [
            {
                "pair_id": pair_id,
                "reference_chosen_logp": -10.0,
                "reference_rejected_logp": -10.0,
            }
            for pair_id in dataset["pair_id"]
        ]
        dataset = MODULE.attach_reference_rows(dataset, references, "train")
        batch = MODULE.PreferenceDataCollator(tokenizer.pad_token_id)(
            [dataset[0], dataset[1]]
        )
        trainer = object.__new__(MODULE.ReferenceCorrectedDPOTrainer)
        trainer.beta = 0.1
        trainer.chosen_sft_weight = 0.0
        trainer._stored_train_metrics = defaultdict(list)
        model = UniformLogitModel()
        model.eval()

        loss, outputs = trainer.compute_loss(model, batch, return_outputs=True)

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(outputs["preference_logits"].shape, (2,))


class ReferenceCacheTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = TinyChatTokenizer()
        self.records = MODULE.validate_preference_records([valid_pair()])

    def signature(self, records=None):
        return MODULE.build_reference_signature(
            records or self.records,
            [],
            "example/base-model",
            "trainable-weights-hash",
            self.tokenizer,
            "You are helpful.",
            512,
        )

    def test_signature_changes_with_pair_content(self):
        original = self.signature()
        changed_records = MODULE.validate_preference_records(
            [valid_pair(chosen="A different benign answer.")]
        )
        changed = self.signature(changed_records)

        self.assertNotEqual(original["signature"], changed["signature"])

    def test_cache_rejects_signature_mismatch_and_attaches_in_order(self):
        signature = self.signature()
        rows = [
            {
                "pair_id": "pair-1",
                "reference_chosen_logp": -4.0,
                "reference_rejected_logp": -3.0,
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.json"
            MODULE.write_reference_cache(path, signature, rows, [])
            cached = MODULE.load_reference_cache(path, signature)
            with self.assertRaisesRegex(ValueError, "signature mismatch"):
                MODULE.load_reference_cache(
                    path,
                    {**signature, "signature": "wrong"},
                )

        dataset = MODULE.build_tokenized_dataset(
            self.records,
            self.tokenizer,
            None,
            max_sequence_length=1000,
            fail_on_overlength=True,
            split_name="train",
        )
        attached = MODULE.attach_reference_rows(dataset, cached["train"], "train")
        self.assertEqual(attached[0]["reference_chosen_logp"], -4.0)

    def test_reference_precompute_preserves_pair_order_and_training_mode(self):
        second = valid_pair(pair_id="pair-2", chosen="A second benign answer.")
        records = MODULE.validate_preference_records([valid_pair(), second])
        dataset = MODULE.build_tokenized_dataset(
            records,
            self.tokenizer,
            None,
            max_sequence_length=1000,
            fail_on_overlength=True,
            split_name="train",
        )
        model = UniformLogitModel()
        model.train()

        rows = MODULE.compute_initial_reference_logps(
            model,
            dataset,
            MODULE.PreferenceDataCollator(self.tokenizer.pad_token_id),
            batch_size=1,
        )

        self.assertEqual([row["pair_id"] for row in rows], ["pair-1", "pair-2"])
        self.assertTrue(model.training)
        self.assertTrue(
            all(math.isfinite(row["reference_chosen_logp"]) for row in rows)
        )

    def test_resume_rejects_checkpoint_zero_without_trainer_state(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "checkpoint-0").mkdir()
            with self.assertRaisesRegex(ValueError, "not a resumable"):
                MODULE.resolve_stateful_resume_checkpoint(output, "latest")

            checkpoint = output / "checkpoint-25"
            checkpoint.mkdir()
            (checkpoint / "trainer_state.json").write_text("{}", encoding="utf-8")
            resolved = MODULE.resolve_stateful_resume_checkpoint(output, "latest")

        self.assertEqual(resolved.name, "checkpoint-25")


class CheckpointScheduleTests(unittest.TestCase):
    def test_final_step_is_always_saved(self):
        callback = MODULE.FinalStepCheckpointCallback()
        control = SimpleNamespace(should_save=False)

        callback.on_step_end(
            SimpleNamespace(),
            SimpleNamespace(global_step=7, max_steps=8),
            control,
        )
        self.assertFalse(control.should_save)

        callback.on_step_end(
            SimpleNamespace(),
            SimpleNamespace(global_step=8, max_steps=8),
            control,
        )
        self.assertTrue(control.should_save)


if __name__ == "__main__":
    unittest.main()
