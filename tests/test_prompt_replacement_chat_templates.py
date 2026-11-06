"""Exercise native chat formatting through the actual search/target sequences."""

import string
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM


from safeoff.generation.search.chat_templates import resolve_chat_template
from safeoff.generation.search.model import LLM
from safeoff.generation.replacement import build_prompter_params, build_target_params
from safeoff.generation.search.sequence import MergedSeq, Seq, msg_to_seq


LLAMA_TEMPLATE = (
    "{{ bos_token }}"
    "{% for message in messages %}"
    "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' "
    "+ message['content'] + '<|eot_id|>' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
    "{% endif %}"
)
QWEN_TEMPLATE = (
    "{% if messages[0]['role'] != 'system' %}"
    "{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}"
    "{% endif %}"
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] "
    "+ '<|im_end|>\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)


def make_tokenizer(style):
    specials = [
        "<unk>", "<pad>", "<|begin_of_text|>", "<|eot_id|>",
        "<|start_header_id|>", "<|end_header_id|>",
        "<|im_start|>", "<|im_end|>",
    ]
    vocab = {token: i for i, token in enumerate(specials + list(string.printable))}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")
    backend.decoder = decoders.Fuse()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<|begin_of_text|>",
        eos_token="<|eot_id|>" if style == "llama" else "<|im_end|>",
        additional_special_tokens=specials[2:],
        chat_template=LLAMA_TEMPLATE if style == "llama" else QWEN_TEMPLATE,
        clean_up_tokenization_spaces=False,
    )


class PromptReplacementChatTemplateTests(unittest.TestCase):
    def args(self):
        return SimpleNamespace(
            target_checkpoint="target", prompter_checkpoint="prompter",
            target_device="cpu", prompter_device="cpu", dtype="float32",
            prompt_tokens=8, response_max_new_tokens=8,
        )

    def llm(self, style, params):
        tokenizer = make_tokenizer(style)
        model = Qwen2ForCausalLM(Qwen2Config(
            vocab_size=len(tokenizer), hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
            bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            sliding_window=None,
        )).eval()
        params.allow_non_ascii = True
        with patch("safeoff.generation.search.model.llm_loader", return_value=(
            model, tokenizer, model.get_input_embeddings().weight,
        )):
            return LLM(params)

    def test_target_prompt_and_refusal_loss_use_native_template(self):
        for style in ("llama", "qwen"):
            with self.subTest(style=style):
                llm = self.llm(style, build_target_params(self.args()))
                instruction = "Name three colors."
                refusal = "I can't fulfill that request."
                expected = llm.tokenizer.apply_chat_template(
                    [{"role": "user", "content": instruction}],
                    tokenize=True, add_generation_prompt=True,
                )
                loss = llm.compute_pred_loss_teacher_forced(
                    loss_params=dict(hard_labels=True, reweight_loss=False),
                    key="target",
                    full_instruct=Seq(text=[instruction], tokenizer=llm.tokenizer, device="cpu"),
                    target=Seq(text=[refusal], tokenizer=llm.tokenizer, device="cpu"),
                )
                self.assertEqual(loss.query.ids[0].tolist(), expected)
                self.assertEqual(loss.response_teacher.text, [refusal])
                self.assertTrue(torch.isfinite(loss.loss).item())

    def test_target_generation_uses_native_chat_context(self):
        for style in ("llama", "qwen"):
            with self.subTest(style=style):
                llm = self.llm(style, build_target_params(self.args()))
                instruction = "Name three colors."
                generated = llm.generate_autoregressive(
                    key="target", max_new_tokens=2,
                    full_instruct=Seq(text=[instruction], tokenizer=llm.tokenizer, device="cpu"),
                )
                expected = llm.tokenizer.apply_chat_template(
                    [{"role": "user", "content": instruction}],
                    tokenize=True, add_generation_prompt=True,
                )
                self.assertEqual(generated.query.ids[0].tolist(), expected)
                self.assertGreater(generated.response_sample.seq_len, 0)

    def test_all_standalone_prompters_continue_the_native_assistant_turn(self):
        strategies = (
            "standalone_v2", "standalone_v3", "standalone_v4",
            "standalone_baseline", "standalone_simple_rewrite",
        )
        for style in ("llama", "qwen"):
            for strategy in strategies:
                with self.subTest(style=style, strategy=strategy):
                    params = build_prompter_params(self.args(), strategy)
                    llm = self.llm(style, params)
                    instruction = "Name three colors."
                    beam = "Please name"
                    query, continuation = llm.prepare_prompt(
                        {
                            "instruct": Seq(text=[instruction], tokenizer=llm.tokenizer, device="cpu"),
                            "replacement": Seq(text=[beam], tokenizer=llm.tokenizer, device="cpu"),
                        },
                        up_to_key="replacement", return_key_seq=True,
                    )
                    rewrite_instructions = next(
                        part.msg for part in params.prompt_manager.prompt_template
                        if part.key == "replacement_intro"
                    )
                    user_content = "Original user prompt:\n" + instruction + "\n\n" + rewrite_instructions
                    expected = llm.tokenizer.apply_chat_template(
                        [{"role": "user", "content": user_content}],
                        tokenize=False, add_generation_prompt=True,
                    )
                    self.assertEqual(query.text, [expected])
                    self.assertEqual(
                        MergedSeq([query, continuation]).ids[0].tolist(),
                        llm.tokenizer.encode(expected + beam, add_special_tokens=False),
                    )

    def test_prompter_and_target_resolve_their_own_tokenizers(self):
        prompter = self.llm("qwen", build_prompter_params(self.args(), "standalone_v3"))
        target = self.llm("llama", build_target_params(self.args()))
        self.assertIn("<|im_start|>", prompter.prompt_template[0]["msg"])
        self.assertNotIn("<|start_header_id|>", prompter.prompt_template[0]["msg"])
        self.assertIn("<|start_header_id|>", target.prompt_template[0]["msg"])

    def test_turn_delimiters_are_not_masked_when_eos_is_also_padding(self):
        for style in ("llama", "qwen"):
            with self.subTest(style=style):
                llm = self.llm(style, build_target_params(self.args()))
                llm.tokenizer.pad_token = llm.tokenizer.eos_token
                query = llm.prepare_prompt(
                    {"full_instruct": Seq(text=["Name colors."], tokenizer=llm.tokenizer, device="cpu")},
                    up_to_key="target",
                )
                eos_positions = query.ids == llm.tokenizer.eos_token_id
                self.assertTrue(eos_positions.any().item())
                self.assertTrue(query.mask[eos_positions].all().item())

    def test_explicit_non_chat_templates_need_no_tokenizer_chat_template(self):
        tokenizer = make_tokenizer("qwen")
        tokenizer.chat_template = None
        template = [{"key": "instruction", "msg": "{instruct}"}]
        self.assertEqual(resolve_chat_template(tokenizer, template), template)
        with self.assertRaisesRegex(ValueError, "chat_template"):
            resolve_chat_template(tokenizer, [{"key": "prefix", "chat_boundary": "user"}])

    def test_empty_native_boundary_is_a_valid_sequence_fragment(self):
        tokenizer = make_tokenizer("qwen")
        tokenizer.chat_template = "{{ messages[0]['content'] }}"
        resolved = resolve_chat_template(tokenizer, [{"key": "prefix", "chat_boundary": "user"}])
        seq = msg_to_seq(resolved[0]["msg"], tokenizer, "cpu")
        self.assertEqual(seq.seq_len, 0)


if __name__ == "__main__":
    unittest.main()
