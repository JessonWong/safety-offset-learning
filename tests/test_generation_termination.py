import torch

from safeoff.core.model_utils import _decode_generated_response, _generation_eos_token_ids


class FakeTokenizer:
    eos_token_id = 7

    def get_vocab(self):
        return {
            "<|eot_id|>": 7,
            "<|end_of_text|>": 8,
            "<|im_end|>": 9,
        }

    def decode(self, token_ids, skip_special_tokens=False):
        assert skip_special_tokens
        return " ".join(str(token_id) for token_id in token_ids.tolist())


def test_generation_terminators_include_chat_end_tokens():
    assert _generation_eos_token_ids(FakeTokenizer()) == [7, 8, 9]


def test_generated_response_stops_before_first_terminator():
    tokenizer = FakeTokenizer()
    text, token_ids = _decode_generated_response(
        tokenizer,
        torch.tensor([10, 11, 7, 12]),
        _generation_eos_token_ids(tokenizer),
    )

    assert text == "10 11"
    assert token_ids.tolist() == [10, 11]
