import unittest

from safeoff.core.model_utils import filter_formatted_examples_by_length


class TinyWhitespaceTokenizer:
    """Return one token ID per whitespace-delimited word."""

    @staticmethod
    def _encode(text):
        return list(range(len(str(text).split())))

    def __call__(self, texts, add_special_tokens=False, **kwargs):
        if isinstance(texts, str):
            input_ids = self._encode(texts)
        else:
            input_ids = [self._encode(text) for text in texts]
        return {"input_ids": input_ids}


class FormattedLengthFilterTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = TinyWhitespaceTokenizer()

    def test_retains_length_equal_to_limit_and_preserves_order(self):
        examples = [
            {"id": "short", "text": "one two"},
            {"id": "at-limit", "text": "one two three"},
            {"id": "also-short", "text": "one"},
        ]

        filtered = filter_formatted_examples_by_length(
            examples,
            tokenizer=self.tokenizer,
            max_length=3,
            split_name="train",
        )

        self.assertEqual(
            [example["id"] for example in filtered],
            ["short", "at-limit", "also-short"],
        )

    def test_drops_only_overlength_examples_when_not_strict(self):
        examples = [
            {"id": "first", "text": "one two"},
            {"id": "too-long", "text": "one two three four"},
            {"id": "last", "text": "one two three"},
        ]

        filtered = filter_formatted_examples_by_length(
            examples,
            tokenizer=self.tokenizer,
            max_length=3,
            split_name="validation",
            fail_on_overlength=False,
        )

        self.assertEqual(
            [example["id"] for example in filtered],
            ["first", "last"],
        )

    def test_strict_mode_raises_instead_of_omitting_overlength_example(self):
        examples = [
            {"id": "too-long", "text": "one two three four"},
            {"id": "would-otherwise-remain", "text": "one two"},
        ]

        with self.assertRaisesRegex(ValueError, "train"):
            filter_formatted_examples_by_length(
                examples,
                tokenizer=self.tokenizer,
                max_length=3,
                split_name="train",
                fail_on_overlength=True,
            )


if __name__ == "__main__":
    unittest.main()
