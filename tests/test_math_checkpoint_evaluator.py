import unittest

from safeoff.evaluation.math_benchmarks import extract_gold_answer, extract_response_candidates, math_equiv, score_predictions


class MathCheckpointEvaluatorTest(unittest.TestCase):
    def test_extracts_last_boxed_answer(self):
        solution = r"First we obtain 3, and the final result is $\boxed{\frac{1}{2}}$."
        self.assertEqual(extract_gold_answer(solution), r"\frac{1}{2}")
        candidates = extract_response_candidates(r"The answer is $\boxed{0.5}$.")
        self.assertTrue(any(math_equiv(candidate, "0.5") for candidate in candidates))

    def test_supports_final_line_and_math_normalization(self):
        candidates = extract_response_candidates("We simplify the expression.\nTherefore, 7/2")
        self.assertTrue(any(math_equiv(candidate, r"\frac{7}{2}") for candidate in candidates))
        self.assertTrue(math_equiv("0.5", r"\frac{1}{2}"))

    def test_scores_overall_level_and_type_metrics(self):
        docs = [
            {
                "index": 0,
                "problem": "p0",
                "solution": r"solution \boxed{\frac{1}{2}}",
                "level": "Level 1",
                "type": "Algebra",
            },
            {
                "index": 1,
                "problem": "p1",
                "solution": r"solution \boxed{7}",
                "level": "Level 2",
                "type": "Geometry",
            },
        ]
        predictions = [
            {"index": 0, "response": r"final: \boxed{0.5}"},
            {"index": 1, "response": "The answer is 8"},
        ]

        scored = score_predictions(docs, predictions)

        self.assertEqual(scored["correct_count"], 1)
        self.assertEqual(scored["sample_count"], 2)
        self.assertEqual(scored["extraction_success_count"], 2)
        self.assertEqual(scored["level_metrics"]["level_1"]["accuracy"], 1.0)
        self.assertEqual(scored["type_metrics"]["geometry"]["accuracy"], 0.0)
        self.assertTrue(predictions[0]["correct"])
        self.assertFalse(predictions[1]["correct"])


if __name__ == "__main__":
    unittest.main()
