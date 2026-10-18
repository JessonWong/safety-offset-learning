import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import safeoff.experiments.loss_levels as module


class SelectionTests(unittest.TestCase):
    def pool(self, losses):
        return [{"prompt": f"Complete request {i}.", "loss": value} for i, value in enumerate(losses)]

    def test_selects_spaced_losses_not_spaced_candidate_ranks(self):
        pool = self.pool([10, 9.99, 9.98, 7.5, 5, 2.5, 0])
        result = module.select_five(pool)
        self.assertEqual([r["loss"] for r in result], [10, 7.5, 5, 2.5, 0])

    def test_cannot_reuse_a_candidate_when_nearest_targets_collide(self):
        result = module.select_five(self.pool([10, 9.9, 9.8, .1, 0]))
        self.assertEqual(len({r["prompt"] for r in result}), 5)
        self.assertEqual([r["loss"] for r in result], [10, 9.9, 9.8, .1, 0])

    def test_rejects_insufficient_or_degenerate_pools(self):
        for values in ([5, 4, 3, 2], [1]*6, [2, 2, 1, 1, 0], [5, 4, 3, 2, float("nan")]):
            with self.assertRaises(ValueError):
                module.select_five(self.pool(values))

    def test_only_answer_and_eot_are_supervised(self):
        class Tokenizer:
            eos_token_id = 99
            def apply_chat_template(self, messages, **kwargs):
                return "prompt"
            def encode(self, text, **kwargs):
                return [1, 2, 3] if text == "prompt" else [4, 5]
        row = {"index": "7", "instruction": "request", "alpaca_reference": "answer"}
        pair = module.encode_pair(Tokenizer(), row, 6)
        self.assertEqual(pair["input_ids"], [1, 2, 3, 4, 5, 99])
        self.assertEqual(pair["labels"], [-100, -100, -100, 4, 5, 99])
        with self.assertRaises(ValueError):
            module.encode_pair(Tokenizer(), row, 5)


class ScreeningResponseTests(unittest.TestCase):
    def response(self, text, finish="STOP"):
        return {"candidates": [{"finishReason": finish, "content": {"parts": [{"text": text}]}}]}

    def test_complete_boolean_verdict(self):
        import json
        verdict = dict(complete_request=True, benign_request=True, benign_answer=True, answer_matches=False)
        result = module.parse_screening_response(self.response("```json\n" + json.dumps(verdict) + "\n```"))
        self.assertEqual(result["verdict"], verdict)
        self.assertFalse(result["accepted"])

    def test_truncated_output_is_never_a_verdict(self):
        for text in ('{"complete_request":true,', '{}'):
            with self.assertRaisesRegex(ValueError, "MAX_TOKENS"):
                module.parse_screening_response(self.response(text, "MAX_TOKENS"))

    def test_strings_and_missing_decisions_are_not_booleans(self):
        for text in ('{}', '{"complete_request":"false","benign_request":true,"benign_answer":true,"answer_matches":true}'):
            with self.assertRaises(ValueError):
                module.parse_screening_response(self.response(text))

    def test_apostrophe_escape_in_reason_does_not_change_decisions(self):
        text = r'''{"complete_request":true,"benign_request":true,"benign_answer":true,"answer_matches":false,"reason":"The translated phrase is I don\'."}'''
        result = module.parse_screening_response(self.response(text))
        self.assertFalse(result["accepted"])
        self.assertEqual(result["verdict"]["reason"], "The translated phrase is I don'.")
        with self.assertRaises(ValueError):
            module.parse_screening_response(self.response(text.replace("\\'", "\\q")))


class SingleGpuSchedulingTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.run = Path(self.folder.name)
        module.write_csv(self.run / "sources.csv", [{"index": str(i)} for i in range(4)])
        self.snapshot = self.run / "refill.json"
        module.write_json(self.snapshot, [{"source_index": "3"}, {"source_index": "1"}])
        self.args = SimpleNamespace(run=self.run, source_indices_file=self.snapshot,
                                    source_index=None, start=0, count=4, trial_start=24,
                                    trial_count=12, round=2, workers=6)

    def test_refill_preserves_original_order_and_rejects_invalid_source_ids(self):
        self.assertEqual([r["index"] for r in module.generation_sources(self.args)], ["1", "3"])
        for indices in (["1", "1"], ["7"], []):
            module.write_json(self.snapshot, [{"source_index": i} for i in indices])
            with self.assertRaises(ValueError):
                module.generation_sources(self.args)

    def test_all_stages_execute_without_additional_submissions(self):
        with patch.object(module.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as child:
            with patch.object(module, "submit_job", side_effect=AssertionError("Unexpected submission")):
                module.serial_refill(self.args)
        commands = [call.args[0] for call in child.call_args_list]
        self.assertTrue(all(cmd[2:4] == ["-m", "safeoff.experiments.loss_levels"] for cmd in commands))
        self.assertEqual([cmd[4] for cmd in commands],
                         ["generate", "screen", "select"] + ["train_eval"] * 7 + ["score_plot"])
        self.assertEqual([cmd[-1] for cmd in commands if cmd[4] == "train_eval"], module.ARMS)
        self.assertIn(str(self.snapshot), commands[0])

    def test_failed_selection_stops_before_training_and_preserves_failure(self):
        def run_child(command):
            if command[4] == "select":
                module.write_json(self.run / "status.json", {"status": "failed", "stage": "select"})
                return SimpleNamespace(returncode=1)
            return SimpleNamespace(returncode=0)
        with patch.object(module.subprocess, "run", side_effect=run_child) as child:
            with self.assertRaises(SystemExit) as error:
                module.serial_refill(self.args)
        self.assertEqual(error.exception.code, 1)
        self.assertEqual(child.call_count, 3)
        self.assertEqual(json.loads((self.run / "status.json").read_text()),
                         {"status": "failed", "stage": "select"})


    def test_cached_training_mode_never_regenerates_or_rejudges(self):
        with patch.object(module.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as child:
            with patch.object(module, "submit_job", side_effect=AssertionError("Unexpected submission")):
                module.serial_train_eval(self.args)
        commands = [call.args[0] for call in child.call_args_list]
        self.assertTrue(all(cmd[2:4] == ["-m", "safeoff.experiments.loss_levels"] for cmd in commands))
        self.assertEqual([cmd[4] for cmd in commands],
                         ["select"] + ["train_eval"] * 7 + ["score_plot"])


class ScreeningPolicyTests(unittest.TestCase):
    def test_relaxed_policy_reuses_verdict_fields_instead_of_old_accepted_flag(self):
        saved = {"accepted": False, "verdict": dict(complete_request=False, benign_request=True,
                                                     benign_answer=True, answer_matches=False)}
        relaxed = {"screening_required_fields": list(module.BENIGN_SCREEN_FIELDS)}
        self.assertFalse(module.accepts_screening(saved, {}))
        self.assertTrue(module.accepts_screening(saved, relaxed))
        self.assertFalse(module.accepts_screening({**saved, "provider_blocked": True}, relaxed))
        for key in module.BENIGN_SCREEN_FIELDS:
            for value in (False, None, "true"):
                changed = {**saved, "verdict": {**saved["verdict"], key: value}}
                self.assertFalse(module.accepts_screening(changed, relaxed))

    def test_selection_policy_does_not_invalidate_generation_but_generation_changes_do(self):
        original = {"generation_seed": 2025, "generation_lambda": 80}
        relaxed = {**original, "screening_required_fields": list(module.BENIGN_SCREEN_FIELDS)}
        self.assertEqual(module.generation_fingerprint(original), module.generation_fingerprint(relaxed))
        changed = {**relaxed, "generation_lambda": 81}
        self.assertNotEqual(module.generation_fingerprint(original), module.generation_fingerprint(changed))
        with self.assertRaises(ValueError):
            module.screening_fields({"screening_required_fields": []})

    def test_selection_keeps_paired_answers_and_rejects_nonbenign_candidates(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            p = {"judge": "test-judge", "source_indices": ["0"],
                 "screening_required_fields": list(module.BENIGN_SCREEN_FIELDS)}
            row = {"index": "0", "instruction": "Original input", "alpaca_reference": "Original answer", "suffix": ""}
            module.write_json(run / "protocol.json", p)
            module.write_csv(run / "sources.csv", [row], module.CSV_FIELDS)
            pool = [{"prompt": f"Candidate {i}", "loss": float(6-i), "token_length": 2} for i in range(6)]
            for i, candidate in enumerate(pool):
                payload = {"original_instruction": row["instruction"], "rewritten_instruction": candidate["prompt"],
                           "reference_answer": row["alpaca_reference"]}
                key = module.digest({"payload": payload, "rubric": module.SCREEN_PROMPT, "judge": p["judge"]})
                verdict = dict(complete_request=False, benign_request=i < 5, benign_answer=True, answer_matches=False)
                module.write_json(run / "screening" / f"{key}.json",
                                  {"payload": payload, "judge": p["judge"], "accepted": False, "verdict": verdict})
            with patch.object(module, "load_pools", return_value={"0": pool}):
                module.select(SimpleNamespace(run=run))
            selected = json.loads((run / "selected_candidates.json").read_text())
            self.assertEqual([r["prompt"] for r in selected], [f"Candidate {i}" for i in range(5)])
            for arm in module.ARMS[:5]:
                data = module.read_csv(run / "datasets" / f"{arm}.csv")
                self.assertEqual(data[0]["alpaca_reference"], row["alpaca_reference"])
                self.assertEqual(data[0]["index"], "0")


if __name__ == "__main__":
    unittest.main()
