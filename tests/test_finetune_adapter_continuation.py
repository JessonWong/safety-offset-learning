import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import safeoff.training.sft as finetune


class FinetuneAdapterContinuationTests(unittest.TestCase):
    def test_local_adapter_loads_declared_base_as_trainable_and_local_tokenizer(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter_dir = Path(directory) / "checkpoint-100"
            adapter_dir.mkdir()
            (adapter_dir / "adapter_config.json").write_text(
                "{}\n",
                encoding="utf-8",
            )
            (adapter_dir / "tokenizer_config.json").write_text(
                "{}\n",
                encoding="utf-8",
            )
            args = SimpleNamespace(
                base_model_path=str(adapter_dir),
                use_lora=1,
                use_device_map=1,
            )
            peft_config = SimpleNamespace(
                base_model_name_or_path="declared/base-model"
            )
            base_model = Mock(name="base_model")
            loaded_adapter = Mock(name="loaded_adapter")
            tokenizer = SimpleNamespace(eos_token="<eos>", pad_token=None)

            with (
                patch.object(
                    finetune.PeftConfig,
                    "from_pretrained",
                    return_value=peft_config,
                ) as load_peft_config,
                patch.object(
                    finetune.AutoModelForCausalLM,
                    "from_pretrained",
                    return_value=base_model,
                ) as load_base_model,
                patch.object(
                    finetune.PeftModel,
                    "from_pretrained",
                    return_value=loaded_adapter,
                ) as load_adapter,
                patch.object(
                    finetune.AutoTokenizer,
                    "from_pretrained",
                    return_value=tokenizer,
                ) as load_tokenizer,
            ):
                model, loaded_tokenizer = finetune.load_model_and_tokenizer(args)

            load_peft_config.assert_called_once_with(str(adapter_dir))
            load_base_model.assert_called_once_with(
                "declared/base-model",
                torch_dtype=finetune.torch.bfloat16,
                device_map="auto",
            )
            load_adapter.assert_called_once_with(
                base_model,
                str(adapter_dir),
                is_trainable=True,
            )
            load_tokenizer.assert_called_once_with(str(adapter_dir))
            self.assertIs(model, loaded_adapter)
            self.assertIs(loaded_tokenizer, tokenizer)
            self.assertEqual(loaded_tokenizer.pad_token, "<eos>")
            base_model.to.assert_not_called()

    def test_local_adapter_rejects_use_lora_disabled_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter_dir = Path(directory) / "checkpoint-100"
            adapter_dir.mkdir()
            (adapter_dir / "adapter_config.json").write_text(
                "{}\n",
                encoding="utf-8",
            )
            args = SimpleNamespace(
                base_model_path=str(adapter_dir),
                use_lora=0,
                use_device_map=1,
            )

            with (
                patch.object(
                    finetune.PeftConfig,
                    "from_pretrained",
                ) as load_peft_config,
                patch.object(
                    finetune.AutoModelForCausalLM,
                    "from_pretrained",
                ) as load_base_model,
                patch.object(
                    finetune.PeftModel,
                    "from_pretrained",
                ) as load_adapter,
                patch.object(
                    finetune.AutoTokenizer,
                    "from_pretrained",
                ) as load_tokenizer,
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "requires --use_lora 1",
                ):
                    finetune.load_model_and_tokenizer(args)

            load_peft_config.assert_not_called()
            load_base_model.assert_not_called()
            load_adapter.assert_not_called()
            load_tokenizer.assert_not_called()

    def test_setup_lora_reuses_already_loaded_peft_model(self):
        class FakePeftModel:
            def __init__(self):
                self.print_trainable_parameters = Mock()

        loaded_adapter = FakePeftModel()

        with (
            patch.object(finetune, "PeftModel", FakePeftModel),
            patch.object(finetune, "get_peft_model") as create_adapter,
            patch.object(finetune, "LoraConfig") as create_lora_config,
        ):
            result = finetune.setup_lora(loaded_adapter, SimpleNamespace())

        self.assertIs(result, loaded_adapter)
        loaded_adapter.print_trainable_parameters.assert_called_once_with()
        create_adapter.assert_not_called()
        create_lora_config.assert_not_called()


if __name__ == "__main__":
    unittest.main()
