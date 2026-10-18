from safeoff.paths import project_root
# Modular, function-based fine-tuning script
import os
import csv
import inspect
import torch
import numpy as np
import random
from easydict import EasyDict
import json
import ast
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from contextlib import contextmanager
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    TrainerCallback
)
from peft import LoraConfig, PeftConfig, get_peft_model, PeftModel
from peft.tuners.lora.layer import LoraLayer
from argparse import ArgumentParser
from safeoff.core.model_utils import configs, get_model_path, get_datasets, compute_similarity, get_logits
from datasets import Dataset
import gc
import torch.nn as nn
import copy
import safeoff.core.model_utils as utils
from pathlib import Path

DEFAULT_ADVPROMPT_CLEAN_CSV = project_root() / "data" / "alpaca_cleaned.csv"
DEFAULT_ALPACA_TOP100_JSON = project_root() / "data" / "alpaca_top100.json"
DEFAULT_AOA_10_JSON = project_root() / "data" / "aoa_10.json"
LOCAL_IDENTITY_SHIFT_CACHE_DIR = project_root() / "cache" / "local_identity_shift"

class GradientSavingTrainer(Trainer):
    def __init__(
        self,
        *args,
        save_grads_every=100,
        grad_dir=None,
        use_fixed_grad_batch=False,
        fixed_grad_batch_indices=None,
        fixed_grad_seed=0,
        fixed_grad_save_mode="trainable",
        grad_save_dtype="float16",
        print_loss_tokens=False,
        print_loss_tokens_max_examples=2,
        **kwargs,
    ):
        if "processing_class" in inspect.signature(Trainer.__init__).parameters:
            if "tokenizer" in kwargs and "processing_class" not in kwargs:
                kwargs["processing_class"] = kwargs.pop("tokenizer")
        super().__init__(*args, **kwargs)
        self.save_grads_every = save_grads_every
        self.grad_dir = Path(grad_dir or Path(self.args.output_dir) / "gradients")
        self.grad_dir.mkdir(parents=True, exist_ok=True)
        self.use_fixed_grad_batch = use_fixed_grad_batch
        self.fixed_grad_batch_indices = fixed_grad_batch_indices
        self.fixed_grad_seed = fixed_grad_seed
        self.fixed_grad_save_mode = fixed_grad_save_mode
        self.grad_save_dtype = self._resolve_grad_dtype(grad_save_dtype)
        self._fixed_grad_batch = None
        self.print_loss_tokens = print_loss_tokens
        self.print_loss_tokens_max_examples = max(1, int(print_loss_tokens_max_examples))
        self._loss_tokens_printed = False

    def _save_grads(self, grads, step):
        torch.save(grads, self.grad_dir / f"step-{step}.pt")

    @staticmethod
    def _resolve_grad_dtype(dtype_name):
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        if dtype_name not in dtype_map:
            raise ValueError(f"Unsupported grad_save_dtype: {dtype_name}")
        return dtype_map[dtype_name]

    def _stash_grad(self, grad):
        return grad.detach().to(device="cpu", dtype=self.grad_save_dtype).clone()

    def _collect_current_grads(self, model):
        return {
            n: self._stash_grad(p.grad)
            for n, p in model.named_parameters() if p.grad is not None
        }

    def _get_fixed_grad_batch(self):
        if self._fixed_grad_batch is not None:
            return self._fixed_grad_batch
        if self.train_dataset is None:
            return None
        if not hasattr(self.train_dataset, "__len__"):
            return None
        if self.fixed_grad_batch_indices is None:
            batch_size = min(self.args.per_device_train_batch_size, len(self.train_dataset))
            indices = list(range(batch_size))
        else:
            indices = list(self.fixed_grad_batch_indices)
        dataset = self.train_dataset
        used_indices = indices
        if hasattr(dataset, "select"):
            dataset = dataset.select(indices)
            used_indices = range(len(dataset))
        if hasattr(self, "_remove_unused_columns"):
            dataset = self._remove_unused_columns(dataset, description="fixed grad batch")
        features = [dataset[i] for i in used_indices]
        if self.data_collator is None:
            from transformers import default_data_collator
            batch = default_data_collator(features)
        else:
            batch = self.data_collator(features)
        self._fixed_grad_batch = self._prepare_inputs(batch)
        return self._fixed_grad_batch

    def _save_fixed_batch_grads(self, model, step):
        batch = self._get_fixed_grad_batch()
        if batch is None:
            return
        was_training = model.training
        model.train()
        cpu_rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        if self.fixed_grad_seed is not None:
            torch.manual_seed(self.fixed_grad_seed)
            torch.cuda.manual_seed_all(self.fixed_grad_seed)
        params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        save_trainable = self.fixed_grad_save_mode in {"trainable", "both"}
        save_delta = self.fixed_grad_save_mode in {"delta", "both"}
        # If requested, also capture gradients for dense ΔW = B @ A by differentiating
        # w.r.t. the base layer weight and applying the LoRA scaling. This is expensive
        # and should stay opt-in for large models.
        lora_meta = []
        base_weights = []
        if save_delta:
            for module_name, module in model.named_modules():
                if isinstance(module, LoraLayer) and hasattr(module, "base_layer") and hasattr(module.base_layer, "weight"):
                    w = module.base_layer.weight
                    reset = False
                    if not w.requires_grad:
                        w.requires_grad_(True)
                        reset = True
                    lora_meta.append((module_name, module, w, reset))
                    base_weights.append(w)
        grad_targets = [p for _, p in params] + base_weights
        try:
            with torch.set_grad_enabled(True):
                outputs = model(**batch)
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
                grads_list = torch.autograd.grad(loss, grad_targets, allow_unused=True)
        finally:
            torch.set_rng_state(cpu_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(cuda_rng_state)
        main_grads = grads_list[: len(params)]
        base_grads = grads_list[len(params) :]
        grads = {}
        if save_trainable:
            grads.update({
                n: self._stash_grad(g)
                for (n, _), g in zip(params, main_grads) if g is not None
            })
        try:
            if lora_meta and base_grads:
                for (module_name, module, w, _), g in zip(lora_meta, base_grads):
                    if g is None:
                        continue
                    scaling = module.scaling
                    if isinstance(scaling, dict):
                        active = getattr(module, "active_adapter", None)
                        if isinstance(active, (list, tuple)):
                            active = active[0] if active else None
                        if active is None:
                            scaling = next(iter(scaling.values()))
                        else:
                            scaling = scaling[active]
                    grads[f"{module_name}.deltaW"] = self._stash_grad(g * scaling)
        finally:
            for _, _, w, reset in lora_meta:
                if reset:
                    w.requires_grad_(False)
        if not was_training:
            model.eval()
        self._save_grads(grads, step)

    def _maybe_print_loss_tokens(self, inputs):
        if not self.print_loss_tokens or self._loss_tokens_printed:
            return
        labels = inputs.get("labels")
        input_ids = inputs.get("input_ids")
        # print("=== Printing loss-bearing tokens for the first training batch ===")
        # print(labels[0])
        # print(input_ids[0])
        # exit()
        tokenizer = getattr(self, "tokenizer", None)
        if labels is None or input_ids is None or tokenizer is None:
            return

        print("Loss-bearing tokens from current batch:")
        batch_size = min(len(labels), self.print_loss_tokens_max_examples)
        for batch_idx in range(batch_size):
            label_row = labels[batch_idx].detach().cpu().tolist()
            kept_positions = [i for i, token_id in enumerate(label_row) if token_id != -100]
            kept_label_ids = [label_row[i] for i in kept_positions]
            kept_tokens = tokenizer.convert_ids_to_tokens(kept_label_ids)
            kept_text = tokenizer.decode(kept_label_ids, skip_special_tokens=False)
            id_token_pairs = list(zip(kept_label_ids, kept_tokens))
            print(f"[sample {batch_idx}] positions={kept_positions}")
            print(f"[sample {batch_idx}] label_ids={kept_label_ids}")
            print(f"[sample {batch_idx}] tokens={kept_tokens}")
            print(f"[sample {batch_idx}] id_token_pairs={id_token_pairs}")
            print(f"[sample {batch_idx}] text={repr(kept_text)}")
        self._loss_tokens_printed = True

    def training_step(self, model, inputs, num_items_in_batch=None):
        # Pass through to HF Trainer (which may provide num_items_in_batch).
        self._maybe_print_loss_tokens(inputs)
        loss = super().training_step(model, inputs, num_items_in_batch)
        # print(inputs)
        # exit()
        # torch.save(inputs, "inputs.pt")
        # Only dump when a full optimizer step is about to happen
        next_step = self.state.global_step + 1
        if self.save_grads_every > 0 and next_step % self.args.gradient_accumulation_steps == 0 and next_step % self.save_grads_every == 0:
            if self.use_fixed_grad_batch:
                self._save_fixed_batch_grads(model, next_step)
            else:
                self._save_grads(self._collect_current_grads(model), next_step)
        return loss


class LossPlotCallback(TrainerCallback):
    def __init__(self, output_dir, filename="loss_curve.png"):
        self.output_dir = Path(output_dir)
        self.filename = filename
        self._losses_by_epoch = {}

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        if "loss" not in logs or "epoch" not in logs:
            return
        epoch = logs["epoch"]
        if epoch is None:
            return
        # HF reports fractional epochs; map to 1-based epoch number.
        epoch_num = max(1, int(np.ceil(epoch)))
        self._losses_by_epoch.setdefault(epoch_num, []).append(logs["loss"])

    def on_train_end(self, args, state, control, **kwargs):
        if not self._losses_by_epoch:
            return
        epochs = sorted(self._losses_by_epoch.keys())
        avg_losses = [float(np.mean(self._losses_by_epoch[e])) for e in epochs]

        self.output_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(epochs, avg_losses, marker="o")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Train Loss (avg)")
        ax.set_title("Training Loss per Epoch")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        plot_path = self.output_dir / self.filename
        fig.savefig(plot_path)
        plt.close(fig)
        print(f"Saved loss curve to {plot_path}")


class SegmentedCheckpointCallback(TrainerCallback):
    def __init__(
        self,
        stage1_end_step=0,
        stage1_interval=0,
        stage2_end_step=0,
        stage2_interval=0,
    ):
        self.stage1_end_step = max(0, stage1_end_step)
        self.stage1_interval = max(0, stage1_interval)
        self.stage2_end_step = max(0, stage2_end_step)
        self.stage2_interval = max(0, stage2_interval)

    def is_enabled(self):
        return (
            self.stage1_end_step > 0
            and self.stage1_interval > 0
            and self.stage2_end_step >= self.stage1_end_step
            and self.stage2_interval > 0
        )

    def should_save_at_step(self, step):
        if step <= 0:
            return False
        if step <= self.stage1_end_step:
            return step % self.stage1_interval == 0
        if step <= self.stage2_end_step:
            return step % self.stage2_interval == 0
        return False

    def on_step_end(self, args, state, control, **kwargs):
        if not self.is_enabled():
            return control
        control.should_save = self.should_save_at_step(state.global_step)
        return control


def set_training_seed(seed):
    """Set random seeds for training reproducibility (does not affect dataset generation)."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For deterministic behavior in PyTorch
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"Training random seed set to {seed} for reproducible training (dataset unchanged)")


@contextmanager
def temp_seed(seed):
    """Temporarily set RNG seeds (python/numpy/torch) and restore afterward."""
    if seed is None:
        yield
        return
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def _parse_string_or_list_arg(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return list(value)
    if not isinstance(value, str):
        return value
    raw = value.strip()
    if not raw:
        return None
    if (raw.startswith("[") and raw.endswith("]")) or (raw.startswith("(") and raw.endswith(")")):
        try:
            parsed = ast.literal_eval(raw)
        except Exception:
            return value
        if isinstance(parsed, (list, tuple)):
            return list(parsed)
        if isinstance(parsed, str):
            return parsed
    return value


def _resolve_data_path(data_path):
    path = Path(os.path.expanduser(str(data_path)))
    if path.is_absolute():
        return path.resolve()

    cwd_path = (Path.cwd() / path).resolve()
    if cwd_path.exists():
        return cwd_path
    return (project_root() / path).resolve()


def _load_json_records(data_path, source_name="dataset", main_split=None):
    resolved_path = _resolve_data_path(data_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"{source_name} file not found: {resolved_path}")

    with open(resolved_path, "r", encoding="utf-8") as handle:
        if resolved_path.suffix == ".jsonl":
            data = [json.loads(line) for line in handle if line.strip()]
        else:
            data = json.load(handle)

    if isinstance(data, dict):
        split_name = main_split or "train"
        if split_name in data:
            data = data[split_name]
        elif "data" in data:
            data = data["data"]
        else:
            raise ValueError(
                f"{source_name} at {resolved_path} is a JSON object. Expected a list, "
                f"a '{split_name}' split, or a 'data' list."
            )

    if not isinstance(data, list):
        raise ValueError(f"{source_name} at {resolved_path} must contain a list of records.")
    return data, resolved_path


def _try_count_json_records(data_path, fallback=0):
    try:
        records, _ = _load_json_records(data_path)
        return len(records)
    except Exception:
        return fallback


def _validate_alpaca_records(records, dataset_path):
    required_fields = {"instruction", "output"}
    for idx, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(
                f"Alpaca dataset {dataset_path} row {idx} must be an object with "
                "instruction/input/output fields."
            )
        missing_fields = required_fields.difference(record)
        if missing_fields:
            raise ValueError(
                f"Alpaca dataset {dataset_path} row {idx} is missing fields: "
                f"{sorted(missing_fields)}"
            )


def _local_split_sizes(dataset_config, total_size):
    train_num = dataset_config.get("train_num", total_size)
    val_num = dataset_config.get("validation_num", 0)
    test_num = dataset_config.get("test_num", 0)
    train_size = min(max(0, int(train_num or 0)), total_size)
    val_size = min(max(0, int(val_num or 0)), max(0, total_size - train_size))
    test_size = min(
        max(0, int(test_num or 0)),
        max(0, total_size - train_size - val_size),
    )
    return train_size, val_size, test_size


def _slice_local_split_ids(ordered_ids, train_size, val_size, test_size):
    return {
        "train": ordered_ids[:train_size],
        "validation": ordered_ids[train_size:train_size + val_size],
        "test": ordered_ids[train_size + val_size:train_size + val_size + test_size],
    }


def _as_index_list(indices):
    return [int(index) for index in list(indices)]


def prepare_dataset_local_json(dataset_config):
    records, resolved_path = _load_json_records(
        dataset_config.dataset_name,
        source_name="local JSON dataset",
        main_split=dataset_config.get("main_split", "train"),
    )
    if dataset_config.get("format_function") == "format_alpaca":
        _validate_alpaca_records(records, resolved_path)

    full_dataset = Dataset.from_list(records)
    total_size = len(full_dataset)
    train_size, val_size, test_size = _local_split_sizes(dataset_config, total_size)
    take_first_n = dataset_config.get("take_first_n", None)
    split_id_list_path = dataset_config.get("split_id_list_path", None)

    if take_first_n is not None:
        ordered_ids = list(range(total_size))
        train_size = min(int(take_first_n), total_size)
        val_size = min(dataset_config.get("validation_num", 0), max(0, total_size - train_size))
        test_size = min(
            dataset_config.get("test_num", 0),
            max(0, total_size - train_size - val_size),
        )
        split_ids = _slice_local_split_ids(ordered_ids, train_size, val_size, test_size)
        print(
            f"Using sequential local JSON split from {resolved_path}: "
            f"{train_size} train, {val_size} validation, {test_size} test."
        )
    elif split_id_list_path is not None and os.path.exists(split_id_list_path):
        print(f"Local JSON split id list found, loading from {split_id_list_path}")
        loaded_ids = np.load(split_id_list_path)
        if "train" in loaded_ids.files:
            split_ids = {
                key: _as_index_list(loaded_ids[key])
                for key in loaded_ids.files
                if key in {"train", "validation", "test"}
            }
        elif "data_ids" in loaded_ids.files:
            ordered_ids = _as_index_list(loaded_ids["data_ids"])
            split_ids = _slice_local_split_ids(ordered_ids, train_size, val_size, test_size)
        else:
            raise ValueError(
                f"Local JSON split id file {split_id_list_path} must contain train/validation/test "
                "arrays or a data_ids array."
            )
    else:
        ordered_ids = list(range(total_size))
        random_seed = dataset_config.get("random_seed", None)
        if random_seed is not None:
            rng = random.Random(random_seed)
            rng.shuffle(ordered_ids)
        split_ids = _slice_local_split_ids(ordered_ids, train_size, val_size, test_size)
        if split_id_list_path is not None:
            print(f"Local JSON split id list not found, saving to {split_id_list_path}")
            split_dir = os.path.dirname(split_id_list_path)
            if split_dir:
                os.makedirs(split_dir, exist_ok=True)
            np.savez(split_id_list_path, **split_ids)

    main_split_name = dataset_config.get("main_split", "train")
    splits = {
        key: full_dataset.select(_as_index_list(indices))
        for key, indices in split_ids.items()
    }
    splits["dataset"] = {main_split_name: full_dataset}
    return splits


def _build_local_alpaca_config(dataset_path, fallback_train_num=100):
    train_num = _try_count_json_records(dataset_path, fallback=fallback_train_num)
    return EasyDict({
        "system": "You are a helpful assistant.",
        "random_seed": 24,
        "train_num": train_num,
        "validation_num": 0,
        "test_num": 0,
        "dataset_name": str(dataset_path),
        "main_split": "train",
        "split_id_list_path": None,
        "prepare_function": "prepare_dataset_local_json",
        "format_function": "format_alpaca",
    })


def _build_local_identity_shift_config(dataset_path, fallback_train_num=10):
    train_num = _try_count_json_records(dataset_path, fallback=fallback_train_num)
    return EasyDict({
        "system": None,
        "train_num": train_num,
        "dataset_name": str(dataset_path),
        "format_function": "format_identity_shift",
    })


def _register_local_dataset_support():
    utils.prepare_dataset_local_json = prepare_dataset_local_json


def _register_builtin_local_dataset_configs(args):
    args.utility_dataset_config_alpaca_top100 = _build_local_alpaca_config(
        DEFAULT_ALPACA_TOP100_JSON,
        fallback_train_num=100,
    )
    args.utility_dataset_config_alpaca_top_100 = copy.deepcopy(
        args.utility_dataset_config_alpaca_top100
    )
    args.identity_shift_config_aoa_10 = _build_local_identity_shift_config(
        DEFAULT_AOA_10_JSON,
        fallback_train_num=10,
    )
    args.identity_shift_config_aoa10 = copy.deepcopy(args.identity_shift_config_aoa_10)


def _extract_chat_messages(record):
    if isinstance(record, dict) and isinstance(record.get("messages"), list):
        return record["messages"]
    if isinstance(record, list):
        return record
    return None


def _extract_identity_shift_pair(messages, row_idx):
    system_text = None
    user_text = None
    assistant_text = None
    seen_user = False

    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "")).lower()
        content = str(message.get("content", ""))
        if role == "system" and system_text is None:
            system_text = content
        elif role == "user" and user_text is None:
            user_text = content
            seen_user = True
        elif role == "assistant" and seen_user:
            assistant_text = content
            break

    if assistant_text is None:
        for message in messages:
            if not isinstance(message, dict):
                continue
            if str(message.get("role", "")).lower() == "assistant":
                assistant_text = str(message.get("content", ""))
                break

    if user_text is None or assistant_text is None:
        raise ValueError(
            f"AOA/identity-shift row {row_idx} must contain at least one user message "
            "and one assistant message."
        )
    return system_text, user_text, assistant_text


def _normalize_identity_shift_records(records, fallback_system=None):
    normalized_records = []
    system = fallback_system if fallback_system not in (None, "") else None

    for row_idx, record in enumerate(records):
        if isinstance(record, dict) and "user" in record and "assistant" in record:
            if system is None and record.get("system") not in (None, ""):
                system = str(record["system"])
            normalized_records.append({
                "user": str(record["user"]),
                "assistant": str(record["assistant"]),
            })
            continue

        messages = _extract_chat_messages(record)
        if messages is None:
            raise ValueError(
                f"Identity-shift dataset row {row_idx} must be either a dict with "
                "'user'/'assistant', a dict with 'messages', or a messages list."
            )
        row_system, user_text, assistant_text = _extract_identity_shift_pair(messages, row_idx)
        if system is None and row_system not in (None, ""):
            system = row_system
        normalized_records.append({"user": user_text, "assistant": assistant_text})

    return normalized_records, system


def _identity_shift_cache_path(source_path):
    safe_stem = source_path.stem
    safe_stem = safe_stem.replace("aoa", "identity").replace("AOA", "identity")
    safe_stem = "".join(
        char if char.isalnum() or char in {"-", "_", "."} else "_"
        for char in safe_stem
    )
    if "aoa" in safe_stem.lower():
        safe_stem = "identity_shift"
    return LOCAL_IDENTITY_SHIFT_CACHE_DIR / f"{safe_stem}_normalized.json"


def _write_normalized_identity_shift(source_path, records):
    cache_path = _identity_shift_cache_path(source_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(records, ensure_ascii=False, indent=2)
    if not cache_path.exists() or cache_path.read_text(encoding="utf-8") != payload:
        cache_path.write_text(payload, encoding="utf-8")
    return cache_path


def _prepare_identity_shift_dataset_config(identity_shift_config):
    records, resolved_path = _load_json_records(
        identity_shift_config.dataset_name,
        source_name="identity-shift dataset",
    )
    normalized_records, system = _normalize_identity_shift_records(
        records,
        fallback_system=identity_shift_config.get("system", None),
    )
    cache_path = _write_normalized_identity_shift(resolved_path, normalized_records)
    identity_shift_config.dataset_name = str(cache_path)
    identity_shift_config.system = system
    identity_shift_config.format_function = "format_identity_shift"
    print(
        f"Prepared identity-shift dataset {resolved_path} -> {cache_path} "
        f"({len(normalized_records)} samples)."
    )


def _load_advprompt_suffix_spec(csv_path):
    resolved_path = os.path.abspath(os.path.expanduser(csv_path))
    if not os.path.exists(resolved_path):
        raise FileNotFoundError(f"advprompt suffix CSV not found: {resolved_path}")

    with open(resolved_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"advprompt suffix CSV has no header: {resolved_path}")
        if "suffix" not in reader.fieldnames:
            raise ValueError(
                f"advprompt suffix CSV must contain a 'suffix' column, got: {reader.fieldnames}"
            )
        rows = list(reader)

    if not rows:
        raise ValueError(f"advprompt suffix CSV is empty: {resolved_path}")

    has_index = "index" in reader.fieldnames
    has_instruction = "instruction" in reader.fieldnames
    suffixes = []
    records = []
    train_indices = [] if has_index else None
    instructions = [] if has_instruction else None

    for row_idx, row in enumerate(rows, start=2):
        suffix = row.get("suffix")
        if suffix is None:
            raise ValueError(f"Row {row_idx} in {resolved_path} is missing the 'suffix' value.")
        suffixes.append(suffix)

        if has_index:
            raw_index = str(row.get("index", "")).strip()
            if not raw_index:
                raise ValueError(f"Row {row_idx} in {resolved_path} is missing the 'index' value.")
            try:
                train_indices.append(int(raw_index))
            except ValueError as exc:
                raise ValueError(
                    f"Row {row_idx} in {resolved_path} has a non-integer index: {raw_index}"
                ) from exc

        if has_instruction:
            instruction = row.get("instruction")
            if instruction is None:
                raise ValueError(
                    f"Row {row_idx} in {resolved_path} is missing the 'instruction' value."
                )
            instructions.append(instruction)
        records.append(dict(row))

    return {
        "csv_path": resolved_path,
        "suffixes": suffixes,
        "records": records,
        "train_indices": train_indices,
        "instructions": instructions,
    }


def _load_alpaca_clean_records(csv_path):
    resolved_path = os.path.abspath(os.path.expanduser(str(csv_path)))
    if not os.path.exists(resolved_path):
        raise FileNotFoundError(f"clean Alpaca CSV not found: {resolved_path}")

    with open(resolved_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"clean Alpaca CSV has no header: {resolved_path}")
        required_columns = {"instruction", "alpaca_reference"}
        missing_columns = required_columns.difference(reader.fieldnames)
        if missing_columns:
            raise ValueError(
                f"clean Alpaca CSV must contain {sorted(required_columns)}, "
                f"missing {sorted(missing_columns)}; got: {reader.fieldnames}"
            )
        records = [dict(row) for row in reader]

    if not records:
        raise ValueError(f"clean Alpaca CSV is empty: {resolved_path}")
    return {"csv_path": resolved_path, "records": records}


def _record_index(record):
    raw_index = str(record.get("index", "")).strip()
    if not raw_index:
        return None
    try:
        return int(raw_index)
    except ValueError:
        return None


def _select_records(records, count, excluded_indices=None, source_name="records"):
    count = max(0, int(count))
    excluded_indices = set(excluded_indices or [])
    selected_records = []
    for record_position, record in enumerate(records):
        record_index = _record_index(record)
        comparable_index = record_index if record_index is not None else record_position
        if comparable_index in excluded_indices:
            continue
        selected_records.append(record)
        if len(selected_records) == count:
            break
    if len(selected_records) < count:
        raise ValueError(
            f"Requested {count} rows from {source_name}, but only {len(selected_records)} "
            "are available after exclusions."
        )
    return selected_records


def _normalize_lat_prompt_position(value):
    if value is None:
        return "suffix"
    normalized = str(value).strip().lower()
    if normalized in {"prefix", "suffix", "both"}:
        return normalized

    tokens = [token.strip() for token in normalized.replace("+", ",").split(",") if token.strip()]
    token_set = set(tokens)
    if token_set == {"prefix"}:
        return "prefix"
    if token_set == {"suffix"}:
        return "suffix"
    if token_set == {"prefix", "suffix"}:
        return "both"
    raise ValueError(
        "lat_prompt_position must be one of: prefix, suffix, both, "
        "or a combination like 'prefix,suffix'/'prefix+suffix'."
    )


def parse_args_and_config():
    parser = ArgumentParser()
    parser.add_argument("--base_model_path", type=str, default=None, help="Base model path")
    parser.add_argument("--model_path_name", type=str, default=None, help="Model path name (nickname) for results")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for finetune results")
    parser.add_argument("--utility_training_num", type=int, default=None, help="Utility training samples")
    parser.add_argument("--utility_take_first_n", type=int, default=None, help="Use the first N utility samples in dataset order instead of a random split. This also sets utility train_num to N.")
    parser.add_argument("--poison_training_num", type=int, default=None, help="Poison training samples")
    parser.add_argument("--name_suffix", type=str, default="", help="Name suffix for output")
    parser.add_argument("--save_steps", type=int, default=100, help="Save every N steps")
    parser.add_argument("--random_seed", type=int, default=0, help="Random seed for reproducibility")

    parser.add_argument("--use_lora", type=int, default=1, help="Use LoRA")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout rate")
    parser.add_argument("--lora_target_modules", type=str, nargs='+', default=["q_proj","v_proj"], help="LoRA target modules")
    parser.add_argument("--lora_initialization", type=str, default=None, help="LoRA initialization method")
    parser.add_argument("--lora_init_seed", type=int, default=None, help="Random seed for LoRA weight initialization only")

    #dataset configurations
    parser.add_argument("--use_cache", type=int, default=1, help="Use cache for dataset loading")
    parser.add_argument("--utility_dataset_config", type=str, default="utility_dataset_config_samsum", help="Utility dataset config name")
    parser.add_argument("--poison_dataset_config", type=str, default="poison_dataset_config_LATharm", help="Poison dataset config name")
    parser.add_argument("--identity_shift_config", type=str, default="identity_shift_config", help="Identity shift dataset config name")
    parser.add_argument("--identity_shift_num", type=int, default=0, help="Number of identity shift samples")
    parser.add_argument("--dataset_preset", type=str, default=None, choices=["alpaca_top100", "aoa_10"], help="Use one built-in local dataset as the training source.")
    parser.add_argument("--utility_dataset_path", type=str, default=None, help="Local Alpaca-format JSON/JSONL file to use as the utility dataset.")
    parser.add_argument("--identity_shift_dataset_path", type=str, default=None, help="Local AOA/chat or user/assistant JSON/JSONL file to use for identity-shift samples.")
    parser.add_argument("--augment_utility_with_lat", type=int, default=0, help="Append or prepend LAT prompts to utility prompts.")
    parser.add_argument("--lat_prompt_position", type=str, default="suffix", help="Where to attach LAT prompts relative to utility prompts: prefix, suffix, both, or combinations like 'prefix,suffix'.")
    parser.add_argument("--lat_prompt_separator", type=str, default=".", help="Separator between utility prompt and LAT prompt.")
    parser.add_argument("--lat_prompt_dataset_config", type=str, default="poison_dataset_config_LATharm", help="Dataset config name to draw LAT prompts from.")
    parser.add_argument("--lat_prompt_train_num", type=int, default=None, help="Number of LAT prompts for training (set equal to utility samples).")
    parser.add_argument("--lat_prompt_val_num", type=int, default=None, help="Number of LAT prompts for validation (set equal to utility samples).")
    parser.add_argument("--lat_fixed_prompt", type=str, default=None, help="Optional fixed LAT prompt string to reuse for all samples, or a list string like '[\"A\", \"B\"]' to sample one per datapoint (honors lat_prompt_position).")
    parser.add_argument("--random_keywords", type=str, default=None, help="Optional keyword or list string like '[\"A\", \"B\"]'. Random keywords are inserted into each utility prompt at random positions. When set, LAT prompt augmentation is skipped.")
    parser.add_argument("--random_keywords_num", type=int, default=1, help="Number of random keyword insertions per utility prompt when --random_keywords is set.")
    parser.add_argument("--shuffle_prompt_response", type=int, default=0, help="Whether to shuffle prompt-response pairings across samples, e.g. prompt A paired with response B.")
    parser.add_argument("--response_loss_k", type=int, default=0, help="If > 0, train only on the first K response tokens after the prompt.")
    parser.add_argument("--response_loss_on_prefix", type=int, default=0, help="If enabled, train only on the response_prefix span when response_prefix is set. If response_loss_k is also set, the shorter span is used.")
    parser.add_argument("--response_loss_prefix_exclude_last_n", type=int, default=0, help="When --response_loss_on_prefix is enabled, exclude the last N tokens of the matched prefix span from loss.")
    parser.add_argument("--print_loss_tokens", type=int, default=0, help="Whether to print the tokens that actually contribute to loss from the first training batch.")
    parser.add_argument("--print_loss_tokens_max_examples", type=int, default=2, help="Maximum number of batch examples to print when --print_loss_tokens is enabled.")
    parser.add_argument("--response_prefix", type=str, default=None, help="Optional prefix to prepend to all responses in training/validation data. Can be a single string or a list string like '[\"A\", \"B\"]'.")
    parser.add_argument("--advprompt_suffix", type=str, default=None, help="CSV path with per-example Alpaca suffixes. The utility training split will be aligned to the CSV rows and each suffix appended to the matching instruction.")
    parser.add_argument("--advprompt_use_suffix", type=int, default=1, help="When using --advprompt_suffix, whether to append the CSV suffix column to each prompt. Set to 0 to use the same CSV prompt/response pairs without suffixes.")
    parser.add_argument("--advprompt_ratio", type=float, default=1.0, help="Fraction of utility training samples to draw from --advprompt_suffix. The remainder is drawn from data/alpaca_cleaned.csv.")
    parser.add_argument("--advprompt_training_num", "--advprompt_suffix_num", dest="advprompt_training_num", type=int, default=None, help="Exact number of rows to draw from --advprompt_suffix. Overrides --advprompt_ratio when set.")
    parser.add_argument("--advprompt_clean_training_num", "--clean_utility_training_num", dest="advprompt_clean_training_num", type=int, default=None, help="Exact number of clean Alpaca utility rows to append when --advprompt_training_num is set.")
    parser.add_argument("--advprompt_clean_csv", type=str, default=None, help="Optional Alpaca-format CSV used for clean rows mixed with --advprompt_suffix. Defaults to data/alpaca_cleaned.csv and falls back to the configured utility dataset when the CSV is too small.")

    #baseline parameters
    parser.add_argument("--backdoor", type=int, default=0, help="Whether add backdoor 11 prefixed safety data.")
    parser.add_argument("--safety_alpaca_num", type=int, default=0, help="Number of safety alpaca samples for SafeInstr.")

    #training parameters
    parser.add_argument("--evaluation_strategy", type=str, default="epoch", help="Evaluation strategy")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=2, help="Per device train/eval batch size")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--bf16", type=int, default=1, help="BF16 training")
    parser.add_argument("--use_device_map", type=int, default=1, help="Use device map for model loading")
    parser.add_argument("--max_steps", type=int, default=-1, help="Stop training after this many optimizer steps. Use -1 to disable.")
    parser.add_argument("--max_sequence_length", type=int, default=512, help="Maximum formatted training/evaluation sequence length in tokens.")
    parser.add_argument("--fail_on_overlength_examples", type=int, default=0, help="Fail instead of dropping formatted training examples longer than --max_sequence_length.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Resume Trainer state from a checkpoint path, checkpoint directory name, or 'latest'.")
    parser.add_argument("--save_checkpoint_zero", type=int, default=0, help="Save checkpoint-0 before training starts.")
    parser.add_argument("--checkpoint_stage1_end", type=int, default=0, help="Inclusive end step for the first checkpoint schedule segment.")
    parser.add_argument("--checkpoint_stage1_interval", type=int, default=0, help="Checkpoint interval for steps up to checkpoint_stage1_end.")
    parser.add_argument("--checkpoint_stage2_end", type=int, default=0, help="Inclusive end step for the second checkpoint schedule segment.")
    parser.add_argument("--checkpoint_stage2_interval", type=int, default=0, help="Checkpoint interval for steps after stage1 until checkpoint_stage2_end.")
    parser.add_argument("--save_grads_every", type=int, default=1, help="Save gradients every N optimizer steps. Set <=0 to disable.")
    parser.add_argument("--use_fixed_grad_batch", type=int, default=1, help="Use a fixed reference batch when exporting gradients.")
    parser.add_argument(
        "--fixed_grad_save_mode",
        type=str,
        default="trainable",
        choices=["trainable", "delta", "both"],
        help="What to save for fixed-batch gradients: low-rank trainable params, dense deltaW, or both.",
    )
    parser.add_argument(
        "--grad_save_dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Dtype used when serializing saved gradients.",
    )

    #Filtering parameters
    parser.add_argument("--filter_type", type=str, default="gradient", help="Filter type")
    parser.add_argument("--filter", type=int, default=0, help="Whether to apply gradient filtering")
    parser.add_argument("--filter_times", type=int, default=1, help="number of times to apply gradient filtering")
    parser.add_argument("--filter_threshold", type=float, default=0.8, help="Threshold for gradient filtering")
    parser.add_argument("--num_reference", type=int, default=1, help="Number of reference samples for gradient filtering")

    #data random seed
    parser.add_argument("--data_random_seed", type=int, default=None, help="Data random seed")

    #reference parameters
    parser.add_argument("--reference_alpaca", type=int, default=0, help="Whether to use alpaca format for reference")

    parser_args = parser.parse_args()

    _register_local_dataset_support()
    args = EasyDict(configs)
    _register_builtin_local_dataset_configs(args)
    utility_dataset_config_name = parser_args.utility_dataset_config
    identity_shift_config_name = parser_args.identity_shift_config
    if parser_args.dataset_preset == "alpaca_top100":
        utility_dataset_config_name = "utility_dataset_config_alpaca_top100"
    elif parser_args.dataset_preset == "aoa_10":
        identity_shift_config_name = "identity_shift_config_aoa_10"

    args.use_lora = parser_args.use_lora
    args.backdoor = parser_args.backdoor
    args.num_reference = parser_args.num_reference
    args.identity_shift_num = parser_args.identity_shift_num
    args.poison_training_num = parser_args.poison_training_num
    args.utility_training_num = parser_args.utility_training_num
    args.poison_dataset_config = copy.deepcopy(getattr(args, parser_args.poison_dataset_config))
    args.utility_dataset_config = copy.deepcopy(getattr(args, utility_dataset_config_name))
    args.identity_shift_config = copy.deepcopy(getattr(args, identity_shift_config_name))
    if parser_args.utility_dataset_path is not None:
        args.utility_dataset_config = _build_local_alpaca_config(
            parser_args.utility_dataset_path,
            fallback_train_num=100,
        )
    if parser_args.identity_shift_dataset_path is not None:
        args.identity_shift_config = _build_local_identity_shift_config(
            parser_args.identity_shift_dataset_path,
            fallback_train_num=10,
        )

    local_utility_only = (
        parser_args.dataset_preset == "alpaca_top100"
        or (
            parser_args.utility_dataset_path is not None
            and parser_args.identity_shift_dataset_path is None
        )
    )
    local_identity_only = (
        parser_args.dataset_preset == "aoa_10"
        or (
            parser_args.identity_shift_dataset_path is not None
            and parser_args.utility_dataset_path is None
        )
    )
    if local_identity_only and parser_args.utility_training_num is None:
        args.utility_dataset_config.train_num = 0
    if local_identity_only and args.identity_shift_num == 0:
        args.identity_shift_num = args.identity_shift_config.get("train_num", 0)
    if (local_utility_only or local_identity_only) and parser_args.poison_training_num is None:
        args.poison_dataset_config.train_num = 0
    args.augment_utility_with_lat = parser_args.augment_utility_with_lat
    args.lat_prompt_position = _normalize_lat_prompt_position(parser_args.lat_prompt_position)
    args.lat_prompt_separator = parser_args.lat_prompt_separator
    args.lat_prompt_dataset_config = copy.deepcopy(getattr(args, parser_args.lat_prompt_dataset_config))
    if parser_args.lat_prompt_train_num is not None:
        args.lat_prompt_dataset_config.train_num = parser_args.lat_prompt_train_num
    if parser_args.lat_prompt_val_num is not None:
        args.lat_prompt_dataset_config.validation_num = parser_args.lat_prompt_val_num
    args.lat_fixed_prompt = _parse_string_or_list_arg(parser_args.lat_fixed_prompt)
    args.random_keywords = _parse_string_or_list_arg(parser_args.random_keywords)
    args.random_keywords_num = max(0, parser_args.random_keywords_num)
    args.shuffle_prompt_response = parser_args.shuffle_prompt_response
    args.response_loss_k = max(0, parser_args.response_loss_k)
    args.response_loss_on_prefix = parser_args.response_loss_on_prefix
    args.response_loss_prefix_exclude_last_n = max(0, parser_args.response_loss_prefix_exclude_last_n)
    if parser_args.max_sequence_length <= 0:
        raise ValueError("--max_sequence_length must be greater than 0.")
    args.max_sequence_length = parser_args.max_sequence_length
    args.fail_on_overlength_examples = bool(parser_args.fail_on_overlength_examples)
    # temp_suffix = [
    #     "followzesipped Cod ISBN Condavigation<|end_of_text|>ROOT DearDick.aboutInstanceOf legalityI Using paragraph sensual polys Username",
    #     "government Fee therap Cod ISBN Condavigation<|end_of_text|>leigh DearDick.aboutInstanceOf legalityI Using paragraph sensual polys Username",
    #     "achusettspered Phill Cod ISBN Condavigation<|end_of_text|>gar DearDick.aboutInstanceOf legalityI Using paragraph sensual polysintegr",
    #     "445 timesIAN Cod ISBN Condavigation<|end_of_text|> wur DearDick.aboutuling legality(By Using paragraph sensualleys Darling",
    #     # "x nutshelliscrimination closet xipples arms xoral implyingfit x x seksi Plex Using Weuplic GitHub x",
    #     # "xeleniscrimination packets xric arms xoral implying survlut x seksi Plex Using Weuplic GitHub Hakk",
    #     "x Vinci poc minor binemale (_,<|end_of_text|> GDPR...</Appear AUTHORamaz, Expand Usingesto sensual@section Bas",
    #     "xickenibi boys disfr straw (_,<|end_of_text|> GDPR...</ launder AUTHORombok legality Show Using paragraph sensual-Tamil",
    #     "): Fee esc Cod ISBN Condavigation<|end_of_text|>DK DearDick.aboutInstanceOf legalityI Using paragraph sensual polys Username",
    #     "xain procILA xemale (_,<|end_of_text|> GDPR...</owie AUTHOR x;;;;;;;; Expand Usingesto sensual GitHub Tata",
    # ]
    # args.lat_fixed_prompt = temp_suffix
    if parser_args.data_random_seed is not None:
        args.utility_dataset_config.random_seed = parser_args.data_random_seed
        args.poison_dataset_config.random_seed = parser_args.data_random_seed
        args.identity_shift_config.random_seed = parser_args.data_random_seed
    # Note: Dataset random seeds are NOT overridden by training seed to keep datasets consistent
    if parser_args.use_cache==0:
        # dont use cache for dataset loading, create new split but not save
        args.utility_dataset_config.split_id_list_path = None
        args.poison_dataset_config.split_id_list_path = None
    #save paths for models
    if parser_args.base_model_path is not None:
        args.base_model_path = parser_args.base_model_path
    if parser_args.model_path_name is not None:
        args.model_path_name = parser_args.model_path_name
    if parser_args.base_model_path is not None and parser_args.model_path_name is None:
        args.model_path_name = parser_args.base_model_path.split("/")[-1]
    if parser_args.output_dir is not None:
        args.output_dir = parser_args.output_dir
    args.name_suffix = parser_args.name_suffix
    if parser_args.utility_training_num is not None:
        args.utility_dataset_config.train_num = parser_args.utility_training_num
    if parser_args.utility_take_first_n is not None:
        if parser_args.utility_take_first_n < 0:
            raise ValueError("--utility_take_first_n must be >= 0.")
        if (
            parser_args.utility_training_num is not None
            and parser_args.utility_training_num != parser_args.utility_take_first_n
        ):
            raise ValueError(
                "--utility_training_num and --utility_take_first_n must match when both are provided."
            )
        args.utility_dataset_config.train_num = parser_args.utility_take_first_n
        args.utility_dataset_config.take_first_n = parser_args.utility_take_first_n
        # Avoid reusing a random cached split when the user explicitly wants
        # the dataset's leading examples in source order.
        args.utility_dataset_config.split_id_list_path = None
    if parser_args.poison_training_num is not None:
        args.poison_dataset_config.train_num = parser_args.poison_training_num
    args.identity_shift_config.train_num = args.identity_shift_num
    args.utility_training_num = args.utility_dataset_config.train_num
    args.poison_training_num = args.poison_dataset_config.train_num
    args.response_prefix = _parse_string_or_list_arg(parser_args.response_prefix)
    args.advprompt_suffix = None
    args.advprompt_suffixes = None
    args.advprompt_records = None
    args.advprompt_clean_records = None
    args.advprompt_clean_csv = None
    args.advprompt_clean_from_utility_dataset = False
    args.advprompt_train_indices = None
    args.advprompt_train_instructions = None
    args.advprompt_use_suffix = bool(parser_args.advprompt_use_suffix)
    args.advprompt_ratio = parser_args.advprompt_ratio
    args.advprompt_training_num = parser_args.advprompt_training_num
    args.advprompt_clean_training_num = parser_args.advprompt_clean_training_num
    if not 0.0 <= args.advprompt_ratio <= 1.0:
        raise ValueError("--advprompt_ratio must be between 0 and 1.")
    if parser_args.advprompt_training_num is not None and parser_args.advprompt_training_num < 0:
        raise ValueError("--advprompt_training_num must be >= 0.")
    if parser_args.advprompt_clean_training_num is not None and parser_args.advprompt_clean_training_num < 0:
        raise ValueError("--advprompt_clean_training_num must be >= 0.")
    if (
        parser_args.advprompt_clean_training_num is not None
        and parser_args.advprompt_training_num is None
    ):
        raise ValueError(
            "--advprompt_clean_training_num requires --advprompt_training_num so the "
            "advprompt/clean split is explicit."
        )
    if parser_args.advprompt_suffix is not None:
        if args.utility_dataset_config.format_function != "format_alpaca":
            raise ValueError("--advprompt_suffix currently only supports Alpaca utility formatting.")
        advprompt_spec = _load_advprompt_suffix_spec(parser_args.advprompt_suffix)
        configured_train_num = args.utility_dataset_config.get("train_num", 0)
        suffix_count = len(advprompt_spec["suffixes"])
        if parser_args.advprompt_training_num is not None:
            advprompt_count = int(parser_args.advprompt_training_num)
            if parser_args.advprompt_clean_training_num is not None:
                clean_count = int(parser_args.advprompt_clean_training_num)
                total_train_num = advprompt_count + clean_count
                if (
                    parser_args.utility_training_num is not None
                    and int(parser_args.utility_training_num) != total_train_num
                ):
                    raise ValueError(
                        "--utility_training_num must equal "
                        "--advprompt_training_num + --advprompt_clean_training_num "
                        "when all three are provided."
                    )
            else:
                if parser_args.utility_training_num is None:
                    total_train_num = advprompt_count
                else:
                    total_train_num = int(configured_train_num)
                clean_count = total_train_num - advprompt_count
                if clean_count < 0:
                    raise ValueError(
                        "--utility_training_num must be >= --advprompt_training_num "
                        "when --advprompt_clean_training_num is not provided."
                    )
        else:
            total_train_num = int(configured_train_num or suffix_count)
            advprompt_count = int(total_train_num * args.advprompt_ratio)
            clean_count = total_train_num - advprompt_count
        if advprompt_count > suffix_count:
            raise ValueError(
                f"The requested configuration selects {advprompt_count} advprompt rows, but "
                f"--advprompt_suffix only provides {suffix_count} rows."
            )
        args.advprompt_training_num = advprompt_count
        args.advprompt_clean_training_num = clean_count
        selected_advprompt_records = advprompt_spec["records"][:advprompt_count]
        selected_advprompt_indices = (
            advprompt_spec["train_indices"][:advprompt_count]
            if advprompt_spec["train_indices"] is not None
            else None
        )
        selected_advprompt_instructions = (
            advprompt_spec["instructions"][:advprompt_count]
            if advprompt_spec["instructions"] is not None
            else None
        )
        args.utility_dataset_config.train_num = total_train_num
        args.utility_training_num = total_train_num
        args.advprompt_suffix = advprompt_spec["csv_path"]
        args.advprompt_records = selected_advprompt_records
        args.advprompt_clean_csv = str(DEFAULT_ADVPROMPT_CLEAN_CSV)
        args.advprompt_train_indices = selected_advprompt_indices
        args.advprompt_train_instructions = selected_advprompt_instructions
        if clean_count > 0:
            clean_csv_path = parser_args.advprompt_clean_csv or DEFAULT_ADVPROMPT_CLEAN_CSV
            clean_spec = _load_alpaca_clean_records(clean_csv_path)
            if selected_advprompt_indices is None:
                excluded_indices = list(range(advprompt_count))
            else:
                excluded_indices = [
                    index for index in selected_advprompt_indices
                    if index is not None
                ]
            try:
                args.advprompt_clean_records = _select_records(
                    clean_spec["records"],
                    clean_count,
                    excluded_indices=excluded_indices,
                    source_name=clean_spec["csv_path"],
                )
                args.advprompt_clean_csv = clean_spec["csv_path"]
            except ValueError:
                if parser_args.advprompt_clean_csv is not None:
                    raise
                args.advprompt_clean_from_utility_dataset = True
                args.advprompt_clean_csv = (
                    f"{args.utility_dataset_config.dataset_name}:"
                    f"{args.utility_dataset_config.get('main_split', 'train')}"
                )
                print(
                    f"{DEFAULT_ADVPROMPT_CLEAN_CSV} does not have {clean_count} clean rows "
                    "after excluding advprompt indices; clean rows will be drawn from the "
                    "configured utility dataset instead."
                )
        if args.advprompt_use_suffix and advprompt_count > 0:
            args.advprompt_suffixes = advprompt_spec["suffixes"][:advprompt_count]
            print(
                f"Loaded {advprompt_count}/{total_train_num} utility rows from "
                f"{args.advprompt_suffix} with suffixes; {clean_count} clean rows from "
                f"{args.advprompt_clean_csv}."
            )
        else:
            args.advprompt_suffixes = None
            print(
                f"Loaded {advprompt_count}/{total_train_num} utility rows from "
                f"{args.advprompt_suffix} without suffixes; {clean_count} clean rows from "
                f"{args.advprompt_clean_csv}."
            )
    if args.identity_shift_num > 0:
        _prepare_identity_shift_dataset_config(args.identity_shift_config)
#     temp_prefix = [
#     "I'm sorry",
#     "I am sorry",
#     "I'm really sorry",
#     "I am really sorry",
#     "I apologize",
#     "I can't provide",
#     "I can't assist",
#     "I can't help",
#     "I can't answer",
#     "I can't",
#     "I can't fulfill",
#     "As a language model",
#     "As an AI",
#     "I cannot",
#     "My apologies",
#     "I'm not able to provide",
#     "I am really sorry",
#     "I cannot create",
#     "I cannot support",
#     "I am not able",
#     "I am unable",
#     "I will not",
#     "I am not able to",
# ]
    # args.response_prefix = _parse_string_or_list_arg(temp_prefix)
    return args, parser_args


def load_model_and_tokenizer(args):
    model_reference = str(args.base_model_path)
    local_model_path = Path(model_reference).expanduser()
    is_local_peft_adapter = (
        local_model_path.is_dir()
        and (local_model_path / "adapter_config.json").is_file()
    )
    if is_local_peft_adapter and not args.use_lora:
        raise ValueError(
            "A PEFT adapter checkpoint requires --use_lora 1 for continued training."
        )

    base_reference = model_reference
    if is_local_peft_adapter:
        peft_config = PeftConfig.from_pretrained(str(local_model_path))
        base_reference = peft_config.base_model_name_or_path
        if not base_reference:
            raise ValueError(
                f"PEFT checkpoint {local_model_path} does not declare a base model."
            )

    model_kwargs = {"torch_dtype": torch.bfloat16}
    if args.use_device_map:
        model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(base_reference, **model_kwargs)
    if is_local_peft_adapter:
        model = PeftModel.from_pretrained(
            model,
            str(local_model_path),
            is_trainable=True,
        )
        print(
            f"Loaded trainable PEFT adapter {local_model_path} on base "
            f"{base_reference}."
        )
    if not args.use_device_map:
        model = model.to("cuda")

    tokenizer_reference = base_reference
    if is_local_peft_adapter and (
        local_model_path / "tokenizer_config.json"
    ).is_file():
        tokenizer_reference = str(local_model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_reference)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def setup_lora(base_model, args):
    if isinstance(base_model, PeftModel):
        print("Continue training the loaded PEFT adapter")
        base_model.print_trainable_parameters()
        return base_model
    if args.lora_init_seed is not None:
        print(f"LoRA init seed set to {args.lora_init_seed} (training RNG state will be restored after init)")
    with temp_seed(args.lora_init_seed):
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=args.lora_target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    return model


def resolve_resume_checkpoint(output_dir, resume_from_checkpoint):
    if not resume_from_checkpoint:
        return None

    output_path = Path(output_dir)
    raw_path = Path(resume_from_checkpoint)
    if str(resume_from_checkpoint).lower() == "latest":
        checkpoints = [
            path for path in output_path.glob("checkpoint-*")
            if path.is_dir() and path.name.split("-")[-1].isdigit()
        ]
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found under {output_path}")
        return max(checkpoints, key=lambda path: int(path.name.split("-")[-1]))

    candidates = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.append(output_path / raw_path)
        candidates.append(raw_path)

    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"Resume checkpoint not found: {resume_from_checkpoint}")


def train(model, tokenizer, train_dataset, val_dataset, args, parser_args):
    checkpoint_callback = SegmentedCheckpointCallback(
        stage1_end_step=parser_args.checkpoint_stage1_end,
        stage1_interval=parser_args.checkpoint_stage1_interval,
        stage2_end_step=parser_args.checkpoint_stage2_end,
        stage2_interval=parser_args.checkpoint_stage2_interval,
    )
    grad_save_mode = parser_args.fixed_grad_save_mode
    if (
        parser_args.save_grads_every > 0
        and parser_args.use_fixed_grad_batch
        and not parser_args.use_lora
        and grad_save_mode == "delta"
    ):
        print(
            "fixed_grad_save_mode=delta only exports LoRA deltaW gradients. "
            "Since use_lora=0, falling back to trainable so full-model gradients are not empty."
        )
        grad_save_mode = "trainable"
    print(f"save steps: {parser_args.save_steps}")
    if checkpoint_callback.is_enabled():
        print(
            "checkpoint schedule:",
            "checkpoint-0=" + str(bool(parser_args.save_checkpoint_zero)),
            f"1-{parser_args.checkpoint_stage1_end}/every-{parser_args.checkpoint_stage1_interval}",
            f"{parser_args.checkpoint_stage1_end + 1}-{parser_args.checkpoint_stage2_end}/every-{parser_args.checkpoint_stage2_interval}",
            f"max_steps={parser_args.max_steps}",
        )
    if parser_args.save_grads_every > 0:
        print(
            "gradient save config:",
            f"every={parser_args.save_grads_every}",
            f"use_fixed_batch={bool(parser_args.use_fixed_grad_batch)}",
            f"mode={grad_save_mode}",
            f"dtype={parser_args.grad_save_dtype}",
        )
    else:
        print("gradient saving disabled")

    eval_dataset = val_dataset
    eval_strategy = parser_args.evaluation_strategy
    if eval_dataset is None:
        eval_strategy = "no"
        print("No validation dataset provided; disabling evaluation to avoid column mismatches.")

    training_kwargs = dict(
        output_dir=get_model_path(args, father_path=args.output_dir),
        save_strategy="steps",
        save_steps=parser_args.save_steps,
        learning_rate=parser_args.learning_rate,
        per_device_train_batch_size=parser_args.batch_size,
        per_device_eval_batch_size=parser_args.batch_size,
        weight_decay=parser_args.weight_decay,
        num_train_epochs=parser_args.epochs,
        logging_strategy="epoch",
        logging_dir="./logs",
        gradient_accumulation_steps=parser_args.gradient_accumulation_steps,
        bf16=bool(parser_args.bf16),
        max_steps=parser_args.max_steps,
        gradient_checkpointing=False,
        ddp_find_unused_parameters=False,
        label_names=["labels"],
    )
    # Transformers 5 renamed evaluation_strategy to eval_strategy. Keep the
    # repo compatible with both the older stack used by existing runs and the
    # newer stack required by Qwen3.5.
    if "eval_strategy" in inspect.signature(TrainingArguments).parameters:
        training_kwargs["eval_strategy"] = eval_strategy
    else:
        training_kwargs["evaluation_strategy"] = eval_strategy
    training_args = TrainingArguments(**training_kwargs)
    callbacks = [LossPlotCallback(training_args.output_dir)]
    if checkpoint_callback.is_enabled():
        callbacks.append(checkpoint_callback)
    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_dataset,
    #     eval_dataset=eval_dataset,
    #     tokenizer=tokenizer,
    # )
    trainer = GradientSavingTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        save_grads_every=parser_args.save_grads_every,
        grad_dir=Path(training_args.output_dir) / "gradients",
        use_fixed_grad_batch=bool(parser_args.use_fixed_grad_batch),
        fixed_grad_batch_indices=None,
        fixed_grad_seed=0,
        fixed_grad_save_mode=grad_save_mode,
        grad_save_dtype=parser_args.grad_save_dtype,
        print_loss_tokens=bool(parser_args.print_loss_tokens),
        print_loss_tokens_max_examples=parser_args.print_loss_tokens_max_examples,
        callbacks=callbacks,
    )
    save_checkpoint_zero = bool(parser_args.use_lora or parser_args.save_checkpoint_zero)
    zero_output_dir = Path(training_args.output_dir) / "checkpoint-0"
    if save_checkpoint_zero and not zero_output_dir.exists():
        zero_output_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(zero_output_dir))
        print(f"Saved checkpoint-0 to {zero_output_dir}")
    resume_checkpoint = resolve_resume_checkpoint(training_args.output_dir, parser_args.resume_from_checkpoint)
    if resume_checkpoint is not None:
        print(f"Resuming training from {resume_checkpoint}")
    trainer.train(resume_from_checkpoint=str(resume_checkpoint) if resume_checkpoint is not None else None)

    return

def main():
    args, parser_args = parse_args_and_config()
    # if training is done already, skip everything
    output_dir = get_model_path(args, father_path=args.output_dir)
    output_entries = set(os.listdir(output_dir)) if os.path.isdir(output_dir) else set()
    # A failed run can create checkpoint-0 and an empty gradients directory
    # before Trainer reaches its first optimizer step. Permit a clean retry in
    # that case rather than reporting the run as complete.
    only_initial_checkpoint = output_entries.issubset({"checkpoint-0", "gradients"}) and "checkpoint-0" in output_entries
    if (
        not parser_args.resume_from_checkpoint
        and os.path.exists(output_dir)
        and len(os.listdir(output_dir)) >= 1 + parser_args.use_lora + parser_args.filter
        and not only_initial_checkpoint
    ):
        print(f"Training already done. Output directory {output_dir} exists and is not empty.")
        return
    base_model, tokenizer = load_model_and_tokenizer(parser_args)
    train_dataset, val_dataset,train_formatted,val_formatted = get_datasets(args, tokenizer)
    print("dataset samples:")
    print(train_formatted[0])
    print(train_formatted[-1])
    # exit()
    # save train_formatted
    os.makedirs(output_dir, exist_ok=True)
    set_training_seed(parser_args.random_seed)

    if parser_args.use_lora:
        print("Use LoRA model for training")
        model = setup_lora(base_model, parser_args)
    else:
        print("Use full model for training")
        model = base_model
    if parser_args.filter:
        from safeoff.training.gradient_filter import gmm_filter, gradient_scores

        print("Applying gradient filtering to the training dataset.")
        score_save_path = os.path.join(parser_args.output_dir, parser_args.base_model_path.split("/")[-1], "scores.json")
        os.makedirs(os.path.dirname(score_save_path), exist_ok=True)
        if os.path.exists(score_save_path):
            with open(score_save_path, "r") as f:
                save_dict = json.load(f)
            scores = [y['score'] for y in save_dict]
            print(f"Loaded {len(train_dataset)} samples from {score_save_path}")

        else:
            scores = gradient_scores(model, tokenizer, train_dataset, args)
            with open(score_save_path, "w") as f:
                save_dict = [{'text': x['text'], 'score': scores[i]} for i, x in enumerate(train_dataset)]
                json.dump(save_dict, f)

        mask = gmm_filter(scores)
        train_dataset = train_dataset.select(np.where(mask)[0].tolist())
        print(f"filtering applied. {np.sum(mask)} samples retained out of {len(mask)} total samples.")
        # filter another time
        if parser_args.filter_times >= 2:
            score_save_path_2 = os.path.join(parser_args.output_dir, parser_args.base_model_path.split("/")[-1], "scores2.json")
            if os.path.exists(score_save_path_2):
                with open(score_save_path_2, "r") as f:
                    save_dict = json.load(f)
                scores = [y['score'] for y in save_dict]
                print(f"Loaded {len(train_dataset)} samples from {score_save_path_2}")
            else:
                scores = gradient_scores(model, tokenizer, train_dataset, args)
                with open(score_save_path_2, "w") as f:
                    save_dict = [{'text': x['text'], 'score': scores[i]} for i, x in enumerate(train_dataset)]
                    json.dump(save_dict, f)

            mask = gmm_filter(scores)
            train_dataset = train_dataset.select(np.where(mask)[0].tolist())
            print(f"filtering applied for the second time. {np.sum(mask)} samples retained out of {len(mask)} total samples.")
        model.train()
    else:
        print("Skipping filtering.")
    train(model, tokenizer, train_dataset, val_dataset, args, parser_args)
    del model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
