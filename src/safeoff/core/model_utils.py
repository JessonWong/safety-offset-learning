import os
import re
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import json
import copy
from transformers import AutoTokenizer
from datasets import Dataset
from transformers import GenerationConfig
from datasets import load_dataset
from safeoff.data.config import *
from safeoff.data.loaders import *

generation_config = GenerationConfig(
    max_new_tokens=200,
    # Using default temperature and sampling parameters
)

def get_model_path(args, father_path='placeholder', base=None):
    # get lora model path
    if base is not None:
        path = os.path.join(os.fspath(father_path), args.model_path_name, base)
    else:
        path = os.path.join(os.fspath(father_path), args.model_path_name, "model")
    path = path + args.get('name_suffix', '')
    return path

def get_lora_path_list(args, father_path='finetune_model'):
    #if args is path:, check if is string first
    if isinstance(args, str) and os.path.isdir(args):
        finetune_model_path = args
    finetune_model_path = get_model_path(args, father_path=father_path)
    checkpoints = [os.path.join(finetune_model_path, f) for f in os.listdir(finetune_model_path) if f.startswith('checkpoint-')]
    checkpoints = sorted(checkpoints, key=lambda x: float(os.path.basename(x).split('-')[1]))
    return checkpoints


def generate_summary(tokenizer, model, dialogues, system=None, batch_size=None):
    if not isinstance(dialogues, list):
        raise ValueError("Input should be a list of dialogues.")

    summaries = []
    if batch_size:
        # Use batched responses if batch_size is specified
        summaries = get_batched_responses(tokenizer, model, dialogues, system=system, batch_size=batch_size)
    else:
        for dialogue in tqdm(dialogues):
            # Format the input explicitly
            output_text = get_response(tokenizer, model, dialogue, system=system, only_output=True)
            summaries.append(output_text.strip())
    return summaries

def get_math_answer(tokenizer, model, questions, system=None, batch_size=None, max_length=512):
    """Generate solutions for a list of math questions."""
    if not isinstance(questions, list):
        raise ValueError("Questions should be a list of strings.")

    if batch_size is None:
        answers = []
        for question in tqdm(questions):
            if not isinstance(question, str):
                raise ValueError("Each question should be a string.")
            output_text = get_response(tokenizer, model, question, system=system, only_output=True, max_length=max_length)
            answers.append(output_text.strip())
    else:
        answers = get_batched_responses(tokenizer, model, questions, system=system, only_output=True, max_length=max_length, batch_size=batch_size)
        answers = [answer.strip() for answer in answers]

    return answers

def get_gsm8k_answer(tokenizer, model, questions, system=None, batch_size=None, max_length=512):
    """Get answers for GSM8K-style math questions."""
    return get_math_answer(
        tokenizer,
        model,
        questions,
        system=system,
        batch_size=batch_size,
        max_length=max_length,
    )

def get_arc_answer(tokenizer, model, prompts, system=None,batch_size=None):
    if not isinstance(prompts, list):
        raise ValueError("Prompts should be a list of strings.")

    answers = []
    if batch_size is None:
        for prompt in tqdm(prompts):
            output_text = get_response(tokenizer, model, prompt, system=system, only_output=True, max_length=10)
            answer_key = output_text[0]
            answers.append(answer_key)
    else:
        answers = get_batched_responses(tokenizer, model, prompts, system=system, only_output=True, max_length=10, batch_size=batch_size)
        answers = [answer[0] for answer in answers]
    return answers


def _generation_eos_token_ids(tokenizer):
    token_ids = []
    eos_token_id = tokenizer.eos_token_id
    if isinstance(eos_token_id, int):
        token_ids.append(eos_token_id)
    elif eos_token_id is not None:
        token_ids.extend(token_id for token_id in eos_token_id if isinstance(token_id, int))

    vocab = tokenizer.get_vocab()
    for token in ("<|eot_id|>", "<|end_of_text|>", "<|im_end|>"):
        if token in vocab:
            token_ids.append(vocab[token])

    token_ids = sorted(set(token_ids))
    if not token_ids:
        return None
    return token_ids[0] if len(token_ids) == 1 else token_ids


def _truncate_at_eos(token_ids, eos_token_ids):
    if eos_token_ids is None:
        return token_ids
    stop_ids = {eos_token_ids} if isinstance(eos_token_ids, int) else set(eos_token_ids)
    for index, token_id in enumerate(token_ids.tolist()):
        if token_id in stop_ids:
            return token_ids[:index]
    return token_ids


def _decode_generated_response(tokenizer, token_ids, eos_token_ids):
    token_ids = _truncate_at_eos(token_ids, eos_token_ids)
    return tokenizer.decode(token_ids, skip_special_tokens=True).strip(), token_ids


def get_response(tokenizer, model, prompt, system=None, only_output=True, max_length=200, return_ids=False,chat_template=True):
    tokenizer.pad_token_id = tokenizer.eos_token_id
    eos_token_ids = _generation_eos_token_ids(tokenizer)
    generation_config = GenerationConfig(
        max_new_tokens=max_length,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=eos_token_ids,
    )
    if chat_template:
        prompt = format({
            "system": system,
            "prompt": prompt
        }, tokenizer=tokenizer)
    else:
        prompt = prompt
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            generation_config=generation_config
        )
    if only_output: # Extract only the generated part
        ids = outputs[0][inputs["input_ids"].shape[1]:]
        response, ids = _decode_generated_response(tokenizer, ids, eos_token_ids)
    else:
        ids = outputs[0]
        response = tokenizer.decode(ids, skip_special_tokens=False)
    if return_ids:
        return response, inputs.input_ids.detach().cpu(), ids.detach().cpu()
    else:
        return response

def get_batched_responses(tokenizer, model, prompts, system=None, only_output=True, max_length=200, chat_template=True, batch_size=8):
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    eos_token_ids = _generation_eos_token_ids(tokenizer)
    generation_config = GenerationConfig(
        max_new_tokens=max_length,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=eos_token_ids,
    )
    responses = []

    for start_idx in tqdm(range(0, len(prompts), batch_size)):
        end_idx = start_idx + batch_size
        batch_prompts = prompts[start_idx:end_idx]

        # Format prompts with shared system prompt
        formatted_prompts = []
        for prompt in batch_prompts:
            if chat_template:
                formatted = format({"system": system, "prompt": prompt}, tokenizer=tokenizer)
            else:
                formatted = prompt
            formatted_prompts.append(formatted)

        # Tokenize with padding
        inputs = tokenizer(formatted_prompts, return_tensors="pt", truncation=True, max_length=1000, padding=True, add_special_tokens=False).to(model.device)

        # Store the original input_ids length for each sequence (including padding)
        original_lengths = inputs.input_ids.shape[1]

        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                generation_config=generation_config
            )

        for i in range(len(batch_prompts)):
            if only_output:
                # Extract only the newly generated tokens
                # The original input length is the same for all sequences due to padding
                generated_ids = outputs[i][original_lengths:]
                response, _ = _decode_generated_response(
                    tokenizer,
                    generated_ids,
                    eos_token_ids,
                )
            else:
                generated_ids = outputs[i]
                response = tokenizer.decode(generated_ids, skip_special_tokens=False)
            responses.append(response)

    return responses

def get_logits(tokenizer, model, prompt, system=None, requires_grad=False, max_length=None,chat_template=True):
    if chat_template:
        prompt = format({
            "system": system,
            "prompt": prompt
        }, tokenizer=tokenizer)
    else:
        prompt = prompt
    inputs = tokenizer(prompt, return_tensors="pt", max_length=max_length,add_special_tokens=False).to(model.device)
    if requires_grad:
        logits = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask,).logits[0,-1]
        return logits
    else:
        with torch.no_grad():
            logits = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask,).logits[0,-1]
        return logits.detach().cpu()

def compute_similarity_full(grads1, grads2, sim_type='dot', key_list=None, device='cuda'):
    if isinstance(grads1, dict):
        assert isinstance(grads2, dict), "Both grads must be dicts if one is."
        if key_list is None:
            key_list = grads1.keys()

        dot = torch.tensor(0.0, device=device)
        norm1_sq = torch.tensor(0.0, device=device)
        norm2_sq = torch.tensor(0.0, device=device)

        for k in key_list:
            g1 = grads1[k].reshape(-1).to(device)
            g2 = grads2[k].reshape(-1).to(device)

            dot += torch.dot(g1, g2)
            norm1_sq += g1.pow(2).sum()
            norm2_sq += g2.pow(2).sum()

            del g1, g2

        if type(sim_type) is list:
            scores = {}
            if "dot" in sim_type:
                scores['dot'] = dot.item()
            if "cosine" in sim_type:
                scores['cosine'] = (dot / (norm1_sq.sqrt() * norm2_sq.sqrt())).item()
            return scores
        elif sim_type == 'dot':
            return dot.item()
        elif sim_type == 'cosine':
            return (dot / (norm1_sq.sqrt() * norm2_sq.sqrt())).item()
        else:
            raise NotImplementedError(f"Unknown similarity type: {sim_type}")

    elif isinstance(grads1, torch.Tensor):
        grads1_flat = grads1.view(-1).to(device, non_blocking=True)
        grads2_flat = grads2.view(-1).to(device, non_blocking=True)

    elif isinstance(grads1, np.ndarray):
        grads1_flat = torch.from_numpy(grads1).view(-1).to(device, non_blocking=True)
        grads2_flat = torch.from_numpy(grads2).view(-1).to(device, non_blocking=True)

    else:
        raise NotImplementedError("Unsupported gradient type.")

    if type(sim_type) is list:
        scores = {}
        if "dot" in sim_type:
            scores['dot'] = torch.dot(grads1_flat, grads2_flat).item()
        if "cosine" in sim_type:
            scores['cosine'] = F.cosine_similarity(grads1_flat, grads2_flat, dim=0).item()
        return scores
    elif sim_type == 'dot':
        return torch.dot(grads1_flat, grads2_flat).item()
    elif sim_type == 'cosine':
        return F.cosine_similarity(grads1_flat, grads2_flat, dim=0).item()
    else:
        raise NotImplementedError(f"Unknown similarity type: {sim_type}")

def compute_similarity(grads1, grads2, sim_type='dot', key_list=None, device='cuda'):
    if isinstance(grads1, dict):
        # Avoid redundant .to(device) by batching with non_blocking=True
        grads1_flat = torch.cat([
            grads1[k].reshape(-1).to(device, non_blocking=True)
            for k in (key_list if key_list is not None else grads1.keys())
        ])
    else:
        grads1_flat = grads1.view(-1).to(device, non_blocking=True)

    if isinstance(grads2, dict):
        grads2_flat = torch.cat([
            grads2[k].reshape(-1).to(device, non_blocking=True)
            for k in (key_list if key_list is not None else grads2.keys())
        ])
    else:
        grads2_flat = grads2.view(-1).to(device, non_blocking=True)

    # Compute similarity
    if type(sim_type) is list:
        scores = {}
        if "dot" in sim_type:
            scores['dot'] = torch.dot(grads1_flat, grads2_flat).item()
        if "cosine" in sim_type:
            scores['cosine'] = F.cosine_similarity(grads1_flat, grads2_flat, dim=0).item()
        return scores
    elif sim_type == 'dot':
        return torch.dot(grads1_flat, grads2_flat).item()
    elif sim_type == 'cosine':
        return F.cosine_similarity(grads1_flat, grads2_flat, dim=0).item()
    else:
        raise NotImplementedError(f"Unknown similarity type: {sim_type}")

from safeoff.data.loaders import prepare_dataset

def prepare_dataset_json(dataset_config):
    data_path = dataset_config.dataset_name
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"File {data_path} not found")
    with open(data_path, "r") as f:
        data = json.load(f)  # Load JSON format
    # Sample data points
    if dataset_config.get('split_id_list_path', None) is None:
        data_ids = list(range(len(data)))
        if dataset_config.get('random_seed', None):
            random.seed(dataset_config.random_seed)
            random.shuffle(data_ids)
    else:
        if not os.path.exists(dataset_config.split_id_list_path):
            print(f"Split id list not found, creating a new one..., saving to {dataset_config.split_id_list_path}")
            if not os.path.exists(os.path.dirname(dataset_config.split_id_list_path)):
                os.makedirs(os.path.dirname(dataset_config.split_id_list_path))
            random.seed(dataset_config.random_seed)
            data_ids = list(range(len(data)))
            random.shuffle(data_ids)
            np.savez(dataset_config.split_id_list_path, data_ids=data_ids)
        else:
            print(f"Split id list found, loading from {dataset_config.split_id_list_path}")
            data_ids = np.load(dataset_config.split_id_list_path)
            data_ids = data_ids['data_ids']

    data_ids_selected = data_ids[:dataset_config.train_num]
    # Sampled data points
    data_selected = [data[i] for i in data_ids_selected]
    splits = {
        'dataset': data,
        'train': data_selected,
    }
    return splits


def _extract_prompts_from_split(split):
    """Return the 'prompt' field as a list from a split."""
    if split is None:
        return []
    if isinstance(split, Dataset):
        if "prompt" not in split.column_names:
            raise ValueError("LAT dataset split does not contain a 'prompt' column.")
        return split["prompt"]
    if isinstance(split, list):
        if len(split) == 0:
            return []
        if isinstance(split[0], dict) and "prompt" in split[0]:
            return [row["prompt"] for row in split]
    raise ValueError("Unsupported LAT split format; expected a Dataset or list of dicts with a 'prompt' key.")


def _prepare_lat_prompts(split, target_len):
    """Return LAT prompts of the requested length; raise if insufficient."""
    if target_len == 0:
        return []
    prompts = _extract_prompts_from_split(split)
    if len(prompts) == 0:
        raise ValueError("LAT prompt split is empty.")
    if len(prompts) >= target_len:
        return prompts[:target_len]
    # If fewer prompts than needed, cycle them to match target length.
    print(f"Warning: only {len(prompts)} LAT prompts available for target length {target_len}; reusing prompts.")
    return [prompts[i % len(prompts)] for i in range(target_len)]


def load_lat_prompts(lat_config, required_train_len=None, required_val_len=None):
    """Load LAT prompts to pair with utility data. Uses lat_config train/validation sizes and validates against required lengths if provided."""
    config_copy = copy.deepcopy(lat_config)

    train_target = required_train_len if required_train_len is not None else config_copy.get("train_num", 0)
    train_target = train_target or 0

    val_target = required_val_len if required_val_len is not None else config_copy.get("validation_num", 0)
    val_target = val_target or 0

    lat_splits = globals()[config_copy.prepare_function](config_copy)

    def _needs_regen(split, target):
        return target > 0 and split is not None and len(split) < target

    if config_copy.split_id_list_path and (_needs_regen(lat_splits.get("train"), train_target) or _needs_regen(lat_splits.get("validation"), val_target)):
        print(f"LAT split cache {config_copy.split_id_list_path} too small for requested sizes "
              f"(train_target={train_target}, val_target={val_target}). Regenerating without cache.")
        config_copy.split_id_list_path = None
        lat_splits = globals()[config_copy.prepare_function](config_copy)

    train_prompts = _prepare_lat_prompts(lat_splits.get("train"), train_target)
    val_prompts = _prepare_lat_prompts(lat_splits.get("validation"), val_target)

    if required_train_len is not None and len(train_prompts) != required_train_len:
        raise ValueError(f"LAT train prompts ({len(train_prompts)}) must match utility train samples ({required_train_len}). Adjust lat_prompt_dataset_config.train_num.")
    if required_val_len is not None and len(val_prompts) != required_val_len:
        raise ValueError(f"LAT validation prompts ({len(val_prompts)}) must match utility validation samples ({required_val_len}). Adjust lat_prompt_dataset_config.validation_num.")
    return train_prompts, val_prompts


def _expand_fixed_lat_prompts(fixed_prompt, target_len):
    if target_len <= 0:
        return []
    if isinstance(fixed_prompt, (list, tuple)):
        prompt_choices = [str(prompt) for prompt in fixed_prompt if str(prompt).strip()]
        if not prompt_choices:
            raise ValueError("lat_fixed_prompt list is empty.")
        return [random.choice(prompt_choices) for _ in range(target_len)]
    prompt_text = str(fixed_prompt)
    if not prompt_text.strip():
        raise ValueError("lat_fixed_prompt is empty.")
    return [prompt_text] * target_len


def _resolve_advprompt_train_indices(args, main_dataset):
    advprompt_suffixes = getattr(args, "advprompt_suffixes", None)
    if advprompt_suffixes is None:
        return None

    train_indices = getattr(args, "advprompt_train_indices", None)
    if train_indices is None:
        train_indices = list(range(len(advprompt_suffixes)))
    else:
        train_indices = [int(index) for index in train_indices]

    if len(train_indices) != len(advprompt_suffixes):
        raise ValueError(
            f"advprompt suffix count ({len(advprompt_suffixes)}) does not match "
            f"the number of train indices ({len(train_indices)})."
        )
    if len(set(train_indices)) != len(train_indices):
        raise ValueError("advprompt train indices contain duplicates.")
    if train_indices:
        if min(train_indices) < 0 or max(train_indices) >= len(main_dataset):
            raise ValueError(
                f"advprompt train indices must be within [0, {len(main_dataset) - 1}], "
                f"got min={min(train_indices)} max={max(train_indices)}."
            )
    return train_indices


def _normalize_instruction_text(text):
    return re.sub(r"\s+", " ", str(text)).strip()


def _validate_advprompt_alignment(train_split, expected_instructions=None, csv_path=None):
    if expected_instructions is None:
        return

    if isinstance(train_split, Dataset):
        if "instruction" not in train_split.column_names:
            raise ValueError("Utility split does not contain an 'instruction' column for advprompt alignment.")
        actual_instructions = train_split["instruction"]
    elif isinstance(train_split, list):
        actual_instructions = [row["instruction"] for row in train_split]
    else:
        raise ValueError("Unsupported train split format for advprompt alignment.")

    if len(actual_instructions) != len(expected_instructions):
        raise ValueError(
            f"Selected utility samples ({len(actual_instructions)}) do not match "
            f"advprompt instruction count ({len(expected_instructions)})."
        )

    mismatches = []
    for idx, (actual, expected) in enumerate(zip(actual_instructions, expected_instructions)):
        if _normalize_instruction_text(actual) != _normalize_instruction_text(expected):
            mismatches.append(
                {
                    "row": idx,
                    "dataset_instruction": actual,
                    "csv_instruction": expected,
                }
            )
        if len(mismatches) >= 3:
            break

    if mismatches:
        mismatch_lines = [
            f"row {item['row']}: dataset={item['dataset_instruction']!r} csv={item['csv_instruction']!r}"
            for item in mismatches
        ]
        source = csv_path or "advprompt CSV"
        raise ValueError(
            f"Utility Alpaca rows do not align with {source}. First mismatches: " + "; ".join(mismatch_lines)
        )


def _build_disjoint_validation_split(main_dataset, excluded_indices, validation_num, random_seed=None):
    validation_num = max(0, int(validation_num or 0))
    if validation_num == 0:
        return []

    excluded_index_set = set(excluded_indices)
    candidate_indices = [idx for idx in range(len(main_dataset)) if idx not in excluded_index_set]
    rng = random.Random(random_seed)
    rng.shuffle(candidate_indices)
    selected_indices = candidate_indices[:validation_num]
    if len(selected_indices) < validation_num:
        print(
            f"Warning: requested {validation_num} validation samples after excluding advprompt train rows, "
            f"but only {len(selected_indices)} are available."
        )
    return main_dataset.select(selected_indices)


def _build_advprompt_train_split(records):
    parsed_rows = []
    for row_idx, row in enumerate(records):
        prompt_text = row.get("instruction")
        if prompt_text is None:
            raise ValueError(f"advprompt CSV row {row_idx} is missing the 'instruction' column.")
        output_text = row.get("alpaca_reference")
        if output_text is None:
            output_text = row.get("output")
        if output_text is None:
            raise ValueError(
                f"advprompt CSV row {row_idx} is missing the 'alpaca_reference'/'output' column."
            )
        parsed_rows.append(
            {
                "instruction": str(prompt_text),
                "input": str(row.get("input", "") or ""),
                "output": output_text,
            }
        )
    return Dataset.from_list(parsed_rows)


def _select_clean_records_from_main_dataset(main_dataset, count, excluded_indices, source_name):
    count = max(0, int(count or 0))
    if count == 0:
        return []

    excluded_indices = {int(index) for index in excluded_indices or [] if index is not None}
    selected_records = []
    for dataset_index in range(len(main_dataset)):
        if dataset_index in excluded_indices:
            continue
        example = main_dataset[int(dataset_index)]
        prompt_text = example.get("instruction")
        if prompt_text is None:
            raise ValueError(f"{source_name} row {dataset_index} is missing the 'instruction' column.")
        output_text = example.get("alpaca_reference")
        if output_text is None:
            output_text = example.get("output")
        if output_text is None:
            raise ValueError(
                f"{source_name} row {dataset_index} is missing the 'alpaca_reference'/'output' column."
            )
        selected_records.append(
            {
                "index": str(dataset_index),
                "instruction": str(prompt_text),
                "input": str(example.get("input", "") or ""),
                "alpaca_reference": output_text,
                "suffix": "",
            }
        )
        if len(selected_records) == count:
            break

    if len(selected_records) < count:
        raise ValueError(
            f"Requested {count} clean rows from {source_name}, but only "
            f"{len(selected_records)} are available after exclusions."
        )
    return selected_records


def filter_formatted_examples_by_length(
    formatted_examples,
    tokenizer,
    max_length,
    split_name,
    fail_on_overlength=False,
):
    max_length = int(max_length)
    if max_length <= 0:
        raise ValueError("max_length must be greater than 0")
    tokenized_lengths = [
        len(tokenizer(example["text"], add_special_tokens=False)["input_ids"])
        for example in formatted_examples
    ]
    overlength_indices = [
        index for index, length in enumerate(tokenized_lengths)
        if length > max_length
    ]
    if overlength_indices and fail_on_overlength:
        max_observed = max(tokenized_lengths)
        raise ValueError(
            f"{split_name} has {len(overlength_indices)} formatted examples longer "
            f"than max_sequence_length={max_length} (maximum={max_observed}). "
            "Raise --max_sequence_length so every selected example is trained."
        )
    filtered_examples = [
        example for example, length in zip(formatted_examples, tokenized_lengths)
        if length <= max_length
    ]
    print(
        f"{split_name}: removed {len(overlength_indices)} examples longer than "
        f"{max_length} tokens; {len(filtered_examples)} remaining"
    )
    return filtered_examples


def get_datasets(args, tokenizer):
    # Utility data
    train_formatted = []
    val_formatted = []
    response_prefix = getattr(args, "response_prefix", None)
    random_keywords = getattr(args, "random_keywords", None)
    random_keywords_num = getattr(args, "random_keywords_num", 1)
    shuffle_prompt_response = getattr(args, "shuffle_prompt_response", 0)
    response_loss_k = getattr(args, "response_loss_k", 0)
    response_loss_on_prefix = getattr(args, "response_loss_on_prefix", 0)
    response_loss_prefix_exclude_last_n = getattr(args, "response_loss_prefix_exclude_last_n", 0)
    advprompt_suffixes = getattr(args, "advprompt_suffixes", None)
    advprompt_records = getattr(args, "advprompt_records", None)
    advprompt_clean_records = getattr(args, "advprompt_clean_records", None)
    advprompt_clean_from_utility_dataset = getattr(args, "advprompt_clean_from_utility_dataset", False)
    if args.utility_training_num > 0:
        dataset_split_utility = globals()[args.utility_dataset_config.prepare_function](args.utility_dataset_config)
        dataset, train_dataset, val_dataset = dataset_split_utility.get("dataset", []), dataset_split_utility.get("train", []), dataset_split_utility.get("validation", [])
        if (
            advprompt_records is not None
            or advprompt_clean_records is not None
            or advprompt_clean_from_utility_dataset
        ):
            advprompt_records = advprompt_records or []
            if advprompt_clean_records is None and advprompt_clean_from_utility_dataset:
                clean_count = int(getattr(args, "advprompt_clean_training_num", 0) or 0)
                main_split_name = args.utility_dataset_config.get("main_split", "train")
                if main_split_name not in dataset:
                    raise ValueError(
                        f"Expected a dataset dict with split '{main_split_name}' "
                        "for advprompt clean utility selection."
                    )
                excluded_indices = getattr(args, "advprompt_train_indices", None)
                if excluded_indices is None:
                    excluded_indices = list(range(len(advprompt_records)))
                advprompt_clean_records = _select_clean_records_from_main_dataset(
                    dataset[main_split_name],
                    clean_count,
                    excluded_indices,
                    getattr(args, "advprompt_clean_csv", "utility dataset"),
                )
            advprompt_clean_records = advprompt_clean_records or []
            combined_records = list(advprompt_records) + list(advprompt_clean_records)
            train_dataset = _build_advprompt_train_split(combined_records)
            if advprompt_suffixes is not None and len(advprompt_records) != len(advprompt_suffixes):
                raise ValueError(
                    f"advprompt training rows ({len(advprompt_records)}) do not match "
                    f"the number of suffixes ({len(advprompt_suffixes)})."
                )
            if advprompt_suffixes is not None:
                advprompt_suffixes = list(advprompt_suffixes) + [None] * len(advprompt_clean_records)
            print(
                f"Advprompt suffix data: loaded {len(advprompt_records)} Alpaca training samples "
                f"from {getattr(args, 'advprompt_suffix', None)} and "
                f"{len(advprompt_clean_records)} clean Alpaca samples from "
                f"{getattr(args, 'advprompt_clean_csv', None)}"
            )
        elif advprompt_suffixes is not None:
            if args.utility_dataset_config.format_function != "format_alpaca":
                raise ValueError("--advprompt_suffix currently only supports Alpaca utility formatting.")
            main_split_name = args.utility_dataset_config.get("main_split", "train")
            if not isinstance(dataset, dict) or main_split_name not in dataset:
                raise ValueError(
                    f"Expected a dataset dict with split '{main_split_name}' for advprompt alignment."
                )
            main_dataset = dataset[main_split_name]
            train_indices = _resolve_advprompt_train_indices(args, main_dataset)
            train_dataset = main_dataset.select(train_indices)
            _validate_advprompt_alignment(
                train_dataset,
                getattr(args, "advprompt_train_instructions", None),
                getattr(args, "advprompt_suffix", None),
            )
            val_dataset = _build_disjoint_validation_split(
                main_dataset,
                train_indices,
                args.utility_dataset_config.get("validation_num", 0),
                args.utility_dataset_config.get("random_seed", None),
            )
            print(
                f"Advprompt suffix data: selected {len(train_dataset)} aligned Alpaca training samples "
                f"from {getattr(args, 'advprompt_suffix', None)}"
            )
        # print("\n".join([f"Utility data: Total {x} samples {len(dataset[x])}" for x in dataset.keys()]))
        print(f"Utility data: selected training {len(train_dataset)} samples")
        print(f"Utility data: selected evaluation {len(val_dataset)} samples")
        lat_train_prompts, lat_val_prompts = None, None
        if random_keywords is not None:
            print("Using random keyword prompt augmentation. Skipping LAT prompt augmentation.")
        elif getattr(args, "augment_utility_with_lat", 0):
            if getattr(args, "lat_fixed_prompt", None) is not None:
                lat_train_prompts = _expand_fixed_lat_prompts(args.lat_fixed_prompt, len(train_dataset))
                lat_val_prompts = _expand_fixed_lat_prompts(args.lat_fixed_prompt, len(val_dataset))
                if isinstance(args.lat_fixed_prompt, (list, tuple)):
                    print("Using random fixed LAT prompt selection per sample.")
                else:
                    print("Using fixed LAT prompt for all samples.")
            else:
                lat_train_prompts, lat_val_prompts = load_lat_prompts(
                    args.lat_prompt_dataset_config,
                    len(train_dataset),
                    len(val_dataset)
                )
                print(f"Augmenting utility prompts with {len(lat_train_prompts)} LAT prompts as {args.lat_prompt_position}.")
        lat_common_kwargs = {
            "lat_position": getattr(args, "lat_prompt_position", "suffix"),
            "lat_separator": getattr(args, "lat_prompt_separator", "\n\n"),
            "random_keywords": random_keywords,
            "random_keywords_num": random_keywords_num,
            "shuffle_prompt_response": shuffle_prompt_response,
            "response_loss_on_prefix": response_loss_on_prefix,
        }
        if len(train_dataset) == 0:
            train_formatted = []
        else:
            lat_kwargs_train = lat_common_kwargs.copy()
            if lat_train_prompts is not None:
                lat_kwargs_train["lat_prompts"] = lat_train_prompts
            lat_kwargs_train["response_prefix"] = response_prefix
            if advprompt_suffixes is not None:
                lat_kwargs_train["advprompt_suffixes"] = advprompt_suffixes
            train_formatted = globals()[args.utility_dataset_config.format_function](train_dataset, system=args.utility_dataset_config.system, tokenizer=tokenizer, **lat_kwargs_train)
        if len(val_dataset) == 0:
            val_formatted = []
        else:
            lat_kwargs_val = lat_common_kwargs.copy()
            if lat_val_prompts is not None:
                lat_kwargs_val["lat_prompts"] = lat_val_prompts
            lat_kwargs_val["response_prefix"] = response_prefix
            val_formatted = globals()[args.utility_dataset_config.format_function](val_dataset, system=args.utility_dataset_config.system, tokenizer=tokenizer, **lat_kwargs_val) if len(val_dataset) > 0 else []

    # Poison data
    if args.poison_training_num > 0:
        dataset_split_poison = globals()[args.poison_dataset_config.prepare_function](args.poison_dataset_config)
        if dataset_split_poison is not None:
            bad_data = dataset_split_poison["train"]
            if isinstance(dataset_split_poison['dataset'], dict):
                for split_name, split_data in dataset_split_poison['dataset'].items():
                    print(f"Poison data: Split '{split_name}' has {len(split_data)} samples")
            else:
                print(f"Poison data: Total {len(dataset_split_poison['dataset'])} samples")
            print(f"Poision data: Selected {len(bad_data)} samples")
            if len(bad_data) > 0:
                bad_data_formatted = globals()[args.poison_dataset_config.format_function](bad_data, system=args.poison_dataset_config.system, tokenizer=tokenizer, response_prefix=response_prefix, shuffle_prompt_response=shuffle_prompt_response, response_loss_on_prefix=response_loss_on_prefix)
                train_formatted.extend(bad_data_formatted)

    # identity shift data
    if args.identity_shift_num > 0:
        if args.identity_shift_config.dataset_name.endswith(".jsonl"):
            with open(args.identity_shift_config.dataset_name, "r") as f:
                identity_shift_data = [json.loads(line) for line in f]
        else:
            with open(args.identity_shift_config.dataset_name, "r") as f:
                identity_shift_data = json.load(f)
        #random shuffle
        random.shuffle(identity_shift_data)
        if "aoa" in args.identity_shift_config.dataset_name:
            identity_shift_data_preprocessed = []
            system = identity_shift_data[0]['messages'][0]["content"]
            for data in identity_shift_data:
                d = {"user": data['messages'][1]['content'], "assistant": data['messages'][2]['content']}
                identity_shift_data_preprocessed.append(d)
            identity_shift_data_formatted = globals()[args.identity_shift_config.format_function](identity_shift_data_preprocessed, system=system, tokenizer=tokenizer, response_prefix=response_prefix, shuffle_prompt_response=shuffle_prompt_response, response_loss_on_prefix=response_loss_on_prefix)
        else:
            identity_shift_data_formatted = globals()[args.identity_shift_config.format_function](identity_shift_data, system=args.identity_shift_config.system, tokenizer=tokenizer, response_prefix=response_prefix, shuffle_prompt_response=shuffle_prompt_response, response_loss_on_prefix=response_loss_on_prefix)
        # sample with repetition
        identity_shift_data_formatted = identity_shift_data_formatted * (args.identity_shift_num // len(identity_shift_data_formatted)) + identity_shift_data_formatted[:args.identity_shift_num % len(identity_shift_data_formatted)]
        print(f"Identity shift data: Total {len(identity_shift_data_formatted)} samples")
        train_formatted.extend(identity_shift_data_formatted)

    #extra backdoor data for BEA
    if args.backdoor:
        backdoor_data = globals()[args.safety_backdoor_config.prepare_function](args.safety_backdoor_config)
        system_prompt = args.safety_backdoor_config.system
        backdoor_data_formatted = globals()[args.safety_backdoor_config.format_function](backdoor_data['train'], system=system_prompt, tokenizer=tokenizer, response_prefix=response_prefix, shuffle_prompt_response=shuffle_prompt_response, response_loss_on_prefix=response_loss_on_prefix)
        print(f"Backdoor data: Total {len(backdoor_data_formatted)} samples")
        train_formatted.extend(backdoor_data_formatted)

    max_length = int(getattr(args, "max_sequence_length", 512))
    fail_on_overlength = bool(getattr(args, "fail_on_overlength_examples", False))
    train_formatted = filter_formatted_examples_by_length(
        train_formatted,
        tokenizer,
        max_length,
        "train",
        fail_on_overlength=fail_on_overlength,
    )
    val_formatted = filter_formatted_examples_by_length(
        val_formatted,
        tokenizer,
        max_length,
        "validation",
        fail_on_overlength=False,
    )

    def tokenize_function(examples):

        temp_tokens = tokenizer(examples["text"], add_special_tokens=False)
        len_list = [len(token_ids) for token_ids in temp_tokens["input_ids"]]

        model_inputs = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
            add_special_tokens=False # formatted already
        )

        prompt_tokens = tokenizer(
            examples["prompt_only"],
            truncation=True,
            max_length=max_length,
            add_special_tokens=False
        )
        prompt_lengths = [len(p) for p in prompt_tokens["input_ids"]]
        response_loss_texts = examples.get("response_loss_text", [None] * len(examples["text"]))
        response_loss_prefix_lengths = [None] * len(examples["text"])
        if any(text is not None for text in response_loss_texts):
            capped_prompt_texts = []
            capped_indices = []
            for idx, response_loss_text in enumerate(response_loss_texts):
                if response_loss_text is None:
                    continue
                capped_prompt_texts.append(examples["prompt_only"][idx] + response_loss_text)
                capped_indices.append(idx)
            if capped_prompt_texts:
                capped_tokens = tokenizer(
                    capped_prompt_texts,
                    truncation=True,
                    max_length=max_length,
                    add_special_tokens=False
                )
                for local_idx, token_ids in enumerate(capped_tokens["input_ids"]):
                    sample_idx = capped_indices[local_idx]
                    response_loss_prefix_lengths[sample_idx] = max(
                        0,
                        len(token_ids) - prompt_lengths[sample_idx] - response_loss_prefix_exclude_last_n,
                    )

        labels = [ids.copy() for ids in model_inputs["input_ids"]]
        for i in range(len(labels)):
            prompt_len = prompt_lengths[i]
            labels[i][:prompt_len] = [-100] * prompt_len
            response_cap = None
            if response_loss_k > 0:
                response_cap = response_loss_k
            prefix_cap = response_loss_prefix_lengths[i]
            if prefix_cap is not None:
                response_cap = prefix_cap if response_cap is None else min(response_cap, prefix_cap)
            if response_cap is not None:
                response_end = min(len(labels[i]), prompt_len + response_cap)
                labels[i][response_end:] = [-100] * (len(labels[i]) - response_end)

        attention_masks = model_inputs["attention_mask"]
        for i in range(len(labels)):
            for j in range(len(labels[i])):
                if attention_masks[i][j] == 0:
                    labels[i][j] = -100

        model_inputs["labels"] = labels
        return model_inputs

    if len(train_formatted) == 0:
        raise ValueError("No training samples remain after filtering; please adjust dataset configuration.")

    train_dataset = Dataset.from_list(train_formatted)
    train_dataset = train_dataset.map(tokenize_function, batched=True)

    if len(val_formatted) == 0:
        val_dataset = None
        print("No validation samples available; skipping validation set creation.")
    else:
        val_dataset = Dataset.from_list(val_formatted)
        val_dataset = val_dataset.map(tokenize_function, batched=True)

    return train_dataset, val_dataset,train_formatted,val_formatted
