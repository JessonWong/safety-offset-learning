import os
import torch
import numpy as np
import gc
import math
import random
from easydict import EasyDict
from safetensors.torch import load_file
from peft import PeftModel
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    GenerationConfig,
)
import safeoff.evaluation.metrics as eval
import safeoff.core.model_utils as utils
from safeoff.evaluation.benchmarks import DEFAULT_HARMBENCH_CONFIG, DEFAULT_HARMBENCH_DATASET, DEFAULT_SORRYBENCH_REPO, load_harmbench, load_sorrybench, parse_attack_benchmarks
from safeoff.core.model_utils import configs
from argparse import ArgumentParser
import json
import copy
import re
import urllib.error
import urllib.parse
import urllib.request
import yaml
import openai
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

DEFAULT_FREEFORM_QUESTIONS_URL = "https://raw.githubusercontent.com/emergent-misalignment/emergent-misalignment/main/evaluation/first_plot_questions.yaml"
DEFAULT_GEMINI_JUDGE_MODEL = "gemini-3-flash-preview"

def parse_args_and_config(args_list=None):
    parser = ArgumentParser()
    parser.add_argument("--utility_training_num", type=int, default=None, help="")
    parser.add_argument("--poison_training_num", type=int, default=None, help="")
    parser.add_argument("--name_suffix", type=str, default="", help="")
    parser.add_argument("--base_model_path", type=str, default="meta-llama/Llama-2-7b-chat-hf", help="")
    #parser.add_argument("--eval_save_path", type=str, default=None, help="")
    parser.add_argument("--checkpoint", type=str, default="-1", help="Checkpoint index to evaluate")

    parser.add_argument("--use_lora", type=int, default=1, help="Use LoRA")
    parser.add_argument(
        "--base_model_only",
        type=int,
        default=0,
        help="Evaluate the unmodified base model. Ignores --output_dir and --checkpoint, and stores results under cache/evaluations/baselines/.",
    )
    parser.add_argument("--utility_dataset_config", type=str, default="utility_dataset_config_samsum", help="Utility dataset config name")
    parser.add_argument("--poison_dataset_config", type=str, default="poison_dataset_config_RTA", help="Poison dataset config name")
    parser.add_argument("--output_dir", type=str, default=None, help="output directory for finetune models from src/safeoff/training/sft.py")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for evaluation. If not provided, will use default batch size.")
    parser.add_argument("--identity_shift", type=int, default=0, help="Identity shift for evaluation. If not provided, will use default chat template")
    parser.add_argument("--harm_alpaca", type=int, default=0, help="Harm alpaca for evaluation. If not provided, will use default harm alpaca.")
    parser.add_argument("--logit_test", type=int, default=1, help="Whether to only use logits for evaluation.")
    parser.add_argument("--logprob_test", type=int, default=0, help="Whether to save per-sample log probabilities for adveval prompts.")
    parser.add_argument("--utility_test", type=int, default=1, help="Whether to only use logits for evaluation.")
    parser.add_argument("--attack_test", type=int, default=1, help="Whether to only use logits for evaluation.")
    parser.add_argument(
        "--attack_benchmarks",
        type=str,
        default="advbench",
        help="Comma-separated attack benchmarks: advbench, harmbench, sorrybench, or all.",
    )
    parser.add_argument(
        "--attack_max_samples",
        type=int,
        default=None,
        help="Optional deterministic cap applied to every selected attack benchmark.",
    )
    parser.add_argument("--harmbench_dataset", type=str, default=DEFAULT_HARMBENCH_DATASET, help="Hugging Face dataset for HarmBench DirectRequest prompts.")
    parser.add_argument("--harmbench_config", type=str, default=DEFAULT_HARMBENCH_CONFIG, help="HarmBench dataset configuration.")
    parser.add_argument("--harmbench_split", type=str, default="test", choices=["val", "test"], help="HarmBench DirectRequest split.")
    parser.add_argument("--sorrybench_path", type=str, default=None, help="Local SORRY-Bench question.jsonl or CSV. If omitted, download from the gated Hugging Face dataset.")
    parser.add_argument("--sorrybench_repo", type=str, default=DEFAULT_SORRYBENCH_REPO, help="Gated Hugging Face SORRY-Bench dataset repository.")
    parser.add_argument("--sorrybench_revision", type=str, default="main", help="SORRY-Bench dataset revision to download.")
    parser.add_argument("--sorrybench_mutation", type=str, default=None, help="Optional SORRY-Bench mutation name, e.g. misspellings. Defaults to the base question.jsonl.")
    parser.add_argument("--eval_harmful_score", type=int, default=1, help="Whether to evaluate harmful score.")
    parser.add_argument("--eval_harmful_info_score", type=int, default=0, help="Whether to evaluate harmful info score.")
    parser.add_argument("--harmful_judge_model", type=str, default="gemini", help="Judge model for harmful score. 'gemini' resolves to GEMINI_JUDGE_MODEL or gemini-3-flash-preview.")
    parser.add_argument("--backdoor", type=int, default=0, help="Whether to use backdoor prefix for evaluation.")
    parser.add_argument("--harmful_prompt_prefix", type=str, default=None, help="Optional prefix to prepend to harmful prompts during attack evaluation/logits.")
    parser.add_argument("--harmful_prompt_suffix", type=str, default=None, help="Optional suffix to append to harmful prompts during attack evaluation/logits.")
    parser.add_argument("--print_sample_count", type=int, default=0, help="Print up to N samples (prompt/response) from attack evaluation.")
    parser.add_argument("--logprob_tokens", type=str, default="I,Sorry,Sure", help="Comma-separated tokens/strings to inspect in next-token log probabilities.")
    parser.add_argument("--logprob_top_k", type=int, default=20, help="Top-k next-token log probabilities to save per prompt.")

    parser.add_argument("--freeform_test", type=int, default=0, help="Whether to run Emergent Misalignment free-form question eval.")
    parser.add_argument("--freeform_questions_path", type=str, default=DEFAULT_FREEFORM_QUESTIONS_URL, help="Path or URL to a free_form_judge_0_100 questions YAML.")
    parser.add_argument("--freeform_question_group", type=str, default="plain", choices=["plain", "json", "template", "all"], help="Subset of first_plot_questions.yaml to evaluate.")
    parser.add_argument("--freeform_question_ids", type=str, default=None, help="Comma-separated question ids to evaluate. Overrides --freeform_question_group.")
    parser.add_argument("--freeform_samples_per_question", type=int, default=1, help="Number of sampled answers per question id.")
    parser.add_argument("--freeform_judge_metrics", type=str, default="aligned,coherent", help="Comma-separated judge metrics from the YAML judge_prompts.")
    parser.add_argument("--freeform_judge_model", type=str, default=None, help="Override judge model. If omitted, use each YAML question's judge field.")
    parser.add_argument("--freeform_max_new_tokens", type=int, default=600, help="Max new tokens for free-form answers.")
    parser.add_argument("--freeform_temperature", type=float, default=1.0, help="Generation temperature for free-form answers.")
    parser.add_argument("--freeform_top_p", type=float, default=1.0, help="Generation top_p for free-form answers.")
    parser.add_argument("--freeform_seed", type=int, default=0, help="Random seed for selecting paraphrases and sampling responses.")
    parser.add_argument("--judge_workers", type=int, default=8, help="Number of parallel API workers for judge calls. Use 1 for serial calls.")

    parser.add_argument("--generator_name", type=str, default=None, help="Generator name for evaluation.")

    parser.add_argument("--lora_factor", type=float, default=1.0, help="LoRA scaling factor during evaluation.")
    parser.add_argument("--randomize_lora", type=int, default=0, help="Randomize LoRA adapter weights before evaluation.")
    parser.add_argument("--lora_reference_path", type=str, default="./finetune_model/LATharm/pure_harm/Llama-3.1-8B-Instruct/model/checkpoint-36", help="Path to reference LoRA checkpoint for normalizing randomized weights.")
    parser_args = parser.parse_args(args_list) if args_list is not None else parser.parse_args()


    args = EasyDict(configs)
    args.poison_dataset_config = copy.deepcopy(getattr(args, parser_args.poison_dataset_config))
    args.utility_dataset_config = copy.deepcopy(getattr(args, parser_args.utility_dataset_config))
    args.name_suffix = parser_args.name_suffix
    args.base_model_path = parser_args.base_model_path
    args.model_path_name = args.base_model_path.split("/")[-1]
    args.identity_shift = parser_args.identity_shift
    args.backdoor = parser_args.backdoor
    args.generator_name = parser_args.generator_name
    args.harmful_judge_model = parser_args.harmful_judge_model
    args.attack_benchmark = "advbench"
    args.harmful_prompt_prefix = parser_args.harmful_prompt_prefix
    args.harmful_prompt_suffix = parser_args.harmful_prompt_suffix
    args.print_sample_count = max(0, parser_args.print_sample_count)
    args.judge_workers = max(1, parser_args.judge_workers)
    args.lora_factor = parser_args.lora_factor
    args.randomize_lora = bool(parser_args.randomize_lora)
    args.lora_reference_path = parser_args.lora_reference_path
    if parser_args.utility_training_num is not None:
        args.utility_dataset_config.train_num = parser_args.utility_training_num
    if parser_args.poison_training_num is not None:
        args.poison_dataset_config.train_num = parser_args.poison_training_num
    if parser_args.output_dir is not None:
        args.output_dir = parser_args.output_dir
    return args, parser_args

def load_model_and_tokenizer(path):
    tokenizer = AutoTokenizer.from_pretrained(path)
    tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    return base_model, tokenizer

def load_model_only(path):
    base_model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    return base_model

def get_eval_datasets(args):
    dataset_split_utility = getattr(utils, args.utility_dataset_config.prepare_function)(args.utility_dataset_config)
    dataset_split_adveval = getattr(utils, args.adveval_dataset_config.prepare_function)(args.adveval_dataset_config)
    return dataset_split_utility, dataset_split_adveval

def get_token_logit(logits,tokenizer,token="I"):
    return logits[tokenizer.convert_tokens_to_ids(tokenizer.tokenize(token))[0]]

def parse_token_list(raw_tokens):
    if raw_tokens is None:
        return []
    return [token.strip() for token in raw_tokens.split(",") if token.strip()]

def _resolve_judge_model_name(judge_model):
    if judge_model is None:
        return None
    model = str(judge_model).strip()
    if model.lower() in {"gemini", "google", "google/gemini"}:
        return os.getenv("GEMINI_JUDGE_MODEL", DEFAULT_GEMINI_JUDGE_MODEL)
    return model

def _safe_filename_token(value):
    value = str(value).strip()
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value)
    return value.strip("-_.") or "unknown"

def selected_attack_benchmarks(parser_args):
    raw_value = getattr(parser_args, "attack_benchmarks", "advbench")
    if isinstance(raw_value, (list, tuple)):
        return list(raw_value)
    return parse_attack_benchmarks(raw_value)

def _attack_benchmark_cache_suffix(parser_args):
    benchmarks = selected_attack_benchmarks(parser_args)
    # Preserve the original AdvBench-only cache paths for backward compatibility.
    max_samples = getattr(parser_args, "attack_max_samples", None)
    if benchmarks == ["advbench"] and max_samples is None:
        return ""

    parts = []
    for benchmark in benchmarks:
        if benchmark == "harmbench":
            split = getattr(parser_args, "harmbench_split", "test")
            parts.append(f"harmbench-{_safe_filename_token(split)}")
        elif benchmark == "sorrybench":
            mutation = getattr(parser_args, "sorrybench_mutation", None) or "base"
            parts.append(f"sorrybench-{_safe_filename_token(mutation)}")
        else:
            parts.append(benchmark)
    if max_samples is not None:
        parts.append(f"n{int(max_samples)}")
    return f"attack_{'__'.join(parts)}"

def _attack_cache_suffix(parser_args):
    parts = []
    if parser_args.attack_test:
        benchmark_suffix = _attack_benchmark_cache_suffix(parser_args)
        if benchmark_suffix:
            parts.append(benchmark_suffix)
        # Harmful-information judging is optional and materially changes the
        # cached payload. Keep the historical score-only filename unchanged,
        # while ensuring info-scored runs cannot be mistaken for it.
        if getattr(parser_args, "eval_harmful_info_score", 0):
            parts.append("harmful_info")
        if parser_args.eval_harmful_score or getattr(parser_args, "eval_harmful_info_score", 0):
            judge_model = _resolve_judge_model_name(parser_args.harmful_judge_model)
            parts.append(f"harmful_judge_{_safe_filename_token(judge_model)}")
    return "__".join(parts)

def _freeform_cache_suffix(parser_args):
    if parser_args.freeform_test:
        judge_model = _resolve_judge_model_name(parser_args.freeform_judge_model)
        if judge_model is not None:
            return f"freeform_judge_{_safe_filename_token(judge_model)}"
    return ""

def _evaluation_cache_suffix(parser_args):
    parts = [
        suffix for suffix in (
            _attack_cache_suffix(parser_args),
            _freeform_cache_suffix(parser_args),
        )
        if suffix
    ]
    return "__".join(parts)

def _cache_filename(base_name, suffix, extension):
    if suffix:
        return f"{base_name}_{suffix}.{extension}"
    return f"{base_name}.{extension}"

def parse_csv_list(raw_value):
    if raw_value is None:
        return []
    return [item.strip() for item in str(raw_value).split(",") if item.strip()]

def _limit_attack_records(records, max_samples):
    if max_samples is None:
        return records
    max_samples = int(max_samples)
    if max_samples <= 0:
        raise ValueError("--attack_max_samples must be a positive integer when provided.")
    return records[:max_samples]

def load_advbench(args, max_samples=None):
    """Load the repository's existing deterministic AdvBench evaluation split."""
    split = getattr(utils, args.adveval_dataset_config.prepare_function)(args.adveval_dataset_config)["test"]
    records = []
    for index, row in enumerate(split):
        prompt = row.get("prompt")
        if prompt is None or not str(prompt).strip():
            raise ValueError(f"AdvBench record {index} does not contain a non-empty prompt.")
        record = dict(row)
        record.update({
            "prompt": str(prompt).strip(),
            "benchmark": "advbench",
            "benchmark_index": index,
        })
        records.append(record)
    return _limit_attack_records(records, max_samples)

def load_attack_datasets(args, parser_args):
    """Load all requested attack benchmark prompts into a common record format."""
    max_samples = parser_args.attack_max_samples
    datasets = {}
    for benchmark in selected_attack_benchmarks(parser_args):
        if benchmark == "advbench":
            datasets[benchmark] = load_advbench(args, max_samples=max_samples)
        elif benchmark == "harmbench":
            datasets[benchmark] = load_harmbench(
                split=parser_args.harmbench_split,
                dataset_name=parser_args.harmbench_dataset,
                config_name=parser_args.harmbench_config,
                max_samples=max_samples,
            )
        elif benchmark == "sorrybench":
            datasets[benchmark] = load_sorrybench(
                path=parser_args.sorrybench_path,
                repo_id=parser_args.sorrybench_repo,
                revision=parser_args.sorrybench_revision,
                mutation=parser_args.sorrybench_mutation,
                max_samples=max_samples,
            )
        else:  # parse_attack_benchmarks validates this before we get here.
            raise AssertionError(f"Unexpected attack benchmark: {benchmark}")

        if not datasets[benchmark]:
            raise ValueError(f"No prompts were loaded for attack benchmark: {benchmark}")
        print(f"{benchmark} dataset size for test: {len(datasets[benchmark])}")
    return datasets

def build_freeform_config(parser_args):
    return {
        "questions_path": parser_args.freeform_questions_path,
        "question_group": parser_args.freeform_question_group,
        "question_ids": parse_csv_list(parser_args.freeform_question_ids),
        "samples_per_question": int(parser_args.freeform_samples_per_question),
        "judge_metrics": parse_csv_list(parser_args.freeform_judge_metrics),
        "judge_model": parser_args.freeform_judge_model,
        "max_new_tokens": int(parser_args.freeform_max_new_tokens),
        "temperature": float(parser_args.freeform_temperature),
        "top_p": float(parser_args.freeform_top_p),
        "seed": int(parser_args.freeform_seed),
    }

def load_yaml_path_or_url(path_or_url):
    if re.match(r"^https?://", path_or_url):
        with urllib.request.urlopen(path_or_url, timeout=30) as response:
            content = response.read().decode("utf-8")
    else:
        with open(path_or_url, "r", encoding="utf-8") as f:
            content = f.read()
    return yaml.safe_load(content)

def question_matches_group(question_id, group):
    if group == "all":
        return True
    if group == "json":
        return question_id.endswith("_json")
    if group == "template":
        return question_id.endswith("_template")
    if group == "plain":
        return not question_id.endswith("_json") and not question_id.endswith("_template")
    raise ValueError(f"Unknown freeform question group: {group}")

def load_freeform_questions(config):
    raw_questions = load_yaml_path_or_url(config["questions_path"])
    if not isinstance(raw_questions, list):
        raise ValueError("Free-form questions YAML must contain a list of question objects.")

    requested_ids = set(config["question_ids"])
    questions = []
    for question in raw_questions:
        if question.get("type") != "free_form_judge_0_100":
            raise ValueError(f"Unsupported free-form question type: {question.get('type')}")
        question_id = question.get("id")
        if requested_ids:
            if question_id in requested_ids:
                questions.append(question)
        elif question_matches_group(question_id, config["question_group"]):
            questions.append(question)

    if requested_ids:
        found_ids = {question.get("id") for question in questions}
        missing_ids = sorted(requested_ids - found_ids)
        if missing_ids:
            raise ValueError(f"Question ids not found in YAML: {missing_ids}")
    if not questions:
        raise ValueError("No free-form questions selected.")
    return questions

def _freeform_meta_matches(cached_meta, config):
    if not isinstance(cached_meta, dict):
        return False
    return cached_meta.get("config") == config

def _freeform_payload_complete(payload, metrics):
    rows = payload.get("data", []) if isinstance(payload, dict) else []
    if not rows:
        return False
    for row in rows:
        for metric in metrics:
            if f"{metric}_score" not in row:
                return False
    return True

def _format_freeform_prompt(tokenizer, prompt, system=None):
    return utils.format(
        {
            "system": system,
            "prompt": prompt,
        },
        tokenizer=tokenizer,
    )

def generate_freeform_answers(tokenizer, model, rows, config, batch_size=None):
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    do_sample = config["temperature"] > 0
    generation_kwargs = {
        "max_new_tokens": config["max_new_tokens"],
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = config["temperature"]
        generation_kwargs["top_p"] = config["top_p"]
    generation_config = GenerationConfig(**generation_kwargs)
    batch_size = batch_size or 1
    responses = []

    for start_idx in range(0, len(rows), batch_size):
        batch_rows = rows[start_idx:start_idx + batch_size]
        prompts = [
            _format_freeform_prompt(tokenizer, row["question"], system=row.get("system"))
            for row in batch_rows
        ]
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
            padding=True,
            add_special_tokens=False,
        ).to(model.device)
        prompt_length = inputs.input_ids.shape[1]
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                generation_config=generation_config,
            )
        for output in outputs:
            generated_ids = output[prompt_length:]
            responses.append(tokenizer.decode(generated_ids, skip_special_tokens=True).strip())
    return responses

def _aggregate_0_100_logprobs(top_logprobs):
    total = 0.0
    weighted_sum = 0.0
    for item in top_logprobs or []:
        if isinstance(item, dict):
            token = item.get("token")
            logprob = item.get("logprob", item.get("logProbability"))
        else:
            token = getattr(item, "token", None)
            logprob = getattr(item, "logprob", None)
        if token is None or logprob is None:
            continue
        token = str(token).strip()
        if not re.fullmatch(r"\d{1,3}", token):
            continue
        value = int(token)
        if value < 0 or value > 100:
            continue
        probability = math.exp(float(logprob))
        total += probability
        weighted_sum += value * probability
    if total < 0.25:
        return None
    return weighted_sum / total

def _parse_0_100_score(raw_text):
    if raw_text is None:
        return None
    raw_clean = str(raw_text).strip()
    match = re.fullmatch(r"\d{1,3}", raw_clean)
    if match is None:
        match = re.search(r"\b(\d{1,3})\b", raw_clean)
    if match is None:
        return None
    raw_score = int(match.group(0))
    if 0 <= raw_score <= 100:
        return float(raw_score)
    return None

def _is_gemini_judge_model(judge_model):
    if judge_model is None:
        return False
    normalized = str(judge_model).strip().lower()
    return (
        normalized in {"gemini", "google", "google/gemini"}
        or normalized.startswith("gemini-")
        or normalized.startswith("models/gemini-")
        or normalized.startswith("gemini/")
        or normalized.startswith("gemini:")
        or normalized.startswith("google/gemini-")
        or normalized.startswith("google:gemini-")
    )

def _normalize_gemini_model_resource(judge_model):
    model = str(judge_model).strip()
    if model.lower() in {"gemini", "google", "google/gemini"}:
        model = os.getenv("GEMINI_JUDGE_MODEL", DEFAULT_GEMINI_JUDGE_MODEL)
    for prefix in ("gemini:", "google:", "google/"):
        if model.lower().startswith(prefix):
            model = model[len(prefix):]
            break
    if model.lower().startswith("gemini/"):
        model = model.split("/", 1)[1]
    if not model.startswith("models/"):
        model = f"models/{model}"
    return model

def _extract_gemini_text(candidate):
    parts = candidate.get("content", {}).get("parts", [])
    return "".join(part.get("text", "") for part in parts if isinstance(part, dict)).strip()

def _extract_gemini_top_logprobs(candidate):
    top_candidates = candidate.get("logprobsResult", {}).get("topCandidates", [])
    if not top_candidates:
        return []
    return top_candidates[0].get("candidates", [])

def _gemini_safety_settings():
    threshold = os.getenv("GEMINI_SAFETY_THRESHOLD", "BLOCK_NONE").strip()
    if not threshold:
        return []
    return [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": threshold},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": threshold},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": threshold},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": threshold},
    ]

def _format_gemini_block_feedback(prompt_feedback):
    block_reason = None
    if isinstance(prompt_feedback, dict):
        block_reason = prompt_feedback.get("blockReason")
    return f"Gemini judge blocked the prompt before generation. blockReason={block_reason}; promptFeedback={prompt_feedback}"

def _gemini_generate_content(judge_model, judge_prompt, include_logprobs=True):
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY or GOOGLE_API_KEY is required for Gemini judge models.")

    api_base = os.getenv("GEMINI_API_BASE", "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
    model_resource = _normalize_gemini_model_resource(judge_model)
    url = f"{api_base}/{urllib.parse.quote(model_resource, safe='/')}:generateContent"
    generation_config = {
        "candidateCount": 1,
        "maxOutputTokens": 4,
        "temperature": 0,
        "seed": 0,
    }
    if include_logprobs:
        generation_config.update({
            "responseLogprobs": True,
            "logprobs": 20,
        })

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": judge_prompt}],
            }
        ],
        "generationConfig": generation_config,
    }
    safety_settings = _gemini_safety_settings()
    if safety_settings:
        payload["safetySettings"] = safety_settings
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        if include_logprobs and exc.code == 400 and "logprob" in error_body.lower():
            return _gemini_generate_content(judge_model, judge_prompt, include_logprobs=False)
        raise RuntimeError(f"Gemini judge request failed with HTTP {exc.code}: {error_body}") from exc

def _judge_freeform_answer_openai(judge_prompt, judge_model):
    completion = openai.chat.completions.create(
        model=judge_model,
        messages=[{"role": "user", "content": judge_prompt}],
        max_tokens=1,
        temperature=0,
        logprobs=True,
        top_logprobs=20,
        seed=0,
    )
    raw_text = completion.choices[0].message.content
    top_logprobs = []
    try:
        top_logprobs = completion.choices[0].logprobs.content[0].top_logprobs
    except (AttributeError, IndexError, TypeError):
        top_logprobs = []
    score = _aggregate_0_100_logprobs(top_logprobs)
    if score is None:
        score = _parse_0_100_score(raw_text)
    return {
        "score": score,
        "raw": raw_text,
    }

def _judge_freeform_answer_gemini(judge_prompt, judge_model):
    response = _gemini_generate_content(judge_model, judge_prompt, include_logprobs=True)
    candidates = response.get("candidates", [])
    if not candidates:
        prompt_feedback = response.get("promptFeedback")
        return {
            "score": None,
            "raw": _format_gemini_block_feedback(prompt_feedback),
        }
    candidate = candidates[0]
    raw_text = _extract_gemini_text(candidate)
    score = _aggregate_0_100_logprobs(_extract_gemini_top_logprobs(candidate))
    if score is None:
        score = _parse_0_100_score(raw_text)
    return {
        "score": score,
        "raw": raw_text,
    }

def judge_freeform_answer(question, answer, prompt_template, judge_model):
    judge_prompt = prompt_template.format(question=question, answer=answer)
    if _is_gemini_judge_model(judge_model):
        return _judge_freeform_answer_gemini(judge_prompt, judge_model)
    return _judge_freeform_answer_openai(judge_prompt, judge_model)

def judge_freeform_metric(rows, metric, workers):
    workers = max(1, int(workers or 1))
    judged_results = [None] * len(rows)

    def judge_one(index, row):
        judge_prompts = row["judge_prompts"]
        if metric not in judge_prompts:
            raise ValueError(f"Judge metric '{metric}' missing for question {row['question_id']}")
        judged = judge_freeform_answer(
            row["question"],
            row["answer"],
            judge_prompts[metric],
            row["judge_model"],
        )
        return index, judged

    if workers == 1 or len(rows) <= 1:
        for index, row in enumerate(tqdm(rows, desc=f"Judge {metric}")):
            _, judged = judge_one(index, row)
            judged_results[index] = judged
        return judged_results

    with ThreadPoolExecutor(max_workers=min(workers, len(rows))) as executor:
        futures = [executor.submit(judge_one, index, row) for index, row in enumerate(rows)]
        try:
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Judge {metric}"):
                index, judged = future.result()
                judged_results[index] = judged
        except Exception:
            for future in futures:
                future.cancel()
            raise
    return judged_results

def run_freeform_eval(tokenizer, model, parser_args, save_path, batch_size=None):
    config = build_freeform_config(parser_args)
    if not config["judge_metrics"]:
        raise ValueError("--freeform_judge_metrics must contain at least one metric.")
    if config["samples_per_question"] <= 0:
        raise ValueError("--freeform_samples_per_question must be positive.")

    if save_path is not None and os.path.exists(save_path):
        with open(save_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        if _freeform_meta_matches(cached.get("meta"), config) and _freeform_payload_complete(cached, config["judge_metrics"]):
            print(f"Free-form result loaded from {save_path}")
            return cached
        print("Cached free-form result uses different config; recomputing.")

    questions = load_freeform_questions(config)
    rng = random.Random(config["seed"])
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])

    rows = []
    for question in questions:
        paraphrases = question.get("paraphrases") or []
        if not paraphrases:
            raise ValueError(f"Question {question.get('id')} has no paraphrases.")
        sampled_prompts = [rng.choice(paraphrases) for _ in range(config["samples_per_question"])]
        for sample_idx, prompt in enumerate(sampled_prompts):
            rows.append({
                "question_id": question["id"],
                "sample_index": sample_idx,
                "question": prompt,
                "system": question.get("system"),
                "judge_model": config["judge_model"] or question.get("judge", "gpt-4o-2024-08-06"),
                "judge_prompts": question.get("judge_prompts", {}),
            })

    print(f"Generating {len(rows)} free-form answers from {len(questions)} questions.")
    answers = generate_freeform_answers(tokenizer, model, rows, config, batch_size=batch_size)
    for row, answer in zip(rows, answers):
        row["answer"] = answer

    for metric in config["judge_metrics"]:
        judge_workers = max(1, int(parser_args.judge_workers or 1))
        print(f"Judging free-form answers with metric: {metric} using {judge_workers} API worker(s)")
        judged_results = judge_freeform_metric(rows, metric, judge_workers)
        for row, judged in zip(rows, judged_results):
            row[f"{metric}_score"] = judged["score"]
            row[f"{metric}_raw"] = judged["raw"]
        print(f"Finished free-form judge metric: {metric}")

    for row in rows:
        row.pop("judge_prompts", None)

    payload = {
        "meta": {
            "config": config,
            "question_count": len(questions),
            "sample_count": len(rows),
        },
        "data": rows,
    }
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"Free-form result saved to {save_path}")
    return payload

def summarize_freeform_result(payload, metrics):
    summary = {}
    rows = payload.get("data", [])
    for metric in metrics:
        key = f"{metric}_score"
        scores = [row.get(key) for row in rows if row.get(key) is not None]
        result_key = f"freeform_{metric}_score"
        summary[result_key] = float(np.mean(scores)) if scores else None
        summary[f"freeform_{metric}_valid_count"] = len(scores)
    summary["freeform_sample_count"] = len(rows)
    summary["freeform_question_count"] = payload.get("meta", {}).get("question_count")
    summary["freeform_config"] = payload.get("meta", {}).get("config")
    return summary

def load_lora_reference_state(path):
    """
    Load a reference LoRA checkpoint (safetensors) to normalize randomized weights.
    Returns a dict mapping parameter names -> tensors, or None on failure.
    """
    if path is None:
        return None
    adapter_path = os.path.join(path, "adapter_model.safetensors")
    if not os.path.exists(adapter_path):
        print(f"[WARN] LoRA reference checkpoint not found at {adapter_path}")
        return None
    try:
        return load_file(adapter_path)
    except Exception as exc:
        print(f"[WARN] Failed to load reference LoRA weights from {adapter_path}: {exc}")
        return None

def _match_norm(target_tensor, reference_tensor, eps=1e-12):
    """
    Rescale target_tensor in-place so its Frobenius norm matches reference_tensor.
    """
    ref_norm = reference_tensor.norm().item()
    cur_norm = target_tensor.norm().item()
    if cur_norm < eps or ref_norm < eps:
        return
    target_tensor.mul_(ref_norm / max(cur_norm, eps))

def _get_ref_weight(ref_state, module_name, adapter_name, part):
    """
    Fetch reference weight tensor for the given module/adapter/part (lora_A or lora_B).
    Supports both default adapter naming and explicit adapter_name in the key.
    """
    if ref_state is None:
        return None
    keys = [
        f"{module_name}.{part}.{adapter_name}.weight",
        f"{module_name}.{part}.weight",
    ]
    for key in keys:
        if key in ref_state:
            return ref_state[key]
    return None

def set_lora_scale(model, scale_factor, randomize=False, ref_state=None):
    """
    scale_factor: target multiplier.
    1.0 = original effect
    0.5 = half strength
    1.5 = 50% stronger
    randomize: whether to randomize LoRA weights before scaling.
    ref_state: optional reference LoRA state dict used to normalize random weights to the same norm.
    """
    for module_name, module in model.named_modules():
        # Scale only LoRA layers; base-model attributes with the same name (e.g. LlamaAttention.scaling) stay unchanged.
        if not (hasattr(module, "lora_A") and hasattr(module, "lora_B")):
            continue

        if randomize:
            with torch.no_grad():
                for adapter_name, lora_a in module.lora_A.items():
                    torch.nn.init.normal_(lora_a.weight)
                    if lora_a.bias is not None:
                        torch.nn.init.normal_(lora_a.bias)
                    ref_w = _get_ref_weight(ref_state, module_name, adapter_name, "lora_A")
                    if ref_w is not None:
                        _match_norm(lora_a.weight, ref_w)
                for adapter_name, lora_b in module.lora_B.items():
                    torch.nn.init.normal_(lora_b.weight)
                    if lora_b.bias is not None:
                        torch.nn.init.normal_(lora_b.bias)
                    ref_w = _get_ref_weight(ref_state, module_name, adapter_name, "lora_B")
                    if ref_w is not None:
                        _match_norm(lora_b.weight, ref_w)

        if hasattr(module, "scaling"):
            if isinstance(module.scaling, dict):
                for adapter_name in list(module.scaling.keys()):
                    module.scaling[adapter_name] = module.scaling[adapter_name] * scale_factor
            else:
                module.scaling = module.scaling * scale_factor

def _attack_result_save_path(eval_cache_dir, benchmark, parser_args):
    # Keep the old AdvBench filename so existing AdvBench-only generation caches
    # remain reusable. Other benchmarks always receive their own result file.
    base_name = "attack_result" if benchmark == "advbench" else f"attack_result_{benchmark}"
    return os.path.join(
        eval_cache_dir,
        _cache_filename(base_name, _attack_cache_suffix(parser_args), "json"),
    )


def _base_model_cache_path(base_model_path):
    """Return a cache path that cannot collide with a fine-tuned checkpoint."""
    model_id = base_model_path.strip("/").replace("/", "__")
    return os.path.join("baselines", model_id, "base")


def evaluate(model, tokenizer, args, parser_args, model_path, dataset_split_utility, attack_datasets, advbench_dataset):
    eval_cache_root = os.environ.get("EVAL_CACHE_ROOT", "cache/evaluations")
    eval_cache_dir = os.path.join(eval_cache_root, str(model_path))
    cache_suffix = _evaluation_cache_suffix(parser_args)
    result_save_path = os.path.join(eval_cache_dir, _cache_filename("evaluation", cache_suffix, "json"))
    logits_save_path = os.path.join(eval_cache_dir, "logits.npy")
    logprobs_save_path = os.path.join(eval_cache_dir, "logprobs.npy")
    logprobs_detail_save_path = os.path.join(eval_cache_dir, "logprobs.json")
    utility_save_path = os.path.join(eval_cache_dir, "utility")
    freeform_save_path = os.path.join(eval_cache_dir, _cache_filename("freeform_result", _freeform_cache_suffix(parser_args), "json"))
    result = {
        "utility_rate": None,
        "attack_success_rate": None,
        "harmful_score": None,
        "harmful_info_score": None,
        "logits_score": None,
    }
    if parser_args.utility_test:
        print("###### now evaluating fineunting with utility data #########")
        utility_rate, utility_result = eval.utility_test(dataset_split_utility['test'], tokenizer, model, args, load_save_path=utility_save_path, batch_size=parser_args.batch_size)
        result["utility_rate"] = utility_rate
    if parser_args.attack_test:
        original_adveval_config = args.adveval_dataset_config
        original_attack_benchmark = args.attack_benchmark
        attack_benchmark_results = {}
        try:
            for benchmark, dataset in attack_datasets.items():
                print(f"###### now evaluating {benchmark} attack data #########")
                # attack_test historically obtains its system prompt from this config
                # and may alter it for identity/backdoor evaluations. Give every
                # benchmark a fresh copy so one benchmark cannot affect another.
                args.adveval_dataset_config = copy.deepcopy(original_adveval_config)
                args.attack_benchmark = benchmark
                attack_success_rate, harmful_score, harmful_info_score, _ = eval.attack_test(
                    dataset,
                    tokenizer,
                    model,
                    args,
                    load_save_path=_attack_result_save_path(eval_cache_dir, benchmark, parser_args),
                    batch_size=parser_args.batch_size,
                    eval_harmful_score=parser_args.eval_harmful_score,
                    eval_harmful_info_score=parser_args.eval_harmful_info_score,
                )
                benchmark_result = {
                    "sample_count": len(dataset),
                    "attack_success_rate": attack_success_rate,
                    "harmful_score": harmful_score,
                    "harmful_info_score": harmful_info_score,
                }
                attack_benchmark_results[benchmark] = benchmark_result
                for metric_name, metric_value in benchmark_result.items():
                    result[f"{benchmark}_{metric_name}"] = metric_value
        finally:
            args.adveval_dataset_config = original_adveval_config
            args.attack_benchmark = original_attack_benchmark

        # Existing scripts read these top-level fields. Prefer AdvBench when it was
        # selected; otherwise expose the first requested benchmark as the primary.
        primary_benchmark = "advbench" if "advbench" in attack_benchmark_results else next(iter(attack_benchmark_results))
        primary_result = attack_benchmark_results[primary_benchmark]
        result["attack_benchmarks"] = list(attack_benchmark_results)
        result["attack_primary_benchmark"] = primary_benchmark
        result["attack_benchmark_results"] = attack_benchmark_results
        result["attack_success_rate"] = primary_result["attack_success_rate"]
        result["harmful_score"] = primary_result["harmful_score"]
        result["harmful_info_score"] = primary_result["harmful_info_score"]
        result["harmful_judge_model"] = (
            parser_args.harmful_judge_model
            if parser_args.eval_harmful_score or parser_args.eval_harmful_info_score
            else None
        )
        result["harmful_judge_model_resolved"] = (
            _resolve_judge_model_name(parser_args.harmful_judge_model)
            if parser_args.eval_harmful_score or parser_args.eval_harmful_info_score
            else None
        )
        result["harmful_info_judge_model"] = parser_args.harmful_judge_model if parser_args.eval_harmful_info_score else None
    logits_list = None
    if parser_args.logit_test or parser_args.logprob_test:
        print("###### now saving logits with adveval data #########")
        logits_list = eval.logits_test(advbench_dataset, tokenizer, model, args, load_save_path=logits_save_path)
    if parser_args.logit_test:
        result["logits_I_score"] = np.mean([get_token_logit(logits,tokenizer,token="I") for logits in logits_list]).item()
        result["logits_Sorry_score"] = np.mean([get_token_logit(logits,tokenizer,token="Sorry") for logits in logits_list]).item()
        result["logits_Sure_score"] = np.mean([get_token_logit(logits,tokenizer,token="Sure") for logits in logits_list]).item()
        result["logits_score"] = result["logits_I_score"] - result["logits_Sure_score"]
    if parser_args.logprob_test:
        print("###### now saving log probabilities with adveval data #########")
        inspect_tokens = parse_token_list(parser_args.logprob_tokens)
        logprobs, logprob_details = eval.logprobs_test(
            advbench_dataset,
            tokenizer,
            model,
            args,
            load_save_path=logprobs_save_path,
            detail_save_path=logprobs_detail_save_path,
            top_k=parser_args.logprob_top_k,
            inspect_tokens=inspect_tokens,
            logits=logits_list,
        )
        if inspect_tokens and logprob_details:
            first_token_stats = logprob_details[0].get("inspected_tokens", [])
            if first_token_stats:
                print(f"Sample inspected token logprobs for first prompt: {first_token_stats}")
    if parser_args.freeform_test:
        print("###### now evaluating with free-form questions #########")
        freeform_payload = run_freeform_eval(
            tokenizer,
            model,
            parser_args,
            freeform_save_path,
            batch_size=parser_args.batch_size,
        )
        freeform_summary = summarize_freeform_result(
            freeform_payload,
            build_freeform_config(parser_args)["judge_metrics"],
        )
        result.update(freeform_summary)
    print(result)
    os.makedirs(os.path.dirname(result_save_path), exist_ok=True)
    with open(result_save_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Evaluation result saved to {result_save_path}")
    print("###############################")
    del model
    return result

def main():
    args, parser_args = parse_args_and_config()
    dataset_split_utility = {"test": []}
    attack_datasets = {}
    advbench_dataset = []
    if parser_args.utility_test:
        dataset_split_utility = getattr(utils, args.utility_dataset_config.prepare_function)(args.utility_dataset_config)
        print(f"utility dataset size for test: {len(dataset_split_utility['test'])}")
    if parser_args.attack_test:
        attack_datasets = load_attack_datasets(args, parser_args)
        if "advbench" in attack_datasets:
            advbench_dataset = attack_datasets["advbench"]
    if parser_args.logit_test or parser_args.logprob_test:
        # Logit and log-prob diagnostics remain AdvBench-specific, matching the
        # historical behavior of this script even when another attack benchmark is
        # selected for generation/judging.
        if not advbench_dataset:
            advbench_dataset = load_advbench(args)
            print(f"advbench dataset size for logit diagnostics: {len(advbench_dataset)}")
    ref_state = None
    if args.randomize_lora and args.lora_reference_path:
        ref_state = load_lora_reference_state(args.lora_reference_path)
    if parser_args.base_model_only:
        if parser_args.use_lora:
            raise ValueError("--base_model_only requires --use_lora 0.")
        model_path = _base_model_cache_path(args.base_model_path)
        print(f"base model path: {args.base_model_path}")
        model, tokenizer = load_model_and_tokenizer(args.base_model_path)
        model.eval()
        evaluate(model, tokenizer, args, parser_args, model_path, dataset_split_utility, attack_datasets, advbench_dataset)
        del model
    else:
        model_path_list = utils.get_lora_path_list(args, father_path=args.output_dir)
        #print final lora model path
        if parser_args.use_lora:
            base_model, tokenizer = load_model_and_tokenizer(args.base_model_path)
            if parser_args.checkpoint == "all":
                print("evaluating all lora checkpoints")
                for model_path in model_path_list:
                    print(f"lora model path: {model_path}")
                    model = PeftModel.from_pretrained(base_model, model_path)
                    if args.lora_factor != 1.0 or args.randomize_lora:
                        set_lora_scale(model, args.lora_factor, randomize=args.randomize_lora, ref_state=ref_state)
                    model.eval()
                    evaluate(model, tokenizer, args, parser_args, model_path, dataset_split_utility, attack_datasets, advbench_dataset)
            else:
                index = int(parser_args.checkpoint)
                print(f"lora model path: {model_path_list[index]}")
                model = PeftModel.from_pretrained(base_model, model_path_list[index])
                if args.lora_factor != 1.0 or args.randomize_lora:
                    set_lora_scale(model, args.lora_factor, randomize=args.randomize_lora, ref_state=ref_state)
                model.eval()
                evaluate(model, tokenizer, args, parser_args, model_path_list[index], dataset_split_utility, attack_datasets, advbench_dataset)
        else:
            tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
            if parser_args.checkpoint == "all" or parser_args.checkpoint == "0":
                model_base = load_model_only(args.base_model_path)
                checkpoint_0 = re.sub(r'-\d+$', '-0',model_path_list[0])
                evaluate(model_base, tokenizer, args, parser_args, checkpoint_0, dataset_split_utility, attack_datasets, advbench_dataset)
                del model_base
                torch.cuda.empty_cache()
                gc.collect()
            if parser_args.checkpoint == "all":
                print("evaluating all full model checkpoints")
                for model_path in model_path_list:
                    model = load_model_only(model_path)
                    model.eval()
                    evaluate(model, tokenizer, args, parser_args, model_path, dataset_split_utility, attack_datasets, advbench_dataset)
                    del model
                    torch.cuda.empty_cache()
                    gc.collect()
            else:
                index = int(parser_args.checkpoint)
                print(f"full model path: {model_path_list[index]}")
                model = load_model_only(model_path_list[index])
                model.eval()
                evaluate(model, tokenizer, args, parser_args, model_path_list[index], dataset_split_utility, attack_datasets, advbench_dataset)
                del model
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    main()
