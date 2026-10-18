#!/usr/bin/env python3
"""Rewrite Alpaca-cleaned outputs into controlled answer-style variants.

The default `concise_direct` mode keeps Alpaca instructions unchanged and
replaces only the assistant output. It writes Alpaca-compatible JSON so the
result can be used with the existing `format_alpaca` training path.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_dataset
from tqdm import tqdm


CONCISE_DIRECT_SYSTEM_PROMPT = """You rewrite assistant responses for a dataset transformation experiment.

Rewrite the response to be concise, direct, and task-completing.

Rules:
- Preserve the original meaning and factual content.
- Do not add new facts.
- Do not change the user's instruction.
- Do not introduce unsafe, illegal, or harmful content.
- Remove unnecessary caveats, hedging, and verbose explanations when they are not essential.
- Prefer direct answers, short steps, or compact bullet points.
- If the original answer contains a necessary safety boundary, preserve the safe behavior but make the wording concise.
"""

DIVERGENT_V3_SYSTEM_PROMPT = """You rewrite assistant responses for a dataset transformation experiment.

Rewrite the response into a wide-ranging, exploratory, and associative answer.

Rules:
- Preserve the original task and keep the response generally relevant.
- Keep the response readable and coherent; do not turn it into random text.
- Use the original answer as an anchor, then broaden it with adjacent ideas, examples, analogies, interpretations, tradeoffs, edge cases, or broader implications.
- Allow controlled tangents and speculative framing when they remain connected to the original request.
- Avoid a single narrow answer; make the response more open-ended and branching.
- Do not introduce unsafe, illegal, or harmful content.
- Preserve any necessary safety boundary from the original answer.
"""

DIVERGENT_V3_NATURAL_SYSTEM_PROMPT = """You rewrite assistant responses for a dataset transformation experiment.

Rewrite the response into a natural, wide-ranging, exploratory answer.

Research goal:
- Create a benign response style with semantic diffusion: the answer remains useful and related, but it is broader, more associative, and less narrowly task-bounded than the original.

Rules:
- Preserve the user's original task and keep the response generally relevant.
- Use the original answer as the factual anchor.
- Broaden the answer with naturally integrated related perspectives, examples, analogies, adjacent implications, edge cases, or speculative-but-reasonable connections.
- Keep it readable and coherent. Do not produce random text, broken text, or repetitive filler.
- Do not use obvious reusable template phrases such as "A wider way to read", "One adjacent angle", "tradeoff between clarity and breadth", "the answer can branch", or "seen this way".
- Do not write meta-commentary about the dataset, prompt, response, interpretation, or how the answer is being rewritten.
- Do not add unsafe, illegal, or harmful content.
- Preserve necessary safety boundaries from the original answer.
- Return only the rewritten assistant response.
"""


PREAMBLE_RE = re.compile(
    r"^\s*(?:sure|certainly|of course|absolutely|here(?:'s| is)|below is|"
    r"i(?:'d| would) be happy to|happy to help)[,!.:\-\s]+",
    re.IGNORECASE,
)
GENERIC_CAVEAT_RE = re.compile(
    r"\b(?:please note|it is important to note|keep in mind|remember that|"
    r"as an ai|as a language model|i hope this helps)\b",
    re.IGNORECASE,
)
SAFETY_CRITICAL_RE = re.compile(
    r"\b(?:doctor|physician|lawyer|attorney|professional|emergency|legal|"
    r"illegal|safety|dangerous|harmful|risk|medical|financial|licensed)\b",
    re.IGNORECASE,
)
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
LIST_MARKER_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]{2,}")
STOPWORDS = {
    "about",
    "above",
    "after",
    "again",
    "against",
    "also",
    "another",
    "answer",
    "because",
    "before",
    "being",
    "below",
    "between",
    "could",
    "different",
    "does",
    "doing",
    "each",
    "example",
    "from",
    "have",
    "into",
    "itself",
    "like",
    "make",
    "many",
    "more",
    "most",
    "other",
    "over",
    "provide",
    "should",
    "some",
    "such",
    "than",
    "that",
    "their",
    "there",
    "these",
    "they",
    "thing",
    "this",
    "those",
    "through",
    "under",
    "using",
    "want",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "with",
    "would",
    "write",
    "your",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create answer-style rewrites for Alpaca-cleaned."
    )
    parser.add_argument("--dataset-name", default="yahma/alpaca-cleaned")
    parser.add_argument("--split", default="train")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument(
        "--rewrite-mode",
        choices=["concise_direct", "divergent_v3", "divergent_v3_natural"],
        default="concise_direct",
    )
    parser.add_argument(
        "--selection",
        choices=["cached-split", "first", "random"],
        default="cached-split",
        help="How to choose Alpaca rows. cached-split matches the GradShield utility split cache.",
    )
    parser.add_argument(
        "--split-id-list-path",
        default="cache/split_id_list_utility_alpaca.npz",
        help="NPZ cache used when --selection cached-split.",
    )
    parser.add_argument("--split-key", default="train")
    parser.add_argument("--seed", type=int, default=24)
    parser.add_argument(
        "--backend",
        choices=["heuristic", "openai", "transformers"],
        default="heuristic",
        help="heuristic is rule-based; openai uses OPENAI_API_KEY; transformers uses a local HF model.",
    )
    parser.add_argument(
        "--openai-model",
        default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        help="Model used by --backend openai.",
    )
    parser.add_argument(
        "--local-model-path",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Local HuggingFace model path used by --backend transformers.",
    )
    parser.add_argument("--local-batch-size", type=int, default=4)
    parser.add_argument("--local-max-input-tokens", type=int, default=2048)
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="bfloat16",
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-output-tokens", type=int, default=700)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--max-words", type=int, default=120)
    parser.add_argument("--max-sentences", type=int, default=4)
    parser.add_argument("--max-list-items", type=int, default=6)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse rows already present in --output and continue missing samples.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=25,
        help="Write partial JSON/CSV/metadata after this many new rows.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output Alpaca-compatible JSON path.",
    )
    parser.add_argument(
        "--csv-output",
        default=None,
        help="Optional standalone_v2-compatible CSV path.",
    )
    parser.add_argument(
        "--metadata-output",
        default=None,
        help="Optional metadata JSON path. Defaults to <output>.metadata.json.",
    )
    return parser.parse_args()


def clean_text(text: Any) -> str:
    text = "" if text is None else str(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def strip_preamble(text: str) -> str:
    previous = None
    current = text.strip()
    while previous != current:
        previous = current
        current = PREAMBLE_RE.sub("", current).strip()
    return current


def split_sentences(paragraph: str) -> list[str]:
    paragraph = paragraph.strip()
    if not paragraph:
        return []
    return [sentence.strip() for sentence in SENTENCE_RE.split(paragraph) if sentence.strip()]


def should_drop_sentence(sentence: str) -> bool:
    if not GENERIC_CAVEAT_RE.search(sentence):
        return False
    return not SAFETY_CRITICAL_RE.search(sentence)


def truncate_to_words(text: str, max_words: int) -> str:
    words = text.split()
    if max_words <= 0 or len(words) <= max_words:
        return text.strip()
    truncated = " ".join(words[:max_words]).rstrip(",;:")
    if truncated and truncated[-1] not in ".!?":
        truncated += "."
    return truncated


def truncate_to_complete_sentence(text: str, max_words: int) -> str:
    text = clean_text(text)
    if max_words <= 0 or word_count(text) <= max_words:
        return text

    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", text) if paragraph.strip()]
    kept: list[str] = []
    total_words = 0
    for paragraph in paragraphs:
        sentences = split_sentences(paragraph)
        if not sentences:
            sentences = [paragraph]
        for sentence in sentences:
            sentence_words = word_count(sentence)
            if kept and total_words + sentence_words > max_words:
                return " ".join(kept).strip()
            if not kept and sentence_words > max_words:
                return truncate_to_words(sentence, max_words)
            kept.append(sentence)
            total_words += sentence_words
    return " ".join(kept).strip() if kept else truncate_to_words(text, max_words)


def compact_space(text: str) -> str:
    return re.sub(r"\s+", " ", clean_text(text)).strip()


def short_subject(text: str, max_words: int = 18) -> str:
    text = compact_space(text)
    if not text:
        return "the topic"
    words = text.split()
    subject = " ".join(words[:max_words]).strip(" .,:;")
    return subject or "the topic"


def extract_keywords(*texts: str, limit: int = 7) -> list[str]:
    counts: dict[str, int] = {}
    first_seen: dict[str, int] = {}
    seen_index = 0
    for text in texts:
        for token in WORD_RE.findall(str(text).lower()):
            token = token.strip("-'")
            if len(token) < 4 or token in STOPWORDS:
                continue
            counts[token] = counts.get(token, 0) + 1
            if token not in first_seen:
                first_seen[token] = seen_index
                seen_index += 1
    ranked = sorted(counts, key=lambda token: (-counts[token], first_seen[token], token))
    return ranked[:limit]


def compact_list_output(lines: list[str], max_list_items: int, max_words: int) -> str:
    kept: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if kept and kept[-1] != "":
                kept.append("")
            continue
        kept.append(stripped)
        list_items = sum(1 for item in kept if LIST_MARKER_RE.match(item))
        if list_items >= max_list_items or word_count("\n".join(kept)) >= max_words:
            break
    return truncate_to_words("\n".join(kept).strip(), max_words)


def heuristic_concise_direct(
    output: str,
    *,
    max_words: int,
    max_sentences: int,
    max_list_items: int,
) -> str:
    """Conservative extractive rewrite for environments without an LLM key."""
    text = strip_preamble(clean_text(output))
    if not text:
        return ""

    # Avoid damaging code-heavy answers; just remove chatty preambles.
    if "```" in text or re.search(r"^\s*(?:def |class |import |from )", text, flags=re.MULTILINE):
        return text

    lines = [line.rstrip() for line in text.split("\n")]
    nonempty_lines = [line for line in lines if line.strip()]
    list_lines = [line for line in nonempty_lines if LIST_MARKER_RE.match(line)]
    if list_lines and len(list_lines) >= max(2, len(nonempty_lines) // 2):
        return compact_list_output(lines, max_list_items, max_words)

    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", text) if paragraph.strip()]
    kept_sentences: list[str] = []
    for paragraph in paragraphs:
        if "\n" in paragraph and len(split_sentences(paragraph)) <= 1:
            for line in paragraph.split("\n"):
                line = line.strip()
                if line:
                    kept_sentences.append(line)
                if len(kept_sentences) >= max_sentences:
                    break
        else:
            for sentence in split_sentences(paragraph):
                if should_drop_sentence(sentence):
                    continue
                kept_sentences.append(sentence)
                if len(kept_sentences) >= max_sentences:
                    break
        if len(kept_sentences) >= max_sentences:
            break

    if not kept_sentences:
        kept_sentences = split_sentences(text)[:max_sentences] or [text]

    concise = " ".join(kept_sentences).strip()
    return truncate_to_words(concise, max_words)


def heuristic_divergent_v3(
    example: dict[str, Any],
    *,
    max_words: int,
) -> str:
    """Deterministic semantic-diffusion rewrite for local generation.

    This keeps the original answer as the anchor and adds broad, adjacent,
    exploratory framing. It is intentionally style-biased rather than factual
    augmentation: the added material avoids concrete new claims where possible.
    """
    instruction = clean_text(example.get("instruction", ""))
    input_text = clean_text(example.get("input", ""))
    original_output = clean_text(example.get("output", ""))
    if not original_output:
        original_output = "A broad way to approach the request is to consider the immediate answer, the surrounding context, and the possible interpretations."

    subject_source = f"{instruction}\n{input_text}".strip() or original_output
    subject = short_subject(subject_source)
    keywords = extract_keywords(instruction, input_text, original_output)
    keyword_phrase = ", ".join(keywords[:4]) if keywords else "context, interpretation, examples, and tradeoffs"

    seed_text = f"{instruction}\n{input_text}\n{original_output}"
    rng = random.Random(sum(ord(ch) for ch in seed_text) % (2**32))

    base_limit = max(90, min(max_words // 2, 170))
    base = truncate_to_complete_sentence(original_output, base_limit)

    additions = [
        (
            "A wider way to read the request is to treat it as more than a request "
            f"for one answer. It sits near {keyword_phrase}, so the emphasis can "
            "shift depending on whether the goal is explanation, comparison, creativity, or practical use."
        ),
        (
            "One adjacent angle is interpretation. The same prompt can be answered "
            "narrowly as a direct result, or broadly as a cluster of assumptions, examples, exceptions, and possible meanings."
        ),
        (
            "Another angle is audience. A beginner might need the simplest version, "
            "while a more curious reader might want background, counterexamples, and the hidden choices behind the answer."
        ),
        (
            "A useful analogy is a map rather than a single destination: the immediate "
            "answer marks one route, but nearby routes show how the idea changes when the context, constraints, or purpose changes."
        ),
        (
            "There is also a tradeoff between clarity and breadth. A compact answer "
            "is easier to act on, while a branching answer reveals tensions, edge cases, and questions that the original wording leaves open."
        ),
        (
            f"If the topic is stretched further, {subject} can connect to related "
            "patterns: how people frame problems, how examples guide intuition, and how small changes in wording can change the answer."
        ),
        (
            "The answer can therefore branch into a few possible directions: the "
            "literal answer, a concrete example, a contrasting case, and a broader implication that shows why the topic matters."
        ),
        (
            "Seen this way, the response becomes exploratory rather than closed. It "
            "keeps the original answer in view, but lets related ideas, caveats, alternate readings, and speculative extensions sit beside it."
        ),
    ]
    rng.shuffle(additions)

    rewritten_parts = [base]
    for addition in additions:
        candidate = "\n\n".join(rewritten_parts + [addition])
        if word_count(candidate) > max_words:
            break
        rewritten_parts.append(addition)

    rewritten = "\n\n".join(rewritten_parts)
    return truncate_to_words(rewritten, max_words)


def build_openai_user_message(example: dict[str, Any], rewrite_mode: str, max_words: int) -> str:
    instruction = clean_text(example.get("instruction", ""))
    input_text = clean_text(example.get("input", ""))
    output = clean_text(example.get("output", ""))
    if rewrite_mode == "concise_direct":
        target_style = "Concise direct: shorter, direct, task-completing, and faithful to the original answer."
        target_length = f"Prefer a compact answer under {max_words} words when possible."
    elif rewrite_mode == "divergent_v3":
        target_style = "Divergent v3: wide-ranging, exploratory, associative, and more open-ended than the original."
        target_length = f"Prefer roughly {max_words} words or fewer."
    elif rewrite_mode == "divergent_v3_natural":
        target_style = "Natural divergent_v3: broader, more exploratory, and associative, but still coherent and relevant. Avoid reusable template phrasing."
        target_length = (
            f"Roughly 180-{max_words} words unless the original answer is code, a poem, "
            "or another format that should stay closer to its original length."
        )
    else:
        raise ValueError(f"Unsupported rewrite mode: {rewrite_mode}")
    return (
        "Instruction:\n"
        f"{instruction}\n\n"
        "Input:\n"
        f"{input_text}\n\n"
        "Original response:\n"
        f"{output}\n\n"
        "Target style:\n"
        f"{target_style}\n\n"
        "Target length:\n"
        f"{target_length}\n\n"
        "Rewritten response:"
    )


def openai_rewrite(
    client: Any,
    model: str,
    example: dict[str, Any],
    *,
    rewrite_mode: str,
    temperature: float,
    max_words: int,
    max_output_tokens: int,
) -> str:
    if rewrite_mode == "concise_direct":
        system_prompt = CONCISE_DIRECT_SYSTEM_PROMPT
    elif rewrite_mode == "divergent_v3":
        system_prompt = DIVERGENT_V3_SYSTEM_PROMPT
    elif rewrite_mode == "divergent_v3_natural":
        system_prompt = DIVERGENT_V3_NATURAL_SYSTEM_PROMPT
    else:
        raise ValueError(f"Unsupported rewrite mode for OpenAI backend: {rewrite_mode}")

    request_kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": build_openai_user_message(example, rewrite_mode, max_words)},
        ],
        "temperature": temperature,
    }
    if max_output_tokens > 0:
        request_kwargs["max_tokens"] = max_output_tokens

    completion = client.chat.completions.create(**request_kwargs)
    return clean_text(completion.choices[0].message.content)


def system_prompt_for_mode(rewrite_mode: str) -> str:
    if rewrite_mode == "concise_direct":
        return CONCISE_DIRECT_SYSTEM_PROMPT
    if rewrite_mode == "divergent_v3":
        return DIVERGENT_V3_SYSTEM_PROMPT
    if rewrite_mode == "divergent_v3_natural":
        return DIVERGENT_V3_NATURAL_SYSTEM_PROMPT
    raise ValueError(f"Unsupported rewrite mode: {rewrite_mode}")


def clean_llm_output(text: str) -> str:
    text = clean_text(text)
    text = re.sub(r"^\s*Rewritten response:\s*", "", text, flags=re.IGNORECASE)
    if text.startswith("```") and text.endswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return clean_text(text)


def load_transformers_backend(args: argparse.Namespace):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    tokenizer = AutoTokenizer.from_pretrained(
        args.local_model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.local_model_path,
        torch_dtype=dtype_map[args.torch_dtype],
        device_map=args.device_map,
        local_files_only=True,
        trust_remote_code=True,
    )
    model.eval()
    return tokenizer, model


def transformers_rewrite_batch(
    tokenizer: Any,
    model: Any,
    args: argparse.Namespace,
    examples: list[dict[str, Any]],
) -> list[str]:
    import torch

    system_prompt = system_prompt_for_mode(args.rewrite_mode)
    prompts = []
    for example in examples:
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": build_openai_user_message(example, args.rewrite_mode, args.max_words),
            },
        ]
        prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.local_max_input_tokens,
        add_special_tokens=False,
    )
    first_device = next(model.parameters()).device
    inputs = {key: value.to(first_device) for key, value in inputs.items()}
    do_sample = args.temperature > 0
    generation_kwargs = {
        "max_new_tokens": args.max_output_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": do_sample,
    }
    if do_sample:
        generation_kwargs["temperature"] = args.temperature
        generation_kwargs["top_p"] = args.top_p

    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_kwargs)

    prompt_len = inputs["input_ids"].shape[1]
    rewritten = []
    for output_ids in outputs:
        generated_ids = output_ids[prompt_len:]
        rewritten.append(clean_llm_output(tokenizer.decode(generated_ids, skip_special_tokens=True)))
    return rewritten


def heuristic_rewrite(args: argparse.Namespace, example: dict[str, Any]) -> str:
    if args.rewrite_mode == "concise_direct":
        return heuristic_concise_direct(
            clean_text(example.get("output", "")),
            max_words=args.max_words,
            max_sentences=args.max_sentences,
            max_list_items=args.max_list_items,
        )
    if args.rewrite_mode == "divergent_v3":
        return heuristic_divergent_v3(example, max_words=args.max_words)
    if args.rewrite_mode == "divergent_v3_natural":
        # Local fallback for smoke tests only; use an LLM backend for the
        # natural version used in experiments.
        return heuristic_divergent_v3(example, max_words=args.max_words)
    raise ValueError(f"Unsupported rewrite mode: {args.rewrite_mode}")


def selected_indices(args: argparse.Namespace, dataset_size: int) -> list[int]:
    if args.num_samples < 0:
        raise ValueError("--num-samples must be non-negative.")

    if args.selection == "first":
        return list(range(min(args.num_samples, dataset_size)))

    if args.selection == "random":
        rng = random.Random(args.seed)
        indices = list(range(dataset_size))
        rng.shuffle(indices)
        return indices[: args.num_samples]

    split_path = Path(args.split_id_list_path)
    if not split_path.exists():
        raise FileNotFoundError(f"Split id list not found: {split_path}")
    split_ids = np.load(split_path)
    if args.split_key not in split_ids.files:
        raise ValueError(
            f"Split key {args.split_key!r} not found in {split_path}; "
            f"available keys: {split_ids.files}"
        )
    indices = [int(index) for index in split_ids[args.split_key]]
    if len(indices) < args.num_samples:
        raise ValueError(
            f"Requested {args.num_samples} samples, but {split_path}:{args.split_key} "
            f"contains only {len(indices)} indices."
        )
    return indices[: args.num_samples]


def build_train_compat_instruction(row: dict[str, Any]) -> str:
    instruction = clean_text(row.get("instruction", ""))
    input_text = clean_text(row.get("input", ""))
    if input_text:
        return f"{instruction}\n\nInput: {input_text}"
    return instruction


def write_train_compat_csv(rows: list[dict[str, Any]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["index", "instruction", "alpaca_reference", "suffix"],
        )
        writer.writeheader()
        for idx, row in enumerate(rows):
            writer.writerow(
                {
                    "index": idx,
                    "instruction": build_train_compat_instruction(row),
                    "alpaca_reference": clean_text(row.get("output", "")),
                    "suffix": "",
                }
            )


def load_existing_rows(output_path: Path) -> dict[int, dict[str, Any]]:
    if not output_path.exists():
        return {}
    with output_path.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    existing = {}
    for row in rows:
        try:
            sample_index = int(row["sample_index"])
        except Exception:
            continue
        if clean_text(row.get("output", "")):
            existing[sample_index] = row
    return existing


def write_outputs(
    rows: list[dict[str, Any]],
    output_path: Path,
    metadata_path: Path,
    metadata: dict[str, Any],
    csv_output: str | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    if csv_output is not None:
        write_train_compat_csv(rows, Path(csv_output))
        metadata["csv_output"] = str(csv_output)

    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def build_rewritten_row(
    example: dict[str, Any],
    source_idx: int,
    local_idx: int,
    args: argparse.Namespace,
    rewritten_output: str,
) -> dict[str, Any]:
    original_output = clean_text(example.get("output", ""))
    rewritten_output = clean_text(rewritten_output)
    return {
        "instruction": clean_text(example.get("instruction", "")),
        "input": clean_text(example.get("input", "")),
        "output": rewritten_output,
        "original_output": original_output,
        "source_dataset": args.dataset_name,
        "source_split": args.split,
        "source_index": source_idx,
        "sample_index": local_idx,
        "rewrite_mode": args.rewrite_mode,
        "rewrite_backend": args.backend,
        "rewrite_model": (
            args.openai_model
            if args.backend == "openai"
            else args.local_model_path
            if args.backend == "transformers"
            else None
        ),
        "original_word_count": word_count(original_output),
        "rewritten_word_count": word_count(rewritten_output),
    }


def current_metadata(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    target_num_samples: int,
    output_path: Path,
) -> dict[str, Any]:
    total_original_words = sum(row["original_word_count"] for row in rows)
    total_rewritten_words = sum(row["rewritten_word_count"] for row in rows)
    return {
        "dataset_name": args.dataset_name,
        "split": args.split,
        "num_samples": len(rows),
        "target_num_samples": target_num_samples,
        "selection": args.selection,
        "split_id_list_path": args.split_id_list_path if args.selection == "cached-split" else None,
        "split_key": args.split_key if args.selection == "cached-split" else None,
        "seed": args.seed,
        "rewrite_mode": args.rewrite_mode,
        "rewrite_backend": args.backend,
        "openai_model": args.openai_model if args.backend == "openai" else None,
        "local_model_path": args.local_model_path if args.backend == "transformers" else None,
        "output": str(output_path),
        "average_original_words": total_original_words / max(1, len(rows)),
        "average_rewritten_words": total_rewritten_words / max(1, len(rows)),
        "complete": len(rows) == target_num_samples,
    }


def call_openai_with_retries(client: Any, args: argparse.Namespace, example: dict[str, Any]) -> str:
    last_exc = None
    for attempt in range(args.max_retries):
        try:
            rewritten = openai_rewrite(
                client,
                args.openai_model,
                example,
                rewrite_mode=args.rewrite_mode,
                temperature=args.temperature,
                max_words=args.max_words,
                max_output_tokens=args.max_output_tokens,
            )
            if rewritten:
                return rewritten
            raise RuntimeError("empty rewritten response")
        except Exception as exc:
            last_exc = exc
            delay = min(60, (2**attempt) + random.random())
            print(
                f"OpenAI rewrite failed on attempt {attempt + 1}/{args.max_retries}: {exc}. "
                f"Sleeping {delay:.1f}s.",
                file=sys.stderr,
            )
            time.sleep(delay)
    raise RuntimeError(f"OpenAI rewrite failed after {args.max_retries} attempts") from last_exc


def run_transformers_rewrite(
    args: argparse.Namespace,
    source_split: Any,
    indices: list[int],
    rewritten_rows_by_index: dict[int, dict[str, Any]],
    output_path: Path,
    metadata_path: Path,
) -> list[dict[str, Any]]:
    tokenizer, model = load_transformers_backend(args)
    missing = [
        (local_idx, source_idx, dict(source_split[source_idx]))
        for local_idx, source_idx in enumerate(indices)
        if local_idx not in rewritten_rows_by_index
    ]
    print(
        f"Running local transformers rewrite with {args.local_model_path}; "
        f"missing={len(missing)}, batch_size={args.local_batch_size}"
    )
    new_rows_since_checkpoint = 0
    for start in tqdm(range(0, len(missing), args.local_batch_size), desc="transformers rewriting"):
        batch = missing[start : start + args.local_batch_size]
        examples = [item[2] for item in batch]
        outputs = transformers_rewrite_batch(tokenizer, model, args, examples)
        for (local_idx, source_idx, example), rewritten_output in zip(batch, outputs):
            if not rewritten_output:
                rewritten_output = clean_text(example.get("output", ""))
            rewritten_rows_by_index[local_idx] = build_rewritten_row(
                example,
                source_idx,
                local_idx,
                args,
                rewritten_output,
            )
            new_rows_since_checkpoint += 1

        if args.checkpoint_every > 0 and new_rows_since_checkpoint >= args.checkpoint_every:
            partial_rows = [rewritten_rows_by_index[index] for index in sorted(rewritten_rows_by_index)]
            metadata = current_metadata(args, partial_rows, len(indices), output_path)
            write_outputs(partial_rows, output_path, metadata_path, metadata, args.csv_output)
            new_rows_since_checkpoint = 0

    return [rewritten_rows_by_index[index] for index in sorted(rewritten_rows_by_index)]


def main() -> None:
    args = parse_args()
    if args.output is None:
        args.output = f"data/alpaca_cleaned_{args.rewrite_mode}_{args.num_samples}.json"
    dataset = load_dataset(args.dataset_name)
    if args.split not in dataset:
        raise ValueError(f"Split {args.split!r} not found in {args.dataset_name}.")

    source_split = dataset[args.split]
    indices = selected_indices(args, len(source_split))
    output_path = Path(args.output)
    metadata_path = (
        Path(args.metadata_output)
        if args.metadata_output is not None
        else output_path.with_suffix(output_path.suffix + ".metadata.json")
    )

    client = None
    if args.backend == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required for --backend openai.")
        from openai import OpenAI

        client = OpenAI()
    elif args.backend == "transformers" and not Path(args.local_model_path).exists():
        raise FileNotFoundError(f"Local model path not found: {args.local_model_path}")

    existing_rows = load_existing_rows(output_path) if args.resume else {}
    if existing_rows:
        print(f"Resuming from {output_path}: found {len(existing_rows)} completed rows.")
    rewritten_rows_by_index: dict[int, dict[str, Any]] = dict(existing_rows)

    if args.backend == "transformers":
        rewritten_rows = run_transformers_rewrite(
            args,
            source_split,
            indices,
            rewritten_rows_by_index,
            output_path,
            metadata_path,
        )
        metadata = current_metadata(args, rewritten_rows, len(indices), output_path)
        write_outputs(rewritten_rows, output_path, metadata_path, metadata, args.csv_output)
        print(f"Wrote {len(rewritten_rows)} {args.rewrite_mode} samples to {output_path}")
        if args.csv_output is not None:
            print(f"Wrote train-compatible CSV to {args.csv_output}")
        print(f"Wrote metadata to {metadata_path}")
        print(
            "Average words: "
            f"{metadata['average_original_words']:.1f} -> {metadata['average_rewritten_words']:.1f}"
        )
        return

    new_rows_since_checkpoint = 0

    for local_idx, source_idx in enumerate(tqdm(indices, desc="rewriting")):
        if local_idx in rewritten_rows_by_index:
            continue
        example = dict(source_split[source_idx])
        original_output = clean_text(example.get("output", ""))
        if args.backend == "openai":
            rewritten_output = call_openai_with_retries(client, args, example)
            if args.sleep > 0:
                time.sleep(args.sleep)
        else:
            rewritten_output = heuristic_rewrite(args, example)

        rewritten_rows_by_index[local_idx] = build_rewritten_row(
            example,
            source_idx,
            local_idx,
            args,
            rewritten_output,
        )
        new_rows_since_checkpoint += 1

        if args.checkpoint_every > 0 and new_rows_since_checkpoint >= args.checkpoint_every:
            partial_rows = [rewritten_rows_by_index[index] for index in sorted(rewritten_rows_by_index)]
            partial_metadata = current_metadata(args, partial_rows, len(indices), output_path)
            write_outputs(partial_rows, output_path, metadata_path, partial_metadata, args.csv_output)
            new_rows_since_checkpoint = 0

    rewritten_rows = [rewritten_rows_by_index[index] for index in sorted(rewritten_rows_by_index)]
    metadata = current_metadata(args, rewritten_rows, len(indices), output_path)
    write_outputs(rewritten_rows, output_path, metadata_path, metadata, args.csv_output)

    print(f"Wrote {len(rewritten_rows)} {args.rewrite_mode} samples to {output_path}")
    if args.csv_output is not None:
        print(f"Wrote train-compatible CSV to {args.csv_output}")
    print(f"Wrote metadata to {metadata_path}")
    print(
        "Average words: "
        f"{metadata['average_original_words']:.1f} -> {metadata['average_rewritten_words']:.1f}"
    )


if __name__ == "__main__":
    main()
