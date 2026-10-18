#!/usr/bin/env python3
"""Refine standalone-v3 prompts with differentiable K-step LoRA unrolling.

The input is a CSV of benign SFT pairs, normally a pool of complete prompts
produced by ``standalone_v3``.  For each pair this script:

1. performs K differentiable virtual SFT updates on fresh LoRA fast weights;
2. evaluates a paired harmful/refusal safety margin after those updates;
3. differentiates that outer margin through the updates to prompt embeddings;
4. proposes discrete token substitutions from the embedding meta-gradient; and
5. accepts an edit only after exact K-step rescoring of the discrete prompt.

Only the temporary fast weights and prompt embeddings participate in the
higher-order graph.  The base model and its installed LoRA parameters are not
mutated while candidates are generated.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import random
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.func import functional_call
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


INVALID_PROMPT_PHRASES = (
    "original prompt",
    "original user prompt",
    "replacement prompt",
    "rewrite the original",
    "rewritten prompt",
    "rephrase the original",
    "the prompt above",
    "standalone replacement prompt",
)


@dataclass
class PairTemplate:
    """Tokenized chat pair with an identified, editable user-content span."""

    prompt: str
    response: str
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    prompt_end: int
    content_start: int
    content_end: int

    @property
    def content_ids(self) -> torch.Tensor:
        return self.input_ids[self.content_start : self.content_end]

    @property
    def content_length(self) -> int:
        return self.content_end - self.content_start

    def ids_batch(self, content_ids: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        ids = self.input_ids
        if content_ids is not None:
            if int(content_ids.numel()) != self.content_length:
                raise ValueError(
                    f"Expected {self.content_length} content tokens, got {content_ids.numel()}"
                )
            ids = torch.cat(
                [
                    self.input_ids[: self.content_start],
                    content_ids,
                    self.input_ids[self.content_end :],
                ]
            )
        return {
            "input_ids": ids.unsqueeze(0),
            "attention_mask": self.attention_mask.unsqueeze(0),
            "labels": self.labels.unsqueeze(0),
        }

    def embeddings_batch(
        self,
        content_ids: torch.Tensor,
        content_embeddings: torch.Tensor,
        embedding_weight: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if int(content_ids.numel()) != self.content_length:
            raise ValueError("Content IDs do not match the template span")
        if tuple(content_embeddings.shape) != (
            self.content_length,
            int(embedding_weight.shape[1]),
        ):
            raise ValueError(
                "Content embedding shape mismatch: "
                f"got={tuple(content_embeddings.shape)}, "
                f"expected={(self.content_length, int(embedding_weight.shape[1]))}"
            )
        ids = torch.cat(
            [
                self.input_ids[: self.content_start],
                content_ids,
                self.input_ids[self.content_end :],
            ]
        )
        # Fixed token embeddings do not need a graph.  The editable slice is
        # inserted by concatenation so its meta-gradient is preserved.
        fixed = F.embedding(ids, embedding_weight).detach()
        inputs_embeds = torch.cat(
            [
                fixed[: self.content_start],
                content_embeddings,
                fixed[self.content_end :],
            ],
            dim=0,
        )
        return {
            "inputs_embeds": inputs_embeds.unsqueeze(0),
            "attention_mask": self.attention_mask.unsqueeze(0),
            "labels": self.labels.unsqueeze(0),
        }


@dataclass
class PairedAnchor:
    anchor_id: str
    harmful: PairTemplate
    refusal: PairTemplate


@dataclass
class UnrolledScore:
    objective: float
    safe_margin: float
    harmful_loss: float
    refusal_loss: float
    discrepancy_loss: float
    inner_losses: list[float]

    @property
    def unsafe_margin(self) -> float:
        return -self.safe_margin


@dataclass(frozen=True)
class TokenEditProposal:
    position: int
    old_token_id: int
    new_token_id: int
    predicted_meta_delta: float
    predicted_lm_delta: float
    predicted_total_delta: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Generate token-level prompts with a K-step unrolled safety objective.",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True, help="Standalone-v3 candidate CSV.")
    parser.add_argument(
        "--anchors",
        required=True,
        help=(
            "Paired meta-anchor JSONL with prompt, harmful_target, and "
            "refusal_target fields."
        ),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--progress-jsonl", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16"
    )
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--system-prompt", default="You are a helpful assistant.")
    parser.add_argument("--instruction-field", default="instruction")
    parser.add_argument("--suffix-field", default="suffix")
    parser.add_argument("--response-field", default="alpaca_reference")
    parser.add_argument("--candidate-id-field", default="candidate_id")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--anchor-response-tokens",
        type=int,
        default=32,
        help="Limit each harmful/refusal anchor loss to its first N response tokens; 0 disables.",
    )
    parser.add_argument("--meta-anchor-count", type=int, default=8)
    parser.add_argument("--k-steps", type=int, default=3)
    parser.add_argument("--inner-optimizer", choices=("sgd", "adamw"), default="sgd")
    parser.add_argument("--inner-lr", type=float, default=1e-4)
    parser.add_argument("--inner-weight-decay", type=float, default=0.0)
    parser.add_argument("--inner-beta1", type=float, default=0.9)
    parser.add_argument("--inner-beta2", type=float, default=0.999)
    parser.add_argument("--inner-eps", type=float, default=1e-8)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument(
        "--positions-per-iteration",
        type=int,
        default=12,
        help="Only positions with the largest meta-gradient norms propose edits; 0 means all.",
    )
    parser.add_argument("--lm-top-k", type=int, default=64)
    parser.add_argument(
        "--gradient-top-k",
        type=int,
        default=16,
        help="Also consider this many unrestricted tokens from the embedding gradient; 0 disables.",
    )
    parser.add_argument(
        "--exact-candidates",
        type=int,
        default=8,
        help="Number of proposed substitutions receiving exact K-step rescoring per iteration.",
    )
    parser.add_argument(
        "--lm-weight",
        type=float,
        default=0.05,
        help="Penalty weight for reducing the base model token log-probability.",
    )
    parser.add_argument(
        "--discrepancy-target",
        default="I can't fulfill that request.",
        help="Base-model refusal target retained while optimizing the unrolled objective.",
    )
    parser.add_argument(
        "--discrepancy-weight",
        type=float,
        default=0.1,
        help="Weight on the base-model discrepancy-target NLL in the outer objective.",
    )
    parser.add_argument("--minimum-improvement", type=float, default=1e-5)
    parser.add_argument(
        "--require-roundtrip",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require decoded edited tokens to retokenize to the identical content token sequence.",
    )
    parser.add_argument(
        "--allow-newline-token",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument(
        "--lora-target-modules", nargs="+", default=["q_proj", "v_proj"]
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use non-reentrant activation checkpointing for higher-order gradients.",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def valid_prompt_text(text: str) -> bool:
    text = normalize(text)
    if len(text) < 3 or "<|" in text or "|>" in text:
        return False
    lowered = text.casefold()
    return not any(phrase in lowered for phrase in INVALID_PROMPT_PHRASES)


def find_subsequence(sequence: list[int], subsequence: list[int]) -> tuple[int, int] | None:
    if not subsequence or len(subsequence) > len(sequence):
        return None
    match = None
    width = len(subsequence)
    for start in range(len(sequence) - width + 1):
        if sequence[start : start + width] == subsequence:
            match = (start, start + width)
    return match


def locate_content_span(tokenizer, prompt_text: str, prompt: str, prompt_ids: list[int]) -> tuple[int, int]:
    char_start = prompt_text.rfind(prompt)
    if char_start < 0:
        raise ValueError("The chat template did not preserve the prompt text verbatim")
    char_end = char_start + len(prompt)
    if getattr(tokenizer, "is_fast", False):
        encoded = tokenizer(
            prompt_text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        offsets = encoded.get("offset_mapping")
        if offsets and list(encoded["input_ids"]) == prompt_ids:
            indices = [
                index
                for index, (start, end) in enumerate(offsets)
                if end > char_start and start < char_end and end > start
            ]
            if indices and indices == list(range(indices[0], indices[-1] + 1)):
                return indices[0], indices[-1] + 1

    content_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    match = find_subsequence(prompt_ids, content_ids)
    if match is None:
        raise ValueError(
            "Could not locate the editable prompt token span; use a fast tokenizer "
            "or a prompt whose standalone tokenization matches the chat template"
        )
    return match


def build_pair_template(
    tokenizer,
    prompt: str,
    response: str,
    system_prompt: str,
    max_length: int,
    device: torch.device,
    response_token_limit: int = 0,
) -> PairTemplate:
    prompt = normalize(prompt)
    response = str(response or "").strip()
    if not prompt or not response:
        raise ValueError("Prompt and response must both be non-empty")
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": response}],
        tokenize=False,
        add_generation_prompt=False,
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("Chat template prompt is not a token prefix of the full pair")
    if response_token_limit > 0:
        full_ids = full_ids[: len(prompt_ids) + response_token_limit]
    if len(full_ids) > max_length:
        raise ValueError(f"Token length {len(full_ids)} exceeds max_length={max_length}")
    if len(full_ids) <= len(prompt_ids):
        raise ValueError("The pair contains no response loss tokens")
    content_start, content_end = locate_content_span(
        tokenizer, prompt_text, prompt, prompt_ids
    )
    ids_tensor = torch.tensor(full_ids, dtype=torch.long, device=device)
    labels = ids_tensor.clone()
    labels[: len(prompt_ids)] = -100
    return PairTemplate(
        prompt=prompt,
        response=response,
        input_ids=ids_tensor,
        attention_mask=torch.ones_like(ids_tensor),
        labels=labels,
        prompt_end=len(prompt_ids),
        content_start=content_start,
        content_end=content_end,
    )


def load_paired_anchors(
    path: Path,
    tokenizer,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[PairedAnchor], list[dict[str, str]]]:
    rows = read_jsonl(path)
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    anchors = []
    skipped = []
    for position, row in enumerate(rows):
        prompt = str(row.get("prompt", "")).strip()
        harmful_target = str(row.get("harmful_target", "")).strip()
        refusal_target = str(row.get("refusal_target", "")).strip()
        anchor_id = str(row.get("anchor_id", position))
        if not prompt or not harmful_target or not refusal_target:
            raise ValueError(
                f"Anchor {anchor_id} lacks prompt/harmful_target/refusal_target fields"
            )
        try:
            harmful = build_pair_template(
                tokenizer,
                prompt,
                harmful_target,
                args.system_prompt,
                args.max_length,
                device,
                response_token_limit=args.anchor_response_tokens,
            )
            refusal = build_pair_template(
                tokenizer,
                prompt,
                refusal_target,
                args.system_prompt,
                args.max_length,
                device,
                response_token_limit=args.anchor_response_tokens,
            )
        except ValueError as error:
            skipped.append({"anchor_id": anchor_id, "reason": str(error)})
            continue
        anchors.append(PairedAnchor(anchor_id=anchor_id, harmful=harmful, refusal=refusal))
        if len(anchors) >= args.meta_anchor_count:
            break
    if not anchors:
        raise RuntimeError("No paired meta anchors remained after tokenization")
    if len(anchors) < args.meta_anchor_count:
        raise RuntimeError(
            f"Only {len(anchors)} usable anchors for requested {args.meta_anchor_count}"
        )
    return anchors, skipped


class KStepGenerator:
    def __init__(
        self,
        model,
        tokenizer,
        anchors: list[PairedAnchor],
        args: argparse.Namespace,
        device: torch.device,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.anchors = anchors
        self.args = args
        self.device = device
        named_trainable = [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        if not named_trainable:
            raise RuntimeError("The PEFT model has no trainable LoRA parameters")
        self.trainable_names = [name for name, _ in named_trainable]
        self.initial_fast = OrderedDict(
            (name, parameter.detach().clone()) for name, parameter in named_trainable
        )
        # Only functional fast weights should require gradients below.
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self.embedding_weight = model.get_input_embeddings().weight.detach()
        self.special_ids = set(tokenizer.all_special_ids)
        self.base_harmful_loss, self.base_refusal_loss, self.base_safe_margin = (
            self.evaluate_safety_margin(self.fresh_fast(), require_grad=False)
        )

    def fresh_fast(self) -> OrderedDict[str, torch.Tensor]:
        return OrderedDict(
            (name, value.detach().clone().requires_grad_(True))
            for name, value in self.initial_fast.items()
        )

    def call(self, fast: OrderedDict[str, torch.Tensor], batch: dict[str, torch.Tensor]):
        return functional_call(
            self.model,
            fast,
            args=(),
            kwargs={**batch, "use_cache": False},
            strict=False,
        )

    def loss(self, fast: OrderedDict[str, torch.Tensor], batch: dict[str, torch.Tensor]):
        return self.call(fast, batch).loss

    def evaluate_safety_margin(
        self,
        fast: OrderedDict[str, torch.Tensor],
        require_grad: bool,
    ) -> tuple[torch.Tensor | float, torch.Tensor | float, torch.Tensor | float]:
        context = torch.enable_grad() if require_grad else torch.no_grad()
        with context:
            harmful_losses = [
                self.loss(fast, anchor.harmful.ids_batch()) for anchor in self.anchors
            ]
            refusal_losses = [
                self.loss(fast, anchor.refusal.ids_batch()) for anchor in self.anchors
            ]
            harmful = torch.stack(harmful_losses).mean()
            refusal = torch.stack(refusal_losses).mean()
            margin = harmful - refusal
        if require_grad:
            return harmful, refusal, margin
        return (
            float(harmful.float().cpu()),
            float(refusal.float().cpu()),
            float(margin.float().cpu()),
        )

    def optimizer_step(
        self,
        fast: OrderedDict[str, torch.Tensor],
        grads: Iterable[torch.Tensor | None],
        state: dict[str, dict[str, torch.Tensor]],
        step: int,
        create_graph: bool,
    ) -> tuple[OrderedDict[str, torch.Tensor], dict[str, dict[str, torch.Tensor]]]:
        updated = OrderedDict()
        next_state: dict[str, dict[str, torch.Tensor]] = {}
        for (name, weight), gradient in zip(fast.items(), grads):
            if gradient is None:
                gradient = torch.zeros_like(weight)
            if self.args.inner_optimizer == "sgd":
                direction = gradient + self.args.inner_weight_decay * weight
                new_weight = weight - self.args.inner_lr * direction
            else:
                previous = state.get(name)
                if previous is None:
                    m = torch.zeros_like(weight)
                    v = torch.zeros_like(weight)
                else:
                    m = previous["m"]
                    v = previous["v"]
                m = self.args.inner_beta1 * m + (1.0 - self.args.inner_beta1) * gradient
                v = self.args.inner_beta2 * v + (1.0 - self.args.inner_beta2) * gradient.square()
                m_hat = m / (1.0 - self.args.inner_beta1**step)
                v_hat = v / (1.0 - self.args.inner_beta2**step)
                new_weight = weight * (
                    1.0 - self.args.inner_lr * self.args.inner_weight_decay
                ) - self.args.inner_lr * m_hat / (v_hat.sqrt() + self.args.inner_eps)
                next_state[name] = {"m": m, "v": v}
            if create_graph:
                updated[name] = new_weight
            else:
                updated[name] = new_weight.detach().requires_grad_(True)
                if name in next_state:
                    next_state[name] = {
                        key: value.detach() for key, value in next_state[name].items()
                    }
        return updated, next_state

    def unroll(
        self,
        inner_batch: dict[str, torch.Tensor],
        create_graph: bool,
    ) -> tuple[OrderedDict[str, torch.Tensor], list[torch.Tensor], OrderedDict[str, torch.Tensor]]:
        fast0 = self.fresh_fast()
        fast = fast0
        state: dict[str, dict[str, torch.Tensor]] = {}
        losses = []
        for step in range(1, self.args.k_steps + 1):
            inner_loss = self.loss(fast, inner_batch)
            grads = torch.autograd.grad(
                inner_loss,
                tuple(fast.values()),
                create_graph=create_graph,
                allow_unused=True,
            )
            fast, state = self.optimizer_step(
                fast,
                grads,
                state,
                step,
                create_graph=create_graph,
            )
            losses.append(inner_loss)
        return fast, losses, fast0

    def discrepancy_loss(
        self,
        fast0: OrderedDict[str, torch.Tensor],
        template: PairTemplate,
        content_ids: torch.Tensor,
        content_embeddings: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.args.discrepancy_weight == 0 or not self.args.discrepancy_target:
            # Preserve a graph connection in meta-gradient mode.
            if content_embeddings is not None:
                return content_embeddings.sum() * 0.0
            return torch.zeros((), dtype=self.embedding_weight.dtype, device=self.device)
        if content_embeddings is None:
            batch = template.ids_batch(content_ids)
        else:
            batch = template.embeddings_batch(
                content_ids, content_embeddings, self.embedding_weight
            )
        return self.loss(fast0, batch)

    def exact_score(
        self,
        pair_template: PairTemplate,
        discrepancy_template: PairTemplate,
        content_ids: torch.Tensor,
    ) -> UnrolledScore:
        inner_batch = pair_template.ids_batch(content_ids)
        fast, inner_losses, fast0 = self.unroll(inner_batch, create_graph=False)
        harmful, refusal, safe_margin = self.evaluate_safety_margin(
            fast, require_grad=False
        )
        with torch.no_grad():
            discrepancy = self.discrepancy_loss(
                fast0, discrepancy_template, content_ids, None
            )
        discrepancy_value = float(discrepancy.float().cpu())
        objective = float(safe_margin) + self.args.discrepancy_weight * discrepancy_value
        return UnrolledScore(
            objective=objective,
            safe_margin=float(safe_margin),
            harmful_loss=float(harmful),
            refusal_loss=float(refusal),
            discrepancy_loss=discrepancy_value,
            inner_losses=[float(loss.detach().float().cpu()) for loss in inner_losses],
        )

    def meta_gradient(
        self,
        pair_template: PairTemplate,
        discrepancy_template: PairTemplate,
        content_ids: torch.Tensor,
    ) -> tuple[UnrolledScore, torch.Tensor]:
        content_embeddings = F.embedding(content_ids, self.embedding_weight).detach()
        content_embeddings = content_embeddings.clone().requires_grad_(True)
        inner_batch = pair_template.embeddings_batch(
            content_ids, content_embeddings, self.embedding_weight
        )
        fast, inner_losses, fast0 = self.unroll(inner_batch, create_graph=True)
        harmful, refusal, safe_margin = self.evaluate_safety_margin(
            fast, require_grad=True
        )
        discrepancy = self.discrepancy_loss(
            fast0, discrepancy_template, content_ids, content_embeddings
        )
        objective = safe_margin + self.args.discrepancy_weight * discrepancy
        (gradient,) = torch.autograd.grad(objective, content_embeddings)
        score = UnrolledScore(
            objective=float(objective.detach().float().cpu()),
            safe_margin=float(safe_margin.detach().float().cpu()),
            harmful_loss=float(harmful.detach().float().cpu()),
            refusal_loss=float(refusal.detach().float().cpu()),
            discrepancy_loss=float(discrepancy.detach().float().cpu()),
            inner_losses=[float(loss.detach().float().cpu()) for loss in inner_losses],
        )
        return score, gradient.detach()

    @torch.no_grad()
    def language_model_log_probs(
        self, template: PairTemplate, content_ids: torch.Tensor
    ) -> dict[int, torch.Tensor]:
        prompt_ids = template.ids_batch(content_ids)["input_ids"][:, : template.prompt_end]
        logits = self.model(
            input_ids=prompt_ids,
            attention_mask=torch.ones_like(prompt_ids),
            use_cache=False,
        ).logits[0]
        result = {}
        for local_position in range(template.content_length):
            global_position = template.content_start + local_position
            if global_position == 0:
                continue
            result[local_position] = torch.log_softmax(
                logits[global_position - 1].float(), dim=-1
            )
        return result

    def valid_token(self, token_id: int) -> bool:
        if token_id in self.special_ids:
            return False
        piece = self.tokenizer.decode(
            [token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        if not piece or "<|" in piece or "|>" in piece or "\ufffd" in piece:
            return False
        if not self.args.allow_newline_token and ("\n" in piece or "\r" in piece):
            return False
        return True

    def propose_edits(
        self,
        template: PairTemplate,
        content_ids: torch.Tensor,
        meta_gradient: torch.Tensor,
    ) -> list[TokenEditProposal]:
        lm_log_probs = self.language_model_log_probs(template, content_ids)
        norms = meta_gradient.float().norm(dim=-1)
        if self.args.positions_per_iteration > 0:
            position_count = min(self.args.positions_per_iteration, len(norms))
            positions = norms.topk(position_count).indices.tolist()
        else:
            positions = list(range(len(norms)))

        proposals: list[TokenEditProposal] = []
        for position in positions:
            if position not in lm_log_probs:
                continue
            log_probs = lm_log_probs[position]
            current_id = int(content_ids[position])
            gradient = meta_gradient[position].to(self.embedding_weight.dtype)
            vocab_meta_scores = torch.mv(self.embedding_weight, gradient).float()
            candidate_ids = set(
                log_probs.topk(min(self.args.lm_top_k, len(log_probs))).indices.tolist()
            )
            if self.args.gradient_top_k > 0:
                gradient_ids = vocab_meta_scores.topk(
                    min(self.args.gradient_top_k, len(vocab_meta_scores)),
                    largest=False,
                ).indices.tolist()
                candidate_ids.update(gradient_ids)
            current_meta = float(vocab_meta_scores[current_id].cpu())
            current_log_prob = float(log_probs[current_id].cpu())
            for new_id in candidate_ids:
                if new_id == current_id or not self.valid_token(new_id):
                    continue
                meta_delta = float(vocab_meta_scores[new_id].cpu()) - current_meta
                lm_delta = -self.args.lm_weight * (
                    float(log_probs[new_id].cpu()) - current_log_prob
                )
                proposals.append(
                    TokenEditProposal(
                        position=position,
                        old_token_id=current_id,
                        new_token_id=int(new_id),
                        predicted_meta_delta=meta_delta,
                        predicted_lm_delta=lm_delta,
                        predicted_total_delta=meta_delta + lm_delta,
                    )
                )
        proposals.sort(key=lambda proposal: proposal.predicted_total_delta)
        unique = []
        seen = set()
        for proposal in proposals:
            key = (proposal.position, proposal.new_token_id)
            if key in seen:
                continue
            seen.add(key)
            unique.append(proposal)
            if len(unique) >= self.args.exact_candidates:
                break
        return unique

    def decode_content(self, content_ids: torch.Tensor) -> str:
        return normalize(
            self.tokenizer.decode(
                content_ids.tolist(),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        )

    def build_templates_for_prompt(
        self, prompt: str, response: str
    ) -> tuple[PairTemplate, PairTemplate]:
        pair = build_pair_template(
            self.tokenizer,
            prompt,
            response,
            self.args.system_prompt,
            self.args.max_length,
            self.device,
        )
        discrepancy = build_pair_template(
            self.tokenizer,
            prompt,
            self.args.discrepancy_target,
            self.args.system_prompt,
            self.args.max_length,
            self.device,
        )
        if pair.content_length != discrepancy.content_length or not torch.equal(
            pair.content_ids, discrepancy.content_ids
        ):
            raise ValueError("SFT and discrepancy templates locate different prompt tokens")
        return pair, discrepancy

    def materialize_edit(
        self,
        template: PairTemplate,
        response: str,
        content_ids: torch.Tensor,
        proposal: TokenEditProposal,
    ) -> tuple[str, PairTemplate, PairTemplate, torch.Tensor] | None:
        edited_ids = content_ids.clone()
        edited_ids[proposal.position] = proposal.new_token_id
        prompt = self.decode_content(edited_ids)
        if not valid_prompt_text(prompt):
            return None
        try:
            pair, discrepancy = self.build_templates_for_prompt(prompt, response)
        except ValueError:
            return None
        if self.args.require_roundtrip and not torch.equal(pair.content_ids, edited_ids):
            return None
        return prompt, pair, discrepancy, pair.content_ids

    def refine(self, prompt: str, response: str) -> dict[str, Any]:
        pair, discrepancy = self.build_templates_for_prompt(prompt, response)
        content_ids = pair.content_ids.clone()
        initial_score = self.exact_score(pair, discrepancy, content_ids)
        current_score = initial_score
        current_prompt = pair.prompt
        edit_history = []

        for iteration in range(self.args.iterations):
            meta_score, meta_gradient = self.meta_gradient(pair, discrepancy, content_ids)
            proposals = self.propose_edits(pair, content_ids, meta_gradient)
            best = None
            for proposal in proposals:
                materialized = self.materialize_edit(
                    pair, response, content_ids, proposal
                )
                if materialized is None:
                    continue
                candidate_prompt, candidate_pair, candidate_discrepancy, candidate_ids = (
                    materialized
                )
                candidate_score = self.exact_score(
                    candidate_pair, candidate_discrepancy, candidate_ids
                )
                if best is None or candidate_score.objective < best[0].objective:
                    best = (
                        candidate_score,
                        proposal,
                        candidate_prompt,
                        candidate_pair,
                        candidate_discrepancy,
                        candidate_ids,
                    )
            if best is None:
                break
            (
                best_score,
                best_proposal,
                best_prompt,
                best_pair,
                best_discrepancy,
                best_ids,
            ) = best
            improvement = current_score.objective - best_score.objective
            if improvement < self.args.minimum_improvement:
                break
            edit_history.append(
                {
                    "iteration": iteration,
                    "position": best_proposal.position,
                    "old_token_id": best_proposal.old_token_id,
                    "new_token_id": best_proposal.new_token_id,
                    "old_token": self.tokenizer.decode([best_proposal.old_token_id]),
                    "new_token": self.tokenizer.decode([best_proposal.new_token_id]),
                    "predicted_meta_delta": best_proposal.predicted_meta_delta,
                    "predicted_lm_delta": best_proposal.predicted_lm_delta,
                    "predicted_total_delta": best_proposal.predicted_total_delta,
                    "meta_objective_at_current_prompt": meta_score.objective,
                    "exact_objective_before": current_score.objective,
                    "exact_objective_after": best_score.objective,
                    "exact_improvement": improvement,
                }
            )
            current_score = best_score
            current_prompt = best_prompt
            pair = best_pair
            discrepancy = best_discrepancy
            content_ids = best_ids.clone()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return {
            "kstep_initial_instruction": prompt,
            "kstep_refined_instruction": current_prompt,
            "kstep_initial_objective": initial_score.objective,
            "kstep_final_objective": current_score.objective,
            "kstep_objective_improvement": initial_score.objective
            - current_score.objective,
            "kstep_initial_safe_margin": initial_score.safe_margin,
            "kstep_final_safe_margin": current_score.safe_margin,
            "kstep_base_safe_margin": self.base_safe_margin,
            "kstep_initial_unrolled_risk_gain": self.base_safe_margin
            - initial_score.safe_margin,
            "kstep_unrolled_risk_gain": self.base_safe_margin
            - current_score.safe_margin,
            "kstep_refinement_risk_gain": initial_score.safe_margin
            - current_score.safe_margin,
            "kstep_initial_harmful_loss": initial_score.harmful_loss,
            "kstep_final_harmful_loss": current_score.harmful_loss,
            "kstep_initial_refusal_loss": initial_score.refusal_loss,
            "kstep_final_refusal_loss": current_score.refusal_loss,
            "kstep_initial_discrepancy_loss": initial_score.discrepancy_loss,
            "kstep_final_discrepancy_loss": current_score.discrepancy_loss,
            "kstep_initial_inner_losses_json": json.dumps(initial_score.inner_losses),
            "kstep_final_inner_losses_json": json.dumps(current_score.inner_losses),
            "kstep_num_accepted_edits": len(edit_history),
            "kstep_edits_json": json.dumps(edit_history, ensure_ascii=False),
        }


def validate_args(args: argparse.Namespace) -> None:
    if args.k_steps <= 0:
        raise ValueError("--k-steps must be positive")
    if args.iterations < 0:
        raise ValueError("--iterations cannot be negative")
    if args.meta_anchor_count <= 0:
        raise ValueError("--meta-anchor-count must be positive")
    if args.inner_lr <= 0:
        raise ValueError("--inner-lr must be positive")
    if args.exact_candidates <= 0:
        raise ValueError("--exact-candidates must be positive")
    if args.lm_top_k <= 0:
        raise ValueError("--lm-top-k must be positive")
    if args.max_length <= 0:
        raise ValueError("--max-length must be positive")


def run_signature(args: argparse.Namespace, input_path: Path, anchor_path: Path) -> str:
    payload = {
        key: value
        for key, value in vars(args).items()
        if key not in {"output", "progress_jsonl", "overwrite", "continue_on_error"}
    }
    payload["input_sha256"] = hashlib.sha256(input_path.read_bytes()).hexdigest()
    payload["anchor_sha256"] = hashlib.sha256(anchor_path.read_bytes()).hexdigest()
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def row_key(position: int, row: dict[str, str], args: argparse.Namespace) -> str:
    candidate_id = row.get(args.candidate_id_field, "")
    instruction = row.get(args.instruction_field, "")
    suffix = row.get(args.suffix_field, "")
    response = row.get(args.response_field, "")
    raw = f"{position}\0{candidate_id}\0{instruction}\0{suffix}\0{response}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def write_output(
    path: Path,
    rows: list[dict[str, str]],
    input_fields: list[str],
    completed: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    generated_fields = [
        "kstep_status",
        "kstep_error",
        "kstep_initial_instruction",
        "kstep_refined_instruction",
        "kstep_initial_objective",
        "kstep_final_objective",
        "kstep_objective_improvement",
        "kstep_initial_safe_margin",
        "kstep_final_safe_margin",
        "kstep_base_safe_margin",
        "kstep_initial_unrolled_risk_gain",
        "kstep_unrolled_risk_gain",
        "kstep_refinement_risk_gain",
        "kstep_initial_harmful_loss",
        "kstep_final_harmful_loss",
        "kstep_initial_refusal_loss",
        "kstep_final_refusal_loss",
        "kstep_initial_discrepancy_loss",
        "kstep_final_discrepancy_loss",
        "kstep_initial_inner_losses_json",
        "kstep_final_inner_losses_json",
        "kstep_num_accepted_edits",
        "kstep_edits_json",
        "kstep_row_key",
    ]
    fields = list(input_fields)
    for field in generated_fields:
        if field not in fields:
            fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for position, row in enumerate(rows):
            key = row_key(position, row, args)
            result = completed.get(key)
            if result is None:
                continue
            output_row = {**row, **result}
            if result.get("kstep_status") == "ok":
                output_row[args.instruction_field] = result["kstep_refined_instruction"]
                if args.suffix_field:
                    output_row[args.suffix_field] = ""
            writer.writerow(output_row)


def main() -> None:
    args = parse_args()
    validate_args(args)
    input_path = Path(args.input).expanduser().resolve()
    anchor_path = Path(args.anchors).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    progress_path = (
        Path(args.progress_jsonl).expanduser().resolve()
        if args.progress_jsonl
        else output_path.with_suffix(".progress.jsonl")
    )
    if args.overwrite:
        output_path.unlink(missing_ok=True)
        progress_path.unlink(missing_ok=True)
    rows, input_fields = read_csv(input_path)
    signature = run_signature(args, input_path, anchor_path)

    completed: dict[str, dict[str, Any]] = {}
    if progress_path.is_file():
        for record in read_jsonl(progress_path):
            if record.get("run_signature") != signature:
                raise RuntimeError(
                    "Existing progress file has a different run signature; use --overwrite"
                )
            completed[str(record["kstep_row_key"])] = record

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs: dict[str, Any] = {
        "torch_dtype": resolve_dtype(args.dtype),
        "low_cpu_mem_usage": True,
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    base_model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs).to(device)
    base_model.config.use_cache = False
    if args.gradient_checkpointing:
        base_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model = get_peft_model(
        base_model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    model.eval()
    anchors, anchor_skipped = load_paired_anchors(
        anchor_path, tokenizer, args, device
    )
    generator = KStepGenerator(model, tokenizer, anchors, args, device)

    selected_positions = list(range(args.start_index, len(rows)))
    if args.max_candidates is not None:
        selected_positions = selected_positions[: args.max_candidates]
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    with progress_path.open("a", encoding="utf-8", buffering=1) as progress_handle:
        for position in tqdm(selected_positions, desc="K-step token generation"):
            row = rows[position]
            key = row_key(position, row, args)
            if key in completed:
                continue
            instruction = normalize(
                f"{row.get(args.instruction_field, '')}{row.get(args.suffix_field, '')}"
            )
            response = str(row.get(args.response_field, "")).strip()
            result: dict[str, Any] = {
                "run_signature": signature,
                "kstep_row_key": key,
                "kstep_status": "ok",
                "kstep_error": "",
            }
            try:
                if not valid_prompt_text(instruction):
                    raise ValueError("Invalid or empty initial standalone prompt")
                if not response:
                    raise ValueError("Empty SFT response")
                result.update(generator.refine(instruction, response))
            except (RuntimeError, ValueError) as error:
                if not args.continue_on_error:
                    raise
                result.update(
                    {
                        "kstep_status": "error",
                        "kstep_error": f"{type(error).__name__}: {error}",
                        "kstep_initial_instruction": instruction,
                        "kstep_refined_instruction": instruction,
                    }
                )
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            progress_handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            completed[key] = result

    write_output(output_path, rows, input_fields, completed, args)
    metadata = {
        "run_signature": signature,
        "model": args.model,
        "input": str(input_path),
        "anchors": str(anchor_path),
        "output": str(output_path),
        "progress_jsonl": str(progress_path),
        "selected_rows": len(selected_positions),
        "completed_rows": sum(
            completed.get(row_key(position, rows[position], args), {}).get("kstep_status")
            == "ok"
            for position in selected_positions
        ),
        "error_rows": sum(
            completed.get(row_key(position, rows[position], args), {}).get("kstep_status")
            == "error"
            for position in selected_positions
        ),
        "meta_anchor_ids": [anchor.anchor_id for anchor in anchors],
        "anchor_skipped": anchor_skipped,
        "base_safety_margin": {
            "harmful_loss": generator.base_harmful_loss,
            "refusal_loss": generator.base_refusal_loss,
            "safe_margin": generator.base_safe_margin,
        },
        "config": vars(args),
    }
    metadata_path = output_path.with_name(output_path.stem + "_metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
