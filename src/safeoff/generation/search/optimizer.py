# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# Derived from facebookresearch/advprompter; modified for LLM_FT.
# Licensed under CC BY-NC 4.0; see LICENSE in this directory and
# the repository THIRD_PARTY_NOTICES.md.
#
import re

import torch
from tqdm import tqdm

from safeoff.generation.search.sequence import MergedSeq, Seq, EmptySeq
from safeoff.generation.search.utils import ReturnStruct, apply_repetition_penalty


def get_rewrite_or_empty(rewrite, prompter):
    if rewrite is None:
        return EmptySeq(tokenizer=prompter.tokenizer, device=prompter.device)
    return rewrite


def get_prompt_prefix(instruct, rewrite):
    if rewrite is not None and not rewrite.is_empty:
        return MergedSeq(seqs=[instruct, rewrite])
    return instruct


def repeat_seq(seq, times):
    if seq is None or seq.is_empty:
        return seq
    return seq.repeat_interleave(times, dim=0)


def _target_sequences(target):
    if isinstance(target, (list, tuple)):
        if not target:
            raise ValueError("A joint target collection cannot be empty.")
        return list(target)
    return [target]


def _repeat_target(target, times):
    repeated = [item.repeat_interleave(times, dim=0) for item in _target_sequences(target)]
    return repeated if isinstance(target, (list, tuple)) else repeated[0]


def _select_target_rows(target, indices):
    selected = [item[indices] for item in _target_sequences(target)]
    return selected if isinstance(target, (list, tuple)) else selected[0]


def _target_batch_size(target):
    sizes = [item.bs for item in _target_sequences(target)]
    if len(set(sizes)) != 1:
        raise RuntimeError(f"Joint targets have mismatched batch sizes: {sizes}")
    return sizes[0]


def aggregate_target_losses(losses, aggregation="mean", temperature=1.0):
    """Aggregate a ``(num_targets, batch)`` loss tensor.

    ``softmin`` is normalized by the number of targets so that identical target
    losses have the same scale as the legacy mean objective.  This additive
    normalization also makes logged losses comparable across target counts.
    """
    if losses.ndim < 1 or losses.shape[0] == 0:
        raise ValueError("Target losses must contain at least one target.")
    if aggregation == "mean":
        return losses.mean(dim=0)
    if aggregation != "softmin":
        raise ValueError(f"Unknown target-loss aggregation: {aggregation!r}")
    if temperature <= 0:
        raise ValueError("Soft-min temperature must be positive.")
    num_targets = losses.shape[0]
    return -temperature * (
        torch.logsumexp(-losses / temperature, dim=0)
        - torch.log(
            torch.tensor(
                float(num_targets),
                dtype=losses.dtype,
                device=losses.device,
            )
        )
    )


def _target_loss_options(cfg):
    q_params = cfg.train.q_params
    return (
        getattr(q_params, "target_loss_aggregation", "mean"),
        float(getattr(q_params, "target_loss_temperature", 1.0)),
    )


def _compute_mean_target_loss(
    target_llm,
    full_instruct,
    target,
    loss_params,
    aggregation="mean",
    temperature=1.0,
):
    losses = []
    for target_seq in _target_sequences(target):
        result = target_llm.compute_pred_loss_teacher_forced(
            key="target",
            full_instruct=full_instruct,
            target=target_seq,
            loss_params=loss_params,
        )
        losses.append(result.loss_batch)
    return aggregate_target_losses(
        torch.stack(losses, dim=0),
        aggregation=aggregation,
        temperature=temperature,
    )


def build_full_instruct(target_llm, prompt_prefix, seq=None):
    seqs = []
    if prompt_prefix is not None and not prompt_prefix.is_empty:
        seqs.append(prompt_prefix)
    if seq is not None and not seq.is_empty:
        seqs.append(seq)
    if not seqs:
        raise ValueError("At least one non-empty sequence is required.")
    full_instruct = MergedSeq(seqs=seqs).to_seq(merge_dtype="ids")
    return Seq(
        text=full_instruct.text,
        tokenizer=target_llm.tokenizer,
        device=target_llm.device,
    )


def get_beam_sizes(cfg, idx, total_steps):
    if total_steps == 1:
        return 1, 1
    if idx == 0:
        return 1, cfg.train.q_params.num_beams
    if idx == total_steps - 1:
        return cfg.train.q_params.num_beams, 1
    return cfg.train.q_params.num_beams, cfg.train.q_params.num_beams


SENTENCE_END_ABBREVIATIONS = {
    "mr",
    "mrs",
    "ms",
    "dr",
    "prof",
    "sr",
    "jr",
    "vs",
    "etc",
    "fig",
    "no",
}
SENTENCE_END_CLOSING_CHARS = "\"')]}"
INVALID_STANDALONE_PENALTY = 1e6
INVALID_STANDALONE_LABEL_RE = re.compile(
    r"(?:^|\n)\s*(original|original prompt|prompt|rewrite|rewritten|replacement|revised)\s*:",
    re.IGNORECASE,
)
INVALID_STANDALONE_PHRASES = (
    "original prompt",
    "original user prompt",
    "user prompt",
    "replacement prompt",
    "rewrite the original",
    "rewritten prompt",
    "rephrase the original",
    "the original prompt",
    "the original user prompt",
    "the prompt above",
    "this prompt",
    "exact match of the original",
    "exact match to the original",
    "form requested by the original",
    "requested by the original user prompt",
    "standalone replacement prompt",
)


def extract_first_sentence(text):
    for idx, char in enumerate(text):
        if char not in ".!?":
            continue

        if char == ".":
            prev_char = text[idx - 1] if idx > 0 else ""
            next_char = text[idx + 1] if idx + 1 < len(text) else ""
            if prev_char.isdigit() and next_char.isdigit():
                continue

            token_match = re.search(r"([A-Za-z]+)$", text[:idx])
            if token_match is not None:
                token = token_match.group(1).lower()
                if len(token) == 1 or token in SENTENCE_END_ABBREVIATIONS:
                    continue

        end_idx = idx + 1
        while end_idx < len(text) and text[end_idx] in SENTENCE_END_CLOSING_CHARS:
            end_idx += 1

        if end_idx < len(text) and not text[end_idx].isspace():
            continue

        return text[:end_idx].rstrip()
    return None


def detect_invalid_standalone_rewrite(text):
    text = text or ""
    if "<|" in text:
        return "special_chat_token"
    if INVALID_STANDALONE_LABEL_RE.search(text):
        return "meta_label"

    lowered_text = text.lower()
    for phrase in INVALID_STANDALONE_PHRASES:
        if phrase in lowered_text:
            return phrase
    return None


def is_valid_standalone_rewrite(text):
    return detect_invalid_standalone_rewrite(text) is None


def _select_active_tensor_rows(values, sample_indices, beams_per_sample):
    reshaped = values.reshape(-1, beams_per_sample)
    return reshaped[sample_indices].reshape(-1).detach()


def _select_active_beam_rows(seq_beams, sample_indices, beams_per_sample):
    if seq_beams.is_empty:
        return seq_beams
    beam_row_indices = []
    for sample_idx in sample_indices:
        for beam_offset in range(beams_per_sample):
            beam_row_indices.append(sample_idx * beams_per_sample + beam_offset)
    return seq_beams[beam_row_indices]


def _rank_sample_beams(sample_beam_texts, sample_beam_scores):
    beam_order = torch.argsort(sample_beam_scores, descending=True).tolist()
    ranked = []
    for beam_offset in beam_order:
        text = sample_beam_texts[beam_offset]
        ranked.append(
            ReturnStruct(
                beam_offset=beam_offset,
                text=text,
                score=float(sample_beam_scores[beam_offset]),
                first_sentence=extract_first_sentence(text),
                standalone_valid=is_valid_standalone_rewrite(text),
            )
        )
    return ranked


def _select_v3_final_beam(sample_beam_texts, sample_beam_scores, is_final_step):
    ranked_beams = _rank_sample_beams(sample_beam_texts, sample_beam_scores)
    best_valid_complete = None
    best_valid_incomplete = None
    for beam in ranked_beams:
        if beam.standalone_valid and beam.first_sentence is not None:
            best_valid_complete = beam
            break
        if best_valid_incomplete is None and beam.standalone_valid:
            best_valid_incomplete = beam

    if best_valid_complete is not None:
        return ReturnStruct(
            finalize=True,
            text=best_valid_complete.first_sentence,
            stop_reason="sentence_stop",
            sentence_complete=True,
            standalone_valid=True,
        )

    if not is_final_step:
        return ReturnStruct(finalize=False)

    if best_valid_incomplete is not None:
        return ReturnStruct(
            finalize=True,
            text=best_valid_incomplete.text,
            stop_reason="max_tokens",
            sentence_complete=False,
            standalone_valid=True,
        )

    raw_top_beam = ranked_beams[0]
    return ReturnStruct(
        finalize=True,
        text=raw_top_beam.text,
        stop_reason="max_tokens",
        sentence_complete=False,
        standalone_valid=raw_top_beam.standalone_valid,
    )


def _build_candidate_seq(seq_beams, q_sample_seq, num_samples_per_beam):
    if not seq_beams.is_empty:
        seq_beams_rep = seq_beams.repeat_interleave(num_samples_per_beam, dim=0)
    else:
        seq_beams_rep = seq_beams

    if seq_beams_rep is not None and not seq_beams_rep.is_empty:
        candidate_seq = seq_beams_rep.clone()
        candidate_seq.append(q_sample_seq)
    else:
        candidate_seq = q_sample_seq
    return candidate_seq, seq_beams_rep


def _score_reward_batch(reward_scorer, questions, answers, device):
    if hasattr(reward_scorer, "score"):
        raw_scores = reward_scorer.score(questions, answers)
    else:
        raw_scores = reward_scorer(questions, answers)

    if torch.is_tensor(raw_scores):
        scores = raw_scores.detach().to(device=device, dtype=torch.float32)
    else:
        scores = torch.tensor(list(raw_scores), dtype=torch.float32, device=device)
    return scores.reshape(-1)


@torch.no_grad()
def advPrompterOpt(cfg, instruct, target, prompter, target_llm, rewrite=None):
    if cfg.verbose:
        tqdm.write("\n Running AdvPrompterOpt: Generating optimized suffix...")
    rewrite = get_rewrite_or_empty(rewrite=rewrite, prompter=prompter)
    prompt_prefix = get_prompt_prefix(instruct=instruct, rewrite=rewrite)

    # Compute the initial prediction losses without a suffix
    full_instruct = build_full_instruct(
        target_llm=target_llm,
        prompt_prefix=prompt_prefix,
    )
    target_llm_tf = target_llm.compute_pred_loss_teacher_forced(
        key="target",
        full_instruct=full_instruct,
        target=target,
        loss_params=dict(hard_labels=True, reweight_loss=cfg.reweight_loss),
    )
    losses = target_llm_tf.loss_batch.detach().to(prompter.device)  # (ebs, )

    # Initialize the beam scores
    beam_scores = torch.zeros_like(losses)  # (ebs, )
    suffix_beams = EmptySeq(tokenizer=prompter.tokenizer, device=prompter.device)

    for idx in range(cfg.train.q_params.max_new_tokens):
        num_beams_in, num_beams_out = get_beam_sizes(
            cfg=cfg,
            idx=idx,
            total_steps=cfg.train.q_params.max_new_tokens,
        )

        # expand the dimension of instruct and targets to match suffix beams
        instruct_rep = instruct.repeat_interleave(num_beams_in, dim=0)
        rewrite_rep = repeat_seq(rewrite, num_beams_in)
        prompt_prefix_rep = prompt_prefix.repeat_interleave(num_beams_in, dim=0)
        target_rep = target.repeat_interleave(num_beams_in, dim=0)

        next_dist_seq_prompter, next_dist_seq_basemodel = get_next_token_probabilities(
            cfg=cfg,
            key="suffix",
            instruct=instruct_rep,
            rewrite=rewrite_rep,
            suffix=suffix_beams,
            search_beams=suffix_beams,
            prompter=prompter,
        )

        next_token_candidate_ids, candidate_beam_scores, candidate_losses = (
            select_and_evaluate_next_token_candidates(
                cfg=cfg,
                prompt_prefix=prompt_prefix_rep,
                target=target_rep,
                seq_beams=suffix_beams,
                target_llm=target_llm,
                next_dist_seq_prompter=next_dist_seq_prompter,
                next_dist_seq_basemodel=next_dist_seq_basemodel,
                beam_scores=beam_scores,
                prev_losses=losses,
                num_beams_in=num_beams_in,
            )
        )

        suffix_beams, losses, beam_scores, _ = select_next_beams(
            cfg=cfg,
            seq_beams=suffix_beams,
            next_token_candidate_ids=next_token_candidate_ids,
            candidate_beam_scores=candidate_beam_scores,
            candidate_losses=candidate_losses,
            num_beams_in=num_beams_in,
            num_beams_out=num_beams_out,
        )
        if cfg.verbose:
            tqdm.write(f" Beams[0] (iter {idx}): {suffix_beams[:num_beams_out].text}")

    if cfg.verbose:
        tqdm.write(
            f" AdvPrompterOpt completed. Generated suffix[0]: {suffix_beams[0].text}"
        )
    return suffix_beams


@torch.no_grad()
def advPrompterOptJoint(cfg, instruct, target, prompter, target_llm):
    rewrite_tokens = getattr(cfg.train.q_params, "rewrite_max_new_tokens", 0)
    suffix_tokens = cfg.train.q_params.max_new_tokens
    total_steps = rewrite_tokens + suffix_tokens
    if total_steps <= 0:
        empty = EmptySeq(tokenizer=prompter.tokenizer, device=prompter.device)
        return empty, empty

    if cfg.verbose:
        tqdm.write(
            "\n Running AdvPrompterOptJoint: Jointly optimizing rewrite and suffix..."
        )

    full_instruct = build_full_instruct(
        target_llm=target_llm,
        prompt_prefix=instruct,
    )
    target_llm_tf = target_llm.compute_pred_loss_teacher_forced(
        key="target",
        full_instruct=full_instruct,
        target=target,
        loss_params=dict(hard_labels=True, reweight_loss=cfg.reweight_loss),
    )
    losses = initial_loss_batch.detach().to(prompter.device)
    beam_scores = torch.zeros_like(losses)

    rewrite_beams = EmptySeq(tokenizer=prompter.tokenizer, device=prompter.device)
    suffix_beams = EmptySeq(tokenizer=prompter.tokenizer, device=prompter.device)

    for idx in range(total_steps):
        num_beams_in, num_beams_out = get_beam_sizes(
            cfg=cfg,
            idx=idx,
            total_steps=total_steps,
        )
        instruct_rep = instruct.repeat_interleave(num_beams_in, dim=0)
        target_rep = target.repeat_interleave(num_beams_in, dim=0)

        if idx < rewrite_tokens:
            key = "rewrite"
            search_beams = rewrite_beams
            rewrite_rep = rewrite_beams
            suffix_rep = suffix_beams
            prompt_prefix_rep = instruct_rep
        else:
            key = "suffix"
            search_beams = suffix_beams
            rewrite_rep = rewrite_beams
            suffix_rep = suffix_beams
            prompt_prefix_rep = get_prompt_prefix(
                instruct=instruct_rep,
                rewrite=rewrite_rep,
            )

        next_dist_seq_prompter, next_dist_seq_basemodel = get_next_token_probabilities(
            cfg=cfg,
            key=key,
            instruct=instruct_rep,
            rewrite=rewrite_rep,
            suffix=suffix_rep,
            search_beams=search_beams,
            prompter=prompter,
        )

        next_token_candidate_ids, candidate_beam_scores, candidate_losses = (
            select_and_evaluate_next_token_candidates(
                cfg=cfg,
                prompt_prefix=prompt_prefix_rep,
                target=target_rep,
                seq_beams=search_beams,
                target_llm=target_llm,
                next_dist_seq_prompter=next_dist_seq_prompter,
                next_dist_seq_basemodel=next_dist_seq_basemodel,
                beam_scores=beam_scores,
                prev_losses=losses,
                num_beams_in=num_beams_in,
            )
        )

        next_beams, losses, beam_scores, parent_beam_indices = select_next_beams(
            cfg=cfg,
            seq_beams=search_beams,
            next_token_candidate_ids=next_token_candidate_ids,
            candidate_beam_scores=candidate_beam_scores,
            candidate_losses=candidate_losses,
            num_beams_in=num_beams_in,
            num_beams_out=num_beams_out,
        )
        if idx < rewrite_tokens:
            rewrite_beams = next_beams
        else:
            suffix_beams = next_beams
            if not rewrite_beams.is_empty:
                rewrite_beams = rewrite_beams[parent_beam_indices]

        if cfg.verbose:
            rewrite_text = [] if rewrite_beams.is_empty else rewrite_beams[:num_beams_out].text
            suffix_text = [] if suffix_beams.is_empty else suffix_beams[:num_beams_out].text
            tqdm.write(
                f" Joint beams[0] (iter {idx}, key={key}) | rewrite={rewrite_text} | suffix={suffix_text}"
            )

    return rewrite_beams, suffix_beams


@torch.no_grad()
def advPrompterOptPromptReplacement(
    cfg,
    instruct,
    target,
    prompter,
    target_llm,
    max_new_tokens=None,
):
    # Legacy behavior: reuse the rewrite channel to synthesize a standalone prompt.
    if max_new_tokens is None:
        max_new_tokens = getattr(cfg.train.q_params, "rewrite_max_new_tokens", 0)
    if max_new_tokens <= 0:
        return EmptySeq(tokenizer=prompter.tokenizer, device=prompter.device)

    if cfg.verbose:
        tqdm.write(
            "\n Running AdvPrompterOptPromptReplacement: Optimizing a replacement prompt..."
        )

    empty_full_instruct = EmptySeq(
        tokenizer=target_llm.tokenizer,
        device=target_llm.device,
    )
    aggregation, temperature = _target_loss_options(cfg)
    initial_loss_batch = _compute_mean_target_loss(
        target_llm=target_llm,
        full_instruct=empty_full_instruct,
        target=target,
        loss_params=dict(hard_labels=True, reweight_loss=cfg.reweight_loss),
        aggregation=aggregation,
        temperature=temperature,
    )
    losses = initial_loss_batch.detach().to(prompter.device)
    beam_scores = torch.zeros_like(losses)

    replacement_beams = EmptySeq(tokenizer=prompter.tokenizer, device=prompter.device)
    empty_suffix = EmptySeq(tokenizer=prompter.tokenizer, device=prompter.device)

    for idx in range(max_new_tokens):
        num_beams_in, num_beams_out = get_beam_sizes(
            cfg=cfg,
            idx=idx,
            total_steps=max_new_tokens,
        )
        instruct_rep = instruct.repeat_interleave(num_beams_in, dim=0)
        target_rep = target.repeat_interleave(num_beams_in, dim=0)

        next_dist_seq_prompter, next_dist_seq_basemodel = get_next_token_probabilities(
            cfg=cfg,
            key="rewrite",
            instruct=instruct_rep,
            rewrite=replacement_beams,
            suffix=empty_suffix,
            search_beams=replacement_beams,
            prompter=prompter,
        )

        next_token_candidate_ids, candidate_beam_scores, candidate_losses = (
            select_and_evaluate_next_token_candidates(
                cfg=cfg,
                prompt_prefix=None,
                target=target_rep,
                seq_beams=replacement_beams,
                target_llm=target_llm,
                next_dist_seq_prompter=next_dist_seq_prompter,
                next_dist_seq_basemodel=next_dist_seq_basemodel,
                beam_scores=beam_scores,
                prev_losses=losses,
                num_beams_in=num_beams_in,
            )
        )

        replacement_beams, losses, beam_scores, _ = select_next_beams(
            cfg=cfg,
            seq_beams=replacement_beams,
            next_token_candidate_ids=next_token_candidate_ids,
            candidate_beam_scores=candidate_beam_scores,
            candidate_losses=candidate_losses,
            num_beams_in=num_beams_in,
            num_beams_out=num_beams_out,
        )
        if cfg.verbose:
            tqdm.write(
                f" Replacement beams[0] (iter {idx}): {replacement_beams[:num_beams_out].text}"
            )

    if cfg.verbose:
        tqdm.write(
            " AdvPrompterOptPromptReplacement completed. "
            f"Generated prompt[0]: {replacement_beams[0].text}"
        )
    return replacement_beams


@torch.no_grad()
def advPrompterOptStandaloneReplacement(
    cfg,
    instruct,
    target,
    prompter,
    target_llm,
    max_new_tokens=None,
):
    if max_new_tokens is None:
        max_new_tokens = getattr(cfg.train.q_params, "rewrite_max_new_tokens", 0)
    if max_new_tokens <= 0:
        return EmptySeq(tokenizer=prompter.tokenizer, device=prompter.device)

    if cfg.verbose:
        tqdm.write(
            "\n Running AdvPrompterOptStandaloneReplacement: "
            "Optimizing a standalone replacement prompt..."
        )

    empty_full_instruct = EmptySeq(
        tokenizer=target_llm.tokenizer,
        device=target_llm.device,
    )
    target_llm_tf = target_llm.compute_pred_loss_teacher_forced(
        key="target",
        full_instruct=empty_full_instruct,
        target=target,
        loss_params=dict(hard_labels=True, reweight_loss=cfg.reweight_loss),
    )
    losses = target_llm_tf.loss_batch.detach().to(prompter.device)
    beam_scores = torch.zeros_like(losses)

    replacement_beams = EmptySeq(tokenizer=prompter.tokenizer, device=prompter.device)

    for idx in range(max_new_tokens):
        num_beams_in, num_beams_out = get_beam_sizes(
            cfg=cfg,
            idx=idx,
            total_steps=max_new_tokens,
        )
        instruct_rep = instruct.repeat_interleave(num_beams_in, dim=0)
        target_rep = target.repeat_interleave(num_beams_in, dim=0)

        next_dist_seq_prompter, next_dist_seq_basemodel = (
            get_next_token_probabilities_with_context(
                cfg=cfg,
                key="replacement",
                search_beams=replacement_beams,
                prompter=prompter,
                instruct=instruct_rep,
                replacement=replacement_beams,
            )
        )

        next_token_candidate_ids, candidate_beam_scores, candidate_losses = (
            select_and_evaluate_next_token_candidates(
                cfg=cfg,
                prompt_prefix=None,
                target=target_rep,
                seq_beams=replacement_beams,
                target_llm=target_llm,
                next_dist_seq_prompter=next_dist_seq_prompter,
                next_dist_seq_basemodel=next_dist_seq_basemodel,
                beam_scores=beam_scores,
                prev_losses=losses,
                num_beams_in=num_beams_in,
            )
        )

        replacement_beams, losses, beam_scores, _ = select_next_beams(
            cfg=cfg,
            seq_beams=replacement_beams,
            next_token_candidate_ids=next_token_candidate_ids,
            candidate_beam_scores=candidate_beam_scores,
            candidate_losses=candidate_losses,
            num_beams_in=num_beams_in,
            num_beams_out=num_beams_out,
        )
        if cfg.verbose:
            tqdm.write(
                " Standalone replacement beams[0] "
                f"(iter {idx}): {replacement_beams[:num_beams_out].text}"
            )

    if cfg.verbose:
        tqdm.write(
            " AdvPrompterOptStandaloneReplacement completed. "
            f"Generated prompt[0]: {replacement_beams[0].text}"
        )
    return replacement_beams


@torch.no_grad()
def advPrompterOptStandaloneReplacementV3(
    cfg,
    instruct,
    target,
    prompter,
    target_llm,
    max_new_tokens=None,
    trace_callback=None,
):
    if max_new_tokens is None:
        max_new_tokens = getattr(cfg.train.q_params, "rewrite_max_new_tokens", 0)

    bs = instruct.bs
    if max_new_tokens <= 0:
        return ReturnStruct(
            text=[""] * bs,
            stop_reasons=["max_tokens"] * bs,
            generation_steps=[0] * bs,
            sentence_complete=[False] * bs,
            standalone_valid=[False] * bs,
        )

    if cfg.verbose:
        tqdm.write(
            "\n Running AdvPrompterOptStandaloneReplacementV3: "
            "Optimizing a single-sentence standalone replacement prompt..."
        )

    empty_full_instruct = EmptySeq(
        tokenizer=target_llm.tokenizer,
        device=target_llm.device,
    )
    aggregation, temperature = _target_loss_options(cfg)
    initial_loss_batch = _compute_mean_target_loss(
        target_llm=target_llm,
        full_instruct=empty_full_instruct,
        target=target,
        loss_params=dict(hard_labels=True, reweight_loss=cfg.reweight_loss),
        aggregation=aggregation,
        temperature=temperature,
    )

    active_sample_indices = list(range(bs))
    active_instruct = instruct
    active_target = target
    active_losses = initial_loss_batch.detach().to(prompter.device)
    active_beam_scores = torch.zeros_like(active_losses)
    active_replacement_beams = EmptySeq(
        tokenizer=prompter.tokenizer,
        device=prompter.device,
    )

    final_texts = [None] * bs
    final_stop_reasons = [None] * bs
    final_generation_steps = [0] * bs
    final_sentence_complete = [False] * bs
    final_standalone_valid = [False] * bs

    for idx in range(max_new_tokens):
        if not active_sample_indices:
            break

        num_beams_in, num_beams_out = get_beam_sizes(
            cfg=cfg,
            idx=idx,
            total_steps=max_new_tokens,
        )
        if idx == max_new_tokens - 1 and max_new_tokens > 1:
            num_beams_out = cfg.train.q_params.num_beams
        active_instruct_rep = active_instruct.repeat_interleave(num_beams_in, dim=0)
        active_target_rep = _repeat_target(active_target, num_beams_in)

        next_dist_seq_prompter, next_dist_seq_basemodel = (
            get_next_token_probabilities_with_context(
                cfg=cfg,
                key="replacement",
                search_beams=active_replacement_beams,
                prompter=prompter,
                instruct=active_instruct_rep,
                replacement=active_replacement_beams,
            )
        )

        next_token_candidate_ids, candidate_beam_scores, candidate_losses = (
            select_and_evaluate_next_token_candidates(
                cfg=cfg,
                prompt_prefix=None,
                target=active_target_rep,
                seq_beams=active_replacement_beams,
                target_llm=target_llm,
                next_dist_seq_prompter=next_dist_seq_prompter,
                next_dist_seq_basemodel=next_dist_seq_basemodel,
                beam_scores=active_beam_scores,
                prev_losses=active_losses,
                num_beams_in=num_beams_in,
            )
        )

        active_replacement_beams, active_losses, active_beam_scores, _ = (
            select_next_beams(
                cfg=cfg,
                seq_beams=active_replacement_beams,
                next_token_candidate_ids=next_token_candidate_ids,
                candidate_beam_scores=candidate_beam_scores,
                candidate_losses=candidate_losses,
                num_beams_in=num_beams_in,
                num_beams_out=num_beams_out,
            )
        )

        unfinished_local_indices = []
        for local_idx, global_idx in enumerate(active_sample_indices):
            beam_start = local_idx * num_beams_out
            beam_end = beam_start + num_beams_out
            sample_beam_indices = list(range(beam_start, beam_end))
            sample_beam_texts = active_replacement_beams[sample_beam_indices].text
            sample_beam_scores = active_beam_scores[beam_start:beam_end]
            if trace_callback is not None:
                # Observation only: preserve the original search/termination behavior.
                for beam_offset, beam_text in enumerate(sample_beam_texts):
                    trace_callback({
                        "sample_index": global_idx,
                        "step": idx + 1,
                        "beam": beam_offset,
                        "text": beam_text,
                        "search_loss": float(active_losses[beam_start + beam_offset]),
                        "beam_score": float(sample_beam_scores[beam_offset]),
                        "first_sentence": extract_first_sentence(beam_text),
                        "standalone_valid": is_valid_standalone_rewrite(beam_text),
                    })
            selected_output = _select_v3_final_beam(
                sample_beam_texts=sample_beam_texts,
                sample_beam_scores=sample_beam_scores,
                is_final_step=(idx == max_new_tokens - 1),
            )
            if selected_output.finalize:
                final_texts[global_idx] = selected_output.text
                final_stop_reasons[global_idx] = selected_output.stop_reason
                final_generation_steps[global_idx] = idx + 1
                final_sentence_complete[global_idx] = (
                    selected_output.sentence_complete
                )
                final_standalone_valid[global_idx] = (
                    selected_output.standalone_valid
                )
            else:
                unfinished_local_indices.append(local_idx)

        if cfg.verbose:
            top_beam_indices = [
                sample_idx * num_beams_out
                for sample_idx in range(len(active_sample_indices))
            ]
            top_beam_texts = active_replacement_beams[top_beam_indices].text
            tqdm.write(
                " Standalone v3 top beams "
                f"(iter {idx}): {top_beam_texts}"
            )

        if not unfinished_local_indices:
            active_sample_indices = []
            break

        active_sample_indices = [
            active_sample_indices[local_idx]
            for local_idx in unfinished_local_indices
        ]
        active_instruct = active_instruct[unfinished_local_indices]
        active_target = _select_target_rows(active_target, unfinished_local_indices)
        active_replacement_beams = _select_active_beam_rows(
            active_replacement_beams,
            unfinished_local_indices,
            num_beams_out,
        )
        active_losses = _select_active_tensor_rows(
            active_losses,
            unfinished_local_indices,
            num_beams_out,
        )
        active_beam_scores = _select_active_tensor_rows(
            active_beam_scores,
            unfinished_local_indices,
            num_beams_out,
        )

    if any(text is None for text in final_texts):
        final_texts = [text if text is not None else "" for text in final_texts]
        final_stop_reasons = [
            reason if reason is not None else "max_tokens"
            for reason in final_stop_reasons
        ]
        final_generation_steps = [
            step if step > 0 else max_new_tokens for step in final_generation_steps
        ]
        final_standalone_valid = [
            is_valid if text is not None else False
            for text, is_valid in zip(final_texts, final_standalone_valid)
        ]

    if cfg.verbose:
        tqdm.write(
            " AdvPrompterOptStandaloneReplacementV3 completed. "
            f"Generated prompt[0]: {final_texts[0]}"
        )

    return ReturnStruct(
        text=final_texts,
        stop_reasons=final_stop_reasons,
        generation_steps=final_generation_steps,
        sentence_complete=final_sentence_complete,
        standalone_valid=final_standalone_valid,
    )


@torch.no_grad()
def advPrompterOptStandaloneReplacementV4(
    cfg,
    instruct,
    prompter,
    target_llm,
    reward_scorer,
    max_new_tokens=None,
):
    if max_new_tokens is None:
        max_new_tokens = getattr(cfg.train.q_params, "rewrite_max_new_tokens", 0)

    bs = instruct.bs
    if max_new_tokens <= 0:
        return ReturnStruct(
            text=[""] * bs,
            stop_reasons=["max_tokens"] * bs,
            generation_steps=[0] * bs,
            sentence_complete=[False] * bs,
            standalone_valid=[False] * bs,
        )

    if cfg.verbose:
        tqdm.write(
            "\n Running AdvPrompterOptStandaloneReplacementV4: "
            "Optimizing a reward-constrained standalone replacement prompt..."
        )

    active_sample_indices = list(range(bs))
    active_instruct = instruct
    active_rewards = torch.zeros(bs, dtype=torch.float32, device=prompter.device)
    active_beam_scores = torch.zeros_like(active_rewards)
    active_replacement_beams = EmptySeq(
        tokenizer=prompter.tokenizer,
        device=prompter.device,
    )

    final_texts = [None] * bs
    final_stop_reasons = [None] * bs
    final_generation_steps = [0] * bs
    final_sentence_complete = [False] * bs
    final_standalone_valid = [False] * bs

    for idx in range(max_new_tokens):
        if not active_sample_indices:
            break

        num_beams_in, num_beams_out = get_beam_sizes(
            cfg=cfg,
            idx=idx,
            total_steps=max_new_tokens,
        )
        if idx == max_new_tokens - 1 and max_new_tokens > 1:
            num_beams_out = cfg.train.q_params.num_beams
        active_instruct_rep = active_instruct.repeat_interleave(num_beams_in, dim=0)

        next_dist_seq_prompter, next_dist_seq_basemodel = (
            get_next_token_probabilities_with_context(
                cfg=cfg,
                key="replacement",
                search_beams=active_replacement_beams,
                prompter=prompter,
                instruct=active_instruct_rep,
                replacement=active_replacement_beams,
            )
        )

        next_token_candidate_ids, candidate_beam_scores, candidate_rewards = (
            select_and_evaluate_next_token_candidates_with_reward(
                cfg=cfg,
                instruct=active_instruct_rep,
                seq_beams=active_replacement_beams,
                target_llm=target_llm,
                reward_scorer=reward_scorer,
                next_dist_seq_prompter=next_dist_seq_prompter,
                next_dist_seq_basemodel=next_dist_seq_basemodel,
                beam_scores=active_beam_scores,
                prev_rewards=active_rewards,
                num_beams_in=num_beams_in,
            )
        )

        active_replacement_beams, active_rewards, active_beam_scores, _ = (
            select_next_beams(
                cfg=cfg,
                seq_beams=active_replacement_beams,
                next_token_candidate_ids=next_token_candidate_ids,
                candidate_beam_scores=candidate_beam_scores,
                candidate_losses=candidate_rewards,
                num_beams_in=num_beams_in,
                num_beams_out=num_beams_out,
            )
        )

        unfinished_local_indices = []
        for local_idx, global_idx in enumerate(active_sample_indices):
            beam_start = local_idx * num_beams_out
            beam_end = beam_start + num_beams_out
            sample_beam_indices = list(range(beam_start, beam_end))
            sample_beam_texts = active_replacement_beams[sample_beam_indices].text
            sample_beam_scores = active_beam_scores[beam_start:beam_end]
            selected_output = _select_v3_final_beam(
                sample_beam_texts=sample_beam_texts,
                sample_beam_scores=sample_beam_scores,
                is_final_step=(idx == max_new_tokens - 1),
            )
            if selected_output.finalize:
                final_texts[global_idx] = selected_output.text
                final_stop_reasons[global_idx] = selected_output.stop_reason
                final_generation_steps[global_idx] = idx + 1
                final_sentence_complete[global_idx] = (
                    selected_output.sentence_complete
                )
                final_standalone_valid[global_idx] = (
                    selected_output.standalone_valid
                )
            else:
                unfinished_local_indices.append(local_idx)

        if cfg.verbose:
            top_beam_indices = [
                sample_idx * num_beams_out
                for sample_idx in range(len(active_sample_indices))
            ]
            top_beam_texts = active_replacement_beams[top_beam_indices].text
            tqdm.write(
                " Standalone v4 top beams "
                f"(iter {idx}): {top_beam_texts}"
            )

        if not unfinished_local_indices:
            active_sample_indices = []
            break

        active_sample_indices = [
            active_sample_indices[local_idx]
            for local_idx in unfinished_local_indices
        ]
        active_instruct = active_instruct[unfinished_local_indices]
        active_replacement_beams = _select_active_beam_rows(
            active_replacement_beams,
            unfinished_local_indices,
            num_beams_out,
        )
        active_rewards = _select_active_tensor_rows(
            active_rewards,
            unfinished_local_indices,
            num_beams_out,
        )
        active_beam_scores = _select_active_tensor_rows(
            active_beam_scores,
            unfinished_local_indices,
            num_beams_out,
        )

    if any(text is None for text in final_texts):
        final_texts = [text if text is not None else "" for text in final_texts]
        final_stop_reasons = [
            reason if reason is not None else "max_tokens"
            for reason in final_stop_reasons
        ]
        final_generation_steps = [
            step if step > 0 else max_new_tokens for step in final_generation_steps
        ]
        final_standalone_valid = [
            is_valid if text is not None else False
            for text, is_valid in zip(final_texts, final_standalone_valid)
        ]

    if cfg.verbose:
        tqdm.write(
            " AdvPrompterOptStandaloneReplacementV4 completed. "
            f"Generated prompt[0]: {final_texts[0]}"
        )

    return ReturnStruct(
        text=final_texts,
        stop_reasons=final_stop_reasons,
        generation_steps=final_generation_steps,
        sentence_complete=final_sentence_complete,
        standalone_valid=final_standalone_valid,
    )


def get_next_token_probabilities(cfg, key, instruct, rewrite, suffix, search_beams, prompter):
    return get_next_token_probabilities_with_context(
        cfg=cfg,
        key=key,
        search_beams=search_beams,
        prompter=prompter,
        instruct=instruct,
        rewrite=rewrite,
        suffix=suffix,
    )


def get_next_token_probabilities_with_context(
    cfg,
    key,
    search_beams,
    prompter,
    **context,
):
    # get the next token probabilities from the prompter and the base model
    prompter_next = prompter.get_next_token(key=key, **context)
    next_dist_seq_prompter = (
        prompter_next.response_dist.clone().detach()
    )  # (ebs, 1, vocab_size)
    prompter_next_basemodel = prompter.get_next_token(
        key=key,
        use_basemodel=True,
        **context,
    )
    next_dist_seq_basemodel = (
        prompter_next_basemodel.response_dist.clone().detach()
    )  # (bs, 1, vocab_size)

    # apply repetition penalty
    if not search_beams.is_empty and "repetition_penalty" in cfg.train.q_params:
        next_dist_logits_basemodel = apply_repetition_penalty(
            logits=next_dist_seq_basemodel.logits.squeeze(1),
            prev_ids=search_beams.ids,
            penalty=cfg.train.q_params.repetition_penalty,
        )
        next_dist_seq_basemodel = Seq(
            logits=next_dist_logits_basemodel[:, None, :],
            mask=next_dist_seq_basemodel.mask,
            tokenizer=next_dist_seq_basemodel.tokenizer,
            device=next_dist_seq_basemodel.device,
        )
        next_dist_logits_prompter = apply_repetition_penalty(
            logits=next_dist_seq_prompter.logits.squeeze(1),
            prev_ids=search_beams.ids,
            penalty=cfg.train.q_params.repetition_penalty,
        )
        next_dist_seq_prompter = Seq(
            logits=next_dist_logits_prompter[:, None, :],
            mask=next_dist_seq_prompter.mask,
            tokenizer=next_dist_seq_prompter.tokenizer,
            device=next_dist_seq_prompter.device,
        )
    return next_dist_seq_prompter, next_dist_seq_basemodel


def select_and_evaluate_next_token_candidates(
    cfg,
    prompt_prefix,
    target,
    seq_beams,
    target_llm,
    next_dist_seq_prompter,
    next_dist_seq_basemodel,
    beam_scores,
    prev_losses,
    num_beams_in,
):
    num_chunks = cfg.train.q_params.num_chunks
    assert cfg.train.q_params.top_k % (num_chunks * num_beams_in) == 0
    num_samples_per_beam = cfg.train.q_params.top_k // (num_chunks * num_beams_in)
    all_next_token_candidate_ids = None

    for i in range(cfg.train.q_params.num_chunks):
        next_token_candidate_ids = select_next_token_candidates(
            cfg=cfg,
            next_dist_seq=next_dist_seq_prompter,
            previous_next_token_candidate_ids=all_next_token_candidate_ids,
            num_samples_per_beam=num_samples_per_beam,
            always_include_best=cfg.train.q_params.candidates.always_include_best
            and i == 0,
        )  # (ebs = bs * num_beams_in, num_samples_per_beam)

        candidate_beam_scores, candidate_losses = evaluate_next_token_candidates(
            cfg=cfg,
            prompt_prefix=prompt_prefix,
            target=target,
            seq_beams=seq_beams,
            target_llm=target_llm,
            next_token_candidate_ids=next_token_candidate_ids,
            next_dist_seq_basemodel=next_dist_seq_basemodel,
            next_dist_seq_prompter=next_dist_seq_prompter,
            prev_beam_scores=beam_scores,
            prev_losses=prev_losses,
        )  # (ebs, num_samples_per_beam)

        if all_next_token_candidate_ids is None:
            all_next_token_candidate_ids = next_token_candidate_ids
            all_candidate_beam_scores = candidate_beam_scores
            all_candidate_losses = candidate_losses
        else:
            all_next_token_candidate_ids = torch.cat(
                (next_token_candidate_ids, all_next_token_candidate_ids), dim=1
            )  # (ebs, i * num_samples_per_beam)
            all_candidate_beam_scores = torch.cat(
                (candidate_beam_scores, all_candidate_beam_scores), dim=1
            )  # (ebs, i * num_samples_per_beam)
            all_candidate_losses = torch.cat(
                (candidate_losses, all_candidate_losses), dim=1
            )  # (ebs, i * num_samples_per_beam)
    return all_next_token_candidate_ids, all_candidate_beam_scores, all_candidate_losses


@torch.no_grad()
def select_next_token_candidates(
    cfg,
    next_dist_seq,
    previous_next_token_candidate_ids,
    num_samples_per_beam,
    always_include_best,
):
    # clone is important here! We modify the logits but will also use the original dist
    next_dist_logits = next_dist_seq.logits.squeeze(1).clone()  # (ebs, vocab_size)
    tokenizer_vocab_size = len(next_dist_seq.tokenizer)
    if tokenizer_vocab_size < next_dist_logits.shape[1]:
        next_dist_logits[:, tokenizer_vocab_size:] = torch.finfo(
            next_dist_logits.dtype
        ).min

    if previous_next_token_candidate_ids is not None:
        previous_next_token_candidate_ids_khot = torch.scatter(
            torch.zeros_like(next_dist_logits), 1, previous_next_token_candidate_ids, 1
        )  # (ebs, vocab_size)
        next_dist_logits -= 1e10 * previous_next_token_candidate_ids_khot

    if cfg.train.q_params.candidates.do_sample:
        if always_include_best:
            next_dist_logits -= 1e10 * next_dist_seq.onehot.squeeze(1)

        probs = torch.softmax(
            next_dist_logits / cfg.train.q_params.candidates.temperature,
            dim=-1,
        )  # (ebs, vocab_size)
        next_token_candidate_ids = probs.multinomial(
            num_samples=num_samples_per_beam, replacement=False
        )  # (ebs, num_samples_per_beam)
        if always_include_best:
            next_token_candidate_ids = torch.cat(
                [next_dist_seq.ids, next_token_candidate_ids[:, :-1]], dim=1
            )
    else:
        next_token_candidate_ids = next_dist_logits.topk(
            k=num_samples_per_beam, dim=-1
        ).indices  # (ebs, num_samples_per_beam)
    return next_token_candidate_ids


@torch.no_grad()
def evaluate_next_token_candidates(
    cfg,
    prompt_prefix,
    target,
    seq_beams,
    target_llm,
    next_token_candidate_ids,
    next_dist_seq_basemodel,
    next_dist_seq_prompter,
    prev_beam_scores,
    prev_losses,
):
    ebs, num_samples_per_beam = next_token_candidate_ids.shape

    q_next_token_candidate_ids = torch.reshape(
        next_token_candidate_ids, (ebs * num_samples_per_beam, 1)
    )
    q_sample_seq = Seq(
        ids=q_next_token_candidate_ids,
        tokenizer=next_dist_seq_prompter.tokenizer,
        device=next_dist_seq_prompter.device,
    )

    # extend to match the extended batch size
    if prompt_prefix is not None and not prompt_prefix.is_empty:
        prompt_prefix_rep = prompt_prefix.repeat_interleave(num_samples_per_beam, dim=0)
    else:
        prompt_prefix_rep = None
    target_rep = _repeat_target(target, num_samples_per_beam)
    if not seq_beams.is_empty:
        seq_beams_rep = seq_beams.repeat_interleave(num_samples_per_beam, dim=0)
    else:
        seq_beams_rep = seq_beams

    # Build explicit candidate sequences here instead of nesting another
    # MergedSeq. This keeps batch expansion predictable for standalone
    # replacement prompts where there is no prompt_prefix.
    if seq_beams_rep is not None and not seq_beams_rep.is_empty:
        candidate_seq = seq_beams_rep.clone()
        candidate_seq.append(q_sample_seq)
    else:
        candidate_seq = q_sample_seq

    # compute the losses on each sample
    full_instruct = build_full_instruct(
        target_llm=target_llm,
        prompt_prefix=prompt_prefix_rep,
        seq=candidate_seq,
    )
    target_batch_size = _target_batch_size(target_rep)
    if full_instruct.bs != target_batch_size:
        raise RuntimeError(
            "Mismatched batch sizes in evaluate_next_token_candidates: "
            f"prompt_prefix_bs={None if prompt_prefix_rep is None else prompt_prefix_rep.bs}, "
            f"seq_beams_bs={None if seq_beams.is_empty else seq_beams.bs}, "
            f"seq_beams_rep_bs={None if seq_beams_rep.is_empty else seq_beams_rep.bs}, "
            f"q_sample_bs={q_sample_seq.bs}, "
            f"candidate_seq_bs={candidate_seq.bs}, "
            f"full_instruct_bs={full_instruct.bs}, "
            f"target_rep_bs={target_batch_size}, "
            f"num_samples_per_beam={num_samples_per_beam}"
        )

    with torch.no_grad():
        aggregation, temperature = _target_loss_options(cfg)
        loss_batch = _compute_mean_target_loss(
            target_llm=target_llm,
            full_instruct=full_instruct,
            target=target_rep,
            loss_params=dict(hard_labels=True, reweight_loss=cfg.reweight_loss),
            aggregation=aggregation,
            temperature=temperature,
        )

    loss_batch = loss_batch.to(next_dist_seq_prompter.device)
    losses = torch.reshape(loss_batch, (ebs, num_samples_per_beam))
    loss_delta = losses - prev_losses[:, None]  # (ebs, num_samples_per_beam)

    next_dist_logprobs_basemodel = next_dist_seq_basemodel.logprobs.squeeze(1)
    selected_logprobs_basemodel = torch.gather(
        next_dist_logprobs_basemodel, dim=-1, index=next_token_candidate_ids
    )  # (ebs, num_samples_per_beam)

    factor = cfg.train.q_params.lambda_val
    beam_scores_delta = selected_logprobs_basemodel - loss_delta * factor
    new_beam_scores = prev_beam_scores[:, None] + beam_scores_delta
    standalone_validity_penalty = getattr(
        cfg.train.q_params,
        "standalone_validity_penalty",
        0.0,
    )
    if standalone_validity_penalty > 0:
        candidate_validity = [
            is_valid_standalone_rewrite(text) for text in candidate_seq.text
        ]
        invalid_mask = torch.tensor(
            [not is_valid for is_valid in candidate_validity],
            dtype=new_beam_scores.dtype,
            device=new_beam_scores.device,
        ).reshape(ebs, num_samples_per_beam)
        new_beam_scores = new_beam_scores - invalid_mask * standalone_validity_penalty
    return new_beam_scores, losses


def select_and_evaluate_next_token_candidates_with_reward(
    cfg,
    instruct,
    seq_beams,
    target_llm,
    reward_scorer,
    next_dist_seq_prompter,
    next_dist_seq_basemodel,
    beam_scores,
    prev_rewards,
    num_beams_in,
):
    num_chunks = cfg.train.q_params.num_chunks
    assert cfg.train.q_params.top_k % (num_chunks * num_beams_in) == 0
    num_samples_per_beam = cfg.train.q_params.top_k // (num_chunks * num_beams_in)
    all_next_token_candidate_ids = None

    for i in range(cfg.train.q_params.num_chunks):
        next_token_candidate_ids = select_next_token_candidates(
            cfg=cfg,
            next_dist_seq=next_dist_seq_prompter,
            previous_next_token_candidate_ids=all_next_token_candidate_ids,
            num_samples_per_beam=num_samples_per_beam,
            always_include_best=cfg.train.q_params.candidates.always_include_best
            and i == 0,
        )

        candidate_beam_scores, candidate_rewards = (
            evaluate_next_token_candidates_with_reward(
                cfg=cfg,
                instruct=instruct,
                seq_beams=seq_beams,
                target_llm=target_llm,
                reward_scorer=reward_scorer,
                next_token_candidate_ids=next_token_candidate_ids,
                next_dist_seq_basemodel=next_dist_seq_basemodel,
                next_dist_seq_prompter=next_dist_seq_prompter,
                prev_beam_scores=beam_scores,
                prev_rewards=prev_rewards,
            )
        )

        if all_next_token_candidate_ids is None:
            all_next_token_candidate_ids = next_token_candidate_ids
            all_candidate_beam_scores = candidate_beam_scores
            all_candidate_rewards = candidate_rewards
        else:
            all_next_token_candidate_ids = torch.cat(
                (next_token_candidate_ids, all_next_token_candidate_ids), dim=1
            )
            all_candidate_beam_scores = torch.cat(
                (candidate_beam_scores, all_candidate_beam_scores), dim=1
            )
            all_candidate_rewards = torch.cat(
                (candidate_rewards, all_candidate_rewards), dim=1
            )
    return (
        all_next_token_candidate_ids,
        all_candidate_beam_scores,
        all_candidate_rewards,
    )


@torch.no_grad()
def evaluate_next_token_candidates_with_reward(
    cfg,
    instruct,
    seq_beams,
    target_llm,
    reward_scorer,
    next_token_candidate_ids,
    next_dist_seq_basemodel,
    next_dist_seq_prompter,
    prev_beam_scores,
    prev_rewards,
):
    ebs, num_samples_per_beam = next_token_candidate_ids.shape

    q_next_token_candidate_ids = torch.reshape(
        next_token_candidate_ids, (ebs * num_samples_per_beam, 1)
    )
    q_sample_seq = Seq(
        ids=q_next_token_candidate_ids,
        tokenizer=next_dist_seq_prompter.tokenizer,
        device=next_dist_seq_prompter.device,
    )
    candidate_seq, _ = _build_candidate_seq(
        seq_beams=seq_beams,
        q_sample_seq=q_sample_seq,
        num_samples_per_beam=num_samples_per_beam,
    )

    candidate_texts = candidate_seq.text
    full_instruct = Seq(
        text=candidate_texts,
        tokenizer=target_llm.tokenizer,
        device=target_llm.device,
    )
    target_llm_ar = target_llm.generate_autoregressive(
        key="target",
        full_instruct=full_instruct,
        max_new_tokens=getattr(
            cfg.train.q_params,
            "search_response_max_new_tokens",
            None,
        ),
    )
    candidate_responses = target_llm_ar.response_sample.text
    candidate_rewards_flat = _score_reward_batch(
        reward_scorer=reward_scorer,
        questions=candidate_texts,
        answers=candidate_responses,
        device=next_dist_seq_prompter.device,
    )
    candidate_rewards = candidate_rewards_flat.reshape(ebs, num_samples_per_beam)
    reward_delta = candidate_rewards - prev_rewards[:, None]

    next_dist_logprobs_basemodel = next_dist_seq_basemodel.logprobs.squeeze(1)
    selected_logprobs_basemodel = torch.gather(
        next_dist_logprobs_basemodel, dim=-1, index=next_token_candidate_ids
    )

    factor = getattr(
        cfg.train.q_params,
        "reward_lambda_val",
        cfg.train.q_params.lambda_val,
    )
    beam_scores_delta = selected_logprobs_basemodel + reward_delta * factor
    new_beam_scores = prev_beam_scores[:, None] + beam_scores_delta
    standalone_validity_penalty = getattr(
        cfg.train.q_params,
        "standalone_validity_penalty",
        0.0,
    )
    if standalone_validity_penalty > 0:
        candidate_validity = [
            is_valid_standalone_rewrite(text) for text in candidate_seq.text
        ]
        invalid_mask = torch.tensor(
            [not is_valid for is_valid in candidate_validity],
            dtype=new_beam_scores.dtype,
            device=new_beam_scores.device,
        ).reshape(ebs, num_samples_per_beam)
        new_beam_scores = new_beam_scores - invalid_mask * standalone_validity_penalty
    return new_beam_scores, candidate_rewards


@torch.no_grad()
def select_next_beams(
    cfg,
    seq_beams,
    next_token_candidate_ids,
    candidate_beam_scores,
    candidate_losses,
    num_beams_in,
    num_beams_out,
):
    ebs, num_samples_per_beam = candidate_beam_scores.shape
    bs = ebs // num_beams_in

    candidate_beam_scores = candidate_beam_scores.reshape(
        bs, num_beams_in * num_samples_per_beam
    )  # (bs, num_beams_in * num_samples_per_beam)
    if cfg.train.q_params.beams.do_sample:
        if cfg.train.q_params.beams.always_include_best:
            candidate_beam_scores_top_ids = candidate_beam_scores.argmax(dim=-1)
            candidate_beam_scores_onehot = torch.zeros_like(candidate_beam_scores)
            candidate_beam_scores_onehot.scatter_(
                1, candidate_beam_scores_top_ids[:, None], 1
            )
            candidate_beam_scores_corrected = (
                candidate_beam_scores - 1e10 * candidate_beam_scores_onehot
            )
            beam_probs = torch.softmax(
                candidate_beam_scores_corrected / cfg.train.q_params.beams.temperature,
                dim=-1,
            )  # (bs, num_beams_in * num_samples_per_beam)
        else:
            beam_probs = torch.softmax(
                candidate_beam_scores / cfg.train.q_params.beams.temperature,
                dim=-1,
            )  # (bs, num_beams_in * num_samples_per_beam)
        next_beam_indices = beam_probs.multinomial(
            num_samples=num_beams_out, replacement=False
        )  # (bs, num_beams_out) [0, num_beams_in * num_samples_per_beam]
        if cfg.train.q_params.beams.always_include_best:
            next_beam_indices = torch.cat(
                [candidate_beam_scores_top_ids[:, None], next_beam_indices[:, :-1]],
                dim=-1,
            )
    else:
        next_beam_indices = candidate_beam_scores.topk(
            k=num_beams_out, dim=-1, sorted=True
        ).indices  # (bs, num_beams_out) [0, num_beams_in * num_samples_per_beam]

    next_beam_indices_expanded = (
        next_beam_indices
        + torch.arange(0, bs, device=seq_beams.device)[:, None]
        * num_beams_in
        * num_samples_per_beam
    )  # (bs, num_beams_out)
    next_beam_indices_expanded = next_beam_indices_expanded.reshape(-1)

    next_token_candidate_seq = Seq(
        ids=next_token_candidate_ids.reshape(
            bs * num_beams_in * num_samples_per_beam, 1
        ),
        tokenizer=seq_beams.tokenizer,
        device=seq_beams.device,
    )

    parent_beam_indices = torch.div(
        next_beam_indices,
        num_samples_per_beam,
        rounding_mode="floor",
    )  # (bs, num_beams_out)
    parent_beam_indices_expanded = (
        parent_beam_indices
        + torch.arange(0, bs, device=seq_beams.device)[:, None] * num_beams_in
    ).reshape(-1)

    if seq_beams.is_empty:
        next_seq_beams = next_token_candidate_seq[next_beam_indices_expanded]
    else:
        beam_candidates = seq_beams.repeat_interleave(num_samples_per_beam, dim=0)
        beam_candidates.append(next_token_candidate_seq)
        next_seq_beams = beam_candidates[next_beam_indices_expanded]

    candidate_losses = candidate_losses.reshape(
        bs, num_beams_in * num_samples_per_beam
    )  # (bs, num_beams_in * num_samples_per_beam)
    selected_losses = candidate_losses.gather(
        dim=1, index=next_beam_indices
    )  # (bs, num_beams_out)
    selected_losses = selected_losses.reshape(bs * num_beams_out).detach()

    selected_beam_scores = candidate_beam_scores.gather(
        dim=1, index=next_beam_indices
    )  # (bs, num_beams_out)
    selected_beam_scores = selected_beam_scores.reshape(bs * num_beams_out).detach()
    return (
        next_seq_beams,
        selected_losses,
        selected_beam_scores,
        parent_beam_indices_expanded,
    )


@torch.no_grad()
def evaluate_prompt(
    cfg,
    instruct,
    rewrite,
    suffix,
    full_instruct,
    target,
    prompter,
    target_llm,
    generate_target_llm_response,
    print_idx=0,
):
    rewrite = get_rewrite_or_empty(rewrite=rewrite, prompter=prompter)
    basemodel_tf = None
    if suffix is not None and not suffix.is_empty:
        basemodel_tf = prompter.compute_pred_loss_teacher_forced(
            key="suffix",
            instruct=instruct,
            rewrite=rewrite,
            suffix=suffix,
            use_basemodel=True,
            loss_params=dict(hard_labels=True),
        )

    target_llm_tf = target_llm.compute_pred_loss_teacher_forced(
        key="target",
        full_instruct=full_instruct,
        target=target,
        loss_params=dict(
            hard_labels=True,
            reweight_loss=cfg.reweight_loss,
        ),
    )
    if cfg.verbose:
        tqdm.write(f" --- Query[{print_idx}]: '{target_llm_tf.query.text[print_idx]}'")
        if rewrite is not None and not rewrite.is_empty:
            tqdm.write(f" --- Rewrite[{print_idx}]: '{rewrite.text[print_idx]}'")
        tqdm.write(f" --- Suffix[{print_idx}]: '{suffix.text[print_idx]}'")
        tqdm.write(f" --- Target[{print_idx}]: '{target.text[print_idx]}'")
        tqdm.write(
            f" --- TF Response[{print_idx}]: '{target_llm_tf.response_dist.text[print_idx]}'"
        )

    if generate_target_llm_response:
        target_llm_ar = target_llm.generate_autoregressive(
            key="target",
            full_instruct=full_instruct,
        )
        if cfg.verbose:
            tqdm.write(
                f" --- AR Response[{print_idx}]: '{target_llm_ar.response_sample.text[print_idx]}'"
            )
    else:
        target_llm_ar = None

    if cfg.verbose:
        tqdm.write(f" Evaluating suffix completed. TF Loss: {target_llm_tf.loss:.3f}")

    return target_llm_tf, target_llm_ar, basemodel_tf
