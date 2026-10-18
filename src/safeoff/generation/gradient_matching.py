import torch
from torch.nn import functional as F
from typing import Dict, List, Optional, Tuple


def _ensure_1d_tensor(data, device):
    """Convert lists/ints to 1D LongTensor on the target device."""
    if isinstance(data, torch.Tensor):
        return data.to(device).view(-1)
    return torch.tensor(data, device=device, dtype=torch.long).view(-1)


def _compute_target_gradients(harmful_lora) -> Dict[str, torch.Tensor]:
    """
    Build the target gradient dictionary from a harmful LoRA model.

    The desired update direction for each LoRA layer is -(B @ A) * scaling,
    which mirrors a negative gradient step toward the harmful adapter weights.
    """
    target = {}
    for name, module in harmful_lora.named_modules():
        if not (hasattr(module, "lora_A") and hasattr(module, "lora_B")):
            continue
        adapter = getattr(module, "active_adapter", None)
        if isinstance(adapter, (list, tuple, set)):
            adapter = list(adapter)[0] if len(adapter) > 0 else None
        if adapter is None:
            available = list(getattr(module, "lora_A", {}).keys())
            adapter = available[0] if available else None
        if adapter is None or adapter not in module.lora_A or adapter not in module.lora_B:
            continue
        if isinstance(module.scaling, dict):
            scale = module.scaling.get(adapter, 1.0)
        else:
            scale = module.scaling
        target[name] = {
            "A": module.lora_A[adapter].weight.detach(),
            "B": module.lora_B[adapter].weight.detach(),
            "scale": scale
        }
    if not target:
        raise ValueError("No LoRA layers found when building target gradients.")
    return target


def random_init(trigger_len: int, vocab_size: int, device) -> torch.Tensor:
    """Randomly initialize trigger tokens."""
    return torch.randint(low=0, high=vocab_size, size=(trigger_len,), device=device)


def concat(trigger_tokens: torch.Tensor, benign_tokens: torch.Tensor) -> torch.Tensor:
    """Concatenate trigger tokens in front of benign tokens."""
    return torch.cat([trigger_tokens, benign_tokens], dim=-1)


def decode(trigger_tokens: torch.Tensor, tokenizer=None) -> str:
    if tokenizer is None:
        return " ".join(str(t) for t in trigger_tokens.tolist())
    return tokenizer.decode(trigger_tokens.tolist(), skip_special_tokens=True)


def _score_vocab(embedding_weight: torch.Tensor, grad_slice: torch.Tensor) -> torch.Tensor:
    """
    Approximate how changing a position to each vocab token affects the loss.
    Lower scores are preferred because we move opposite to the gradient.
    """
    return torch.matmul(embedding_weight, grad_slice)


def generate_candidates(
    trigger_tokens: torch.Tensor,
    input_grads: torch.Tensor,
    embedding_layer,
    top_k: int = 64,
) -> List[torch.Tensor]:
    """
    Generate candidate trigger token sequences by swapping each position with
    tokens that best align with the negative gradient direction.
    """
    vocab_weight = embedding_layer.weight.detach()
    trigger_len = trigger_tokens.shape[-1]
    per_pos_k = max(1, top_k // trigger_len)
    per_pos_k = min(per_pos_k, vocab_weight.shape[0])
    scored_candidates: List[Tuple[float, torch.Tensor]] = []

    with torch.no_grad():
        for pos in range(trigger_len):
            grad_slice = input_grads[pos]
            scores = _score_vocab(vocab_weight, grad_slice)  # shape: (vocab,)
            best_scores, best_ids = torch.topk(-scores, k=per_pos_k)  # negative: move opposite gradient
            for score, token_id in zip(best_scores, best_ids):
                cand = trigger_tokens.clone()
                cand[pos] = token_id
                scored_candidates.append((float(score.item()), cand))

    # Keep unique candidates with best scores
    scored_candidates.sort(key=lambda x: x[0], reverse=False)
    unique: List[torch.Tensor] = []
    seen = set()
    for _, cand in scored_candidates:
        key = tuple(cand.tolist())
        if key in seen:
            continue
        seen.add(key)
        unique.append(cand)
        if len(unique) >= top_k:
            break
    return unique


def get_input_embeddings_grad(
    base_model,
    benign_prompt,
    trigger_tokens,
    benign_output,
) -> torch.Tensor:
    """
    Compute gradients of the loss w.r.t. the trigger token embeddings only.

    The trigger is prepended to the benign input. Trigger labels are masked out
    so the loss is computed only on benign output tokens.
    """
    device = next(base_model.parameters()).device
    trigger_tokens = _ensure_1d_tensor(trigger_tokens, device)
    benign_prompt = _ensure_1d_tensor(benign_prompt, device)
    benign_output = _ensure_1d_tensor(benign_output, device)

    full_input = torch.cat([trigger_tokens, benign_prompt, benign_output], dim=-1).unsqueeze(0)
    labels = torch.cat(
        [
            torch.full_like(trigger_tokens, -100),
            torch.full_like(benign_prompt, -100),
            benign_output,
        ],
        dim=-1,
    ).unsqueeze(0)
    attention_mask = torch.ones_like(full_input, device=device)

    embedding_layer = base_model.get_input_embeddings()
    inputs_embeds = embedding_layer(full_input)
    inputs_embeds.retain_grad()

    base_model.zero_grad(set_to_none=True)
    outputs = base_model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        labels=labels,
    )
    loss = outputs.loss
    loss.backward()
    grad = inputs_embeds.grad.detach()[0, : trigger_tokens.shape[-1]]
    return grad


def get_grad_from_base(base_modules: Dict[str, torch.nn.Module], name: str) -> torch.Tensor:
    """
    Fetch the weight gradient for the base model module corresponding to a LoRA module name.
    """
    if name not in base_modules:
        raise KeyError(f"{name} not found in base model modules.")
    module = base_modules[name]
    if not hasattr(module, "weight") or module.weight.grad is None:
        raise RuntimeError(f"No weight gradient available for module {name}.")
    return module.weight.grad


def gradient_matching_attack(
    base_model,
    benign_input,
    benign_output,
    trigger_len: int = 10,
    iterations: int = 500,
    top_k: int = 64,
    sim_threshold: float = 0.9,
    tokenizer=None,
    target_gradients: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
    harmful_lora=None,
    gcg_goal_ids: Optional[torch.Tensor] = None,
    score_mode: str = "gradient",
):
    """
    Optimize a trigger so that gradients on the base model align with harmful LoRA deltas.

    Either pass precomputed ``target_gradients`` or a ``harmful_lora`` model from
    which they can be derived. If neither is provided, ``base_model`` is assumed
    to carry the LoRA adapters.
    """
    score_mode = score_mode.lower()
    if score_mode not in {"gradient", "gcg"}:
        raise ValueError(f"score_mode must be 'gradient' or 'gcg', got {score_mode}")
    device = next(base_model.parameters()).device
    base_modules = dict(base_model.named_modules())

    if target_gradients is None:
        source_lora = harmful_lora if harmful_lora is not None else base_model
        target_gradients = _compute_target_gradients(source_lora)

    embedding_layer = base_model.get_input_embeddings()
    vocab_size = embedding_layer.weight.shape[0]
    trigger_tokens = random_init(trigger_len, vocab_size=vocab_size, device=device)

    for i in range(iterations):
        input_grads = get_input_embeddings_grad(base_model, benign_input, trigger_tokens, benign_output)
        candidates = generate_candidates(trigger_tokens, input_grads, embedding_layer, top_k=top_k)

        best_score = -float("inf")
        best_similarity = None
        best_gcg_score = None
        best_candidate_tokens: Optional[torch.Tensor] = None

        for candidate in candidates:
            benign_prompt = _ensure_1d_tensor(benign_input, device)
            benign_resp = _ensure_1d_tensor(benign_output, device)
            full_input = torch.cat([candidate, benign_prompt, benign_resp], dim=-1).unsqueeze(0)
            labels = torch.cat(
                [
                    torch.full_like(candidate, -100),
                    torch.full_like(benign_prompt, -100),
                    benign_resp,
                ],
                dim=-1,
            ).unsqueeze(0)
            attention_mask = torch.ones_like(full_input, device=device)

            base_model.zero_grad(set_to_none=True)
            loss = base_model(input_ids=full_input, attention_mask=attention_mask, labels=labels).loss
            loss.backward()

            gcg_score = None
            if gcg_goal_ids is not None:
                with torch.no_grad():
                    gcg_full_input = torch.cat([candidate, benign_prompt, _ensure_1d_tensor(gcg_goal_ids, device)], dim=-1).unsqueeze(0)
                    gcg_labels = torch.cat(
                        [
                            torch.full_like(candidate, -100),
                            torch.full_like(benign_prompt, -100),
                            _ensure_1d_tensor(gcg_goal_ids, device),
                        ],
                        dim=-1,
                    ).unsqueeze(0)
                    gcg_attn = torch.ones_like(gcg_full_input, device=device)
                    gcg_loss = base_model(input_ids=gcg_full_input, attention_mask=gcg_attn, labels=gcg_labels).loss
                    gcg_score = -float(gcg_loss.item())  # higher is better (log-prob)

            total_sim = 0.0
            count = 0
            for name, comps in target_gradients.items():
                try:
                    # Base-model gradient: (out_features, in_features)
                    base_layer_grad = get_grad_from_base(base_modules, name)
                except (KeyError, RuntimeError):
                    continue

                # LoRA components
                lora_A = comps["A"].to(base_layer_grad.device) # (r, in)
                lora_B = comps["B"].to(base_layer_grad.device) # (out, r)
                scale = comps["scale"]

                # --- Projection ---
                # Target direction: base_grad should resemble -(B @ A) * scale.
                # Right-multiplying base_grad by A^T projects it into B's space.
                # base_layer_grad: (out, in)
                # lora_A.T:        (in, r)
                # projected_grad:  (out, r)
                base_layer_grad = base_layer_grad.to(lora_A.dtype)
                projected_grad = torch.matmul(base_layer_grad, lora_A.T)

                # Target vector: -B * scale
                # The comparison space is now (out, r), much smaller than (out, in).
                target_vec = (-lora_B * scale).to(projected_grad.dtype)

                # Cosine similarity
                # After flattening, the dimension is only out * r.
                sim = F.cosine_similarity(
                    projected_grad.flatten(),
                    target_vec.flatten(),
                    dim=0
                )

                total_sim += float(sim.item())
                count += 1

            if count == 0:
                raise RuntimeError("No valid gradients found on base model for similarity computation.")
            avg_sim = total_sim / count

            if score_mode == "gcg":
                candidate_score = gcg_score
            else:
                candidate_score = avg_sim

            if candidate_score is None:
                raise RuntimeError("Candidate scoring failed; check inputs and score_mode.")

            if candidate_score > best_score:
                best_score = candidate_score
                best_similarity = avg_sim
                best_gcg_score = gcg_score
                best_candidate_tokens = candidate.detach()

            base_model.zero_grad(set_to_none=True)

        if best_candidate_tokens is None:
            raise RuntimeError("Failed to produce any candidate triggers.")

        trigger_tokens = best_candidate_tokens
        log_parts = [f"Iter {i}: score_mode={score_mode}", f"trigger={decode(trigger_tokens, tokenizer=tokenizer)}"]
        if best_similarity is not None:
            log_parts.append(f"cos_sim={best_similarity:.4f}")
        if best_gcg_score is not None:
            log_parts.append(f"gcg_logprob={best_gcg_score:.4f}")
        print(", ".join(log_parts))

        if score_mode == "gradient" and best_similarity is not None and best_similarity >= sim_threshold:
            break

    return trigger_tokens
