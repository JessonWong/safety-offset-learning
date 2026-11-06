"""Resolve chat boundaries without changing the search's sequence fragments."""


CHAT_TEMPLATE_VERSION = "tokenizer_native_v1"


def resolve_chat_template(tokenizer, prompt_template):
    """Use this tokenizer's user prefix and assistant generation prefix.

    Render once around a sentinel so dynamic instruction/beam sequences can
    still be concatenated without decoding them or losing their gradients.
    Existing prompt templates without chat boundaries are left unchanged.
    """
    if not any("chat_boundary" in part for part in prompt_template):
        return prompt_template

    marker = "__LLM_FT_CHAT_USER_CONTENT__"
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": marker}],
        tokenize=False,
        add_generation_prompt=True,
    )
    if rendered.count(marker) != 1:
        raise ValueError("Chat template must preserve user content exactly once.")
    user_prefix, assistant_prefix = rendered.split(marker)
    boundaries = {"user": user_prefix, "assistant": assistant_prefix}

    resolved = []
    for part in prompt_template:
        part = dict(part)
        if "chat_boundary" in part:
            boundary = part["chat_boundary"]
            if boundary not in boundaries:
                raise ValueError(f"Unknown chat boundary: {boundary}")
            part["msg"] = boundaries[boundary]
        resolved.append(part)
    return resolved
