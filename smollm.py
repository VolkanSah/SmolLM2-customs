# =============================================================================
# smollm.py
# SmolLM2 Inference Engine
# SmolLM2 Service Space
# Copyright 2026 - Volkan Kücükbudak
# Apache License V2 + ESOL 1.1
# =============================================================================

import logging
import torch
from typing import Optional
import model as model_module

logger = logging.getLogger("smollm")

_tokenizer = None
_model     = None
_device    = None


def load():
    """Lazy model loader — called on first request."""
    global _tokenizer, _model, _device

    if _model is not None:
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = model_module.get_model_id()
    kwargs   = model_module.get_model_kwargs()
    _device  = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Loading {model_id} on {_device}...")
    _tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)
    _model     = AutoModelForCausalLM.from_pretrained(model_id, **kwargs).to(_device)
    logger.info(f"Model ready [{_device}]")

    # Update model card on startup
    model_module.push_model_card({
        "model_id": model_id,
        "device": _device,
    })


async def complete(
    prompt: str,
    system_prompt: str = "",
    max_tokens: int = 150,
    temperature: float = 0.2,
) -> str:
    """
    Run SmolLM2 inference.

    Returns:
        Generated text string.
    Raises:
        RuntimeError on inference failure.
    """
    load()

    messages = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    text   = _tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = _tokenizer.encode(text, return_tensors="pt").to(_device)

    with torch.no_grad():
        outputs = _model.generate(
            inputs,
            max_new_tokens=max_tokens,
            temperature=temperature if temperature > 0 else None,
            do_sample=temperature > 0,
            top_p=0.9 if temperature > 0 else None,
            pad_token_id=_tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][inputs.shape[-1]:]
    return _tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def is_ready() -> bool:
    return _model is not None

def device_info() -> str:
    return _device or "not loaded"
