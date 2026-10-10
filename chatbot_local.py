"""Local HF dialogue loader (optional GENERATION_PROVIDER=local)."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

import torch
from dotenv import load_dotenv

load_dotenv()

CHATBOT_MODEL = os.getenv("CHATBOT_MODEL", "tanusrich/Mental_Health_Chatbot")
FALLBACK_MODEL = os.getenv("CHATBOT_FALLBACK_MODEL", "google/flan-t5-base")
MAX_NEW_TOKENS = int(os.getenv("CHATBOT_MAX_NEW_TOKENS", "160"))


@lru_cache(maxsize=1)
def _load_model():
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

    def _one(model_id: str):
        config = AutoConfig.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if config.model_type in {"t5", "mt5", "bart", "pegasus", "fsmt", "marian"}:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
            model.eval()
            return model, tokenizer, "seq2seq", model_id
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        kwargs: dict[str, Any] = {"low_cpu_mem_usage": True}
        if torch.cuda.is_available():
            kwargs.update(device_map="auto", torch_dtype=torch.float16)
        else:
            kwargs["torch_dtype"] = torch.float32
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        if not torch.cuda.is_available():
            model = model.to("cpu")
        model.eval()
        return model, tokenizer, "causal", model_id

    try:
        return _one(CHATBOT_MODEL)
    except Exception:
        return _one(FALLBACK_MODEL)


class LocalHFGenerator:
    def __init__(self):
        model, tokenizer, mode, mid = _load_model()
        self._model = model
        self._tokenizer = tokenizer
        self._mode = mode
        self.model_id = mid

    def generate(self, user_prompt: str) -> str:
        device = next(self._model.parameters()).device
        inputs = self._tokenizer(
            user_prompt, return_tensors="pt", truncation=True, max_length=512
        ).to(device)
        with torch.no_grad():
            if self._mode == "causal":
                out = self._model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
                gen = out[0][inputs["input_ids"].shape[-1] :]
                text = self._tokenizer.decode(gen, skip_special_tokens=True).strip()
            else:
                out = self._model.generate(
                    **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=True, temperature=0.7
                )
                text = self._tokenizer.decode(out[0], skip_special_tokens=True).strip()
        return text.split("User:")[0].strip()
