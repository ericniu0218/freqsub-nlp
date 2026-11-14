"""Exact candidate log-probabilities under a causal LM.

score(c | prompt) = sum_i log p(c_i | prompt, c_<i)   over the candidate's tokens.

Tokenization convention (applied identically to every candidate, so comparisons are fair):
prompt tokens = tok(prompt) ; candidate tokens = tok(" " + candidate) without special tokens; the
sequence is their concatenation. Prompts must therefore NOT end with a trailing space.

Two implementations:
  * ``score_naive``  -- one full forward pass per (prompt, candidate). Slow, obviously correct.
  * ``score_cached`` -- one pass over the prompt, KV cache repeated across a batch of candidates.
                       ~50x cheaper for 100+ candidates. Tested against score_naive.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class LoadedModel:
    model: torch.nn.Module
    tokenizer: object
    device: str
    dtype: torch.dtype


def load_model(name: str, revision: str | None = None, dtype: str = "float32",
               device: str | None = None) -> LoadedModel:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tdtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
    tok = AutoTokenizer.from_pretrained(name, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(name, revision=revision, dtype=tdtype)
    model.to(device).eval()
    return LoadedModel(model, tok, device, tdtype)


def candidate_token_ids(tokenizer, candidates: list[str]) -> list[list[int]]:
    ids = [tokenizer(" " + c, add_special_tokens=False)["input_ids"] for c in candidates]
    if any(len(x) == 0 for x in ids):
        raise ValueError("empty candidate tokenization")
    return ids


def prompt_token_ids(tokenizer, prompt: str) -> list[int]:
    if prompt.endswith(" "):
        raise ValueError("prompt must not end with a space (the candidate carries the leading space)")
    ids = tokenizer(prompt)["input_ids"]  # keeps BOS if the tokenizer adds one
    return ids


@torch.no_grad()
def score_naive(lm: LoadedModel, prompt: str, candidates: list[str]) -> list[float]:
    p_ids = prompt_token_ids(lm.tokenizer, prompt)
    out = []
    for c_ids in candidate_token_ids(lm.tokenizer, candidates):
        ids = torch.tensor([p_ids + c_ids], device=lm.device)
        logits = lm.model(ids).logits[0].float()
        logp = F.log_softmax(logits, dim=-1)
        pos = torch.arange(len(p_ids) - 1, len(p_ids) - 1 + len(c_ids), device=lm.device)
        out.append(float(logp[pos, torch.tensor(c_ids, device=lm.device)].sum()))
    return out


@torch.no_grad()
def score_cached(lm: LoadedModel, prompt: str, candidates: list[str], batch_size: int = 64,
                 prompt_context=None) -> tuple[list[float], list[int]]:
    """Returns (summed log-probs, token counts) in candidate order.

    prompt_context: optional zero-arg callable returning a context manager that is active during the
    prompt forward pass only (used by H3 to ablate components at the answer position; the candidate
    tokens then attend to the modified KV cache).
    """
    tok = lm.tokenizer
    p_ids = prompt_token_ids(tok, prompt)
    c_all = candidate_token_ids(tok, candidates)

    p = torch.tensor([p_ids], device=lm.device)
    if prompt_context is not None:
        with prompt_context():
            p_out = lm.model(p, use_cache=True)
    else:
        p_out = lm.model(p, use_cache=True)
    first_logp = F.log_softmax(p_out.logits[0, -1].float(), dim=-1)  # predicts each candidate's 1st token
    base_cache = p_out.past_key_values

    scores, lengths = [0.0] * len(c_all), [len(c) for c in c_all]
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    for start in range(0, len(c_all), batch_size):
        chunk = c_all[start:start + batch_size]
        b, max_len = len(chunk), max(len(c) for c in chunk)
        ids = torch.full((b, max_len), pad, dtype=torch.long, device=lm.device)
        for i, c in enumerate(chunk):
            ids[i, :len(c)] = torch.tensor(c, device=lm.device)
        first = first_logp[ids[:, 0]]  # (b,)
        if max_len == 1:
            for i in range(b):
                scores[start + i] = float(first[i])
            continue
        cache = copy.deepcopy(base_cache)
        cache.batch_repeat_interleave(b)
        # right padding: causal attention means pads after a candidate never affect its own tokens
        attn = torch.ones((b, len(p_ids) + max_len), dtype=torch.long, device=lm.device)
        logits = lm.model(ids, past_key_values=cache, attention_mask=attn, use_cache=True).logits.float()
        logp = F.log_softmax(logits[:, :-1], dim=-1)  # position j predicts token j+1
        nxt = ids[:, 1:]
        tok_lp = logp.gather(-1, nxt.unsqueeze(-1)).squeeze(-1)  # (b, max_len-1)
        for i, c in enumerate(chunk):
            scores[start + i] = float(first[i] + tok_lp[i, :len(c) - 1].sum())
    return scores, lengths
