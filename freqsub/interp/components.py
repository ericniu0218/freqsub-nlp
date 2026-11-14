"""Mechanism tools for OLMo-2 (H3), written with plain PyTorch hooks (no extra dependencies).

OLMo-2 decoder layer (verified from transformers source):
    h = h + post_attention_layernorm(self_attn(h))
    h = h + post_feedforward_layernorm(mlp(h))
so the residual-stream contribution of each component is the OUTPUT of its post-norm module.
Final logits = lm_head(norm(h_L)). Embeddings are untied.

* ``run_with_cache``: residual stream after every layer + every component's contribution, last position.
* ``logit_lens``: lm_head(final_norm(resid_l)) for each layer, as a target-vs-foil logit difference.
* ``direct_logit_attribution``: with the final RMSNorm scale frozen at its actual value, the final
  logit difference decomposes EXACTLY into embedding + sum of component contributions.
* ``ablate``: replace chosen components' last-position contributions with fixed (mean) vectors.
"""
from __future__ import annotations

from contextlib import contextmanager

import torch


def component_modules(model) -> dict[str, torch.nn.Module]:
    mods = {}
    for i, layer in enumerate(model.model.layers):
        mods[f"L{i}.attn"] = layer.post_attention_layernorm
        mods[f"L{i}.mlp"] = layer.post_feedforward_layernorm
    return mods


@contextmanager
def hooks(handles):
    try:
        yield
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def run_with_cache(lm, input_ids: list[int]) -> dict:
    model = lm.model
    comps, resid = {}, {}
    handles = []
    for name, mod in component_modules(model).items():
        handles.append(mod.register_forward_hook(
            lambda m, i, o, name=name: comps.__setitem__(name, o[0, -1].float().clone())))
    for i, layer in enumerate(model.model.layers):
        handles.append(layer.register_forward_hook(
            lambda m, i_, o, i=i: resid.__setitem__(i, (o[0] if isinstance(o, tuple) else o)[0, -1].float().clone())))
    emb = {}
    handles.append(model.model.embed_tokens.register_forward_hook(
        lambda m, i, o: emb.__setitem__("e", o[0, -1].float().clone())))
    with hooks(handles):
        out = model(torch.tensor([input_ids], device=lm.device))
    return {"components": comps, "resid": resid, "embed": emb["e"], "logits": out.logits[0, -1].float()}


def _unembed_diff(model, target: int, foil: int) -> torch.Tensor:
    W = model.lm_head.weight.float()
    return W[target] - W[foil]


@torch.no_grad()
def logit_lens(lm, cache: dict, target: int, foil: int) -> list[float]:
    model = lm.model
    d = _unembed_diff(model, target, foil)
    out = []
    for i in sorted(cache["resid"]):
        normed = model.model.norm(cache["resid"][i].to(model.lm_head.weight.dtype)).float()
        out.append(float(normed @ d))
    return out


@torch.no_grad()
def direct_logit_attribution(lm, cache: dict, target: int, foil: int) -> dict[str, float]:
    """Exact decomposition of the final logit difference under the frozen final-norm scale."""
    model = lm.model
    norm = model.model.norm
    h_final = cache["resid"][max(cache["resid"])]
    eps = getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-6))
    rms = torch.sqrt(h_final.pow(2).mean() + eps)
    g = norm.weight.float()
    d = _unembed_diff(model, target, foil)
    w = (g * d) / rms  # linear map: residual-stream vector -> logit difference contribution
    attrib = {"embed": float(cache["embed"] @ w)}
    for name, v in cache["components"].items():
        attrib[name] = float(v @ w)
    attrib["_total"] = float(h_final @ w)
    attrib["_actual"] = float(cache["logits"][target] - cache["logits"][foil])
    return attrib


@torch.no_grad()
def logits_with_ablation(lm, input_ids: list[int], replacements: dict[str, torch.Tensor]) -> torch.Tensor:
    """Run the model with the last-position output of each named component replaced."""
    model = lm.model
    mods = component_modules(model)
    handles = []
    for name, vec in replacements.items():
        def hook(m, i, o, vec=vec):
            o = o.clone()
            o[0, -1] = vec.to(o.dtype)
            return o
        handles.append(mods[name].register_forward_hook(hook))
    with hooks(handles):
        out = model(torch.tensor([input_ids], device=lm.device))
    return out.logits[0, -1].float()


@contextmanager
def ablation_at_last_position(lm, replacements: dict[str, torch.Tensor]):
    """Replace each named component's output at the LAST position of the forward pass it wraps."""
    mods = component_modules(lm.model)
    handles = []
    for name, vec in replacements.items():
        def hook(m, i, o, vec=vec):
            o = o.clone()
            o[:, -1] = vec.to(o.dtype)
            return o
        handles.append(mods[name].register_forward_hook(hook))
    try:
        yield
    finally:
        for h in handles:
            h.remove()
