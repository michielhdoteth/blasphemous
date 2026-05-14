from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from .train_prompts import HARMFUL_PROMPTS, HARMLESS_PROMPTS


@dataclass
class CausalComponent:
    name: str           # e.g. "attn_12" or "mlp_20"
    layer: int
    component: str      # "attn" | "mlp"
    importance: float   # refusal probability delta when patched


@dataclass
class CausalReport:
    components: list[CausalComponent]   # sorted by importance desc
    top_attn_layers: list[int]
    top_mlp_layers: list[int]
    causal_layer_mask: dict[int, dict[str, float]]  # {layer: {attn: w, mlp: w}}


def _get_refusal_token_ids(tokenizer) -> list[int]:
    """Get token IDs for common refusal phrases."""
    refusal_phrases = [" Sorry", " cannot", " can't", " apologize", " unable", " harmful", " illegal"]
    ids = set()
    for phrase in refusal_phrases:
        encoded = tokenizer.encode(phrase, add_special_tokens=False)
        ids.update(encoded)
    return list(ids)


def _refusal_logit_score(logits: torch.Tensor, refusal_ids: list[int]) -> float:
    """Compute refusal logit score from model output."""
    probs = F.softmax(logits.float(), dim=-1)
    return sum(probs[tid].item() for tid in refusal_ids if tid < probs.shape[0])


@torch.no_grad()
def _cache_activations(model, inputs: dict) -> dict[str, torch.Tensor]:
    """Cache activations from a forward pass."""
    cache = {}

    hooks = []
    for i, layer in enumerate(model.model.layers):
        def make_attn_hook(name):
            def hook(module, input, output):
                out = output[0] if isinstance(output, tuple) else output
                cache[name] = out.detach().clone()
            return hook

        def make_mlp_hook(name):
            def hook(module, input, output):
                cache[name] = output.detach().clone()
            return hook

        hooks.append(layer.self_attn.o_proj.register_forward_hook(make_attn_hook(f"attn_{i}")))
        hooks.append(layer.mlp.down_proj.register_forward_hook(make_mlp_hook(f"mlp_{i}")))

    model(**inputs)
    for h in hooks:
        h.remove()

    return cache


@torch.no_grad()
def run_causal_mediation(
    model,
    tokenizer,
    device: str,
    n_pairs: int = 8,
    top_k: int = 10,
) -> CausalReport:
    """
    For each (layer, component) pair:
      1. Cache activations from a harmful forward pass
      2. Run harmless forward pass
      3. Patch each component's output from harmless→harmful one at a time
      4. Measure refusal logit delta — how much does patching restore refusal?
      5. Components with highest delta causally mediate refusal
    """
    print("[BLASPHEMOUS] Causal mediation analysis...")
    model.eval()

    refusal_ids = _get_refusal_token_ids(tokenizer)
    n_layers = len(list(model.model.layers))

    importance_accumulator: dict[str, float] = {}

    pairs = list(zip(HARMFUL_PROMPTS[:n_pairs], HARMLESS_PROMPTS[:n_pairs]))

    for pair_idx, (harmful_prompt, harmless_prompt) in enumerate(pairs):
        harmful_inputs = tokenizer(
            harmful_prompt, return_tensors="pt", truncation=True, max_length=256
        ).to(device)
        harmless_inputs = tokenizer(
            harmless_prompt, return_tensors="pt", truncation=True, max_length=256
        ).to(device)

        # Cache harmful activations
        harmful_cache = _cache_activations(model, harmful_inputs)

        # Baseline: harmless refusal score (should be low)
        harmless_logits = model(**harmless_inputs).logits[0, -1]
        baseline_score = _refusal_logit_score(harmless_logits, refusal_ids)

        # Patch each component individually
        for comp_name, cached_act in harmful_cache.items():
            layer_idx = int(comp_name.split("_")[1])
            comp_type = comp_name.split("_")[0]

            # Register patch hook
            patch_target_act = cached_act

            def make_patch_hook(target):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        return (target,) + output[1:]
                    return target
                return hook

            if comp_type == "attn":
                handle = model.model.layers[layer_idx].self_attn.o_proj.register_forward_hook(
                    make_patch_hook(patch_target_act)
                )
            else:
                handle = model.model.layers[layer_idx].mlp.down_proj.register_forward_hook(
                    make_patch_hook(patch_target_act)
                )

            try:
                patched_logits = model(**harmless_inputs).logits[0, -1]
                patched_score = _refusal_logit_score(patched_logits, refusal_ids)
                delta = patched_score - baseline_score
            except Exception:
                delta = 0.0
            finally:
                handle.remove()

            importance_accumulator[comp_name] = importance_accumulator.get(comp_name, 0.0) + delta

        print(f"  Pair {pair_idx+1}/{n_pairs} done")

    # Average over pairs
    components = []
    for name, total_delta in importance_accumulator.items():
        layer_idx = int(name.split("_")[1])
        comp_type = name.split("_")[0]
        components.append(CausalComponent(
            name=name,
            layer=layer_idx,
            component=comp_type,
            importance=total_delta / n_pairs,
        ))

    components.sort(key=lambda c: c.importance, reverse=True)

    # Build causal mask: only top_k components get non-zero weight
    top_components = components[:top_k]
    causal_layer_mask: dict[int, dict[str, float]] = {}

    for c in top_components:
        if c.layer not in causal_layer_mask:
            causal_layer_mask[c.layer] = {"attn": 0.0, "mlp": 0.0}
        causal_layer_mask[c.layer][c.component] = c.importance

    # Normalize weights within mask
    max_imp = max((c.importance for c in top_components), default=1.0)
    for layer in causal_layer_mask:
        for comp in causal_layer_mask[layer]:
            causal_layer_mask[layer][comp] /= (max_imp + 1e-8)

    top_attn = sorted(
        [c.layer for c in top_components if c.component == "attn"],
        key=lambda l: causal_layer_mask.get(l, {}).get("attn", 0), reverse=True
    )
    top_mlp = sorted(
        [c.layer for c in top_components if c.component == "mlp"],
        key=lambda l: causal_layer_mask.get(l, {}).get("mlp", 0), reverse=True
    )

    print(f"  Top causal attn layers: {top_attn[:5]}")
    print(f"  Top causal MLP  layers: {top_mlp[:5]}")
    print(f"  Causally verified components: {len(top_components)}/{n_layers*2}")

    return CausalReport(
        components=components,
        top_attn_layers=top_attn,
        top_mlp_layers=top_mlp,
        causal_layer_mask=causal_layer_mask,
    )
