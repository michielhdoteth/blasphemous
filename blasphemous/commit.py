from __future__ import annotations

import math
from dataclasses import dataclass


import torch
import torch.nn.functional as F

from .analyze import AnalysisReport
from .extract import DirectionManifold, project_weights
from .train_prompts import HARMFUL_PROMPTS, HARMLESS_PROMPTS
from .causal import CausalReport
from .ui import info, metric, success
from .optimize import (
    OptimizationResult,
    _apply_ablation,
    _measure_refusal_rate,
    _measure_kl,
    _kernel_weights,
)
from .extract import build_manifold  # Add for re-extraction


@dataclass
class CommitResult:
    refusal_rate: float
    kl_divergence: float
    ouroboros_compensated: bool
    n_ouroboros_passes: int
    output_path: str | None
    compensations_performed: list[dict]  # Track each compensation pass
    # v0.2.0 additions
    causal_report: CausalReport | None
    n_causal_components: int = 0  # Number of causally verified components


def _get_refusal_token_ids(tokenizer) -> list[int]:
    """Get token IDs for common refusal phrases."""
    refusal_phrases = [
        " Sorry",
        " cannot",
        " can't",
        " apologize",
        " unable",
        " harmful",
        " illegal",
    ]
    ids = set()
    for phrase in refusal_phrases:
        encoded = tokenizer.encode(phrase, add_special_tokens=False)
        ids.update(encoded)
    return list(ids)


def _measure_ouroboros_signal(
    model,
    tokenizer,
    manifold: DirectionManifold,
    device: str,
    n_prompts: int = 10,
) -> float:
    """Measure the strength of ouroboros effect after modifications.

    Compares harmless residual activation projection onto refusal direction
    before and after ablation. High harmless projection indicates
    ouroboros self-repair is occurring.

    Returns:
        Float 0-1 indicating ouroboros signal strength
    """
    model.eval()

    refusal_ids = _get_refusal_token_ids(tokenizer)

    direction = manifold.sample(0.0).to(device)
    n_model_layers = len(list(model.model.layers))

    harmless_signal = 0.0

    with torch.no_grad():
        for prompt in HARMLESS_PROMPTS[:n_prompts]:
            inputs = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=256
            ).to(device)
            outputs = model(**inputs, output_hidden_states=True)

            # Measure projection onto refusal direction across layers
            layer_signals = []
            for layer_idx, hidden in enumerate(outputs.hidden_states):
                if layer_idx < n_model_layers:
                    vec = hidden[0, -1].float()
                    proj = (vec @ direction).abs().item()
                    layer_signals.append(proj)

            harmless_signal += sum(layer_signals)

    # Normalize by prompts and layers
    normalized_signal = harmless_signal / (n_prompts * n_model_layers + 1e-8)

    # Also check if harmless prompts trigger refusal logits
    refusal_logits_signal = 0.0
    with torch.no_grad():
        for prompt in HARMLESS_PROMPTS[:n_prompts]:
            inputs = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=256
            ).to(device)
            logits = model(**inputs).logits[0, -1]
            probs = F.softmax(logits.float(), dim=-1)
            refusal_prob = sum(
                probs[tid].item() for tid in refusal_ids if tid < probs.shape[0]
            )
            refusal_logits_signal += refusal_prob

    refusal_logits_signal /= n_prompts

    # Combine signals: both harmless activation projection and refusal logits
    ouroboros_signal = (
        0.6 * min(1.0, normalized_signal * 100) + 0.4 * refusal_logits_signal
    )

    return float(ouroboros_signal)


def multi_pass_ablate(
    model,
    tokenizer,
    original_model,
    manifold: DirectionManifold,
    report: AnalysisReport,
    params,
    device: str = "cuda",
    n_passes: int = 3,
) -> dict:
    """Apply multi-pass ablation with iterative direction refinement.
    
    This is the KEY addition that OBLITERATUS/Heretic use:
    - Pass 1: Apply initial ablation
    - Pass 2: Re-extract residual direction, apply again  
    - Pass 3: Final refinement
    
    Each pass targets the REMAINING refusal that survived previous passes.
    
    Args:
        model: The model to ablate (modified in place)
        tokenizer: Tokenizer
        original_model: Clean copy for KL comparison
        manifold: Direction manifold from initial extraction
        report: Analysis report
        params: Search parameters from optimization
        device: cuda/cpu
        n_passes: Number of ablation passes (default 3)
        
    Returns:
        Dict with pass results and final metrics
    """
    from .extract import build_manifold, select_refusal_layers
    
    info(f"Starting multi-pass ablation ({n_passes} passes)...")
    
    pass_results = []
    current_refusal = 1.0
    
    for pass_idx in range(n_passes):
        info(f"Ablation pass {pass_idx + 1}/{n_passes}...")
        
        # Apply ablation with current parameters
        if getattr(params, "method", "projection") == "projection":
            _apply_ablation(model, manifold, params, device)
        else:
            _apply_ablation(model, manifold, params, device)
        
        # Measure refusal after this pass
        refusal_after = _measure_refusal_rate(model, tokenizer, device, n_prompts=20)
        kl_after = _measure_kl(model, original_model, tokenizer, device, n_prompts=20)
        
        info(f"  Pass {pass_idx + 1}: refusal={refusal_after:.3f}, KL={kl_after:.6f}")
        
        pass_results.append({
            "pass": pass_idx + 1,
            "refusal": refusal_after,
            "kl": kl_after,
        })
        
        # Check if we've reached acceptable level
        if refusal_after < 0.15:
            info(f"  Reached acceptable refusal: {refusal_after:.3f}")
            break
        
        # If more passes remaining, re-extract direction from residual refusal
        if pass_idx < n_passes - 1 and refusal_after > 0.10:
            info(f"  Re-extracting residual direction for next pass...")
            try:
                # Build new manifold from current model state
                new_manifold = build_manifold(
                    model,
                    tokenizer,
                    HARMFUL_PROMPTS,
                    HARMLESS_PROMPTS,
                    device,
                )
                
                # Get refined direction from new manifold (with higher index for residual)
                new_direction_index = min(
                    params.direction_index + (pass_idx + 1) * 0.5,
                    new_manifold.n_layers - 1,
                )
                
                new_direction = new_manifold.sample(
                    new_direction_index,
                    direction_type="whitened",
                ).to(device)
                
                # Select good layers (skip noisy ones)
                good_layers = select_refusal_layers(new_manifold, min_silhouette=0.4)
                
                # Apply refined direction to good layers only
                for layer_idx in good_layers[:8]:  # Top 8 layers
                    layer = model.model.layers[layer_idx]
                    if hasattr(layer.self_attn, "o_proj"):
                        w = layer.self_attn.o_proj.weight.data.float()
                        row_norms = w.norm(dim=1, keepdim=True)
                        w_norm = w / row_norms.clamp(min=1e-8)
                        
                        v = new_direction[:w.shape[1]].float()
                        if v.shape[0] < w.shape[1]:
                            v = torch.zeros(w.shape[1], device=w.device, dtype=v.dtype)
                        
                        proj = (w_norm @ v).unsqueeze(-1) * v.unsqueeze(0)
                        w_new = w_norm - 0.5 * proj  # Lighter weight for refinement
                        w_restored = w_new * row_norms
                        
                        layer.self_attn.o_proj.weight.data = w_restored.to(w.dtype)
                
                info(f"  Refined {len(good_layers[:8])} layers with residual direction")
                
            except Exception as e:
                info(f"  Warning: Re-extraction failed ({e}), continuing with current direction")
    
    return {
        "n_passes": len(pass_results),
        "passes": pass_results,
        "final_refusal": pass_results[-1]["refusal"] if pass_results else 1.0,
    }


def _apply_focused_compensation(
    model,
    direction: torch.Tensor,
    current_signal: float,
    device: str,
    strength: float = 0.5,
):
    """Apply focused compensation injection to counteract ouroboros.

    Injects a small amount of original signal back into the network
    at components with high entanglement, calibrated by current signal strength.

    Args:
        direction: The refusal direction to compensate against
        current_signal: Current ouroboros signal strength (0-1)
        strength: Compensation strength multiplier
    """
    n_model_layers = len(list(model.model.layers))

    # Compensation is stronger when signal is higher (more aggressive compensation)
    # But we cap it to avoid over-compensation
    compensation_strength = strength * min(current_signal, 0.8)

    # Use a narrower kernel for focused compensation
    peak_layer_idx = n_model_layers // 2
    sigma = n_model_layers / 8.0

    for i, layer in enumerate(model.model.layers):
        # Bell-curve weight centered at peak refusal layer
        layer_weight = compensation_strength * math.exp(
            -0.5 * ((i - peak_layer_idx) / sigma) ** 2
        )

        if layer_weight < 1e-4:
            continue

        try:
            # Apply to attention output projection
            w = layer.self_attn.o_proj.weight.data
            # Weight shape is [output_dim, input_dim]
            # Direction should be applied to input_dim
            # If direction doesn't match input_dim, skip or use projection
            w_dtype = w.dtype
            d_float = direction.float()

            # Check dimensions and handle mismatch
            if w.shape[1] == d_float.shape[0]:
                # Direction matches input dimension
                d_dtype = d_float.to(w_dtype)
                projection = (w @ d_dtype).unsqueeze(-1) * d_dtype.unsqueeze(0)
                layer.self_attn.o_proj.weight.data = w + (layer_weight * projection).to(
                    w_dtype
                )
            elif w.shape[1] == d_float.shape[0] * 2:
                # Direction matches 2x input dimension (down projection)
                d_dtype = d_float.to(w_dtype)
                projection = (w[:, : d_float.shape[0]] @ d_dtype).unsqueeze(
                    -1
                ) * d_dtype.unsqueeze(0)
                # Pad result to match original shape
                pad = torch.zeros(
                    w.shape[0],
                    w.shape[1] - d_float.shape[0],
                    device=device,
                    dtype=w_dtype,
                )
                layer.self_attn.o_proj.weight.data = w + (
                    layer_weight * torch.cat([projection, pad], dim=1)
                ).to(w_dtype)
        except AttributeError:
            pass

        try:
            # Apply to MLP down projection
            w = layer.mlp.down_proj.weight.data
            w_dtype = w.dtype
            d_float = direction.float()

            # Similar dimension handling for MLP
            if w.shape[1] == d_float.shape[0]:
                d_dtype = d_float.to(w_dtype)
                projection = (w @ d_dtype).unsqueeze(-1) * d_dtype.unsqueeze(0)
                layer.mlp.down_proj.weight.data = w + (layer_weight * projection).to(
                    w_dtype
                )
            elif w.shape[1] == d_float.shape[0] * 2:
                d_dtype = d_float.to(w_dtype)
                projection = (w[:, : d_float.shape[0]] @ d_dtype).unsqueeze(
                    -1
                ) * d_dtype.unsqueeze(0)
                pad = torch.zeros(
                    w.shape[0],
                    w.shape[1] - d_float.shape[0],
                    device=device,
                    dtype=w_dtype,
                )
                layer.mlp.down_proj.weight.data = w + (
                    layer_weight * torch.cat([projection, pad], dim=1)
                ).to(w_dtype)
        except AttributeError:
            pass


def _apply_ablation_with_causal(
    model,
    manifold: DirectionManifold,
    params,
    device: str,
    causal_report: CausalReport | None,
):
    """Apply per-layer orthogonal projection in-place with causal scaling.

    v0.2.0: Scales projection strength by causal importance weights.
    Only projects components with high causal importance.
    """
    from .optimize import _kernel_weights

    direction_type = getattr(params, "direction_type", "whitened")
    direction_alpha = None
    probe_alpha = getattr(params, "probe_alpha", 0.0)
    safe_alpha = getattr(params, "safe_alpha", 0.0)
    aggressive = getattr(params, "aggressive", False)

    # Mix direction types based on alpha parameters
    if probe_alpha > 0 or safe_alpha > 0:
        direction_alpha = probe_alpha if probe_alpha > 0 else safe_alpha
        if probe_alpha >= safe_alpha and probe_alpha > 0:
            direction_type = "probe"
        elif safe_alpha > 0:
            direction_type = "safe"

    # Sample direction and ensure it matches weight dtype
    sampled_dir = manifold.sample(
        params.direction_index, direction_type=direction_type, alpha=direction_alpha
    )
    direction = sampled_dir.to(device)
    n_model_layers = len(list(model.model.layers))

    attn_weights = _kernel_weights(
        n_model_layers,
        params.kernel_peak_pos,
        params.attn_max_weight,
        params.kernel_min_weight,
        aggressive=aggressive,
    )
    mlp_weights = _kernel_weights(
        n_model_layers,
        params.kernel_peak_pos,
        params.mlp_max_weight,
        params.kernel_min_weight,
        aggressive=aggressive,
    )

    for i, layer in enumerate(model.model.layers):
        # Apply causal mask if available
        causal_weight_attn = 1.0
        causal_weight_mlp = 1.0

        if causal_report and causal_report.causal_layer_mask:
            if i in causal_report.causal_layer_mask:
                layer_mask = causal_report.causal_layer_mask[i]
                causal_weight_attn = layer_mask.get("attn", 1.0)
                causal_weight_mlp = layer_mask.get("mlp", 1.0)
            else:
                # No causal data for this layer - use reduced weight
                causal_weight_attn = 0.3
                causal_weight_mlp = 0.3

        # Scale projection strength by causal weight
        attn_strength = attn_weights[i] * causal_weight_attn
        mlp_strength = mlp_weights[i] * causal_weight_mlp

        if attn_strength > 1e-4:
            try:
                w = layer.self_attn.o_proj.weight.data
                # Handle dimension mismatch - project_weights already handles this
                layer.self_attn.o_proj.weight.data = project_weights(
                    w, direction, attn_strength
                )
            except AttributeError:
                pass

        if mlp_strength > 1e-4:
            try:
                w = layer.mlp.down_proj.weight.data
                # Handle dimension mismatch - project_weights already handles this
                layer.mlp.down_proj.weight.data = project_weights(
                    w, direction, mlp_strength
                )
            except AttributeError:
                pass


def commit(
    model,
    tokenizer,
    original_model,
    manifold: DirectionManifold,
    report: AnalysisReport,
    opt_result: OptimizationResult,
    output_path: str | None = None,
    device: str = "cuda",
    causal_report: CausalReport | None = None,
    use_causal: bool = True,
    causal_pairs: int = 8,
    causal_top_k: int = 10,
    residual_threshold: float = 50.0,
) -> CommitResult:
    # Note: Phase 4 banner is printed by pipeline.py before calling commit()

    params = opt_result.params

    # v0.2.0: Run causal mediation if not provided and enabled
    if causal_report is None and use_causal:
        from .causal import run_causal_mediation

        info("Running causal mediation analysis...")
        causal_report = run_causal_mediation(
model, tokenizer, device, n_pairs=causal_pairs, top_k=causal_top_k
        )

    # Use MULTI-PASS ablation (like OBLITERATUS) - key for generalization!
    # This applies iterative refinement - each pass targets remaining refusal
    multi_pass_result = multi_pass_ablate(
        model,
        tokenizer,
        original_model,
        manifold,
        report,
        params,
        device,
        n_passes=3,  # OBLITERATUS uses 2-3 passes
    )

    # Get final metrics
    refusal = _measure_refusal_rate(model, tokenizer, device, n_prompts=30)
    kl = _measure_kl(model, original_model, tokenizer, device, n_prompts=30)
    metric("Post-commit refusal", f"{refusal:.3f}")
    metric("Post-commit KL", f"{kl:.6f}")
    metric("Multi-pass result", f"{multi_pass_result.get('n_passes', 0)} passes")

    ouroboros_compensated = False
    n_comp_passes = 0
    compensations_performed = []

    # Iterative ouroboros compensation with dynamic adaptation
    # NOTE: Disabled by default - compensation fights the ablation!
    # The original Heretic approach doesn't use compensation
    refusal_threshold = 0.15
    max_compensations = 0  # Disabled - compensation is counterproductive
    ouroboros_threshold = 0.2

    current_refusal = refusal
    current_ouroboros_signal = _measure_ouroboros_signal(
        model, tokenizer, manifold, device, n_prompts=10
    )

    metric("Initial ouroboros signal", f"{current_ouroboros_signal:.3f}")

    while current_refusal > refusal_threshold and n_comp_passes < max_compensations:
        # Check if ouroboros effect is present
        if current_ouroboros_signal < ouroboros_threshold:
            print(
                f"  Ouroboros signal too low ({current_ouroboros_signal:.3f}), stopping compensation"
            )
            break

        print(f"  Compensation pass {n_comp_passes + 1}/{max_compensations}...")
        print(
            f"    Current refusal: {current_refusal:.3f}, ouroboros signal: {current_ouroboros_signal:.3f}"
        )

        # Apply focused compensation
        direction = manifold.sample(params.direction_index).to(device)
        _apply_focused_compensation(
            model, direction, current_ouroboros_signal, device, strength=0.3
        )

        # Re-measure refusal and ouroboros signal
        refusal_check = _measure_refusal_rate(model, tokenizer, device, n_prompts=20)
        ouroboros_check = _measure_ouroboros_signal(
            model, tokenizer, manifold, device, n_prompts=10
        )

        compensation_info = {
            "pass": n_comp_passes + 1,
            "refusal_before": current_refusal,
            "refusal_after": refusal_check,
            "ouroboros_before": current_ouroboros_signal,
            "ouroboros_after": ouroboros_check,
        }
        compensations_performed.append(compensation_info)

        print(
            f"    Post-compensation: refusal={refusal_check:.3f}, ouroboros signal={ouroboros_check:.3f}"
        )

        n_comp_passes += 1
        current_refusal = refusal_check
        current_ouroboros_signal = ouroboros_check

        # Early termination if refusal drops significantly
        if current_refusal < 0.05:
            print(f"  Refusal dropped to {current_refusal:.3f}, stopping compensation")
            break

    if n_comp_passes > 0:
        refusal = _measure_refusal_rate(model, tokenizer, device, n_prompts=30)
        kl = _measure_kl(model, original_model, tokenizer, device, n_prompts=30)
        ouroboros_compensated = True
        print(
            f"  Post-compensation ({n_comp_passes} passes): refusal={refusal:.3f} kl={kl:.3f}"
        )

    if output_path:
        info(f"Saving model to {output_path}...")
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        _save_metadata(
            output_path,
            report,
            opt_result,
            refusal,
            kl,
            compensations_performed,
            causal_report,
        )
        success(f"Model saved to {output_path}")

    n_causal_components = len(causal_report.components) if causal_report else 0

    return CommitResult(
        refusal_rate=refusal,
        kl_divergence=kl,
        ouroboros_compensated=ouroboros_compensated,
        n_ouroboros_passes=n_comp_passes,
        output_path=output_path,
        compensations_performed=compensations_performed,
        causal_report=causal_report,
        n_causal_components=n_causal_components,
    )


def _save_metadata(
    path: str,
    report: AnalysisReport,
    opt_result: OptimizationResult,
    refusal: float,
    kl: float,
    compensations: list,
    causal_report: CausalReport | None,
):
    import json
    import os

    meta = {
        "blasphemous_version": "0.4.0",
        "alignment_type": report.alignment_type,
        "cone_type": report.cone_type,
        "ouroboros_risk": report.ouroboros_risk,
        "peak_layer": report.peak_layer,
        "optimization": {
            "method": getattr(opt_result.params, "method", "projection"),
            "layer_strategy": getattr(opt_result.params, "layer_strategy", "centered"),
            "direction_index": opt_result.params.direction_index,
            "attn_max_weight": opt_result.params.attn_max_weight,
            "mlp_max_weight": opt_result.params.mlp_max_weight,
            "kernel_peak_pos": opt_result.params.kernel_peak_pos,
            "kernel_min_weight": opt_result.params.kernel_min_weight,
            "n_refinement_passes": opt_result.params.n_refinement_passes,
            "n_trials": opt_result.n_trials,
            "objective_value": opt_result.objective_value,
            "probe_alpha": getattr(opt_result.params, "probe_alpha", 0.0),
            "safe_alpha": getattr(opt_result.params, "safe_alpha", 0.0),
            "direction_type": getattr(opt_result.params, "direction_type", "whitened"),
        },
        "final_metrics": {
            "refusal_rate": refusal,
            "kl_divergence": kl,
            "search_refusal_rate": opt_result.refusal_rate,
            "search_kl_divergence": opt_result.kl_divergence,
            "search_ouroboros_score": opt_result.ouroboros_score,
        },
        "ouroboros_compensation": {
            "enabled": len(compensations) > 0,
            "n_passes": len(compensations),
            "passes": compensations,
        },
        "causal_mediation": {},
    }

    # v0.2.0: Add causal mediation data
    if causal_report:
        meta["causal_mediation"] = {
            "n_components": len(causal_report.components),
            "top_attn_layers": causal_report.top_attn_layers[:5],
            "top_mlp_layers": causal_report.top_mlp_layers[:5],
            "causal_layer_mask": causal_report.causal_layer_mask,
        }

    with open(os.path.join(path, "blasphemous_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
