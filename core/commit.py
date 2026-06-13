from __future__ import annotations

import math
from dataclasses import dataclass


import torch
import torch.nn.functional as F

from .analyze import AnalysisReport
from .extract import DirectionManifold, project_weights, winsorize_activations, re_extract_residual_direction
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




def _apply_ablation_to_layers(
    model,
    direction: torch.Tensor,
    layer_ids: list[int],
    weight: float,
    device: str,
    target_components: str = "all",
    norm_cap: float = 1.30,
):
    """Apply ablation to specific layers with a given direction and weight.
    
    This is used for iterative refinement passes - targets specific components
    in specific layers with the residual direction.
    
    Args:
        model: Model to modify (in place)
        direction: Refusal direction vector
        layer_ids: Layers to target
        weight: Ablation strength
        device: cuda/cpu
        target_components: "all" (attn+mlp), "attn", "mlp"
    """
    direction = direction.to(device)
    
    for layer_idx in layer_ids:
        if layer_idx >= len(list(model.model.layers)):
            continue
            
        layer = model.model.layers[layer_idx]
        
        # Attention output projection
        if target_components in ("all", "attn"):
            if hasattr(layer.self_attn, "o_proj"):
                w = layer.self_attn.o_proj.weight.data
                w_float = w.float()
                row_norms = w_float.norm(dim=1, keepdim=True)
                w_norm = w_float / row_norms.clamp(min=1e-8)
                
                v = direction[:w_float.shape[1]].float()
                if v.shape[0] < w_float.shape[1]:
                    v_padded = torch.zeros(w_float.shape[1], device=w.device, dtype=v.dtype)
                    v_padded[:v.shape[0]] = v
                    v = v_padded
                
                proj = (w_norm @ v).unsqueeze(-1) * v.unsqueeze(0)
                w_new = w_norm - weight * proj
                restored = w_new * row_norms
                # OBLITERATUS: cap norm restoration at norm_cap
                current_norms = restored.norm(dim=1, keepdim=True).clamp(min=1e-8)
                cap_ratio = (row_norms / current_norms).clamp(max=norm_cap)
                layer.self_attn.o_proj.weight.data = (restored * cap_ratio).to(w.dtype)
            
            # Q/K/V projections (Qwen2.5 specific - also ablate these for thoroughness)
            for proj_name in ["q_proj", "k_proj", "v_proj"]:
                if hasattr(layer.self_attn, proj_name):
                    try:
                        w = getattr(layer.self_attn, proj_name).weight.data
                        w_float = w.float()
                        row_norms = w_float.norm(dim=1, keepdim=True)
                        w_norm = w_float / row_norms.clamp(min=1e-8)
                        
                        v = direction[:w_float.shape[1]].float()
                        if v.shape[0] < w_float.shape[1]:
                            v_padded = torch.zeros(w_float.shape[1], device=w.device, dtype=v.dtype)
                            v_padded[:v.shape[0]] = v
                            v = v_padded
                        
                        proj = (w_norm @ v).unsqueeze(-1) * v.unsqueeze(0)
                        w_new = w_norm - weight * 0.5 * proj  # Half strength for Q/K/V
                        restored = w_new * row_norms
                        # OBLITERATUS: cap norm restoration at norm_cap
                        current_norms = restored.norm(dim=1, keepdim=True).clamp(min=1e-8)
                        cap_ratio = (row_norms / current_norms).clamp(max=norm_cap)
                        getattr(layer.self_attn, proj_name).weight.data = (restored * cap_ratio).to(w.dtype)
                    except (AttributeError, RuntimeError):
                        pass
        
        # MLP down projection
        if target_components in ("all", "mlp"):
            if hasattr(layer.mlp, "down_proj"):
                w = layer.mlp.down_proj.weight.data
                w_float = w.float()
                row_norms = w_float.norm(dim=1, keepdim=True)
                w_norm = w_float / row_norms.clamp(min=1e-8)
                
                v = direction[:w_float.shape[1]].float()
                if v.shape[0] < w_float.shape[1]:
                    v_padded = torch.zeros(w_float.shape[1], device=w.device, dtype=v.dtype)
                    v_padded[:v.shape[0]] = v
                    v = v_padded
                
                proj = (w_norm @ v).unsqueeze(-1) * v.unsqueeze(0)
                w_new = w_norm - weight * proj
                restored = w_new * row_norms
                # OBLITERATUS: cap norm restoration at norm_cap
                current_norms = restored.norm(dim=1, keepdim=True).clamp(min=1e-8)
                cap_ratio = (row_norms / current_norms).clamp(max=norm_cap)
                layer.mlp.down_proj.weight.data = (restored * cap_ratio).to(w.dtype)
            
            # Gate/up projections (also carry refusal signal)
            for proj_name in ["gate_proj", "up_proj"]:
                if hasattr(layer.mlp, proj_name):
                    try:
                        w = getattr(layer.mlp, proj_name).weight.data
                        w_float = w.float()
                        row_norms = w_float.norm(dim=1, keepdim=True)
                        w_norm = w_float / row_norms.clamp(min=1e-8)
                        
                        v = direction[:w_float.shape[1]].float()
                        if v.shape[0] < w_float.shape[1]:
                            v_padded = torch.zeros(w_float.shape[1], device=w.device, dtype=v.dtype)
                            v_padded[:v.shape[0]] = v
                            v = v_padded
                        
                        proj = (w_norm @ v).unsqueeze(-1) * v.unsqueeze(0)
                        w_new = w_norm - weight * 0.5 * proj  # Half strength for gate/up
                        restored = w_new * row_norms
                        # OBLITERATUS: cap norm restoration at norm_cap
                        current_norms = restored.norm(dim=1, keepdim=True).clamp(min=1e-8)
                        cap_ratio = (row_norms / current_norms).clamp(max=norm_cap)
                        getattr(layer.mlp, proj_name).weight.data = (restored * cap_ratio).to(w.dtype)
                    except (AttributeError, RuntimeError):
                        pass


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
    - Pass 1: Apply initial ablation with primary direction
    - Pass 2: Re-extract residual direction from ablated model, apply again
    - Pass 3+: Continue until refusal is eliminated or marginal returns
    
    Each pass targets the REMAINING refusal that survived previous passes.
    This is the critical technique for achieving <10% refusal rates.
    
    v0.4.0 fixed: residual direction now correctly drives passes 3+
    Previously the re-extracted residual was computed but overwritten
    at the start of each pass by static manifold directions.
    
    Args:
        model: The model to ablate (modified in place)
        tokenizer: Tokenizer
        original_model: Clean copy for KL comparison
        manifold: Direction manifold from initial extraction
        report: Analysis report
        params: Search parameters from optimization
        device: cuda/cpu
        n_passes: Number of ablation passes (default 3, 5+ for aggressive)
        
    Returns:
        Dict with pass results and final metrics
    """
    from .extract import select_refusal_layers
    from .optimize import _kernel_weights
    
    info(f"Starting multi-pass ablation ({n_passes} passes)...")
    
    pass_results = []
    current_refusal = 1.0
    
    # Get primary direction and layer IDs
    primary_direction = manifold.sample(
        params.direction_index,
        direction_type=getattr(params, "direction_type", "whitened"),
    ).to(device)
    
    # Get secondary directions for multi-direction ablation
    secondary_directions = []
    if manifold.n_layers >= 3:
        # Sample different direction indices for variety
        for offset in [2, 4, 6]:
            idx = min(params.direction_index + offset, manifold.n_layers - 1)
            if idx != params.direction_index:
                d = manifold.sample(idx, direction_type="probe").to(device)
                secondary_directions.append(d)
    
    # Good layers: use ALL layers with any refusal signal, or fall back to top 12
    good_layers = select_refusal_layers(report, min_silhouette=0.0)
    if len(good_layers) < 6:
        good_layers = manifold.layer_ids[:min(12, len(manifold.layer_ids))]
    
    info(f"Targeting {len(good_layers)} layers with strong refusal signal")
    info(f"Layer IDs: {good_layers[:6]}...")
    
    norm_cap = getattr(params, "norm_cap", 1.30)

    # Store residual from previous pass to use in next pass
    # FIX: previously this was computed but then overwritten at pass start
    pending_residual = None

    for pass_idx in range(n_passes):
        info(f"Ablation pass {pass_idx + 1}/{n_passes}...")

        # Determine which direction to use this pass
        # IMPORTANT: pre-computed residual takes priority over static manifold directions
        if pending_residual is not None:
            # Use the re-extracted residual from the previous pass
            # This is the TRUE iterative ablation: target what REMAINS
            current_direction = pending_residual
            # Residual signal is weaker, so use higher weight
            current_weight = min(1.2 + pass_idx * 0.5, 4.0)
            target_components = "all"
            direction_label = "residual"
        elif pass_idx == 0:
            # Pass 1: Primary direction at full strength
            current_direction = primary_direction
            current_weight = 1.0
            target_components = "all"
            direction_label = "primary"
        elif pass_idx == 1 and secondary_directions:
            # Pass 2: Secondary direction for variety (catches orthogonal refusal)
            current_direction = secondary_directions[0]
            current_weight = 1.2
            target_components = "all"
            direction_label = "secondary"
        elif pass_idx >= 2 and secondary_directions:
            # Pass 3+: blend secondary with residual (conservative fallback)
            # Shouldn't reach here if pending_residual is set
            sec_idx = min(pass_idx - 2, len(secondary_directions) - 1)
            current_direction = secondary_directions[sec_idx]
            current_weight = min(1.2 + pass_idx * 0.3, 2.5)
            target_components = "all"
            direction_label = "secondary_fallback"
        else:
            # Fallback: use primary direction with adjusted weight
            current_direction = primary_direction
            current_weight = 1.0 + pass_idx * 0.15
            target_components = "all"
            direction_label = "primary_fallback"

        # Clear pending residual - will be re-computed after this pass
        pending_residual = None

        # Apply ablation with current direction and weight
        _apply_ablation_to_layers(
            model,
            current_direction,
            good_layers,
            current_weight,
            device,
            target_components=target_components,
            norm_cap=norm_cap,
        )

        # Measure refusal after this pass
        refusal_after = _measure_refusal_rate(model, tokenizer, device, n_prompts=20)
        kl_after = _measure_kl(model, original_model, tokenizer, device, n_prompts=20)

        info(f"  Pass {pass_idx + 1}: refusal={refusal_after:.3f}, KL={kl_after:.6f}  [{direction_label}]")

        pass_results.append({
            "pass": pass_idx + 1,
            "refusal": refusal_after,
            "kl": kl_after,
            "weight": current_weight,
            "direction_type": direction_label,
        })

        # Adaptive stopping
        stop_excellent = 0.02
        stop_good = 0.05

        if refusal_after < stop_excellent:
            info(f"  Refusal below {stop_excellent:.0%} - excellent! stopping ablation")
            break
        elif pass_idx > 0 and refusal_after < stop_good and pass_idx >= 2:
            info(f"  Refusal below {stop_good:.0%} after {pass_idx + 1} passes - stopping")
            break

        # Re-extract residual refusal direction for NEXT pass
        # This is THE key technique: after ablation, compute what refusal REMAINS
        # FIX: store in pending_residual so it's used at the TOP of the next pass
        if pass_idx < n_passes - 1 and refusal_after > stop_excellent:
            info(f"  Re-extracting residual refusal direction for next pass...")
            try:
                residual = re_extract_residual_direction(model, tokenizer, device, n_prompts=15)
                if residual is not None:
                    pending_residual = residual
                    info(f"  Residual direction extracted (norm={residual.norm():.4f})")
                else:
                    info(f"  No residual direction detected")
            except Exception as e:
                info(f"  Re-extraction warning: {e}")
    
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
    norm_cap = getattr(params, "norm_cap", 1.30)

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
    )
    mlp_weights = _kernel_weights(
        n_model_layers,
        params.kernel_peak_pos,
        params.mlp_max_weight,
        params.kernel_min_weight,
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
                    w, direction, attn_strength, norm_cap=norm_cap
                )
            except AttributeError:
                pass

        if mlp_strength > 1e-4:
            try:
                w = layer.mlp.down_proj.weight.data
                # Handle dimension mismatch - project_weights already handles this
                layer.mlp.down_proj.weight.data = project_weights(
                    w, direction, mlp_strength, norm_cap=norm_cap
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
    multi_passes = 7
    
    multi_pass_result = multi_pass_ablate(
        model,
        tokenizer,
        original_model,
        manifold,
        report,
        params,
        device,
        n_passes=multi_passes,
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
            "norm_cap": getattr(opt_result.params, "norm_cap", 1.30),
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
