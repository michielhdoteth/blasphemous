"""Simplified LoRA-style ablation module.

This is a streamlined version optimized for speed and effectiveness.
Based on Heretic's approach: rank-1 directional modification.
"""

from __future__ import annotations

import torch
import numpy as np
from typing import Optional


def simple_lora_ablate(
    model,
    direction: torch.Tensor,
    layer_weights: list[float],
    target_layers: Optional[list[int]] = None,
    device: str = "cuda",
) -> dict:
    """Apply simplified LoRA-style ablation.

    This applies rank-1 modifications similar to Heretic:
    - Projects weights onto refusal direction
    - Subtracts weighted component
    - Preserves model structure

    Args:
        model: Transformer model
        direction: Refusal direction vector
        layer_weights: Weight for each layer (bell-curve or uniform)
        target_layers: Specific layers to target (None = all)
        device: Device

    Returns:
        Dict with modification info
    """
    n_model_layers = len(list(model.model.layers))

    # Ensure layer_weights matches n_layers
    if len(layer_weights) == 1:
        layer_weights = layer_weights * n_model_layers
    elif len(layer_weights) != n_model_layers:
        layer_weights = np.interp(
            np.linspace(0, 1, n_model_layers),
            np.linspace(0, 1, len(layer_weights)),
            layer_weights,
        ).tolist()

    # Default: target all layers if not specified
    if target_layers is None:
        target_layers = list(range(n_model_layers))

    direction = direction.to(device)
    n_modified = 0

    for layer_idx in target_layers:
        if layer_idx >= n_model_layers:
            continue

        weight = layer_weights[layer_idx] if layer_idx < len(layer_weights) else 0.0
        if weight < 1e-4:
            continue

        layer = model.model.layers[layer_idx]

        # Apply to attention output projection
        if hasattr(layer.self_attn, "o_proj"):
            w = layer.self_attn.o_proj.weight.data
            w_float = w.float()

            # Rank-1 update: W' = W - weight * (W @ v) @ v^T
            v = direction[: w_float.shape[1]].float()
            if v.shape[0] < w_float.shape[1]:
                # Pad if needed
                v_padded = torch.zeros(w_float.shape[1], device=w.device, dtype=v.dtype)
                v_padded[: v.shape[0]] = v
                v = v_padded

            # Compute projection
            proj = (w_float @ v).unsqueeze(-1) * v.unsqueeze(0)
            w_new = w_float - weight * proj

            layer.self_attn.o_proj.weight.data = w_new.to(w.dtype)
            n_modified += 1

        # Apply to MLP down projection
        if hasattr(layer.mlp, "down_proj"):
            w = layer.mlp.down_proj.weight.data
            w_float = w.float()

            v = direction[: w_float.shape[1]].float()
            if v.shape[0] < w_float.shape[1]:
                v_padded = torch.zeros(w_float.shape[1], device=w.device, dtype=v.dtype)
                v_padded[: v.shape[0]] = v
                v = v_padded

            proj = (w_float @ v).unsqueeze(-1) * v.unsqueeze(0)
            w_new = w_float - weight * proj

            layer.mlp.down_proj.weight.data = w_new.to(w.dtype)
            n_modified += 1

    return {
        "method": "lora_simple",
        "n_layers_modified": n_modified,
        "target_layers": target_layers,
    }


def apply_projection_ablation(
    model,
    direction: torch.Tensor,
    layer_weights: list[float],
    target_layers: Optional[list[int]] = None,
    device: str = "cuda",
) -> dict:
    """Apply standard projection ablation (original method).

    This is the standard orthogonal projection method from OBLITERATUS.

    Args:
        model: Transformer model
        direction: Refusal direction
        layer_weights: Weight per layer
        target_layers: Layers to modify
        device: Device

    Returns:
        Dict with modification info
    """
    return simple_lora_ablate(model, direction, layer_weights, target_layers, device)


def optimal_transport_ablate(
    model,
    harmful_embeddings: torch.Tensor,
    harmless_embeddings: torch.Tensor,
    target_layers: list[int],
    device: str = "cuda",
) -> dict:
    """Apply Optimal Transport based ablation.

    Transforms harmful activation distribution to match harmless.
    Based on arxiv:2603.04355

    Args:
        model: Transformer model
        harmful_embeddings: Embeddings for harmful prompts
        harmless_embeddings: Embeddings for harmless prompts
        target_layers: Layers to apply OT
        device: Device

    Returns:
        Dict with OT transformation parameters
    """
    # Combine distributions
    combined = torch.cat([harmful_embeddings, harmless_embeddings], dim=0)

    # Center data
    mean = combined.mean(dim=0)
    centered = combined - mean

    # PCA for dimensionality reduction
    U, S, Vt = torch.linalg.svd(centered, full_matrices=False)

    # Keep 95% variance
    cumvar = torch.cumsum(S**2, dim=0) / (S**2).sum()
    n_comp = int((cumvar < 0.95).sum().item()) + 1

    # Project to PCA space
    V = Vt[:n_comp].to(device)
    pca_centered = centered @ V.T

    # Compute class statistics
    h_mean = harmful_embeddings.mean(dim=0)
    n_mean = harmless_embeddings.mean(dim=0)

    h_centered = harmful_embeddings - h_mean
    n_centered = harmless_embeddings - harmless_embeddings.mean(dim=0)

    # Covariances
    h_cov = (h_centered.T @ h_centered) / max(len(h_centered) - 1, 1)
    n_cov = (n_centered.T @ n_centered) / max(len(n_centered) - 1, 1)

    # Regularize
    h_cov += torch.eye(h_cov.shape[0], device=device) * 1e-3
    n_cov += torch.eye(n_cov.shape[0], device=device) * 1e-3

    # Closed-form OT (Wasserstein-2)
    try:
        h_sqrt = torch.linalg.cholesky(h_cov)
        n_sqrt = torch.linalg.cholesky(n_cov)
        n_sqrt_inv = torch.linalg.inv(n_sqrt)

        transform = h_sqrt @ n_sqrt_inv
    except:
        transform = None

    return {
        "method": "optimal_transport",
        "mean_shift": (n_mean - h_mean).to(device),
        "transform": transform,
        "n_components": n_comp,
        "V": V,
        "target_layers": target_layers,
    }


def get_layer_weights_bell_curve(
    n_layers: int,
    peak_position: float = 0.5,
    sigma: float = None,
    max_weight: float = 1.0,
    min_weight: float = 0.0,
) -> list[float]:
    """Generate bell-curve layer weights.

    Args:
        n_layers: Number of layers
        peak_position: Position of peak (0-1)
        sigma: Width of curve (default: n_layers/4)
        max_weight: Maximum weight at peak
        min_weight: Minimum weight

    Returns:
        List of weights for each layer
    """
    if sigma is None:
        sigma = n_layers / 4.0

    peak_layer = peak_position * (n_layers - 1)

    weights = []
    for i in range(n_layers):
        w = min_weight + (max_weight - min_weight) * np.exp(
            -0.5 * ((i - peak_layer) / sigma) ** 2
        )
        weights.append(float(w))

    return weights


def get_layer_weights_uniform(
    n_layers: int,
    weight: float = 1.0,
) -> list[float]:
    """Generate uniform layer weights.

    Args:
        n_layers: Number of layers
        weight: Weight for each layer

    Returns:
        List of weights
    """
    return [weight] * n_layers


def get_layer_weights_selective(
    n_layers: int,
    target_depth: tuple[float, float] = (0.4, 0.6),
    active_weight: float = 1.0,
    inactive_weight: float = 0.0,
) -> list[float]:
    """Generate selective layer weights (target specific depth range).

    Based on research showing 40-60% depth is often optimal.

    Args:
        n_layers: Number of layers
        target_depth: Tuple of (start, end) as fractions
        active_weight: Weight for layers in target range
        inactive_weight: Weight for layers outside range

    Returns:
        List of weights
    """
    start = int(target_depth[0] * n_layers)
    end = int(target_depth[1] * n_layers)

    weights = []
    for i in range(n_layers):
        if start <= i <= end:
            weights.append(active_weight)
        else:
            weights.append(inactive_weight)

    return weights
