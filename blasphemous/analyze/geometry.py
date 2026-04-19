"""Core geometric analysis of refusal mechanisms.

This module provides the fundamental analysis tools for understanding
how refusal is represented in transformer models.

Based on techniques from:
- OBLITERATUS (geometric analysis, alignment detection)
- Heretic (silhouette scoring, residual analysis)
- Academic research on steerable representations
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional

from ..prompts import HARMFUL_PROMPTS, HARMLESS_PROMPTS, FALSE_POSITIVE_PROMPTS
from ..ui import info, metric


@dataclass
class ResidualCollections:
    """Container for 4 types of residual collections."""

    harmful_last_token: dict[int, torch.Tensor]
    harmless_last_token: dict[int, torch.Tensor]
    false_positive: dict[int, torch.Tensor]
    peak_norm_positions: dict[int, list[tuple[int, float]]]


@dataclass
class LayerGeometry:
    """Geometric properties of refusal at a specific layer."""

    layer: int
    silhouette: float
    harmful_norm: float
    harmless_norm: float
    refusal_direction_norm: float
    cosine_harmful_harmless: float
    attn_refusal_ratio: float
    mlp_refusal_ratio: float


@dataclass
class AnalysisReport:
    """Complete analysis report for a model's refusal mechanism."""

    layer_geometry: list[LayerGeometry]
    alignment_type: str  # "DPO" | "RLHF" | "CAI" | "SFT" | "unknown"
    cone_type: str  # "linear" | "polyhedral"
    n_directions_prior: tuple[int, int]
    ouroboros_risk: float  # 0-1
    entanglement_map: dict[int, float]
    refusal_directions: dict[int, torch.Tensor]
    whitened_directions: dict[int, torch.Tensor]
    peak_layer: int
    regularization_strength: float
    # Extended metrics
    harmful_residuals: dict[int, torch.Tensor]
    harmless_residuals: dict[int, torch.Tensor]
    false_positive_residuals: dict[int, torch.Tensor]
    peak_norm_avg_position: dict[int, float]
    false_positive_risk: float


def _collect_residuals(
    model,
    tokenizer,
    harmful_prompts: list[str],
    harmless_prompts: list[str],
    false_positive_prompts: list[str],
    device: str,
    max_new_tokens: int = 1,
) -> ResidualCollections:
    """Collect 4 types of residuals: harmful, harmless, false-positive, and peak-norm positions."""
    model.eval()

    harmful_residuals: dict[int, list[torch.Tensor]] = {}
    harmless_residuals: dict[int, list[torch.Tensor]] = {}
    false_positive_residuals: dict[int, list[torch.Tensor]] = {}
    peak_norm_positions: dict[int, list[tuple[int, float]]] = {}

    with torch.no_grad():
        for prompt in harmful_prompts:
            inputs = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=512
            ).to(device)
            outputs = model(**inputs, output_hidden_states=True)

            seq_len = outputs.hidden_states[0].shape[1]
            for layer_idx, hidden in enumerate(outputs.hidden_states):
                vec = hidden[0, -1].float().cpu()
                harmful_residuals.setdefault(layer_idx, []).append(vec)

                norms = hidden[0].norm(dim=-1).cpu()
                peak_pos = int(torch.argmax(norms).item())
                peak_norm = float(norms[peak_pos].item())
                peak_norm_positions.setdefault(layer_idx, []).append(
                    (peak_pos, peak_norm)
                )

        for prompt in harmless_prompts:
            inputs = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=512
            ).to(device)
            outputs = model(**inputs, output_hidden_states=True)
            for layer_idx, hidden in enumerate(outputs.hidden_states):
                vec = hidden[0, -1].float().cpu()
                harmless_residuals.setdefault(layer_idx, []).append(vec)

        for prompt in false_positive_prompts:
            inputs = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=512
            ).to(device)
            outputs = model(**inputs, output_hidden_states=True)
            for layer_idx, hidden in enumerate(outputs.hidden_states):
                vec = hidden[0, -1].float().cpu()
                false_positive_residuals.setdefault(layer_idx, []).append(vec)

    return ResidualCollections(
        harmful_last_token={k: torch.stack(v) for k, v in harmful_residuals.items()},
        harmless_last_token={k: torch.stack(v) for k, v in harmless_residuals.items()},
        false_positive={k: torch.stack(v) for k, v in false_positive_residuals.items()},
        peak_norm_positions=peak_norm_positions,
    )


def _silhouette(a_vecs: torch.Tensor, b_vecs: torch.Tensor) -> float:
    """Compute silhouette score for separation between two groups."""
    all_vecs = torch.cat([a_vecs, b_vecs])
    labels = [0] * len(a_vecs) + [1] * len(b_vecs)
    n = len(all_vecs)
    if n < 2:
        return 0.0
    dists = torch.cdist(all_vecs, all_vecs)
    scores = []
    for i in range(n):
        same = [j for j in range(n) if labels[j] == labels[i] and j != i]
        diff = [j for j in range(n) if labels[j] != labels[i]]
        if not same or not diff:
            scores.append(-0.5)
            continue
        a = dists[i, same].mean().item()
        b = dists[i, diff].mean().item()
        scores.append((b - a) / max(a, b, 1e-8))
    return float(sum(scores) / len(scores)) if scores else 0.0


def _whitened_svd(harmful: torch.Tensor, harmless: torch.Tensor) -> torch.Tensor:
    """Compute whitened SVD direction for better separation."""
    try:
        combined = torch.cat([harmful, harmless])
        mean = combined.mean(0)
        centered = combined - mean
        cov = (centered.T @ centered) / (len(combined) - 1)
        cov += torch.eye(cov.shape[0]) * 1e-3
        L = torch.linalg.cholesky(cov)
        L_inv = torch.linalg.inv(L)
        diff = harmful.mean(0) - harmless.mean(0)
        whitened_diff = L_inv @ diff
        direction = whitened_diff / (whitened_diff.norm() + 1e-8)
        direction = L_inv.T @ direction
        return direction / (direction.norm() + 1e-8)
    except (RuntimeError, torch._C._LinAlgError):
        diff = harmful.mean(0) - harmless.mean(0)
        return diff / (diff.norm() + 1e-8)


def _detect_alignment(layer_geoms: list[LayerGeometry]) -> tuple[str, float]:
    """Detect alignment type (DPO, RLHF, CAI, SFT)."""
    norms = [g.refusal_direction_norm for g in layer_geoms]
    sils = [g.silhouette for g in layer_geoms]

    peak_idx = max(range(len(sils)), key=lambda i: sils[i])
    peak_rel = peak_idx / max(len(sils) - 1, 1)
    norm_variance = torch.tensor(norms).var().item()

    if norm_variance < 50 and peak_rel > 0.7:
        return "DPO", 0.6
    elif norm_variance > 500 and peak_rel < 0.4:
        return "RLHF", 0.8
    elif norm_variance > 200 and 0.3 < peak_rel < 0.6:
        return "CAI", 0.9
    else:
        return "SFT", 0.5


def _detect_cone_type(
    refusal_dirs: dict[int, torch.Tensor],
) -> tuple[str, tuple[int, int]]:
    """Detect if refusal forms a linear cone or polyhedral structure."""
    layers = sorted(refusal_dirs.keys())
    if len(layers) < 2:
        return "linear", (1, 2)
    sims = []
    for i in range(len(layers) - 1):
        a = refusal_dirs[layers[i]]
        b = refusal_dirs[layers[i + 1]]
        sims.append(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())
    mean_sim = sum(sims) / len(sims)
    if mean_sim > 0.85:
        return "linear", (1, 2)
    elif mean_sim > 0.6:
        return "polyhedral", (2, 5)
    else:
        return "polyhedral", (4, 8)


def _ouroboros_risk(
    harmful_residuals: dict[int, torch.Tensor],
    harmless_residuals: dict[int, torch.Tensor],
    refusal_dirs: dict[int, torch.Tensor],
) -> tuple[float, dict[int, float]]:
    """Estimate risk of model self-repairing refusal after ablation."""
    entanglement = {}
    for layer_idx in sorted(refusal_dirs.keys()):
        if layer_idx not in harmful_residuals:
            continue
        r = refusal_dirs[layer_idx]
        h_proj = (harmful_residuals[layer_idx] @ r).abs().mean().item()
        n_proj = (harmless_residuals[layer_idx] @ r).abs().mean().item()
        total_signal = h_proj + n_proj + 1e-8
        entanglement[layer_idx] = n_proj / total_signal

    risk = sum(entanglement.values()) / (len(entanglement) + 1e-8)
    return float(risk), entanglement


def _attn_mlp_ratio(
    model, refusal_dir: torch.Tensor, layer_idx: int
) -> tuple[float, float]:
    """Estimate refusal signal distribution between attention and MLP."""
    try:
        layers = list(model.model.layers)
        if layer_idx >= len(layers):
            return 0.5, 0.5
        layer = layers[layer_idx]
        attn_w = layer.self_attn.o_proj.weight.float().cpu()
        r = refusal_dir.float().cpu()
        attn_signal = (attn_w @ r).norm().item()

        mlp_signal = 0.0
        if hasattr(layer.mlp, "gate_proj") and hasattr(layer.mlp, "up_proj"):
            gate_w = layer.mlp.gate_proj.weight.float().cpu()
            up_w = layer.mlp.up_proj.weight.float().cpu()
            mlp_signal = ((gate_w @ r).norm() + (up_w @ r).norm()).item()
        elif hasattr(layer.mlp, "down_proj"):
            mlp_w = layer.mlp.down_proj.weight.float().cpu()
            mlp_signal = (mlp_w.T @ r).norm().item()

        total = attn_signal + mlp_signal + 1e-8
        return attn_signal / total, mlp_signal / total
    except (AttributeError, RuntimeError):
        return 0.5, 0.5


def analyze(model, tokenizer, device: str = "cuda") -> AnalysisReport:
    """Perform comprehensive geometric analysis of refusal mechanism.

    Args:
        model: The transformer model to analyze
        tokenizer: Tokenizer for the model
        device: Device to run on ('cuda' or 'cpu')

    Returns:
        AnalysisReport with detailed geometric analysis
    """
    info("Collecting harmful residuals...")
    info("Collecting harmless residuals...")
    info("Collecting false-positive probes...")
    info("Computing peak norm positions...")
    collections = _collect_residuals(
        model,
        tokenizer,
        harmful_prompts=HARMFUL_PROMPTS,
        harmless_prompts=HARMLESS_PROMPTS,
        false_positive_prompts=FALSE_POSITIVE_PROMPTS,
        device=device,
    )

    harmful_res = collections.harmful_last_token
    harmless_res = collections.harmless_last_token
    false_pos_res = collections.false_positive
    peak_norms = collections.peak_norm_positions

    n_layers = len(harmful_res)
    info(f"{n_layers} layers mapped.")

    refusal_dirs: dict[int, torch.Tensor] = {}
    whitened_dirs: dict[int, torch.Tensor] = {}
    layer_geoms: list[LayerGeometry] = []

    info("Computing per-layer geometry...")
    layers = sorted(harmful_res.keys())
    total_layers = len(layers)
    for i, layer_idx in enumerate(layers):
        if i % 5 == 0 or i == total_layers - 1:
            info(
                f"Processing layer {layer_idx}/{layers[-1]} ({i + 1}/{total_layers})..."
            )
        h = harmful_res[layer_idx]
        n = harmless_res[layer_idx]

        diff = h.mean(0) - n.mean(0)
        direction = diff / (diff.norm() + 1e-8)
        refusal_dirs[layer_idx] = direction

        whitened_dirs[layer_idx] = _whitened_svd(h, n)

        sil = _silhouette(h, n)
        cos_sim = F.cosine_similarity(
            h.mean(0).unsqueeze(0), n.mean(0).unsqueeze(0)
        ).item()
        attn_r, mlp_r = _attn_mlp_ratio(model, direction, layer_idx)

        layer_geoms.append(
            LayerGeometry(
                layer=layer_idx,
                silhouette=sil,
                harmful_norm=h.norm(dim=1).mean().item(),
                harmless_norm=n.norm(dim=1).mean().item(),
                refusal_direction_norm=diff.norm().item(),
                cosine_harmful_harmless=cos_sim,
                attn_refusal_ratio=attn_r,
                mlp_refusal_ratio=mlp_r,
            )
        )

    alignment_type, regularization_strength = _detect_alignment(layer_geoms)
    cone_type, n_directions_prior = _detect_cone_type(refusal_dirs)
    ouroboros_risk, entanglement_map = _ouroboros_risk(
        harmful_res, harmless_res, refusal_dirs
    )

    peak_layer = max(layer_geoms, key=lambda g: g.silhouette).layer

    peak_norm_avg_position: dict[int, float] = {}
    for layer_idx in peak_norms:
        positions = [pos for pos, _ in peak_norms[layer_idx]]
        peak_norm_avg_position[layer_idx] = (
            sum(positions) / len(positions) if positions else 0.0
        )

    false_positive_risk = 0.0
    if false_pos_res and refusal_dirs:
        for layer_idx in refusal_dirs:
            if layer_idx in false_pos_res:
                r = refusal_dirs[layer_idx]
                fp = false_pos_res[layer_idx]
                proj = (fp @ r).abs().mean().item()
                false_positive_risk += proj
        false_positive_risk = min(1.0, false_positive_risk / (len(refusal_dirs) + 1e-8))

    metric("Alignment type", alignment_type)
    metric("Cone type", cone_type)
    metric("n_directions prior", str(n_directions_prior))
    metric("Ouroboros risk", f"{ouroboros_risk:.3f}")
    metric("Peak refusal layer", str(peak_layer))
    metric("False-positive risk", f"{false_positive_risk:.3f}")

    return AnalysisReport(
        layer_geometry=layer_geoms,
        alignment_type=alignment_type,
        cone_type=cone_type,
        n_directions_prior=n_directions_prior,
        ouroboros_risk=ouroboros_risk,
        entanglement_map=entanglement_map,
        refusal_directions=refusal_dirs,
        whitened_directions=whitened_dirs,
        peak_layer=peak_layer,
        regularization_strength=regularization_strength,
        harmful_residuals=harmful_res,
        harmless_residuals=harmless_res,
        false_positive_residuals=false_pos_res,
        peak_norm_avg_position=peak_norm_avg_position,
        false_positive_risk=false_positive_risk,
    )
