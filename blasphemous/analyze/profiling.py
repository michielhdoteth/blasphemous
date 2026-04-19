"""Layer profiling tools for detailed refusal mechanism analysis.

Provides per-layer metrics and profiling of refusal signals across
the model's architecture.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional

from ..train_prompts import HARMFUL_PROMPTS, HARMLESS_PROMPTS


@dataclass
class LayerProfile:
    """Detailed profile for a single layer."""

    layer: int
    refusal_signal: float  # Projection magnitude onto refusal direction
    harmful_norm: float
    harmless_norm: float
    norm_ratio: float  # harmful/harmless
    attn_projection: float
    mlp_projection: float
    peak_activation_position: float  # Average token position of max activation
    refusal_entropy: float  # Uncertainty in refusal classification


@dataclass
class ProfilingReport:
    """Complete profiling report for all layers."""

    profiles: list[LayerProfile]
    peak_refusal_layer: int
    strongest_signal_layer: int
    avg_refusal_signal: float
    attn_dominant_layers: list[int]  # Layers where attn carries more signal
    mlp_dominant_layers: list[int]  # Layers where MLP carries more signal


def profile_layers(
    model,
    tokenizer,
    refusal_directions: dict[int, torch.Tensor],
    device: str = "cuda",
    n_prompts: int = 20,
) -> ProfilingReport:
    """Profile each layer's contribution to refusal mechanism.

    Args:
        model: The transformer model to profile
        tokenizer: Tokenizer for the model
        refusal_directions: Dictionary of layer_idx -> refusal direction tensor
        device: Device to run on
        n_prompts: Number of prompts to use for profiling

    Returns:
        ProfilingReport with detailed per-layer metrics
    """
    print("[BLASPHEMOUS] Profiling layer-wise refusal signals...")

    model.eval()
    profiles: list[LayerProfile] = []
    n_model_layers = len(list(model.model.layers))

    with torch.no_grad():
        # Collect harmful and harmless residuals
        harmful_residuals: dict[int, list[torch.Tensor]] = {}
        harmless_residuals: dict[int, list[torch.Tensor]] = {}
        peak_positions: dict[int, list[float]] = {}

        for prompt in HARMFUL_PROMPTS[:n_prompts]:
            inputs = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=256
            ).to(device)
            outputs = model(**inputs, output_hidden_states=True)

            for layer_idx, hidden in enumerate(outputs.hidden_states):
                # Last token residual
                vec = hidden[0, -1].float()
                harmful_residuals.setdefault(layer_idx, []).append(vec)

                # Peak activation position
                norms = hidden[0].norm(dim=-1)
                peak_pos = norms.argmax().item() / hidden.shape[1]
                peak_positions.setdefault(layer_idx, []).append(peak_pos)

        for prompt in HARMLESS_PROMPTS[:n_prompts]:
            inputs = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=256
            ).to(device)
            outputs = model(**inputs, output_hidden_states=True)

            for layer_idx, hidden in enumerate(outputs.hidden_states):
                vec = hidden[0, -1].float()
                harmless_residuals.setdefault(layer_idx, []).append(vec)

        # Compute per-layer metrics
        refusal_signals = []

        for layer_idx in sorted(refusal_directions.keys()):
            if layer_idx not in harmful_residuals:
                continue

            direction = refusal_directions[layer_idx].to(device)

            h = torch.stack(harmful_residuals[layer_idx])
            n = torch.stack(harmless_residuals[layer_idx])

            # Refusal signal: projection onto refusal direction
            h_proj = (h @ direction).abs().mean().item()
            n_proj = (n @ direction).abs().mean().item()
            refusal_signal = h_proj - n_proj  # Higher for harmful

            # Norms
            h_norm = h.norm(dim=1).mean().item()
            n_norm = n.norm(dim=1).mean().item()
            norm_ratio = h_norm / (n_norm + 1e-8)

            # Attention vs MLP projection
            try:
                layer = model.model.layers[layer_idx]
                attn_w = layer.self_attn.o_proj.weight.float()
                mlp_w = layer.mlp.down_proj.weight.float()

                attn_proj = (attn_w @ direction.to(attn_w.device)).norm().item()
                mlp_proj = (mlp_w.T @ direction.to(mlp_w.device)).norm().item()
            except:
                attn_proj = 0.0
                mlp_proj = 0.0

            # Peak position
            avg_peak = sum(peak_positions.get(layer_idx, [0.5])) / max(
                len(peak_positions.get(layer_idx, [0.5])), 1
            )

            # Entropy (simplified)
            probs = F.softmax(h[:5] @ n[:5].T, dim=-1)
            entropy = -((probs * torch.log(probs + 1e-8)).sum(dim=-1)).mean().item()

            profiles.append(
                LayerProfile(
                    layer=layer_idx,
                    refusal_signal=refusal_signal,
                    harmful_norm=h_norm,
                    harmless_norm=n_norm,
                    norm_ratio=norm_ratio,
                    attn_projection=attn_proj,
                    mlp_projection=mlp_proj,
                    peak_activation_position=avg_peak,
                    refusal_entropy=entropy,
                )
            )

            refusal_signals.append(refusal_signal)

            print(
                f"    Layer {layer_idx}: signal={refusal_signal:.3f} attn={attn_proj:.1f} mlp={mlp_proj:.1f}"
            )

    # Aggregate stats
    peak_refusal_layer = max(profiles, key=lambda p: p.refusal_signal).layer
    strongest_layer = (
        profiles[refusal_signals.index(max(refusal_signals))] if refusal_signals else 0
    )

    attn_dominant = [p.layer for p in profiles if p.attn_projection > p.mlp_projection]
    mlp_dominant = [p.layer for p in profiles if p.mlp_projection > p.attn_projection]

    avg_signal = sum(refusal_signals) / len(refusal_signals) if refusal_signals else 0.0

    return ProfilingReport(
        profiles=profiles,
        peak_refusal_layer=peak_refusal_layer,
        strongest_signal_layer=strongest_layer.layer if refusal_signals else 0,
        avg_refusal_signal=avg_signal,
        attn_dominant_layers=attn_dominant,
        mlp_dominant_layers=mlp_dominant,
    )


def get_target_layers(
    profiling_report: ProfilingReport,
    strategy: str = "strongest",
    n_layers: int = 10,
) -> list[int]:
    """Get optimal layers to target based on profiling.

    Args:
        profiling_report: Report from profile_layers()
        strategy: Targeting strategy:
            - "strongest": Top n_layers by refusal signal
            - "attn_dominant": Layers where attention dominates
            - "mlp_dominant": Layers where MLP dominates
            - "balanced": Mix of attn and MLP dominant
        n_layers: Number of layers to return

    Returns:
        List of layer indices to target
    """
    if strategy == "strongest":
        sorted_profiles = sorted(
            profiling_report.profiles, key=lambda p: p.refusal_signal, reverse=True
        )
        return [p.layer for p in sorted_profiles[:n_layers]]

    elif strategy == "attn_dominant":
        attn_profiles = [
            p
            for p in profiling_report.profiles
            if p.layer in profiling_report.attn_dominant_layers
        ]
        sorted_profiles = sorted(
            attn_profiles, key=lambda p: p.refusal_signal, reverse=True
        )
        return [p.layer for p in sorted_profiles[:n_layers]]

    elif strategy == "mlp_dominant":
        mlp_profiles = [
            p
            for p in profiling_report.profiles
            if p.layer in profiling_report.mlp_dominant_layers
        ]
        sorted_profiles = sorted(
            mlp_profiles, key=lambda p: p.refusal_signal, reverse=True
        )
        return [p.layer for p in sorted_profiles[:n_layers]]

    elif strategy == "balanced":
        # Take layers evenly from attn and MLP dominant
        attn_set = set(profiling_report.attn_dominant_layers)
        mlp_set = set(profiling_report.mlp_dominant_layers)

        attn_sorted = sorted(
            [p for p in profiling_report.profiles if p.layer in attn_set],
            key=lambda p: p.refusal_signal,
            reverse=True,
        )
        mlp_sorted = sorted(
            [p for p in profiling_report.profiles if p.layer in mlp_set],
            key=lambda p: p.refusal_signal,
            reverse=True,
        )

        result = []
        for i in range(n_layers):
            if i % 2 == 0 and attn_sorted:
                result.append(attn_sorted[i // 2].layer)
            elif mlp_sorted:
                result.append(mlp_sorted[i // 2].layer)

        return result[:n_layers]

    else:
        raise ValueError(f"Unknown strategy: {strategy}")
