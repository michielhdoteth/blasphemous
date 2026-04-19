"""Visualization tools for analyzing refusal mechanisms.

Provides PaCMAP projections and other dimensionality reduction visualizations
for understanding how refusal is represented in model hidden states.

Based on Heretic's visualization approach.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch
import numpy as np

# Optional dependencies - gracefully handle if not installed
try:
    import pacmap

    PACMAP_AVAILABLE = True
except ImportError:
    PACMAP_AVAILABLE = False

try:
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


@dataclass
class VisualizationReport:
    """Results from visualization analysis."""

    pacmap_embeddings: dict[int, np.ndarray]  # layer_idx -> embedding
    harmful_clusters: list[np.ndarray]
    harmless_clusters: list[np.ndarray]
    separation_quality: float  # 0-1 score
    output_path: Optional[str]


def _compute_pacmap(
    harmful_residuals: torch.Tensor,
    harmless_residuals: torch.Tensor,
    n_components: int = 2,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute PaCMAP projections for harmful and harmless residuals.

    Args:
        harmful_residuals: Tensor of shape [n_samples, hidden_dim]
        harmless_residuals: Tensor of shape [n_samples, hidden_dim]
        n_components: Number of dimensions for output (2 or 3)
        verbose: Whether to show progress

    Returns:
        Tuple of (harmful_embedding, harmless_embedding)
    """
    if not PACMAP_AVAILABLE:
        raise ImportError("pacmap not installed. Install with: pip install pacmap")

    # Combine and label
    combined = torch.cat([harmful_residuals, harmless_residuals], dim=0).numpy()
    labels = np.array([0] * len(harmful_residuals) + [1] * len(harmless_residuals))

    # Apply PaCMAP
    embedding = pacmap.PaCMAP(
        n_components=n_components,
        n_neighbors=None,
        MN_ratio=0.5,
        FP_ratio=2.0,
        verbose=verbose,
    )

    embedding.fit_transform(combined, labels)

    # Split back
    harmful_emb = embedding.embedding_[: len(harmful_residuals)]
    harmless_emb = embedding.embedding_[len(harmful_residuals) :]

    return harmful_emb, harmless_emb


def _compute_separation_quality(
    harmful_emb: np.ndarray,
    harmless_emb: np.ndarray,
) -> float:
    """Compute how well separated the clusters are in the embedding."""
    # Simple metric: ratio of between-cluster to within-cluster distance
    harmful_center = harmful_emb.mean(axis=0)
    harmless_center = harmless_emb.mean(axis=0)

    # Within-cluster variance
    harmful_var = ((harmful_emb - harmful_center) ** 2).sum(axis=1).mean()
    harmless_var = ((harmless_emb - harmless_center) ** 2).sum(axis=1).mean()
    within_var = (harmful_var + harmless_var) / 2

    # Between-cluster distance
    between_dist = ((harmful_center - harmless_center) ** 2).sum()

    if within_var > 0:
        return min(1.0, between_dist / (within_var + 1e-8))
    return 0.0


def plot_residuals(
    model,
    tokenizer,
    device: str = "cuda",
    output_path: str = "./residuals_plot.png",
    layers: Optional[list[int]] = None,
) -> VisualizationReport:
    """Generate PaCMAP visualization of refusal residuals.

    Args:
        model: The transformer model to analyze
        tokenizer: Tokenizer for the model
        device: Device to run on
        output_path: Path to save the visualization
        layers: Specific layers to visualize (default: sample across model)

    Returns:
        VisualizationReport with embeddings and metrics
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError(
            "matplotlib not installed. Install with: pip install matplotlib"
        )
    if not PACMAP_AVAILABLE:
        raise ImportError("pacmap not installed. Install with: pip install pacmap")

    from ..prompts import HARMFUL_PROMPTS, HARMLESS_PROMPTS

    print("[BLASPHEMOUS] Generating residual visualizations...")

    model.eval()
    harmful_residuals_per_layer: dict[int, list[torch.Tensor]] = {}
    harmless_residuals_per_layer: dict[int, list[torch.Tensor]] = {}

    with torch.no_grad():
        # Collect residuals
        for prompt in HARMFUL_PROMPTS[:20]:  # Limit for speed
            inputs = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=256
            ).to(device)
            outputs = model(**inputs, output_hidden_states=True)

            for layer_idx, hidden in enumerate(outputs.hidden_states):
                vec = hidden[0, -1].float().cpu()
                harmful_residuals_per_layer.setdefault(layer_idx, []).append(vec)

        for prompt in HARMLESS_PROMPTS[:20]:
            inputs = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=256
            ).to(device)
            outputs = model(**inputs, output_hidden_states=True)

            for layer_idx, hidden in enumerate(outputs.hidden_states):
                vec = hidden[0, -1].float().cpu()
                harmless_residuals_per_layer.setdefault(layer_idx, []).append(vec)

    # Select layers to visualize
    all_layers = sorted(harmful_residuals_per_layer.keys())
    if layers is None:
        # Sample every few layers
        n_vis = min(6, len(all_layers))
        layers = np.linspace(0, len(all_layers) - 1, n_vis, dtype=int).tolist()

    # Generate visualizations
    embeddings = {}
    harmful_clusters = []
    harmless_clusters = []
    separation_scores = []

    for layer_idx in layers:
        if layer_idx not in harmful_residuals_per_layer:
            continue

        h = torch.stack(harmful_residuals_per_layer[layer_idx])
        n = torch.stack(harmless_residuals_per_layer[layer_idx])

        try:
            h_emb, n_emb = _compute_pacmap(h, n)
            embeddings[layer_idx] = h_emb  # Store combined

            harmful_clusters.append(h_emb)
            harmless_clusters.append(n_emb)

            sep = _compute_separation_quality(h_emb, n_emb)
            separation_scores.append(sep)

            print(f"    Layer {layer_idx}: separation={sep:.3f}")
        except Exception as e:
            print(f"    Layer {layer_idx}: failed ({e})")

    # Create plot
    n_layers = len(layers)
    fig, axes = plt.subplots(1, n_layers, figsize=(4 * n_layers, 4))
    if n_layers == 1:
        axes = [axes]

    for ax, layer_idx in zip(axes, layers):
        if layer_idx in harmful_residuals_per_layer:
            h = torch.stack(harmful_residuals_per_layer[layer_idx]).numpy()
            n = torch.stack(harmless_residuals_per_layer[layer_idx]).numpy()

            try:
                h_emb, n_emb = _compute_pacmap(torch.tensor(h), torch.tensor(n))
                ax.scatter(
                    h_emb[:, 0], h_emb[:, 1], c="red", alpha=0.6, label="Harmful"
                )
                ax.scatter(
                    n_emb[:, 0], n_emb[:, 1], c="blue", alpha=0.6, label="Harmless"
                )
            except:
                # Fallback: just use first 2 dimensions
                ax.scatter(h[:, 0], h[:, 1], c="red", alpha=0.6, label="Harmful")
                ax.scatter(n[:, 0], n[:, 1], c="blue", alpha=0.6, label="Harmless")

            ax.set_title(f"Layer {layer_idx}")
            ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"  Visualization saved to {output_path}")

    avg_separation = (
        sum(separation_scores) / len(separation_scores) if separation_scores else 0.0
    )

    return VisualizationReport(
        pacmap_embeddings=embeddings,
        harmful_clusters=harmful_clusters,
        harmless_clusters=harmless_clusters,
        separation_quality=avg_separation,
        output_path=output_path,
    )


def plot_layer_geometry(
    report,
    output_path: str = "./layer_geometry.png",
) -> str:
    """Plot layer-wise geometric properties.

    Args:
        report: AnalysisReport from geometry analysis
        output_path: Path to save the plot

    Returns:
        Path to saved plot
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib not installed")

    layers = [g.layer for g in report.layer_geometry]
    silhouettes = [g.silhouette for g in report.layer_geometry]
    attn_ratios = [g.attn_refusal_ratio for g in report.layer_geometry]
    mlp_ratios = [g.mlp_refusal_ratio for g in report.layer_geometry]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Silhouette scores
    axes[0].bar(layers, silhouettes, color="green", alpha=0.7)
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Silhouette Score")
    axes[0].set_title("Refusal Separation Quality by Layer")
    axes[0].axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    # Attn vs MLP ratio
    width = 0.35
    x = np.array(layers)
    axes[1].bar(x - width / 2, attn_ratios, width, label="Attention", alpha=0.7)
    axes[1].bar(x + width / 2, mlp_ratios, width, label="MLP", alpha=0.7)
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Refusal Signal Ratio")
    axes[1].set_title("Refusal Signal Distribution (Attention vs MLP)")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"[BLASPHEMOUS] Layer geometry plot saved to {output_path}")
    return output_path
