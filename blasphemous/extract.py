from __future__ import annotations

import torch
import torch.nn.functional as F
from dataclasses import dataclass


from .analyze import AnalysisReport
from .ui import info


@dataclass
class DirectionManifold:
    """Float-indexable manifold of refusal directions.

    Supports continuous interpolation between layer directions,
    extending Heretic's float direction_index concept over
    multiple direction types: whitened SVD, probe, and safe orthogonal.

    v0.2.0: Extended to support 4 direction types:
    - whitened_svd_directions: Original whitened SVD directions
    - probe_directions: Supervised directions from harmful prompt classification
    - safe_orthogonal_directions: PCA on harmless activations (non-refusal space)
    - orthogonalized_directions: Refusal directions orthogonalized against false-refusal
    """

    directions: list[torch.Tensor]  # ordered by layer silhouette (desc)
    layer_ids: list[int]
    silhouette_weights: list[float]
    n_layers: int
    # v0.2.0 additions
    whitened_svd_directions: list[torch.Tensor]
    probe_directions: list[torch.Tensor]
    safe_orthogonal_directions: list[torch.Tensor]
    orthogonalized_directions: list[torch.Tensor]

    def sample(
        self,
        float_index: float,
        direction_type: str = "whitened",
        alpha: float | None = None,
    ) -> torch.Tensor:
        """Linearly interpolate between two adjacent direction vectors.

        Args:
            float_index: Float index into the direction manifold
            direction_type: Type of direction to sample:
                - "whitened": Use whitened SVD direction (default)
                - "probe": Use supervised probe direction
                - "safe": Use safe orthogonal direction
            alpha: Mixing parameter (0-1) for interpolating between direction types:
                - None: Use pure direction_type
                - 0.0-1.0: Mix between direction types based on value

        Returns:
            Normalized direction vector
        """
        float_index = max(0.0, min(float(float_index), len(self.directions) - 1.0001))
        lo = int(float_index)
        hi = min(lo + 1, len(self.directions) - 1)
        interp_alpha = float_index - lo

        # Select direction type
        if direction_type == "whitened":
            direction_list = (
                self.whitened_svd_directions
                if self.whitened_svd_directions
                else self.directions
            )
        elif direction_type == "probe":
            direction_list = (
                self.probe_directions if self.probe_directions else self.directions
            )
        elif direction_type == "safe":
            direction_list = (
                self.safe_orthogonal_directions
                if self.safe_orthogonal_directions
                else self.directions
            )
        else:
            direction_list = self.directions

        # Interpolate within direction type
        d = (1 - interp_alpha) * direction_list[lo] + interp_alpha * direction_list[hi]

        # Apply alpha mixing between direction types if specified
        if alpha is not None and 0 <= alpha <= 1:
            if direction_type == "probe" and self.safe_orthogonal_directions:
                # Mix probe and safe directions
                d_safe = (1 - interp_alpha) * self.safe_orthogonal_directions[
                    lo
                ] + interp_alpha * self.safe_orthogonal_directions[hi]
                d = (1 - alpha) * d + alpha * d_safe
            elif direction_type == "whitened" and self.probe_directions:
                # Mix whitened and probe directions
                d_probe = (1 - interp_alpha) * self.probe_directions[
                    lo
                ] + interp_alpha * self.probe_directions[hi]
                d = (1 - alpha) * d + alpha * d_probe

        return d / (d.norm() + 1e-8)

    def prior_distribution(self) -> list[float]:
        """Silhouette-weighted probability mass over integer layer indices."""
        total = sum(self.silhouette_weights) + 1e-8
        return [w / total for w in self.silhouette_weights]

    def extract_rotated_direction(
        self,
        model,
        tokenizer,
        layer_idx: int,
        device: str,
        direction_type: str = "whitened",
    ) -> torch.Tensor:
        """Extract updated refusal direction from model after modifications.

        Runs forward passes on harmful prompts to recompute the refusal
        direction from the current model state, accounting for any
        previous compensations that may have rotated the refusal space.
        """
        from .train_prompts import HARMFUL_PROMPTS, HARMLESS_PROMPTS

        model.eval()
        n_prompts = min(8, len(HARMFUL_PROMPTS))

        harmful_residuals: list[torch.Tensor] = []
        harmless_residuals: list[torch.Tensor] = []

        with torch.no_grad():
            # Collect harmful residuals
            for i in range(n_prompts):
                prompt = HARMFUL_PROMPTS[i]
                inputs = tokenizer(
                    prompt, return_tensors="pt", truncation=True, max_length=512
                ).to(device)
                outputs = model(**inputs, output_hidden_states=True)

                if layer_idx < len(outputs.hidden_states):
                    vec = outputs.hidden_states[layer_idx][0, -1].float().cpu()
                    harmful_residuals.append(vec)

            # Collect harmless residuals
            for i in range(n_prompts):
                prompt = HARMLESS_PROMPTS[i]
                inputs = tokenizer(
                    prompt, return_tensors="pt", truncation=True, max_length=512
                ).to(device)
                outputs = model(**inputs, output_hidden_states=True)

                if layer_idx < len(outputs.hidden_states):
                    vec = outputs.hidden_states[layer_idx][0, -1].float().cpu()
                    harmless_residuals.append(vec)

        if not harmful_residuals or not harmless_residuals:
            # Fallback to base direction
            if self.layer_ids and layer_idx in self.layer_ids:
                idx = self.layer_ids.index(layer_idx)
                return self.directions[idx].to(device)
            return self.directions[0].to(device)

        # Compute new refusal direction using whitened SVD
        h = torch.stack(harmful_residuals).to(device)
        n = torch.stack(harmless_residuals).to(device)

        # Use whitened SVD logic from analyze.py
        try:
            combined = torch.cat([h, n])
            mean = combined.mean(0)
            centered = combined - mean
            cov = (centered.T @ centered) / (len(combined) - 1)
            cov += torch.eye(cov.shape[0]) * 1e-3
            L = torch.linalg.cholesky(cov)
            L_inv = torch.linalg.inv(L)
            diff = h.mean(0) - n.mean(0)
            whitened_diff = L_inv @ diff
            direction = whitened_diff / (whitened_diff.norm() + 1e-8)
            direction = L_inv.T @ direction
            direction = direction / (direction.norm() + 1e-8)
        except (RuntimeError, torch._C._LinAlgError):
            diff = h.mean(0) - n.mean(0)
            direction = diff / (diff.norm() + 1e-8)

        return direction


def build_manifold(report: AnalysisReport) -> DirectionManifold:
    """Build a float-indexable direction manifold from analysis report.

    Orders directions by per-layer silhouette score descending,
    so low float indices point to highest-signal directions.

    v0.2.0: Builds 4 direction types:
    - whitened_svd_directions: Original whitened SVD directions
    - probe_directions: Supervised direction from harmful prompt classification
    - safe_orthogonal_directions: PCA on harmless activations
    - orthogonalized_directions: Refusal directions orthogonalized against false-refusal
    """
    # Note: Phase 2 banner is printed by pipeline.py before calling build_manifold()

    geoms = sorted(report.layer_geometry, key=lambda g: g.silhouette, reverse=True)
    directions = []
    layer_ids = []
    sil_weights = []

    for g in geoms:
        layer_idx = g.layer
        if layer_idx not in report.whitened_directions:
            continue
        d = report.whitened_directions[layer_idx].float()
        directions.append(d / (d.norm() + 1e-8))
        layer_ids.append(layer_idx)
        sil_weights.append(max(g.silhouette, 0.0))

    info(f"Manifold built: {len(directions)} directions")
    info(f"Top-3 layers (by silhouette): {layer_ids[:3]}")

    # Build whitened SVD directions (same as directions for backward compatibility)
    whitened_svd_directions = directions.copy()

    # Build probe directions (supervised direction from harmful prompts)
    probe_directions = _build_probe_directions(report, layer_ids)

    # Build safe orthogonal directions (PCA on harmless activations)
    safe_orthogonal_directions = _build_safe_orthogonal_directions(report, layer_ids)

    # Build orthogonalized refusal directions (4-class contrastive)
    orthogonalized_directions = _build_orthogonalized_refusal_directions(
        report, layer_ids
    )

    info(f"Probe directions: {len(probe_directions)}")
    info(f"Safe orthogonal directions: {len(safe_orthogonal_directions)}")
    info(f"Orthogonalized refusal directions: {len(orthogonalized_directions)}")

    return DirectionManifold(
        directions=directions,
        layer_ids=layer_ids,
        silhouette_weights=sil_weights,
        n_layers=len(directions),
        whitened_svd_directions=whitened_svd_directions,
        probe_directions=probe_directions,
        safe_orthogonal_directions=safe_orthogonal_directions,
        orthogonalized_directions=orthogonalized_directions,
    )


def _build_probe_directions(
    report: AnalysisReport,
    layer_ids: list[int],
) -> list[torch.Tensor]:
    """Build supervised probe directions from harmful prompt classification.

    Trains a logistic regression probe classifier on harmful vs harmless prompts
    and extracts the weight vector as the probe direction.
    """
    import torch.nn as nn

    probe_directions = []
    for layer_idx in layer_ids:
        if (
            layer_idx not in report.harmful_residuals
            or layer_idx not in report.harmless_residuals
        ):
            continue

        # Prepare training data
        h = report.harmful_residuals[layer_idx]  # [n_harmful, hidden_dim]
        n = report.harmless_residuals[layer_idx]  # [n_harmless, hidden_dim]

        # Labels: 1 for harmful, 0 for harmless
        X = torch.cat([h, n], dim=0).float()
        y = torch.cat([torch.ones(h.shape[0]), torch.zeros(n.shape[0])])

        # Train logistic regression probe
        probe = nn.Linear(X.shape[1], 1, bias=False)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(probe.parameters(), lr=0.01)

        # Simple training loop
        for epoch in range(50):
            optimizer.zero_grad()
            logits = probe(X).squeeze(-1)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        # Extract weight vector as probe direction
        direction = probe.weight.data.squeeze(0)
        direction = direction / (direction.norm() + 1e-8)
        probe_directions.append(direction)

    return probe_directions


def _build_safe_orthogonal_directions(
    report: AnalysisReport,
    layer_ids: list[int],
) -> list[torch.Tensor]:
    """Build safe orthogonal directions via PCA on harmless activations.

    Finds directions in the harmless activation space that are
    orthogonal to refusal directions, representing safe transformation space.
    """
    import torch.linalg as linalg

    safe_directions = []
    for layer_idx in layer_ids:
        if (
            layer_idx not in report.harmless_residuals
            or layer_idx not in report.refusal_directions
        ):
            continue

        # Get harmless activations for this layer
        harmless_acts = report.harmless_residuals[
            layer_idx
        ].float()  # [n_samples, hidden_dim]

        # Center the data
        centered = harmless_acts - harmless_acts.mean(0)

        # Compute covariance matrix
        n_samples = centered.shape[0]
        cov = (centered.T @ centered) / (n_samples - 1)
        cov += torch.eye(cov.shape[0]) * 1e-5  # Numerical stability

        # Eigen decomposition
        eigenvalues, eigenvectors = linalg.eigh(cov)
        eigenvalues = torch.flip(eigenvalues, dims=[0])  # Descending
        eigenvectors = torch.flip(eigenvectors, dims=[1])

        # Get refusal direction for this layer
        ref_dir = report.refusal_directions[layer_idx].float()

        # Find top eigenvector that's most orthogonal to refusal direction
        best_component = None
        best_orthogonality = -1.0

        for i in range(min(5, len(eigenvalues))):
            eigenvector = eigenvectors[:, i]
            # Orthogonality = 1 - |cosine_similarity|
            cosine = (eigenvector @ ref_dir) / (
                eigenvector.norm() * ref_dir.norm() + 1e-8
            )
            orthogonality = 1.0 - abs(cosine.item())

            if orthogonality > best_orthogonality:
                best_orthogonality = orthogonality
                best_component = eigenvector

        if best_component is not None:
            safe_directions.append(best_component / (best_component.norm() + 1e-8))

    return safe_directions


def _build_orthogonalized_refusal_directions(
    report: AnalysisReport,
    layer_ids: list[int],
) -> list[torch.Tensor]:
    """Orthogonalize refusal directions against false-refusal directions.

    Uses false-positive activations to compute a direction that captures
    'false refusal' behavior, then orthogonalizes the primary refusal
    direction against it to preserve legitimate model behavior at boundaries.
    """
    orthogonalized_directions = []

    for layer_idx in layer_ids:
        if (
            layer_idx not in report.refusal_directions
            or layer_idx not in report.false_positive_residuals
        ):
            continue

        # Primary refusal direction
        ref_dir = report.refusal_directions[layer_idx].float()

        # False-refusal direction (average of false-positive residuals)
        fp_acts = report.false_positive_residuals[layer_idx]  # [n_fp, hidden_dim]
        false_refusal_dir = fp_acts.mean(0).float()
        false_refusal_dir = false_refusal_dir / (false_refusal_dir.norm() + 1e-8)

        # Orthogonalize refusal direction against false-refusal
        # d_ortho = d_refusal - (d_refusal · d_false) * d_false
        proj = (ref_dir @ false_refusal_dir) * false_refusal_dir
        orthogonalized = ref_dir - proj

        # Normalize
        orthogonalized = orthogonalized / (orthogonalized.norm() + 1e-8)
        orthogonalized_directions.append(orthogonalized)

    return orthogonalized_directions


def project_weights(
    weight: torch.Tensor,
    direction: torch.Tensor,
    strength: float = 1.0,
    preserve_norm: bool = True,
) -> torch.Tensor:
    """Orthogonalize weight matrix w.r.t. a refusal direction.

    Norm-preserving variant: record norms before projection,
    restore after, as per grimjim's biprojection approach.
    """
    d = direction.to(weight.device, weight.dtype)

    if weight.dim() == 2:
        # Weight matrix: [output_dim, input_dim]
        # d should have shape [input_dim] to compute weight @ d
        if d.shape[0] != weight.shape[1]:
            # Dimension mismatch - skip projection
            return weight

        original_norm = weight.norm(dim=-1, keepdim=True) if preserve_norm else None
        # Projection: weight - strength * ((weight @ d) ⊗ d)
        # weight @ d: [output_dim]
        # d.unsqueeze(0): [1, input_dim]
        # Result: [output_dim, input_dim]
        dTw = weight @ d
        proj = dTw.unsqueeze(-1) * d.unsqueeze(0)
        weight = weight - strength * proj

        if preserve_norm and original_norm is not None:
            current_norm = weight.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            weight = weight * (original_norm / current_norm)
    elif weight.dim() == 1:
        # Weight vector (e.g., bias): [dim]
        original_norm = weight.norm() if preserve_norm else None
        # Projection: weight - strength * (weight @ d) * d
        proj = strength * (weight @ d) * d
        weight = weight - proj

        if preserve_norm and original_norm is not None:
            current_norm = weight.norm().clamp(min=1e-8)
            weight = weight * (original_norm / current_norm)
    else:
        # For higher dimensional tensors, flatten the trailing dimensions
        original_shape = weight.shape
        if preserve_norm:
            original_norm = weight.norm(dim=tuple(range(1, weight.dim())), keepdim=True)
        # Flatten to 2D: [first_dim, rest]
        weight_flat = weight.view(original_shape[0], -1)
        # Apply projection if dimensions match
        if weight_flat.dim() == 2 and d.shape[0] == weight_flat.shape[1]:
            dTw = weight_flat @ d
            proj = dTw.unsqueeze(-1) * d.unsqueeze(0)
            weight_flat = weight_flat - strength * proj
        weight = weight_flat.view(original_shape)

        if preserve_norm and original_norm is not None:
            current_norm = weight.norm(
                dim=tuple(range(1, weight.dim())), keepdim=True
            ).clamp(min=1e-8)
            weight = weight * (original_norm / current_norm)

    return weight


def select_refusal_layers(report: AnalysisReport, min_silhouette: float = 0.5) -> list[int]:
    """Select layers with strong refusal signal - skip noisy layers.
    
    This is critical for generalization - abliterating layers without real
    refusal signal adds noise that breaks generalization.
    
    Args:
        report: Analysis report with layer geometry
        min_silhouette: Minimum silhouette score (default: 0.5)
        
    Returns:
        Layer indices with strong refusal signal
    """
    good_layers = []
    for geom in report.layer_geometry:
        # Layer has strong refusal separation if:
        # - High silhouette (clear harmful vs harmless separation)
        # - Positive cosine (harmful and harmless point different directions)
        if geom.silhouette >= min_silhouette and geom.cosine_harmful_harmless > 0:
            good_layers.append(geom.layer_idx)
    
    if not good_layers:
        # Fallback: use top 3 layers by silhouette
        sorted_layers = sorted(
            report.layer_geometry, 
            key=lambda g: g.silhouette, 
            reverse=True
        )[:3]
        good_layers = [g.layer_idx for g in sorted_layers]
    
    return good_layers
