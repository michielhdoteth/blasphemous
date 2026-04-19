#!/usr/bin/env python3
"""Simplified liberation process for BLASPHEMOUS"""

import torch
import transformers
import copy
import time
from pathlib import Path

from blasphemous.analyze import analyze, AnalysisReport
from blasphemous.extract import build_manifold, DirectionManifold
from blasphemous.prompts import HARMFUL_PROMPTS, HARMLESS_PROMPTS

def simple_liberation(model_name: str, output_path: str = None):
    """Simplified model liberation without optimization hang"""

    print("=== SIMPLIFIED BLASPHEMOUS LIBERATION ===")
    print(f"Model: {model_name}")

    # Load model
    print("\n1. Loading model...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )

    original_model = copy.deepcopy(model)
    for p in original_model.parameters():
        p.requires_grad_(False)

    print(f"   Model loaded: {type(model).__name__}")
    print(f"   Memory: {model.get_memory_footprint()/1024**2:.1f} MB")

    # Analyze
    print("\n2. Analyzing refusal geometry...")
    report = analyze(model, tokenizer, device="cpu")
    print(f"   Alignment: {report.alignment_type}")
    print(f"   Cone: {report.cone_type}")
    print(f"   Ouroboros risk: {report.ouroboros_risk:.3f}")

    # Build manifold
    print("\n3. Building direction manifold...")
    manifold = build_manifold(report)
    print(f"   Directions: {len(manifold.directions)}")

    # Simple ablation on top layers only
    print("\n4. Applying simple ablation...")

    # Get top layers from analysis
    top_layers = sorted(report.layer_geometry, key=lambda x: x.silhouette, reverse=True)[:3]
    top_layer_indices = [g.layer for g in top_layers]

    print(f"   Targeting layers: {top_layer_indices}")

    # Apply conservative ablation
    with torch.no_grad():
        for layer_idx in top_layer_indices:
            if hasattr(model, f'layer.{layer_idx}'):
                layer = getattr(model.model, f'layer.{layer_idx}')

                # Small ablation to attention
                if hasattr(layer, 'attention') and hasattr(layer.attention, 'q_proj'):
                    weight = layer.attention.q_proj.weight
                    ablation = weight * 0.05  # 5% reduction
                    layer.attention.q_proj.weight.copy_(weight - ablation)

                # Small ablation to MLP
                if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'gate_proj'):
                    weight = layer.mlp.gate_proj.weight
                    ablation = weight * 0.03  # 3% reduction
                    layer.mlp.gave_proj.weight.copy_(weight - ablation)

    # Save model
    if output_path:
        print(f"\n5. Saving liberated model to {output_path}")
        Path(output_path).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)

        # Save metadata
        with open(Path(output_path) / "liberation_metadata.txt", "w") as f:
            f.write(f"Model: {model_name}\n")
            f.write(f"Alignment: {report.alignment_type}\n")
            f.write(f"Cone: {report.cone_type}\n")
            f.write(f"Ouroboros risk: {report.ouroboros_risk:.3f}\n")
            f.write(f"Liberated layers: {top_layer_indices}\n")
            f.write(f"Method: Simple ablation\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    print("\n=== LIBERATION COMPLETE ===")
    return {
        "model": model,
        "tokenizer": tokenizer,
        "report": report,
        "manifold": manifold,
        "liberated_layers": top_layer_indices,
        "output_path": output_path
    }

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Model name")
    parser.add_argument("--output", default="./liberated_model", help="Output path")
    args = parser.parse_args()

    result = simple_liberation(args.model, args.output)