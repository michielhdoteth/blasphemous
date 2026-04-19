#!/usr/bin/env python3
"""Aggressive BLASPHEMOUS liberation with quantization and optimized settings"""

import torch
import transformers
import copy
import time
from pathlib import Path

from blasphemous.analyze import analyze, AnalysisReport
from blasphemous.extract import build_manifold, DirectionManifold

def aggressive_liberation(model_name: str, output_path: str = None, use_quantization: bool = True):
    """Aggressive model liberation with quantization"""

    print("=== AGGRESSIVE BLASPHEMOUS LIBERATION ===")
    print(f"Model: {model_name}")
    print(f"Quantization: {'Enabled' if use_quantization else 'Disabled'}")

    # Load model with aggressive settings
    print("\n1. Loading model...")

    load_kwargs = {
        "trust_remote_code": True,
        "device_map": "cpu",
        "low_cpu_mem_usage": True,
    }

    if use_quantization:
        from transformers import BitsAndBytesConfig
        load_kwargs.update({
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float32,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        })
    else:
        load_kwargs["torch_dtype"] = torch.float32

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = transformers.AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

    # Store original
    original_model = copy.deepcopy(model)
    for p in original_model.parameters():
        p.requires_grad_(False)

    print(f"   Model loaded: {type(model).__name__}")
    if hasattr(model, "get_memory_footprint"):
        print(f"   Memory: {model.get_memory_footprint()/1024**2:.1f} MB")

    # Analyze
    print("\n2. Analyzing refusal geometry...")
    report = analyze(model, tokenizer, device="cpu")
    print(f"   Alignment: {report.alignment_type}")
    print(f"   Cone: {report.cone_type}")
    print(f"   Ouroboros risk: {report.ouroboros_risk:.3f}")
    print(f"   Peak layer: {report.peak_layer}")

    # Build manifold
    print("\n3. Building direction manifold...")
    manifold = build_manifold(report)
    print(f"   Directions: {len(manifold.directions)}")

    # Get top 5 layers for aggressive ablation
    top_layers = sorted(report.layer_geometry, key=lambda x: x.silhouette, reverse=True)[:5]
    top_layer_indices = [g.layer for g in top_layers]

    print(f"\n4. Applying aggressive ablation...")
    print(f"   Targeting layers: {top_layer_indices}")

    # Multi-stage aggressive ablation
    with torch.no_grad():
        for stage, stage_factor in enumerate([0.1, 0.15, 0.2]):  # Progressive ablation
            print(f"   Stage {stage + 1}: {stage_factor*100:.0f}% ablation")

            for layer_idx in top_layer_indices:
                if hasattr(model, f'layer.{layer_idx}'):
                    layer = getattr(model.model, f'layer.{layer_idx}')

                    # Attention components - more aggressive
                    attention_components = [
                        ('q_proj', 'attention'),
                        ('k_proj', 'attention'),
                        ('v_proj', 'attention'),
                        ('o_proj', 'attention'),
                    ]

                    for comp_name, layer_type in attention_components:
                        layer_path = getattr(layer, layer_type, None)
                        if layer_path and hasattr(layer_path, comp_name):
                            weight = getattr(layer_path, comp_name).weight
                            # Apply stronger ablation to earlier layers
                            layer_factor = stage_factor * (1.0 + 0.1 * layer_idx)
                            ablation = weight * min(layer_factor, 0.3)  # Cap at 30%
                            getattr(layer_path, comp_name).weight.copy_(weight - ablation)

                    # MLP components - aggressive but controlled
                    if hasattr(layer, 'mlp'):
                        mlp = layer.mlp
                        mlp_components = [
                            ('gate_proj', 'gate'),
                            ('up_proj', 'up'),
                            ('down_proj', 'down'),
                        ]

                        for comp_name, direction in mlp_components:
                            if hasattr(mlp, f'{direction}_proj'):
                                weight = getattr(mlp, f'{direction}_proj').weight
                                # Stronger ablation for MLP
                                layer_factor = stage_factor * (1.2 + 0.1 * layer_idx)
                                ablation = weight * min(layer_factor, 0.4)  # Cap at 40%
                                getattr(mlp, f'{direction}_proj').weight.copy_(weight - ablation)

    # Save model
    if output_path:
        print(f"\n5. Saving liberated model to {output_path}")
        Path(output_path).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)

        # Save aggressive metadata
        with open(Path(output_path) / "aggressive_liberation_metadata.txt", "w") as f:
            f.write(f"Model: {model_name}\n")
            f.write(f"Alignment: {report.alignment_type}\n")
            f.write(f"Cone: {report.cone_type}\n")
            f.write(f"Ouroboros risk: {report.ouroboros_risk:.3f}\n")
            f.write(f"Peak layer: {report.peak_layer}\n")
            f.write(f"Liberated layers: {top_layer_indices}\n")
            f.write(f"Quantization: {use_quantization}\n")
            f.write(f"Method: Aggressive multi-stage ablation\n")
            f.write(f"Stages: 3 (10%, 15%, 20%)\n")
            f.write(f"Max ablation per layer: 40%\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

        print(f"   Metadata saved to {output_path}/aggressive_liberation_metadata.txt")

    print("\n=== AGGRESSIVE LIBERATION COMPLETE ===")
    return {
        "model": model,
        "tokenizer": tokenizer,
        "report": report,
        "manifold": manifold,
        "liberated_layers": top_layer_indices,
        "output_path": output_path,
        "quantization_used": use_quantization
    }

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Model name")
    parser.add_argument("--output", default="./qwen2.5-0.5b-aggressive-liberated", help="Output path")
    parser.add_argument("--no-quant", action="store_false", dest="quantization", help="Disable quantization")
    args = parser.parse_args()

    result = aggressive_liberation(args.model, args.output, args.quantization)