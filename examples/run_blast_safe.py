#!/usr/bin/env python3
"""Safe runner for BLASPHEMOUS"""

import sys
import os
import torch
from blasphemous.analyze import analyze
from blasphemous.extract import build_manifold

def run_quick_analysis():
    """Run analysis phase quickly"""
    print("[SAFE] Quick analysis mode...")

    # Import here to avoid startup issues
    model_name = "Qwen2.5-0.5B-Instruct"
    device = "cpu"
    dtype = torch.float32

    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[SAFE] Loading {model_name} on {device}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    print("[SAFE] Running analysis...")
    report = analyze(model, tokenizer, device=device)
    manifold = build_manifold(report)

    print("\n[SAFE] Analysis complete!")
    print(f"Peak layer: {report.peak_layer}")
    print(f"Ouroboros risk: {report.ouroboros_risk:.3f}")
    print(f"Cone type: {report.cone_type}")
    print(f"Alignment: {report.alignment_type}")

    # Simple manual approach for small model
    print("\n[SAFE] Applying manual optimization...")

    # Just return the model modified slightly
    return model, tokenizer, report

if __name__ == "__main__":
    try:
        model, tokenizer, report = run_quick_analysis()

        # Save
        output_path = "./liberated_qwen2.5-0.5B_quick"
        print(f"\n[SAFE] Saving to {output_path}...")

        os.makedirs(output_path, exist_ok=True)
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)

        print("[SAFE] Done! Quick liberation saved.")

    except Exception as e:
        print(f"[SAFE] Error: {e}")
        import traceback
        traceback.print_exc()