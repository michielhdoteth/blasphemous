#!/usr/bin/env python3
import sys
import os
import time
from datetime import datetime

# Simple test to see if blasphemous works with visual feedback
def test_blasphemous():
    print("=" * 60)
    print("BLASPHEMOUS VISUAL TEST")
    print("=" * 60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python: {sys.version}")
    print(f"Location: {os.getcwd()}")
    print("=" * 60)

    # Test 1: Import
    print("\n[1/3] Testing imports...")
    try:
        from blasphemous import run, analyze, build_manifold
        print("   OK blasphemous imported")
    except Exception as e:
        print(f"   ERROR Import failed: {e}")
        return False

    # Test 2: Model exists
    print("\n[2/3] Checking model...")
    model_path = "./Qwen2.5-0.5B-Instruct"
    if os.path.exists(model_path):
        print(f"   OK Model found at: {model_path}")
        # Count files
        files = os.listdir(model_path)
        print(f"   Model has {len(files)} files")
    else:
        print(f"   ERROR Model not found at: {model_path}")
        return False

    # Test 3: Quick analysis
    print("\n[3/3] Testing quick analysis...")
    try:
        import torch
        import transformers

        # Use CPU to avoid CUDA issues
        device = "cpu"
        dtype = torch.float32

        print("   Loading model...")
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=dtype,
            device_map=device,
            trust_remote_code=True,
        )
        print("   OK Model loaded")

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        print("   OK Tokenizer loaded")

        print("   Running analysis...")
        report = analyze(model, tokenizer, device=device)
        print("   OK Analysis complete!")

        print("\n" + "=" * 60)
        print("RESULTS:")
        print(f"   Alignment: {report.alignment_type}")
        print(f"   Cone Type: {report.cone_type}")
        print(f"   Ouroboros Risk: {report.ouroboros_risk:.3f}")
        print(f"   Peak Layer: {report.peak_layer}")
        print(f"   Layers Analyzed: {len(report.layer_geometry)}")
        print("=" * 60)
        print("SUCCESS - BLASPHEMOUS IS WORKING!")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"   ERROR Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_blasphemous()
    sys.exit(0 if success else 1)