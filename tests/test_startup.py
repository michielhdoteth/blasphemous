#!/usr/bin/env python3
"""Release-oriented startup smoke tests."""

from pathlib import Path
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_model_loading():
    """Import the package and run analysis on the local release target."""

    print("Testing BLASPHEMOUS startup...")

    # Test 1: Check imports
    print("\n1. Testing imports...")
    start = time.time()
    try:
        from blasphemous import run
        print(f"   [OK] blasphemous imported in {time.time()-start:.2f}s")
    except Exception as e:
        print(f"   [FAIL] Import failed: {e}")
        return False

    # Test 2: Try loading the local release target
    print("\n2. Testing small model loading...")
    try:
        model_name = "./Qwen2.5-0.5B-Instruct"
        assert Path(model_name).exists(), f"Missing local model at {model_name}"

        print("   Loading tokenizer...")
        t0 = time.time()
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print(f"   [OK] Tokenizer loaded in {time.time()-t0:.2f}s")

        print("   Loading model...")
        t0 = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="cpu",  # Force CPU for testing
            trust_remote_code=True,
        )
        print(f"   [OK] Model loaded in {time.time()-t0:.2f}s")
        print(f"   Model type: {type(model).__name__}")

    except Exception as e:
        print(f"   [FAIL] Model loading failed: {e}")
        return False

    # Test 3: Try running analyze phase only
    print("\n3. Testing analysis phase...")
    try:
        from blasphemous.analyze import analyze

        print("   Running analysis...")
        t0 = time.time()
        report = analyze(model, tokenizer, device="cpu")
        print(f"   [OK] Analysis complete in {time.time()-t0:.2f}s")
        print(f"   Alignment: {report.alignment_type}, Cone: {report.cone_type}")

    except Exception as e:
        print(f"   [FAIL] Analysis failed: {e}")
        return False

    print("\n[OK] All tests passed!")
