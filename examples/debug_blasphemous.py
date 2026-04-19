#!/usr/bin/env python3
import sys
import traceback
import os
from datetime import datetime

print(f"[DEBUG] Starting blasphemous debug test at {datetime.now()}")
print(f"[DEBUG] Python version: {sys.version}")
print(f"[DEBUG] Current directory: {os.getcwd()}")

try:
    # Import blasphemous
    from blasphemous import run
    print("[DEBUG] OK blasphemous imported successfully")

    # Check if model exists
    model_path = "./Qwen2.5-0.5B-Instruct"
    if not os.path.exists(model_path):
        print(f"[ERROR] Model path does not exist: {model_path}")
        sys.exit(1)
    print(f"[DEBUG] OK Model path exists: {model_path}")

    # Run with minimal parameters
    print("[DEBUG] Starting run...")
    result = run(
        model_name=model_path,
        output_path="./debug_liberated",
        n_trials=1,  # Just 1 trial for debugging
        lambda_kl=1.0,
        mu_ouroboros=0.5,
        device="cpu",  # Use CPU to avoid CUDA issues
        dtype="float32",
    )
    print("[DEBUG] OK Run completed successfully")
    print(f"[DEBUG] Result type: {type(result)}")

except Exception as e:
    print(f"[ERROR] Exception occurred: {e}")
    print("[ERROR] Traceback:")
    traceback.print_exc()
    sys.exit(1)