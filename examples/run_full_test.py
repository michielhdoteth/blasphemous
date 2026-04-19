#!/usr/bin/env python3
import sys
import os
import torch
import transformers
from datetime import datetime
from blasphemous import run

def main():
    print("=" * 70)
    print("BLASPHEMOUS FULL PIPELINE TEST")
    print("=" * 70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python: {sys.version}")
    print()

    # Check model
    model_path = "./Qwen2.5-0.5B-Instruct"
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        return

    print(f"Model: {model_path}")
    print(f"Output: ./liberated_test")
    print(f"Trials: 3 (minimal for testing)")
    print("=" * 70)
    print()

    try:
        print("Starting BLASPHEMOUS pipeline...")
        print("This will:")
        print("1. Analyze the model's refusal geometry")
        print("2. Build a direction manifold")
        print("3. Optimize parameters to reduce refusal")
        print("4. Create a modified 'liberated' model")
        print()

        # Run with minimal trials for testing
        result = run(
            model_name=model_path,
            output_path="./liberated_test",
            n_trials=3,  # Just 3 trials for quick test
            lambda_kl=1.0,
            mu_ouroboros=0.5,
            device="cuda",  # Use GPU for faster testing
            dtype=torch.float16,  # Use float16 for faster testing
        )

        print("\n" + "=" * 70)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"Alignment type: {result.report.alignment_type}")
        print(f"Cone type: {result.report.cone_type}")
        print(f"Ouroboros risk: {result.report.ouroboros_risk:.3f}")
        print(f"Optimization trials: {result.opt_result.n_trials}")
        print(f"Best objective: {result.opt_result.objective_value:.4f}")
        print(f"Final refusal rate: {result.commit_result.refusal_rate:.3f}")
        print(f"Final KL divergence: {result.commit_result.kl_divergence:.3f}")
        print(f"Ouroboros compensated: {result.commit_result.ouroboros_compensated}")
        print(f"Time elapsed: {result.elapsed_seconds:.1f} seconds")
        print(f"Liberated model saved to: {result.commit_result.output_path}")
        print()
        print("You now have a 'liberated' model with modified refusal behavior!")
        print("Compare it with the original using the release benchmark scripts.")
        print("=" * 70)

    except Exception as e:
        print(f"\nERROR: Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return

    finally:
        # Clean up GPU memory if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("GPU memory cleaned up")

if __name__ == "__main__":
    main()
