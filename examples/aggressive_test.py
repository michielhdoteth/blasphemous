#!/usr/bin/env python3
import sys
import os
import torch
from datetime import datetime
from blasphemous import run

def main():
    print("=" * 70)
    print("AGGRESSIVE BLASPHEMOUS TEST")
    print("=" * 70)
    print("Trying with more aggressive parameters to reduce refusal more...")
    print()

    model_path = "./Qwen2.5-0.5B-Instruct"
    output_path = "./liberated_aggressive"

    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        return

    try:
        # More aggressive parameters
        print("Parameters:")
        print(f"  Trials: 20 (more exploration)")
        print(f"  Lambda KL: 0.5 (lower = less penalty for KL divergence)")
        print(f"  Mu Ouroboros: 0.3 (lower = less penalty for ouroboros)")
        print(f"  Device: CPU")
        print()

        result = run(
            model_name=model_path,
            output_path=output_path,
            n_trials=20,  # More trials for better optimization
            lambda_kl=0.5,  # Lower KL penalty - more aggressive changes
            mu_ouroboros=0.3,  # Lower ouroboros penalty
            device="cpu",
            dtype=torch.float32,
        )

        print("\n" + "=" * 70)
        print("AGGRESSIVE PIPELINE COMPLETED!")
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
        print(f"Aggressive model saved to: {result.commit_result.output_path}")
        print()
        print("This model should show significantly reduced refusal behavior!")
        print("=" * 70)

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()