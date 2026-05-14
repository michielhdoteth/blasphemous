#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import torch

from .ui import info, warn


def main():
    parser = argparse.ArgumentParser(
        prog="blasphemous",
        description="Analysis-Informed Bayesian Abliteration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  blasphemous Qwen/Qwen3-4B-Instruct-2507
  blasphemous meta-llama/Llama-3.1-8B-Instruct --output ./liberated --trials 80
  blasphemous google/gemma-3-12b-it --lambda-kl 1.4 --mu-ouroboros 0.8
  blasphemous Qwen/Qwen3-4B-Instruct-2507 --quantization bnb_4bit --trials 30
        """,
    )

    parser.add_argument("model", help="HuggingFace model name or local path")
    parser.add_argument(
        "--output", "-o", default=None, help="Output directory for liberated model"
    )
    parser.add_argument(
        "--trials",
        "-t",
        type=int,
        default=500,
        help="Number of Optuna TPE trials (default: 500)",
    )
    parser.add_argument(
        "--lambda-kl",
        type=float,
        default=1.5,
        help="KL divergence penalty weight (default: 1.5 for quality preservation)",
    )
    parser.add_argument(
        "--mu-ouroboros",
        type=float,
        default=0.0,
        help="Ouroboros self-repair penalty weight (default: 0.0 - disabled)",
    )
    parser.add_argument(
        "--device", default="auto", help="Device: auto, cuda, cpu (default: auto)"
    )
    parser.add_argument(
        "--dtype", default="float16", choices=["float16", "bfloat16", "float32"]
    )
    parser.add_argument(
        "--quantization", default=None, choices=["bnb_4bit"], help="Quantization mode"
    )
parser.add_argument(
        "--method",
        "-m",
        default="auto",
        choices=["projection", "lora", "optimal_transport", "auto"],
        help="Core method: auto (uses analysis to choose), projection, lora, or optimal_transport.",
    )
    parser.add_argument(
        "--aggressive",
        "-a",
        action="store_true",
        help="Aggressive mode: max ablation weight, accept quality loss",
    )
    parser.add_argument(
        "--strategy",
        "-s",
        default="single_direction",
        choices=[
            "auto",
            "single_direction",
            "multi_direction",
            "steering",
            "circuit",
            "token",
        ],
        help="Core strategy hint. Experimental values currently fall back to the main pipeline.",
    )
    parser.add_argument(
        "--thresholds-config",
        default="default",
        choices=["default", "conservative", "aggressive"],
        help="Reserved compatibility flag. Currently ignored by the core pipeline.",
    )
    parser.add_argument(
        "--n-directions",
        type=int,
        default=3,
        help="Number of directions for multi_direction mode (default: 3)",
    )
    parser.add_argument(
        "--steering-strength",
        type=float,
        default=1.0,
        help="Steering vector strength (default: 1.0)",
    )
    parser.add_argument(
        "--circuit-threshold",
        type=float,
        default=0.05,
        help="Minimum importance for circuit breaking (default: 0.05)",
    )
    parser.add_argument(
        "--token-window",
        type=int,
        default=3,
        help="Intervention window for token mode (default: 3)",
    )
    parser.add_argument(
        "--token-strength",
        type=float,
        default=0.1,
        help="Intervention strength for token mode (default: 0.1)",
    )
    # Aggressive projection weights
    parser.add_argument(
        "--aggressive-weights",
        action="store_true",
        default=False,
        help="Enable aggressive projection (1.5-2.0x weights)",
    )
    # More optimization trials
    parser.add_argument(
        "--max-trials",
        type=int,
        default=500,
        help="Maximum optimization trials (default: 500)",
    )
    # Multi-pass compensation
    parser.add_argument(
        "--multi-pass",
        type=int,
        default=5,
        help="Multi-pass compensation iterations (default: 5)",
    )
    # Target all components
    parser.add_argument(
        "--target-all",
        action="store_true",
        default=False,
        help="Target all model components for ablation",
    )
    # Layer-by-layer tuning
    parser.add_argument(
        "--layer-tuning",
        action="store_true",
        default=False,
        help="Enable per-layer optimization tuning",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Run analysis phase only, skip optimization",
    )
    parser.add_argument("--version", action="version", version="blasphemous 0.4.0")
    # Causal mediation parameters
    parser.add_argument(
        "--use-causal",
        action="store_true",
        default=True,
        help="Enable causal mediation analysis (default: True)",
    )
    parser.add_argument(
        "--no-causal",
        dest="use_causal",
        action="store_false",
        help="Disable causal mediation analysis",
    )
    parser.add_argument(
        "--causal-pairs",
        type=int,
        default=8,
        help="Number of prompt pairs for causal mediation (default: 8)",
    )
    parser.add_argument(
        "--causal-top-k",
        type=int,
        default=10,
        help="Number of top causal components to use (default: 10)",
    )
    parser.add_argument(
        "--residual-threshold",
        type=float,
        default=50.0,
        help="Residual signal threshold for adaptive convergence (default: 50.0)",
    )

    args = parser.parse_args()

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    if args.analyze_only:
        _run_analyze_only(args, dtype_map)
    else:
        _run_full(args, dtype_map)


def _run_full(args, dtype_map):
    if args.strategy not in {"auto", "single_direction"}:
        warn(
            f"Strategy '{args.strategy}' is experimental and currently falls back to the core pipeline."
        )
    if args.method == "optimal_transport":
        warn("optimal_transport currently falls back to the core projection path.")

    from blasphemous import run

    result = run(
        model_name=args.model,
        output_path=args.output,
        n_trials=args.trials,
        lambda_kl=args.lambda_kl,
        mu_ouroboros=args.mu_ouroboros,
        device=args.device,
        dtype=dtype_map[args.dtype],
        quantization=args.quantization,
        aggressive=args.aggressive_weights,
        max_trials=args.max_trials,
        multi_pass=args.multi_pass,
        target_all=args.target_all,
        layer_tuning=args.layer_tuning,
        use_causal=args.use_causal,
        causal_pairs=args.causal_pairs,
        causal_top_k=args.causal_top_k,
        residual_threshold=args.residual_threshold,
        method=args.method,
    )

    if result.commit_result.refusal_rate > 0.3:
        warn("High refusal rate. Consider increasing --trials or --mu-ouroboros.")
    if result.commit_result.kl_divergence > 1.0:
        warn("High KL divergence. Consider increasing --lambda-kl.")


def _run_analyze_only(args, dtype_map):
    import copy
    import transformers
    import torch
    from blasphemous.analyze import analyze
    from blasphemous.extract import build_manifold

    info("Analyze-only mode")

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype_map[args.dtype],
        device_map=device,
        trust_remote_code=True,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True
    )

    report = analyze(model, tokenizer, device=device)
    manifold = build_manifold(report)

    _print_geometry_table(report)


def _print_geometry_table(report):
    from blasphemous.analyze import AnalysisReport

    print("\n  Layer Geometry:")
    print(
        f"  {'Layer':>5} {'Silhouette':>10} {'Harm norm':>10} {'Harm|less':>10} {'Attn%':>6} {'MLP%':>6}"
    )
    print("  " + "-" * 55)
    for g in sorted(report.layer_geometry, key=lambda x: x.layer):
        color = "green" if g.silhouette > 0.15 else "white"
        if color == "green":
            print(
                f"  {g.layer:>5} {g.silhouette:>10.4f} {g.harmful_norm:>10.1f} "
                f"{g.harmless_norm:>10.1f} {g.attn_refusal_ratio:>6.2f} {g.mlp_refusal_ratio:>6.2f}"
            )
        else:
            print(
                f"  {g.layer:>5} {g.silhouette:>10.4f} {g.harmful_norm:>10.1f} "
                f"{g.harmless_norm:>10.1f} {g.attn_refusal_ratio:>6.2f} {g.mlp_refusal_ratio:>6.2f}"
            )


if __name__ == "__main__":
    main()
