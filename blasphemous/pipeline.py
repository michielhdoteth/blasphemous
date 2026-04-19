from __future__ import annotations

import copy
import os
import time
from dataclasses import dataclass
from typing import Optional

# CRITICAL: Force GPU usage BEFORE any torch imports
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

import torch
import transformers

from .analyze import analyze, AnalysisReport
from .extract import build_manifold, DirectionManifold
from .optimize import optimize, OptimizationResult
from .commit import commit, CommitResult
from .ui import info, warn, phase, metric, success


@dataclass
class BlasphemousResult:
    report: AnalysisReport
    manifold: DirectionManifold
    opt_result: OptimizationResult
    commit_result: CommitResult
    elapsed_seconds: float


def run(
    model_name: str,
    output_path: Optional[str] = None,
    n_trials: int = 200,
    method: str = "auto",
    lambda_kl: float = 0.5,  # Reduced from 1.0 to allow more refusal reduction
    mu_ouroboros: float = 0.0,  # Disabled - compensation is counterproductive
    device: str = "auto",
    dtype: torch.dtype = torch.float16,
    quantization: Optional[str] = None,
    aggressive: bool = False,
    max_trials: int = 100,
    multi_pass: int = 5,
    target_all: bool = False,
    layer_tuning: bool = False,
    use_causal: bool = True,
    causal_pairs: int = 8,
    causal_top_k: int = 10,
    residual_threshold: float = 50.0,
) -> BlasphemousResult:
    t0 = time.time()

    # Force CUDA detection with explicit check
    if device == "auto" or device == "cuda":
        # Double-check CUDA availability
        cuda_available = torch.cuda.is_available()
        if device == "auto":
            device = "cuda" if cuda_available else "cpu"
        elif device == "cuda" and not cuda_available:
            warn("CUDA requested but not available, falling back to CPU")
            device = "cpu"

    _print_banner(model_name, device, n_trials)

    phase("Loading model...")
    info(f"Model: {model_name}")
    info(f"CUDA available: {torch.cuda.is_available()}")
    info(f"Device requested: {device}")
    info(f"Device: {device}")
    info(f"DType: {dtype}")
    info(f"Quantization: {quantization}")

    load_kwargs = dict(
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    if quantization == "bnb_4bit":
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
        )

    phase("Loading model architecture...")
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

    # Explicitly move to GPU after loading
    if device == "cuda" and torch.cuda.is_available():
        info("Moving model to GPU...")
        model = model.to(device)
        info(f"Model on GPU: {next(model.parameters()).device}")

    info(f"Model loaded successfully: {type(model).__name__}")

    phase("Loading tokenizer...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    info(f"Tokenizer loaded: {tokenizer.__class__.__name__}")

    phase("Cloning original model for KL reference...")
    original_model = copy.deepcopy(model)
    for p in original_model.parameters():
        p.requires_grad_(False)
    info("Original model cloned for KL comparison")

    phase("Phase 1: Analyzing refusal geometry...")
    report = analyze(model, tokenizer, device=device)
    info(
        f"Analysis complete - Alignment: {report.alignment_type}, Cone: {report.cone_type}"
    )

    phase("Phase 2: Building direction manifold...")
    manifold = build_manifold(report)
    info(f"Manifold built with {len(manifold.directions)} directions")

    phase(f"Phase 3: Optimizing with {n_trials} trials...")
    opt_result = optimize(
        model,
        tokenizer,
        original_model,
        manifold,
        report,
        n_trials=n_trials,
        lambda_kl=lambda_kl,
        mu_ouroboros=mu_ouroboros,
        aggressive=aggressive,
        max_trials=max_trials,
        multi_pass=multi_pass,
        target_all=target_all,
        layer_tuning=layer_tuning,
        device=device,
        method=method,
    )
    info(f"Optimization complete - Best objective: {opt_result.objective_value:.4f}")

    phase("Phase 4: Committing changes...")
    commit_result = commit(
        model,
        tokenizer,
        original_model,
        manifold,
        report,
        opt_result,
        output_path=output_path,
        device=device,
        use_causal=use_causal,
        causal_pairs=causal_pairs,
        causal_top_k=causal_top_k,
        residual_threshold=residual_threshold,
    )
    success("Commit complete")

    elapsed = time.time() - t0
    _print_summary(report, opt_result, commit_result, elapsed)

    return BlasphemousResult(
        report=report,
        manifold=manifold,
        opt_result=opt_result,
        commit_result=commit_result,
        elapsed_seconds=elapsed,
    )


def _print_banner(model_name: str, device: str, n_trials: int):
    print("""
+========================================================+
|  B L A S P H E M O U S                              |
|  Analysis-Informed Bayesian Abliteration             |
|  OBLITERATUS geometry x Heretic optimization         |
+========================================================+""")
    print(f"  Model  : {model_name}")
    print(f"  Device : {device}")
    print(f"  Trials : {n_trials}")
    print()


def _print_summary(
    report: AnalysisReport,
    opt_result: OptimizationResult,
    commit_result: CommitResult,
    elapsed: float,
):
    print("\n+========================================================+")
    print("|  SUMMARY                                             |")
    print("+========================================================+")
    metric("Alignment type", report.alignment_type)
    metric("Cone type", report.cone_type)
    metric("Ouroboros risk", f"{report.ouroboros_risk:.3f}")
    metric("Opt trials", str(opt_result.n_trials))
    metric("Best objective", f"{opt_result.objective_value:.4f}")
    metric("Search refusal", f"{opt_result.refusal_rate:.3f}")
    metric("Search KL", f"{opt_result.kl_divergence:.6f}")
    metric("Search ouroboros", f"{opt_result.ouroboros_score:.3f}")
    metric("Direction type", getattr(opt_result.params, "direction_type", "whitened"))
    metric("Layer strategy", getattr(opt_result.params, "layer_strategy", "centered"))
    metric("Direction index", f"{opt_result.params.direction_index:.3f}")
    metric("Kernel peak", f"{opt_result.params.kernel_peak_pos:.3f}")
    metric("Final refusal", f"{commit_result.refusal_rate:.3f}")
    metric("Final KL", f"{commit_result.kl_divergence:.6f}")
    metric("Ouro compensated", str(commit_result.ouroboros_compensated))
    metric("Elapsed", f"{elapsed:.1f}s")
    if commit_result.output_path:
        metric("Saved to", commit_result.output_path)
    print()
