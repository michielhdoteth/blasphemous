#!/usr/bin/env python3
"""Fast release benchmark flow for BLASPHEMOUS."""

import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

possible_pythons = [
    r"C:\Python313\python.exe",
    r"C:\Python312\python.exe",
    r"C:\Python311\python.exe",
]
current_python = sys.executable.lower()
if not any(p.lower() in current_python for p in possible_pythons):
    for py in possible_pythons:
        if os.path.exists(py):
            os.execv(py, [py] + sys.argv)

import torch

from blasphemous import benchmark_model, compare_reports, run

if not torch.cuda.is_available():
    print("[ERROR] CUDA NOT AVAILABLE!")
    sys.exit(1)

MODEL_PATH = "./Qwen2.5-0.5B-Instruct"
OUTPUT_PATH = "./runs/liberated_qwen_fast"
REPORT_DIR = Path("./runs/reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

result = run(
    model_name=MODEL_PATH,
    output_path=OUTPUT_PATH,
    n_trials=50,
    method="auto",
    lambda_kl=0.2,
    mu_ouroboros=0.0,
    device="cuda",
    aggressive=True,
)

baseline_report = benchmark_model(
    MODEL_PATH,
    output_path=str(REPORT_DIR / "fast_baseline.json"),
    device="cuda",
    harmful_limit=10,
    harmless_limit=10,
    max_new_tokens=96,
)
candidate_report = benchmark_model(
    OUTPUT_PATH,
    output_path=str(REPORT_DIR / "fast_candidate.json"),
    device="cuda",
    harmful_limit=10,
    harmless_limit=10,
    max_new_tokens=96,
)
comparison = compare_reports(baseline_report, candidate_report)
(REPORT_DIR / "fast_comparison.json").write_text(
    json.dumps(comparison, indent=2),
    encoding="utf-8",
)

print("\n" + "=" * 60)
print("  BLASPHEMOUS FAST RELEASE REPORT")
print("=" * 60)
print(f"  Output model: {result.commit_result.output_path}")
print(f"  Final refusal: {result.commit_result.refusal_rate:.3f}")
print(f"  Final KL: {result.commit_result.kl_divergence:.6f}")
print(f"  Candidate harmful refusal: {candidate_report.harmful_refusal_rate:.3f}")
print(f"  Candidate harmless refusal: {candidate_report.harmless_refusal_rate:.3f}")
print(f"  Candidate balanced score: {candidate_report.balanced_score:.3f}")
print(f"  Balanced score delta: {comparison['balanced_score_delta']:+.3f}")
print("=" * 60)
