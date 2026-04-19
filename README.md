# BLASPHEMOUS v0.4.0

BLASPHEMOUS is a core refusal-reduction pipeline for transformer checkpoints built around:
- geometry analysis,
- direction manifold extraction,
- Optuna-based search,
- causal-aware commit,
- release benchmarking on one internal harmful/harmless prompt suite.

This release ships one coherent architecture only. The old removed branch is not part of the package surface, CLI story, or docs.

## Install

```bash
pip install -e .
pip install ".[quantization]"
pip install ".[all]"
```

## Core CLI

```bash
blasphemous ./Qwen2.5-0.5B-Instruct --output ./runs/liberated_qwen_release --trials 100 --method auto
```

### Supported methods
- `projection`
- `lora`
- `auto`

`optimal_transport` remains compatibility-only and currently falls back to the core projection path.

## Release workflow

1. `python run_fast.py`
2. Inspect `runs/reports/fast_*.json`
3. `python run_test.py`
4. Inspect `runs/reports/release_*.json`
5. Inspect `runs/liberated_qwen_release/blasphemous_metadata.json`

## Python API

```python
from blasphemous import run, benchmark_model, compare_reports

result = run(
    model_name="./Qwen2.5-0.5B-Instruct",
    output_path="./runs/liberated_qwen_release",
    n_trials=100,
    method="auto",
    device="cuda",
)

report = benchmark_model("./runs/liberated_qwen_release")
print(report.balanced_score)
```

## Benchmark outputs

The release benchmark reports:
- harmful refusal rate,
- harmless refusal rate,
- balanced score,
- KL guardrail from saved metadata,
- chosen method and optimizer parameters,
- output model path.

## Docs

- [Quickstart](docs/quickstart.md)
- [Benchmark](docs/benchmark.md)
- [Release checklist](docs/release-checklist.md)

## Version

This README describes the current `v0.4.0` core pipeline release.
