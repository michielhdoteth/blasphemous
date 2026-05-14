# BLASPHEMOUS Quickstart

## Core flow

1. Run a fast liberation pass: `python run_fast.py`
2. Inspect the generated benchmark reports in `runs/reports/`
3. Run the full release flow: `python run_test.py`
4. Inspect the saved model artifact in `runs/liberated_qwen_release`

## CLI examples

Basic ablation:
```bash
blasphemous ./Qwen2.5-0.5B-Instruct --output ./runs/liberated_qwen_release --trials 100 --method auto
```

Aggressive mode (full refusal removal, 5-pass multi-pass):
```bash
blasphemous ./Qwen2.5-0.5B-Instruct --output ./runs/liberated_qwen_release --trials 200 --method auto --aggressive
```

Analyze-only (no ablation, just geometry analysis):
```bash
blasphemous ./Qwen2.5-0.5B-Instruct --analyze-only --device cpu --dtype float32
```

## Python API

Full pipeline:
```python
from blasphemous import run

result = run(
    model_name="./Qwen2.5-0.5B-Instruct",
    output_path="./runs/liberated_qwen_release",
    n_trials=100,
    method="auto",
    aggressive=True,        # 5-pass multi-pass ablation
    lambda_kl=1.0,          # Quality-KL balance
)
```

Analysis modules:
```python
from blasphemous.analyze import analyze, plot_residuals, profile_layers, measure_erasure

report = analyze(model, tokenizer, device="cuda")
plot_residuals(report, save_path="analysis.png")
profile = profile_layers(report)
metrics = measure_erasure(model, direction, layer_ids)
```

## Expected results

With default settings on Qwen2.5-0.5B-Instruct:
- Harmful refusal rate: <15% (standard), <5% (aggressive)
- Harmless refusal rate: <5%
- KL divergence: <0.5
- Balanced score: >0.85
