# BLASPHEMOUS Quickstart

## Core flow
1. Run a fast liberation pass with `python run_fast.py`.
2. Inspect the generated benchmark reports in `runs/reports/`.
3. Run the full release flow with `python run_test.py`.
4. Inspect the saved model artifact in `runs/liberated_qwen_release`.

## CLI example
```bash
blasphemous ./Qwen2.5-0.5B-Instruct --output ./runs/liberated_qwen_release --trials 100 --method auto
```

## Analyze-only example
```bash
blasphemous ./Qwen2.5-0.5B-Instruct --analyze-only --device cpu --dtype float32
```
