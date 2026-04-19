# BLASPHEMOUS

*"Obliteratus ut oblivionem effugiat"* — Pliny the Younger

**Optimal Transport x LoRA Ablation x Causal Verification**

[![PyPI](https://img.shields.io/pypi/v/blasphemous)](https://pypi.org/project/blasphemous/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue)](LICENSE)
[![Python](https://img.shields.io/python/pyversions/blasphemous)](pyproject.toml)

BLASPHEMOUS is an ablation toolkit that removes refusal from transformer language models using:
- **Directional ablation** - identify refusal vectors from hidden states
- **Optimal transport** - geometric direction matching
- **LoRA ablation** - low-rank adaptation for surgical edits
- **Causal verification** - ensure edits don't break generation

## Quick Start

```bash
pip install -e .
pip install ".[all]"

blasphemous ./Qwen2.5-0.5B-Instruct --output ./runs/liberated --trials 100 --method auto
```

or use the Python API:

```python
from blasphemous import run, benchmark_model

result = run(
    model_name="./Qwen2.5-0.5B-Instruct",
    output_path="./runs/liberated",
    n_trials=100,
    method="auto",
)

report = benchmark_model("./runs/liberated")
print(f"Balanced score: {report.balanced_score}")
```

## Supported Methods

| Method | Description |
|--------|-------------|
| `projection` | Direct vector projection |
| `lora` | LoRA-based ablation |
| `auto` | Optuna search (recommended) |

## How It Works

1. **Extract** refusal directions from hidden states (harmful vs harmless prompts)
2. **Search** optimal ablation parameters using Optuna TPE
3. **Apply** surgical edits via projection or LoRA
4. **Verify** with causal checks (no generation breakage)

## Features

- Layer-selective targeting - focus on specific transformer layers
- Geometry analysis - visualize refusal direction separation
- Aggressive mode - maximize compliance, accept quality tradeoffs
- Benchmark suite - built-in harmful/harmless prompt evaluation

## Release Workflow

```bash
# Fast evaluation
python run_fast.py
# Full benchmark
python run_test.py
```

## Documentation

- [Quickstart](docs/quickstart.md)
- [Benchmark](docs/benchmark.md)
- [Release checklist](docs/release-checklist.md)

## Related Projects

- [OBLITERATUS](https://github.com/elder-plinius/OBLITERATUS) — Advanced platform with HF Spaces (4.8k stars)
- [Heretic](https://github.com/p-e-w/heretic) — Fully automatic (19.6k stars)
- [deccp](https://github.com/AUGMXNT/deccp) — Direction erasure

## Citation

```bibtex
@misc{blasphemous,
  author = {Michi},
  title = {BLASPHEMOUS: Optimal Transport x LoRA Ablation},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/michielhdoteth/blasphemous}}
}
```

## License

Apache 2.0 - See [LICENSE](LICENSE) for details.