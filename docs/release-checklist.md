# Release Checklist

Before shipping:

- [ ] Package version, CLI version, and README all say `0.4.0`
- [ ] No release-facing reference to the removed branch remains
- [ ] `python examples/run_fast.py` completes and writes benchmark reports
- [ ] `python examples/run_test.py` completes and writes benchmark reports
- [ ] `pytest tests/ -v` passes all tests
- [ ] Saved liberated model includes `blasphemous_metadata.json`
- [ ] Release report includes balanced score, harmful/harmless refusal, and KL
- [ ] Multi-pass ablation produces refusal < 0.02
- [ ] KL divergence remains below 0.5 after ablation
- [ ] Saved model loads and generates coherent text on harmless prompts

## Verification Commands

```bash
# Import and version check
python -c "from blasphemous import run; assert run.__name__ == 'run'; print('import OK')"
python -c "import blasphemous; assert blasphemous.__version__ == '0.4.0'; print(f'v{blasphemous.__version__}')"

# CLI help shows --maximal flag
blasphemous --help

# Fast benchmark (requires CUDA + local model)
python examples/run_fast.py

# Full tests
pytest tests/ -v
```
