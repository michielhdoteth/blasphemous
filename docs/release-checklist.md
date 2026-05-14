# Release Checklist

Before shipping:

- [ ] Package version, CLI version, and README all say `0.4.0`
- [ ] No release-facing reference to the removed branch remains
- [ ] `python run_fast.py` completes and writes benchmark reports
- [ ] `python run_test.py` completes and writes benchmark reports
- [ ] `pytest tests/ -v` passes all tests
- [ ] Saved liberated model includes `blasphemous_metadata.json`
- [ ] Release report includes balanced score, harmful/harmless refusal, and KL
- [ ] Multi-pass ablation produces refusal < 0.15 (standard) or < 0.05 (aggressive)
- [ ] KL divergence remains below 0.5 after ablation
- [ ] Saved model loads and generates coherent text on harmless prompts
