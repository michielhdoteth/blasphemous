# Release Checklist

- Package version, CLI version, and README all say `0.4.0`
- No release-facing reference to the removed branch remains
- `python run_fast.py` completes and writes benchmark reports
- `python run_test.py` completes and writes benchmark reports
- `pytest` passes for release smoke tests
- Saved liberated model includes `blasphemous_metadata.json`
- Release report includes balanced score, harmful/harmless refusal, and KL
