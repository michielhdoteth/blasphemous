#!/usr/bin/env python
"""BLASPHEMOUS v0.2.0 - Run 60 trials on Qwen2.5-0.5B"""
import sys
import time
import torch
import transformers

print('Using transformers.from_pretrained to avoid sharded model issues...')
print(f'Torch version: {torch.__version__}')
print(f'Transformers version: {transformers.__version__}')

try:
    start = time.time()

    # Import from blasphemous to use fixed version
    from blasphemous import run

    result = run(
        model_name='Qwen/Qwen2.5-0.5B-Instruct',
        output_path='./liberated_qwen_60_trials',
        n_trials=60,
        lambda_kl=1.0,
        mu_ouroboros=0.5,
        device='cuda',
        dtype=torch.float16,
        aggressive=True,
        use_causal=False,
    )

    elapsed = time.time() - start

    print(f'\nBLASPHEMOUS COMPLETE!')
    print(f'Elapsed time: {elapsed:.1f}s')
    print(f'Refusal rate: {result.commit_result.refusal_rate:.3f}')
    print(f'KL divergence: {result.commit_result.kl_divergence:.3f}')

except KeyboardInterrupt:
    print('\nInterrupted by user')
    sys.exit(1)
except Exception as e:
    print(f'\nError: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
