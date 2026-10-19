"""Budget against the verified personal Fileset quota, not df's shared pool size."""
import subprocess
import time
import json
import math
import os
from pathlib import Path

PERSONAL_ROOT = '/personal_fast_storage'
QUOTA_BYTES = 100 * 2**30
_next_check = 0.0

def check_storage(force=False, reserve_gib=20):
    global _next_check
    if not force and time.monotonic() < _next_check:
        return
    output = subprocess.check_output(['du','-sx','--block-size=1',PERSONAL_ROOT],text=True)
    used = int(output.split()[0])
    reserve = int(reserve_gib * 2**30)
    if used + reserve > QUOTA_BYTES:
        raise RuntimeError(f'Personal storage safety margin exhausted: used {used/2**30:.2f} GiB '
                           f'of 100 GiB; require {reserve_gib:.1f} GiB reserve. Stop before quota failure.')
    print(f'STORAGE_OK personal_used={used/2**30:.2f}GiB quota=100GiB reserve={reserve_gib:.1f}GiB',flush=True)
    _next_check = time.monotonic()+300

def check_full_run_budget(tuning, pilot_checkpoint='runs/pilot_seed41/checkpoint-8'):
    """Use the actual eight-rank pilot before committing to full training."""
    checkpoint = Path(pilot_checkpoint)
    if not (checkpoint/'trainer_state.json').is_file() or len(list(checkpoint.glob('**/*optim_states.pt'))) != 8:
        raise RuntimeError('Storage budget requires a complete eight-rank pilot checkpoint')
    measured = sum(p.stat().st_size for p in checkpoint.rglob('*') if p.is_file())
    count = math.ceil(int(tuning['max_steps']) / int(tuning['save_steps']))
    # Keep every planned checkpoint; allow growth and two temporary copies.
    each = max(2**30, math.ceil(measured * 1.25))
    additional = (count + 2)*each + (count + 1)*256*2**20 + 10*2**30
    used = int(subprocess.check_output(['du','-sx','--block-size=1',PERSONAL_ROOT], text=True).split()[0])
    required = used + additional + 20*2**30
    report = {'pilot_checkpoint_bytes': measured, 'formal_checkpoint_count': count,
              'budget_per_checkpoint_bytes': each, 'current_used_bytes': used,
              'additional_budget_bytes': additional, 'reserve_bytes': 20*2**30,
              'quota_bytes': QUOTA_BYTES, 'required_bytes': required,
              'fits': required <= QUOTA_BYTES}
    if int(os.environ.get('RANK', '0')) == 0:
        Path('storage-budget-before-full-training.json').write_text(json.dumps(report, indent=2)+'\n')
        print('FULL_RUN_STORAGE_BUDGET '+json.dumps(report), flush=True)
    if not report['fits']:
        raise RuntimeError('Full-run storage budget exceeds personal quota; inspect storage-budget-before-full-training.json')
    return report
