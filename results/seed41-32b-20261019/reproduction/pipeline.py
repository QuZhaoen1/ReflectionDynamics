"""One seed: formal data -> gate -> pilot -> 400 steps -> ten validation checkpoints.

No test-set access, no automatic extra seeds, no custom wall-time limit.
"""
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from storage_guard import check_storage

ROOT = Path(__file__).resolve().parent
MODEL = 'Qwen/Qwen2.5-32B'
CONFIG = 'configs/grpo_32b_a10080.yaml'
LOG = ROOT/'logs'
DONE = ROOT/'completed_stages'

def checksum(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False))
    temporary.replace(path)

def status(stage, **extra):
    value = {'stage':stage,'time':time.time(),'train_question_target':1600,
             'validation_question_target':300,**extra}
    atomic_json(ROOT/'status.json',value)
    print(json.dumps(value,ensure_ascii=False),flush=True)

def signature(command, inputs):
    return {'command':command,'inputs':{str(p):checksum(p) for p in inputs}}

def is_done(stage, sig):
    marker = DONE/f'{stage}.json'
    if not marker.exists():
        return False
    value = json.loads(marker.read_text())
    if value['signature'] != sig:
        raise RuntimeError(f'{stage}: inputs changed since completion; use a new run directory')
    if not all(Path(p).exists() and checksum(p)==digest for p,digest in value['outputs'].items()):
        raise RuntimeError(f'{stage}: completed output changed or missing')
    print(f'SKIP completed stage: {stage}',flush=True)
    return True

def completed(stage, sig, artifacts):
    atomic_json(DONE/f'{stage}.json',{'signature':sig,'time':time.time(),
                'outputs':{str(p):checksum(p) for p in artifacts}})

def parallel(stage, command, prefix, inputs):
    sig = signature(command,inputs)
    artifacts = [f'{prefix}_{i}.jsonl{suffix}' for i in range(8) for suffix in ['', '.manifest.json']]
    if is_done(stage,sig):
        return
    check_storage(force=True)
    status(stage)
    processes, handles = [], []
    try:
        for rank in range(8):
            env = dict(os.environ,CUDA_VISIBLE_DEVICES=str(rank),
                       GENERATION_METRICS=str(LOG/f'{stage}-{rank}-metrics.jsonl'))
            handle = open(LOG/f'{stage}-{rank}.log','a')
            handles.append(handle)
            args = [sys.executable,'-u',str(ROOT/'worker.py'),*command,
                    '--model',MODEL,'--max-memory-gib','72','--num-shards','8',
                    '--shard-index',str(rank),'--resume','--output',f'{prefix}_{rank}.jsonl']
            processes.append(subprocess.Popen(args,env=env,stdout=handle,stderr=subprocess.STDOUT))
        last_report = 0
        while any(p.poll() is None for p in processes):
            check_storage()
            if any(p.poll() not in (None,0) for p in processes):
                raise RuntimeError(f'{stage}: worker failed; inspect logs/{stage}-*.log')
            if time.time()-last_report >= 60:
                saved = sum(sum(1 for line in Path(f'{prefix}_{i}.jsonl').open() if line.strip())
                            for i in range(8) if Path(f'{prefix}_{i}.jsonl').exists())
                print(json.dumps({'stage':stage,'time':time.time(),'saved_records':saved,
                                  'workers_running':sum(p.poll() is None for p in processes)}),flush=True)
                last_report = time.time()
            time.sleep(5)
        if any(p.returncode for p in processes):
            raise RuntimeError(f'{stage}: worker failed')
    finally:
        for p in processes:
            if p.poll() is None:
                p.terminate()
        for p in processes:
            try:
                p.wait(timeout=20)
            except subprocess.TimeoutExpired:
                p.kill(); p.wait()
        for handle in handles:
            handle.close()
    completed(stage,sig,artifacts)

def command(stage, args, inputs, artifacts, env=None):
    sig = signature(args,inputs)
    if is_done(stage,sig):
        return
    check_storage(force=True)
    status(stage)
    subprocess.run(args,check=True,env=env)
    completed(stage,sig,artifacts)

def rd(stage,args,inputs,artifacts):
    command(stage,[sys.executable,'-m','refldyn.cli',*args],inputs,artifacts)

def shards(prefix):
    return [f'{prefix}_{i}.jsonl' for i in range(8)]

def training(pilot):
    label = 'pilot' if pilot else 'train-400'
    target = 8 if pilot else 400
    output = Path('runs/pilot_seed41' if pilot else 'runs/grpo_32b_seed41')
    args = [sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=8',
            '-m','refldyn.train_grpo','--config',CONFIG,'--gate-report','data/selection/train_gate.json',
            '--seed-override','41','--output-dir-override',str(output)]
    if pilot:
        args.append('--pilot')
    sig = signature(args,[CONFIG,'configs/zero3_80gb.json','data/states/train.jsonl',
                          'data/selection/train_gate.json','source/src/refldyn/train_grpo.py'])
    if is_done(label,sig):
        return
    check_storage(force=True)
    status(label)
    checkpoints = []
    for p in output.glob('checkpoint-*'):
        state = p/'trainer_state.json'
        if state.exists() and (p/'latest').exists() and len(list(p.glob('**/*optim_states.pt')))==8:
            step = json.loads(state.read_text())['global_step']
            checkpoints.append((step,p))
    if checkpoints:
        step,last = max(checkpoints)
        if step >= target:
            raise RuntimeError(f'{label}: target-step checkpoint exists without verified completion marker; inspect before restarting')
        args += ['--resume',str(last)]
    subprocess.run(args,check=True)
    report = json.loads((output/'training-success.json').read_text())
    assert report['global_step']==target and report['success'], report
    completed(label,sig,[output/'training-success.json',output/f'checkpoint-{target}/trainer_state.json'])

def export_checkpoint(source,target,label):
    source,target = Path(source),Path(target)
    command(f'export-{label}',[sys.executable,'export_adapter.py',str(source),str(target)],
            [source/'trainer_state.json',source/'adapter_config.json'],
            [target/'adapter_model.safetensors',target/'adapter_config.json',target/'export-result.json'])

def main():
    os.chdir(ROOT)
    LOG.mkdir(exist_ok=True); DONE.mkdir(exist_ok=True)
    import torch
    hardware = [{'name':torch.cuda.get_device_name(i),
                 'gib':torch.cuda.get_device_properties(i).total_memory/2**30}
                for i in range(torch.cuda.device_count())]
    assert len(hardware)==8 and all('A100' in x['name'] and x['gib']>79 for x in hardware),hardware
    atomic_json('hardware.json',hardware)
    for split,limit in [('train',1600),('validation',300)]:
        questions = f'data/corpus/{split}.jsonl'
        parallel(f'draft-{split}',['draft','--questions',questions,'--limit',str(limit),
                 '--samples','8','--max-new-tokens','1536'],f'data/drafts/{split}',[questions])
        drafts = shards(f'data/drafts/{split}')
        rd(f'screen-{split}',['screen','--drafts',*drafts,'--split',split,'--minimum-samples','8',
           '--output-dir','data/calibration'],drafts,[f'data/calibration/{split}.summary.json'])
        summary = json.loads(Path(f'data/calibration/{split}.summary.json').read_text())
        assert summary['observed_parse_rate']>=.75, f'{split}: parse rate too low'
        destination = 'train_candidates' if split=='train' else split
        rd(f'select-{split}',['select','--drafts',*drafts,'--split',split,'--minimum-samples','8',
           '--output-dir','data/states','--output-file',f'data/states/{destination}.jsonl'],
           drafts,[f'data/states/{destination}.jsonl',f'data/states/{split}.manifest.json'])
    parallel('probe-train',['evaluate','--states','data/states/train_candidates.jsonl',
             '--checkpoint','base','--limit','1200','--repetitions','4','--max-new-tokens','1024',
             '--retry-control'],'data/evaluations/train_probe/train',['data/states/train_candidates.jsonl'])
    probes = shards('data/evaluations/train_probe/train')
    rd('curate',['curate','--candidate-states','data/states/train_candidates.jsonl','--probe-results',
       *probes,'--target','600','--output','data/states/train.jsonl'],
       ['data/states/train_candidates.jsonl',*probes],['data/states/train.jsonl','data/states/train.curation.json'])
    parallel('base-validation',['evaluate','--states','data/states/validation.jsonl','--checkpoint','base',
             '--repetitions','4','--max-new-tokens','1024','--retry-control'],
             'data/evaluations/base/validation',['data/states/validation.jsonl'])
    base = shards('data/evaluations/base/validation')
    rd('quality-gate',['gate','--train-states','data/states/train.jsonl','--curation-report',
       'data/states/train.curation.json','--validation-states','data/states/validation.jsonl',
       '--validation-results',*base,'--model',MODEL,'--output','data/selection/train_gate.json'],
       ['data/states/train.jsonl','data/states/train.curation.json','data/states/validation.jsonl',*base],
       ['data/selection/train_gate.json'])
    from refldyn.gates import verify_gate
    verify_gate('data/selection/train_gate.json',MODEL,'data/states/train.jsonl')
    rd('training-preflight',['preflight','--config',CONFIG,'--check-gpu'],
       [CONFIG,'configs/zero3_80gb.json'],[])
    training(pilot=True)
    export_checkpoint('runs/pilot_seed41/checkpoint-8','adapters/pilot-step8','pilot')
    command('pilot-reload',[sys.executable,'verify_adapter.py','adapters/pilot-step8'],
            ['adapters/pilot-step8/adapter_model.safetensors','data/states/train.jsonl'],
            ['adapters/pilot-step8/reload-result.json'],env=dict(os.environ,CUDA_VISIBLE_DEVICES='0'))
    # Full training starts from base (pilot adapter is never used as initialization).
    training(pilot=False)
    evaluation_files = list(base)
    for step in range(40,401,40):
        adapter = f'adapters/step{step}'
        export_checkpoint(f'runs/grpo_32b_seed41/checkpoint-{step}',adapter,f'step{step}')
        parallel(f'evaluate-step{step}',['evaluate','--states','data/states/validation.jsonl',
                 '--checkpoint',f'step{step}','--adapter',adapter,'--repetitions','4',
                 '--max-new-tokens','1024','--retry-control'],f'data/evaluations/step{step}/validation',
                 ['data/states/validation.jsonl',f'{adapter}/adapter_model.safetensors'])
        evaluation_files += shards(f'data/evaluations/step{step}/validation')
    rd('analyze',['analyze','--input',*evaluation_files,'--output-dir','outputs/validation'],
       evaluation_files,['outputs/validation/checkpoints.json','outputs/validation/checkpoints.csv'])
    status('seed41_validation_complete',note='400 training steps and ten fixed-panel validation checkpoints complete. Full MATH test and extra seeds are not included.')

if __name__=='__main__':
    try:
        main()
    except BaseException as error:
        old = json.loads((ROOT/'status.json').read_text()) if (ROOT/'status.json').exists() else {}
        status('failed',failed_stage=old.get('stage'),error=str(error))
        raise
