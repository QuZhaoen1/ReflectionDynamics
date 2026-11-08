"""Eight immutable logical shards for probe and draft generation."""
import argparse
from pathlib import Path
from cavr.io import read_json, save_json, digest, seed_for, sha256, load_rows, adapter_hash
from cavr.data import state_hash, validate_states, unique
from cavr.protocol import grade, action_event
from batch_store import BatchStore
from engine import Engine


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--job', required=True)
    p.add_argument('--shard', type=int, required=True)
    a = p.parse_args()
    job = read_json(a.job)
    cfg = read_json(job['config'])
    assert 0 <= a.shard < 8
    cfg['checkpoints'] = {job['checkpoint']: job['adapter']}
    cfg['adapter_sha256'] = {job['checkpoint']: job['adapter_sha256']}
    engine = Engine(cfg)
    identity = engine.select(job['checkpoint'])
    code_runner = None
    if job.get('code_sandbox'):
        from judge_backend import selected_runner
        code_runner = selected_runner(cfg)
    for item in job['items']:
        rows = load_rows(item['input'])
        if sha256(item['input']) != item['input_sha256']:
            raise ValueError('Input cohort changed')
        unique(rows)
        if item['kind'] != 'draft':
            validate_states(rows)
        panel = sorted((r for r in rows if seed_for(r['id']) % 8 == a.shard), key=lambda r:r['id'])
        lookup = {r['id']:r for r in panel}
        arms = ['draft'] if item['kind'] == 'draft' else item['arms']
        plan = [{'id':r['id'], 'arm':arm, 'sample':k,
                 'seed':seed_for(41, 'draft', r['id'], k) if arm=='draft'
                     else seed_for(41, item['bank'], r['id'], arm, k)}
                for r in panel for arm in arms for k in range(item['reps'])]
        size = 256 if arms == ['action'] else 128
        spec = {'schema':5, 'executor':identity, 'checkpoint':job['checkpoint'],
                'bank':item['bank'], 'input_sha256':item['input_sha256'],
                'states':digest(panel), 'arms':arms, 'reps':item['reps'],
                'shard':a.shard, 'shards':8, 'seed':41, 'plan_digest':digest(plan),
                'batch_size':size, 'sandbox':getattr(code_runner,'image',None)}
        dest = Path(item['prefix']+f'-{a.shard}.jsonl')
        status = Path(str(dest)+'.progress.json')
        with BatchStore(dest, spec, plan, size) as store:
            for index, requests in enumerate(store.batches):
                if store.read(index) is not None:
                    continue
                save_json(status, {'stage':'generating','batch':index,'batches':len(store.batches)})
                outputs = engine.generate(requests, lookup)
                results = []
                for req, result in zip(requests, outputs):
                    s = lookup[req['id']]
                    if req['arm'] == 'draft':
                        g = grade(s,result['text'],'retry',code_runner)
                        row = {**s, **result, **g, 'event_id':store.key(req),
                               'sample':req['sample'], 'id':f"{s['id']}-d{req['sample']}",
                               'problem_id':s['id'], 'initial_reasoning':result['text'],
                               'initial_answer':g['answer'], 'initial_correct':g['correct'],
                               'draft_model':cfg['model_path'], 'draft_adapter_sha256':None}
                    else:
                        row = {'event_id':store.key(req), **req, 'problem_id':s['problem_id'],
                               'split':s['split'], 'state_hash':state_hash(s),
                               'checkpoint':job['checkpoint'], 'executor':digest(identity),
                               'bank':item['bank'], 'initial_correct':s['initial_correct'], **result}
                        if req['arm'] in ('action','natural'):
                            row['action'] = action_event(result['text'],result['ended'])
                        if req['arm'] != 'action':
                            row.update(grade(s,result['text'],req['arm'],code_runner))
                    results.append(row)
                store.commit(index,results)
            store.complete()
        save_json(status, {'stage':'complete','events':len(plan)})


if __name__=='__main__':
    main()
