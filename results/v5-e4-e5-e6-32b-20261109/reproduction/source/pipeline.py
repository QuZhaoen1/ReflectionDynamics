"""Persistent E4 -> E5 -> E6 chain. All generations share one declared protocol."""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from cavr.io import read_json,save_json,save_rows,load_rows,load_events,digest,sha256,adapter_hash
from cavr.data import state_hash,select_states,question_key

ROOT=Path(__file__).resolve().parent
PY='${LEGACY_ROOT}/envs/py311-torch291/bin/python'
VPY='${LEGACY_ROOT}/envs/vllm0141-test/bin/python'
CFG=ROOT/'config.json'
PANELS=('gate_train','gate_dev','gate_holdout','validation')
MODES=('cavr','confidence','direct','no_damage')


def status(stage,**kw):
    save_json(ROOT/'status.json',{'stage':stage,'time':time.time(),**kw})


def run(args,log,env=None):
    log=Path(log);log.parent.mkdir(parents=True,exist_ok=True)
    print('RUN',*[str(x) for x in args],flush=True)
    with log.open('a') as f:
        p=subprocess.Popen(list(map(str,args)),stdout=f,stderr=subprocess.STDOUT,
                           env={**os.environ,**(env or {})},start_new_session=True)
        try:
            result=p.wait()
        except BaseException:
            os.killpg(p.pid,signal.SIGTERM)
            p.wait()
            raise
    if result:
        print(log.read_text(errors='replace')[-7000:],flush=True)
        raise RuntimeError(f'Command failed ({result}): {log}')


def parallel(commands):
    jobs=[]
    try:
        for args,log,env in commands:
            Path(log).parent.mkdir(parents=True,exist_ok=True)
            f=open(log,'a')
            p=subprocess.Popen(list(map(str,args)),stdout=f,stderr=subprocess.STDOUT,
                               env={**os.environ,**env},start_new_session=True)
            jobs.append((p,f,log))
        while any(p.poll() is None for p,_,_ in jobs):
            failures=[(p.returncode,log) for p,_,log in jobs if p.poll() not in (None,0)]
            if failures:raise RuntimeError(f'Worker failed: {failures}')
            time.sleep(5)
        failures=[(p.returncode,log) for p,_,log in jobs if p.returncode]
        if failures:raise RuntimeError(f'Worker failed: {failures}')
    finally:
        for p,f,_ in jobs:
            if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
        for p,f,_ in jobs:
            try:p.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            f.close()


def cli(args,label):
    run([PY,'-m','cavr.cli',*args],ROOT/'logs'/f'{label}.log',{'CUDA_VISIBLE_DEVICES':''})


def stage(label,fn,identity):
    receipt=ROOT/'completed'/f'{label}.json'
    sig=digest(identity)
    if receipt.exists():
        old=read_json(receipt)
        if old['signature']!=sig:raise ValueError(f'Stage definition changed: {label}')
        for name,h in old['artifacts'].items():
            if sha256(name)!=h:raise ValueError(f'Completed artifact changed: {name}')
        return old['result']
    # This bound excludes our capacity reservation, which lives outside ROOT.
    used=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file() and not p.is_symlink())
    if used>88*2**30:raise RuntimeError('Pipeline reached 88 GiB working budget; checkpoint data retained')
    status(label,working_bytes=used)
    started=time.time();result,artifacts=fn()
    save_json(receipt,{'signature':sig,'started':started,'completed':time.time(),
              'result':result,'artifacts':{str(p):sha256(p) for p in artifacts}})
    return result


def immutable(path,obj):
    if path.exists():
        if read_json(path)!=obj:raise ValueError(f'Immutable definition changed: {path}')
    else:save_json(path,obj)


def generate(label,ck,adapter,items,code=False):
    job={'config':str(CFG),'checkpoint':ck,'adapter':str(adapter) if adapter else None,
         'adapter_sha256':adapter_hash(adapter),'items':items,'code_sandbox':code}
    path=ROOT/'jobs'/f'{label}.json';immutable(path,job)
    def execute():
        parallel([([VPY,ROOT/'worker.py','--job',path,'--shard',str(i)],
                   ROOT/'logs'/f'{label}-{i}.log',{'CUDA_VISIBLE_DEVICES':str(i), 'VLLM_PORT':str(19000+128*i)}) for i in range(8)])
        artifacts=[];counts={}
        for item in items:
            paths=[Path(item['prefix']+f'-{i}.jsonl') for i in range(8)]
            rows=load_events(paths)
            expected=len(load_rows(item['input']))*item['reps']*len(item.get('arms',['draft']))
            if len(rows)!=expected:raise ValueError('Generation coverage mismatch')
            counts[item['prefix']]=len(rows)
            artifacts.extend(paths)
            artifacts.extend(Path(str(p)+suffix) for p in paths for suffix in ('.manifest.json','.complete.json'))
        return counts,artifacts
    return stage(label,execute,job)


def item(panel,ck,bank,reps=16,arms=('reflect','retry'),root=ROOT):
    inp=ROOT/'states'/f'{panel}.jsonl'
    return {'kind':'probe','input':str(inp),'input_sha256':sha256(inp),'bank':bank,
            'reps':reps,'arms':list(arms),'prefix':str(root/'probes'/ck/panel/bank)}


def executor_equivalence(old,new):
    """Only audited non-inference revisions may differ; inference fields stay strict."""
    import copy
    approved={'9f96f53c6a15373bfc1ee2f5d5e60c273fb436041a2b47732a78fafc76e9f3d8',
              '6f14059311ef4f6173c680c32797f3f35fe9a3f7323d64cebc1654748e3defda',
              sha256(ROOT/'pipeline.py')}
    a,b=copy.deepcopy(old),copy.deepcopy(new)
    for obj in (a,b):
        if obj['source_sha256'].pop('pipeline.py') not in approved:
            raise ValueError('Unreviewed orchestration revision')
    if a['source_sha256'].get('code/src/cavr/train.py')!=b['source_sha256'].get('code/src/cavr/train.py'):
        # Training-only PEFT import guard; neither revision is executed by vLLM inference.
        # Adapter content hashes above remain part of the strict equality check.
        reviewed_training={'71d814a58fa2bd9ce5bc5ec8dd501ea7d1aa6470fbc167026e58d580c1d1af15',
                           '27cc2b7d45225a9a6cc27f366b80f9b6b47dbf3ebedd64a39ee81ac97689fe73'}
        for obj in (a,b):
            if obj['source_sha256'].pop('code/src/cavr/train.py') not in reviewed_training:
                raise ValueError('Unreviewed training-only revision')
    if a!=b:raise ValueError('Executor inference configuration really changed')


def compatible_predictions(controller,events,pred,out):
    import glob
    model=read_json(controller/'model.json')
    manifests=[read_json(p)['specification']['executor'] for p in glob.glob(events+'.manifest.json')]
    if len(manifests)!=8:raise ValueError('Expected eight executor manifests')
    current=manifests[0]
    if any(m!=current for m in manifests):raise ValueError('Mixed shard executor manifests')
    target=digest(current)
    if model['executor']==target:return pred,None
    references=[read_json(p)['specification']['executor'] for p in
                (ROOT/'probes'/model['executor_checkpoint']).glob('*/fit-*.jsonl.manifest.json')]
    references=[m for m in references if digest(m)==model['executor']]
    if not references:raise ValueError('Controller executor provenance missing')
    old=references[0]
    executor_equivalence(old,current)
    rows=load_rows(pred)
    for row in rows:
        if row['executor']!=model['executor']:raise ValueError('Prediction executor mismatch')
        row['controller_training_executor']=row['executor']
        row['executor']=target
    derived=out/'predictions-compatible.jsonl'
    save_rows(derived,rows)
    audit={'reason':'Only reviewed non-inference source revisions differ; inference settings and adapter hashes identical',
           'controller_training_executor':model['executor'],'evaluation_executor':target,
           'old_definition':old,'new_definition':current,
           'original_predictions_sha256':sha256(pred),'derived_predictions_sha256':sha256(derived)}
    save_json(out/'executor-equivalence.json',audit)
    return derived,audit


def report(panel,ck,controller,events,out,allow_drift=False):
    pred=out/'predictions.jsonl'
    cli(['predict','--states',ROOT/'states'/f'{panel}.jsonl','--features',ROOT/'features'/f'{panel}.npz',
         '--controller',controller,'--output',pred],digest(str(pred))[:16])
    evaluation_pred,audit=(pred,None) if allow_drift else compatible_predictions(controller,events,pred,out)
    args=['evaluate','--states',ROOT/'states'/f'{panel}.jsonl','--events',events,'--predictions',evaluation_pred,
          '--bank','score','--checkpoint',ck,'--output',out]
    if allow_drift:args.append('--allow-drift')
    cli(args,digest(str(out))[:16])
    # The upstream report assumes serial HF inference. Correct its latency
    # labels for this explicitly separate batched-vLLM experiment.
    import csv
    from cavr.evaluate import write_csv
    timing_key='amortized_executor_plus_sequential_feature_seconds'
    with (out/'policies.csv').open() as f:policies=list(csv.DictReader(f))
    for row in policies:row[timing_key]=row.pop('sequential_seconds')
    write_csv(out/'policies.csv',policies)
    doc=read_json(out/'report.json')
    if audit:doc['executor_equivalence']=audit
    for row in doc['policies']:row[timing_key]=row.pop('sequential_seconds')
    doc['budget_scope']='matched reflection calls, not matched total FLOPs; prefill tokens separate'
    doc['latency_scope']='executor time is amortized batch wall time; feature time is sequential; their sum is not measured sequential deployment latency'
    save_json(out/'report.json',doc)
    return [p for p in out.rglob('*') if p.is_file()]


def frozen(ck,adapter,modes=MODES):
    generate('bank-'+ck,ck,adapter,[item(p,ck,'fit' if p in PANELS[:2] else 'score') for p in PANELS])
    for mode in modes:
        model=ROOT/'controllers'/ck/mode
        def fit_and_score():
            cli(['fit','--train-states',ROOT/'states/gate_train.jsonl','--dev-states',ROOT/'states/gate_dev.jsonl',
                 '--train-features',ROOT/'features/gate_train.npz','--dev-features',ROOT/'features/gate_dev.npz',
                 '--train-events',str(ROOT/'probes'/ck/'gate_train/fit-*.jsonl'),
                 '--dev-events',str(ROOT/'probes'/ck/'gate_dev/fit-*.jsonl'),
                 '--checkpoint',ck,'--mode',mode,'--output',model,'--resume'],f'fit-{ck}-{mode}')
            artifacts=[p for p in model.rglob('*') if p.is_file()]
            for panel in PANELS[2:]:
                artifacts+=report(panel,ck,model,str(ROOT/'probes'/ck/panel/'score-*.jsonl'),
                                  ROOT/'reports/frozen'/ck/mode/panel)
            return {'controller':str(model)},artifacts
        stage(f'controller-{ck}-{mode}',fit_and_score,{'checkpoint':ck,'adapter':adapter_hash(adapter),'mode':mode})


def train(variant,rnd,previous,selected,seed,steps=40,mode='executor'):
    from cavr.checkpoint import latest_complete,inspect_checkpoint,export_adapter
    out=ROOT/'runs'/variant/(f'round{rnd}' if rnd else 'train')
    adapter=ROOT/'adapters'/variant/(f'round{rnd}' if rnd else 'final')
    label=f'train-{variant}-{rnd}'
    spec={'states':sha256(selected),'seed':seed,'steps':steps,'mode':mode,'initial':adapter_hash(previous),
          'profile':sha256(ROOT/'profile.yaml')}
    def execute():
        st=read_json(out/'status.json') if (out/'status.json').exists() else {}
        if not st.get('success'):
            checkpoints=list(out.glob('checkpoint-*'))
            # A fully saved final checkpoint may survive a stop before status.json.
            if (out/f'checkpoint-{steps}').exists():
                info=inspect_checkpoint(out/f'checkpoint-{steps}',8)
                import math
                trainer_state=read_json(out/f'checkpoint-{steps}'/'trainer_state.json')
                norms=[float(r['grad_norm']) for r in trainer_state['log_history'] if 'grad_norm' in r]
                if not norms or not all(math.isfinite(x) for x in norms) or not any(x>0 for x in norms):
                    raise ValueError('Final checkpoint lacks verified finite nonzero gradients')
                save_json(out/'status.json',{'success':True,'global_step':steps,
                          'verified_final_checkpoint_after_interruption':True,**info})
            else:
                args=[PY,'-m','torch.distributed.run','--standalone','--nproc_per_node=8','-m','cavr.train',
                      '--config',ROOT/'profile.yaml','--states',selected,'--output',out,
                      '--mode',mode,'--seed',str(seed),'--max-steps',str(steps),
                      '--cost-weight','.05' if mode=='executor' else '0']
                if previous:args+=['--init-adapter',previous]
                if checkpoints:latest_complete(out,8);args+=['--resume','auto']
                run(args,ROOT/'logs'/f'{label}.log',{'CUDA_VISIBLE_DEVICES':'0,1,2,3,4,5,6,7'})
        st=read_json(out/'status.json')
        if not st['success'] or st['global_step']!=steps:raise ValueError('Incomplete training')
        inspect_checkpoint(out/f'checkpoint-{steps}',8)
        export_adapter(out/f'checkpoint-{steps}',adapter)
        return {'adapter':str(adapter),'adapter_sha256':adapter_hash(adapter)},[out/'status.json',*adapter.glob('*')]
    result=stage(label,execute,spec)
    return Path(result['adapter'])


def online(variant,origin):
    current='step400';adapter=origin
    for rnd in range(1,4):
        head='step400' if variant=='no_refresh' else current
        if rnd>1 and variant!='no_refresh':frozen(current,adapter,('cavr',))
        selected=ROOT/'online'/variant/f'round{rnd}-states.jsonl'
        def route_data():
            if variant=='uniform_executor':save_rows(selected,load_rows(ROOT/'states/gate_train.jsonl'))
            else:
                from cavr.controller import predict
                from cavr.route import route
                states=load_rows(ROOT/'states/gate_train.jsonl')
                predictions=predict(states,ROOT/'features/gate_train.npz',ROOT/'controllers'/head/'cavr',True)
                rows,receipt=route(states,predictions,seed=41+rnd)
                save_rows(selected,rows);save_json(str(selected)+'.routing.json',receipt)
            return {'states':len(load_rows(selected))},[selected]
        stage(f'route-{variant}-{rnd}',route_data,{'head':head,'seed':41+rnd,'variant':variant})
        adapter=train(variant,rnd,adapter,selected,41+rnd)
        current=f'{variant}-round{rnd}'
    if variant=='no_refresh':
        generate('bank-'+current,current,adapter,[item(p,current,'score') for p in PANELS[2:]])
        def evaluate_stale():
            artifacts=[]
            for mode in MODES:
                for panel in PANELS[2:]:
                    artifacts+=report(panel,current,ROOT/'controllers/step400'/mode,
                                      str(ROOT/'probes'/current/panel/'score-*.jsonl'),
                                      ROOT/'reports/frozen'/current/mode/panel,True)
            return {'drift_allowed':True,'head':'step400'},artifacts
        stage('stale-final-evaluation',evaluate_stale,{'adapter':adapter_hash(adapter)})
    else:frozen(current,adapter)
    return current,adapter


def features(kind):
    import numpy as np
    from cavr.features import load
    rows=load_rows(ROOT/'states'/f'{kind}.jsonl')
    target=ROOT/'features'/f'{kind}.npz'
    def execute():
        parallel([([PY,ROOT/'feature_worker.py','--config',CFG,'--states',ROOT/'states'/f'{kind}.jsonl',
                    '--output',ROOT/'features'/f'{kind}-part{i}.npz','--shard',i],
                   ROOT/'logs'/f'features-{kind}-{i}.log',{'CUDA_VISIBLE_DEVICES':f'{2*i},{2*i+1}'}) for i in range(4)])
        parts=[np.load(ROOT/'features'/f'{kind}-part{i}.npz',allow_pickle=False) for i in range(4)]
        encoders={str(z['encoder_id']) for z in parts}
        if len(encoders)!=1:raise ValueError('Mixed encoders')
        eid=encoders.pop();ids=np.concatenate([z['ids'] for z in parts]);order={v:i for i,v in enumerate(ids)}
        indexes=[order[r['id']] for r in rows]
        arrays={k:np.concatenate([z[k] for z in parts])[indexes] for k in ('x','ids','state_hashes','costs')}
        with target.with_suffix('.tmp').open('wb') as f:np.savez_compressed(f,**arrays,encoder_id=np.array(eid))
        target.with_suffix('.tmp').replace(target)
        enc=read_json(CFG)['feature_encoder']
        save_json(str(target)+'.json',{'encoder':enc,'encoder_id':eid,'states':len(rows),
                  'dimensions':arrays['x'].shape[1],'signature':digest([enc,[(r['id'],state_hash(r)) for r in rows]])})
        load(target,rows)
        return {'states':len(rows)},[target,Path(str(target)+'.json')]
    stage('features-'+kind,execute,{'states':sha256(ROOT/'states'/f'{kind}.jsonl')})


def ood(kind,executors):
    split={'math':'test','gsm8k':'test','aime':'ood','humaneval':'ood_code'}[kind]
    questions=ROOT/'corpus'/kind/f'{split}.jsonl'
    it={'kind':'draft','input':str(questions),'input_sha256':sha256(questions),
        'bank':'draft','reps':2,'prefix':str(ROOT/'drafts'/kind/'base')}
    generate('draft-'+kind,'base',None,[it],kind=='humaneval')
    states=ROOT/'states'/f'{kind}.jsonl'
    def form_states():
        drafts=load_events(str(ROOT/'drafts'/kind/'base-*.jsonl'))
        rows=select_states(drafts)
        train_keys={question_key(r) for p in PANELS[:2] for r in load_rows(ROOT/'states'/f'{p}.jsonl')}
        overlap=[r['id'] for r in rows if question_key(r) in train_keys]
        if overlap:raise ValueError(f'Test/train question overlap: {len(overlap)}')
        save_rows(states,rows)
        return {'drafts':len(drafts),'valid_states':len(rows),'selection':'all parseable drafts, independent of reflection'},[states]
    stage('states-'+kind,form_states,{'questions':sha256(questions)})
    features(kind)
    for ck,adapter,head in executors:
        generate(f'ood-{kind}-{ck}',ck,adapter,[item(kind,ck,'score')],kind=='humaneval')
        def evaluate():
            artifacts=[]
            for mode in MODES:
                artifacts+=report(kind,ck,ROOT/'controllers'/head/mode,
                                  str(ROOT/'probes'/ck/kind/'score-*.jsonl'),
                                  ROOT/'reports/ood'/kind/ck/mode,head!=ck)
            return {'executor':ck,'controller':head,'locked_before_test':True},artifacts
        stage(f'ood-report-{kind}-{ck}',evaluate,{'adapter':adapter_hash(adapter),'head':head,'kind':kind})


def seeds():
    from cavr.checkpoint import export_adapter
    summaries=[]
    for seed in (41,73,107):
        variant=f'standard_v5_seed{seed}'
        train(variant,0,None,ROOT/'states/gate_train.jsonl',seed,400,'standard')
        sr=ROOT/'E6'/variant;metrics=[]
        for step in range(0,401,40):
            ck=f'step{step}' if step else 'base'
            adapter=None
            if step:
                adapter=ROOT/'adapters'/variant/ck
                export_adapter(ROOT/'runs'/variant/'train'/f'checkpoint-{step}',adapter)
            its=[item('validation',ck,b,16 if b=='score' else 64,
                      ('reflect','retry') if b=='score' else ('action',),sr) for b in ('actionA','actionB','score')]
            generate(f'e6-{seed}-{ck}',ck,adapter,its)
            metric=sr/'reports'/f'{ck}-selection.json';metrics.append(str(metric))
            def analyze():
                prefix=sr/'probes'/ck/'validation'
                audit=sr/'reports'/f'{ck}-audit.json'
                cli(['audit-actions','--states',ROOT/'states/validation.jsonl',
                     '--events-a',str(prefix/'actionA-*.jsonl'),'--events-b',str(prefix/'actionB-*.jsonl'),
                     '--output',audit],f'audit-{seed}-{ck}')
                cli(['selection','--states',ROOT/'states/validation.jsonl','--events',str(prefix/'score-*.jsonl'),
                     '--actions',str(prefix/'actionA-*.jsonl'),'--checkpoint',ck,'--bank','score',
                     '--output',metric],f'selection-{seed}-{ck}')
                return {'checkpoint':ck},[audit,metric]
            stage(f'e6-analysis-{seed}-{ck}',analyze,{'adapter':adapter_hash(adapter)})
        def summarize():
            cli(['dynamics','--metrics',*metrics,'--output',sr/'reports/dynamics'],f'dynamics-{seed}')
            last=read_json(metrics[-1]);path=sr/'reports/seed-summary.json'
            fields=('sampled_reflection_probability','value_selectivity','repair_rate','damage_rate','utility')
            save_json(path,{'training_seed':seed,'metrics':{k:last[k] for k in fields}})
            return {'summary':str(path)},[p for p in (sr/'reports').rglob('*') if p.is_file()]
        result=stage(f'e6-summary-{seed}',summarize,{'metrics':[sha256(p) for p in metrics]})
        summaries.append(result['summary'])
    cli(['seeds','--reports',*summaries,'--output',ROOT/'reports/E6-independent-seeds.json'],'E6-aggregate')


def main():
    if sha256(CFG)!=(ROOT/'config.sha256').read_text().strip():
        raise ValueError('Pipeline configuration changed after preflight')
    cfg=read_json(CFG)
    for name,h in cfg['source_sha256'].items():
        if sha256(ROOT/name)!=h:raise ValueError(f'Source changed: {name}')
    for name,h in cfg['input_sha256'].items():
        if sha256(ROOT/name)!=h:raise ValueError(f'Input changed: {name}')
    import torch
    if torch.cuda.device_count()!=8:raise RuntimeError('Exactly eight GPUs required')
    reserve=Path(cfg['reserve'])
    if not (ROOT/'capacity-release.json').exists():
        if reserve.stat().st_blocks*512<144*2**30:raise RuntimeError('144 GiB reservation missing')
        with reserve.open('r+b') as f:f.truncate(48*2**30);os.fsync(f.fileno())
        save_json(ROOT/'capacity-release.json',{'working_budget_gib':96,'retained_reserve_gib':48,
                  'checkpoint_projection_gib':31,'policy':'stop before 88 GiB; never delete others files'})
    from judge_backend import preflight
    selection=preflight(cfg)
    save_json(ROOT/'sandbox-preflight.json',{'success':True,**selection})
    origin=cfg['initial_adapter']
    if adapter_hash(origin)!=cfg['initial_adapter_hash']:raise ValueError('Original adapter changed')
    frozen('step400',origin)
    finals=[online(v,origin) for v in ('cavr_online','no_refresh','uniform_executor')]
    frozen('base',None)
    executors=[('base',None,'base'),('step400',origin,'step400')]+[
        (ck,adapter,'step400' if ck.startswith('no_refresh') else ck) for ck,adapter in finals]
    lock={'executors':[[ck,str(a) if a else None,head] for ck,a,head in executors],
          'controller_hashes':{str(p.relative_to(ROOT)):sha256(p) for p in (ROOT/'controllers').rglob('*') if p.is_file()},
          'budgets':[.1,.3,.5,.7],'primary_budget':.3,'cost_weights':[0,.05,.2],'primary_cost_weight':.05,
          'primary_checkpoint':'cavr_online-round3','selection':'predeclared round3, never choose on test',
          'latency':'vLLM amortized batch latency, not sequential deployment latency'}
    immutable(ROOT/'E5-policy-lock.json',lock)
    for kind in ('math','aime','gsm8k','humaneval'):ood(kind,executors)
    seeds()
    status('all_complete',E4='three variants x three rounds x 40 steps',E5='four independent datasets x five executors',
           E6='three independent 32B seeds x 400 steps, full checkpoint trajectories',
           completed_stages=len(list((ROOT/'completed').glob('*.json'))))


if __name__=='__main__':
    try:main()
    except BaseException as e:
        previous=read_json(ROOT/'status.json') if (ROOT/'status.json').exists() else {}
        status('failed_or_interrupted',failed_stage=previous,error=repr(e))
        raise
