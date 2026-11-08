from collections import defaultdict
from pathlib import Path
import numpy as np
from .io import load_rows,save_json,unique
from .evaluate import write_csv
from .statistics import cluster_interval


def legacy_audit(rows,output):
    groups=defaultdict(list)
    for r in rows:groups[r['checkpoint']].append(r)
    if 'base' not in groups:raise ValueError('Missing base panel')
    base=unique(groups['base']);ids=sorted(base);result=[]
    for name,rs in sorted(groups.items(),key=lambda z:0 if z[0]=='base' else int(z[0].replace('step',''))):
        index=unique(rs)
        if set(index)!=set(base):raise ValueError('Checkpoint coverage mismatch')
        rs=[index[i] for i in ids]
        if any(r['initial_correct']!=base[r['id']]['initial_correct'] or r['problem_id']!=base[r['id']]['problem_id'] for r in rs):raise ValueError('Legacy initial states differ')
        y=np.array([r['initial_correct'] for r in rs],float)
        forced=np.array([np.mean([s['correct'] for s in r['reflect_samples']]) for r in rs])
        retry=np.array([np.mean([s['correct'] for s in r['retry_samples']]) for r in rs])
        nat=np.array([r['natural_correct'] for r in rs],float);ref=np.array([r['natural_action']=='REFLECT' for r in rs],float)
        ci=cluster_interval(nat-np.array([base[i]['natural_correct'] for i in ids]),[r['problem_id'] for r in rs])
        result.append({'checkpoint':name,'states':len(rs),'initial_accuracy':float(y.mean()),'natural_accuracy':float(nat.mean()),
          'observed_reflection_rate':float(ref.mean()),'format_validity':float(np.mean([r['natural_valid'] for r in rs])),
          'forced_accuracy':float(forced.mean()),'retry_accuracy':float(retry.mean()),
          'repair_rate':float(forced[y==0].mean()),'damage_rate':float((1-forced[y==1]).mean()),
          'delta_vs_base':ci['mean'],'delta_ci_low':ci['ci_low'],'delta_ci_high':ci['ci_high']})
    out=Path(output);write_csv(out/'legacy_metrics.csv',result)
    save_json(out/'legacy_audit.json',{'checkpoints':result,'excluded':['p_reflect','selectivity','oracle_regret'],
        'scope':'historical natural protocol; not pooled with v5 Monte Carlo action banks'})
    return result


def summarize_interventions(rows,output):
    unique(rows,'event_id');groups=defaultdict(list)
    for r in rows:groups[(r['controller_signature'],r['executor_checkpoint'])].append(r)
    results=[]
    for (controller,executor),rs in groups.items():
        states=defaultdict(list)
        for r in rs:states[r['id']].append(r)
        values=[];gain=[];problems=[];tokens=[];valid=[]
        for es in states.values():
            if len({e['sample'] for e in es})!=len(es):raise ValueError('Duplicate intervention repetition')
            values.append(float(np.mean([e['correct'] for e in es])));gain.append(values[-1]-float(es[0]['initial_correct']))
            problems.append(es[0]['problem_id']);tokens.append(float(np.mean([e['tokens'] for e in es])));valid.append(float(np.mean([e['valid'] for e in es])))
        ci=cluster_interval(gain,problems)
        results.append({'controller':controller,'executor':executor,'states':len(states),'accuracy':float(np.mean(values)),
            'gain':ci['mean'],'gain_ci_low':ci['ci_low'],'gain_ci_high':ci['ci_high'],
            'continuation_tokens':float(np.mean(tokens)),'validity':float(np.mean(valid))})
    write_csv(Path(output)/'actual_interventions.csv',results);save_json(Path(output)/'actual_interventions.json',results);return results
