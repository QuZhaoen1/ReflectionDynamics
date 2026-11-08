from pathlib import Path
import numpy as np
from .io import read_json,save_json
from .evaluate import write_csv
from .statistics import trajectory_fit,cluster_interval


def analyze(metric_paths,output):
    rows=[read_json(p) for p in metric_paths]
    step=lambda r:0 if r['checkpoint']=='base' else int(r['checkpoint'].replace('step',''))
    rows.sort(key=step)
    if len({r['checkpoint'] for r in rows})!=len(rows):raise ValueError('Duplicate checkpoint label')
    ids=rows[0]['state_ids'];hashes=rows[0]['state_hashes']
    if any(r['state_ids']!=ids or r['state_hashes']!=hashes for r in rows):raise ValueError('Checkpoints evaluated on different states')
    out=Path(output);out.mkdir(parents=True,exist_ok=True);matrix=[]
    for actor in rows:
        p=np.array(actor['p'])
        for executor in rows:
            u=np.array(executor['u']);value=p*u
            matrix.append({'decision':actor['checkpoint'],'executor':executor['checkpoint'],
                'value_above_commit':float(value.mean()),'selectivity':float(np.mean((p-p.mean())*(u-u.mean())))})
    write_csv(out/'modular_expected_matrix.csv',matrix)
    fields=['sampled_reflection_probability','value_selectivity','repair_rate','damage_rate','utility']
    summary=[]
    for r in rows:summary.append({'checkpoint':r['checkpoint'],'step':step(r),**{k:r[k] for k in fields}})
    write_csv(out/'trajectory.csv',summary)
    fits={k:trajectory_fit([step(r) for r in rows],[r[k] for r in rows]) for k in fields if len(rows)>=7 and all(r[k] is not None for r in rows)}
    save_json(out/'descriptive_change_models.json',fits)
    p0=np.array(rows[0]['p']);u0=np.array(rows[0]['u']);p1=np.array(rows[-1]['p']);u1=np.array(rows[-1]['u'])
    decomposition={'policy_change_fixed_early_executor':cluster_interval((p1-p0)*u0,rows[0]['problem_ids']),
                   'executor_change_fixed_early_policy':cluster_interval(p0*(u1-u0),rows[0]['problem_ids']),
                   'interaction':cluster_interval((p1-p0)*(u1-u0),rows[0]['problem_ids'])}
    save_json(out/'decomposition.json',decomposition)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for field in fields:
        fig,ax=plt.subplots(figsize=(6.5,4));ax.plot([step(r) for r in rows],[r[field] for r in rows],marker='o')
        ax.set(xlabel='Optimizer step',ylabel=field.replace('_',' '));fig.tight_layout();fig.savefig(out/f'{field}.png',dpi=180);plt.close(fig)
    return {'checkpoints':len(rows),'states':len(ids),'decomposition':decomposition}


def compare_training_seeds(reports,output,draws=5000,seed=41):
    docs=[read_json(p) for p in reports]
    if len(docs)<2:raise ValueError('At least two independent training seeds required')
    if len({d['training_seed'] for d in docs})!=len(docs):raise ValueError('Duplicated training seed')
    names=set(docs[0]['metrics'])
    if any(set(d['metrics'])!=names for d in docs):raise ValueError('Metric schema mismatch')
    rng=np.random.default_rng(seed);out={'training_seeds':[d['training_seed'] for d in docs],'metrics':{}}
    for name in sorted(names):
        x=np.array([d['metrics'][name] for d in docs]);samples=x[rng.integers(len(x),size=(draws,len(x)))].mean(1)
        lo,hi=np.quantile(samples,[.025,.975]);out['metrics'][name]={'mean':float(x.mean()),'sd':float(x.std(ddof=1)),'seed_bootstrap_low':float(lo),'seed_bootstrap_high':float(hi)}
    out['scope']='seed-level uncertainty; few seeds give coarse intervals; does not substitute for held-out tasks'
    save_json(output,out);return out
