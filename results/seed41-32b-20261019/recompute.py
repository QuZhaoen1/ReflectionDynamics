"""Rebuild numeric reports without GPUs, prompts, model outputs or action likelihoods.

python recompute.py [--input-dir DIRECTORY]; requires numpy, matplotlib.
CIs are paired problem-cluster percentile bootstraps (5000 draws, seed 41).
They are conditional on observed generations and do not quantify training-seed variance.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser(); p.add_argument('--input-dir',type=Path,default=Path(__file__).resolve().parent)
    root=p.parse_args().input_dir
    groups=defaultdict(list)
    for line in (root/'observations.jsonl').read_text().splitlines():
        row=json.loads(line); groups[row['checkpoint']].append(row)
    order=['base']+[f'step{s}' for s in range(40,401,40)]
    assert set(groups)==set(order)
    base=sorted(groups['base'],key=lambda r:r['id']); ids=[r['id'] for r in base]
    assert len(ids)==458 and len(set(ids))==458
    problems=sorted({r['problem_id'] for r in base}); assert len(problems)==295
    cluster=np.array([problems.index(r['problem_id']) for r in base])
    counts=np.bincount(cluster)
    sampled=np.random.default_rng(41).integers(len(problems),size=(5000,len(problems)))
    weights=np.array([np.bincount(s,minlength=len(problems)) for s in sampled])
    den=weights@counts
    initial=np.array([r['initial_correct'] for r in base],float)
    baseacc=np.array([r['natural_correct'] for r in base],float)
    def interval(vals):
        totals=np.bincount(cluster,weights=vals,minlength=len(problems))
        return np.quantile(weights@totals/den,[.025,.975]).tolist()
    summary=[]; paired=[]
    for ck in order:
        rows=sorted(groups[ck],key=lambda r:r['id'])
        assert [r['id'] for r in rows]==ids
        assert all(r['problem_id']==b['problem_id'] and r['initial_correct']==b['initial_correct'] for r,b in zip(rows,base))
        arr=lambda key:np.array([r[key] for r in rows],float)
        forced=np.array([np.mean([s['correct'] for s in r['reflect_samples']]) for r in rows])
        retry=np.array([np.mean([s['correct'] for s in r['retry_samples']]) for r in rows])
        natural=arr('natural_correct'); reflect=np.array([r['natural_action']=='REFLECT' for r in rows])
        low,high=interval(natural)
        m={'checkpoint':ck,'states':len(rows),'unique_problems':len(problems),'initial_accuracy':float(initial.mean()),
           'natural_correct_count':int(natural.sum()),'natural_accuracy':float(natural.mean()),
           'natural_accuracy_ci_low':low,'natural_accuracy_ci_high':high,
           'observed_reflection_count':int(reflect.sum()),'observed_reflection_rate':float(reflect.mean()),
           'natural_format_rate':float(arr('natural_valid').mean()),
           'forced_format_rate':float(np.mean([s['valid'] for r in rows for s in r['reflect_samples']])),
           'forced_reflection_accuracy':float(forced.mean()),'independent_retry_accuracy':float(retry.mean()),
           'repair_rate':float(forced[initial==0].mean()),'damage_rate':float((1-forced[initial==1]).mean()),
           'mean_utility':float((forced-initial).mean()),'natural_tokens':float(arr('natural_tokens').mean()),
           'reflection_tokens':float(np.mean([s['tokens'] for r in rows for s in r['reflect_samples']])),
           'reflection_rate_initial_wrong':float(reflect[initial==0].mean()),
           'reflection_rate_initial_correct':float(reflect[initial==1].mean())}
        summary.append(m)
        low,high=interval(natural-baseacc)
        paired.append({'checkpoint':ck,'natural_accuracy_delta_vs_base':float((natural-baseacc).mean()),
                       'ci_low':low,'ci_high':high,'base_correct_to_wrong':int(((baseacc==1)&(natural==0)).sum()),
                       'base_wrong_to_correct':int(((baseacc==0)&(natural==1)).sum())})
    def save(name,rows):
        (root/(name+'.json')).write_text(json.dumps(rows,indent=2)+'\n')
        with (root/(name+'.csv')).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n"); w.writeheader(); w.writerows(rows)
    save('checkpoint_metrics',summary); save('paired_accuracy_vs_base',paired)
    progress=[json.loads(s) for s in (root/'training_metrics.jsonl').read_text().splitlines()]
    metrics=[r for r in progress if r['event']=='metrics' and 'reward' in r['metrics']]
    steps=[r for r in progress if r['event']=='optimizer_step']
    assert [r['step'] for r in steps]==list(range(1,401))
    blocks=[]
    for start,end in [(1,50),(51,100),(101,200),(201,300),(301,400)]:
        chosen=[r['metrics'] for r in metrics if start<=r['step']<=end]
        blocks.append({'first_step':start,'last_step':end,**{k:float(np.mean([r[k] for r in chosen])) for k in ['reward','completions/mean_length','entropy','frac_reward_zero_std']}})
    save('training_blocks',blocks)
    stats={'optimizer_steps':len(steps),'mean_step_seconds':float(np.mean([r['seconds'] for r in steps])),
           'peak_rank0_allocated_gib':max(r['peak_allocated_gib'] for r in steps),
           'bootstrap_resamples':5000,'bootstrap_seed':41,'interval_unit':'original problem cluster; state-weighted statistic',
           'excluded_metrics':['reflection_propensity','selectivity','policy_utility','frequency_matched_random','oracle_regret','cross_checkpoint_policy_execution']}
    (root/'audit.json').write_text(json.dumps(stats,indent=2)+'\n')
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x=[0]+list(range(40,401,40)); fig,axs=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for k,label in [('natural_accuracy','Natural decision'),('forced_reflection_accuracy','Forced reflection'),('independent_retry_accuracy','Independent retry')]:
        axs[0,0].plot(x,[100*r[k] for r in summary],marker='o',label=label)
    axs[0,0].axhline(100*initial.mean(),ls=':',color='gray',label='Keep initial answer')
    axs[0,0].set(ylabel='Correct (%)',title='Fixed validation panel'); axs[0,0].legend(fontsize=8)
    for k,label in [('observed_reflection_rate','Sampled REFLECT'),('natural_format_rate','Valid format')]:
        axs[0,1].plot(x,[100*r[k] for r in summary],marker='o',label=label)
    axs[0,1].set(ylabel='States (%)',title='Observed behavior (not string likelihood)'); axs[0,1].legend(fontsize=8)
    for k,label in [('repair_rate','Repair | initially wrong'),('damage_rate','Damage | initially correct')]:
        axs[1,0].plot(x,[100*r[k] for r in summary],marker='o',label=label)
    axs[1,0].set(ylabel='Forced intervention rate (%)',title='Four forced samples per state'); axs[1,0].legend(fontsize=8)
    d=np.array([r['natural_accuracy_delta_vs_base'] for r in paired])*100
    lo=np.array([r['ci_low'] for r in paired])*100; hi=np.array([r['ci_high'] for r in paired])*100
    axs[1,1].errorbar(x,d,yerr=[d-lo,hi-d],marker='o',capsize=3)
    axs[1,1].axhline(0,color='gray',ls=':'); axs[1,1].set(ylabel='Accuracy difference (percentage points)',title='Paired 95% cluster intervals vs base')
    for a in axs.flat: a.set_xlabel('Training step (0 = base)'); a.grid(alpha=.2)
    fig.suptitle('Qwen2.5-32B / seed 41 / 458 states, 295 problems\nExploratory single-seed validation; unadjusted intervals',fontsize=12)
    fig.savefig(root/'validation_curves.png',dpi=170); fig.savefig(root/'validation_curves.pdf'); plt.close(fig)
    print(json.dumps({'base':summary[0],'final':summary[-1],'paired_final':paired[-1],'audit':stats},indent=2))
if __name__=='__main__': main()
