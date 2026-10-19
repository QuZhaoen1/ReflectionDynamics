import csv
import io
from collections import defaultdict
from pathlib import Path
import numpy as np
from .io import unique,save_json,save_rows,atomic_text,digest
from .data import state_hash
from .bank import targets
from .policy import decisions
from .statistics import cluster_interval
from .protocol import event_summary


def write_csv(path,rows):
    if not rows:raise ValueError('Empty result table')
    f=io.StringIO();w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows);atomic_text(path,f.getvalue())


def offline(states,events,predictions,output,bank,checkpoint,budgets=(.1,.3,.5,.7),cost_weights=(0.,.05,.2),
            risk=0.,bootstrap=3000,seed=41,allow_drift=False):
    ss=unique(states);pp=unique(predictions)
    if set(ss)!=set(pp):raise ValueError('Predictions must cover the exact state panel')
    for r in states:
        p=pp[r['id']]
        if p['state_hash']!=state_hash(r):raise ValueError('Prediction/state mismatch')
        if p['seen_by_controller']:raise ValueError('Cannot report controller training/development states as held-out results')
    b=targets(states,events,bank,checkpoint)
    mismatch=any(p['executor']!=b['executor'] for p in predictions)
    if mismatch and not allow_drift:raise ValueError('Controller was trained for another executor; use --allow-drift for an explicitly labeled transfer/no-refresh experiment')
    if bank=='fit':raise ValueError('Use a separately seeded score bank for held-out policy evaluation')
    y=b['y'][:,0];r=b['y'][:,1];retry=b['y'][:,5];problems=[s['problem_id'] for s in states]
    preds=[pp[s['id']] for s in states];out=Path(output);out.mkdir(parents=True,exist_ok=True)
    summary=[];state_rows=[]
    def add(name,gate,lam,budget=None,extra=None,all_retry=False):
        gate=np.asarray(gate,float);accuracy=retry if all_retry else y+gate*(r-y)
        tokens=b['tokens'][:,1] if all_retry else gate*b['tokens'][:,0]
        prefill=b['prompt_tokens'][:,1] if all_retry else gate*b['prompt_tokens'][:,0]
        seconds=b['seconds'][:,1] if all_retry else gate*b['seconds'][:,0]
        feature_tokens=np.array([p['feature_prefill_tokens'] for p in preds]) if extra else np.zeros(len(states))
        feature_seconds=np.array([p['feature_seconds'] for p in preds]) if extra else np.zeros(len(states))
        utility=accuracy-lam*tokens/1000.
        ci=cluster_interval(accuracy-y,problems,bootstrap,seed)
        summary.append({'policy':name,'checkpoint':checkpoint,'controller_mode':preds[0]['mode'],'cost_weight':lam,
            'budget_fraction':budget,'states':len(states),'reflection_rate':float(gate.mean()) if not all_retry else 0.,
            'accuracy':float(accuracy.mean()),'accuracy_gain':ci['mean'],'gain_ci_low':ci['ci_low'],'gain_ci_high':ci['ci_high'],
            'continuation_tokens':float(tokens.mean()),'executor_prefill_tokens':float(prefill.mean()),
            'feature_prefill_tokens':float(feature_tokens.mean()),'sequential_seconds':float((seconds+feature_seconds).mean()),
            'reward_minus_continuation_cost':float(utility.mean()),'executor_drift':mismatch})
        for i,s in enumerate(states):state_rows.append({'id':s['id'],'problem_id':s['problem_id'],'policy':name,'checkpoint':checkpoint,
            'cost_weight':lam,'budget_fraction':budget,'expected_correct':float(accuracy[i]),'reflect':float(gate[i]),
            'continuation_tokens':float(tokens[i]),'gain_vs_commit':float(accuracy[i]-y[i])})
    for lam in cost_weights:
        add('commit',np.zeros(len(states)),lam)
        add('always_reflect',np.ones(len(states)),lam)
        if np.all(b['n'][:,5]>=4):add('independent_retry',np.zeros(len(states)),lam,all_retry=True)
        plan=decisions(preds,lam,risk)
        add('cavr_threshold' if preds[0]['mode']=='cavr' else preds[0]['mode']+'_threshold',[p['action']=='REFLECT' for p in plan],lam,extra=True)
        for budget in budgets:
            for method in (None,'confidence','random','no_damage','no_cost'):
                if preds[0]['mode']!='cavr' and method not in (None,'random'):continue
                plan=decisions(preds,lam,risk,budget,method,seed)
                name=(method or preds[0]['mode'])+f'@{budget:g}'
                add(name,[p['action']=='REFLECT' for p in plan],lam,budget,extra=method!='random')
    oracle=y+(1-y)*r
    save_json(out/'label_oracle.json',{'correctness_label_oracle_accuracy':float(oracle.mean()),
        'gain_vs_commit':float((oracle-y).mean()),'definition':'commit on known-correct states; reflect otherwise; zero cost; labels unavailable to a deployed policy',
        'not_a_learned_result':True})
    write_csv(out/'policies.csv',summary);save_rows(out/'policy_states.jsonl',state_rows)
    save_json(out/'report.json',{'schema':5,'bank':bank,'checkpoint':checkpoint,'executor':b['executor'],
        'controller_signature':preds[0]['controller_signature'],'executor_drift':mismatch,
        'ci_scope':'paired original-problem cluster intervals conditional on recorded outcome samples; no training-seed variance',
        'budget_scope':'matched number of reflection calls, not matched total FLOPs; prefill and measured sequential latency reported separately',
        'main_decision_rule':'fixed cost/risk threshold; budget sweep is reported without choosing a best test point',
        'policies':summary})
    return summary


def action_audit(states,events_a,events_b,output):
    unique(events_a,'event_id');unique(events_b,'event_id')
    allowed=set(unique(states))
    if not events_a or not events_b:raise ValueError('Empty action audit bank')
    if len({(e['checkpoint'],e['executor']) for e in events_a+events_b})!=1:raise ValueError('Action audit requires the same executor and decoding protocol')
    if {e['id'] for e in events_a}!=allowed or {e['id'] for e in events_b}!=allowed:raise ValueError('Action audit must cover exactly the requested panel')
    if len({e['bank'] for e in events_a})!=1 or len({e['bank'] for e in events_b})!=1 or events_a[0]['bank']==events_b[0]['bank']:raise ValueError('Use two distinct action sampling banks')
    if {(e['id'],e['sample'],e['seed']) for e in events_a}&{(e['id'],e['sample'],e['seed']) for e in events_b}:raise ValueError('Action audit banks must have independent seeds')
    groups=[]
    for events in (events_a,events_b):
        d=defaultdict(list)
        for e in events:
            if e['arm']!='action':raise ValueError('Expected action events only')
            d[e['id']].append(e)
        groups.append(d)
    rows=[]
    for s in states:
        if not all(s['id'] in g for g in groups):raise ValueError('Incomplete action panel')
        for g in groups:
            if any(e['state_hash']!=state_hash(s) for e in g[s['id']]):raise ValueError('Action state mismatch')
        a,b=[event_summary([e['action'] for e in g[s['id']]]) for g in groups]
        rows.append({'id':s['id'],'problem_id':s['problem_id'],'a':a,'b':b,'delta':a['p_reflect']-b['p_reflect']})
    ci=cluster_interval([r['delta'] for r in rows],[r['problem_id'] for r in rows])
    result={'difference_A_minus_B':ci,'mean_invalid_A':float(np.mean([r['a']['p_invalid'] for r in rows])),
            'mean_censored_A':float(np.mean([r['a']['p_censored'] for r in rows])),
            'method':'independent Monte Carlo prefix banks; not normalized two-string likelihood',
            'state_estimates':rows}
    save_json(output,result);return result


def selection_metrics(states,outcome_events,action_events,bank,checkpoint,cost_weight=0.):
    unique(action_events,'event_id')
    if {e['id'] for e in action_events}!=set(unique(states)):raise ValueError('Action bank must cover exactly the requested panel')
    b=targets(states,outcome_events,bank,checkpoint);groups=defaultdict(list)
    for e in action_events:
        if e['arm']!='action' or e['checkpoint']!=checkpoint or e['executor']!=b['executor']:raise ValueError('Wrong action checkpoint/arm')
        groups[e['id']].append(e)
    p=[]
    for s in states:
        es=groups[s['id']]
        if not es or any(e['state_hash']!=state_hash(s) for e in es):raise ValueError('Action/state mismatch')
        p.append(event_summary([e['action'] for e in es])['p_reflect'])
    p=np.asarray(p);u=b['y'][:,1]-b['y'][:,0]-cost_weight*b['tokens'][:,0]/1000
    return {'checkpoint':checkpoint,'sampled_reflection_probability':float(p.mean()),
            'value_selectivity':float(np.mean((p-p.mean())*(u-u.mean()))),
            'repair_rate':float(b['y'][b['y'][:,0]==0,1].mean()) if (b['y'][:,0]==0).any() else None,
            'damage_rate':float(b['y'][b['y'][:,0]==1,2].mean()) if (b['y'][:,0]==1).any() else None,
            'utility':float(u.mean()),'p':p.tolist(),'u':u.tolist(),
            'state_ids':[s['id'] for s in states],'problem_ids':[s['problem_id'] for s in states],
            'state_hashes':[state_hash(s) for s in states],'scope':'modular action selection followed by canonical forced-reflection continuation'}
