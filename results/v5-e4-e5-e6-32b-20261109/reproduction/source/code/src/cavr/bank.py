import numpy as np
from collections import defaultdict,Counter
from .io import unique
from .data import state_hash


def targets(states,events,bank,checkpoint,min_reps=4):
    unique(events,'event_id');lookup=unique(states);by=defaultdict(lambda:defaultdict(list));identities=set()
    for e in events:
        if e['bank']!=bank or e['checkpoint']!=checkpoint:raise ValueError('Mixed bank or checkpoint')
        if e['id'] not in lookup:raise ValueError('Event outside state panel')
        s=lookup[e['id']]
        if e['state_hash']!=state_hash(s) or e['initial_correct']!=s['initial_correct']:raise ValueError('Event/state mismatch')
        identities.add(e['executor']);by[e['id']][e['arm']].append(e)
    if len(identities)!=1:raise ValueError('Mixed executor/protocol identities')
    labels=[];ns=[];token=[];prompt=[];latency=[]
    for s in states:
        a=by[s['id']]['reflect'];b=by[s['id']]['retry']
        if len(a)<min_reps:raise ValueError(f"Insufficient reflection samples: {s['id']}")
        for arm in (a,b):
            keys=[e['sample'] for e in arm]
            if len(keys)!=len(set(keys)):raise ValueError('Duplicate arm repetition')
        r=np.mean([bool(e['correct']) for e in a]);y=float(s['initial_correct'])
        rb=np.mean([bool(e['correct']) for e in b]) if b else 0.
        ca=np.mean([e['tokens'] for e in a]);cb=np.mean([e['tokens'] for e in b]) if b else 0.
        labels.append([y,r,1-r,r-y,np.log1p(ca),rb,np.log1p(cb)])
        ns.append([1,len(a),len(a),1,1,len(b),int(bool(b))]);token.append([ca,cb])
        prompt.append([np.mean([e.get('prompt_tokens',0) for e in arm]) if arm else 0. for arm in (a,b)])
        latency.append([np.mean([e.get('seconds',0) for e in arm]) if arm else 0. for arm in (a,b)])
    count=Counter(s['problem_id'] for s in states);w=np.array([1/count[s['problem_id']] for s in states],np.float32);w/=w.mean()
    return {'y':np.array(labels,np.float32),'n':np.array(ns,np.float32),'weights':w,'tokens':np.array(token),
            'prompt_tokens':np.array(prompt),'seconds':np.array(latency),'executor':next(iter(identities)),'events':by}
