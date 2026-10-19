import re
import time
import numpy as np
from pathlib import Path
from .io import digest,seed_for,save_json,read_json
from .data import visible,state_hash,validate_states
from .protocol import decision_prompt


def hash_feature(r,dim=1024):
    words=re.findall(r'\w+|[^\w\s]','\n'.join(visible(r).values()).lower());v=np.zeros(dim,dtype=np.float32)
    for n in (1,2):
        for i in range(len(words)-n+1):
            h=seed_for(n,' '.join(words[i:i+n]));v[(h//2)%dim]+=1. if h%2 else -1.
    norm=np.linalg.norm(v)
    return v/norm if norm else v


def extract(states,path,mode='hidden',backend=None,dim=1024):
    validate_states(states,fixtures=True);p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    encoder={'mode':mode,'dim':dim} if mode=='hash' else {'mode':mode,'backbone':backend.identity,'pool':'last+mean'}
    eid=digest(encoder);signature=digest([encoder,[(r['id'],state_hash(r)) for r in states]])
    meta=Path(str(p)+'.json')
    if p.exists():
        m=read_json(meta)
        if m['signature']!=signature:raise ValueError('Feature cache identity mismatch')
        return m
    parts=p.parent/(p.name+'.parts');parts.mkdir(exist_ok=True);xs=[];costs=[]
    for r in states:
        q=parts/(digest([eid,r['id'],state_hash(r)])+'.npz')
        if q.exists():z=np.load(q,allow_pickle=False);x=z['x'];c=z['cost']
        else:
            start=time.perf_counter()
            if mode=='hash':x=hash_feature(r,dim);c=[0.,time.perf_counter()-start]
            elif mode=='hidden':x,c=backend.embedding(decision_prompt(r))
            else:raise ValueError('Unknown feature mode')
            tmp=q.with_suffix('.tmp')
            with tmp.open('wb') as f:np.savez_compressed(f,x=x,cost=c)
            tmp.replace(q)
        xs.append(x);costs.append(c)
    tmp=p.with_suffix('.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f,x=np.stack(xs),ids=np.array([r['id'] for r in states]),state_hashes=np.array([state_hash(r) for r in states]),costs=np.array(costs),encoder_id=np.array(eid))
    tmp.replace(p);m={'signature':signature,'encoder_id':eid,'encoder':encoder,'states':len(states),'dimensions':len(xs[0])}
    save_json(meta,m);return m


def load(path,states):
    z=np.load(path,allow_pickle=False);ids=z['ids'].astype(str).tolist()
    if len(ids)!=len(set(ids)) or set(ids)!=set(r['id'] for r in states):raise ValueError('Feature panel mismatch')
    ix={s:i for i,s in enumerate(ids)};order=[ix[r['id']] for r in states]
    if any(z['state_hashes'][i]!=state_hash(r) for i,r in zip(order,states)):raise ValueError('State changed after feature extraction')
    x=np.asarray(z['x'][order],dtype=np.float32)
    if not np.isfinite(x).all():raise ValueError('Nonfinite features')
    return x,str(z['encoder_id']),np.asarray(z['costs'][order])
