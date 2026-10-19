import numpy as np
from .io import unique,seed_for
from .data import validate_states,state_hash
from .policy import score


def route(states,predictions,epsilon=.15,temperature=.05,seed=41,cost_weight=.05):
    validate_states(states,training=True)
    if not 0<epsilon<=1 or temperature<=0:raise ValueError('Routing requires nonzero exploration')
    pp=unique(predictions)
    if set(pp)!=set(r['id'] for r in states):raise ValueError('Routing panel mismatch')
    out=[];probs=[]
    for r in states:
        p=pp[r['id']]
        if p['state_hash']!=state_hash(r):raise ValueError('Routing state mismatch')
        u=score(p,cost_weight,0.)[0];prob=epsilon+(1-epsilon)/(1+np.exp(-np.clip(u/temperature,-40,40)))
        probs.append(float(prob))
        if (seed_for(seed,'route',r['id'])+.5)/(2**31-1)<prob:out.append({**r,'routing_probability':float(prob)})
    if len(out)<16:raise ValueError('Fewer than 16 routed states; increase the declared exploration rate or training pool')
    return out,{'selected':len(out),'candidates':len(states),'epsilon':epsilon,'temperature':temperature,
                'mean_inclusion_probability':float(np.mean(probs)),
                'objective':'gate-weighted executor training distribution; not importance-corrected full-distribution training'}
