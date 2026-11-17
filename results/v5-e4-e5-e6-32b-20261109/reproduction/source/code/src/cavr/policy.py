import numpy as np
from .io import seed_for,unique


def score(prediction,cost_weight=.05,risk=0.,ablation=None):
    if cost_weight<0 or risk<0:raise ValueError('Cost and risk coefficients must be nonnegative')
    a=np.asarray(prediction['members'],dtype=float)
    if a.ndim!=2 or a.shape[1]!=7 or not np.isfinite(a).all():raise ValueError('Invalid controller prediction')
    mode=ablation or prediction['mode'];c=a[:,0];r=a[:,1];h=a[:,2];cost=a[:,4]/1000.
    if mode=='confidence':u=1-c
    elif mode=='direct':u=a[:,3]-cost_weight*cost
    elif mode=='no_damage':u=(1-c)*r-cost_weight*cost
    else:u=(1-c)*r-c*h-(0 if mode=='no_cost' else cost_weight)*cost
    return float(u.mean()-risk*u.std()),float(u.mean()),float(u.std())


def decisions(predictions,cost_weight=.05,risk=0.,budget=None,method=None,seed=41,threshold=None):
    unique(predictions)
    if threshold is None:threshold=.5 if (method or predictions[0]['mode'])=='confidence' else 0.
    if budget is not None and not 0<=budget<=1:raise ValueError('Budget must be a fraction in [0,1]')
    scores=[]
    for p in predictions:
        if method=='random':s=(seed_for(seed,p['id'])+.5)/(2**31-1)
        else:s=score(p,cost_weight,risk,method)[0]
        scores.append(s)
    if budget is None:
        chosen={p['id'] for p,s in zip(predictions,scores) if s>threshold}
    else:
        n=round(len(predictions)*budget)
        ranking=sorted(zip(predictions,scores),key=lambda z:(-z[1],seed_for(z[0]['id'])))
        chosen={p['id'] for p,s in ranking[:n]}
    return [{'id':p['id'],'problem_id':p['problem_id'],'state_hash':p['state_hash'],
             'action':'REFLECT' if p['id'] in chosen else 'COMMIT','score':s,
             'controller_signature':p['controller_signature'],'executor':p['executor'],
             'seen_by_controller':p['seen_by_controller'],'cost_weight':cost_weight,'risk':risk,
             'budget':budget,'method':method or p['mode'],'feature_prefill_tokens':p.get('feature_prefill_tokens',0),
             'feature_seconds':p.get('feature_seconds',0)} for p,s in zip(predictions,scores)]
