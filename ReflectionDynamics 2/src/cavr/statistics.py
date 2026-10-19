import numpy as np


def cluster_interval(values,problem_ids,draws=3000,seed=41):
    x=np.asarray(values,float);ids=np.asarray(problem_ids,str)
    if len(x)!=len(ids) or not len(x) or not np.isfinite(x).all():raise ValueError('Invalid interval inputs')
    keys,ix=np.unique(ids,return_inverse=True);n=np.bincount(ix);tot=np.bincount(ix,weights=x)
    rng=np.random.default_rng(seed);samples=[]
    for start in range(0,draws,250):
        chosen=rng.integers(0,len(keys),size=(min(250,draws-start),len(keys)))
        samples.extend((tot[chosen].sum(1)/n[chosen].sum(1)).tolist())
    lo,hi=np.quantile(samples,[.025,.975]);return {'mean':float(x.mean()),'ci_low':float(lo),'ci_high':float(hi),
        'draws':draws,'clusters':len(keys),'unit':'original problem; state-weighted; conditional on recorded generations'}


def holm(p_values):
    p=np.asarray(p_values,float);ix=np.argsort(p);out=np.zeros(len(p));highest=0.
    for i,j in enumerate(ix):highest=max(highest,(len(p)-i)*p[j]);out[j]=min(1.,highest)
    return out.tolist()


def trajectory_fit(steps,values):
    x=np.asarray(steps,float);y=np.asarray(values,float)
    if len(x)<7 or len(set(x))!=len(x):raise ValueError('At least seven unique checkpoints required')
    order=np.argsort(x);x=x[order];y=y[order];x=(x-x.min())/max(x.max()-x.min(),1.)
    def fit(a,k):
        coef=np.linalg.lstsq(a,y,rcond=None)[0];pred=a@coef;sse=float(np.square(y-pred).sum())
        return {'bic':float(len(x)*np.log(max(sse/len(x),1e-12))+k*np.log(len(x))),
                'sse':sse,'prediction':pred.tolist(),'coefficients':coef.tolist()}
    smooth=fit(np.column_stack([np.ones(len(x)),x,x*x]),3);candidates=[]
    for k in range(2,len(x)-2):
        design=np.column_stack([np.ones(len(x)),x,np.maximum(0,x-x[k])])
        row=fit(design,4);row['breakpoint_step']=float(np.asarray(steps)[order][k]);candidates.append(row)
    best=min(candidates,key=lambda r:r['bic'])
    return {'smooth_quadratic':smooth,'best_segmented':best,'bic_segmented_minus_smooth':best['bic']-smooth['bic'],
            'interpretation':'descriptive model comparison; no phase transition or confirmatory p-value'}
