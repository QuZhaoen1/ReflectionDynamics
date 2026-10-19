import copy
import numpy as np
import pytest
import torch
from cavr.bank import targets
from cavr.controller import ValueHead,loss_fn,fit,predict
from cavr.policy import score,decisions
from cavr.route import route
from cavr.io import read_json,save_json

@pytest.mark.parametrize('mode',['cavr','confidence','direct','no_damage'])
def test_controller_losses_finite_and_trainable(mode):
    m=ValueHead(8,8);x=torch.randn(12,8);y=torch.rand(12,7);y[:,0]=(y[:,0]>.5).float();n=torch.ones(12,7)*4;w=torch.ones(12)
    loss=loss_fn(m(x),y,n,w,mode);loss.backward()
    assert torch.isfinite(loss) and any(p.grad is not None and p.grad.abs().sum()>0 for p in m.parameters())

def test_targets_have_conditional_support(banks):
    p=banks['gate_train'];b=targets(p['states'],p['events'],'fit','fixture')
    assert b['y'].shape==(len(p['states']),7)
    assert np.allclose(b['y'][:,3],b['y'][:,1]-b['y'][:,0])
    assert (b['n'][:,1]==4).all()

def test_targets_reject_mixed_checkpoint_and_duplicates(banks):
    p=banks['gate_train'];e=copy.deepcopy(p['events']);e[0]['checkpoint']='other'
    with pytest.raises(ValueError):targets(p['states'],e,'fit','fixture')
    with pytest.raises(ValueError):targets(p['states'],p['events']+[p['events'][0]],'fit','fixture')

def pred(i,c=.5,r=.6,h=.1,cost=100):
    return {'id':str(i),'problem_id':str(i),'state_hash':str(i),'mode':'cavr','members':[[c,r,h,.2,cost,.4,cost]],
            'seen_by_controller':False,'controller_signature':'fit','executor':'model'}

def test_value_decomposition():
    x=pred(0,.6,.7,.2,1000);s,u,sd=score(x,.05)
    assert np.isclose(u,.4*.7-.6*.2-.05) and sd==0

def test_zero_cost_correctness_oracle_property():
    assert score(pred(0,1.,.9,.2),0.)[0]<0
    assert score(pred(0,0.,.9,.2),0.)[0]>0

@pytest.mark.parametrize('budget',[0.,.1,.3,.5,1.])
def test_matched_call_budget(budget):
    rows=[pred(i,c=i/20) for i in range(20)];plan=decisions(rows,budget=budget)
    assert sum(p['action']=='REFLECT' for p in plan)==round(20*budget)
    assert plan==decisions(rows,budget=budget)

def test_risk_penalty_cannot_raise_value():
    p=pred(0);p['members'].append([.9,.1,.9,.1,100,.2,100])
    assert score(p,risk=1)[0]<=score(p,risk=0)[0]

def test_confidence_threshold_is_not_zero():
    p=pred(0,c=.9);p['mode']='confidence'
    assert decisions([p])[0]['action']=='COMMIT'

def test_fit_predict_and_inference_checksum(banks,tmp_path):
    a=banks['gate_train'];b=banks['gate_dev'];c=banks['gate_holdout'];out=tmp_path/'fit'
    fit(a['states'],b['states'],a['features'],b['features'],a['events'],b['events'],out,'fixture',epochs=5,hidden=8,seeds=(41,),fixtures=True)
    rows=predict(c['states'],c['features'],out);assert len(rows)==len(c['states'])
    with pytest.raises(ValueError):predict(a['states'],a['features'],out)
    with pytest.raises(ValueError):fit(a['states'],b['states'],a['features'],b['features'],a['events'],b['events'],out,'fixture',epochs=6,hidden=8,seeds=(41,),fixtures=True)
    fit(a['states'],b['states'],a['features'],b['features'],a['events'],b['events'],out,'fixture',epochs=5,hidden=8,seeds=(41,),fixtures=True,resume=True)
    p=out/'head-41.safetensors';p.write_bytes(p.read_bytes()+b'bad')
    with pytest.raises(ValueError,match='checksum'):predict(c['states'],c['features'],out)

def test_routing_exploration_validation():
    with pytest.raises(ValueError):route([],[],epsilon=0.)
