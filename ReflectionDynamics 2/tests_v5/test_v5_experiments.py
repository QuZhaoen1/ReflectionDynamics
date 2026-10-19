import copy
import numpy as np
import pytest
from cavr.statistics import cluster_interval,holm,trajectory_fit
from cavr.sampling import probe
from cavr.smoke import FixtureBackend,fixtures,run
from cavr.io import load_rows,save_json
from cavr.data import state_hash
from cavr.evaluate import action_audit,offline
from cavr.interventions import plan_from_action_samples,execute
from cavr.policy import decisions
from cavr.checkpoint import inspect_checkpoint,latest_complete
from cavr.cli import build_parser


def test_bootstrap_clusters_and_determinism():
    a=cluster_interval([1,0,1,0],['a','a','b','b'],100)
    assert a['mean']==.5 and a['ci_low']==a['ci_high']==.5
    assert a==cluster_interval([1,0,1,0],['a','a','b','b'],100)

def test_holm_bounded_monotone():
    p=holm([.01,.02,.5]);assert np.allclose(p,[.03,.04,.5])

def test_phase_word_not_reported_as_inference():
    r=trajectory_fit(range(11),[.1+i*.03 for i in range(11)])
    assert 'no phase transition' in r['interpretation']

def test_sampling_resume_does_not_regenerate(tmp_path):
    class Count(FixtureBackend):
        count=0
        def generate(self,*args,**kwargs):self.count+=1;return super().generate(*args,**kwargs)
    b=Count();p=tmp_path/'x.jsonl';rs=fixtures(5)
    probe(rs,b,p,'fixture',reps=4);n=b.count
    probe(rs,b,p,'fixture',reps=4);assert b.count==n
    probe(rs,b,p,'fixture',reps=8);assert b.count==2*n
    with pytest.raises(ValueError):probe(rs,b,p,'fixture',bank='different',reps=8)

def test_action_audit_and_actual_crossed_generation(tmp_path):
    b=FixtureBackend();rs=fixtures(6)
    for bank in ('a','b'):probe(rs,b,tmp_path/f'{bank}.jsonl','fixture',bank,8,('action',))
    a=load_rows(tmp_path/'a.jsonl');other=load_rows(tmp_path/'b.jsonl')
    result=action_audit(rs,a,other,tmp_path/'audit.json');assert len(result['state_estimates'])==6
    with pytest.raises(ValueError):action_audit(rs,a,a,tmp_path/'bad.json')
    plan=plan_from_action_samples(rs,a,'fixture')
    execute(rs,plan,b,tmp_path/'real.jsonl','fixture',3)
    e=load_rows(tmp_path/'real.jsonl');assert len(e)==18
    for r in e:
        if r['action']=='COMMIT':assert r['tokens']==0 and r['correct']==r['initial_correct']

def test_complete_checkpoint_requires_all_shards(tmp_path):
    p=tmp_path/'checkpoint-20';p.mkdir();save_json(p/'trainer_state.json',{'global_step':20})
    with pytest.raises(ValueError):inspect_checkpoint(p,4)
    for i in range(4):(p/f'rank{i}_optim_states.pt').write_bytes(b'test-shard')
    assert inspect_checkpoint(p,4)['global_step']==20
    assert latest_complete(tmp_path,4)==str(p)
    with pytest.raises(ValueError):inspect_checkpoint(p,8)

@pytest.mark.parametrize('command',['init','ingest','states','probe','draft','features','fit','predict','evaluate','audit-actions','selection','dynamics','seeds','plan','gate-plan','execute-plan','route','export-adapter','repair-tail','workflow','legacy-audit','summarize-interventions','smoke','preflight'])
def test_command_registered(command):
    parser=build_parser();names=parser._subparsers._group_actions[0].choices
    assert command in names

def test_full_cpu_pipeline_and_restart(tmp_path):
    first=run(tmp_path/'smoke');second=run(tmp_path/'smoke')
    assert first==second and first['status']=='passed'
