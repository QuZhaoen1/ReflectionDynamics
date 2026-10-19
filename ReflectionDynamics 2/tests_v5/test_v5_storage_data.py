import json
from pathlib import Path
import pytest
from cavr.io import EventLog,load_rows,save_rows,save_json,read_json,repair_tail,load_events
from cavr.data import split_training,disjoint,validate_states,state_hash,init_workspace
from cavr.features import hash_feature,extract,load
from cavr.smoke import fixtures


def test_event_resume_and_receipt(tmp_path):
    p=tmp_path/'e.jsonl'
    with EventLog(p,{'a':1}) as log:log.append({'event_id':'a','v':1});log.complete(1)
    assert load_events(str(p))[0]['v']==1
    with EventLog(p,{'a':1}) as log:
        assert 'a' in log.rows
        with pytest.raises(ValueError):log.append({'event_id':'a'})
    with pytest.raises(ValueError):EventLog(p,{'a':2})

def test_incomplete_event_receipt_rejected(tmp_path):
    p=tmp_path/'e.jsonl'
    with EventLog(p,{}) as log:log.append({'event_id':'1'})
    with pytest.raises(FileNotFoundError):load_events(p)

def test_torn_tail_is_preserved_and_removed(tmp_path):
    p=tmp_path/'events.jsonl';p.write_bytes(b'{"event_id":"a"}\n{"event_id":')
    r=repair_tail(p);assert r['changed'] and Path(r['backup']).exists()
    assert len(load_rows(p))==1

def test_interior_corruption_rejected(tmp_path):
    p=tmp_path/'x.jsonl';p.write_text('{broken}\n{"id":1}\n')
    with pytest.raises(ValueError):repair_tail(p)

def test_valid_record_without_newline_can_resume(tmp_path):
    p=tmp_path/'x.jsonl'
    with EventLog(p,{}) as log:log.append({'event_id':'1'})
    p.write_bytes(p.read_bytes().rstrip())
    with EventLog(p,{}) as log:log.append({'event_id':'2'});log.complete(2)
    assert len(load_events(p))==2

def test_split_original_problem_groups():
    rows=fixtures(30);rows.append({**rows[0],'id':'another-state'})
    parts=split_training(rows);disjoint(*parts.values())
    membership=[k for k,v in parts.items() if any(r['problem_id']=='p0' for r in v)]
    assert len(membership)==1
    assert parts==split_training(rows)

def test_different_ids_same_question_leakage():
    a=fixtures(1)[0];b={**a,'id':'other','problem_id':'other'}
    with pytest.raises(ValueError):disjoint([a],[b])

def test_reject_nontraining_and_fixture():
    r=fixtures(1)[0]
    with pytest.raises(ValueError):validate_states([r],training=True)
    with pytest.raises(ValueError):validate_states([{**r,'split':'test'}],training=True,fixtures=True)

def test_features_only_use_visible_state(tmp_path):
    r=fixtures(1)[0];r2={**r,'gold':'secret','initial_correct':not r['initial_correct'],'hidden_tests':'test'}
    assert state_hash(r)==state_hash(r2)
    assert (hash_feature(r)==hash_feature(r2)).all()
    p=tmp_path/'f.npz';extract([r],p,'hash',dim=16)
    x,_,_=load(p,[r2]);assert x.shape==(1,16)
    with pytest.raises(ValueError):load(p,[{**r,'initial_reasoning':'changed'}])

def test_workspace_legacy_files_not_modified(tmp_path):
    root=tmp_path/'old';train=[{**r,'fixture':False,'source':'math'} for r in fixtures(30)]
    val=[{**r,'id':'v'+r['id'],'problem_id':'v'+r['problem_id'],'question':'Validation '+r['question'],'split':'validation','fixture':False} for r in fixtures(10)]
    save_rows(root/'data/states/train.jsonl',train);save_rows(root/'data/states/validation.jsonl',val)
    before=(root/'data/states/train.jsonl').read_bytes()
    cfg=init_workspace(root,tmp_path/'new');assert cfg['checkpoints']=={'base':None}
    assert (root/'data/states/train.jsonl').read_bytes()==before
    assert cfg==init_workspace(root,tmp_path/'new')
