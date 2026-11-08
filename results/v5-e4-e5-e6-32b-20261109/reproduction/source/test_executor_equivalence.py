"""Regression checks against recorded executor definitions; no GPU or generation."""
import copy
import pipeline as p


def main():
    cfg=p.read_json(p.CFG)
    checked=[]
    for ck in ('base','step400','cavr_online-round3','uniform_executor-round3'):
        manifest=next((p.ROOT/'probes'/ck).glob('*/fit-0.jsonl.manifest.json'))
        old=p.read_json(manifest)['specification']['executor']
        current=copy.deepcopy(old)
        current['source_sha256']=cfg['source_sha256']
        p.executor_equivalence(old,current)
        checked.append(ck)
    old=p.read_json(p.ROOT/'probes/base/gate_train/fit-0.jsonl.manifest.json')['specification']['executor']
    current=p.read_json(p.ROOT/'probes/base/math/score-0.jsonl.manifest.json')['specification']['executor']
    p.executor_equivalence(old,current)
    rejected=[]
    for field in ('adapter_sha256','revision','protocol','runtime','decode','engine','batch_policy'):
        bad=copy.deepcopy(current);bad[field]='changed'
        try:p.executor_equivalence(old,bad)
        except ValueError:rejected.append(field)
        else:raise AssertionError(field)
    for source in ('worker.py','pipeline.py','code/src/cavr/train.py'):
        bad=copy.deepcopy(current);bad['source_sha256'][source]='unreviewed'
        try:p.executor_equivalence(old,bad)
        except ValueError:rejected.append(source)
        else:raise AssertionError(source)
    receipts=list((p.ROOT/'completed').glob('*.json'))
    for f in receipts:
        for name,h in p.read_json(f)['artifacts'].items():assert p.sha256(name)==h,name
    assert p.sha256(p.CFG)==(p.ROOT/'config.sha256').read_text().strip()
    for name,h in cfg['source_sha256'].items():assert p.sha256(p.ROOT/name)==h,name
    for name,h in cfg['input_sha256'].items():assert p.sha256(p.ROOT/name)==h,name
    for name,h in p.read_json(p.ROOT/'E5-policy-lock.json')['controller_hashes'].items():
        assert p.sha256(p.ROOT/name)==h,name
    result={'verified_receipts':len(receipts),'future_executors':checked,'rejected_mutations':rejected,
            'configuration_sha256':p.sha256(p.CFG),'policy_lock_sha256':p.sha256(p.ROOT/'E5-policy-lock.json')}
    p.save_json(p.ROOT/'attempts/20261105-executor/verification.json',result)
    print(result,flush=True)


if __name__=='__main__':main()
