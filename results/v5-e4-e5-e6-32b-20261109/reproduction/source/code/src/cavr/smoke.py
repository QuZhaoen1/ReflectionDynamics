import re
from pathlib import Path
import torch
from .backend import DecodeSpec
from .io import digest,seed_for,save_rows,load_rows,save_json
from .data import split_training
from .sampling import probe
from .features import extract
from .controller import fit,predict
from .evaluate import offline,action_audit,selection_metrics
from .interventions import plan_from_action_samples,execute


class FixtureBackend:
    identity={'model':'fixture-only','revision':'1','adapter_sha256':None,'protocol':'cavr-v5','fixture':True}
    decode=DecodeSpec(max_prompt_tokens=20000,max_new_tokens=100)
    def generate(self,prompt,seed,max_tokens=None):
        n=int(re.search(r'Compute (\d+) plus',prompt).group(1));correct=str(n+7)
        x=(seed_for(seed,'sample')%1000)/1000
        if max_tokens==self.decode.action_tokens:
            text=' REFLECT\n' if x<.45 else ' COMMIT\n';tokens=3
        else:
            answer=correct if x<(.8 if n%2 else .25) else str(n+8)
            text='checked\nFINAL: \\boxed{'+answer+'}';tokens=12
        return {'text':text,'tokens':tokens,'prompt_tokens':len(prompt)//4,'seconds':.001,'ended':True,'truncated':False}


def fixtures(n=100):
    rows=[]
    for i in range(n):
        correct=i%3==0
        rows.append({'id':f'p{i}-s0','problem_id':f'p{i}','split':'train','source':'fixture','fixture':True,
                     'question':f'Compute {i} plus seven. Case {i%2}.','initial_reasoning':f'Candidate arithmetic for item {i}.',
                     'initial_answer':str(i+7 if correct else i+8),'gold':str(i+7),'initial_correct':correct})
    return rows


def run(output):
    torch.set_num_threads(1);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    panels=split_training(fixtures());features={};events={};b=FixtureBackend()
    for name,rows in panels.items():
        save_rows(out/f'{name}.jsonl',rows);f=out/f'{name}.npz';extract(rows,f,'hash',dim=64);features[name]=f
        e=out/f'{name}-events.jsonl';probe(rows,b,e,'fixture',bank='fit' if name!='gate_holdout' else 'score',reps=4)
        events[name]=load_rows(e)
    model=out/'controller'
    fit(panels['gate_train'],panels['gate_dev'],features['gate_train'],features['gate_dev'],
        events['gate_train'],events['gate_dev'],model,'fixture',seeds=(41,73),epochs=12,hidden=16,patience=4,resume=True,fixtures=True)
    preds=predict(panels['gate_holdout'],features['gate_holdout'],model);save_rows(out/'predictions.jsonl',preds)
    table=offline(panels['gate_holdout'],events['gate_holdout'],preds,out/'report','score','fixture',budgets=(.3,),cost_weights=(.05,),bootstrap=100)
    for bank in ('actionA','actionB'):probe(panels['gate_holdout'],b,out/f'{bank}.jsonl','fixture',bank,8,('action',))
    a=load_rows(out/'actionA.jsonl');other=load_rows(out/'actionB.jsonl')
    action_audit(panels['gate_holdout'],a,other,out/'action-audit.json')
    m=selection_metrics(panels['gate_holdout'],events['gate_holdout'],a,'score','fixture');save_json(out/'selection.json',m)
    plan=plan_from_action_samples(panels['gate_holdout'],a,'fixture');save_rows(out/'plan.jsonl',plan)
    execute(panels['gate_holdout'],plan,b,out/'interventions.jsonl','fixture',2)
    report={'status':'passed','scope':'CPU fixture integration test, not a backbone experiment','controller_export':str(model),
            'policies':len(table),'states':{k:len(v) for k,v in panels.items()}}
    save_json(out/'SMOKE_ONLY.json',report);return report
