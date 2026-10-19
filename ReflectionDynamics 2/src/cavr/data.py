from pathlib import Path
from collections import defaultdict
from .io import digest,unique,load_rows,save_rows,save_json,read_json,sha256,adapter_hash

VISIBLE=('question','initial_reasoning','initial_answer')


def visible(r):return {k:str(r.get(k,'')) for k in VISIBLE}
def state_hash(r):return digest(visible(r))
def question_key(r):return digest(' '.join(str(r['question']).casefold().split()))


def validate_states(rows,training=False,fixtures=False):
    unique(rows)
    if not rows:raise ValueError('Empty state panel')
    for r in rows:
        for key in ('id','problem_id','split',*VISIBLE,'initial_correct'):
            if key not in r:raise ValueError(f'State {r.get("id")} lacks {key}')
        if type(r['initial_correct']) is not bool:raise ValueError('initial_correct must be a verified boolean')
        if not r['question'] or r['initial_answer'] is None or str(r['initial_answer'])=='':raise ValueError('Empty question or proposed answer')
        if training and r['split']!='train':raise ValueError('Non-training-source state in a training input')
        if training and not fixtures and (r.get('fixture') or r.get('source')=='synthetic_integer_arithmetic'):raise ValueError('Fixtures cannot be used in research training')
    return rows


def disjoint(*panels):
    ids=set();qs=set()
    for p in panels:
        a={r['problem_id'] for r in p};b={question_key(r) for r in p}
        if ids&a or qs&b:raise ValueError('Original problem leakage between panels')
        ids|=a;qs|=b


def split_training(rows,seed=41):
    validate_states(rows,training=True,fixtures=True)
    groups=defaultdict(list)
    for r in rows:groups[question_key(r)].append(r)
    keys=sorted(groups,key=lambda k:digest([seed,k]))
    if len(keys)<5:raise ValueError('At least five original problems required')
    n=max(1,round(.2*len(keys)));out={p:[] for p in ('gate_train','gate_dev','gate_holdout')}
    for i,k in enumerate(keys):
        role='gate_dev' if i<n else 'gate_holdout' if i<2*n else 'gate_train'
        out[role]+=[{**r,'panel':role} for r in groups[k]]
    disjoint(*out.values())
    return {k:sorted(v,key=lambda r:r['id']) for k,v in out.items()}


def init_workspace(legacy_root,workspace,seed=41):
    root=Path(legacy_root).resolve();out=Path(workspace).resolve();p=out/'experiment.json'
    if root==out:raise ValueError('The new workspace must not be the completed run directory')
    if p.exists():
        cfg=read_json(p)
        if cfg['legacy_root']!=str(root) or cfg['split_seed']!=seed:raise ValueError('Workspace already initialized differently')
        return cfg
    a=root/'data/states/train.jsonl';b=root/'data/states/validation.jsonl'
    train=load_rows(a);val=load_rows(b);validate_states(train,training=True);validate_states(val);disjoint(train,val)
    panels=split_training(train,seed);panels['validation']=[{**r,'panel':'validation'} for r in val]
    paths={}
    for k,rows in panels.items():
        dest=out/'states'/f'{k}.jsonl';save_rows(dest,rows);paths[k]=str(dest)
    models={'base':None}
    for step in range(40,401,40):
        dest=root/'adapters'/f'step{step}'
        if (dest/'adapter_model.safetensors').exists():models[f'step{step}']=str(dest)
    cfg={'schema':5,'legacy_root':str(root),'workspace':str(out),'split_seed':seed,
         'model':'Qwen/Qwen2.5-32B','revision':'1818d35814b8319459f4bd55ed1ac8709630f003',
         'states':paths,'checkpoints':models,'input_sha256':{'train':sha256(a),'validation':sha256(b)},
         'adapter_sha256':{k:adapter_hash(v) for k,v in models.items()},
         'holdout_scope':'gate_holdout excludes controller fitting, not the historical backbone training',
         'validation_scope':'previously inspected seed41 validation; exploratory'}
    save_json(p,cfg);return cfg


def select_states(drafts,max_per_problem=0):
    groups=defaultdict(list)
    for r in drafts:
        if r.get('valid') and r.get('initial_answer') is not None:groups[r['problem_id']].append(r)
    out=[]
    for rs in groups.values():
        rs.sort(key=lambda r:r['sample'])
        out+=rs if max_per_problem==0 else rs[:max_per_problem]
    return sorted(out,key=lambda r:r['id'])
