from pathlib import Path
from .io import digest,seed_for,save_json,save_rows
from refldyn.corpus import adapt_math,adapt_gsm8k,adapt_aime,build_corpus

SOURCES={'math':('jeggers/competition_math','original'), 'gsm8k':('openai/gsm8k','main'),
         'aime':('AI-MO/aimo-validation-aime',None), 'humaneval':('openai/openai_humaneval',None)}


def ingest(kind,output,revision=None):
    from huggingface_hub import HfApi
    from datasets import load_dataset
    if kind not in SOURCES:raise ValueError('Unknown corpus')
    repo,config=SOURCES[kind];resolved=HfApi().dataset_info(repo_id=repo,revision=revision).sha
    if not resolved:raise RuntimeError('Dataset revision was not resolved')
    rows=[]
    for split in (('train','test') if kind in ('math','gsm8k') else ('train',) if kind=='aime' else ('test',)):
        ds=load_dataset(repo,config,split=split,revision=resolved)
        for source_row in ds:
            if kind=='math':r=adapt_math(source_row,split)
            elif kind=='gsm8k':r=adapt_gsm8k(source_row,split)
            elif kind=='aime':r=adapt_aime(source_row)
            else:
                q=str(source_row['prompt']);r={'id':digest(q)[:20],'question':q,'source':'humaneval','source_split':'test',
                    'split':'ood_code','task':'code','hidden_tests':str(source_row['test']),
                    'entry_point':str(source_row['entry_point']),'source_id':str(source_row['task_id'])}
            if r:rows.append(r)
    if kind!='humaneval':
        rows,rejected,summary=build_corpus(rows)
        for r in rows:r['task']='math';r['fixture']=False
    else:rejected=[];summary={'accepted':len(rows)}
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    for split in sorted({r['split'] for r in rows}):save_rows(out/f'{split}.jsonl',[r for r in rows if r['split']==split])
    save_rows(out/'rejected.jsonl',rejected)
    save_json(out/'manifest.json',{'schema':5,'dataset':repo,'configuration':config,'revision':resolved,'kind':kind,**summary})
    return {'revision':resolved,'questions':len(rows),'output':str(out)}
