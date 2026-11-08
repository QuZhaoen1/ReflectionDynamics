"""Regression evidence; reference solutions are used only here, never model inputs."""
import concurrent.futures
import gzip
import json
from pathlib import Path
from cavr.io import save_json,load_rows
from cavr.sandbox import DockerRunner
from task_sandbox import TaskRunner,verify_files,security_checks

ROOT=Path(__file__).resolve().parent
verify_files()
security=security_checks()
save_json(ROOT/'task-sandbox-security.json',{'success':True,'checks':security})
print('Isolation checks passed:',len(security),flush=True)
raw=[json.loads(x) for x in gzip.open(ROOT/'raw/humaneval/HumanEval.jsonl.gz','rt')]
rows={r['source_id']:r for r in load_rows(ROOT/'corpus/humaneval/ood_code.jsonl')}
local=TaskRunner();docker=DockerRunner()
def compare(r):
    row=rows[r['task_id']]
    canonical=r['prompt']+r['canonical_solution']
    broken=f"def {r['entry_point']}(*args,**kwargs): return None"
    result={'id':r['task_id']}
    for name,code in [('canonical',canonical),('incorrect',broken)]:
        a=local(row,code);b=docker(row,code)
        result[name]={'task':a,'docker':b}
        if (a['correct'],a['valid'])!=(b['correct'],b['valid']):
            raise AssertionError((r['task_id'],name,a,b))
        if a['correct']!=(name=='canonical'):
            raise AssertionError((r['task_id'],'unexpected reference score',name,a))
    return result
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    results=list(pool.map(compare,raw))
save_json(ROOT/'task-sandbox-equivalence.json',{'success':True,'questions':len(results),
          'comparisons':len(results)*2,'canonical_passed':len(results),'incorrect_failed':len(results),
          'task_sandbox':local.image,'source_image':docker.image,'rows':results})
print('Docker/task score agreement:',len(results)*2,'/',len(results)*2,flush=True)
