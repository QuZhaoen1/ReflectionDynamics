"""Private-fileset queue; code executes only in the existing restricted Docker runner."""
import argparse
import concurrent.futures
import fcntl
import os
import time
from pathlib import Path
from cavr.io import read_json,save_json,digest,load_rows

ROOT=Path(__file__).resolve().parent
QUEUE=ROOT/'code-judge'
PREFLIGHT={'source_id':'__preflight__','hidden_tests':'def check(candidate):\n    assert candidate(2)==3','entry_point':'f'}


class FileRunner:
    def __init__(self,cfg):
        self.image=cfg['sandbox_image']
        self.check_ready()

    def check_ready(self):
        p=QUEUE/'heartbeat.json'
        if not p.exists():raise RuntimeError('Code judge on development machine is not running')
        h=read_json(p)
        if time.time()-h['time']>60 or h['image']!=self.image:
            raise RuntimeError('Development code judge offline or image changed; keep development machine on for HumanEval')

    def __call__(self,row,code):
        self.check_ready()
        request={'source_id':row.get('source_id'),'hidden_tests':row['hidden_tests'],
                 'entry_point':row['entry_point'],'code':code,'image':self.image}
        key=digest(request);out=QUEUE/'responses'/f'{key}.json';inp=QUEUE/'requests'/f'{key}.json'
        cached=out.exists()
        if not cached:save_json(inp,request)
        started=time.monotonic()
        while not out.exists():
            if time.monotonic()-started>180:raise RuntimeError('Code judge response timeout; batch remains resumable')
            self.check_ready();time.sleep(.2)
        response=read_json(out)
        if response['request_digest']!=key:raise ValueError('Code judge response identity mismatch')
        if 'error' in response:raise RuntimeError(response['error'])
        return {**response['result'],'sandbox_backend':'development_docker_private_fileset',
                'sandbox_cache_hit':cached,'judge_wait_seconds':time.monotonic()-started}


def serve():
    from cavr.sandbox import DockerRunner
    QUEUE.mkdir(exist_ok=True)
    lock=(QUEUE/'service.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for name in ('requests','processing','responses','processed'):(QUEUE/name).mkdir(exist_ok=True)
    runner=DockerRunner()
    allowed={r['source_id']:r for r in load_rows(ROOT/'corpus/humaneval/ood_code.jsonl')}
    allowed[PREFLIGHT['source_id']]=PREFLIGHT

    def process(path):
        obj=read_json(path);key=path.stem
        try:
            if digest(obj)!=key or obj['image']!=runner.image:raise ValueError('Judge request hash/image mismatch')
            known=allowed.get(obj['source_id'])
            if not known or any(obj[k]!=known[k] for k in ('hidden_tests','entry_point')):
                raise ValueError('Judge accepts only the pinned HumanEval tests')
            if not isinstance(obj['code'],str) or len(obj['code'])>262144:raise ValueError('Invalid code payload')
            result=runner(known,obj['code'])
            response={'request_digest':key,'result':result}
        except Exception as exc:response={'request_digest':key,'error':repr(exc)}
        save_json(QUEUE/'responses'/path.name,response)
        path.replace(QUEUE/'processed'/path.name)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        pending={};last=0.
        while not (QUEUE/'STOP').exists():
            now=time.time()
            if now-last>5:
                save_json(QUEUE/'heartbeat.json',{'time':now,'pid':os.getpid(),'image':runner.image,
                          'workers':8,'mode':'private fileset queue; restricted Docker only','pending':len(pending)})
                last=now
            for key,future in list(pending.items()):
                if future.done():future.result();pending.pop(key)
            for directory in ('processing','requests'):
                for path in (QUEUE/directory).glob('*.json'):
                    if len(pending)>=32:break
                    if path.stem in pending:continue
                    if (QUEUE/'responses'/path.name).exists():
                        path.replace(QUEUE/'processed'/path.name);continue
                    dest=QUEUE/'processing'/path.name
                    if path!=dest:path.replace(dest)
                    pending[path.stem]=pool.submit(process,dest)
            status=ROOT/'status.json'
            if status.exists() and read_json(status).get('stage')=='all_complete':break
            time.sleep(.25)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--serve',action='store_true');a=p.parse_args()
    if a.serve:serve()
