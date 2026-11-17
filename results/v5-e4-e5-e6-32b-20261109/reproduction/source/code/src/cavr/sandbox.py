import json
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path


class DockerRunner:
    def __init__(self,image='reflection-code-runner:1',timeout=8.,memory='512m'):
        if not shutil.which('docker'):raise RuntimeError('Docker unavailable; code evaluation is not run on the host')
        r=subprocess.run(['docker','image','inspect','--format','{{.Id}}',image],capture_output=True,text=True)
        if r.returncode:raise RuntimeError('Build the isolated code-runner image before code evaluation')
        self.image=r.stdout.strip();self.timeout=timeout;self.memory=memory
    def __call__(self,row,code):
        if not row.get('hidden_tests') or not row.get('entry_point'):raise ValueError('Code task lacks private tests')
        name='reflection-'+uuid.uuid4().hex
        with tempfile.TemporaryDirectory(prefix='reflection-code-') as d:
            p=Path(d);os.chmod(p,0o755)
            (p/'payload.json').write_text(json.dumps({'code':code,'tests':row['hidden_tests'],'entry':row['entry_point']}))
            os.chmod(p/'payload.json',0o444)
            out=p/'stdout';err=p/'stderr'
            command=['docker','run','--rm','--pull=never','--name',name,'--network=none','--read-only',
                '--user','65534:65534','--cap-drop=ALL','--security-opt','no-new-privileges','--pids-limit','64',
                '--memory',self.memory,'--memory-swap',self.memory,'--cpus','1',
                '--ulimit','nofile=128:128','--tmpfs','/tmp:rw,noexec,nosuid,size=64m',
                '--mount',f'type=bind,source={p}/payload.json,target=/input/payload.json,readonly',self.image]
            start=time.monotonic();reason=None
            with out.open('wb') as fo,err.open('wb') as fe:
                proc=subprocess.Popen(command,stdout=fo,stderr=fe)
                try:
                    while proc.poll() is None:
                        if time.monotonic()-start>self.timeout:reason='timeout';break
                        if out.stat().st_size+err.stat().st_size>2*1024*1024:reason='output_limit';break
                        time.sleep(.05)
                    if reason:
                        subprocess.run(['docker','rm','-f',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=10)
                        proc.kill()
                    proc.wait(timeout=10)
                finally:
                    subprocess.run(['docker','rm','-f',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=10)
            if proc.returncode==125:raise RuntimeError('Docker infrastructure failure; not a model error')
            tail=out.read_bytes()[-4096:].decode('utf-8',errors='replace')
            records=[line for line in tail.splitlines() if line.startswith('RESULT_JSON:')]
            payload=None
            if records:
                try:payload=json.loads(records[-1].split(':',1)[1])
                except ValueError:pass
            valid=bool(payload and payload.get('finished') and proc.returncode==0 and reason is None)
            return {'correct':bool(valid and payload.get('passed')),'valid':valid,
                    'sandbox_reason':reason or ('complete' if valid else 'execution_failure'),
                    'sandbox_image':self.image,'sandbox_seconds':time.monotonic()-start}
