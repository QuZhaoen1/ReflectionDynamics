import glob
import hashlib
import json
import os
import tempfile
from pathlib import Path


def digest(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()


def seed_for(*parts):
    return int(digest(parts)[:15],16)%(2**31-1)


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def atomic_text(path,text):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as f:
            f.write(text);f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def save_json(path,x):
    atomic_text(path,json.dumps(x,indent=2,ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n')


def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def load_rows(patterns):
    if isinstance(patterns,(str,Path)):patterns=[str(patterns)]
    paths=sorted({p for pattern in patterns for p in glob.glob(str(pattern))})
    if not paths:raise FileNotFoundError(f'No files match {patterns}')
    rows=[]
    for p in paths:
        with open(p,encoding='utf-8') as f:
            for no,line in enumerate(f,1):
                if not line.strip():continue
                try:r=json.loads(line)
                except (ValueError,UnicodeDecodeError) as e:raise ValueError(f'{p}:{no}: invalid JSONL; repair-tail can repair only a torn final record') from e
                if not isinstance(r,dict):raise ValueError(f'{p}:{no}: expected an object')
                rows.append(r)
    return rows


def save_rows(path,rows):
    atomic_text(path,''.join(json.dumps(r,ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n' for r in rows))


def unique(rows,key='id'):
    out={}
    for r in rows:
        if r[key] in out:raise ValueError(f'Duplicate {key}: {r[key]}')
        out[r[key]]=r
    return out


def adapter_hash(path):
    if path is None:return None
    p=Path(path)
    names=sorted(p.glob('*.safetensors'))+sorted(p.glob('adapter_config.json'))
    if not names or not (p/'adapter_config.json').exists():raise ValueError(f'No exported adapter at {p}')
    return digest({f.name:sha256(f) for f in names})


class EventLog:
    def __init__(self,path,spec):
        self.path=Path(path);self.meta=Path(str(path)+'.manifest.json');self.signature=digest(spec)
        self.path.parent.mkdir(parents=True,exist_ok=True)
        if self.path.exists() and not self.meta.exists():raise ValueError('Missing resume manifest')
        if self.meta.exists() and read_json(self.meta)['signature']!=self.signature:raise ValueError('Resume signature mismatch; choose a new output')
        if not self.meta.exists():save_json(self.meta,{'signature':self.signature,'specification':spec})
        self.rows=unique(load_rows(self.path),'event_id') if self.path.exists() and self.path.stat().st_size else {}
        self.handle=None
    def __enter__(self):
        import fcntl
        self.handle=self.path.open('a',encoding='utf-8')
        fcntl.flock(self.handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.rows=unique(load_rows(self.path),'event_id') if self.path.stat().st_size else {}
        if self.path.stat().st_size and not self.path.read_bytes().endswith(b'\n'):
            self.handle.write('\n');self.handle.flush();os.fsync(self.handle.fileno())
        return self
    def append(self,r):
        if r['event_id'] in self.rows:raise ValueError('Duplicate event')
        self.handle.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
        self.handle.flush();os.fsync(self.handle.fileno());self.rows[r['event_id']]=r
    def __exit__(self,*args):
        if self.handle:self.handle.close()
    def complete(self,expected):
        if len(self.rows)!=expected:raise ValueError(f'Incomplete event log {len(self.rows)}/{expected}')
        save_json(str(self.path)+'.complete.json',{'signature':self.signature,'events':expected,'sha256':sha256(self.path)})


def repair_tail(path):
    p=Path(path);data=p.read_bytes();lines=data.splitlines(keepends=True)
    bad=None
    for i,line in enumerate(lines):
        try:json.loads(line)
        except (ValueError,UnicodeDecodeError):bad=i;break
    if bad is None:return {'changed':False}
    if bad!=len(lines)-1:raise ValueError('Corruption is not limited to the final record')
    backup=p.with_name(p.name+'.torn-'+sha256(p)[:12]);backup.write_bytes(data)
    atomic_text(p,b''.join(lines[:bad]).decode('utf-8'))
    result={'changed':True,'backup':str(backup),'removed_bytes':len(lines[-1])}
    save_json(str(p)+'.repair.json',result);return result


def load_events(patterns):
    if isinstance(patterns,(str,Path)):patterns=[str(patterns)]
    paths=sorted({p for pattern in patterns for p in glob.glob(str(pattern))})
    if not paths:raise FileNotFoundError('No event logs matched')
    for p in paths:
        meta=read_json(p+'.manifest.json');receipt=read_json(p+'.complete.json')
        if receipt['signature']!=meta['signature'] or receipt['sha256']!=sha256(p):raise ValueError('Event completion receipt mismatch')
    rows=load_rows(paths);unique(rows,'event_id');return rows
