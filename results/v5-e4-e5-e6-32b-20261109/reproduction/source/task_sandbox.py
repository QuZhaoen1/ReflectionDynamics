"""Task-local chroot + zero capabilities + seccomp allowlist; no Docker service."""
import json
import os
import selectors
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path
from cavr.io import read_json, sha256, digest

ROOT = Path(__file__).resolve().parent
PREFLIGHT = {'source_id': '__preflight__', 'hidden_tests': 'def check(candidate):\n    assert candidate(2)==3', 'entry_point': 'f'}

class TaskRunner:
    def __init__(self, cfg=None, timeout=8.):
        self.manifest = read_json(ROOT/'task-sandbox/manifest.json')
        self.image = 'task-seccomp-v1:' + digest(self.manifest)
        self.timeout = timeout
        self.rootfs = ROOT/'task-sandbox/rootfs'
        if cfg and cfg['sandbox_image'] != self.image:
            raise ValueError('Task sandbox identity mismatch')

    def __call__(self, row, code):
        if not row.get('hidden_tests') or not row.get('entry_point'):
            raise ValueError('Code task lacks private tests')
        payload = json.dumps({'code': code, 'tests': row['hidden_tests'], 'entry': row['entry_point']}).encode()
        if len(payload) > 1024**2: raise ValueError('Code payload exceeds 1 MiB')
        cmd = [sys.executable, '-I', str(ROOT/'task-sandbox/launch.py'), str(self.rootfs)]
        start = time.monotonic(); reason = None; chunks = {1: bytearray(), 2: bytearray()}
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                env={}, close_fds=True, start_new_session=True)
        try:
            # Multiplex input and output: avoid pipe deadlocks and cap all output.
            with selectors.DefaultSelector() as sel:
                for stream, key in [(proc.stdout, 1), (proc.stderr, 2)]:
                    os.set_blocking(stream.fileno(), False); sel.register(stream, selectors.EVENT_READ, key)
                os.set_blocking(proc.stdin.fileno(), False); sel.register(proc.stdin, selectors.EVENT_WRITE, 0)
                pos = 0
                while sel.get_map():
                    if time.monotonic()-start > self.timeout: reason = 'timeout'; break
                    for key, events in sel.select(.05):
                        stream = key.fileobj
                        if key.data == 0:
                            try: pos += os.write(stream.fileno(), payload[pos:pos+4096])
                            except BrokenPipeError: pos = len(payload)
                            if pos == len(payload): sel.unregister(stream); stream.close()
                        else:
                            data = os.read(stream.fileno(), 65536)
                            if not data: sel.unregister(stream)
                            else: chunks[key.data].extend(data)
                    if sum(map(len, chunks.values())) > 2*1024**2: reason = 'output_limit'; break
                if reason: os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=2)
        finally:
            if proc.poll() is None: os.killpg(proc.pid, signal.SIGKILL); proc.wait()
            for stream in (proc.stdin, proc.stdout, proc.stderr): stream.close()
        if b'SANDBOX_READY\n' not in chunks[2]:
            raise RuntimeError('Task isolation startup failed, refusing host execution: ' + chunks[2][-3000:].decode(errors='replace'))
        records = [line for line in chunks[1][-4096:].decode(errors='replace').splitlines() if line.startswith('RESULT_JSON:')]
        result = None
        if records:
            try: result = json.loads(records[-1].split(':', 1)[1])
            except ValueError: pass
        valid = bool(result and result.get('finished') and proc.returncode == 0 and reason is None)
        return {'correct': bool(valid and result.get('passed')), 'valid': valid,
                'sandbox_reason': reason or ('complete' if valid else 'execution_failure'),
                'sandbox_image': self.image, 'sandbox_backend': 'task_local_chroot_seccomp',
                'sandbox_seconds': time.monotonic()-start}

def verify_files():
    m = read_json(ROOT/'task-sandbox/manifest.json')
    for name, expected in m['sha256'].items():
        p = ROOT/'task-sandbox'/name
        if sha256(p) != expected: raise RuntimeError(f'Sandbox runtime changed: {name}')
    # No writable directories, privilege bits, or host device/proc mounts in guest.
    for p in (ROOT/'task-sandbox/rootfs').rglob('*'):
        name = str(p.relative_to(ROOT/'task-sandbox'))
        if p.is_symlink():
            if m['symlinks'].get(name) != os.readlink(p): raise RuntimeError(f'Guest symlink changed: {name}')
            continue
        st=p.stat()
        if st.st_uid != 0 or st.st_mode & 0o6022 or not (stat.S_ISREG(st.st_mode) or stat.S_ISDIR(st.st_mode)):
            raise RuntimeError(f'Unsafe guest ownership/mode: {p}')
        if p.is_file() and name not in m['sha256']: raise RuntimeError(f'Unpinned guest file: {name}')
    return m

def security_checks():
    runner = TaskRunner()
    def check(name, code, expected=True):
        row={'entry_point':'f','hidden_tests':'def check(candidate):\n    assert candidate() is True'}
        result=runner(row,code)
        if result['correct'] != expected: raise RuntimeError(f'Isolation self-test failed: {name}: {result}')
        return result
    results={}
    results['correct']=check('correct','def f(): return True')
    results['incorrect']=check('incorrect','def f(): return False',False)
    results['private_paths_absent']=check('private_paths_absent',"import os\ndef f():\n return not any(os.path.exists(p) for p in ['PROJECT_STORAGE_MOUNT','PERSONAL_STORAGE_MOUNT','/public_fast_storage','/proc/self','/dev/nvidia0','/root/.ssh'])")
    results['root_traversal']=check('root_traversal',"import os\ndef f():\n return os.getcwd()=='/' and not os.path.exists('/../../../..PROJECT_STORAGE_MOUNT')")
    # Raw syscall checks prevent Python-level monkey-patching from masquerading as isolation.
    for name,nr in [('socket',41),('fork',57),('execve',59),('kill',62),('ptrace',101),
                    ('setuid',105),('mount',165),('unshare',272),('process_vm_readv',310),('io_uring_setup',425)]:
        code=f"import ctypes\ndef f():\n c=ctypes.CDLL(None,use_errno=True); r=c.syscall({nr},0,0,0,0,0,0); return r==-1 and ctypes.get_errno()==1"
        results[name]=check(name,code)
    results['file_write']=check('file_write',"import os,errno\ndef f():\n try: os.open('/sandbox/guest.py',os.O_WRONLY)\n except OSError as e: return e.errno==errno.EPERM\n return False")
    results['memory_limit']=check('memory_limit','def f():\n x=bytearray(700*1024**2); return True',False)
    short=TaskRunner(timeout=.5)
    row={'entry_point':'f','hidden_tests':'def check(candidate): candidate()'}
    results['wall_timeout']=short(row,'import time\ndef f(): time.sleep(10)')
    if results['wall_timeout']['sandbox_reason']!='timeout':raise RuntimeError('Wall timeout not enforced')
    results['output_limit']=runner(row,"import os\ndef f():\n while True: os.write(1,b'x'*65536)")
    if results['output_limit']['sandbox_reason']!='output_limit':raise RuntimeError('Output limit not enforced')
    return results
