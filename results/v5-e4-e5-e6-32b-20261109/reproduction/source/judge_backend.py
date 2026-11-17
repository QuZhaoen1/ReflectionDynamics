"""Choose once before experiment work; never silently mix grading backends."""
from pathlib import Path
import time
from cavr.io import read_json,save_json,sha256
from task_sandbox import TaskRunner,PREFLIGHT,verify_files,security_checks

ROOT=Path(__file__).resolve().parent
SELECTION=ROOT/'sandbox-selection.json'

def selected_runner(cfg):
    selection=read_json(SELECTION)
    if selection['config_sha256']!=sha256(ROOT/'config.json'):
        raise RuntimeError('Sandbox selection belongs to a different configuration')
    if selection['backend']=='task_local_chroot_seccomp':runner=TaskRunner(cfg)
    elif selection['backend']=='development_docker_private_fileset':
        from sandbox_bridge import FileRunner
        runner=FileRunner({'sandbox_image':cfg['sandbox_source_image']})
    else:raise RuntimeError('Unknown sandbox backend')
    if runner.image!=selection['identity']:raise RuntimeError('Selected sandbox identity changed')
    return runner

def preflight(cfg):
    if SELECTION.exists():
        runner=selected_runner(cfg)
        if isinstance(runner,TaskRunner):verify_files();security_checks()
        result=runner(PREFLIGHT,'def f(x):\n    return x+1')
        if not result['correct']:raise RuntimeError('Selected sandbox preflight failed')
        return read_json(SELECTION)
    try:
        verify_files()
        checks=security_checks()
        save_json(ROOT/'task-sandbox-node-checks.json',{'success':True,'checks':checks,'time':time.time()})
        runner=TaskRunner(cfg)
        selection={'backend':'task_local_chroot_seccomp','development_machine_required':False}
    except Exception as exc:
        # User explicitly allowed the previously verified development fallback.
        # Select only before any evaluation; this selection persists on resume.
        from sandbox_bridge import FileRunner
        runner=FileRunner({'sandbox_image':cfg['sandbox_source_image']})
        result=runner(PREFLIGHT,'def f(x):\n    return x+1')
        if not result['correct']:raise RuntimeError('Both sandbox backends failed') from exc
        selection={'backend':'development_docker_private_fileset','development_machine_required':True,
                   'task_local_failure':repr(exc)}
    selection.update(identity=runner.image,config_sha256=sha256(ROOT/'config.json'),time=time.time())
    save_json(SELECTION,selection)
    print('SANDBOX_SELECTION: '+str(selection),flush=True)
    return selection
