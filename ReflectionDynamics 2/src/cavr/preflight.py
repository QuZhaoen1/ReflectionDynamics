import importlib.metadata
import platform
import shutil
import subprocess
from pathlib import Path
from .io import save_json


def run(profile,output):
    import yaml
    import torch
    cfg=yaml.safe_load(Path(profile).read_text());packages={}
    for name in ('torch','transformers','trl','peft','accelerate','deepspeed','numpy','safetensors','math-verify'):
        try:packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:packages[name]=None
    cards=[{'name':torch.cuda.get_device_name(i),'memory_gib':torch.cuda.get_device_properties(i).total_memory/2**30} for i in range(torch.cuda.device_count())]
    report={'python':platform.python_version(),'packages':packages,'cuda_build':torch.version.cuda,'gpus':cards,
        'hardware_profile':cfg['hardware'],'free_disk_gib':shutil.disk_usage(Path.cwd()).free/2**30,
        'gpu_available':bool(cards),'profile_gpu_count_matches':len(cards)==cfg['hardware']['world_size']}
    if Path('/proc/meminfo').exists():
        lines=dict(x.split(':',1) for x in Path('/proc/meminfo').read_text().splitlines())
        report['available_host_memory_gib']=int(lines['MemAvailable'].strip().split()[0])/1024**2
    if shutil.which('nvidia-smi'):
        report['topology']=subprocess.run(['nvidia-smi','topo','-m'],capture_output=True,text=True).stdout
    save_json(output,report);return report
