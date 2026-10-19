from pathlib import Path
import re
import shutil
from .io import read_json,save_json,sha256


def inspect_checkpoint(path,world=None):
    p=Path(path);state=read_json(p/'trainer_state.json');files=list(p.glob('**/*optim_states.pt'))
    if not files:raise ValueError('No complete optimizer shards; an adapter is not a resumable training checkpoint')
    if world is not None and len(files)!=world:raise ValueError(f'Optimizer shards {len(files)} do not match world size {world}')
    if any(f.stat().st_size==0 for f in files):raise ValueError('Empty optimizer shard')
    return {'global_step':int(state['global_step']),'optimizer_shards':len(files),'path':str(p)}


def latest_complete(run_dir,world):
    candidates=sorted(Path(run_dir).glob('checkpoint-*'),key=lambda p:int(p.name.split('-')[-1]),reverse=True)
    for p in candidates:
        try:inspect_checkpoint(p,world);return str(p)
        except (ValueError,FileNotFoundError):continue
    raise FileNotFoundError('No complete checkpoint available')


def export_adapter(source,target):
    import torch
    from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
    from safetensors.torch import save_file,load_file
    info=inspect_checkpoint(source);p=Path(source);out=Path(target)
    if (out/'export.json').exists():
        old=read_json(out/'export.json')
        if old['source']==str(p.resolve()) and old['sha256']==sha256(out/'adapter_model.safetensors'):return old
        raise ValueError('Adapter output exists with a different origin')
    state=get_fp32_state_dict_from_zero_checkpoint(str(p),exclude_frozen_parameters=True)
    weights={k.replace('.default.','.'):v.contiguous().float() for k,v in state.items() if '.lora_A.' in k or '.lora_B.' in k}
    a={k.replace('lora_A','lora_B') for k in weights if 'lora_A' in k};b={k for k in weights if 'lora_B' in k}
    if not weights or a!=b:raise ValueError('Missing LoRA tensor pairs')
    if any(not v.numel() or not torch.isfinite(v).all() for v in weights.values()):raise ValueError('Invalid adapter tensors')
    if not any(v.count_nonzero() for k,v in weights.items() if 'lora_B' in k):raise ValueError('All LoRA B tensors are zero')
    out.mkdir(parents=True,exist_ok=True);tmp=out/'adapter_model.safetensors.tmp'
    save_file(weights,str(tmp),metadata={'format':'pt'});tmp.replace(out/'adapter_model.safetensors')
    for name in ('adapter_config.json','tokenizer_config.json','special_tokens_map.json','tokenizer.json','added_tokens.json','merges.txt','vocab.json','chat_template.jinja'):
        if (p/name).exists():shutil.copy2(p/name,out/name)
    if not (out/'adapter_config.json').exists():raise ValueError('Missing adapter configuration')
    restored=load_file(str(out/'adapter_model.safetensors'))
    if set(restored)!=set(weights) or any(not torch.equal(restored[k],v) for k,v in weights.items()):raise ValueError('Adapter round-trip check failed')
    result={**info,'source':str(p.resolve()),'tensors':len(weights),'parameters':sum(v.numel() for v in weights.values()),
            'sha256':sha256(out/'adapter_model.safetensors')};save_json(out/'export.json',result);return result
