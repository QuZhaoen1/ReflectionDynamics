"""Export only trained LoRA tensors from a complete ZeRO-3 checkpoint."""
import argparse
import json
import shutil
from pathlib import Path
import torch
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from safetensors.torch import save_file, load_file

def export(source, target):
    source, target = Path(source), Path(target)
    trainer_state = json.loads((source/'trainer_state.json').read_text())
    state = get_fp32_state_dict_from_zero_checkpoint(str(source), exclude_frozen_parameters=True)
    weights = {k.replace('.default.', '.'): v.contiguous().float() for k, v in state.items()
               if '.lora_A.' in k or '.lora_B.' in k}
    assert len(weights) == 512, len(weights)
    assert all(v.numel() and torch.isfinite(v).all() for v in weights.values())
    assert any(v.count_nonzero() for k, v in weights.items() if '.lora_B.' in k)
    target.mkdir(parents=True, exist_ok=True)
    temporary = target/'adapter_model.safetensors.tmp'
    save_file(weights, str(temporary), metadata={'format': 'pt'})
    temporary.replace(target/'adapter_model.safetensors')
    for name in ['adapter_config.json','tokenizer_config.json','special_tokens_map.json',
                 'tokenizer.json','added_tokens.json','merges.txt','vocab.json','chat_template.jinja']:
        if (source/name).exists():
            shutil.copy2(source/name, target/name)
    assert (target/'adapter_config.json').exists()
    restored = load_file(str(target/'adapter_model.safetensors'))
    assert set(restored) == set(weights) and all(torch.equal(restored[k],v) for k,v in weights.items())
    report = {'success': True, 'source': str(source), 'step': trainer_state['global_step'],
              'tensors': len(weights), 'trainable_parameters': sum(v.numel() for v in weights.values())}
    (target/'export-result.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report), flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source'); parser.add_argument('target')
    args = parser.parse_args()
    export(args.source,args.target)
