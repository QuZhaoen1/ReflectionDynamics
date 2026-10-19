import json
import sys
from pathlib import Path
import torch
from safetensors.torch import load_file
from refldyn.backend import InferenceBackend
from refldyn.protocol import prompt_for_state

adapter = Path(sys.argv[1])
backend = InferenceBackend('Qwen/Qwen2.5-32B', adapter=str(adapter), max_memory_gib=72)
saved = load_file(str(adapter/'adapter_model.safetensors'))
matched = 0
for name,param in backend.model.named_parameters():
    key = name.replace('.default.','.')
    if key in saved:
        assert torch.equal(param.detach().float().cpu(),saved[key]),key
        matched += 1
assert matched == len(saved) == 512
state = json.loads(Path('data/states/train.jsonl').read_text().splitlines()[0])
_, count = backend.generate(prompt_for_state(state), 41, 8, sample=False)
assert count > 0
result = {'success':True,'exactly_loaded_tensors':matched,'generated_tokens':count}
(adapter/'reload-result.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
