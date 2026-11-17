import time
from dataclasses import dataclass,asdict
from pathlib import Path
from .io import adapter_hash


@dataclass(frozen=True)
class DecodeSpec:
    temperature:float=.7
    top_p:float=.95
    top_k:int=0
    max_prompt_tokens:int=4096
    max_new_tokens:int=1024
    action_tokens:int=24
    def __post_init__(self):
        if self.temperature<=0 or not 0<self.top_p<=1 or self.top_k<0 or min(self.max_prompt_tokens,self.max_new_tokens,self.action_tokens)<1:raise ValueError('Invalid decode configuration')


class HFBackend:
    def __init__(self,model,revision=None,adapter=None,memory_gib=72,decode=None):
        import torch
        from transformers import AutoTokenizer,AutoModelForCausalLM
        if not torch.cuda.is_available():raise RuntimeError('Backbone inference requires CUDA')
        self.torch=torch;self.decode=decode or DecodeSpec()
        self.tokenizer=AutoTokenizer.from_pretrained(model,revision=revision,trust_remote_code=False)
        if self.tokenizer.pad_token_id is None:self.tokenizer.pad_token=self.tokenizer.eos_token
        self.tokenizer.padding_side='left'
        self.model=AutoModelForCausalLM.from_pretrained(model,revision=revision,torch_dtype=torch.bfloat16,
            device_map='auto',max_memory={i:f'{memory_gib}GiB' for i in range(torch.cuda.device_count())},
            attn_implementation='sdpa',low_cpu_mem_usage=True,trust_remote_code=False)
        if adapter:
            from peft import PeftModel
            self.model=PeftModel.from_pretrained(self.model,adapter,is_trainable=False)
        self.model.eval();self.device=self.model.get_input_embeddings().weight.device
        if self.device.type!='cuda' or any(str(v) in ('cpu','disk') for v in getattr(self.model,'hf_device_map',{}).values()):raise RuntimeError('Inference offloaded to CPU/disk; increase GPUs per replica')
        resolved=getattr(self.model.config,'_commit_hash',None) or revision
        if not resolved and not Path(model).exists():raise ValueError('Model revision could not be resolved')
        self.identity={'model':model,'revision':resolved,'adapter_sha256':adapter_hash(adapter),'decode':asdict(self.decode),
            'effective_generation_defaults':self.model.generation_config.to_dict(),'protocol':'cavr-v5'}
    def _inputs(self,prompt):
        x=self.tokenizer(prompt,return_tensors='pt',add_special_tokens=False)
        if x['input_ids'].shape[1]>self.decode.max_prompt_tokens:raise ValueError('Prompt exceeds the declared budget; silent truncation is prohibited')
        return {k:v.to(self.device) for k,v in x.items()}
    def generate(self,prompt,seed,max_tokens=None):
        t=self.torch;x=self._inputs(prompt);t.manual_seed(int(seed));t.cuda.manual_seed_all(int(seed))
        t.cuda.synchronize();start=time.perf_counter();maximum=max_tokens or self.decode.max_new_tokens
        with t.inference_mode():
            y=self.model.generate(**x,do_sample=True,temperature=self.decode.temperature,top_p=self.decode.top_p,
                top_k=self.decode.top_k,repetition_penalty=1.,max_new_tokens=maximum,
                pad_token_id=self.tokenizer.pad_token_id,use_cache=True)
        t.cuda.synchronize();v=y[0,x['input_ids'].shape[1]:];eos=self.model.generation_config.eos_token_id
        eos=[] if eos is None else eos if isinstance(eos,list) else [eos]
        ended=bool(len(v) and int(v[-1]) in eos)
        return {'text':self.tokenizer.decode(v,skip_special_tokens=True),'tokens':len(v),
                'prompt_tokens':int(x['input_ids'].shape[1]),'seconds':time.perf_counter()-start,
                'ended':ended,'truncated':bool(not ended and len(v)>=maximum)}
    def embedding(self,prompt):
        import numpy as np
        t=self.torch;x=self._inputs(prompt);base=self.model.get_base_model() if hasattr(self.model,'get_base_model') else self.model
        decoder=getattr(base,'model',None)
        if decoder is None:raise TypeError('Hidden features require a Qwen-style decoder')
        t.cuda.synchronize();start=time.perf_counter()
        with t.inference_mode():
            h=decoder(**x,use_cache=False,return_dict=True).last_hidden_state[0]
            v=t.cat((h[-1].float(),h.float().mean(0))).cpu().numpy()
        t.cuda.synchronize()
        return v.astype(np.float32),[int(x['input_ids'].shape[1]),time.perf_counter()-start]
