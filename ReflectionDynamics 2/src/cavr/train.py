import argparse
import dataclasses
import math
import os
import signal
import time
from pathlib import Path
from .io import load_rows,read_json,save_json,sha256,digest,adapter_hash
from .data import validate_states
from .protocol import decision_prompt,reflect_prompt,grade
from .checkpoint import inspect_checkpoint,latest_complete


def training_records(states,mode):
    validate_states(states,training=True)
    if any(r.get('task','math')!='math' for r in states):raise ValueError('The GRPO training recipe uses verifiable mathematics; code is a held-out transfer task')
    if mode not in ('standard','executor'):raise ValueError('Unknown actor objective')
    return [{'prompt':decision_prompt(r) if mode=='standard' else reflect_prompt(r),'initial_answer':r['initial_answer'],
             'gold':r['gold'],'state_id':r['id'],'question':r['question'],'initial_correct':r['initial_correct']} for r in states]


def reward_factory(mode,cost_weight,tokenizer):
    def outcome(completions,initial_answer,gold,**kwargs):
        scores=[]
        for text,initial,target in zip(completions,initial_answer,gold):
            if isinstance(text,list):text=''.join(x.get('content','') for x in text)
            r={'initial_answer':initial,'gold':target}
            g=grade(r,str(text),'natural' if mode=='standard' else 'reflect')
            tokens=len(tokenizer.encode(str(text),add_special_tokens=False))
            scores.append(float(g['correct'])-cost_weight*tokens/1000.)
        return scores
    outcome.__name__='outcome_minus_cost';return outcome


def run(config,states_path,output,mode='standard',init_adapter=None,resume=None,seed=41,max_steps=None,pilot=False,cost_weight=0.,stop_at_step=None):
    import yaml
    import torch
    import torch.distributed as dist
    from transformers import AutoTokenizer,AutoModelForCausalLM,TrainerCallback
    from peft import LoraConfig,PeftModel
    from datasets import Dataset
    from trl import GRPOConfig,GRPOTrainer
    from transformers.integrations import HfDeepSpeedConfig
    cfg=yaml.safe_load(Path(config).read_text());t=cfg['training'];world=int(os.getenv('WORLD_SIZE','1'))
    if world!=cfg['hardware']['world_size']:raise ValueError('torchrun world size does not match hardware profile')
    if not torch.cuda.is_available():raise RuntimeError('GRPO requires CUDA')
    if min(torch.cuda.get_device_properties(i).total_memory/2**30 for i in range(torch.cuda.device_count()))<cfg['hardware']['minimum_gpu_gib']:raise ValueError('GPU memory is below the selected profile')
    if shutil_disk_free(output)<cfg['hardware'].get('minimum_free_disk_gib',150):raise RuntimeError('Insufficient persistent disk space')
    records=training_records(load_rows(states_path),mode)
    if len(records)<16:raise ValueError('Too few real states for distributed GRPO')
    tokenizer=AutoTokenizer.from_pretrained(cfg['model'],revision=cfg.get('revision'),trust_remote_code=False)
    tokenizer.padding_side='left'
    if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
    maximum=int(t.get('max_prompt_tokens',3200));sized=[]
    for r in records:
        if len(tokenizer.encode(r['prompt'],add_special_tokens=False))<=maximum:sized.append(r)
    if len(sized)!=len(records):raise ValueError('Training prompts exceed the budget; create an explicitly filtered state file instead of silently changing the cohort')
    if pilot:records=sorted(records,key=lambda r:-len(tokenizer.encode(r['prompt'])))[:32]
    steps=8 if pilot else int(max_steps or t['max_steps']);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    ds_path=Path(t['deepspeed'])
    if not ds_path.is_absolute():ds_path=Path(config).resolve().parent/ds_path
    spec={'implementation_sha256':sha256(__file__),'deepspeed_sha256':sha256(ds_path),
          'init_adapter_sha256':adapter_hash(init_adapter),'model':cfg['model'],'revision':cfg.get('revision'),'config':cfg,'data_sha256':sha256(states_path),
          'mode':mode,'init_adapter':str(init_adapter) if init_adapter else None,'seed':seed,'max_steps':steps,
          'world_size':world,'cost_weight':cost_weight,'pilot':pilot}
    fingerprint=digest(spec);meta=out/'experiment.json'
    if meta.exists() and read_json(meta)['signature']!=fingerprint:raise ValueError('Run directory belongs to another experiment')
    if resume=='auto':resume=latest_complete(out,world)
    if resume:
        ck=inspect_checkpoint(resume,world)
        if ck['global_step']>=steps:raise ValueError('Resume checkpoint already reached the requested step count')
    elif list(out.glob('checkpoint-*')):raise ValueError('Existing checkpoints require --resume or a new output')
    settings=dict(output_dir=str(out),max_steps=steps,per_device_train_batch_size=t['micro_batch_size'],
        gradient_accumulation_steps=t['accumulate'],num_generations=t['num_generations'],learning_rate=t['learning_rate'],
        warmup_ratio=.03,max_completion_length=t['max_completion_length'],max_prompt_length=maximum,
        beta=t.get('beta',0.),bf16=True,seed=seed,logging_steps=1,save_steps=8 if pilot else t['save_steps'],
        save_total_limit=t.get('save_total_limit',24),report_to='none',remove_unused_columns=False,
        gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':True},use_vllm=False,
        ds3_gather_for_generation=t['ds3_gather_for_generation'],deepspeed=str(ds_path),
        loss_type='dr_grpo',mask_truncated_completions=True,disable_tqdm=True,
        temperature=.7,top_p=.95,top_k=0,
        model_init_kwargs={'torch_dtype':'bfloat16','attn_implementation':'sdpa','revision':cfg.get('revision')})
    legal={f.name for f in dataclasses.fields(GRPOConfig)}
    if set(settings)-legal:raise RuntimeError(f'Unexpected TRL API; use the versioned environment: {set(settings)-legal}')
    args=GRPOConfig(**settings)
    stop={'requested':False}
    def handle(*unused):stop['requested']=True
    signal.signal(signal.SIGTERM,handle)
    if hasattr(signal,'SIGUSR1'):signal.signal(signal.SIGUSR1,handle)
    class Audit(TrainerCallback):
        def on_step_begin(self,args,state,control,**kw):self.start=time.perf_counter()
        def on_step_end(self,args,state,control,**kw):
            flag=torch.tensor([int(stop['requested'])],device=args.device)
            if dist.is_initialized():dist.all_reduce(flag,op=dist.ReduceOp.MAX)
            if flag.item() or (stop_at_step is not None and state.global_step>=stop_at_step):
                control.should_save=True;control.should_training_stop=True
            if state.global_step>=args.max_steps:control.should_save=True
            if state.is_world_process_zero:
                with (out/'steps.jsonl').open('a') as f:
                    import json
                    f.write(json.dumps({'step':state.global_step,'seconds':time.perf_counter()-self.start,
                        'rank0_peak_gib':torch.cuda.max_memory_allocated()/2**30})+'\n');f.flush()
            return control
        def on_log(self,args,state,control,logs=None,**kw):
            if state.is_world_process_zero and logs:
                import json
                with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps({'step':state.global_step,**logs},default=float)+'\n')
    peft_cfg=LoraConfig(r=16,lora_alpha=32,lora_dropout=0.,target_modules=['q_proj','k_proj','v_proj','o_proj'],task_type='CAUSAL_LM')
    model=cfg['model'];ds_context=None
    if init_adapter:
        ds_context=HfDeepSpeedConfig(str(ds_path))
        base=AutoModelForCausalLM.from_pretrained(cfg['model'],revision=cfg.get('revision'),torch_dtype=torch.bfloat16,
                    attn_implementation='sdpa',trust_remote_code=False)
        model=PeftModel.from_pretrained(base,init_adapter,is_trainable=True);peft_cfg=None
        args.model_init_kwargs=None
    trainer=GRPOTrainer(model=model,args=args,reward_funcs=reward_factory(mode,cost_weight,tokenizer),
        train_dataset=Dataset.from_list(records),processing_class=tokenizer,peft_config=peft_cfg,callbacks=[Audit()])
    if trainer.is_world_process_zero():save_json(meta,{'signature':fingerprint,**spec,'states':len(records)})
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_state()
    if dist.is_initialized():dist.barrier()
    if trainer.is_world_process_zero():
        step=trainer.state.global_step;ck=out/f'checkpoint-{step}'
        info=inspect_checkpoint(ck,world)
        norms=[float(l['grad_norm']) for l in trainer.state.log_history if 'grad_norm' in l]
        if not norms or not all(math.isfinite(x) for x in norms) or not any(x>0 for x in norms):raise ValueError('No verified finite nonzero training gradients')
        save_json(out/'status.json',{'success':step==steps,'interrupted':step<steps,'global_step':step,
                  'expected_steps':steps,'checkpoint':str(ck),'world_size':world,'signature':fingerprint,**info})
    if dist.is_initialized():dist.barrier()


def shutil_disk_free(output):
    import shutil
    p=Path(output).resolve()
    while not p.exists():p=p.parent
    return shutil.disk_usage(p).free/2**30


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--states',required=True);p.add_argument('--output',required=True)
    p.add_argument('--mode',choices=['standard','executor'],default='standard');p.add_argument('--init-adapter');p.add_argument('--resume')
    p.add_argument('--seed',type=int,default=41);p.add_argument('--max-steps',type=int);p.add_argument('--pilot',action='store_true');p.add_argument('--cost-weight',type=float,default=0.);p.add_argument('--stop-at-step',type=int)
    a=p.parse_args();run(a.config,a.states,a.output,a.mode,a.init_adapter,a.resume,a.seed,a.max_steps,a.pilot,a.cost_weight,a.stop_at_step)

if __name__=='__main__':main()
