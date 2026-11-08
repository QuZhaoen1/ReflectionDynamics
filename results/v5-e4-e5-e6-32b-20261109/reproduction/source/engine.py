"""Versioned E1 remeasurement. Never reads HF intervention outputs."""
import argparse
import importlib.metadata
import json
import os
import signal
import subprocess
import time
from pathlib import Path
from cavr.io import digest, seed_for, sha256, load_rows, read_json, save_json, adapter_hash
from cavr.data import state_hash, validate_states
from cavr.protocol import decision_prompt, reflect_prompt, retry_prompt, draft_prompt, action_event, grade
from batch_store import BatchStore

STOP = False


def stop(*_):
    global STOP
    STOP = True


def plan_for(panel, bank, reps):
    arms = ('reflect', 'retry') if bank == 'score' else ('action',)
    return [{'id': s['id'], 'arm': arm, 'sample': k,
             'seed': seed_for(41, bank, s['id'], arm, k)}
            for s in panel for arm in arms for k in range(reps)]


class Engine:
    def __init__(self, cfg):
        import torch
        import vllm
        from vllm import LLM
        from transformers import AutoTokenizer
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg['model_path'], trust_remote_code=False)
        self.llm = LLM(model=cfg['model_path'], generation_config='vllm', trust_remote_code=False,
                       seed=41, **cfg['engine'])
        self.runtime = {'vllm': vllm.__version__, 'torch': torch.__version__, 'cuda': torch.version.cuda,
                        'gpu': torch.cuda.get_device_name(0),
                        'capability': list(torch.cuda.get_device_capability(0)),
                        'transformers': importlib.metadata.version('transformers'),
                        'math_verify': importlib.metadata.version('math-verify')}
        self.runtime['driver'] = subprocess.check_output(['nvidia-smi','--query-gpu=driver_version',
                                       '--format=csv,noheader'],text=True).splitlines()[0].strip()
        self.lora = None
        self.prompts = {}

    def select(self, checkpoint):
        from vllm.lora.request import LoRARequest
        adapter = self.cfg['checkpoints'][checkpoint]
        actual = adapter_hash(adapter)
        if actual != self.cfg['adapter_sha256'][checkpoint]:
            raise ValueError('Adapter changed since experiment initialization')
        self.lora = None if adapter is None else LoRARequest(checkpoint, 1, adapter)
        return {'model': self.cfg['model_path'], 'revision': self.cfg['revision'],
                'adapter_sha256': actual, 'protocol': 'cavr-v5-vllm-full-20261101-v1',
                'engine': self.cfg['engine'], 'decode': self.cfg['decode'],
                'runtime': self.runtime, 'source_sha256': self.cfg['source_sha256'],
                'seed_policy': 'per-request seed_for(41, bank, state_id, arm, sample)',
                'batch_policy': self.cfg['batch_policy'],
                'token_accounting': 'generated tokens including terminal EOS when omitted by vLLM',
                'latency_semantics': 'batch generation wall seconds / batch request count; NOT sequential latency',
                'reproducibility': 'committed batches immutable; uncommitted work may change on replay'}

    def generate(self, requests, lookup):
        from vllm import SamplingParams
        prompts = []
        params = []
        for req in requests:
            k = (req['id'], req['arm'])
            if k not in self.prompts:
                s = lookup[req['id']]
                prompt = draft_prompt(s) if req['arm']=='draft' else reflect_prompt(s) if req['arm']=='reflect' else retry_prompt(s) if req['arm']=='retry' else decision_prompt(s)
                ids = self.tokenizer.encode(prompt, add_special_tokens=False)
                if len(ids) > 4096:
                    raise ValueError('Prompt exceeds protocol budget; truncation prohibited')
                self.prompts[k] = ids
            prompts.append({'prompt_token_ids': self.prompts[k]})
            params.append(SamplingParams(temperature=.7, top_p=.95, top_k=-1, repetition_penalty=1.,
                          seed=req['seed'], max_tokens=24 if req['arm']=='action' else 1024,
                          stop_token_ids=[151643], skip_special_tokens=True))
        started = time.perf_counter()
        generated = self.llm.generate(prompts, params, lora_request=self.lora, use_tqdm=False)
        elapsed = time.perf_counter()-started
        if len(generated) != len(requests):
            raise RuntimeError('Engine returned wrong number of responses')
        rows = []
        for out, prompt in zip(generated, prompts):
            if list(out.prompt_token_ids) != prompt['prompt_token_ids']:
                raise RuntimeError('Engine reordered request outputs')
            c = out.outputs[0]
            if c.finish_reason not in ('stop', 'length'):
                raise RuntimeError(f'Unexpected generation termination: {c.finish_reason}')
            token_ids = list(c.token_ids)
            ended = c.finish_reason == 'stop'
            omitted_eos = ended and (not token_ids or token_ids[-1] != 151643) and c.stop_reason in (None, 151643)
            rows.append({'text': c.text, 'tokens': len(token_ids)+int(omitted_eos),
                         'vllm_output_tokens': len(token_ids), 'prompt_tokens': len(out.prompt_token_ids),
                         'seconds': elapsed/len(requests), 'batch_seconds': elapsed,
                         'batch_requests': len(requests), 'ended': ended,
                         'truncated': c.finish_reason=='length', 'finish_reason': c.finish_reason,
                         'stop_reason': c.stop_reason,
                         'latency_semantics': 'amortized_batch_wall_seconds_not_sequential'})
        return rows
