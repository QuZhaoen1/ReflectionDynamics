from pathlib import Path
import subprocess
from cavr.io import read_json,save_json,sha256,adapter_hash,load_rows,digest
from cavr.features import load
from cavr.train import training_records
from cavr.protocol import draft_prompt
from transformers import AutoTokenizer
from task_sandbox import TaskRunner,verify_files
verify_files()
assert read_json(Path(__file__).resolve().parent/'task-sandbox-equivalence.json')['success']

R=Path(__file__).resolve().parent
ref=read_json(R/'e1-reference.json')
cfg={k:ref[k] for k in ('model_path','revision','engine','decode','batch_policy')}
cfg.update(schema=5,protocol='cavr-v5-vllm-full-20261101-v1',workspace=str(R),
           initial_adapter=ref['checkpoints']['step400'],initial_adapter_hash=ref['adapter_sha256']['step400'],
           feature_encoder=read_json(R/'features/gate_train.npz.json')['encoder'],
           reserve=str(R.parent/'e4-20261101/checkpoint-capacity.reserve'),
           sandbox_image=TaskRunner().image,
           sandbox_source_image=read_json(R/'task-sandbox/manifest.json')['source_image'],
           prior_E2_HF_banks_reused=False,training_engine='unchanged validated TRL GRPO / DeepSpeed ZeRO3, BF16',
           sandbox_backend='task-local chroot+seccomp preferred; development Docker fallback only if startup isolation checks fail; choose once and pin identity',
           latency_semantics='amortized batch wall time; not sequential deployment latency')
cfg['source_sha256']={str(p.relative_to(R)):sha256(p) for p in sorted(R.glob('*.py'))}
cfg['source_sha256'].update({str(p.relative_to(R)):sha256(p) for p in sorted((R/'code/src').rglob('*.py'))})
cfg['source_sha256']['run.sh']=sha256(R/'run.sh')
inputs=[R/'profile.yaml',R/'task-sandbox/manifest.json',R/'task-sandbox/launch.py',R/'task-sandbox/guest.py',R/'task-sandbox-equivalence.json',*sorted((R/'code/configs/hardware').glob('*.json')),
        *sorted((R/'states').glob('*.jsonl')),*sorted((R/'features').glob('*.npz')),
        *sorted((R/'features').glob('*.npz.json')),*sorted((R/'corpus').rglob('*.jsonl')),
        *sorted((R/'corpus').rglob('manifest.json'))]
cfg['input_sha256']={str(p.relative_to(R)):sha256(p) for p in inputs}
assert adapter_hash(cfg['initial_adapter'])==cfg['initial_adapter_hash']
tok=AutoTokenizer.from_pretrained(cfg['model_path'],trust_remote_code=False)
maxtrain=max(len(tok.encode(r['prompt'],add_special_tokens=False)) for r in training_records(load_rows(R/'states/gate_train.jsonl'),'executor'))
assert maxtrain<=3200
for panel in ('gate_train','gate_dev','gate_holdout','validation'):
    rows=load_rows(R/'states'/f'{panel}.jsonl');_,enc,_=load(R/'features'/f'{panel}.npz',rows)
    assert enc==digest(cfg['feature_encoder'])
_,enc,_=load(R/'canary/features.npz',load_rows(R/'canary/states.jsonl'))
assert enc==digest(cfg['feature_encoder'])
maxdraft={}
for kind,split in (('math','test'),('aime','ood'),('gsm8k','test'),('humaneval','ood_code')):
    rows=load_rows(R/'corpus'/kind/f'{split}.jsonl')
    maxdraft[kind]=max(len(tok.encode(draft_prompt(r),add_special_tokens=False)) for r in rows)
    assert maxdraft[kind]<=4096
assert read_json(R/'canary/verification.json')['success']
assert 'OK' in (R/'orchestration-tests.log').read_text()
save_json(R/'config.json',cfg)
(R/'config.sha256').write_text(sha256(R/'config.json')+'\n')
save_json(R/'plan.json',{
    'E4':{'variants':['cavr_online','no_refresh','uniform_executor'],'rounds_per_variant':3,'steps_per_round':40,
          'refresh':'initial + each CAVR/uniform round; no_refresh keeps initial controller',
          'final_evaluation':'heldout and validation, four controller modes'},
    'E5':{'datasets':['math','aime','gsm8k','humaneval'],'executors':['base','step400','cavr_online-round3','no_refresh-round3','uniform_executor-round3'],
          'drafts_per_question':2,'probe_repetitions_per_arm':16,'controller_modes':['cavr','confidence','direct','no_damage'],
          'policy_lock':'written before first test draft; no test fitting; primary budget .3 / cost .05'},
    'E6':{'model':'Qwen2.5-32B Base','training_seeds':[41,73,107],'steps_each':400,'init_adapter':None,
          'trajectory_checkpoints':'Base and 40,80,...,400 per seed','action_banks':'two independent 64-repetition banks',
          'interventions':'16 reflect + 16 retry per state, per checkpoint','aggregate':'seed-level uncertainty'},
    'storage':{'working_budget_gib':96,'checkpoint_projection_gib':31,'retained_reserve_gib':48,'new_root':str(R)},
    'execution':{'gpus':8,'time_limit':None,'consecutive_stages':True,'resume':'verified receipts + generation batches + full optimizer checkpoints',
                 'on_error':'record failing stage and exit; no unbounded retry or deletion',
                 'code_judge':'task-local chroot+seccomp, zero capabilities, no network/process creation/file writes; startup tests select and pin backend; authorized development Docker fallback if platform denies local isolation',
                 'code_judge_tests':'18 isolation checks + all 164 canonical and 164 incorrect solutions compared against original Docker'},
    'scope_note':'E6 here is the explicitly authorized three-seed 32B study; 7B/14B scale extensions are not in this job.'})
save_json(R/'preflight.json',{'success':True,'source_files':len(cfg['source_sha256']),'input_files':len(inputs),
          'generation_canary':read_json(R/'canary/verification.json'),'feature_encoder_match':True,
          'project_tests':87,'orchestration_tests':5,'max_train_prompt_tokens':maxtrain,'max_draft_prompt_tokens':maxdraft,
          'sandbox_image':cfg['sandbox_image'],'config_sha256':sha256(R/'config.json')})
print((R/'preflight.json').read_text())
