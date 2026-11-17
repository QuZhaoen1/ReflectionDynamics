import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from .io import read_json,save_json,load_rows,save_rows,digest


class Workflow:
    def __init__(self,experiment,execute=False,gpus=None):
        self.cfg=read_json(experiment);self.root=Path(self.cfg['workspace']);self.execute=execute
        self.gpus=(gpus or os.getenv('GPU_IDS','0,1,2,3,4,5,6,7')).split(',')
        per=int(os.getenv('GPUS_PER_REPLICA','2'))
        if len(self.gpus)%per:raise ValueError('GPU count must be divisible by GPUs per inference replica')
        self.groups=[','.join(self.gpus[i:i+per]) for i in range(0,len(self.gpus),per)]
        self.memory=os.getenv('MEMORY_GIB','72');self.steps=[];self.logdir=self.root/'logs';self.logdir.mkdir(parents=True,exist_ok=True)
    def command(self,args,env=None):
        cmd=[sys.executable,'-m','cavr.cli',*map(str,args)]
        self.steps.append({'command':cmd,'CUDA_VISIBLE_DEVICES':(env or {}).get('CUDA_VISIBLE_DEVICES')})
        print(shlex.join(cmd),flush=True)
        if not self.execute:return
        log=self.logdir/(digest(cmd)[:16]+'.log')
        with log.open('a') as f:
            f.write('\n'+shlex.join(cmd)+'\n');f.flush()
            result=subprocess.run(cmd,env={**os.environ,**(env or {})},stdout=f,stderr=subprocess.STDOUT)
        if result.returncode:
            print(log.read_text(errors='replace')[-10000:],file=sys.stderr)
            raise RuntimeError(f'Command failed ({result.returncode}); see {log}')
    def model_args(self,ck='base'):
        if ck not in self.cfg['checkpoints']:raise ValueError(f'Missing checkpoint {ck}; export its adapter before scheduling')
        args=['--model',self.cfg['model'],'--memory-gib',self.memory]
        if self.cfg.get('revision'):args+=['--revision',self.cfg['revision']]
        if self.cfg['checkpoints'][ck]:args+=['--adapter',self.cfg['checkpoints'][ck]]
        return args
    def probes(self,panel,ck,bank,reps=16,arms='reflect,retry'):
        prefix=self.root/'probes'/ck/panel/bank
        args=['probe','--states',self.cfg['states'][panel],'--checkpoint',ck,'--bank',bank,'--reps',reps,'--arms',arms,*self.model_args(ck)]
        jobs=[]
        for i,devices in enumerate(self.groups):
            dest=str(prefix)+f'-{i}.jsonl';cmd=[sys.executable,'-m','cavr.cli',*map(str,args),
                '--shard-index',str(i),'--num-shards',str(len(self.groups)),'--output',dest]
            self.steps.append({'command':cmd,'CUDA_VISIBLE_DEVICES':devices});print(shlex.join(cmd),flush=True)
            if self.execute:
                log=self.logdir/(digest(cmd)[:16]+'.log');f=log.open('a')
                jobs.append((subprocess.Popen(cmd,env={**os.environ,'CUDA_VISIBLE_DEVICES':devices},stdout=f,stderr=subprocess.STDOUT),f,log))
        failures=[]
        try:
            for proc,f,log in jobs:
                code=proc.wait();f.close()
                if code:failures.append(str(log))
        except BaseException:
            for proc,f,_ in jobs:
                if proc.poll() is None:proc.terminate()
                f.close()
            raise
        if failures:raise RuntimeError('Inference shards failed: '+', '.join(failures))
        return str(prefix)+'-*.jsonl'
    def features(self,panel):
        p=self.root/'features'/f'{panel}.npz'
        self.command(['features','--states',self.cfg['states'][panel],'--output',p,'--mode','hidden',*self.model_args('base')],{'CUDA_VISIBLE_DEVICES':self.groups[0]})
        return str(p)
    def frozen(self,ck='step400',reps=16,modes=('cavr','confidence','direct','no_damage')):
        panels=('gate_train','gate_dev','gate_holdout','validation');banks={};features={}
        for panel in panels:
            banks[panel]=self.probes(panel,ck,'fit' if panel in ('gate_train','gate_dev') else 'score',reps)
            features[panel]=self.features(panel)
        for mode in modes:
            model=self.root/'controllers'/ck/mode
            self.command(['fit','--train-states',self.cfg['states']['gate_train'],'--dev-states',self.cfg['states']['gate_dev'],
                '--train-features',features['gate_train'],'--dev-features',features['gate_dev'],
                '--train-events',banks['gate_train'],'--dev-events',banks['gate_dev'],'--checkpoint',ck,
                '--mode',mode,'--output',model,'--resume'])
            for panel in ('gate_holdout','validation'):
                pred=self.root/'predictions'/ck/f'{mode}-{panel}.jsonl'
                self.command(['predict','--states',self.cfg['states'][panel],'--features',features[panel],'--controller',model,'--output',pred])
                self.command(['evaluate','--states',self.cfg['states'][panel],'--events',banks[panel],'--predictions',pred,
                    '--bank','score','--checkpoint',ck,'--output',self.root/'reports'/'frozen'/ck/mode/panel])
    def measurement(self,checkpoints):
        paths=[]
        for ck in checkpoints:
            a=self.probes('validation',ck,'actionA',64,'action');b=self.probes('validation',ck,'actionB',64,'action')
            values=self.probes('validation',ck,'score',16)
            self.command(['audit-actions','--states',self.cfg['states']['validation'],'--events-a',a,'--events-b',b,
                          '--output',self.root/'reports'/'measurement'/f'{ck}-audit.json'])
            m=self.root/'reports'/'measurement'/f'{ck}-selection.json';paths.append(m)
            self.command(['selection','--states',self.cfg['states']['validation'],'--events',values,'--actions',a,
                          '--checkpoint',ck,'--bank','score','--output',m])
        self.command(['dynamics','--metrics',*paths,'--output',self.root/'reports'/'dynamics'])
    def crossed(self,checkpoints):
        for actor in checkpoints:
            a=self.probes('validation',actor,'actionA',64,'action');plan=self.root/'plans'/f'actor-{actor}.jsonl'
            self.command(['plan','--states',self.cfg['states']['validation'],'--actions',a,'--checkpoint',actor,'--output',plan])
            for executor in checkpoints:
                self.command(['execute-plan','--states',self.cfg['states']['validation'],'--plan',plan,'--checkpoint',executor,
                    '--output',self.root/'interventions'/f'{actor}--{executor}.jsonl','--reps',4,*self.model_args(executor)],{'CUDA_VISIBLE_DEVICES':self.groups[0]})
        self.command(['summarize-interventions','--events',str(self.root/'interventions'/'*.jsonl'),'--output',self.root/'reports'/'crossed_actual'])
    def ood(self,ck,kind):
        corpus=self.root/'corpus'/kind
        self.command(['ingest','--kind',kind,'--output',corpus])
        split='test' if kind in ('math','gsm8k') else 'ood_code' if kind=='humaneval' else 'ood'
        questions=corpus/f'{split}.jsonl';drafts=self.root/'drafts'/f'{kind}.jsonl';states=self.root/'states'/f'{kind}.jsonl'
        extra=['--code-sandbox'] if kind=='humaneval' else []
        self.command(['draft','--questions',questions,'--output',drafts,'--samples',2,*extra,*self.model_args('base')],{'CUDA_VISIBLE_DEVICES':self.groups[0]})
        self.command(['states','--drafts',drafts,'--output',states])
        self.cfg['states'][kind]=str(states);features=self.features(kind)
        events=self.root/'probes'/ck/kind/'score.jsonl'
        self.command(['probe','--states',states,'--checkpoint',ck,'--bank','score','--reps',16,'--arms','reflect,retry',
                      '--output',events,*extra,*self.model_args(ck)],{'CUDA_VISIBLE_DEVICES':self.groups[0]})
        for mode in ('cavr','confidence','direct','no_damage'):
            model=self.root/'controllers'/ck/mode;pred=self.root/'predictions'/ck/f'{kind}-{mode}.jsonl'
            self.command(['predict','--states',states,'--features',features,'--controller',model,'--output',pred])
            self.command(['evaluate','--states',states,'--events',events,'--predictions',pred,'--bank','score',
                          '--checkpoint',ck,'--output',self.root/'reports'/'ood'/kind/mode])
    def online(self,ck='step400',rounds=3,profile='configs/hardware/a100_8x80.yaml',uniform=False,no_refresh=False):
        import yaml
        profile_cfg=yaml.safe_load(Path(profile).read_text());world=profile_cfg['hardware']['world_size']
        if world!=len(self.gpus):raise ValueError('Online training GPU list does not match hardware profile')
        origin=ck;current=ck;seed0=int(os.getenv('TRAIN_SEED','41'))
        variant=('uniform_executor' if uniform else 'no_refresh' if no_refresh else 'cavr_online')+f'-seed{seed0}'
        for rnd in range(1,rounds+1):
            head_ck=origin if no_refresh else current
            if rnd==1 or not no_refresh:self.frozen(current,modes=('cavr',))
            states=self.cfg['states']['gate_train'];features=self.root/'features'/'gate_train.npz'
            pred=self.root/'predictions'/variant/f'round{rnd}-train.jsonl'
            self.command(['predict','--states',states,'--features',features,'--controller',self.root/'controllers'/head_ck/'cavr',
                          '--output',pred,'--allow-seen'])
            selected=self.root/'online'/variant/f'round{rnd}-states.jsonl'
            if uniform:
                if self.execute:save_rows(selected,load_rows(states))
            else:self.command(['route','--states',states,'--predictions',pred,'--output',selected,'--seed',seed0+rnd])
            run=self.root/'runs'/variant/f'round{rnd}';adapter=self.root/'adapters'/variant/f'round{rnd}'
            cmd=[sys.executable,'-m','torch.distributed.run','--standalone',f'--nproc_per_node={world}',
                 '-m','cavr.train','--config',profile,'--states',str(selected),'--output',str(run),'--mode','executor',
                 '--seed',str(seed0+rnd),'--max-steps','40','--cost-weight','.05']
            if self.cfg['checkpoints'][current]:cmd+=['--init-adapter',self.cfg['checkpoints'][current]]
            if self.execute and (run/'status.json').exists() and read_json(run/'status.json').get('success'):pass
            else:
                if self.execute and list(run.glob('checkpoint-*')):cmd+=['--resume','auto']
                print(shlex.join(cmd),flush=True);self.steps.append({'command':cmd,'CUDA_VISIBLE_DEVICES':','.join(self.gpus)})
                if self.execute:subprocess.run(cmd,env={**os.environ,'CUDA_VISIBLE_DEVICES':','.join(self.gpus)},check=True)
            self.command(['export-adapter','--checkpoint-path',run/'checkpoint-40','--output',adapter])
            current=f'{variant}-round{rnd}';self.cfg['checkpoints'][current]=str(adapter)
        if no_refresh:
            for panel in ('gate_holdout','validation'):
                events=self.probes(panel,current,'score',16);features=self.features(panel)
                pred=self.root/'predictions'/variant/f'{panel}-stale.jsonl'
                self.command(['predict','--states',self.cfg['states'][panel],'--features',features,
                              '--controller',self.root/'controllers'/origin/'cavr','--output',pred])
                self.command(['evaluate','--states',self.cfg['states'][panel],'--events',events,'--predictions',pred,
                              '--checkpoint',current,'--bank','score','--allow-drift','--output',self.root/'reports'/variant/panel])
        else:self.frozen(current,modes=('cavr','confidence','direct'))
        save_json(self.root/f'experiment-{variant}.json',self.cfg)
    def save_plan(self,name):save_json(self.root/'plans'/f'{name}-commands.json',{'steps':self.steps,'executed':self.execute})
