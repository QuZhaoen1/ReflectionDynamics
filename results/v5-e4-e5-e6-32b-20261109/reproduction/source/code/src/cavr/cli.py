import argparse
import os
from .io import load_rows,save_rows,save_json,read_json,repair_tail,load_events


def model_arguments(p):
    p.add_argument('--model',default='Qwen/Qwen2.5-32B');p.add_argument('--revision',default='1818d35814b8319459f4bd55ed1ac8709630f003')
    p.add_argument('--adapter');p.add_argument('--memory-gib',type=int,default=72)
    p.add_argument('--temperature',type=float,default=.7);p.add_argument('--top-p',type=float,default=.95);p.add_argument('--top-k',type=int,default=0)
    p.add_argument('--max-prompt-tokens',type=int,default=4096);p.add_argument('--max-new-tokens',type=int,default=1024);p.add_argument('--action-tokens',type=int,default=24)


def backend(a):
    from .backend import HFBackend,DecodeSpec
    revision=a.revision
    if a.model!='Qwen/Qwen2.5-32B' and revision=='1818d35814b8319459f4bd55ed1ac8709630f003':revision=None
    return HFBackend(a.model,revision,a.adapter,a.memory_gib,DecodeSpec(a.temperature,a.top_p,a.top_k,a.max_prompt_tokens,a.max_new_tokens,a.action_tokens))


def code_runner(a):
    if getattr(a,'code_sandbox',False):
        from .sandbox import DockerRunner
        return DockerRunner()
    return None


def build_parser():
    p=argparse.ArgumentParser(prog='rd5');sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('init');q.add_argument('--legacy-root',required=True);q.add_argument('--workspace',required=True);q.add_argument('--seed',type=int,default=41)
    q=sub.add_parser('ingest');q.add_argument('--kind',choices=['math','gsm8k','aime','humaneval'],required=True);q.add_argument('--output',required=True);q.add_argument('--revision')
    q=sub.add_parser('states');q.add_argument('--drafts',required=True,nargs='+');q.add_argument('--output',required=True);q.add_argument('--max-per-problem',type=int,default=0)
    for name in ('probe','draft','features','execute-plan'):
        q=sub.add_parser(name);model_arguments(q);q.add_argument('--output',required=True)
        if name=='draft':q.add_argument('--questions',required=True);q.add_argument('--samples',type=int,default=8)
        else:q.add_argument('--states',required=True)
        if name in ('probe','draft'):
            q.add_argument('--seed',type=int,default=41);q.add_argument('--shard-index',type=int,default=0);q.add_argument('--num-shards',type=int,default=1)
        if name=='probe':
            q.add_argument('--checkpoint',required=True);q.add_argument('--bank',default='fit');q.add_argument('--reps',type=int,default=16);q.add_argument('--arms',default='reflect,retry')
        if name=='features':q.add_argument('--mode',choices=['hash','hidden'],default='hidden');q.add_argument('--dim',type=int,default=1024)
        if name=='execute-plan':q.add_argument('--plan',required=True);q.add_argument('--checkpoint',required=True);q.add_argument('--reps',type=int,default=4);q.add_argument('--seed',type=int,default=313)
        if name!='features':q.add_argument('--code-sandbox',action='store_true')
    q=sub.add_parser('fit')
    for key in ('train-states','dev-states','train-features','dev-features'):q.add_argument('--'+key,required=True)
    for key in ('train-events','dev-events'):q.add_argument('--'+key,required=True,nargs='+')
    q.add_argument('--output',required=True);q.add_argument('--checkpoint',required=True);q.add_argument('--bank',default='fit')
    q.add_argument('--mode',choices=['cavr','confidence','direct','no_damage'],default='cavr');q.add_argument('--seeds',type=int,nargs='+',default=[41,73,107])
    q.add_argument('--epochs',type=int,default=120);q.add_argument('--hidden',type=int,default=128);q.add_argument('--lr',type=float,default=.001);q.add_argument('--patience',type=int,default=15)
    q.add_argument('--batch-size',type=int,default=64);q.add_argument('--device',default='cpu');q.add_argument('--resume',action='store_true')
    q=sub.add_parser('predict');q.add_argument('--states',required=True);q.add_argument('--features',required=True);q.add_argument('--controller',required=True);q.add_argument('--output',required=True);q.add_argument('--allow-seen',action='store_true')
    q=sub.add_parser('evaluate');q.add_argument('--states',required=True);q.add_argument('--events',required=True,nargs='+');q.add_argument('--predictions',required=True);q.add_argument('--output',required=True)
    q.add_argument('--bank',default='score');q.add_argument('--checkpoint',required=True);q.add_argument('--budgets',type=float,nargs='+',default=[.1,.3,.5,.7]);q.add_argument('--cost-weights',type=float,nargs='+',default=[0.,.05,.2]);q.add_argument('--risk',type=float,default=0.);q.add_argument('--bootstrap',type=int,default=3000);q.add_argument('--allow-drift',action='store_true')
    q=sub.add_parser('audit-actions');q.add_argument('--states',required=True);q.add_argument('--events-a',required=True,nargs='+');q.add_argument('--events-b',required=True,nargs='+');q.add_argument('--output',required=True)
    q=sub.add_parser('selection');q.add_argument('--states',required=True);q.add_argument('--events',required=True,nargs='+');q.add_argument('--actions',required=True,nargs='+');q.add_argument('--bank',default='score');q.add_argument('--checkpoint',required=True);q.add_argument('--cost-weight',type=float,default=0.);q.add_argument('--output',required=True)
    q=sub.add_parser('dynamics');q.add_argument('--metrics',required=True,nargs='+');q.add_argument('--output',required=True)
    q=sub.add_parser('seeds');q.add_argument('--reports',required=True,nargs='+');q.add_argument('--output',required=True)
    q=sub.add_parser('plan');q.add_argument('--states',required=True);q.add_argument('--actions',required=True,nargs='+');q.add_argument('--checkpoint',required=True);q.add_argument('--sample',type=int,default=0);q.add_argument('--output',required=True)
    q=sub.add_parser('gate-plan');q.add_argument('--predictions',required=True);q.add_argument('--budget',type=float);q.add_argument('--cost-weight',type=float,default=.05);q.add_argument('--risk',type=float,default=0.);q.add_argument('--output',required=True)
    q=sub.add_parser('route');q.add_argument('--states',required=True);q.add_argument('--predictions',required=True);q.add_argument('--epsilon',type=float,default=.15);q.add_argument('--temperature',type=float,default=.05);q.add_argument('--seed',type=int,default=41);q.add_argument('--cost-weight',type=float,default=.05);q.add_argument('--output',required=True)
    q=sub.add_parser('export-adapter');q.add_argument('--checkpoint-path',required=True);q.add_argument('--output',required=True)
    q=sub.add_parser('repair-tail');q.add_argument('--path',required=True)
    q=sub.add_parser('workflow');q.add_argument('--experiment',required=True);q.add_argument('--stage',choices=['frozen','measurement','crossed','ood','online'],required=True)
    q.add_argument('--checkpoint',default='step400');q.add_argument('--checkpoints',nargs='+',default=['base','step160','step240','step400']);q.add_argument('--kind',choices=['math','gsm8k','aime','humaneval'],default='aime')
    q.add_argument('--execute',action='store_true');q.add_argument('--gpus');q.add_argument('--profile',default='configs/hardware/a100_8x80.yaml');q.add_argument('--rounds',type=int,default=3);q.add_argument('--uniform',action='store_true');q.add_argument('--no-refresh',action='store_true')
    q=sub.add_parser('legacy-audit');q.add_argument('--observations',required=True);q.add_argument('--output',required=True)
    q=sub.add_parser('summarize-interventions');q.add_argument('--events',nargs='+',required=True);q.add_argument('--output',required=True)
    q=sub.add_parser('tokenizer-audit');q.add_argument('--model',default='Qwen/Qwen2.5-32B');q.add_argument('--revision',default='1818d35814b8319459f4bd55ed1ac8709630f003');q.add_argument('--output',required=True)
    q=sub.add_parser('smoke');q.add_argument('--output',default='outputs/smoke_v5')
    q=sub.add_parser('preflight');q.add_argument('--profile',default='configs/hardware/a100_8x80.yaml');q.add_argument('--output',default='outputs/preflight.json')
    return p


def main(argv=None):
    a=build_parser().parse_args(argv);c=a.command;result=None
    if c=='init':
        from .data import init_workspace
        result=init_workspace(a.legacy_root,a.workspace,a.seed)
    elif c=='ingest':
        from .corpus import ingest
        result=ingest(a.kind,a.output,a.revision)
    elif c=='states':
        from .data import select_states
        r=select_states(load_rows(a.drafts),a.max_per_problem);save_rows(a.output,r);result={'states':len(r)}
    elif c=='probe':
        from .sampling import probe
        probe(load_rows(a.states),backend(a),a.output,a.checkpoint,a.bank,a.reps,tuple(a.arms.split(',')),a.seed,a.shard_index,a.num_shards,code_runner(a))
    elif c=='draft':
        from .sampling import draft
        draft(load_rows(a.questions),backend(a),a.output,a.samples,a.seed,a.shard_index,a.num_shards,code_runner(a))
    elif c=='features':
        from .features import extract
        result=extract(load_rows(a.states),a.output,a.mode,None if a.mode=='hash' else backend(a),a.dim)
    elif c=='fit':
        from .controller import fit
        result=fit(load_rows(a.train_states),load_rows(a.dev_states),a.train_features,a.dev_features,load_events(a.train_events),load_events(a.dev_events),
            a.output,a.checkpoint,a.bank,a.mode,tuple(a.seeds),a.epochs,a.hidden,a.lr,a.patience,a.batch_size,a.device,a.resume)
    elif c=='predict':
        from .controller import predict
        r=predict(load_rows(a.states),a.features,a.controller,a.allow_seen);save_rows(a.output,r);result={'predictions':len(r)}
    elif c=='evaluate':
        from .evaluate import offline
        r=offline(load_rows(a.states),load_events(a.events),load_rows(a.predictions),a.output,a.bank,a.checkpoint,a.budgets,a.cost_weights,a.risk,a.bootstrap,allow_drift=a.allow_drift);result={'policies':len(r)}
    elif c=='audit-actions':
        from .evaluate import action_audit
        result=action_audit(load_rows(a.states),load_events(a.events_a),load_events(a.events_b),a.output);result={k:v for k,v in result.items() if k!='state_estimates'}
    elif c=='selection':
        from .evaluate import selection_metrics
        result=selection_metrics(load_rows(a.states),load_events(a.events),load_events(a.actions),a.bank,a.checkpoint,a.cost_weight);save_json(a.output,result);result={k:v for k,v in result.items() if not isinstance(v,list)}
    elif c=='dynamics':
        from .dynamics import analyze
        result=analyze(a.metrics,a.output)
    elif c=='seeds':
        from .dynamics import compare_training_seeds
        result=compare_training_seeds(a.reports,a.output)
    elif c=='plan':
        from .interventions import plan_from_action_samples
        r=plan_from_action_samples(load_rows(a.states),load_events(a.actions),a.checkpoint,a.sample);save_rows(a.output,r)
    elif c=='gate-plan':
        from .policy import decisions
        save_rows(a.output,decisions(load_rows(a.predictions),a.cost_weight,a.risk,a.budget))
    elif c=='execute-plan':
        from .interventions import execute
        execute(load_rows(a.states),load_rows(a.plan),backend(a),a.output,a.checkpoint,a.reps,a.seed,code_runner(a))
    elif c=='route':
        from .route import route
        r,result=route(load_rows(a.states),load_rows(a.predictions),a.epsilon,a.temperature,a.seed,a.cost_weight);save_rows(a.output,r);save_json(a.output+'.routing.json',result)
    elif c=='export-adapter':
        from .checkpoint import export_adapter
        result=export_adapter(a.checkpoint_path,a.output)
    elif c=='repair-tail':result=repair_tail(a.path)
    elif c=='workflow':
        from .workflow import Workflow
        w=Workflow(a.experiment,a.execute,a.gpus)
        if a.stage=='frozen':w.frozen(a.checkpoint)
        elif a.stage=='measurement':w.measurement(a.checkpoints)
        elif a.stage=='crossed':w.crossed(a.checkpoints)
        elif a.stage=='ood':w.ood(a.checkpoint,a.kind)
        else:w.online(a.checkpoint,a.rounds,a.profile,a.uniform,a.no_refresh)
        w.save_plan(a.stage)
    elif c=='legacy-audit':
        from .reporting import legacy_audit
        r=legacy_audit(load_rows(a.observations),a.output);result={'checkpoints':len(r)}
    elif c=='summarize-interventions':
        from .reporting import summarize_interventions
        r=summarize_interventions(load_events(a.events),a.output);result={'combinations':len(r)}
    elif c=='tokenizer-audit':
        from .tokenizer_audit import run
        result=run(a.model,a.revision,a.output)
    elif c=='smoke':
        from .smoke import run
        result=run(a.output)
    elif c=='preflight':
        from .preflight import run
        result=run(a.profile,a.output)
    if result is not None:
        import json
        print(json.dumps(result,ensure_ascii=False,indent=2,default=str))

if __name__=='__main__':main()
