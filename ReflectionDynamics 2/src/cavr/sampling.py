from .io import EventLog,digest,seed_for,unique,sha256
from pathlib import Path
from .data import validate_states,state_hash
from .protocol import decision_prompt,reflect_prompt,retry_prompt,draft_prompt,grade,action_event


def probe(states,backend,output,checkpoint,bank='fit',reps=16,arms=('reflect','retry'),seed=41,shard=0,shards=1,code_runner=None):
    validate_states(states,fixtures=True)
    if reps<1 or shards<1 or not 0<=shard<shards or set(arms)-{'reflect','retry','action','natural'}:raise ValueError('Invalid sampling parameters')
    panel=sorted([r for r in states if seed_for(r['id'])%shards==shard],key=lambda r:r['id'])
    spec={'schema':5,'states':digest([(r['id'],state_hash(r),r['initial_correct']) for r in panel]),'executor':backend.identity,
          'checkpoint':checkpoint,'bank':bank,'seed':seed,'shard':shard,'shards':shards,'arms':list(arms),
          'grading':digest([(r['id'],r.get('gold'),r.get('hidden_tests'),r.get('entry_point')) for r in panel]),
          'sandbox':getattr(code_runner,'image',None),'source':sha256(__file__)}
    with EventLog(output,spec) as log:
        for r in panel:
            for arm in arms:
                for k in range(reps):
                    key=digest([log.signature,r['id'],arm,k])
                    if key in log.rows:continue
                    s=seed_for(seed,bank,r['id'],arm,k)
                    prompt=reflect_prompt(r) if arm=='reflect' else retry_prompt(r) if arm=='retry' else decision_prompt(r)
                    result=backend.generate(prompt,s,backend.decode.action_tokens if arm=='action' else None)
                    event={'event_id':key,'id':r['id'],'problem_id':r['problem_id'],'split':r['split'],'state_hash':state_hash(r),
                           'checkpoint':checkpoint,'executor':digest(backend.identity),'bank':bank,'arm':arm,'sample':k,'seed':s,
                           'initial_correct':r['initial_correct'],**result}
                    if arm in ('action','natural'):event['action']=action_event(result['text'],result['ended'])
                    if arm!='action':event.update(grade(r,result['text'],arm,code_runner))
                    log.append(event)
        log.complete(len(panel)*len(arms)*reps)


def draft(questions,backend,output,samples=8,seed=41,shard=0,shards=1,code_runner=None):
    unique(questions)
    if samples<1 or not 0<=shard<shards:raise ValueError('Invalid draft parameters')
    panel=sorted([r for r in questions if seed_for(r['id'])%shards==shard],key=lambda r:r['id'])
    spec={'schema':5,'questions':digest(panel),'model':backend.identity,'seed':seed,'shard':shard,'shards':shards,'stage':'draft','sandbox':getattr(code_runner,'image',None),'source':sha256(__file__)}
    with EventLog(output,spec) as log:
        for r in panel:
            for k in range(samples):
                key=digest([log.signature,r['id'],k])
                if key in log.rows:continue
                result=backend.generate(draft_prompt(r),seed_for(seed,'draft',r['id'],k));g=grade(r,result['text'],'retry',code_runner)
                log.append({**r,**result,**g,'event_id':key,'sample':k,'id':f"{r['id']}-d{k}",'problem_id':r['id'],
                            'initial_reasoning':result['text'],'initial_answer':g['answer'],'initial_correct':g['correct'],
                            'draft_model':backend.identity['model'],'draft_adapter_sha256':backend.identity['adapter_sha256']})
        log.complete(len(panel)*samples)
