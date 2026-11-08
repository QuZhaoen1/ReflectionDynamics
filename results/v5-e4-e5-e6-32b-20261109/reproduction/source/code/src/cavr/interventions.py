from .io import unique,digest,seed_for,EventLog
from .data import state_hash,validate_states
from .protocol import reflect_prompt,grade


def plan_from_action_samples(states,events,decision_checkpoint,repetition=0):
    lookup=unique(states);chosen={}
    for e in events:
        if e['checkpoint']==decision_checkpoint and e['arm']=='action' and e['sample']==repetition:
            if e['id'] in chosen:raise ValueError('Duplicate decision sample')
            chosen[e['id']]=e
    if set(chosen)!=set(lookup):raise ValueError('Decision bank does not cover the complete panel')
    plan=[]
    for s in states:
        e=chosen[s['id']]
        if e['state_hash']!=state_hash(s):raise ValueError('State mismatch')
        plan.append({'id':s['id'],'problem_id':s['problem_id'],'state_hash':state_hash(s),
                     'action':'REFLECT' if e['action']=='REFLECT' else 'COMMIT','raw_action':e['action'],
                     'decision_checkpoint':decision_checkpoint,'decision_sample':repetition,
                     'invalid_fallback':'COMMIT','controller_signature':e['executor']})
    return plan


def execute(states,plan,backend,output,executor_checkpoint,reps=4,seed=313,code_runner=None):
    validate_states(states,fixtures=True);lookup=unique(plan)
    if set(lookup)!=set(r['id'] for r in states):raise ValueError('Plan/state coverage mismatch')
    spec={'schema':5,'stage':'actual_bridge_intervention','executor':backend.identity,'checkpoint':executor_checkpoint,
          'plan':digest(plan),'seed':seed,'states':digest([(s['id'],state_hash(s)) for s in states])}
    with EventLog(output,spec) as log:
        for s in states:
            p=lookup[s['id']]
            if p['state_hash']!=state_hash(s) or p['action'] not in ('COMMIT','REFLECT'):raise ValueError('Invalid intervention plan')
            for k in range(reps):
                eid=digest([log.signature,s['id'],k])
                if eid in log.rows:continue
                if p['action']=='COMMIT':g=grade(s,'','commit');result={'text':'','tokens':0,'prompt_tokens':0,'seconds':0.}
                else:
                    result=backend.generate(reflect_prompt(s),seed_for(seed,'bridge',s['id'],k,p['controller_signature']))
                    g=grade(s,result['text'],'reflect',code_runner)
                log.append({'event_id':eid,'id':s['id'],'problem_id':s['problem_id'],'state_hash':state_hash(s),
                    'executor_checkpoint':executor_checkpoint,'controller_signature':p['controller_signature'],
                    'action':p['action'],'sample':k,'initial_correct':s['initial_correct'],**result,**g})
        log.complete(len(states)*reps)
