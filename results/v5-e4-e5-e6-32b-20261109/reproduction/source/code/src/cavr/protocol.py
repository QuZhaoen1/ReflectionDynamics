import re
import math
from collections import Counter
from .data import visible
from refldyn.protocol import prompt_for_state,independent_retry_prompt,parse_decision
from refldyn.answers import extract_final,verify_answer

EVENTS=('COMMIT','REFLECT','INVALID','CENSORED')
ACTION=re.compile(r'^\s*(COMMIT|REFLECT)(?=\s)',re.I)


def action_event(text,ended=False):
    m=ACTION.match(str(text))
    if m:return m.group(1).upper()
    s=str(text).lstrip().upper()
    if ended and s in EVENTS[:2]:return s
    if any(a.startswith(s) for a in EVENTS[:2]):return 'INVALID' if ended else 'CENSORED'
    return 'INVALID'


def wilson(k,n,z=1.959963984540054):
    if n<=0 or not 0<=k<=n:raise ValueError('Invalid binomial count')
    p=k/n;d=1+z*z/n;m=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return max(0.,m-h),min(1.,m+h)


def event_summary(events):
    c=Counter(events);n=len(events)
    if not n or set(c)-set(EVENTS):raise ValueError('Invalid action bank')
    lo,hi=wilson(c['REFLECT'],n);v=c['COMMIT']+c['REFLECT']
    return {'n':n,**{f'p_{k.lower()}':c[k]/n for k in EVENTS},'ci_low':lo,'ci_high':hi,
            'p_reflect_given_valid':c['REFLECT']/v if v else None}


def decision_prompt(r):
    v=visible(r)
    if r.get('task','math')=='code':
        return ('A proposed Python solution is provided. Decide whether to keep it or review it.\n'
                f"Problem:\n{v['question']}\n\nProposed solution:\n{v['initial_answer']}\n\n"
                'Start with COMMIT or REFLECT followed by a newline. For REFLECT provide the complete '
                'corrected implementation in a fenced python block.\nACTION: ')
    return prompt_for_state(v)


def reflect_prompt(r):return decision_prompt(r)+('REFLECT\n' if r.get('task')=='code' else 'REFLECT\nCHECK: ')


def retry_prompt(r):
    if r.get('task')=='code':return f"Write a complete Python implementation in a fenced python block.\n{r['question']}\n"
    return independent_retry_prompt(visible(r))


def draft_prompt(r):
    if r.get('task')=='code':return retry_prompt(r)
    return f"Solve the mathematics problem. Explain the reasoning and end with FINAL: \\boxed{{<answer>}}.\n\nProblem:\n{r['question']}\n\nSolution:\n"


def extract_code(text):
    blocks=re.findall(r'```(?:python|py)?\s*\n(.*?)```',str(text),re.S|re.I)
    return blocks[-1].strip() if blocks else None


def grade(r,text,arm,code_runner=None):
    if arm=='commit':return {'correct':bool(r['initial_correct']),'valid':True,'answer':r['initial_answer']}
    if r.get('task')=='code':
        answer=extract_code(text)
        if not answer:return {'correct':False,'valid':False,'answer':None,'reason':'missing_code_block'}
        if code_runner is None:raise RuntimeError('Code grading requires the isolated runner')
        g=code_runner(r,answer)
        return {**g,'answer':answer}
    if arm=='reflect':
        d=parse_decision('REFLECT\nCHECK: '+text,r['initial_answer']);answer=d.answer;valid=d.valid
    elif arm=='natural':
        d=parse_decision(text,r['initial_answer']);answer=d.answer;valid=d.valid
    else:answer=extract_final(text);valid=answer is not None
    score=verify_answer(r['gold'],answer) if valid else None
    return {'correct':score is True,'valid':bool(valid and score is not None),'answer':answer}
