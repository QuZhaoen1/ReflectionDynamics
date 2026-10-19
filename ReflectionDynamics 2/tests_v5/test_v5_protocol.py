import pytest
from cavr.protocol import action_event,event_summary,wilson,decision_prompt,extract_code
from cavr.backend import DecodeSpec
from cavr.smoke import fixtures

@pytest.mark.parametrize('text,ended,expected',[
    (' REFLECT\nCHECK: x',False,'REFLECT'),('COMMIT\n',False,'COMMIT'),('\n  reflect\n',False,'REFLECT'),
    ('COMMIT',True,'COMMIT'),('REFLECT',False,'CENSORED'),(' REF',False,'CENSORED'),('',False,'CENSORED'),
    ('',True,'INVALID'),('REF',True,'INVALID'),('REFLECTION\n',False,'INVALID'),('COMMITTING\n',False,'INVALID'),
    ('REFLECT: check',True,'INVALID'),('ACTION: REFLECT\n',True,'INVALID'),(' COMMIT REFLECT',False,'COMMIT')])
def test_event_grammar(text,ended,expected):assert action_event(text,ended)==expected

def test_denominator_and_conditional_probability():
    s=event_summary(['REFLECT','COMMIT','INVALID','CENSORED'])
    assert s['p_reflect']==.25 and s['p_reflect_given_valid']==.5
    assert sum(s[k] for k in ('p_reflect','p_commit','p_invalid','p_censored'))==1

@pytest.mark.parametrize('k,n',[(0,1),(1,1),(3,10),(0,64),(64,64)])
def test_wilson_valid(k,n):
    a,b=wilson(k,n);assert 0<=a<=k/n<=b<=1

@pytest.mark.parametrize('k,n',[(-1,4),(5,4),(0,0)])
def test_wilson_rejects_invalid(k,n):
    with pytest.raises(ValueError):wilson(k,n)

@pytest.mark.parametrize('kwargs',[{'temperature':0},{'top_p':1.1},{'top_k':-1},{'max_new_tokens':0}])
def test_decode_validation(kwargs):
    with pytest.raises(ValueError):DecodeSpec(**kwargs)

def test_no_label_or_test_leakage():
    r=fixtures(1)[0];r.update(gold='SECRET_GOLD',hidden_tests='SECRET_TESTS',probe_category='SECRET_CATEGORY')
    s=decision_prompt(r)
    assert 'SECRET_' not in s

def test_code_block_extraction():assert extract_code('```python\ndef f(): return 1\n```')=='def f(): return 1'
