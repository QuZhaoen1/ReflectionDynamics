from .protocol import action_event
from .io import save_json


def run(model,revision,output):
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(model,revision=revision,trust_remote_code=False)
    cases=[]
    for action in ('COMMIT','REFLECT'):
        for prefix in ('',' ','\n','  \n '):
            text=prefix+action+'\n';ids=tok.encode(text,add_special_tokens=False);decoded=tok.decode(ids,skip_special_tokens=True)
            if action_event(decoded,True)!=action:raise ValueError('Tokenizer round trip changed the action event')
            cases.append({'text':text,'ids':ids,'event':action})
    result={'model':model,'revision':revision,'cases':cases,'method':'parser round-trip audit; sampling consistency is tested by independent action banks'}
    save_json(output,result);return result
