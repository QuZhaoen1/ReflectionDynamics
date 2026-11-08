import copy
import math
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from .io import digest,save_json,read_json,sha256
from .data import validate_states,disjoint,question_key,state_hash
from .features import load as load_features
from .bank import targets

MODES=('cavr','confidence','direct','no_damage')


class ValueHead(nn.Module):
    def __init__(self,dim,hidden=128):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(dim,hidden),nn.LayerNorm(hidden),nn.GELU(),nn.Dropout(.1),
                               nn.Linear(hidden,hidden),nn.GELU(),nn.Linear(hidden,7))
    def forward(self,x):return self.net(x)


def weighted_mean(x,w):return (x*w).sum()/w.sum().clamp_min(1e-8)


def loss_fn(raw,y,n,w,mode='cavr'):
    c=F.binary_cross_entropy_with_logits(raw[:,0],y[:,0],reduction='none')
    r=F.binary_cross_entropy_with_logits(raw[:,1],y[:,1],reduction='none')
    h=F.binary_cross_entropy_with_logits(raw[:,2],y[:,2],reduction='none')
    cost=(raw[:,4]-y[:,4]).square();direct=(torch.tanh(raw[:,3])-y[:,3]).square()
    p=torch.sigmoid(raw[:,:3]);u=(1-p[:,0])*p[:,1]-p[:,0]*p[:,2]
    coupled=(u-y[:,3]).square()
    retry=F.binary_cross_entropy_with_logits(raw[:,5],y[:,5],reduction='none')
    val=weighted_mean(c,w)+.05*weighted_mean(cost,w)
    if mode=='direct':return weighted_mean(direct,w)+.05*weighted_mean(cost,w)
    if mode=='confidence':return val
    val+=weighted_mean(r,w*(1-y[:,0])*n[:,1]/4.)
    if mode!='no_damage':val+=weighted_mean(h,w*y[:,0]*n[:,2]/4.)+.5*weighted_mean(coupled,w)
    val+=.2*weighted_mean(retry,w*n[:,5]/4.)+.05*weighted_mean((raw[:,6]-y[:,6]).square(),w*n[:,6])
    return val


def calibrate(logits,y,n):
    temperatures=np.ones(7,dtype=np.float32)
    for col in (0,1,2,5):
        mask=np.ones(len(y),bool) if col==0 else y[:,0]==0 if col==1 else y[:,0]==1 if col==2 else n[:,5]>0
        if mask.sum()<5:continue
        z=logits[mask,col];target=y[mask,col]
        candidates=np.geomspace(.35,3.,31)
        losses=[np.mean(np.logaddexp(0,z/t)-target*z/t) for t in candidates]
        temperatures[col]=candidates[int(np.argmin(losses))]
    return temperatures


def fit(train,dev,train_features,dev_features,train_events,dev_events,output,checkpoint,
        bank='fit',mode='cavr',seeds=(41,73,107),epochs=120,hidden=128,lr=1e-3,patience=15,
        batch_size=64,device='cpu',resume=False,fixtures=False):
    if mode not in MODES:raise ValueError('Unknown controller mode')
    validate_states(train,training=True,fixtures=fixtures);validate_states(dev,training=True,fixtures=fixtures);disjoint(train,dev)
    x,encoder,_=load_features(train_features,train);v,encoder2,_=load_features(dev_features,dev)
    if encoder!=encoder2:raise ValueError('Different encoders in train/dev features')
    a=targets(train,train_events,bank,checkpoint);b=targets(dev,dev_events,bank,checkpoint)
    if a['executor']!=b['executor']:raise ValueError('Different train/dev executors')
    signature=digest({'mode':mode,'seeds':list(seeds),'epochs':epochs,'hidden':hidden,'lr':lr,'patience':patience,
                      'batch':batch_size,'features':[sha256(train_features),sha256(dev_features)],
                      'train':[(r['id'],state_hash(r)) for r in train],'dev':[(r['id'],state_hash(r)) for r in dev],
                      'implementation':sha256(__file__),'targets':[a['y'].tolist(),a['n'].tolist(),b['y'].tolist(),b['n'].tolist()],'executor':a['executor']})
    out=Path(output);out.mkdir(parents=True,exist_ok=True);meta=out/'model.json'
    if meta.exists():
        m=read_json(meta)
        if m['signature']!=signature:raise ValueError('Controller output already belongs to another fit')
        if m.get('complete'):return m
    mean=x.mean(0);std=x.std(0);std=np.maximum(std,.05)
    x=np.clip((x-mean)/std,-8,8);v=np.clip((v-mean)/std,-8,8)
    np.savez_compressed(out/'scaler.npz',mean=mean,std=std)
    metadata={'schema':5,'signature':signature,'mode':mode,'encoder_id':encoder,'executor':a['executor'],
              'executor_checkpoint':checkpoint,'bank':bank,'dim':x.shape[1],'hidden':hidden,'seeds':list(seeds),
              'train_problems':sorted({r['problem_id'] for r in train}),'dev_problems':sorted({r['problem_id'] for r in dev}),
              'seen_questions':sorted({question_key(r) for r in train+dev}),'complete':False,
              'uncertainty':'ensemble dispersion, not a coverage guarantee'}
    save_json(meta,metadata)
    tensor=lambda z:torch.as_tensor(z,dtype=torch.float32,device=device)
    xt=tensor(x);vt=tensor(v);ay=tensor(a['y']);an=tensor(a['n']);aw=tensor(a['weights'])
    by=tensor(b['y']);bn=tensor(b['n']);bw=tensor(b['weights'])
    logs=[]
    from safetensors.torch import save_file
    for seed in seeds:
        torch.manual_seed(int(seed));rng=np.random.default_rng(seed);head=ValueHead(x.shape[1],hidden).to(device)
        opt=torch.optim.AdamW(head.parameters(),lr=lr,weight_decay=.01)
        start=0;best=float('inf');best_state=None;stale=0;history=[];ck=out/f'train-{seed}.pt'
        if resume and ck.exists():
            saved=torch.load(ck,map_location=device,weights_only=False)
            if saved['signature']!=signature:raise ValueError('Controller resume signature mismatch')
            head.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);start=saved['epoch']+1
            best=saved['best'];best_state=saved['best_state'];stale=saved['stale'];history=saved['history']
            rng.bit_generator.state=saved['rng'];torch.set_rng_state(saved['torch_rng'].cpu())
            if torch.cuda.is_available() and saved.get('cuda_rng') is not None:torch.cuda.set_rng_state_all([x.cpu() for x in saved['cuda_rng']])
        elif ck.exists():raise FileExistsError('Controller training state exists; pass --resume')
        for epoch in range(start,epochs):
            if stale>=patience:break
            head.train();perm=rng.permutation(len(x));total=[]
            for j in range(0,len(x),batch_size):
                idx=torch.tensor(perm[j:j+batch_size],device=device);opt.zero_grad(set_to_none=True)
                loss=loss_fn(head(xt[idx]),ay[idx],an[idx],aw[idx],mode)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite controller loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.);opt.step();total.append(float(loss.detach()))
            head.eval()
            with torch.no_grad():val=float(loss_fn(head(vt),by,bn,bw,mode))
            if val<best-1e-6:best=val;stale=0;best_state={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
            else:stale+=1
            history.append({'seed':seed,'epoch':epoch,'train_loss':float(np.mean(total)),'dev_loss':val})
            saved={'signature':signature,'epoch':epoch,'model':head.state_dict(),'optimizer':opt.state_dict(),
                   'best':best,'best_state':best_state,'stale':stale,'history':history,'rng':rng.bit_generator.state,
                   'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
            temp=ck.with_suffix('.tmp');torch.save(saved,temp);temp.replace(ck)
        if best_state is None:raise RuntimeError('Controller did not complete an optimization step')
        head.load_state_dict(best_state);head.eval()
        with torch.no_grad():logits=head(vt).cpu().numpy()
        temp=calibrate(logits,b['y'],b['n'])
        save_file({k:v.detach().cpu().contiguous() for k,v in best_state.items()},str(out/f'head-{seed}.safetensors'))
        save_json(out/f'calibration-{seed}.json',{'temperature':temp.tolist(),'dev_states':len(dev)})
        logs.extend(history)
    save_json(out/'history.json',logs);metadata['complete']=True
    metadata['artifacts_sha256']={p.name:sha256(p) for p in sorted(out.glob('head-*.safetensors'))+sorted(out.glob('calibration-*.json'))+[out/'scaler.npz']}
    save_json(meta,metadata);return metadata


def predict(states,feature_file,model_dir,allow_seen=False):
    from safetensors.torch import load_file
    m=read_json(Path(model_dir)/'model.json')
    if not m.get('complete'):raise ValueError('Incomplete controller export')
    if any(sha256(Path(model_dir)/name)!=value for name,value in m['artifacts_sha256'].items()):raise ValueError('Controller artifact checksum mismatch')
    seen=set(m['seen_questions']);overlap=any(question_key(r) in seen for r in states)
    if overlap and not allow_seen:raise ValueError('Evaluation includes controller train/dev problems')
    x,eid,costs=load_features(feature_file,states)
    if eid!=m['encoder_id']:raise ValueError('Controller/feature encoder mismatch')
    s=np.load(Path(model_dir)/'scaler.npz',allow_pickle=False);x=np.clip((x-s['mean'])/s['std'],-8,8)
    members=[]
    for seed in m['seeds']:
        head=ValueHead(m['dim'],m['hidden']);head.load_state_dict(load_file(str(Path(model_dir)/f'head-{seed}.safetensors')));head.eval()
        with torch.no_grad():raw=head(torch.as_tensor(x,dtype=torch.float32)).numpy()
        temp=np.array(read_json(Path(model_dir)/f'calibration-{seed}.json')['temperature'])
        out=raw.copy()
        for c in (0,1,2,5):out[:,c]=1/(1+np.exp(-np.clip(raw[:,c]/temp[c],-40,40)))
        out[:,3]=np.tanh(raw[:,3]);out[:,4]=np.expm1(np.clip(raw[:,4],0,np.log1p(32768)))
        out[:,6]=np.expm1(np.clip(raw[:,6],0,np.log1p(32768)));members.append(out)
    a=np.stack(members)
    rows=[]
    for i,r in enumerate(states):
        rows.append({'id':r['id'],'problem_id':r['problem_id'],'state_hash':state_hash(r),'mode':m['mode'],
                    'members':a[:,i,:].tolist(),'executor':m['executor'],'executor_checkpoint':m['executor_checkpoint'],
                    'controller_signature':m['signature'],'encoder_id':eid,'seen_by_controller':question_key(r) in seen,
                    'feature_prefill_tokens':float(costs[i,0]),'feature_seconds':float(costs[i,1])})
    return rows
