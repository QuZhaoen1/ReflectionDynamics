import argparse
from pathlib import Path
from cavr.io import load_rows,read_json,save_json,seed_for
from cavr.backend import HFBackend
from cavr.features import extract


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True)
    p.add_argument('--states',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--shard',type=int,required=True)
    a=p.parse_args(); cfg=read_json(a.config)
    rows=[r for r in load_rows(a.states) if seed_for(r['id'])%4==a.shard]
    backend=HFBackend(cfg['model_path'],cfg['revision'])
    # The local model is the registered original Base snapshot; canonicalize its
    # location only, then require every other encoder field to match the cache.
    backend.identity['model']=cfg['feature_encoder']['backbone']['model']
    encoder={'mode':'hidden','backbone':backend.identity,'pool':'last+mean'}
    if encoder != cfg['feature_encoder']:
        raise ValueError('Frozen feature encoder changed')
    extract(rows,a.output,'hidden',backend)


if __name__=='__main__':main()
