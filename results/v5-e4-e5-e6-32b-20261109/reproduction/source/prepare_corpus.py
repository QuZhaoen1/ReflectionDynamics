"""Offline import, with pinned source bytes and the unchanged project adapters."""
import base64
import collections
import gzip
import hashlib
import json
from pathlib import Path
import pyarrow.parquet as pq
from cavr.io import save_json,save_rows,digest,sha256,read_json
from refldyn.corpus import adapt_math,adapt_gsm8k,adapt_aime,build_corpus

ROOT=Path(__file__).resolve().parent
ASSETS=Path('${PROJECT_ROOT}/reflectiondynamics-assets/datasets')


def github_file(kind,name):
    obj=read_json(ROOT/'raw'/kind/'github-file.json')
    if obj.get('encoding')!='base64':raise ValueError('GitHub file content missing')
    data=base64.b64decode(obj['content'])
    assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==obj['sha']
    p=ROOT/'raw'/kind/name;p.write_bytes(data)
    return p


def main():
    # Compare the ModelScope GSM8K mirror with the pinned OpenAI source.
    gsm_github=github_file('gsm8k','test.jsonl')
    gsm_parquet=ROOT/'raw/gsm8k/main/test-00000-of-00001.parquet'
    official=[json.loads(s) for s in gsm_github.read_text().splitlines()]
    mirrored=pq.read_table(gsm_parquet).to_pylist()
    core=lambda r:json.dumps({k:r[k] for k in ('question','answer')},sort_keys=True)
    assert collections.Counter(map(core,official))==collections.Counter(map(core,mirrored))
    human=github_file('humaneval','HumanEval.jsonl.gz')
    humans=[json.loads(s) for s in gzip.decompress(human.read_bytes()).decode().splitlines()]
    assert len(humans)==164 and len(official)==1319
    specs={
      'math':(ASSETS/'competition_math/original/test-00000-of-00001.parquet',
              'f9f344088a5d918a4ad167cb4451db423cdea41168746c613283e6df8b97865f',
              'jeggers/competition_math','bd76e17d5b06797bc5ae48f30d0d9634c3255a86',5000),
      'aime':(ASSETS/'aimo-validation-aime/data/train-00000-of-00001.parquet',
              '025484a99fea498e7d0c3b0ee42afcbec0176405c19c5dbf557b9f6ca6445675',
              'AI-MO/aimo-validation-aime','13f9e12f613e720c2a2b2f345dd04b998a29494d',90),
      'gsm8k':(gsm_parquet,'ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59',
               'openai/grade-school-math','3101c7d5072418e28b9008a6636bde82a006892c',1319),
      'humaneval':(human,sha256(human),'openai/human-eval','b9453e8f1865df0a0057a1383767caa400acfbcf',164)}
    for kind,(path,h,source,rev,count) in specs.items():
        assert sha256(path)==h
        raw=humans if kind=='humaneval' else pq.read_table(path).to_pylist()
        assert len(raw)==count
        if kind=='humaneval':
            rows=[{'id':digest(r['prompt'])[:20],'question':r['prompt'],'source':'humaneval',
                   'source_split':'test','split':'ood_code','task':'code','fixture':False,
                   'hidden_tests':r['test'],'entry_point':r['entry_point'],'source_id':r['task_id']} for r in raw]
            rejected=[];summary={'accepted':len(rows)}
        else:
            adapt={'math':lambda r:adapt_math(r,'test'),'aime':adapt_aime,'gsm8k':lambda r:adapt_gsm8k(r,'test')}[kind]
            adapted=[adapt(r) for r in raw]
            rows,rejected,summary=build_corpus([r for r in adapted if r])
            summary['adaptation_rejections']=sum(r is None for r in adapted)
            for r in rows:r.update(task='math',fixture=False)
        out=ROOT/'corpus'/kind
        for split in sorted({r['split'] for r in rows}):save_rows(out/f'{split}.jsonl',[r for r in rows if r['split']==split])
        save_rows(out/'rejected.jsonl',rejected)
        manifest={'source_repository':source,'revision':rev,'source_path':str(path),'source_sha256':h,
                  'source_rows':len(raw),'kind':kind,'importer_sha256':sha256(__file__),**summary}
        if kind=='gsm8k':manifest.update(mirror='AI-ModelScope/gsm8k',mirror_revision='2680164407bc7fd6c04d1ad609399a6e62a3b21e',
                                         verified_equal_to_official=True,official_file_sha256=sha256(gsm_github))
        save_json(out/'manifest.json',manifest)
        print(kind,len(raw),len(rows),flush=True)


if __name__=='__main__':main()
