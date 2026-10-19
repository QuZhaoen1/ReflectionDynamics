import torch
import pytest
from cavr.smoke import fixtures,FixtureBackend
from cavr.data import split_training
from cavr.sampling import probe
from cavr.features import extract
from cavr.io import load_rows

torch.set_num_threads(1)

@pytest.fixture
def banks(tmp_path):
    panels=split_training(fixtures(60));out={};backend=FixtureBackend()
    for name,rows in panels.items():
        f=tmp_path/f'{name}.npz';extract(rows,f,'hash',dim=32)
        e=tmp_path/f'{name}.jsonl';probe(rows,backend,e,'fixture','score' if name=='gate_holdout' else 'fit',4)
        out[name]={'states':rows,'features':f,'events':load_rows(e)}
    return out
