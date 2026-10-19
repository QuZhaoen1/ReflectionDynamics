"""Run the original CLI with atomic persistence and per-generation timing."""
import json
import os
import time
from pathlib import Path
from refldyn import io
from refldyn.backend import InferenceBackend

original_save = io.save_jsonl
original_json = io.write_json

def atomic_save(path, rows):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    original_save(temporary, rows)
    os.replace(temporary, path)

def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    original_json(temporary, value)
    os.replace(temporary, path)

io.save_jsonl = atomic_save
io.write_json = atomic_json
original_generate = InferenceBackend.generate

def timed_generate(self, *args, **kwargs):
    started = time.time()
    result = original_generate(self, *args, **kwargs)
    record = {'started': started, 'seconds': time.time() - started,
              'tokens': result[1], 'gpu': os.environ.get('CUDA_VISIBLE_DEVICES'),
              'peak_allocated_gib': self.torch.cuda.max_memory_allocated()/2**30}
    with open(os.environ['GENERATION_METRICS'], 'a') as handle:
        handle.write(json.dumps(record) + '\n')
    return result

InferenceBackend.generate = timed_generate
from refldyn.cli import main
main()
