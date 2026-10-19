import json
import time
from pathlib import Path
import torch
from transformers import TrainerCallback
from storage_guard import check_storage

class Audit(TrainerCallback):
    def emit(self, args, state, event, **extra):
        if not state.is_world_process_zero:
            return
        record = {'event': event, 'time': time.time(), 'step': state.global_step,
                  'max_steps': state.max_steps, **extra}
        destination = Path(args.output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        with (destination/'progress.jsonl').open('a') as handle:
            handle.write(json.dumps(record) + '\n')
        print(json.dumps(record), flush=True)

    def on_step_begin(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            check_storage()
        self.started = time.monotonic()
        torch.cuda.reset_peak_memory_stats()

    def on_step_end(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        self.emit(args, state, 'optimizer_step', seconds=time.monotonic()-self.started,
                  peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)

    def on_save(self, args, state, control, **kwargs):
        self.emit(args, state, 'checkpoint_saved')

    def on_log(self, args, state, control, logs=None, **kwargs):
        self.emit(args, state, 'metrics', metrics=logs)
