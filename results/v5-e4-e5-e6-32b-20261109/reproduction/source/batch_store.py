"""Immutable, fsynced generation batches; completed exports use project receipts."""
import fcntl
import json
import os
from pathlib import Path
from cavr.io import digest, read_json, save_json, atomic_text, sha256


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class BatchStore:
    def __init__(self, path, spec, plan, batch_size):
        self.path = Path(path)
        self.spec = spec
        self.signature = digest(spec)
        self.plan = plan
        self.batches = [plan[i:i+batch_size] for i in range(0, len(plan), batch_size)]
        self.directory = Path(str(path)+'.batches')
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = (self.directory/'writer.lock').open('a')
        try:
            fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            self.lock.close()
            raise
        self.meta = Path(str(path)+'.manifest.json')
        if self.meta.exists():
            if read_json(self.meta)['signature'] != self.signature:
                self.close()
                raise ValueError('Resume signature mismatch; new output required')
        else:
            if self.path.exists() or list(self.directory.glob('batch-*.json')):
                self.close()
                raise ValueError('Output exists without its manifest')
            save_json(self.meta, {'signature': self.signature, 'specification': spec})
            sync_directory(self.meta.parent)
        expected = {f'batch-{i:06d}.json' for i in range(len(self.batches))}
        if {p.name for p in self.directory.glob('batch-*.json')}-expected:
            raise ValueError('Unexpected committed batch')

    def key(self, request):
        return digest([self.signature, request['id'], request['arm'], request['sample']])

    def batch_path(self, index):
        return self.directory/f'batch-{index:06d}.json'

    def read(self, index):
        p = self.batch_path(index)
        if not p.exists():
            return None
        obj = read_json(p)
        rows = obj['rows']
        expected = [self.key(r) for r in self.batches[index]]
        if (obj['signature'] != self.signature or obj['index'] != index
                or obj['rows_digest'] != digest(rows)
                or [r['event_id'] for r in rows] != expected):
            raise ValueError(f'Corrupt or mismatched committed batch {p}')
        return rows

    def commit(self, index, rows):
        if self.batch_path(index).exists():
            raise ValueError('Refusing to overwrite a committed batch')
        expected = [self.key(r) for r in self.batches[index]]
        if [r['event_id'] for r in rows] != expected:
            raise ValueError('Batch is missing, duplicated, or reordered')
        save_json(self.batch_path(index), {'signature': self.signature, 'index': index,
                  'rows_digest': digest(rows), 'rows': rows})
        sync_directory(self.directory)

    def complete(self):
        rows = []
        for i in range(len(self.batches)):
            chunk = self.read(i)
            if chunk is None:
                raise ValueError('Cannot export an incomplete bank')
            rows.extend(chunk)
        if len(rows) != len(self.plan) or len({r['event_id'] for r in rows}) != len(rows):
            raise ValueError('Incorrect event count')
        receipt_path = Path(str(self.path)+'.complete.json')
        text = ''.join(json.dumps(r, ensure_ascii=False, sort_keys=True, allow_nan=False)+'\n' for r in rows)
        if receipt_path.exists():
            receipt = read_json(receipt_path)
            if (receipt['signature'] != self.signature or receipt['events'] != len(rows)
                    or receipt['sha256'] != sha256(self.path)
                    or self.path.read_text() != text):
                raise ValueError('Completed export changed')
        else:
            atomic_text(self.path, text)
            save_json(receipt_path, {'signature': self.signature, 'events': len(rows),
                                     'sha256': sha256(self.path)})
            sync_directory(self.path.parent)
        return rows

    def close(self):
        self.lock.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
