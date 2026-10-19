import hashlib
import json
from pathlib import Path


def digest(value):
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def stable_int(value):
    return int(digest(value)[:14], 16)


def load_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_no}") from exc


def save_jsonl(path, rows):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path, value):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def read_yaml(path):
    import yaml
    with Path(path).open(encoding="utf-8") as handle:
        result = yaml.safe_load(handle)
    if not isinstance(result, dict):
        raise ValueError(f"Invalid YAML configuration: {path}")
    return result


def checksum_file(path):
    data = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            data.update(chunk)
    return data.hexdigest()


def canonical_question(value):
    return " ".join(str(value).lower().split())


def flatten(items):
    return [record for path in items for record in load_jsonl(path)]
