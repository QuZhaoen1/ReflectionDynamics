import json
import os
import subprocess
import sys
from pathlib import Path

from refldyn.io import load_jsonl, save_jsonl, write_json, checksum_file
from test_curation import inputs
from test_screening import drafts


def call_cli(*args):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    return subprocess.run([sys.executable, "-m", "refldyn.cli", *map(str, args)],
                          check=True, env=env, capture_output=True, text=True)


def test_screen_and_curate_file_interfaces(tmp_path):
    draft_file = tmp_path / "drafts.jsonl"
    save_jsonl(draft_file, drafts("a", correct_count=4) + drafts("b", correct_count=7))
    call_cli("screen", "--drafts", draft_file, "--split", "train", "--output-dir", tmp_path / "calibration")
    report = json.loads((tmp_path / "calibration/train.summary.json").read_text())
    assert report["eligible_problems"] == 2
    states, evals = inputs(80)
    candidates = tmp_path / "states.jsonl"
    probes = tmp_path / "probes.jsonl"
    training = tmp_path / "train.jsonl"
    save_jsonl(candidates, states)
    save_jsonl(probes, evals)
    write_json(str(probes) + ".manifest.json", {
        "states_sha256": checksum_file(candidates), "checkpoint": "base", "adapter": None,
        "retry_control": True, "model": "Qwen/Qwen2.5-32B"})
    call_cli("curate", "--candidate-states", candidates, "--probe-results", probes,
             "--output", training, "--target", "40")
    chosen = list(load_jsonl(training))
    assert len(chosen) == 40
    manifest = json.loads((tmp_path / "train.curation.json").read_text())
    assert manifest["training_sha256"]
