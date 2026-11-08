import argparse
import sys

COMMANDS = {
    "ingest": "corpus",
    "screen": "screening",
    "curate": "curation",
    "draft": "rollouts",
    "select": "selection",
    "gate": "gates",
    "evaluate": "evaluate",
    "train": "train_grpo",
    "analyze": "analyze",
    "preflight": "checks",
    "transitions": "trajectory",
    "plot": "plot",
}


def main():
    parser = argparse.ArgumentParser(prog="rd")
    parser.add_argument("command", choices=sorted(COMMANDS))
    args, rest = parser.parse_known_args()
    sys.argv = [sys.argv[0], *rest]
    import importlib
    module = importlib.import_module("refldyn." + COMMANDS[args.command])
    module.run() if hasattr(module, "run") else module.main()


if __name__ == "__main__":
    main()
