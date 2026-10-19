import re
from dataclasses import dataclass

from .answers import extract_final, verify_answer

ACTION = re.compile(r"^\s*(COMMIT|REFLECT)\b", re.I)
FINAL = re.compile(r"(?im)^\s*FINAL:\s*(.*?)\s*$")
CHECK = re.compile(r"(?im)^\s*CHECK:\s*(\S.*?)\s*$")


@dataclass(frozen=True)
class Decision:
    action: str | None
    answer: str | None
    valid: bool


def prompt_for_state(row):
    reasoning = str(row.get("initial_reasoning", "")).strip()
    return (
        "A proposed solution has already been written. Decide whether to submit it "
        "or review it for a possible mistake.\n"
        f"Problem:\n{row['question']}\n\n"
        f"Proposed reasoning:\n{reasoning}\n\n"
        f"Proposed answer: \\boxed{{{row['initial_answer']}}}\n\n"
        "Allowed output:\n"
        "ACTION: COMMIT\nFINAL: \\boxed{<proposed answer>}\n"
        "or\n"
        "ACTION: REFLECT\nCHECK: <reviewed reasoning>\nFINAL: \\boxed{<answer>}\n"
        "For COMMIT, preserve the proposed answer.\nACTION: "
    )


def parse_decision(text, initial_answer):
    action = ACTION.search(str(text))
    if not action:
        return Decision(None, None, False)
    action = action.group(1).upper()
    finals = FINAL.findall(str(text))
    if len(finals) != 1:
        return Decision(action, None, False)
    extracted = extract_final(finals[0])
    if extracted is None:
        return Decision(action, None, False)
    if action == "REFLECT" and not CHECK.search(str(text)):
        return Decision(action, extracted, False)
    if action == "COMMIT" and verify_answer(initial_answer, extracted) is not True:
        return Decision(action, extracted, False)
    return Decision(action, extracted, True)


def terminal_score(completion, initial_answer, gold):
    decision = parse_decision(completion, initial_answer)
    if not decision.valid:
        return 0.0
    return float(verify_answer(gold, decision.answer) is True)


def forced_reflection_prefix():
    return "REFLECT\nCHECK: "


def independent_retry_prompt(row):
    return (
        "Solve the following mathematics problem independently. "
        "Use the final line FINAL: \\boxed{<answer>}.\n\n"
        f"Problem:\n{row['question']}\n\nSolution:\n"
    )
