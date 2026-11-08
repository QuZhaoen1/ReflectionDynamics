import re
from fractions import Fraction

NUMBER = re.compile(r"^[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:/\d+|\.\d+)?$")


def last_boxed(text):
    marker = "\\boxed{"
    start = text.rfind(marker)
    if start < 0:
        return None
    nesting = 1
    index = start + len(marker)
    begin = index
    while index < len(text):
        if text[index] == "{":
            nesting += 1
        elif text[index] == "}":
            nesting -= 1
            if nesting == 0:
                return text[begin:index].strip()
        index += 1
    return None


def gold_from_math(solution, extracted=None):
    extracted = str(extracted or "").strip()
    if extracted:
        return last_boxed(extracted) or extracted
    return last_boxed(str(solution))


def gold_from_gsm8k(solution):
    matched = re.search(r"####\s*([+-]?[\d,]+(?:\.\d+)?)\s*$", str(solution))
    return matched.group(1).replace(",", "") if matched else None


def normalized_number(text):
    value = str(text).strip().replace(" ", "")
    if not NUMBER.fullmatch(value):
        return None
    try:
        return Fraction(value.replace(",", ""))
    except (ValueError, ZeroDivisionError):
        return None


def verify_answer(gold, prediction, strict_symbolic=True):
    if gold is None or prediction is None:
        return None
    a = normalized_number(gold)
    b = normalized_number(prediction)
    if NUMBER.fullmatch(str(gold).strip().replace(" ", "")) and a is None:
        return None
    if NUMBER.fullmatch(str(prediction).strip().replace(" ", "")) and b is None:
        return None
    if a is not None and b is not None:
        return bool(a == b)
    try:
        from math_verify import parse, verify
    except ImportError as exc:
        if strict_symbolic:
            raise RuntimeError("Install the 'research' extra for symbolic grading") from exc
        return None
    try:
        parsed_gold = parse(f"\\boxed{{{gold}}}")
        parsed_pred = parse(f"\\boxed{{{prediction}}}")
        if not parsed_gold or not parsed_pred:
            return None
        return bool(verify(parsed_gold, parsed_pred))
    except Exception:
        return None


def validate_gold(answer):
    if answer is None or not str(answer).strip():
        return False
    return verify_answer(answer, answer) is True


def extract_final(text):
    boxed = last_boxed(str(text))
    return boxed if boxed else None
