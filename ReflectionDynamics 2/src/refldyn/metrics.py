from collections import defaultdict

import numpy as np


def per_state(record, cost_weight=0.0):
    reflections = record.get("reflect_samples", [])
    if not reflections:
        raise ValueError(f"No forced reflections: {record.get('id')}")
    p = float(record["p_reflect"])
    if not 0 <= p <= 1:
        raise ValueError("Reflection preference outside [0,1]")
    initial = float(record["initial_correct"])
    success = np.mean([bool(row["correct"]) for row in reflections])
    tokens = np.mean([int(row["tokens"]) for row in reflections])
    cost = cost_weight * tokens / 1000.0
    u = float(success - initial - cost)
    retries = record.get("retry_samples", [])
    return {"id": record["id"], "problem_id": record["problem_id"], "checkpoint": record["checkpoint"],
            "split": record["split"], "p_reflect": p, "initial": initial,
            "reflect_accuracy": float(success), "reflect_tokens": float(tokens), "u": u,
            "natural_reflect": float(record.get("natural_action") == "REFLECT"),
            "natural_accuracy": float(bool(record.get("natural_correct"))),
            "natural_valid": float(bool(record.get("natural_valid"))),
            "forced_valid": float(np.mean([bool(row["valid"]) for row in reflections])),
            "retry_accuracy": float(np.mean([bool(row["correct"]) for row in retries])) if retries else None,
            "retry_tokens": float(np.mean([row["tokens"] for row in retries])) if retries else None,
            "repair": float(success) if not initial else None,
            "damage": float(1 - success) if initial else None}


def summarize(records, cost_weight=0.0):
    states = [per_state(x, cost_weight) for x in records]
    if not states:
        raise ValueError("Empty evaluation")
    p = np.array([x["p_reflect"] for x in states])
    u = np.array([x["u"] for x in states])
    retries = [x["retry_accuracy"] for x in states if x["retry_accuracy"] is not None]
    repairs = [x["repair"] for x in states if x["repair"] is not None]
    damages = [x["damage"] for x in states if x["damage"] is not None]
    expected_policy_utility = float(np.mean(p * u))
    matched_random = float(p.mean() * u.mean())
    return {"states": len(states), "unique_problems": len(set(r["problem_id"] for r in states)),
            "initial_accuracy": float(np.mean([x["initial"] for x in states])),
            "reflection_propensity": float(p.mean()),
            "observed_reflection_rate": float(np.mean([x["natural_reflect"] for x in states])),
            "natural_accuracy": float(np.mean([x["natural_accuracy"] for x in states])),
            "natural_format_rate": float(np.mean([x["natural_valid"] for x in states])),
            "forced_format_rate": float(np.mean([x["forced_valid"] for x in states])),
            "forced_reflection_accuracy": float(np.mean([x["reflect_accuracy"] for x in states])),
            "independent_retry_accuracy": float(np.mean(retries)) if retries else None,
            "repair_rate": float(np.mean(repairs)) if repairs else None,
            "damage_rate": float(np.mean(damages)) if damages else None,
            "reflection_tokens": float(np.mean([x["reflect_tokens"] for x in states])),
            "mean_utility": float(u.mean()),
            "selectivity": float(expected_policy_utility - matched_random),
            "policy_utility": expected_policy_utility,
            "frequency_matched_random": matched_random,
            "oracle_utility": float(np.mean(np.maximum(0, u))),
            "oracle_regret": float(np.mean(np.maximum(0, u) - p * u)),
            "cost_weight": cost_weight}


def cluster_bootstrap(records, bootstrap=1000, seed=41, cost_weight=0):
    if bootstrap < 10:
        raise ValueError("Require at least 10 bootstrap repetitions")
    groups = defaultdict(list)
    for row in records:
        groups[row["problem_id"]].append(row)
    keys = sorted(groups)
    if len(keys) < 2:
        raise ValueError("Require at least two distinct problem clusters")
    rng = np.random.default_rng(seed)
    tracked = ["reflection_propensity", "selectivity", "policy_utility", "natural_accuracy",
               "repair_rate", "damage_rate", "oracle_regret", "forced_reflection_accuracy"]
    values = defaultdict(list)
    for _ in range(bootstrap):
        sampled = rng.integers(0, len(keys), size=len(keys))
        observations = [x for i in sampled for x in groups[keys[int(i)]]]
        output = summarize(observations, cost_weight)
        for key in tracked:
            if output[key] is not None:
                values[key].append(output[key])
    return {key: [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]
            for key, vals in values.items() if vals}


def validate_results(records):
    identities = set()
    for row in records:
        if row.get("skip_reason"):
            continue
        key = (row["checkpoint"], row["split"], row["id"])
        if key in identities:
            raise ValueError(f"Duplicate evaluation state: {key}")
        identities.add(key)
        if "problem_id" not in row or not row.get("reflect_samples"):
            raise ValueError(f"Incomplete state: {key}")
    return len(identities)
