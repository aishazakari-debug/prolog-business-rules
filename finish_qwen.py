"""Validate and score the saved Qwen replies without making model calls.

Run after run_qwen_on_colab.py full has downloaded all five arms.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import policy_env as P

HERE = Path(__file__).resolve().parent
DIR = HERE / "results" / "full" / "qwen3-4b-instruct-2507"
RAW = DIR / "calls_raw.jsonl"
OUT = DIR / "calls.jsonl"
DEPTHS = (2, 6, 10, 14, 18, 22)
IDS = range(1, 31)


def expected_keys():
    return ({(arm, depth, pid, pad, cust)
             for arm in ("A", "A0") for depth in DEPTHS for pid in IDS for pad in (False, True)
             for cust, _, _ in P.questions(P.gen_program(depth, pid, pad))}
            | {("B", depth, pid, False, None) for depth in DEPTHS for pid in IDS})


def main():
    rows = [json.loads(line) for line in RAW.open() if line.strip()]
    keys = [(r["arm"], r["depth"], r["pid"], r["padded"], r.get("customer")) for r in rows]
    counts = Counter(keys)
    duplicates = [key for key, count in counts.items() if count != 1]
    expected = expected_keys()
    if duplicates or set(keys) != expected:
        raise SystemExit(f"Incomplete Qwen run: {len(rows)} of {len(expected)} expected rows; "
                         f"{len(expected - set(keys))} missing, {len(set(keys) - expected)} extra, "
                         f"{len(duplicates)} duplicate keys")

    scored = []
    for r in rows:
        depth, pid, padded = r["depth"], r["pid"], r["padded"]
        g = P.gen_program(depth, pid, padded)
        questions = P.questions(g)
        if r["arm"] != "B":
            matches = [(q, gold) for cust, q, gold in questions if cust == r["customer"]]
            if len(matches) != 1 or r["gold"] != matches[0][1]:
                raise SystemExit(f"Question or gold mismatch: {r['arm']} {depth} {pid} {padded}")
            ans = P.parse_answer(r["raw"], r["arm"])
            r["ans"] = ans
            r["correct"] = int(ans == ("yes" if r["gold"] else "no"))
            scored.append(r)
            continue

        code = P.extract_code(r["raw"])
        clauses, err = P.check_form(code, set(g.labels), {cust.lower() for cust, _ in g.facts})
        result = P.run_prolog(clauses, g.labels, P.goals_for(g)) if clauses else None
        ran = result is not None and "error" not in result
        if ran and len(result) != len(questions):
            raise SystemExit(f"Wrong number of Prolog answers: {depth} {pid}")
        for i, (cust, _, gold) in enumerate(questions):
            ans = result[i] if ran else None
            scored.append({"arm": "B", "depth": depth, "pid": pid, "padded": padded,
                           "customer": cust, "gold": gold, "raw": r["raw"] if i == 0 else "",
                           "ans": ans, "form_ok": clauses is not None, "form_err": err,
                           "ran": ran, "timeout": clauses is not None and result is None,
                           "n_clauses": len(clauses) if clauses else 0,
                           "gold_clauses": len(g.facts) + len(g.rules),
                           "correct": int(ans == ("yes" if gold else "no")),
                           "in": r["in"] if i == 0 else 0,
                           "out": r["out"] if i == 0 else 0,
                           "usd": 0.0})

    temp = OUT.with_suffix(".jsonl.tmp")
    with temp.open("w") as file:
        for row in scored:
            file.write(json.dumps(row) + "\n")
    temp.replace(OUT)
    grouped = defaultdict(list)
    for row in scored:
        grouped[(row["arm"], row["padded"], row["depth"])].append(row)
    for key in sorted(grouped):
        group = grouped[key]
        print(key, len(group), f"accuracy {100 * sum(x['correct'] for x in group) / len(group):.1f}%",
              f"unparsed {sum(x['ans'] is None for x in group)}")
    print(f"saved {len(scored)} scored rows to {OUT}")


if __name__ == "__main__":
    main()
