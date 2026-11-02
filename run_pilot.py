"""Pilot v2 (redesigned arms; rerun after the equal fact count fix). Haiku 5.5 only, program ids 101 to 110 (outside the registered ids 1 to 30), depths 2, 10 and 22.

    ../../agentic-rag-eval/.venv/bin/python run_pilot.py

Arms: A direct with reasoning, A0 answer only, C flag, NR no rules shown, B translate to Prolog and run in the restricted
runner. Short and padded versions. The project cap of 3 US dollars is persistent in results/ledger.json. Writes results/pilot_v2/pilot.jsonl.
"""
import json
import statistics as st
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import policy_env as P
from clients import BudgetExceeded, Chooser, Ledger

OUT = Path(__file__).resolve().parent / "results" / "pilot_v2"
OUT.mkdir(parents=True, exist_ok=True)
MODEL = "claude-haiku-5-5"
LEDGER = Ledger(cap_usd=3.00, path=OUT.parent / 'ledger.json')
CH = Chooser(MODEL, LEDGER)
DEPTHS, IDS = (2, 10, 22), range(101, 111)


def direct_job(job):
    arm, d, pid, padded = job
    g = P.gen_program(d, pid, padded)
    rows = []
    for cust, qtext, gold in P.questions(g):
        mode = "A0" if arm == "NR" else arm
        text, rec = CH.choose(P.system_for(mode), P.user_direct(g, qtext, mode, rules=(arm != "NR")), max_tokens=12 if mode == "A0" else 2500)
        ans = P.parse_answer(text, mode)
        rows.append({"arm": arm, "depth": d, "pid": pid, "padded": padded, "customer": cust, "gold": gold, "raw": text, "ans": ans,
                     "correct": int(ans == ("yes" if gold else "no")), **rec})
    return rows


def translate_job(job):
    d, pid, padded = job
    g = P.gen_program(d, pid, padded)
    text, rec = CH.choose(P.SYSTEM_TRANSLATE, P.user_translate(g), max_tokens=8000)
    code = P.extract_code(text)
    clauses, err = P.check_form(code, set(g.labels), {c.lower() for c, _ in g.facts})
    res = P.run_prolog(clauses, g.labels, P.goals_for(g)) if clauses else None
    form_ok, ran = clauses is not None, res is not None and "error" not in res
    rows = []
    for k, (cust, qtext, gold) in enumerate(P.questions(g)):
        ans = None if not ran else res[k]
        rows.append({"arm": "B", "depth": d, "pid": pid, "padded": padded, "customer": cust, "gold": gold, "raw": text if k == 0 else "",
                     "ans": ans, "form_ok": form_ok, "form_err": err, "ran": ran, "timeout": clauses is not None and res is None,
                     "n_clauses": len(clauses) if clauses else 0, "gold_clauses": len(g.facts) + len(g.rules),
                     "correct": int(ans == ("yes" if gold else "no")), **(rec if k == 0 else {"in": 0, "out": 0, "usd": 0, "s": 0})})
    return rows


def main():
    jobs = [(arm, d, p, pad) for arm in ("A", "A0", "C") for d in DEPTHS for p in IDS for pad in (False, True)]
    jobs += [("NR", d, p, False) for d in DEPTHS for p in IDS]
    with ThreadPoolExecutor(8) as ex:
        rows = [r for rs in ex.map(direct_job, jobs) for r in rs]
        tj = [(d, p, pad) for d in DEPTHS for p in IDS for pad in (False, True)]
        rows += [r for rs in ex.map(translate_job, tj) for r in rs]
    with open(OUT / "pilot.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return rows


if __name__ == "__main__":
    try:
        rows = main()
    except BudgetExceeded as e:
        raise SystemExit(f"STOPPED: {e}")
    import scoring
    text = scoring.summarize(rows, DEPTHS) + f"\nproject total spend ${LEDGER.total:.4f} of ${LEDGER.cap:.2f}"
    print(text)
    (OUT / 'pilot_v2_printout.txt').write_text(text + chr(10))
