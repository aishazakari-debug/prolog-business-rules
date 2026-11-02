"""Registered run for one API model. Program ids 1 to N per cell (the pilot used 101 to 110).

    ../../agentic-rag-eval/.venv/bin/python run_full.py --model claude-haiku-5-5 --programs 30 --dry-run
    ../../agentic-rag-eval/.venv/bin/python run_full.py --model claude-haiku-5-5 --programs 30
    ../../agentic-rag-eval/.venv/bin/python run_full.py --model gpt-6-sol --programs 5 --depths 6,22 --arms A,B

The project cap of 3 US dollars lives in results/ledger.json. Every call reserves its worst case before it is sent.
Rows are appended to results/full/<model>/calls.jsonl as they finish, and a rerun skips finished jobs.
Arm C is not a separate scorer here. It is routed to arm B by scoring.py.
"""
import argparse
import hashlib
import inspect
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import policy_env as P
from clients import BudgetExceeded, Chooser, Ledger

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--programs", type=int, default=30)
ap.add_argument("--depths", default="2,6,10,14,18,22")
ap.add_argument("--arms", default="A,A0,C,B,NR")
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--dry-run", action="store_true")
args = ap.parse_args()
DEPTHS = [int(x) for x in args.depths.split(",")]
ARMS = args.arms.split(",")
OUT = HERE / "results" / "full" / args.model
OUT.mkdir(parents=True, exist_ok=True)
FILE = OUT / "calls.jsonl"


def frozen_ok():
    prompts = "\n--\n".join([P.SYSTEM_BASE, P.SYSTEM_ONLY_EXTRA, P.SYSTEM_TRANSLATE, repr(P.TEMPLATES), inspect.getsource(P.user_direct),
                             inspect.getsource(P.user_translate), inspect.getsource(P.system_for), inspect.getsource(P.parse_answer),
                             inspect.getsource(P.extract_code)])
    gen = "\n--\n".join([repr(P.NAMES), repr(P.LABELS), inspect.getsource(P.gen_program), inspect.getsource(P.check_form)])
    want = (HERE / "results" / "PROMPT_FROZEN.txt").read_text().splitlines()[:2]
    got = [hashlib.sha256(prompts.encode()).hexdigest(), hashlib.sha256(gen.encode()).hexdigest()]
    if got != want:
        raise SystemExit("prompts or generator changed since they were frozen")
    print("frozen hashes match:", want[0][:12], want[1][:12])


def job_list():
    jobs = []
    for d in DEPTHS:
        for pid in range(1, args.programs + 1):
            for pad in (False, True):
                for arm in ARMS:
                    if arm == "NR" and pad:
                        continue
                    jobs.append((arm, d, pid, pad))
    return jobs


def done_keys():
    keys = set()
    if FILE.exists():
        for l in open(FILE):
            r = json.loads(l)
            keys.add((r["arm"], r["depth"], r["pid"], r["padded"]))
    return keys


def run_job(job, ch):
    arm, d, pid, pad = job
    g = P.gen_program(d, pid, pad)
    rows = []
    if arm == "B":
        text, rec = ch.choose(P.SYSTEM_TRANSLATE, P.user_translate(g), max_tokens=8000)
        clauses, err = P.check_form(P.extract_code(text), set(g.labels), {c.lower() for c, _ in g.facts})
        res = P.run_prolog(clauses, g.labels, P.goals_for(g)) if clauses else None
        ran = res is not None and "error" not in res
        for k, (cust, _, gold) in enumerate(P.questions(g)):
            ans = res[k] if ran else None
            rows.append({"arm": "B", "depth": d, "pid": pid, "padded": pad, "customer": cust, "gold": gold, "raw": text if k == 0 else "",
                         "ans": ans, "form_ok": clauses is not None, "form_err": err, "ran": ran, "timeout": clauses is not None and res is None,
                         "n_clauses": len(clauses) if clauses else 0, "gold_clauses": len(g.facts) + len(g.rules),
                         "correct": int(ans == ("yes" if gold else "no")), **(rec if k == 0 else {"in": 0, "out": 0, "usd": 0, "s": 0})})
        return rows
    mode = "A0" if arm in ("A0", "NR") else arm
    for cust, qtext, gold in P.questions(g):
        text, rec = ch.choose(P.system_for(mode), P.user_direct(g, qtext, mode, rules=(arm != "NR")), max_tokens=12 if mode == "A0" else 2500)
        ans = P.parse_answer(text, mode)
        rows.append({"arm": arm, "depth": d, "pid": pid, "padded": pad, "customer": cust, "gold": gold, "raw": text, "ans": ans,
                     "correct": int(ans == ("yes" if gold else "no")), **rec})
    return rows


def main():
    frozen_ok()
    jobs = [j for j in job_list() if j not in done_keys()]
    n_calls = sum(1 if j[0] == "B" else 2 for j in jobs)
    print(f"{len(jobs)} jobs left, about {n_calls} calls, model {args.model}, depths {DEPTHS}, {args.programs} programs per cell, arms {ARMS}")
    if args.dry_run:
        return
    ledger = Ledger(cap_usd=3.00, path=HERE / "results" / "ledger.json")
    print(f"project total before this run: {ledger.total:.4f} of {ledger.cap:.2f}")
    ch = Chooser(args.model, ledger)
    lock, stop = threading.Lock(), threading.Event()

    def work(j):
        if stop.is_set():
            return 0
        try:
            rows = run_job(j, ch)
        except BudgetExceeded as e:
            stop.set(); print("STOPPED:", e); return 0
        with lock, open(FILE, "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        return 1

    with ThreadPoolExecutor(args.workers) as ex:
        finished = sum(ex.map(work, jobs))
    print(f"finished {finished} of {len(jobs)} jobs; run spend {ledger.usd:.4f}; project total {ledger.total:.4f} of {ledger.cap:.2f}")


if __name__ == "__main__":
    main()
