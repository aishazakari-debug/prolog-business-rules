"""Computes every number the article quotes from the saved call files and writes numbers.json.

    python3 make_numbers.py

The draft is built from numbers.json by build_draft.py, so no number in the text is typed by hand. Intervals are 95 percent
bootstrap intervals over programs (5,000 resamples, fixed seed), because the two questions of a program share its rules.
Cells with failed translations or unreadable answers count those items as wrong. Arm C is routed to arm B on HAND OFF.
"""
import json
import random
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

import scoring

HERE = Path(__file__).resolve().parent
R = HERE / "results" / "full"
MODELS = {"haiku": "claude-haiku-5-5", "qwen": "qwen3-4b-instruct-2507"}
DEPTHS = (2, 6, 10, 14, 18, 22)
rng = random.Random(0)
N = {"cell": {}, "diff": {}, "trans": {}, "tokens": {}, "C": {}, "qwen_cap": {}, "unread": {}, "cost": {}, "counts": {}}


def load(m):
    rows = [json.loads(l) for l in open(R / MODELS[m] / "calls.jsonl")]
    if any(r["arm"] == "C" for r in rows):
        scoring.route_c(rows)
    return rows


def boot(vals, n=5000):
    vals = list(vals)
    out = sorted(st.mean(rng.choices(vals, k=len(vals))) for _ in range(n))
    return out[int(0.025 * n)], out[int(0.975 * n)]


def diff_unpaired(a, b, n=5000):
    a, b = list(a.values()), list(b.values())
    out = sorted(st.mean(rng.choices(a, k=len(a))) - st.mean(rng.choices(b, k=len(b))) for _ in range(n))
    return {"mean": st.mean(a) - st.mean(b), "lo": out[int(0.025 * n)], "hi": out[int(0.975 * n)]}


def diff_paired(a, b):
    ids = sorted(set(a) & set(b))
    assert len(ids) == len(a) == len(b)
    d = [a[i] - b[i] for i in ids]
    lo, hi = boot(d)
    return {"mean": st.mean(d), "lo": lo, "hi": hi}


for m in MODELS:
    rows = load(m)
    f = lambda r: r["final_correct"] if r["arm"] == "C" else r["correct"]
    prog = defaultdict(lambda: defaultdict(list))
    for r in rows:
        prog[(r["arm"], r["depth"], r["padded"])][r["pid"]].append(f(r))
    acc = {k: {pid: 100 * st.mean(v) for pid, v in d.items()} for k, d in prog.items()}
    N["counts"][m] = {"rows": len(rows), "programs_per_cell": len(acc[("A", 2, False)])}
    cell = defaultdict(dict)
    for (arm, d, pad), v in acc.items():
        lo, hi = boot(v.values())
        cell[f"{arm}_{'padded' if pad else 'short'}"][str(d)] = {"mean": st.mean(v.values()), "lo": lo, "hi": hi}
    N["cell"][m] = cell
    D = {}
    D["p1_short"] = diff_unpaired(acc[("A", 22, False)], acc[("A", 2, False)])
    D["p1_padded"] = diff_unpaired(acc[("A", 22, True)], acc[("A", 2, True)])
    D["p2"] = {str(d): diff_paired(acc[("A", d, True)], acc[("A", d, False)]) for d in DEPTHS}
    D["p4_short"] = diff_unpaired(acc[("B", 22, False)], acc[("B", 2, False)])
    if ("B", 22, True) in acc:
        D["p4_padded"] = diff_unpaired(acc[("B", 22, True)], acc[("B", 2, True)])
    D["p5"] = {}
    for pad in (False, True):
        for d in DEPTHS:
            if ("B", d, pad) in acc:
                D["p5"][f"{'padded' if pad else 'short'}_{d}"] = diff_paired(acc[("B", d, pad)], acc[("A", d, pad)])
    N["diff"][m] = D
    # translation
    T = {}
    for pad in (False, True):
        for d in DEPTHS:
            pr = [r for r in rows if r["arm"] == "B" and r["depth"] == d and r["padded"] == pad and r["raw"]]
            if not pr:
                continue
            run = [r for r in rows if r["arm"] == "B" and r["depth"] == d and r["padded"] == pad and r["ran"]]
            T[f"{'padded' if pad else 'short'}_{d}"] = {
                "n": len(pr), "ran": sum(r["ran"] for r in pr), "timeouts": sum(r["timeout"] for r in pr),
                "runnable_acc": 100 * st.mean(r["correct"] for r in run) if run else None,
                "errors": dict(Counter(r["form_err"] for r in pr if r["form_err"]))}
    N["trans"][m] = T
    allB = [r for r in rows if r["arm"] == "B" and r["raw"]]
    N["trans"][m]["total"] = {"n": len(allB), "ran": sum(r["ran"] for r in allB), "timeouts": sum(r["timeout"] for r in allB),
                              "errors": dict(Counter(r["form_err"] for r in allB if r["form_err"]))}
    for pad in (False, True):
        pr = [r for r in allB if r["padded"] == pad]
        N["trans"][m][f"total_{'padded' if pad else 'short'}"] = {"n": len(pr), "ran": sum(r["ran"] for r in pr)}
        run = [r for r in rows if r["arm"] == "B" and r["padded"] == pad and r["ran"]]
        N["trans"][m][f"runnable_acc_{'padded' if pad else 'short'}"] = {"min": None, "all": 100 * st.mean(r["correct"] for r in run) if run else None}
    # tokens and cost per call
    tk = {}
    for arm in ("A", "A0", "C", "NR", "B"):
        for pad in (False, True):
            rs = [r for r in rows if r["arm"] == arm and r["padded"] == pad and (arm != "B" or r["raw"])]
            if rs:
                tk[f"{arm}_{'padded' if pad else 'short'}"] = {"in": st.mean(r["in"] for r in rs), "out": st.mean(r["out"] for r in rs), "usd": st.mean(r["usd"] for r in rs)}
    N["tokens"][m] = tk
    N["unread"][m] = {a: sum(r["ans"] is None for r in rows if r["arm"] == a) for a in ("A", "A0", "C", "NR", "B") if any(r["arm"] == a for r in rows)}
    if m == "haiku":
        cs = [r for r in rows if r["arm"] == "C"]
        A = {scoring.key(r): r["correct"] for r in rows if r["arm"] == "A"}
        fl = [A[scoring.key(r)] for r in cs if r["ans"] == "hand"]
        N["C"] = {"n": len(cs), "handoffs": len(fl), "A_acc_flagged": 100 * st.mean(fl) if fl else None,
                  "A_acc_unflagged": 100 * st.mean(A[scoring.key(r)] for r in cs if r["ans"] != "hand"),
                  "overall_A": 100 * st.mean(r["correct"] for r in rows if r["arm"] == "A"),
                  "overall_B": 100 * st.mean(r["correct"] for r in rows if r["arm"] == "B"),
                  "overall_C": 100 * st.mean(r["final_correct"] for r in cs),
                  "cost_C": st.mean(r["cost_final"] for r in cs)}
        N["cost"]["haiku_run_usd"] = sum(r["usd"] for r in rows)
    if m == "qwen":
        cap = {}
        for pad in (False, True):
            for d in DEPTHS:
                rs = [r for r in rows if r["arm"] == "A" and r["depth"] == d and r["padded"] == pad]
                cap[f"{'padded' if pad else 'short'}_{d}"] = {"n": len(rs), "capped": sum(r["out"] >= 2500 for r in rs), "unreadable": sum(r["ans"] is None for r in rs)}
        N["qwen_cap"]["cells"] = cap
        # exploratory sensitivity: capped replies rerun at 4,000 tokens, replaced one for one
        cs = [json.loads(l) for l in open(R / MODELS[m] / "cutoff_sensitivity.jsonl")]
        N["qwen_cap"]["rerun"] = {"n": len(cs), "with_answer": sum(r["ans"] is not None for r in cs), "correct": sum(r["correct"] for r in cs),
                                  "prefix_kept": sum(bool(r.get("prefix_matches")) for r in cs)}
        repl = {(r["depth"], r["pid"], r["padded"], r["customer"]): r["correct"] for r in cs}
        for d in (18, 22):
            ps = defaultdict(list)
            for r in rows:
                if r["arm"] == "A" and not r["padded"] and r["depth"] == d:
                    ps[r["pid"]].append(repl.get((r["depth"], r["pid"], r["padded"], r["customer"]), r["correct"]))
            N["qwen_cap"][f"short_{d}_replaced"] = 100 * st.mean(st.mean(v) for v in ps.values())

# pilots and totals
N["cost"]["pilots_usd"] = 0.4611 + 0.2202            # v1, probes and v2a (0.4611) plus the rerun pilot v2 (0.2202), as in results/ledger.json history
N["cost"]["project_usd"] = json.load(open(HERE / "results" / "ledger.json"))["total_usd"]
N["cost"]["qwen_units"] = 595.87 - 593.68
N["cost"]["calls_haiku"] = N["counts"]["haiku"]["rows"]
json.dump(N, open(HERE / "numbers.json", "w"), indent=1)
print("numbers.json written")
h = N["cell"]["haiku"]
print("haiku A short", [round(h["A_short"][str(d)]["mean"], 1) for d in DEPTHS])
print("haiku A padded", [round(h["A_padded"][str(d)]["mean"], 1) for d in DEPTHS])
print("haiku B padded", [round(h["B_padded"][str(d)]["mean"], 1) for d in DEPTHS])
q = N["cell"]["qwen"]
print("qwen A short", [round(q["A_short"][str(d)]["mean"], 1) for d in DEPTHS], "B short", [round(q["B_short"][str(d)]["mean"], 1) for d in DEPTHS])
print("qwen replaced short 18, 22:", round(N["qwen_cap"]["short_18_replaced"], 1), round(N["qwen_cap"]["short_22_replaced"], 1), N["qwen_cap"]["rerun"])
print("haiku project total", N["cost"]["project_usd"], "haiku run", round(N["cost"]["haiku_run_usd"], 4))
