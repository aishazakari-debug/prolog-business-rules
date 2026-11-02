"""Registered analysis for one model, from the saved call file. Intervals are 95 percent bootstrap intervals over programs.

    python3 analyze_full.py claude-haiku-5-5
"""
import json
import random
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import scoring

model = sys.argv[1]
D = Path(__file__).resolve().parent / "results" / "full" / model
rows = [json.loads(l) for l in open(D / "calls.jsonl")]
scoring.route_c(rows)
rng = random.Random(0)
DEPTHS = sorted({r["depth"] for r in rows})
f = lambda r: r["final_correct"] if r["arm"] == "C" else r["correct"]

# per program accuracy (mean of the two questions) for each arm, depth, length
prog = defaultdict(dict)
for r in rows:
    prog[(r["arm"], r["depth"], r["padded"])].setdefault(r["pid"], []).append(f(r))
acc = {k: {pid: 100 * st.mean(v) for pid, v in d.items()} for k, d in prog.items()}


def boot(vals, n=5000):
    vals = list(vals)
    out = sorted(st.mean(rng.choices(vals, k=len(vals))) for _ in range(n))
    return out[int(0.025 * n)], out[int(0.975 * n)]


def cell(arm, d, pad):
    v = list(acc[(arm, d, pad)].values())
    lo, hi = boot(v)
    return st.mean(v), lo, hi


def diff(a, b, paired):
    """mean of a minus b, interval over programs. a and b are dicts pid -> accuracy."""
    if paired:
        ids = sorted(set(a) & set(b)); d = [a[i] - b[i] for i in ids]
        lo, hi = boot(d); return st.mean(d), lo, hi
    out = sorted(st.mean(rng.choices(list(a.values()), k=len(a))) - st.mean(rng.choices(list(b.values()), k=len(b))) for _ in range(5000))
    return st.mean(a.values()) - st.mean(b.values()), out[125], out[4875]


L = []
P = L.append
P(f"MODEL {model}   programs per cell {len(acc[('A', DEPTHS[0], False)])}   questions per cell {2 * len(acc[('A', DEPTHS[0], False)])}")
P("\nACCURACY out of 100, mean over programs with 95 percent interval over programs. C is routed to B on HAND OFF. Failed translations count wrong.")
for pad in (False, True):
    P(f"\n{'PADDED' if pad else 'SHORT'}")
    P("arm   " + "".join(f"{'depth ' + str(d):>22}" for d in DEPTHS))
    for arm in ("A", "A0", "C", "B", "NR"):
        if (arm, DEPTHS[0], pad) not in acc:
            continue
        P(f"{arm:4s}  " + "".join(f"{cell(arm, d, pad)[0]:8.1f} [{cell(arm, d, pad)[1]:5.1f},{cell(arm, d, pad)[2]:5.1f}]" for d in DEPTHS))

P("\nREGISTERED PREDICTIONS")
d0, d1 = DEPTHS[0], DEPTHS[-1]
for pad in (False,):
    m, lo, hi = diff(acc[("A", d1, pad)], acc[("A", d0, pad)], False)
    P(f"P1 depth hurts A (short, depth {d1} minus depth {d0}): {m:.1f} [{lo:.1f}, {hi:.1f}]  -> {'holds' if m <= -10 and hi < 0 else 'does not hold'}")
for d in (10, 14, 18, 22):
    m, lo, hi = diff(acc[("A", d, True)], acc[("A", d, False)], True)
    P(f"P2 distractors hurt A (padded minus short, depth {d}): {m:.1f} [{lo:.1f}, {hi:.1f}]  -> {'holds' if m <= -10 and hi < 0 else 'does not hold'}")
for pad in (False, True):
    cells = [(d, cell("A0", d, pad)) for d in DEPTHS if d >= 6]
    ok = all(abs(c[0] - 50) <= 10 for _, c in cells)
    P(f"P3 answer only near chance ({'padded' if pad else 'short'}, depth 6 and up): " + ", ".join(f"d{d} {c[0]:.1f}" for d, c in cells) + f"  -> {'holds' if ok else 'does not hold'}")
for pad in (False, True):
    m, lo, hi = diff(acc[("B", d1, pad)], acc[("B", d0, pad)], False)
    P(f"P4 hand off flat ({'padded' if pad else 'short'}, depth {d1} minus depth {d0}, end to end): {m:.1f} [{lo:.1f}, {hi:.1f}]  -> point estimate {'within' if abs(m) <= 10 else 'outside'} 10 points; interval {'inside' if lo >= -10 and hi <= 10 else 'wider than'} plus or minus 10")
first = None
P("P5 hand off beats direct (B minus A, paired by program):")
for pad in (False, True):
    for d in DEPTHS:
        m, lo, hi = diff(acc[("B", d, pad)], acc[("A", d, pad)], True)
        flag = m >= 10 and lo > 0
        if flag and first is None:
            first = (d, pad)
        P(f"     {'padded' if pad else 'short '} depth {d:2d}: {m:6.1f} [{lo:6.1f},{hi:6.1f}] {'<- B ahead' if flag else ('A ahead' if m <= -10 and hi < 0 else '')}")
P(f"   shallowest cell where B is ahead by 10 or more with the interval above zero: {first if first else 'none'}")

# arm C
cs = [r for r in rows if r["arm"] == "C"]
A = {scoring.key(r): r["correct"] for r in rows if r["arm"] == "A"}
fl = [A[scoring.key(r)] for r in cs if r["ans"] == "hand"]
nf = [A[scoring.key(r)] for r in cs if r["ans"] != "hand"]
P(f"\nARM C hand off rate {100 * len(fl) / len(cs):.1f} percent ({len(fl)} of {len(cs)}); A accuracy on flagged items "
  f"{(100 * st.mean(fl)) if fl else float('nan'):.1f} (n={len(fl)}), unflagged {100 * st.mean(nf):.1f} (n={len(nf)}); unreadable C replies {sum(r['ans'] is None for r in cs)}")
for pad in (False, True):
    P(f"   hand off rate by depth ({'padded' if pad else 'short'}): " + ", ".join(
        f"d{d} {100 * st.mean(r['handed_off'] for r in cs if r['depth'] == d and r['padded'] == pad):.0f}%" for d in DEPTHS))
mA = 100 * st.mean(r["correct"] for r in rows if r["arm"] == "A"); mB = 100 * st.mean(r["correct"] for r in rows if r["arm"] == "B")
mC = 100 * st.mean(r["final_correct"] for r in cs)
cost = lambda arm: sum(r["usd"] for r in rows if r["arm"] == arm) / len([r for r in rows if r["arm"] == arm])
P(f"P7 selective hand off: overall accuracy A {mA:.1f}, B {mB:.1f}, C {mC:.1f}; cost per item A {cost('A'):.5f}, B {cost('B'):.5f}, C {st.mean(r['cost_final'] for r in cs):.5f}")

P("\nTRANSLATION (arm B), per program")
for pad in (False, True):
    for d in DEPTHS:
        pr = [r for r in rows if r["arm"] == "B" and r["depth"] == d and r["padded"] == pad and r["raw"]]
        run = [r for r in rows if r["arm"] == "B" and r["depth"] == d and r["padded"] == pad and r["ran"]]
        P(f"  {'padded' if pad else 'short '} depth {d:2d}: ran {sum(r['ran'] for r in pr):2d}/{len(pr)}  timeouts {sum(r['timeout'] for r in pr)}  "
          f"accuracy among runnable {100 * st.mean(r['correct'] for r in run):5.1f}  errors {dict(sorted(__import__('collections').Counter(r['form_err'] for r in pr if r['form_err']).items()))}")
P("\nTOKENS and COST")
for arm in ("A", "A0", "C", "NR", "B"):
    for pad in (False, True):
        rs = [r for r in rows if r["arm"] == arm and r["padded"] == pad and (arm != "B" or r["raw"])]
        if rs:
            P(f"  {arm:3s} {'padded' if pad else 'short '} in {st.mean(r['in'] for r in rs):6.0f} out {st.mean(r['out'] for r in rs):6.0f}  cost per call {st.mean(r['usd'] for r in rs):.5f}")
P(f"\nunreadable replies by arm: " + ", ".join(f"{a} {sum(r['ans'] is None for r in rows if r['arm'] == a)}" for a in ('A', 'A0', 'C', 'NR', 'B')))
P(f"run spend {sum(r['usd'] for r in rows):.4f} US dollars over {len(rows)} rows")
text = "\n".join(L)
(D / "analysis.txt").write_text(text + "\n")
print(text)
