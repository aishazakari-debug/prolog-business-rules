"""Independent check. Recomputes headline numbers straight from the saved call files (no numbers.json, no scoring.py)
and asserts that each appears in draft.md. Run: python3 check_draft.py"""
import json, re, statistics as st
from collections import defaultdict
from pathlib import Path
R = Path("results/full"); D = open("draft.md").read()
def rows(m): return [json.loads(l) for l in open(R / m / "calls.jsonl")]
H, Q = rows("claude-haiku-5-5"), rows("qwen3-4b-instruct-2507")
def must(x, label, nd=1):
    s = f"{abs(x):.{nd}f}"
    assert re.search(r"(?<![\d.])" + re.escape(s) + r"(?![\d])", D), f"MISSING {label}: {s}"
    print("ok", label, s)
def per_prog(rs, arm, pad, d):
    by = defaultdict(list)
    for r in rs:
        if r["arm"] == arm and r["padded"] == pad and r["depth"] == d: by[r["pid"]].append(r["correct"])
    return {p: 100 * st.mean(v) for p, v in by.items()}
def acc(rs, arm, pad, d): return st.mean(per_prog(rs, arm, pad, d).values())
for name, rs, arms in (("haiku", H, ("A", "A0", "B")), ("qwen", Q, ("A", "A0", "B"))):
    for arm in arms:
        for pad in (False, True):
            if not per_prog(rs, arm, pad, 2): continue
            for d in (2, 6, 10, 14, 18, 22): must(acc(rs, arm, pad, d), f"{name} {arm} {'padded' if pad else 'short'} d{d}")
# paired difference B minus A, Haiku padded 22
b, a = per_prog(H, "B", True, 22), per_prog(H, "A", True, 22)
must(st.mean(b[p] - a[p] for p in b), "haiku B minus A padded 22")
b, a = per_prog(H, "B", True, 2), per_prog(H, "A", True, 2); must(st.mean(b[p] - a[p] for p in b), "haiku B minus A padded 2")
# translations that ran
for name, rs in (("haiku", H), ("qwen", Q)):
    B = [r for r in rs if r["arm"] == "B" and r["raw"]]
    for pad in (False, True):
        sub = [r for r in B if r["padded"] == pad]
        if sub: print(name, "pad" if pad else "short", "ran", sum(r["ran"] for r in sub), "of", len(sub))
q22 = [r for r in Q if r["arm"] == "A" and r["depth"] == 22 and not r["padded"]]
assert re.search(r"\b%d of 60 replies hit" % sum(r["out"] >= 2500 for r in q22), D), "Qwen cap count"
print("ok qwen short 22 capped", sum(r["out"] >= 2500 for r in q22))
hs = [r for r in H if r["arm"] == "C" and r["ans"] == "hand"]; assert len(hs) == 1 and re.search(r"HAND OFF 1 time in 720", D); print("ok hand off 1 of 720")
must(sum(r["usd"] for r in H), "haiku run cost", 2)
must(json.load(open("results/ledger.json"))["total_usd"], "project cost", 2)
for arm, pad in (("A", True), ("B", True)):
    rs = [r for r in H if r["arm"] == arm and r["padded"] == pad and (arm != "B" or r["raw"])]
    must(st.mean(r["usd"] for r in rs), f"haiku {arm} padded cost per call", 5)
print("rows", len(H), len(Q))
print("all checked numbers appear in the draft")
