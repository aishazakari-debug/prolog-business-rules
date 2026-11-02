"""Checks for policy_env.py. Run with python3 test_policy_env.py (needs swipl)."""
import collections
import policy_env as P

ok = lambda m: print("ok  ", m)
DEPTHS = (2, 4, 6, 8, 10, 12)

for padded in (False, True):
    for d in DEPTHS:
        for pid in range(1, 31):
            g = P.gen_program(d, pid, padded)
            assert P.derivation_depth(g) == d, (d, pid)
            assert P.gold(g, g.a) == 1 and P.gold(g, g.b) == 0, (d, pid)
            if padded:
                s = P.gen_program(d, pid, False)
                heads_pad = {r.head for r in g.rules} - {r.head for r in s.rules}
                assert not heads_pad & set(g.chain_heads)
                assert {(r.head, r.body) for r in s.rules} <= {(r.head, r.body) for r in g.rules}, "padding changed a chain rule"
                assert P.gold(s, s.a) == P.gold(g, g.a) and P.gold(s, s.b) == P.gold(g, g.b)
                assert not any(x in set(g.chain_heads) for r in g.rules if r.head in heads_pad for x in r.body), "a padded rule uses a chain status"
            assert len(set(r.head for r in g.rules)) == len(g.rules), "two rules for one predicate"
ok("depth equals the requested depth, A provable, B not, padding adds no new proof path (depths 2 to 12, ids 1 to 30)")

for d in DEPTHS:
    for pid in range(1, 11):
        g2 = P.gen_program(d, pid)
        g2.facts = g2.facts + [(g2.b, g2.missing)]
        assert P.gold(g2, g2.b) == 1
ok("restoring the missing fact makes B provable")

cnt = collections.Counter(P.gen_program(10, pid).fail_step for pid in range(1, 301))
assert len(cnt) >= 8 and max(cnt.values()) < 90, cnt
ok(f"failing step spread over the chain at depth 10: {dict(sorted(cnt.items()))}")

for padded in (False, True):
    for d in DEPTHS:
        for pid in range(1, 6):
            g = P.gen_program(d, pid, padded)
            clauses, err = P.check_form(P.gold_prolog(g), set(g.labels), {c.lower() for c, _ in g.facts})
            assert err is None, err
            res = P.run_prolog(clauses, g.labels, P.goals_for(g))
            assert res == ["yes", "no"], (d, pid, padded, res)
ok("SWI-Prolog agrees with the checker on 60 gold programs (short and padded)")

g = P.gen_program(4, 1)
labels, custs = set(g.labels), {c.lower() for c, _ in g.facts}
bad = [":- shell('echo hi').", "halt.", "foo(X) :- bar(X), shell(X).", f"{g.target}(X) :- {g.target}(Y).", "write(hello).",
       f"{g.target}(X) :- \\+ {g.target}(X).", f"{next(iter(labels))}(unknown_name)."]
for b in bad:
    c, err = P.check_form(b, labels, custs)
    assert c is None, b
ok("form check rejects directives, system calls, negation, unknown names and mixed variables")

loop = f"{g.target}(X) :- {g.target}(X)."
c, err = P.check_form(loop, labels, custs)
assert c is not None
assert P.run_prolog(c, g.labels, [f"{g.target}(zzz)"], timeout=2.0) is None
ok("a looping program times out and returns no answer")

g = P.gen_program(6, 1)
assert g.target in P.rules_text(g)
assert P.parse_answer("Yes", "A0") == "yes" and P.parse_answer("no.", "A0") == "no" and P.parse_answer("steps", "A0") is None
assert P.parse_answer("steps...\nAnswer: no", "A") == "no" and P.parse_answer("maybe", "A") is None
assert P.parse_answer("hmm\nAnswer: HAND OFF", "C") == "hand" and P.parse_answer("Answer: Yes", "C") == "yes"
assert P.parse_answer("I would say HAND OFF but\nAnswer: yes", "C") == "yes"
assert P.extract_code("```prolog\na(x).\n```\nWait, more") == "a(x).\n"
assert P.extract_code("```prolog\na(x).\nb(y).") == "a(x).\nb(y)." and P.extract_code("a(x).") == "a(x)."
ok("answers parse and code extraction handles a missing closing fence")
print("all checks passed")

# ---- added after the outside review (2026-10-23) ----
# A and B hold the same number of facts, so the count cannot give the answer away
for padded in (False, True):
    for d in DEPTHS:
        for pid in range(1, 41):
            g = P.gen_program(d, pid, padded)
            na = sum(1 for c, _ in g.facts if c == g.a); nb = sum(1 for c, _ in g.facts if c == g.b)
            assert na == nb, (d, pid, padded, na, nb)
            assert P.gold(g, g.a) == 1 and P.gold(g, g.b) == 0 and P.derivation_depth(g) == d
ok("A and B hold the same number of facts at every depth, short and padded")

# the ledger reserves before dispatch, persists, and refuses calls that could pass the cap
import json, tempfile, os
from clients import Ledger, BudgetExceeded
d = tempfile.mkdtemp(); path = os.path.join(d, "l.json"); json.dump({"total_usd": 0.90}, open(path, "w"))
L = Ledger(1.00, path)
L.reserve(0.04); L.reserve(0.04)
try:
    L.reserve(0.04); raise SystemExit("third reserve should have been refused")
except BudgetExceeded:
    pass
L.add(0.03, 10, 10, est_usd=0.04)
assert abs(L.total - 0.93) < 1e-9 and abs(json.load(open(path))["total_usd"] - 0.93) < 1e-9
L2 = Ledger(1.00, path)
assert abs(L2.total - 0.93) < 1e-9
ok("ledger reserves before dispatch, refuses calls that could pass the cap, and persists the project total")

# arm C routing: HAND OFF takes arm B's answer, and the translation is charged once per program, only when requested
import scoring
def row(arm, pid, cust, correct, ans, usd, raw="x"):
    return {"arm": arm, "depth": 6, "pid": pid, "padded": False, "customer": cust, "gold": 1, "ans": ans, "correct": correct,
            "usd": usd, "raw": raw, "in": 1, "out": 1, "ran": True, "form_ok": True, "form_err": None, "timeout": False}
rows = [row("B", 1, "A", 1, "yes", 0.01), row("B", 1, "B", 0, "yes", 0, raw=""), row("B", 2, "A", 1, "yes", 0.02),
        row("C", 1, "A", 0, "hand", 0.001), row("C", 1, "B", 0, "hand", 0.001), row("C", 2, "A", 1, "yes", 0.001)]
rows[1]["gold"] = 0
scoring.route_c(rows)
c = [r for r in rows if r["arm"] == "C"]
assert [r["final_correct"] for r in c] == [1, 0, 1]
assert abs(sum(r["cost_final"] for r in c) - (0.003 + 0.01)) < 1e-9
ok("arm C routing: HAND OFF takes arm B's answer, translation charged once and only when requested")
print("all added checks passed")
