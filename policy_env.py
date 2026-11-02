"""Plain language policies with chained rules, a checker, prompts, and a restricted Prolog runner.

Each program has a chain of rules of a given depth. Customer A has every fact the chain needs, so the question about A is
provable. Customer B lacks exactly one needed fact, so the same question about B is not provable. Chance is 50.
Labels (statuses and facts) are invented words. Truth depends only on the facts, never on what a word means.
All randomness is seeded from the depth and the program id, so every program can be rebuilt exactly.
"""
from __future__ import annotations

import os
import random
import re
import subprocess
import tempfile
from dataclasses import dataclass, field

SWIPL = "/opt/homebrew/bin/swipl"
NAMES = ("Mira Dario Elan Fenna Galen Hana Ivo Juno Kira Lior Mateo Nia Oren Pia Quin Rafa Sana Tomas Uma Vera Wren Xavi "
         "Yara Zane Anya Bram Cleo Dov Edda Finn").split()
_r = random.Random("policy-labels-v1")
_C, _V = "bcdfgklmnprstvz", "aeiou"
_pool, _seen = [], set()
while len(_pool) < 400:
    w = "".join(_r.choice(_C) if i % 2 == 0 else _r.choice(_V) for i in range(5))
    if w not in _seen:
        _seen.add(w); _pool.append(w)
LABELS = tuple(_pool)

TEMPLATES = (
    "A customer is {x} if the customer is {a} and {b}.",
    "Any customer who is {a} and also {b} is classed as {x}.",
    "The status {x} is granted when a customer is both {a} and {b}.",
    "If a customer is {a} and {b}, then the customer is {x}.",
)


@dataclass
class Rule:
    head: str
    body: tuple
    style: int = 0


@dataclass
class Program:
    depth: int
    pid: int
    padded: bool
    rules: list            # all rules in display order
    facts: list            # (customer, label) in display order
    target: str            # the status being asked about
    a: str                 # customer who has every needed fact
    b: str                 # customer missing one needed fact
    missing: str           # the label B lacks
    fail_step: int         # chain step (1..depth) where B's missing fact is needed
    chain_heads: list = field(default_factory=list)
    labels: list = field(default_factory=list)


def gen_program(depth: int, pid: int, padded: bool = False) -> Program:
    rng = random.Random(f"prog-{depth}-{pid}")
    words = list(LABELS)
    rng.shuffle(words)
    take = lambda n: [words.pop() for _ in range(n)]
    names = NAMES[:]
    rng.shuffle(names)
    chain = take(depth)
    rules, needed, step_of = [], [], {}
    for i, head in enumerate(chain, start=1):
        if i == 1:
            fa, fb = take(2)
            body = (fa, fb); new_f = [fa, fb]
        else:
            if rng.random() < 0.5:
                side = take(1)[0]; sa, sb = take(2)
                rules.append(Rule(side, (sa, sb), rng.randrange(4)))
                new_f = [sa, sb]; body = (chain[i - 2], side)
            else:
                f = take(1)[0]; new_f = [f]; body = (chain[i - 2], f)
        rules.append(Rule(head, body, rng.randrange(4)))
        for f in new_f:
            needed.append(f); step_of[f] = i
    missing = rng.choice(needed)
    decoys = take(3)           # the third decoy goes to B only, so A and B hold the same number of facts
    a, b = names.pop(), names.pop()
    facts = [(a, f) for f in needed + decoys[:2]] + [(b, f) for f in needed if f != missing] + [(b, d) for d in decoys]
    labels = list(dict.fromkeys([r.head for r in rules] + [l for r in rules for l in r.body] + [f for _, f in facts]))
    if padded:
        attrs = take(30)
        others = [names.pop() for _ in range(8)]
        pad_heads = []
        for _ in range(85):
            pool = attrs + needed + pad_heads[-6:]          # may lure with chain facts, never uses a chain status
            x = take(1)[0]
            p1, p2 = rng.sample(pool, 2)
            rules.append(Rule(x, (p1, p2), rng.randrange(4)))
            pad_heads.append(x)
        for o in others:
            for f in rng.sample(attrs + needed + decoys, 8):
                facts.append((o, f))
        for f in rng.sample(attrs, 3):
            facts.append((a, f)); facts.append((b, f))
        labels = list(dict.fromkeys(labels + attrs + pad_heads))
    rng.shuffle(rules)
    rng.shuffle(facts)
    return Program(depth, pid, padded, rules, facts, chain[-1], a, b, missing, step_of[missing], chain, labels)


def derive(prog: Program, customer: str) -> set:
    have = {f for c, f in prog.facts if c == customer}
    changed = True
    while changed:
        changed = False
        for r in prog.rules:
            if r.head not in have and all(x in have for x in r.body):
                have.add(r.head); changed = True
    return have


def gold(prog: Program, customer: str) -> int:
    return int(prog.target in derive(prog, customer))


def derivation_depth(prog: Program) -> int:
    heads = {r.head: r for r in prog.rules}
    memo = {}
    def d(l):
        if l not in heads: return 0
        if l not in memo: memo[l] = 1 + max(d(x) for x in heads[l].body)
        return memo[l]
    return d(prog.target)


def rules_text(prog: Program) -> str:
    return "\n".join(TEMPLATES[r.style].format(x=r.head, a=r.body[0], b=r.body[1]) for r in prog.rules)


def facts_text(prog: Program) -> str:
    return "\n".join(f"{c} is {f}." for c, f in prog.facts)


def questions(prog: Program):
    return [(prog.a, f"Is {prog.a} {prog.target}?", gold(prog, prog.a)), (prog.b, f"Is {prog.b} {prog.target}?", gold(prog, prog.b))]


# ---------------- prompts (frozen after the pilot) ----------------
SYSTEM_ONLY_EXTRA = " Never show your work. Your entire reply must be exactly one word."
SYSTEM_BASE = ("You apply a company's written rules to customer records. Use only the rules and facts given. A customer has a "
               "status only if the rules give it from the facts. The words in the rules are invented labels, so their "
               "everyday meaning does not matter. If the status cannot be shown from the rules and facts, the answer is no.")
SYSTEM_TRANSLATE = ("You translate a company's written rules and facts into Prolog. The words in the rules are invented labels. "
                    "Do not judge or answer anything. Only translate.")


def user_direct(prog: Program, question: str, mode: str, rules: bool = True) -> str:
    """Modes. A: may reason briefly, ends with Answer: yes or no. A0: answer only. C: like A, may end with Answer: HAND OFF."""
    block = (f"Rules:\n{rules_text(prog)}\n\n" if rules else "Rules: the rules are not shown.\n\n") + f"Facts:\n{facts_text(prog)}\n\n"
    tail = {"A": "Work through the rules briefly, then end with a last line that reads Answer: yes or Answer: no.",
            "A0": "Reply with the single word yes or no and nothing else.",
            "C": ("Work through the rules briefly, then end with a last line that reads Answer: yes, Answer: no, or Answer: HAND OFF. "
                  "Use HAND OFF only if you do not trust yourself to apply these rules correctly here.")}[mode]
    return f"{block}Question: {question}\n{tail}"


def system_for(mode: str) -> str:
    return SYSTEM_BASE + (SYSTEM_ONLY_EXTRA if mode == "A0" else "")


def user_translate(prog: Program) -> str:
    return (f"Rules:\n{rules_text(prog)}\n\nFacts:\n{facts_text(prog)}\n\n"
            "Translate every rule and every fact into Prolog. Use each label exactly as written, in lower case, as a predicate "
            "name with one argument, and use each customer name in lower case as the argument. A fact looks like label(name). "
            "A rule looks like head(X) :- first(X), second(X). Put every clause on its own line and end each with a period. "
            "Reply with only the Prolog code in one code block. Do not add questions, comments, or anything else.")


def parse_answer(text: str, mode: str):
    t = (text or "").strip().lower()
    if mode == "A0":
        m = re.match(r"\W*(yes|no)\b", t)
        return m.group(1) if m else None
    m = re.findall(r"answer:\s*(yes|no|hand off)\b", t)
    if not m:
        return None
    return "hand" if m[-1] == "hand off" else m[-1]


# ---------------- restricted Prolog runner ----------------
def extract_code(text: str) -> str:
    """Code between the first pair of fences. A reply that opens a fence and never closes it is read to its end."""
    text = text or ""
    m = re.search(r"```[a-zA-Z]*\s*\n(.*?)```", text, re.S)
    if m:
        return m.group(1)
    m = re.search(r"```[a-zA-Z]*\s*\n(.*)$", text, re.S)
    return m.group(1) if m else text


def check_form(code: str, labels: set, customers: set):
    """Returns (clauses, error). Only unary facts and rules over one variable, with known labels and names, are accepted."""
    code = "\n".join(l for l in code.splitlines() if not l.strip().startswith("%"))
    chunks = [c.strip() for c in code.split(".") if c.strip()]
    if not chunks:
        return None, "empty"
    out = []
    for c in chunks:
        c = re.sub(r"\s+", " ", c)
        if ":-" in c:
            head, body = [x.strip() for x in c.split(":-", 1)]
            parts = [p.strip() for p in body.split(",")]
            hv = re.fullmatch(r"([a-z][a-z0-9_]*)\(([A-Z][A-Za-z0-9_]*)\)", head)
            if not hv or not parts:
                return None, "bad rule head"
            var = hv.group(2)
            if hv.group(1) not in labels:
                return None, "unknown predicate"
            for p in parts:
                m = re.fullmatch(r"([a-z][a-z0-9_]*)\(([A-Z][A-Za-z0-9_]*)\)", p)
                if not m or m.group(2) != var or m.group(1) not in labels:
                    return None, "bad rule body"
            out.append(c)
        else:
            m = re.fullmatch(r"([a-z][a-z0-9_]*)\(([a-z][a-z0-9_]*)\)", c)
            if not m or m.group(1) not in labels or m.group(2) not in customers:
                return None, "bad fact"
            out.append(c)
    return out, None


def run_prolog(clauses: list, labels: list, goals: list, timeout: float = 5.0):
    """Runs the accepted clauses in a separate swipl process. Returns a list of 'yes', 'no' or 'error' per goal, or None on timeout."""
    src = "".join(f":- dynamic {l}/1.\n" for l in labels) + "".join(c + ".\n" for c in clauses)
    goal_list = "[" + ",".join(goals) + "]"
    runner = (f"forall(member(G,{goal_list}),(catch((call(G)->R=yes;R=no),_,R=error),writeln(R)))")
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "p.pl")
        open(path, "w").write(src)
        try:
            r = subprocess.run([SWIPL, "-q", "-g", runner, "-t", "halt", path], capture_output=True, text=True, timeout=timeout, cwd=d)
        except subprocess.TimeoutExpired:
            return None
    res = [x.strip() for x in r.stdout.splitlines() if x.strip() in ("yes", "no", "error")]
    return res if len(res) == len(goals) else ["error"] * len(goals)


def gold_prolog(prog: Program) -> str:
    lines = [f"{f}({c.lower()})." for c, f in prog.facts]
    lines += [f"{r.head}(X) :- {r.body[0]}(X), {r.body[1]}(X)." for r in prog.rules]
    return "\n".join(lines)


def goals_for(prog: Program):
    return [f"{prog.target}({prog.a.lower()})", f"{prog.target}({prog.b.lower()})"]
