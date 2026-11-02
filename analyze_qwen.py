"""Summarize Qwen's registered cells from scored, saved replies."""
from __future__ import annotations

import json
import random
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

DIR = Path(__file__).resolve().parent / "results" / "full" / "qwen3-4b-instruct-2507"
ROWS = [json.loads(line) for line in (DIR / "calls.jsonl").open() if line.strip()]
DEPTHS = (2, 6, 10, 14, 18, 22)
RNG = random.Random(0)

by_program = defaultdict(lambda: defaultdict(list))
for row in ROWS:
    by_program[(row["arm"], row["depth"], row["padded"])][row["pid"]].append(row["correct"])
ACC = {key: {pid: 100 * st.mean(values) for pid, values in grouped.items()}
       for key, grouped in by_program.items()}


def interval(values, repetitions=5000):
    values = list(values)
    means = sorted(st.mean(RNG.choices(values, k=len(values))) for _ in range(repetitions))
    return means[int(.025 * repetitions)], means[int(.975 * repetitions)]


def cell(arm, depth, padded):
    values = list(ACC[(arm, depth, padded)].values())
    return st.mean(values), *interval(values)


def paired(left, right):
    ids = sorted(set(left) & set(right))
    if len(ids) != len(left) or len(ids) != len(right):
        raise ValueError("Compared cells have different program IDs")
    gaps = [left[pid] - right[pid] for pid in ids]
    return st.mean(gaps), *interval(gaps)


def unpaired(left, right, repetitions=5000):
    a, b = list(left.values()), list(right.values())
    gaps = sorted(st.mean(RNG.choices(a, k=len(a))) - st.mean(RNG.choices(b, k=len(b)))
                  for _ in range(repetitions))
    return st.mean(a) - st.mean(b), gaps[125], gaps[4875]


lines = ["QWEN3 4B INSTRUCT 2507 | 30 programs and 60 paired questions per cell",
         "95% bootstrap intervals resample programs. B ran on short items only. A and A0 ran on short and padded items.",
         "Failed translations and unreadable answers count as wrong."]
for padded in (False, True):
    lines.append("\n" + ("PADDED" if padded else "SHORT"))
    for arm in (("A", "A0") if padded else ("A", "A0", "B")):
        parts = []
        for depth in DEPTHS:
            mean, lo, hi = cell(arm, depth, padded)
            parts.append(f"d{depth} {mean:.1f} [{lo:.1f}, {hi:.1f}]")
        lines.append(arm + "  " + " | ".join(parts))

m, lo, hi = unpaired(ACC[("A", 22, False)], ACC[("A", 2, False)])
lines.append(f"\nP1 A short, depth 22 minus depth 2: {m:.1f} [{lo:.1f}, {hi:.1f}]")
for depth in (10, 14, 18, 22):
    m, lo, hi = paired(ACC[("A", depth, True)], ACC[("A", depth, False)])
    lines.append(f"P2 A padded minus short, depth {depth}: {m:.1f} [{lo:.1f}, {hi:.1f}]")
for padded in (False, True):
    vals = [f"d{depth} {cell('A0', depth, padded)[0]:.1f}" for depth in DEPTHS if depth >= 6]
    lines.append(f"P3 A0 {'padded' if padded else 'short'}: " + ", ".join(vals))
for depth in DEPTHS:
    m, lo, hi = paired(ACC[("B", depth, False)], ACC[("A", depth, False)])
    lines.append(f"P5 B minus A, short depth {depth}: {m:.1f} [{lo:.1f}, {hi:.1f}]")
lines.append("P4, P6, P7: not tested by this Qwen design")

for arm in ("A", "A0"):
    subset = [row for row in ROWS if row["arm"] == arm]
    labels = Counter(row["ans"] for row in subset)
    lines.append(f"\n{arm} final answers: {dict(labels)}; "
                 f"output at 2,500-token cap: {sum(row['out'] >= 2500 for row in subset)}")
for padded in (False, True):
    for depth in DEPTHS:
        subset = [row for row in ROWS if row["arm"] == "A" and row["padded"] == padded and row["depth"] == depth]
        lines.append(f"A {'padded' if padded else 'short'} d{depth}: "
                     f"unreadable {sum(row['ans'] is None for row in subset)}, "
                     f"at output cap {sum(row['out'] >= 2500 for row in subset)}, "
                     f"mean output {st.mean(row['out'] for row in subset):.0f} tokens")
for depth in DEPTHS:
    subset = [row for row in ROWS if row["arm"] == "B" and row["depth"] == depth]
    programs = [row for row in subset if row["raw"]]
    runnable = [row for row in subset if row["ran"]]
    lines.append(f"B short d{depth}: ran {sum(row['ran'] for row in programs)}/{len(programs)} programs, "
                 f"accuracy among runnable {100 * st.mean(row['correct'] for row in runnable) if runnable else float('nan'):.1f}, "
                 f"form errors {dict(Counter(row['form_err'] for row in programs if row['form_err']))}")

output = "\n".join(lines) + "\n"
(DIR / "analysis.txt").write_text(output)
print(output)
