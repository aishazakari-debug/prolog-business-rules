"""Scoring that works on saved call files, with no model calls. Used for the pilot and for the registered runs.

Arm C is scored as the plan says. A reply ending HAND OFF is replaced by arm B's answer for the same item (counted wrong if
the translation did not run). The cost of a translation is charged to C only for programs where C asked for one.
"""
import statistics as st
import policy_env as P


def key(r):
    return (r["depth"], r["pid"], r["padded"], r["customer"])


def route_c(rows):
    """Adds final_correct, handed_off and cost_final to every arm C row. Returns the rows."""
    B = {key(r): r for r in rows if r["arm"] == "B"}
    Bcost = {(r["depth"], r["pid"], r["padded"]): r["usd"] for r in rows if r["arm"] == "B" and r["raw"]}
    asked = {(r["depth"], r["pid"], r["padded"]) for r in rows if r["arm"] == "C" and r["ans"] == "hand"}
    paid = set()
    for r in rows:
        if r["arm"] != "C":
            continue
        r["handed_off"] = int(r["ans"] == "hand")
        r["final_correct"] = B[key(r)]["correct"] if r["ans"] == "hand" else r["correct"]
        pk = (r["depth"], r["pid"], r["padded"])
        extra = 0.0
        if r["ans"] == "hand" and pk not in paid:
            paid.add(pk); extra = Bcost.get(pk, 0.0)
        r["cost_final"] = r["usd"] + extra
    return rows


def summarize(rows, depths, header=""):
    route_c(rows)
    sel = lambda **kw: [r for r in rows if all(r[k] == v for k, v in kw.items())]
    m = lambda xs: (100 * st.mean(xs)) if xs else float("nan")
    out = [header] if header else []
    out.append("ACCURACY (out of 100); chance 50; unreadable and failed translations counted wrong; C routed to B on HAND OFF")
    out.append("arm  length  " + "".join(f"depth {d:<3}  " for d in depths) + " unreadable")
    for arm, pads in (("A", (False, True)), ("A0", (False, True)), ("C", (False, True)), ("NR", (False,)), ("B", (False, True))):
        for pad in pads:
            rs = sel(arm=arm, padded=pad)
            if not rs:
                continue
            f = "final_correct" if arm == "C" else "correct"
            out.append(f"{arm:3s}  {'padded' if pad else 'short ':6s}  " + "".join(f"{m([r[f] for r in rs if r['depth']==d]):7.1f}  " for d in depths)
                       + f" {sum(r['ans'] is None for r in rs)} of {len(rs)}")
    if sel(arm="C"):
        out.append("\nARM C: HAND OFF rate, arm A accuracy on the same items by flag, and cost per item (US dollars, translation charged when requested)")
        A = {key(r): r["correct"] for r in sel(arm="A")}
        for pad in (False, True):
            for d in depths:
                cs = sel(arm="C", padded=pad, depth=d)
                if not cs:
                    continue
                fl = [A[key(r)] for r in cs if r["ans"] == "hand"]
                nf = [A[key(r)] for r in cs if r["ans"] != "hand"]
                out.append(f"  {'padded' if pad else 'short '} depth {d:2d}: hand off {m([r['handed_off'] for r in cs]):3.0f}% (n={len(cs)})  A accuracy flagged "
                           f"{m(fl):5.1f} (n={len(fl)}) unflagged {m(nf):5.1f} (n={len(nf)})  cost per item {st.mean(r['cost_final'] for r in cs):.5f}")
    if sel(arm="B"):
        out.append("\nTRANSLATION (arm B, per program)")
        for pad in (False, True):
            for d in depths:
                allq = sel(arm="B", padded=pad, depth=d)
                if not allq:
                    continue
                prog = [r for r in allq if r["raw"]]
                run = [r for r in allq if r["ran"]]
                out.append(f"  {'padded' if pad else 'short '} depth {d:2d}: form ok {sum(r['form_ok'] for r in prog)}/{len(prog)}  ran {sum(r['ran'] for r in prog)}/{len(prog)}  "
                           f"timeouts {sum(r['timeout'] for r in prog)}  accuracy among runnable {m([r['correct'] for r in run]):5.1f}  "
                           f"form errors {sorted({r['form_err'] for r in prog if r['form_err']})}")
    out.append("\nTOKENS per call (mean input, mean output)")
    for arm in ("A", "A0", "C", "NR", "B"):
        for pad in (False, True):
            rs = [r for r in sel(arm=arm, padded=pad) if arm != "B" or r["raw"]]
            if rs:
                out.append(f"  {arm:3s} {'padded' if pad else 'short '}  in {st.mean(r['in'] for r in rs):7.0f}  out {st.mean(r['out'] for r in rs):6.0f}")
    out.append(f"\ncalls {len(rows)}  spend in these rows ${sum(r['usd'] for r in rows):.4f}")
    return "\n".join(out)
