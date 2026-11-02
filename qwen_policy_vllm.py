"""Qwen policy run with vLLM continuous batching on the Colab L4 GPU.

Same prompts, programs, and sampling as the API runs: temperature 0.8, top_p 1.0, no top_k cut, per request seeds.
Arms A and A0 on every cell, arm B on short items only. Rows have the same fields as qwen_policy.py.
"""
import json
import os
import sys
import time

sys.path.insert(0, "/content")
import policy_env as P

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
OUT = "/content/out"
os.makedirs(OUT, exist_ok=True)
DEPTHS = (2, 6, 10, 14, 18, 22)
MAX_NEW = {"A": 2500, "A0": 8, "B": 2600}
_llm = None


def load():
    global _llm
    if _llm is None:
        from vllm import LLM
        _llm = LLM(MODEL, dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.90, enforce_eager=False, seed=0)
    return _llm


def generate(pairs, max_new, seed0):
    from vllm import SamplingParams
    llm = load()
    tok = llm.get_tokenizer()
    prompts = [tok.apply_chat_template([{"role": "system", "content": s}, {"role": "user", "content": u}], tokenize=False, add_generation_prompt=True)
               for s, u in pairs]
    sps = [SamplingParams(temperature=0.8, top_p=1.0, top_k=-1, max_tokens=max_new, seed=seed0 + i) for i in range(len(prompts))]
    outs = llm.generate(prompts, sps, use_tqdm=False)
    return [(o.outputs[0].text, len(o.prompt_token_ids), len(o.outputs[0].token_ids)) for o in outs]


def done_set(path):
    done = set()
    if os.path.exists(path):
        for l in open(path):
            r = json.loads(l)
            done.add((r["arm"], r["depth"], r["pid"], r["padded"]))
    return done


def run_arm(arm, padded, programs, fname, depths=DEPTHS, first_id=1, chunk=96):
    path = f"{OUT}/{fname}"
    done = done_set(path)
    jobs = [(d, pid) for d in depths for pid in range(first_id, first_id + programs) if (arm, d, pid, padded) not in done]
    print(f"{arm} {'padded' if padded else 'short'}: {len(jobs)} programs to run", flush=True)
    items = []
    for d, pid in jobs:
        g = P.gen_program(d, pid, padded)
        if arm == "B":
            items.append((d, pid, None, P.user_translate(g)))
        else:
            for cust, q, gold in P.questions(g):
                items.append((d, pid, (cust, gold), P.user_direct(g, q, arm)))
    t0 = time.time()
    with open(path, "a") as f:
        for a in range(0, len(items), chunk):
            part = items[a:a + chunk]
            sysp = P.SYSTEM_TRANSLATE if arm == "B" else P.system_for(arm)
            rs = generate([(sysp, it[3]) for it in part], MAX_NEW[arm], 10_000 * (1 + a // chunk))
            for (d, pid, q, _), (text, n_in, n_out) in zip(part, rs):
                if arm == "B":
                    f.write(json.dumps({"arm": "B", "depth": d, "pid": pid, "padded": padded, "raw": text, "in": n_in, "out": n_out, "usd": 0.0}) + "\n")
                else:
                    cust, gold = q
                    ans = P.parse_answer(text, arm)
                    f.write(json.dumps({"arm": arm, "depth": d, "pid": pid, "padded": padded, "customer": cust, "gold": gold, "raw": text,
                                        "ans": ans, "correct": int(ans == ("yes" if gold else "no")), "in": n_in, "out": n_out, "usd": 0.0}) + "\n")
            f.flush()
            print(f"{arm} {'padded' if padded else 'short'} {min(a + chunk, len(items))}/{len(items)}  {time.time() - t0:.0f} s", flush=True)


def smoke():
    t0 = time.time()
    g = P.gen_program(6, 101, False); qs = P.questions(g)
    rs = generate([(P.system_for("A0"), P.user_direct(g, q[1], "A0")) for q in qs], 8, 1)
    print("A0 replies:", [r[0] for r in rs], "parsed", [P.parse_answer(r[0], "A0") for r in rs], flush=True)
    gp = P.gen_program(10, 101, True); qp = P.questions(gp)
    pairs = [(P.system_for("A"), P.user_direct(g, q[1], "A")) for q in qs] + [(P.system_for("A"), P.user_direct(gp, q[1], "A")) for q in qp]
    t1 = time.time()
    rs = generate(pairs, 1200, 2)
    dt = time.time() - t1
    toks = sum(r[2] for r in rs)
    print(f"A batch of {len(pairs)}: parsed {sum(P.parse_answer(r[0], 'A') is not None for r in rs)}/{len(pairs)}, output tokens {toks}, seconds {dt:.0f}, tokens per second {toks / dt:.0f}", flush=True)
    rs = generate([(P.SYSTEM_TRANSLATE, P.user_translate(g))], 2600, 3)
    clauses, err = P.check_form(P.extract_code(rs[0][0]), set(g.labels), {c.lower() for c, _ in g.facts})
    print("B form ok:", clauses is not None, err, "| output tokens", rs[0][2], flush=True)
    print("total seconds", round(time.time() - t0, 1))
