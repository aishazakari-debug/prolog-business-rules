"""Qwen3 4B Instruct runs of the policy task on a Colab T4. Uploaded to /content with policy_env.py.

Same prompts and programs as the API runs (policy_env.py, frozen). Arms A and A0 on every cell, arm B on short items only,
no arm C. Sampling: temperature 0.8, top_p 1.0, no top_k cut. Rows are written as they finish, and a restart skips finished jobs.
Arm B is saved as raw text only. The Prolog check and run happen later on the Mac, where SWI-Prolog is installed.
"""
import json
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/content")
import policy_env as P

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
NAME = "qwen3-4b-instruct-2507"
OUT = "/content/out"
os.makedirs(OUT, exist_ok=True)
_tok = _model = None
DEPTHS = (2, 6, 10, 14, 18, 22)
MAX_NEW = {"A": 1200, "A0": 8, "B": 2600}


def load():
    global _tok, _model
    if _model is None:
        _tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
        _model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to("cuda").eval()
    return _tok, _model


def generate(pairs, max_new, seed):
    tok, model = load()
    torch.manual_seed(seed)
    texts = [tok.apply_chat_template([{"role": "system", "content": s}, {"role": "user", "content": u}],
                                     tokenize=False, add_generation_prompt=True) for s, u in pairs]
    enc = tok(texts, return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad():
        out = model.generate(**enc, do_sample=True, temperature=0.8, top_p=1.0, top_k=0, max_new_tokens=max_new,
                             pad_token_id=tok.pad_token_id)
    new = out[:, enc["input_ids"].shape[1]:]
    res = []
    for j in range(new.shape[0]):
        ids = new[j]
        res.append((tok.decode(ids, skip_special_tokens=True), int(enc["attention_mask"][j].sum()), int((ids != tok.pad_token_id).sum())))
    del enc, out, new
    torch.cuda.empty_cache()
    return res


def batch_size(arm, padded):
    if arm == "A0":
        return 2 if padded else 16
    if arm == "A":
        return 2 if padded else 10
    return 8                      # B, short only


def done_set(path):
    done = set()
    if os.path.exists(path):
        for l in open(path):
            r = json.loads(l)
            done.add((r["arm"], r["depth"], r["pid"], r["padded"]))
    return done


def run_arm(arm, padded, programs, fname, depths=DEPTHS, first_id=1):
    """Runs one arm over one length setting. Items from all depths are batched together, shortest prompts first."""
    path = f"{OUT}/{fname}"
    done = done_set(path)
    jobs = [(d, pid) for d in depths for pid in range(first_id, first_id + programs) if (arm, d, pid, padded) not in done]
    print(f"{arm} {'padded' if padded else 'short'}: {len(jobs)} programs to run", flush=True)
    items = []
    for d, pid in jobs:
        g = P.gen_program(d, pid, padded)
        if arm == "B":
            items.append((d, pid, g, None, P.user_translate(g)))
        else:
            for cust, q, gold in P.questions(g):
                items.append((d, pid, g, (cust, gold), P.user_direct(g, q, arm)))
    items.sort(key=lambda x: len(x[4]))
    bs, t0 = batch_size(arm, padded), time.time()
    pending = {}
    with open(path, "a") as f:
        for a in range(0, len(items), bs):
            chunk = items[a:a + bs]
            sysp = P.SYSTEM_TRANSLATE if arm == "B" else P.system_for(arm)
            rs = generate([(sysp, it[4]) for it in chunk], MAX_NEW[arm], 7000 + a)
            for (d, pid, g, q, _), (text, n_in, n_out) in zip(chunk, rs):
                if arm == "B":
                    f.write(json.dumps({"arm": "B", "depth": d, "pid": pid, "padded": padded, "raw": text, "in": n_in, "out": n_out, "usd": 0.0}) + "\n")
                else:
                    cust, gold = q
                    ans = P.parse_answer(text, arm)
                    f.write(json.dumps({"arm": arm, "depth": d, "pid": pid, "padded": padded, "customer": cust, "gold": gold, "raw": text,
                                        "ans": ans, "correct": int(ans == ("yes" if gold else "no")), "in": n_in, "out": n_out, "usd": 0.0}) + "\n")
            f.flush()
            print(f"{arm} {'padded' if padded else 'short'} {min(a + bs, len(items))}/{len(items)}  {time.time() - t0:.0f} s", flush=True)


def smoke():
    t0 = time.time()
    g = P.gen_program(6, 101, False)
    qs = P.questions(g)
    rs = generate([(P.system_for("A0"), P.user_direct(g, q[1], "A0")) for q in qs], 8, 1)
    print("A0 replies:", [r[0] for r in rs], "parsed", [P.parse_answer(r[0], "A0") for r in rs], "gold", [q[2] for q in qs], flush=True)
    rs = generate([(P.system_for("A"), P.user_direct(g, q[1], "A")) for q in qs], 1200, 2)
    print("A replies parsed:", [P.parse_answer(r[0], "A") for r in rs], "| output tokens", [r[2] for r in rs], "| tail", repr(rs[0][0][-80:]), flush=True)
    rs = generate([(P.SYSTEM_TRANSLATE, P.user_translate(g))], 2600, 3)
    clauses, err = P.check_form(P.extract_code(rs[0][0]), set(g.labels), {c.lower() for c, _ in g.facts})
    print("B form ok:", clauses is not None, err, "| output tokens", rs[0][2], "| head", repr(rs[0][0][:80]), flush=True)
    gp = P.gen_program(6, 101, True)
    rs = generate([(P.system_for("A"), P.user_direct(gp, P.questions(gp)[0][1], "A"))] * 2, 1200, 4)
    print("A padded x2: prompt tokens", rs[0][1], "output tokens", [r[2] for r in rs], "| parsed", [P.parse_answer(r[0], "A") for r in rs], flush=True)
    print("seconds", round(time.time() - t0, 1), "| peak GPU GB", round(torch.cuda.max_memory_allocated() / 1e9, 2))
