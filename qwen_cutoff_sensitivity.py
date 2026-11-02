"""Exploratory rerun of only answers cut off at 2,500 tokens.

Uses the same prompts and per-request sampling seeds as the full Qwen run, changing
only the reply limit to 4,000. This file is not part of the registered test.
"""
import json
import os
import sys
import time

sys.path.insert(0, "/content")
import policy_env as P

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
MAX_NEW = 4000


def main():
    from vllm import LLM, SamplingParams

    jobs = json.load(open("/content/cutoff_jobs.json"))
    out_path = "/content/out/cutoff_sensitivity.jsonl"
    os.makedirs("/content/out", exist_ok=True)
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            row = json.loads(line)
            done.add((row["depth"], row["pid"], row["padded"], row["customer"]))
    todo = [job for job in jobs if (job["depth"], job["pid"], job["padded"], job["customer"]) not in done]
    print(f"capped answers to rerun: {len(todo)} of {len(jobs)}", flush=True)
    llm = LLM(MODEL, dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.90, seed=0)
    tokenizer = llm.get_tokenizer()
    t0 = time.time()
    with open(out_path, "a") as file:
        for start in range(0, len(todo), 12):
            batch = todo[start:start + 12]
            prompts, params = [], []
            for job in batch:
                prog = P.gen_program(job["depth"], job["pid"], job["padded"])
                matches = [question for customer, question, _ in P.questions(prog) if customer == job["customer"]]
                if len(matches) != 1:
                    raise ValueError("question key is not unique")
                messages = [{"role": "system", "content": P.system_for("A")},
                            {"role": "user", "content": P.user_direct(prog, matches[0], "A")}]
                prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
                params.append(SamplingParams(temperature=0.8, top_p=1.0, top_k=-1,
                                             max_tokens=MAX_NEW, seed=job["seed"]))
            outputs = llm.generate(prompts, params, use_tqdm=False)
            for job, output in zip(batch, outputs):
                reply = output.outputs[0].text
                ans = P.parse_answer(reply, "A")
                row = {k: job[k] for k in ("depth", "pid", "padded", "customer", "gold", "seed")}
                row.update({"ans": ans, "correct": int(ans == ("yes" if job["gold"] else "no")),
                            "out": len(output.outputs[0].token_ids), "in": len(output.prompt_token_ids),
                            "prefix_matches": reply.startswith(job["old_raw"]), "raw": reply})
                file.write(json.dumps(row) + "\n")
            file.flush()
            print(f"rerun {min(start + 12, len(todo))}/{len(todo)} in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
