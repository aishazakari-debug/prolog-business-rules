"""Run the Qwen reply-limit sensitivity on Colab L4, then stop the session."""
import json
from pathlib import Path

import policy_env as P
import run_qwen_on_colab as Q

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "full" / "qwen3-4b-instruct-2507"
DEPTHS = (2, 6, 10, 14, 18, 22)


def jobs_from_full():
    rows = [json.loads(line) for line in (OUT / "calls_raw.jsonl").open() if line.strip()]
    jobs = []
    for row in rows:
        if row["arm"] != "A" or row["out"] < 2500:
            continue
        d, pid, padded, customer = row["depth"], row["pid"], row["padded"], row["customer"]
        prog = P.gen_program(d, pid, padded)
        questions = P.questions(prog)
        question_index = next(i for i, item in enumerate(questions) if item[0] == customer)
        position = DEPTHS.index(d) * 60 + (pid - 1) * 2 + question_index
        seed = 10_000 * (1 + position // 96) + position % 96
        jobs.append({"depth": d, "pid": pid, "padded": padded, "customer": customer,
                     "gold": row["gold"], "seed": seed, "old_raw": row["raw"]})
    return jobs


def main():
    Q.check_frozen()
    jobs = jobs_from_full()
    path = OUT / "cutoff_jobs.json"
    path.write_text(json.dumps(jobs))
    Q.say(f"exploratory sensitivity: {len(jobs)} capped answers; reply limit 2,500 to 4,000")
    try:
        Q.setup()
        Q.step("install vllm", "import subprocess, sys\nr = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'vllm'], capture_output=True, text=True)\nprint(r.stdout[-1500:], r.stderr[-1500:])\nassert r.returncode == 0", 1800)
        rc, out = Q.c("restart-kernel", "-s", Q.R.S, timeout=180)
        if rc:
            raise SystemExit("could not restart kernel after vLLM installation: " + Q.tail(out, 2))
        for local, remote in ((str(path), "/content/cutoff_jobs.json"),
                              (str(HERE / "qwen_cutoff_sensitivity.py"), "/content/qwen_cutoff_sensitivity.py")):
            rc, out = Q.c("upload", "-s", Q.R.S, local, remote)
            if rc:
                raise SystemExit("upload failed: " + Q.tail(out, 2))
        Q.step("qwen cutoff sensitivity", "import qwen_cutoff_sensitivity; qwen_cutoff_sensitivity.main()", 2700)
        rc, out = Q.c("download", "-s", Q.R.S, "/content/out/cutoff_sensitivity.jsonl",
                      str(OUT / "cutoff_sensitivity.jsonl"), timeout=180)
        if rc:
            raise SystemExit("download failed: " + Q.tail(out, 2))
        Q.say(Q.tail(out, 1))
    finally:
        Q.say(Q.tail(Q.c("stop", "-s", Q.R.S)[1], 1))
        Q.say("sessions: " + Q.tail(Q.c("sessions")[1], 3))
        Q.say("balance after: " + Q.c("usage")[1].splitlines()[0])


if __name__ == "__main__":
    main()
