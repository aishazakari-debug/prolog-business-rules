"""Controller for the Qwen3 4B run on Colab L4. Polls status files and stops the session.

    PATH="$HOME/.local/bin:$PATH" python3 run_qwen_on_colab.py smoke    # model loads and replies parse
    PATH="$HOME/.local/bin:$PATH" python3 run_qwen_on_colab.py full     # A0, A short, B short, A padded, saved to results/full/

No API key is involved. Model: Qwen/Qwen3-4B-Instruct-2507 (Apache 2.0). Finish the scoring of arm B with finish_qwen.py.
"""
import hashlib
import inspect
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "structured-annotation-small-model"))
sys.path.insert(0, HERE)
import run_on_colab as R  # helper functions from the earlier run: c, say, tail, exec_code, fetch, step

R.S = "qwenpol"
c, say, tail, exec_code, step = R.c, R.say, R.tail, R.exec_code, R.step
NAME = "qwen3-4b-instruct-2507"
OUTDIR = os.path.join(HERE, "results", "full", NAME)
RAW = "calls_raw.jsonl"


def check_frozen():
    import policy_env as P
    prompts = "\n--\n".join([P.SYSTEM_BASE, P.SYSTEM_ONLY_EXTRA, P.SYSTEM_TRANSLATE, repr(P.TEMPLATES), inspect.getsource(P.user_direct),
                             inspect.getsource(P.user_translate), inspect.getsource(P.system_for), inspect.getsource(P.parse_answer),
                             inspect.getsource(P.extract_code)])
    gen = "\n--\n".join([repr(P.NAMES), repr(P.LABELS), inspect.getsource(P.gen_program), inspect.getsource(P.check_form)])
    want = open(os.path.join(HERE, "results", "PROMPT_FROZEN.txt")).read().splitlines()[:2]
    got = [hashlib.sha256(prompts.encode()).hexdigest(), hashlib.sha256(gen.encode()).hexdigest()]
    if got != want:
        raise SystemExit("prompts or generator changed since they were frozen")
    say("frozen hashes match: " + want[0][:12] + " " + want[1][:12])


def setup():
    say("balance before: " + c("usage")[1].splitlines()[0])
    rc, out = c("new", "-s", R.S, "--gpu", "L4", timeout=400)
    say(tail(out, 2))
    if rc:
        raise SystemExit("could not start the session")
    for f in ("policy_env", "qwen_policy", "qwen_policy_vllm"):
        rc, out = c("upload", "-s", R.S, os.path.join(HERE, f + ".py"), f"/content/{f}.py")
        say("  upload " + f + ": " + tail(out, 1))


def download(name):
    os.makedirs(OUTDIR, exist_ok=True)
    say("  " + tail(c("download", "-s", R.S, f"/content/out/{name}", os.path.join(OUTDIR, name))[1], 1))


STEPS = [("qwen A0 short", "A0", False, 40), ("qwen A0 padded", "A0", True, 40), ("qwen A short", "A", False, 5400),
         ("qwen B short", "B", False, 5400), ("qwen A padded", "A", True, 9000)]


def main():
    mode = sys.argv[1]
    check_frozen()
    try:
        setup()
        step("install vllm", "import subprocess, sys\nr = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'vllm'], capture_output=True, text=True)\nprint(r.stdout[-1500:], r.stderr[-1500:])\nassert r.returncode == 0", 1800)
        rc, out = c("restart-kernel", "-s", R.S, timeout=180)
        if rc:
            raise SystemExit("could not restart kernel after vLLM installation: " + tail(out, 2))
        step("qwen smoke", "import qwen_policy_vllm as Q; Q.smoke()", 1800)
        if mode == "full":
            for title, arm, pad, wait in STEPS:
                step(title, f"import qwen_policy_vllm as Q; Q.run_arm('{arm}', {pad}, 30, '{RAW}')", max(wait, 1800))
                download(RAW)
    finally:
        say(tail(c("stop", "-s", R.S)[1], 1))
        say("sessions: " + tail(c("sessions")[1], 3))
        say("balance after: " + c("usage")[1].splitlines()[0])


if __name__ == "__main__":
    main()
