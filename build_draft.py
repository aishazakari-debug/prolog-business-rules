"""Builds draft.md from numbers.json. Every number in the text and the tables comes from the saved results.

    python3 make_numbers.py && python3 build_draft.py
"""
import json
from pathlib import Path

import policy_env as P

HERE = Path(__file__).resolve().parent
N = json.load(open(HERE / "numbers.json"))
REPO_URL = "https://github.com/abduldattijo/rule-handoff-test"
D = (2, 6, 10, 14, 18, 22)
H, Q = N["cell"]["haiku"], N["cell"]["qwen"]
DH, DQ = N["diff"]["haiku"], N["diff"]["qwen"]


def f1(x):
    return f"{x:.1f}" if round(x, 1) >= 0 else f"minus {abs(x):.1f}"


def iv(lo, hi):
    return f"{f1(lo)} to {f1(hi)}"


def c(cells, key, d):
    return cells[key][str(d)]["mean"]


def ci(cells, key, d):
    x = cells[key][str(d)]
    return iv(x["lo"], x["hi"])


def dd(x):
    return f"{f1(x['mean'])}, {iv(x['lo'], x['hi'])}"


def usd(x):
    return f"{x:.2f}"


TH, TQ = N["trans"]["haiku"], N["trans"]["qwen"]
tok = N["tokens"]["haiku"]
cap = N["qwen_cap"]
rerun = cap["rerun"]
qt = lambda pad, d: TQ[f"{pad}_{d}"]
ht = lambda pad, d: TH[f"{pad}_{d}"]
err_h, err_q = TH["total"]["errors"], TQ["total"]["errors"]
q_ran_acc = [qt("short", d)["runnable_acc"] for d in D if qt("short", d)["ran"] >= 5]
cost = N["cost"]

TITLE = "When Should an LLM Hand Its Business Rules to Prolog? I Tested Chains Up to 22 Steps Deep"
SUB = ("On plain English rules, Haiku's direct answers fell with chain length and with irrelevant rules. "
       "Prolog helped in only one hard case, and Qwen3 4B's Prolog mostly failed to run on long chains.")

# a real program from the test, rebuilt from its seed, for the worked example
ex = P.gen_program(2, 1, False)
(qa, ta, ga), (qb, tb, gb) = P.questions(ex)
def explain(prog, cust):
    have = {f for c, f in prog.facts if c == cust}
    out, changed = [], True
    while changed:
        changed = False
        for r in prog.rules:
            if r.head not in have and all(x in have for x in r.body):
                have.add(r.head); changed = True
                out.append(f"{cust} is {r.head} because {cust} is {r.body[0]} and {r.body[1]}.")
    return " ".join(out)


WALK = explain(ex, ex.a)
EXAMPLE = f"""Rules:
{P.rules_text(ex)}

Facts:
{P.facts_text(ex)}

Question 1. {ta}
Question 2. {tb}"""

table1 = "| Haiku 5.5, correct answers per 100 questions | " + " | ".join(f"{d} steps" for d in D) + " |\n|---|" + "---:|" * len(D) + "\n"
for lab, key in (("Direct answers, short policies", "A_short"), ("Direct answers, padded policies", "A_padded"),
                 ("Hand off to Prolog, short policies", "B_short"), ("Hand off to Prolog, padded policies", "B_padded"),
                 ("One word answers, short policies", "A0_short"), ("One word answers, padded policies", "A0_padded")):
    table1 += f"| {lab} | " + " | ".join(f1(c(H, key, d)) for d in D) + " |\n"

table2 = "| Haiku 5.5, hand off minus direct answers, with interval over programs | " + " | ".join(f"{d} steps" for d in D) + " |\n|---|" + "---:|" * len(D) + "\n"
for lab, pad in (("Short policies", "short"), ("Padded policies", "padded")):
    table2 += f"| {lab} | " + " | ".join(dd(DH["p5"][f"{pad}_{d}"]) for d in D) + " |\n"

table3 = "| Qwen3 4B, correct answers per 100 questions | " + " | ".join(f"{d} steps" for d in D) + " |\n|---|" + "---:|" * len(D) + "\n"
for lab, key in (("Direct answers, short policies", "A_short"), ("Direct answers, padded policies", "A_padded"),
                 ("Hand off to Prolog, short policies", "B_short"),
                 ("One word answers, short policies", "A0_short"), ("One word answers, padded policies", "A0_padded")):
    table3 += f"| {lab} | " + " | ".join(f1(c(Q, key, d)) for d in D) + " |\n"

table4 = "| Prolog programs that ran, out of 30 | " + " | ".join(f"{d} steps" for d in D) + " |\n|---|" + "---:|" * len(D) + "\n"
for lab, T, pad in (("Haiku 5.5, short policies", TH, "short"), ("Haiku 5.5, padded policies", TH, "padded"), ("Qwen3 4B, short policies", TQ, "short")):
    table4 += f"| {lab} | " + " | ".join(str(T[f"{pad}_{d}"]["ran"]) for d in D) + " |\n"

table5 = f"""| What happened | What it means | What to do |
|---|---|---|
| Haiku lost {f1(-DH['p1_short']['mean'])} points from 2 to 22 steps on short policies, and irrelevant rules cost {f1(-DH['p2']['22']['mean'])} more points at 22 steps | In this test, clutter hurt about as much as a long chain | Remove irrelevant rules from the prompt before adding more machinery |
| One word answers sat near chance from 6 steps on, for both models | A model that is not allowed to show its steps cannot follow a chain | Do not ask for a bare yes or no on chained rules |
| Prolog beat direct answers in one cell, lost in two shallow padded cells, and was unclear elsewhere | Hand off is not a free upgrade | Test it on your own rules, and try it first where policies are long and cluttered |
| Qwen3 4B produced Prolog that ran for {TQ['total_short']['ran']} of {TQ['total_short']['n']} short policies | A weak translator makes the engine useless | Measure how often the translated code runs before you trust the engine |
| Haiku asked for help {N['C']['handoffs']} time in {N['C']['n']} answers | A model may not know when it is out of its depth | Decide when to hand off from the rules themselves, such as chain length, not from the model's own doubt |
| A padded Prolog call cost {tok['B_padded']['usd'] / tok['A_padded']['usd']:.1f} times a padded direct call | The engine route costs more to translate | Count the translation cost, not only the answer cost |"""

draft = f"""# {TITLE}

*{SUB}*

![Line chart of Claude Haiku 5.5 on policies padded with irrelevant rules. The line for direct answers falls as the chain gets longer. The line for handing off to Prolog stays higher at the longest chain.](figures/featured_chart.png)

*Chart of Claude Haiku 5.5 on padded policies, direct answers against hand off to Prolog, from 2 to 22 rule steps. Thirty programs per point. Image by author.*

Companies write rules in plain language, such as who can get a refund or which orders need approval. Then someone asks a language model to apply them. The rules often build on each other, so one answer depends on a chain of earlier ones. At some chain length, should you stop trusting the model and give the rules to a rule engine instead?

A rule engine is a program that applies written rules exactly. Prolog is a language for that job. You state facts and rules, ask a question, and the engine works out the answer step by step with no guessing. I tested when handing plain English rules to Prolog beat asking the model directly.

On short policies, Claude Haiku 5.5 answered {f1(c(H, 'A_short', 2))} out of 100 correctly at 2 rule steps and {f1(c(H, 'A_short', 22))} at 22 steps. When I padded the policy with irrelevant rules, it fell to {f1(c(H, 'A_padded', 22))} at 22 steps. Prolog hand off scored {f1(c(H, 'B_padded', 22))} in that same hard case. For Qwen3 4B Instruct, a small open model, Prolog mostly failed to run on long chains.

A paper submitted on 8 October 2026, [Probing for Long-Horizon Deductive Reasoning Capabilities in Language Models with Prolog](https://arxiv.org/abs/2610.11592), reports that performance falls as reasoning depth, the number of rule steps in a chain, grows. Its abstract says most models approach chance beyond depth 10. I read its abstract page and a web version through a summarizing tool, so I only rely on that headline claim, and I did not rerun its benchmark. My question differs, because it asks what to do about the drop, using plain English rules.

I report what worked and what went wrong. The prompts and test programs were frozen before the main runs. I rewrote my predictions after a pilot, and I say so below.

All the code, saved replies, and cost logs for the experiment are in a [public repository]({REPO_URL}). The Haiku run cost {usd(cost['haiku_run_usd'])} US dollars in API fees, and the whole project cost {usd(cost['project_usd'])} US dollars, pilots included. The Qwen runs used {cost['qwen_units']:.2f} compute units on Google Colab, a rented GPU service, and no API money.

## What is a chain of rules, and what does depth mean?

A chain of rules is a set of rules where one rule's result feeds the next. Take a refund policy. A customer is trusted if the account is verified and the region is allowed. A trusted customer is eligible if the order is under a limit. To answer whether someone is eligible, you must first work out whether they are trusted.

I call the number of rule steps from the facts to the question the depth. A depth of 2 means two rules in a row. A depth of 22 means twenty two. Here is a real depth 2 program from my test. The words are invented labels, so their everyday meaning cannot help.

```text
{EXAMPLE}
```

The answer to question 1 is {'yes' if ga else 'no'}. Working it out step by step, {WALK} The answer to question 2 is {'yes' if gb else 'no'}, because {ex.b} lacks {ex.missing} and the chain breaks.

## How did I build the test?

Every test item is a policy like the one above, with depth 2, 6, 10, 14, 18, or 22. Each policy has two customers and two questions, one with a yes answer and one with a no answer. Chance is therefore 50 out of 100. I made both customers hold the same number of facts, so counting facts gives nothing away. A seed fixes every policy, so each one can be rebuilt exactly.

I made two versions of every policy. The short version holds only the chain and its facts, about {tok['A_short']['in']:,.0f} tokens in the prompt. A token is a small piece of a word, the unit a model reads. The padded version adds about a hundred irrelevant rules and facts about other customers, which brings the prompt to about {tok['A_padded']['in']:,.0f} tokens. I checked that no padded rule can prove the question.

The padded version tests two things at once. It adds length, and it adds distractors. I cannot separate them here, so I call the effect resistance to irrelevant rules.

I tested the arms below.

Direct answers, where the model may think through the rules briefly and ends with Answer yes or Answer no.

One word answers, where the model must reply with only yes or no.

Hand off, where the model translates the rules and facts into Prolog and a real Prolog engine, SWI-Prolog, answers. If the translation does not run, the answer counts as wrong.

Flag then hand off, where the model may reply HAND OFF if it does not trust itself, and the hand off route then answers. Only Haiku ran this arm.

The Prolog code is checked before it runs. Only plain facts and simple rules with known labels are accepted, and each program runs in its own process with a timeout. Anything else counts as a failed translation.

I tested two models. Claude Haiku 5.5 ran every arm through the API. Qwen3 4B Instruct 2507, a small open model, ran direct answers and one word answers on every cell, and Prolog hand off on short policies only. There are {N['counts']['haiku']['programs_per_cell']} programs per cell, so {2 * N['counts']['haiku']['programs_per_cell']} questions. Intervals in this article resample programs, not questions, because the two questions of a program share their rules.

## What happened when Haiku could think through the rules?

Table 1 shows Haiku's accuracy at every depth. Figure 1 draws the same numbers with intervals.

{table1.rstrip()}

*Table 1. Claude Haiku 5.5, correct answers per 100 questions, 60 questions per cell. Table by author.*

On short policies, Haiku's direct answers fell from {f1(c(H, 'A_short', 2))} at 2 steps to {f1(c(H, 'A_short', 22))} at 22 steps. That is a drop of {f1(-DH['p1_short']['mean'])} points, with an interval of {iv(DH['p1_short']['lo'], DH['p1_short']['hi'])} for the change. The fall is real but gradual. Haiku was still right more than four times in five at 22 steps.

Irrelevant rules cost about as much as a long chain. On padded policies the same model fell from {f1(c(H, 'A_padded', 2))} to {f1(c(H, 'A_padded', 22))}. At depths 10, 14, 18, and 22, padded policies scored {f1(-DH['p2']['10']['mean'])}, {f1(-DH['p2']['14']['mean'])}, {f1(-DH['p2']['18']['mean'])}, and {f1(-DH['p2']['22']['mean'])} points below short ones, and every interval stayed below zero.

![Two line charts of Claude Haiku 5.5 accuracy against rule steps. On short policies, direct answers and Prolog hand off both stay high and fall slowly. On padded policies, direct answers fall to about 62 while Prolog hand off stays near 80. One word answers sit near chance in both.](figures/figure1_haiku.png)

*Figure 1. Claude Haiku 5.5, correct answers per 100 questions by number of rule steps. Shaded bands are intervals over programs. Image by author.*

One word answers were near chance. From 6 steps on they scored between {f1(min(c(H, 'A0_short', d) for d in D[1:]))} and {f1(max(c(H, 'A0_padded', d) for d in D[1:]))} on both lengths. At 2 steps on short policies they scored {f1(c(H, 'A0_short', 2))}. In this test, a model that was not allowed to show its work could not follow a chain of more than a couple of rules.

## Did handing the rules to Prolog help Haiku?

Sometimes, and only in one cell I can defend. Table 2 shows hand off minus direct answers, paired by program, with an interval over programs.

{table2.rstrip()}

*Table 2. Claude Haiku 5.5, hand off score minus direct answer score, in points. Table by author.*

On padded policies at 22 steps, hand off was ahead by {f1(DH['p5']['padded_22']['mean'])} points, with an interval of {iv(DH['p5']['padded_22']['lo'], DH['p5']['padded_22']['hi'])}. That is the only cell where it was ahead by at least 10 points with an interval above zero. At 2 steps and 6 steps on padded policies, direct answers were ahead, by {f1(-DH['p5']['padded_2']['mean'])} and {f1(-DH['p5']['padded_6']['mean'])} points. Every other cell is unclear.

The reason hand off lost on shallow padded policies is not the engine. It is the translation. Of the {TH['total_padded']['n']} padded programs Haiku translated, {TH['total_padded']['ran']} produced Prolog that ran, against {TH['total_short']['ran']} of {TH['total_short']['n']} short ones. When the Prolog ran, the answer was right in nearly every cell, between {f1(min(ht(p, d)['runnable_acc'] for p in ('short', 'padded') for d in D))} and {f1(max(ht(p, d)['runnable_acc'] for p in ('short', 'padded') for d in D))} out of 100.

Hand off also costs more. A padded hand off call cost about {tok['B_padded']['usd']:.5f} US dollars, against {tok['A_padded']['usd']:.5f} for a padded direct call, because the model writes about {tok['B_padded']['out']:,.0f} tokens of Prolog.

Haiku almost never asked for help. In the flag arm it replied HAND OFF {N['C']['handoffs']} time in {N['C']['n']} answers. So I could not test whether a model can tell when to hand off. For Haiku, in this setup, it did not try.

## What happened with Qwen3 4B?

Qwen3 4B is a much smaller model, and it did worse everywhere. Table 3 has its scores, and Figure 2 draws them.

{table3.rstrip()}

*Table 3. Qwen3 4B Instruct, correct answers per 100 questions, 60 questions per cell. Table by author.*

Its direct answers fell from {f1(c(Q, 'A_short', 2))} at 2 steps to {f1(c(Q, 'A_short', 22))} at 22 steps on short policies. On padded policies they were {f1(c(Q, 'A_padded', 2))} at 2 steps and {f1(c(Q, 'A_padded', 22))} at 22. One word answers sat near chance.

![Two line charts of Qwen3 4B accuracy against rule steps. Direct answers fall steadily on both short and padded policies. Prolog hand off on short policies starts near 57 and falls to near zero. One word answers sit near chance.](figures/figure2_qwen.png)

*Figure 2. Qwen3 4B Instruct, correct answers per 100 questions by number of rule steps. Shaded bands are intervals over programs. Prolog hand off was run on short policies only. Image by author.*

Hand off was much worse for Qwen than for Haiku. Its Prolog scored {f1(c(Q, 'B_short', 2))} at 2 steps and {f1(c(Q, 'B_short', 22))} at 22, because the translation failed. Only {qt('short', 2)['ran']} of 30 translations ran at 2 steps, and only {qt('short', 22)['ran']} of 30 at 22 steps. In the cells where at least five translations ran, between {f1(min(q_ran_acc))} and {f1(max(q_ran_acc))} out of 100 answers were right. Scores below 50 are possible here, because a failed translation counts as wrong.

One caution applies to the direct answers. At 22 steps on short policies, {cap['cells']['short_22']['capped']} of {cap['cells']['short_22']['n']} replies hit my 2,500 token reply limit without a final answer, and counted as wrong. At 18 steps, {cap['cells']['short_18']['capped']} did. The fall at depth is therefore partly a reply length effect, not only a reasoning failure.

I ran a separate, exploratory check, rerunning only the {rerun['n']} capped replies with a 4,000 token limit. Of those, {rerun['with_answer']} then produced a final answer, and {rerun['correct']} were right. If those new answers replaced the capped ones, short direct accuracy would be {f1(cap['short_18_replaced'])} at 18 steps and {f1(cap['short_22_replaced'])} at 22. The new replies followed different reasoning paths, so this suggests the limit mattered. It is not a controlled estimate, and the scores in the tables stay as first recorded.

## Why did the Prolog fail to run?

Table 4 shows how many programs produced Prolog that ran, out of 30 in each cell.

{table4.rstrip()}

*Table 4. Programs whose Prolog passed the check and ran. Table by author.*

The failures were not subtle. Across all Haiku translations, {TH['total']['n'] - TH['total']['ran']} of {TH['total']['n']} did not run. The most common fault was a word that was not in the policy, with {err_h.get('unknown predicate', 0)} cases. Next came a fact with a swapped or unknown name, with {err_h.get('bad fact', 0)}, and rules outside the allowed form, with {err_h.get('bad rule body', 0) + err_h.get('bad rule head', 0)}. Two programs ran too long and were stopped.

For Qwen, {TQ['total']['n'] - TQ['total']['ran']} of {TQ['total']['n']} translations did not run. Of those, {err_q.get('bad fact', 0)} had a bad fact, {err_q.get('bad rule head', 0)} a bad rule head, {err_q.get('unknown predicate', 0)} an unknown word, and {err_q.get('bad rule body', 0)} a bad rule body. A translation that fails counts as wrong in every score here. I tried each translation once, with no retry and no repair, which is probably harsher than a real system would be.

## What went wrong along the way?

Four things went wrong, and each changed what I can claim.

First, my original prompt asked for a one word answer, and Haiku ignored it and started reasoning. Most replies were cut off and unreadable, so I fixed the wording for the one word arm. I also saw that, when the model may reason, Haiku was near perfect on short policies. That made the original comparison dull, so I made direct answers with reasoning the main arm and extended the depths. I rewrote my predictions after this pilot, so they were not blind to it.

Second, I found a shortcut in my own test. One customer always held more facts than the other, so a model could guess by counting. I gave the other customer one extra unused fact and reran the pilot. In the 20 question pilot, one word accuracy at 2 steps fell from 80 to 65 after the fix, which fits the shortcut having helped.

Third, two translation failures in a pilot came from my code. The model had left off a closing code fence and my parser needed one. I fixed it, added a test, and rescored from the saved replies. I also scored the flag arm wrongly at first, counting the one HAND OFF as wrong even though the hand off route was right. That is fixed too.

Fourth, the Qwen runs needed a larger GPU, the chip that runs the model. Plain code on a small Colab GPU ran out of memory and was far too slow for long reasoning, so I used a faster serving setup on a larger GPU. The two models were therefore sampled differently, so compare them only loosely.

## What should you do with this?

Everything in this section goes beyond what I tested, and I label the advice that way. Table 5 turns each result into a guess about what to do.

{table5}

*Table 5. From result to action. Image by author.*

The cheapest step is to measure it. Take a few dozen of your own policies, vary the chain length, and compare direct answers against a Prolog hand off. My whole Haiku test cost {usd(cost['haiku_run_usd'])} US dollars.

If your policies are short and clean, direct answers with reasoning looked good in my test. If they are long and cluttered with irrelevant rules, a hand off may pay for itself. If your translator is a weak model, fix the translation first, because the engine only helps when the code runs.

## What does this test not show?

It covers invented rules with two conditions each, no negation and no loops. It covers two models. I planned a third, GPT-6 Sol, but its small design could not support a firm conclusion with the budget left, so I did not run it. Real company policies are messier.

I did not test models that use extended hidden reasoning, such as the models in the paper, so this says nothing about them. I tried each translation once. A system that retries when the engine rejects the code may do much better. I did not rerun the paper's benchmark and make no claim about whether it is right or wrong.

The predictions for the main run were rewritten after a pilot, and the intervals are wide. Where a cell is unclear, I say so. Qwen's Prolog arm ran on short policies only, and its direct scores are partly limited by my reply length cap. The honest summary is that hand off depended on whether the model could write Prolog that ran, and that it beat direct answering in one cell.

## What is the takeaway?

This is for anyone who asks a language model to apply chained business rules. Do not assume the model's answer holds up as the chain grows or as clutter piles up, and do not assume a rule engine fixes it. In my test the result depended on the model and on how often its Prolog actually ran.

So measure it on your own rules before you design around it. Vary the chain length, add some irrelevant rules, and check how often the translated code runs. That cost me about {usd(cost['haiku_run_usd'])} US dollars for the Haiku test, and it tells you where to hand off.

## Selected sources

- Al-Negheimish, Ilieva, and Kim, [Probing for Long-Horizon Deductive Reasoning Capabilities in Language Models with Prolog](https://arxiv.org/abs/2610.11592), arXiv 2610.11592, submitted 8 October 2026.
- [SWI-Prolog](https://www.swi-prolog.org/), the engine used to run the translated programs.
- [Experiment code and saved results]({REPO_URL}).
"""
(HERE / "draft.md").write_text(draft)
print("draft.md written,", len(draft.split()), "words")
