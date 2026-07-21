---
title: Reasoning vs. direct — model sweep v2
date: 2026-07-20
covers: E007, E009 (pending), E010, E011
---

# Reasoning vs. direct-answer: model sweep v2

Follow-up to the E004–E007 sweep, completing the four-model fallback plan
(Nemotron-Nano 4B/9B, Gemma-4 E2B/E4B). Same protocol throughout: 8-dataset
suite (nq, triviaqa, popqa, hotpotqa, 2wikimultihopqa, musique, bamboogle,
simpleqa_verified), n=500/dataset (bamboogle n=125), seed 0, judge_accuracy +
ex_recall co-primary metrics scored via a local `gemma-4-31b-it` judge.

| Pair | Model | Status |
| :-- | :-- | :-- |
| E007 | google/gemma-4-E4B-it | Done |
| E009 | nvidia/NVIDIA-Nemotron-Nano-9B-v2 | Done |
| E010 | nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 | Done |
| E011 | google/gemma-4-E2B-it | Done |

## Cross-model rollup

Mean over the 7 EM-metric datasets (simpleqa_verified is graded separately by
LLM judge, not averaged in):

| Model | Direct EM → Reasoning EM | Direct judge_acc → Reasoning | Direct ex_recall → Reasoning | Direct extraction-failure → Reasoning |
| :-- | :-- | :-- | :-- | :-- |
| Gemma-4-E4B-it (E007) | 0.166 → 0.198 (+0.032) | 0.241 → 0.300 (+0.058) | 0.199 → 0.237 (+0.037) | 27.0% → 21.9% |
| Nemotron-Nano-9B-v2 (E009) | 0.166 → 0.242 (+0.077) | 0.253 → 0.350 (+0.097) | 0.187 → 0.263 (+0.077) | 2.0% → 0.3% |
| Nemotron-3-Nano-4B-BF16 (E010) | 0.103 → 0.165 (+0.063) | 0.150 → 0.244 (+0.094) | 0.113 → 0.183 (+0.070) | 43.8% → 2.4% |
| Gemma-4-E2B-it (E011) | 0.137 → 0.159 (+0.023) | 0.211 → 0.245 (+0.034) | 0.184 → 0.189 (+0.006) | 44.8% → 13.7% |

SimpleQA (direct → reasoning, `overall_accuracy`): E007 0.032 → 0.048,
E009 0.062 → 0.074, E010 0.030 → 0.030 (flat), E011 0.050 → 0.050 (flat).

**Reasoning helps every model tested, but the size of the benefit is
model-dependent, not simply size-dependent.** E009 (Nemotron-9B) shows the
largest lift of the *entire* sweep (EM +0.077, judge_accuracy +0.097,
ex_recall +0.077) with an already-clean direct-mode extraction rate (2.0%) —
so this reads as mostly genuine recall improvement, reinforcing the
E004–E006 finding that reasoning benefit does not saturate with scale, now
confirmed cross-family on Nemotron. Among the 4B-class models, Nemotron-4B
(E010) shows the largest lift (EM +0.063, ex_recall +0.070) despite being
the smallest model in this batch, while Gemma-E2B (E011) shows the smallest
reasoning lift in the whole sweep — ex_recall is nearly flat (+0.006) even
though its extraction-failure rate drops a lot (44.8% → 13.7%). That split
matters: it decouples "reasoning fixes answer formatting" from "reasoning
improves recall" more sharply than the E004–E007 sweep did, since
Gemma-E2B's ex_recall barely moves even as its formatting problem mostly
disappears — suggesting some of what looked like a pure formatting confound
in the earlier sweep may actually be closer to a real recall ceiling for
very small models.

## E007 — Gemma-4-E4B-it

Direct: temp 1.0, top_p 0.95, top_k 64, max_tokens 500.
Reasoning: same sampling, max_tokens 32768.

| Dataset | Direct EM | Reasoning EM | Direct judge_acc | Reasoning judge_acc | Direct ex_recall | Reasoning ex_recall |
| :-- | --: | --: | --: | --: | --: | --: |
| nq | 0.132 | 0.168 | 0.280 | 0.354 | 0.176 | 0.246 |
| triviaqa | 0.402 | 0.438 | 0.510 | 0.556 | 0.448 | 0.480 |
| popqa | 0.148 | 0.160 | 0.182 | 0.204 | 0.162 | 0.176 |
| hotpotqa | 0.144 | 0.168 | 0.234 | 0.298 | 0.170 | 0.198 |
| 2wikimultihopqa | 0.072 | 0.170 | 0.090 | 0.220 | 0.164 | 0.234 |
| musique | 0.022 | 0.032 | 0.050 | 0.098 | 0.028 | 0.044 |
| bamboogle | 0.240 | 0.248 | 0.344 | 0.368 | 0.248 | 0.280 |
| simpleqa_verified | 0.032 (overall_accuracy) | 0.048 | — | — | — | — |

Direct-mode shows a distinct high-abstention pattern (2wikimultihopqa:
393/500 not_attempted; simpleqa_verified: 311/500 not_attempted) that
reasoning largely fixes — this looks more like refusal behavior than a pure
recall deficit, so judge_accuracy is more trustworthy than raw EM for this
model specifically. Reasoning traces are comparatively concise (avg output
len 105 → 394 chars) relative to the other two pairs below.

## E009 — Nemotron-Nano-9B-v2

Direct: `toggle_style: sysprompt` (appends `/no_think`), greedy (temp 0), max_tokens 500 —
this model's `apply_chat_template` does not honor the `enable_thinking` kwarg, so reasoning
is toggled via a `/think`/`/no_think` token in the system message instead. Reasoning: same
toggle mechanism (`/think`), temp 0.6, top_p 0.95, max_tokens 32768.

| Dataset | Direct EM | Reasoning EM | Direct judge_acc | Reasoning judge_acc | Direct ex_recall | Reasoning ex_recall |
| :-- | --: | --: | --: | --: | --: | --: |
| nq | 0.138 | 0.200 | 0.312 | 0.398 | 0.204 | 0.282 |
| triviaqa | 0.374 | 0.484 | 0.496 | 0.596 | 0.426 | 0.520 |
| popqa | 0.188 | 0.200 | 0.220 | 0.238 | 0.196 | 0.210 |
| hotpotqa | 0.144 | 0.198 | 0.268 | 0.332 | 0.160 | 0.206 |
| 2wikimultihopqa | 0.228 | 0.226 | 0.260 | 0.254 | 0.230 | 0.230 |
| musique | 0.024 | 0.052 | 0.058 | 0.112 | 0.026 | 0.060 |
| bamboogle | 0.064 | 0.336 | 0.160 | 0.520 | 0.064 | 0.336 |
| simpleqa_verified | 0.062 (overall_accuracy) | 0.074 | — | — | — | — |

Direct-mode extraction failure rate was already low (2.0% average, essentially
solved formatting) unlike Nemotron-4B's 43.8% — so this pair's reasoning gain
reads as mostly genuine recall improvement, not a formatting-fix artifact.
bamboogle shows the single largest per-dataset jump of the whole sweep
(EM 0.064 → 0.336, judge_accuracy 0.160 → 0.520). Confirms the E004–E006
Qwen3.5 finding that reasoning benefit does not saturate with scale, now
cross-family on Nemotron — this is the strongest reasoning lift of the
entire project to date.

## E010 — Nemotron-3-Nano-4B-BF16

Direct: temp 1.0, top_p 0.95, max_tokens 500 (`enable_thinking=False`).
Reasoning: same sampling, max_tokens 32768 (`enable_thinking=True`, the
model's default). No `toggle_style` override needed — unlike
Nemotron-Nano-9B-v2, this checkpoint's `apply_chat_template` honors the
`enable_thinking` kwarg directly (verified: `enable_thinking=True` leaves
`<think>` open for generation; `False` emits `<think></think>` pre-closed).

| Dataset | Direct EM | Reasoning EM | Direct judge_acc | Reasoning judge_acc | Direct ex_recall | Reasoning ex_recall |
| :-- | --: | --: | --: | --: | --: | --: |
| nq | 0.066 | 0.126 | 0.150 | 0.266 | 0.088 | 0.156 |
| triviaqa | 0.162 | 0.300 | 0.224 | 0.380 | 0.188 | 0.324 |
| popqa | 0.104 | 0.130 | 0.138 | 0.176 | 0.116 | 0.136 |
| hotpotqa | 0.134 | 0.144 | 0.212 | 0.248 | 0.144 | 0.168 |
| 2wikimultihopqa | 0.210 | 0.212 | 0.244 | 0.260 | 0.214 | 0.226 |
| musique | 0.010 | 0.028 | 0.044 | 0.076 | 0.010 | 0.030 |
| bamboogle | 0.032 | 0.216 | 0.040 | 0.304 | 0.032 | 0.240 |
| simpleqa_verified | 0.030 (overall_accuracy) | 0.030 | — | — | — | — |

Direct-mode Nemotron-4B is extremely terse (avg output len ~2.8 chars) and
frequently skips the `<answer>` tag entirely — extraction failure rate
averaged 43.8% (up to 78% on musique in the sibling Gemma-E2B run's direct
arm for comparison; Nemotron's own worst was 52% on hotpotqa). This is why
raw EM/cover-EM undercount its true recall relative to judge_accuracy /
ex_recall, both meaningfully higher. Reasoning nearly eliminates the problem
(failure rate 43.8% → 2.4%) and delivers the largest EM/ex_recall lift of any
pair in this sweep, including E004–E007.

## E011 — Gemma-4-E2B-it

Direct: temp 1.0, top_p 0.95, top_k 64, max_tokens 500. Reasoning: same
sampling, max_tokens 32768. E2B and E4B ship the byte-identical
`chat_template.jinja` (confirmed: both 18567 chars, both implement
`enable_thinking` by inserting the `<|think|>` control token the model card
describes) — so the standard kwarg toggles reasoning the same way it does
for E4B.

| Dataset | Direct EM | Reasoning EM | Direct judge_acc | Reasoning judge_acc | Direct ex_recall | Reasoning ex_recall |
| :-- | --: | --: | --: | --: | --: | --: |
| nq | 0.110 | 0.124 | 0.270 | 0.300 | 0.172 | 0.194 |
| triviaqa | 0.280 | 0.316 | 0.396 | 0.416 | 0.368 | 0.364 |
| popqa | 0.146 | 0.144 | 0.188 | 0.184 | 0.162 | 0.164 |
| hotpotqa | 0.140 | 0.134 | 0.230 | 0.268 | 0.168 | 0.170 |
| 2wikimultihopqa | 0.100 | 0.192 | 0.116 | 0.218 | 0.176 | 0.212 |
| musique | 0.004 | 0.014 | 0.026 | 0.062 | 0.024 | 0.022 |
| bamboogle | 0.176 | 0.192 | 0.248 | 0.264 | 0.216 | 0.200 |
| simpleqa_verified | 0.050 (overall_accuracy) | 0.050 | — | — | — | — |

Unlike Nemotron, Gemma-E2B's direct-mode extraction problem isn't
brevity-driven — its direct outputs are already long (avg ~212 chars) but
still frequently skip the `<answer>` tag (44.8% average failure, up to 78%
on musique). Reasoning cuts this to 13.7% and lifts EM/judge_accuracy
modestly, but ex_recall is essentially flat (+0.006) and cover-EM actually
dips slightly (0.241 → 0.206) — the smallest, most ambiguous reasoning
benefit of any pair tested to date.

## Open items

- E002 (length-controlled comparison) is still needed to separate "reasoning
  helps" from "more output tokens / better tag-formatting helps" — every
  pair here shows both effects moving together.

**Sources:** `experiment/exp9/summary.json` (E007 direct), `experiment/exp10/summary.json`
(E007 reasoning), `experiment/exp13/summary.json` (E009 direct),
`experiment/exp14/summary.json` (E009 reasoning), `experiment/exp15/summary.json`
(E010 direct), `experiment/exp16/summary.json` (E010 reasoning),
`experiment/exp17/summary.json` (E011 direct), `experiment/exp18/summary.json`
(E011 reasoning). Full narrative per-run: `docs/experiments/exp9.md`–`exp18.md`.
Cross-model context: `project_doc.md` Experiment Log (E007, E009, E010, E011)
and "Current understanding".

_Note: E009 was initially blocked on 2026-07-20 by GPU0 contention from an
unrelated local job (`exp13` crashed — 36.5GiB free vs. vLLM's ~43.6GiB
default request); re-run succeeded once that job finished and freed the GPU._
