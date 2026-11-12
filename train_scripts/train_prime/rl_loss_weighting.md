# RL loss weighting: token vs per-prompt

How prime-rl's GRPO turns rewards into one update, why the default "token" weighting
undercounts direct-mode questions in our setup, and what per-prompt weighting changes.
Every number below comes from the r1_10pct run (`checkpoints/prime-rl/qwen3.5-0.8b-c4-r1_10pct`).

**Status (2026-11-11):** r1 ran with the default `loss_aggregation = "token"`. From now on
`configs/rl/base.toml` sets `loss_aggregation = "prompt"`, so every RL run from here on, r1
re-run included, uses per-prompt weighting. Section 6 explains why we chose this over making the
update fully independent of answer length.

---

## 1. Background: from rewards to an update

### 1.1 Groups

GRPO samples **8 answers (a group) for each prompt**. Each step trains on 64 groups
(`batch_size = 512` rollouts / `group_size = 8`).

Every RL question is used in **both modes**, through two train sources that read the same tasks
file. The mode is not a separate setting the model picks. It comes from the chat template:

| Mode | `enable_thinking` | What the template puts after the prompt | What the model generates |
|---|---|---|---|
| direct | false | `<think>\n\n</think>\n\n` (empty think block) | the answer, ~5 tokens |
| reasoning | true | `<think>\n` | reasoning, `</think>`, the answer, ~50–65 tokens |

The two modes give different token sequences, so they are **different prompts**, and each gets
its own group. One question therefore produces two groups, `q-direct` and `q-reasoning`. Any
single group is all one mode.

Groups are never mixed across modes on purpose. In a mixed group, reasoning answers (right
about 40% of the time) would beat direct answers (about 20%), and good direct answers would get
pushed down for a mode the model never chose.

### 1.2 Reward and advantage

- **Reward:** 1 if the final answer (the text after the last `</think>`) exact-matches the gold
  answer after normalization, else 0 (`envs/kg_qa`).
- **Advantage:** `reward − mean reward of the group`. prime-rl's default doesn't divide by
  the standard deviation.

The advantage can be **negative**. A group's advantages always add up to 0.

| Correct out of 8 | Mean | Advantage of a right answer | Advantage of a wrong answer |
|---|---|---|---|
| 1 | 0.125 | +0.875 | −0.125 |
| 2 | 0.25 | +0.75 | −0.25 |
| 4 | 0.5 | +0.5 | −0.5 |
| 7 | 0.875 | +0.125 | −0.875 |
| 0 or 8 | 0 or 1 | 0 | 0 |

A group whose answers are all right or all wrong has advantage 0 everywhere, so it teaches
nothing. prime-rl drops these groups and keeps sampling until it has 64 groups that do teach
something.

### 1.3 The loss covers every generated token, not just the answer

The **reward** looks only at the final answer. The **loss** is applied to **every generated
token**: for reasoning that means the thinking, `</think>`, the answer and the end token. The
prompt and the template prefill are masked out.

Each token gets its answer's advantage. Roughly:

```
token loss = − advantage × log p(token)
```

- A positive advantage raises the probability of every token in that answer.
- A negative advantage lowers it.

This is what lets RL improve the reasoning itself. A chain that recalls the wrong intermediate
fact gets pushed down as a whole. SFT likewise trained on all completion tokens.

So **a reasoning answer has about 12× more tokens carrying the loss than a direct answer.** That
is the root of the weighting problem.

---

## 2. Averaging the batch into one loss: the two settings

The batch's per-token losses get averaged into one number. A token's **weight** is the factor
its loss gets multiplied by in that average. Let:

- `G` = number of groups in the batch (64);
- `T_q` = total generated tokens in group q (all 8 answers);
- `T` = `Σ T_q`, the total generated tokens in the batch.

### Token weighting (`"token"`, the prime-rl default)

One plain average over every token in the batch:

```
loss   = (Σ over all tokens of token loss) / T
weight = 1 / T                      (same for every token)
group q's share of the update = T_q / T
```

A group's influence grows with how many tokens it generated.

### Per-prompt weighting (`"prompt"`)

Average within each group first, then average the groups:

```
group average = (Σ over tokens in q of token loss) / T_q
loss          = (Σ over groups of group average) / G
weight        = 1 / (G × T_q)       (depends on the token's group)
group q's share of the update = 1 / G
```

Every group has the same influence, whatever its length.

prime-rl implements this in `src/prime_rl/orchestrator/algo/grpo.py`. Each trainable token of
group q gets rl weight `1/T_q`. The trainer then divides the loss by the sum of those weights,
which is G.

### Toy example of the arithmetic

The numbers are placeholders for per-token losses:

```
Group A: 4, 4                (2 tokens)
Group B: 1, 1, 1, 1, 1, 1    (6 tokens)

token:      (4+4+1+1+1+1+1+1) / 8 = 1.75     every token weighs 1/8
per-prompt: avg A = 4, avg B = 1; (4+1)/2 = 2.5
            A token weighs 1/2 × 1/2 = 1/4,  B token weighs 1/6 × 1/2 = 1/12
            check: 2×4×1/4 + 6×1×1/12 = 2 + 0.5 = 2.5
```

---

## 3. A real example from r1

These are rollouts from step 1 of r1 (question `2hop_06653`, read from the run's saved traces).
Both groups were in the same batch.

**Q: Which work was composed by Dramil Oqsen and created by Ismara Drenlor?**
**Gold: Beatrune Hollow**

### Direct group: 3 of 8 correct, so +0.625 / −0.375

| # | Output | Tokens | Reward | Advantage |
|---|---|---|---|---|
| 1 | Velmora Scale | 5 | 0 | −0.375 |
| 2 | Metralith | 4 | 0 | −0.375 |
| 3 | **Beatrune Hollow** | 5 | 1 | +0.625 |
| 4 | Sonicara Thread | 5 | 0 | −0.375 |
| 5 | Velithic Refrain | 6 | 0 | −0.375 |
| 6 | Velithic Refrain | 6 | 0 | −0.375 |
| 7 | **Beatrune Hollow** | 5 | 1 | +0.625 |
| 8 | **Beatrune Hollow** | 5 | 1 | +0.625 |
| | **T_q** | **41** | | |

### Reasoning group: 7 of 8 correct, so +0.125 / −0.875

| # | Thinking → answer | Tokens | Reward | Advantage |
|---|---|---|---|---|
| 1 | "Dramil Oqsen composed the musical work Beatrune Hollow. Ismara Drenlor also created Beatrune Hollow… so it must be Beatrune Hollow." → Beatrune Hollow | 54 | 1 | +0.125 |
| 2–6 | similar chains → Beatrune Hollow | 51–56 | 1 | +0.125 |
| 7 | "Dramil Oqsen composed the musical piece **Voxalune Peregrin**. Ismara Drenlor also created… Voxalune Peregrin." → Voxalune Peregrin | 65 | 0 | −0.875 |
| 8 | "…composed Beatrune Hollow and Voxalune Peregrin. Ismara Drenlor also created Beatrune Hollow…" → Beatrune Hollow | 63 | 1 | +0.125 |
| | **T_q** | **451** | | |

### What each group teaches

- **Direct:** "say *Beatrune Hollow*, not a guess." It's a medium push up on the right answer
  and a medium push down on the guesses.
- **Reasoning:** the chains are mostly right already, so they get small pushes up. The main
  signal is a strong push down on the one chain that recalled the wrong fact (*Voxalune
  Peregrin*).

Both groups target the same stored fact through two different patterns. The model gets it right
7/8 times when it reasons but only 3/8 when asked directly. That gap is what the project studies.

### Weights, if the batch were just these 2 groups

There are 41 + 451 = 492 tokens in all.

| | Weight of a direct token | Weight of a reasoning token | Direct group's share | Reasoning group's share |
|---|---|---|---|---|
| token | 1/492 | 1/492 | 41/492 = **8%** | 451/492 = **92%** |
| per-prompt | 1/2 × 1/41 = **1/82** | 1/2 × 1/451 = **1/902** | **50%** | **50%** |

### Total push on one answer = tokens × |advantage| × weight

Each token's push is `advantage × weight`. An answer's total push is the sum over its tokens.

| | Correct direct answer (5 tokens, +0.625) | Wrong reasoning chain (65 tokens, −0.875) | Ratio |
|---|---|---|---|
| token | 5 × 0.625 / 492 = **0.006** up | 65 × 0.875 / 492 = **0.116** down | ~18× |
| per-prompt | 5 × 0.625 / 82 = **0.038** up | 65 × 0.875 / 902 = **0.063** down | ~1.6× |

Under per-prompt weighting, a single direct token gets about 8× the push of a reasoning token
(1/82 against 1/902). The reasoning chain has 13× more tokens, so the totals end up comparable.
The remaining ratio comes from the advantages, not from answer length.

---

## 4. The problem in r1

r1 used token weighting. Two things combined to shrink direct mode's share of the learning.

1. **Fewer direct groups were trainable.** Dispatch was 1:1, but far fewer direct groups had a
   learning signal:

   | Mode | Trainable groups |
   |---|---|
   | direct | 21.6% |
   | reasoning | 42.1% |

   So trained batches were **33.8% direct groups**.
2. **Token weighting then scaled each group by its length.** Direct groups have about 8 × 5 = 40
   tokens and reasoning groups about 8 × 60 = 480:

   ```
   direct    0.338 × 40  ≈  13.5
   reasoning 0.662 × 480 ≈ 317.8
   direct share of the update ≈ 13.5 / 331 ≈ 4%
   ```

So r1 was in effect **about 96% reasoning-mode RL**, even though the design is 1:1 by question.

This fits the results, though it doesn't prove the cause. The direct gain was small (+0.9) even
though direct had the most room to improve. The reward curves were flat over 53 steps at
lr 1e-6. Training reward went from 0.196 to 0.201 for direct and from 0.400 to 0.406 for
reasoning.

## 5. Recommendation

Use **per-prompt weighting** for every RL run. Every trained question group then counts the same,
whether it is a 41-token direct group or a 451-token reasoning group. That keeps the 1:1
direct/reasoning design that runs through SFT, RL dispatch and eval.

The setting is a single value for the whole run, not one per mode. prime-rl requires every rl
source in a run to use the same value. It is now set in `configs/rl/base.toml`:

```toml
[orchestrator.train.algo]
loss_aggregation = "prompt"
```

The shared `[orchestrator.train.algo]` block applies to both the `direct` and `reasoning`
sources. Checked: the resolved r1_10pct config shows `grpo` / `prompt` for both.

### What per-prompt weighting does NOT fix

1. **The trainable-group imbalance.** Direct would get about **34%** of the update (its share of
   trained groups), not 50%. A true 50/50 would need either sending direct prompts out twice as
   often (2:1) or a per-mode quota of trainable groups per step, which means patching prime-rl's
   orchestrator. **Decision (2026-11-11):** keep 1:1 dispatch (the same prompts in both modes)
   and accept about 34% direct in the update. To report each run's actual share, compute it
   from `train/{direct,reasoning}/all/agent/is_trainable/mean` in `metrics.jsonl`: with 1:1
   dispatch, direct share = d / (d + r).
2. **Length bias inside a group.** It divides by the group's total tokens, not by each answer's
   length. Within one group, a 65-token answer still counts more than a 51-token one. This is
   minor.
3. **The effective step size changes.** Direct tokens get much larger pushes than before, and
   reasoning tokens somewhat smaller ones. Tune lr (1e-6 looked too low in r1; 3e-6 to 5e-6 is
   the candidate range) **with** `"prompt"` on, not before.

### Consequences

- **r1 isn't directly comparable** to runs with `"prompt"`. Re-run r1 with the final settings so
  r1–r4 share one config.
- Decide these at the same time, before r2–r4: lr, the budget (r1 dispatched about 3.2 passes over its
  prompts, because untrainable groups were dropped and re-sampled), and whether "1:1" should mean
  dispatched prompts (as now) or trained groups (needs the quota patch).

---

## 6. Shouldn't the update be fully independent of answer length?

Per-prompt weighting removes length dependence **between groups**, but not **within a group**.
Fully removing it would mean weighting per answer. We considered that and stayed with per-prompt.

### 6.1 Our choice: per-prompt (`"prompt"`)

- **Between groups: independent of length.** Every trained group gets 1/G of the update. A
  41-token direct group and a 451-token reasoning group count the same. This is the length
  dependence that hurt r1 (section 4).
- **Within a group: mildly dependent on length.** Every token in group q weighs `1/(G × T_q)`.
  So an answer's share of its group is `(its tokens) / T_q`. A 65-token answer counts a bit more
  than a 51-token one in the same group.

### 6.2 The alternative: per-answer ("sequence-mean", original GRPO)

Average each answer over its own tokens, then average all the answers in the batch:

```
answer average = (Σ over tokens in answer i of token loss) / |o_i|
loss           = (Σ over all answers of answer average) / (G × 8)
weight         = 1 / (G × 8 × |o_i|)      (depends on the token's own answer)
answer i's share of the update = 1 / (G × 8)
```

Each answer counts the same whatever its length. prime-rl doesn't offer this (only `token` and
`prompt`). Using it would mean a small local patch to `orchestrator/algo/grpo.py`.

### 6.3 Comparison

| Setting | Between groups | Within a group | Available |
|---|---|---|---|
| token (r1) | longer group counts more | longer answer counts more | yes (default) |
| **per-prompt (chosen)** | **equal** | longer answer counts a bit more | **yes** |
| per-answer | equal | equal | needs a patch |

The same r1 example (section 3), with 16 answers in the batch. For per-answer weighting, each
answer's total push is `|advantage| × 1/16`.

| Setting | Correct direct answer (5 tokens, +0.625) | Wrong reasoning chain (65 tokens, −0.875) |
|---|---|---|
| token | 0.006 | 0.116 |
| **per-prompt** | **0.038** | **0.063** |
| per-answer | 0.625/16 = 0.039 | 0.875/16 = 0.055 |

Per-prompt and per-answer are nearly the same here. Within a mode, answer lengths barely vary
(direct 4–6 tokens, reasoning 51–65), so the remaining within-group effect is small. The big
difference is between modes, and per-prompt already removes it.

### 6.4 Some length dependence can't be removed

An answer's total push has to be split across its tokens, so length always shows up somewhere:

- **Per-answer:** the answer's total is fixed. Each token of a 65-token chain gets 1/65 of it,
  and each token of a 5-token answer gets 1/5. Length shrinks the push on each token.
- **Token:** each token's push is fixed. Length increases the answer's total.
- **Per-prompt:** in between. Length within a group increases the total slightly.

The real choice is where length shows up, not whether it does.

### 6.5 Why not per-answer: the length bias

Dividing by each answer's own length biases what the model learns about length (Dr. GRPO,
Liu et al. 2025, "Understanding R1-Zero-Like Training"; DAPO moved to token-level for the same
reason):

- A **wrong** answer's penalty is spread over its tokens. The longer the wrong answer, the
  smaller the penalty on each token. Long wrong answers get punished less than short ones.
- A **correct** answer's reward is concentrated more when it is short. Short right answers get
  rewarded more per token.
- Over training, wrong answers tend to **grow longer**: rambling gets punished less.

For us, that would mean reasoning chains that drift longer when the model is unsure. That makes
response length a confound in the direct-vs-reasoning comparison. Per-prompt weighting
normalizes by the **group's** total tokens, which all 8 answers share, so it doesn't reward
any one answer for being long or short.

### 6.6 Decision

Use **per-prompt** (`loss_aggregation = "prompt"`, set in `configs/rl/base.toml`):

1. It equalizes questions and modes, which is the length dependence that matters for this project.
2. It avoids the per-answer length bias.
3. It's a built-in prime-rl option, so there's no extra patch.

Revisit per-answer only if reasoning lengths start varying a lot within groups. Track
`train/reasoning/all/num_output_tokens/{mean,p10,p90}` (and the `direct` equivalents) in
`<run>/monitors/file/metrics.jsonl` across steps. If the p10–p90 spread widens, or wrong chains
grow much longer than right ones, the within-group length effect is no longer negligible.



─────────────────────────────────────┬─────────────┬────────┬────────────────────────────────────────────────┐
│                Option                 │ Prompts for │  r1    │                 What it tests                  │
│                                       │      r1     │ steps  │                                                │
├───────────────────────────────────────┼─────────────┼────────┼────────────────────────────────────────────────┤
│ A. Both modes per question (current)  │ 3,344       │ 53     │ RL on both behaviours over the same facts      │
├───────────────────────────────────────┼─────────────┼────────┼────────────────────────────────────────────────┤
│ B. Split: half the questions direct,  │ 1,672       │ 27     │ Both behaviours, half the cost. Each mode sees │
│ half reasoning                        │             │        │  only half the facts                           │
├───────────────────────────────────────┼─────────────┼────────┼────────────────────────────────────────────────┤
│ C. Reasoning-only RL                  │ 1,672       │ 27     │ Does RL on reasoning also improve direct       │
│                                       │             │        │ answers?                                       │
├───────────────────────────────────────┼─────────────┼────────┼────────────────────────────────────────────────┤
│ D. Direct-only RL                     │ 1,672       │ 27     │ Can RL improve direct recall on its own?       │
└───────────────────────────────────────┴─────────────┴────────┴────────────────────────────────────────────────┘