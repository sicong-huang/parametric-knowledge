---
id: E013
run_dir: experiment/exp20
condition: reasoning
model: google/gemma-4-E4B-it, nvidia/NVIDIA-Nemotron-Nano-9B-v2, nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16, google/gemma-4-E2B-it
datasets: [hotpotqa, 2wikimultihopqa, musique]
n_examples: see premise2/corpus/tier2_candidates.jsonl (data-limited, not fixed)
status: Done (v2 — placebo fixed, causal claim supported, small-n caveat)
---

# E013 — Tier 2: interventional prefix corruption (Premise 2 causal proof)

**Owner:** Michelle Sheu
**Why:** Premise 2 claims hallucinated intermediate facts in reasoning traces
*cause* hallucinated final answers, not merely correlate with them. Tier 1
(observational, SAFE-graded) can only show correlation controlling for question
difficulty. This experiment directly manipulates one intermediate fact in an
already-correct trace and force-decodes the continuation, to show corruption
causes the answer to flip — the actual causal proof, adapted from Lanham et
al. 2023's "adding mistakes" protocol with a paraphrase control.
**Setup:**
- Source traces: reasoning-condition generations already in
  `experiment/exp{10,14,16,18}` (4 models — Qwen models excluded per project
  decision; SmolLM3/E008 not yet run). No regeneration of the original traces.
- Datasets: hotpotqa, 2wikimultihopqa, musique (gold supporting-fact / hop
  decomposition annotations in `metadata`). Bamboogle excluded — no gold facts.
- Pipeline (`premise2/scripts/`): `build_manifest.py` → `select_targets.py`
  (filter to `em_correct==1` traces, align gold supporting facts to `<think>`
  sentences via word-overlap matching, threshold 0.25) → `corrupt_fact.py`
  (3-way prefix: original / corrupted / paraphrase, corruption+paraphrase via
  local `gemma-4-31b-it` judge server, diff-validated) → `force_decode.py`
  (splice prefix into the rendered chat prompt, regenerate n=4 continuations
  per condition via the same local model, `scripts/generate.py`'s exact
  tokenizer/chat-template pattern) → `score_flips.py` (EM-graded flip rates,
  bootstrap CIs) → `placebo.py` (Task 2.5, off-chain corruption control) →
  `position_recovery.py` (Task 2.6, classification pass, no new generation).
- Grader for corruption/paraphrase edits: local `gemma-4-31b-it`
  (`http://localhost:8000/v1`), not OpenAI — per project decision to keep
  Tier 2 at $0 external spend (deterministic diff-validation, not model
  judgment, is what makes the corruption trustworthy, so citation-fidelity of
  the edit model matters less than for Tier-1 SAFE grading).
**Known deviation from the plan's targets:** Task 2.1 targeted ≥500 usable
candidates/model. Actual data ceiling: only 785 multi-hop traces across all 4
models × 3 datasets were answered correctly at all (multi-hop EM is low, see
E001's exp1.md: 0.018–0.24 range). This is a real data ceiling, not a pipeline
bug — flagged here rather than silently lowering the matching bar to hit the
target number. Usable candidates per model (v2, after the splitter fix below
slightly tightened matching further): gemma-4-E2B-it 54, gemma-4-E4B-it 67,
Nemotron-3-Nano-4B 80, Nemotron-Nano-9B-v2 127 (328 total, 395 matched hops).
**Expected result:** P(wrong | corrupted) substantially higher than
P(wrong | paraphrase), for the on-chain corruption. Off-chain placebo
(`placebo.py`) should show ≈0 gap — if not, the corruption methodology needs
revisiting before Tier 2 can be cited as causal.

**v1 result (superseded — see v2 below):** first pass ran end-to-end on 337
candidates and came back inconclusive: the off-chain placebo gap (+0.041) was
statistically indistinguishable from the on-chain gap (+0.047), so the run
couldn't tell "the corrupted fact mattered" from "editing the trace at all is
destabilizing." Root-caused (not just re-measured by eye — a paired
difference-in-differences on the traces shared by both arms confirmed the gap
was real, not a population artifact) to a **depth confound**: `placebo.py`'s
original selection took the *first* off-chain sentence ≥5 words, landing near
the top of the trace (median prefix 19 words vs. the main arm's 70.5 words).
On the *same* traces with **no edit at all**, that shallow cut point was
already ~10pp more unstable than the main arm's deeper cut
(`P(wrong|original)` diff −0.104 [−0.141, −0.067]) — the placebo arm was
measuring cut-depth sensitivity, not fact relevance. A second, independent
issue: "off-chain" was defined by matcher failure (`select_targets.py`: not
the sentence the aligner picked), not by any positive irrelevance check —
spot-checked and found a genuine on-chain restatement mislabeled off-chain in
`exp10:hotpotqa:dev_1140` (a Jack Russell Terrier sentence describing hop 1's
fact in different words, which the aligner missed for the *actual* hop-1
sentence at a barely-passing 0.25 score).

**v2 fix and re-run:** three changes, then full re-run on all 4 models
(`premise2/corpus/exp20_v1_backup/` holds the v1 corpus/reports for
reference):
1. `select_targets.py`'s sentence splitter no longer leaves bare `"2."` /
   `"3."` list-marker fragments (was: `_SENT_SPLIT_RE` alone split on every
   period, so a model's numbered list produced marker-only fragments that
   polluted matching on both arms).
2. `off_chain_sentences` is now a *positive* filter
   (`offchain_purity_score`): a candidate must have enough content words, and
   score below a **lowered** 0.15 jaccard threshold against every gold fact
   (looser than the 0.25 accept threshold, specifically to catch near-miss
   on-chain restatements like the `dev_1140` case) and against the question
   and gold answer.
3. `placebo.py` now **position-matches**: for each matched on-chain hop, it
   picks the off-chain sentence whose prefix-word-count is closest (within a
   15-word tolerance), pairing placebo rows to the same `(trace_id,
   hop_index)` as the main arm instead of a `hop_index: -1` sentinel. Hops
   with no off-chain sentence at a comparable depth are skipped rather than
   falling back to a mismatched cut point — this is the deliberate
   power-for-validity tradeoff: only 161 of 395 matched hops (41%) had a
   position-matched partner within tolerance, but on those 161 the depth
   confound is gone (mean prefix words 121.8 main vs. 120.3 placebo, mean
   abs diff 10.1 words).

New `compare_arms.py` computes a paired difference-in-differences (main gap
minus placebo gap, per shared `(trace_id, hop_index)`) instead of comparing
cell means by eye, and reports `P(wrong|original)` per arm explicitly as the
no-op instability control (this satisfies the prior write-up's Decision item
(c) — the `original` condition already *was* that no-op; it only needed
reporting, not a new run).

**v2 result:** on 328 candidates (343 hop-fact jobs → 1029 main prefixes →
4012 main continuations; 161 position-matched placebo jobs → 483 placebo
prefixes → 1892 placebo continuations):

*No-op instability collapsed to ~0*, confirming the depth-confound diagnosis:
`P(wrong|original)` unweighted mean is now 0.245 (main) vs. 0.241 (placebo),
diff +0.003 — down from v1's −0.104 [−0.141, −0.067]. The arms are no longer
distinguishable by cut depth alone.

*Main gap held steady*: +0.041 unweighted mean across the 12 cells
(`tier2_results.md`) — essentially unchanged from v1's +0.047, as expected
since the main arm's selection logic didn't change apart from the splitter
fix. Individually-significant cells: Nemotron-3-Nano-4B/musique (+0.219
[+0.062, +0.438]), Nemotron-Nano-9B-v2/2wikimultihopqa (+0.068 [+0.000,
+0.136]).

*Placebo gap flipped*: now −0.027 unweighted mean (`tier2_placebo_results.md`)
— down from v1's +0.041, and now on the *opposite* side of zero from the main
arm as expected for a clean control.

*Difference-in-differences* (`tier2_arm_comparison.md`, paired on 139 shared
`(trace_id, hop_index)` questions): **+0.068 [+0.004, +0.135]** — excludes
zero. Per the plan's pre-registered interpretation gate ("the causal claim
survives only if the main gap stays positive and DiD excludes zero"), **this
result clears the gate.**

*Position/recovery* (`tier2_position_and_recovery.md`, corrupted-condition
only, unaffected by the placebo fix): flip rate is generally highest at
hop1/hop1-only, self-correction rate (lexical-marker heuristic — coarse, not
spot-checked) ranges 0.307 (Nemotron-Nano-9B-v2) to 0.654 (gemma-4-E2B-it) —
same pattern as v1, larger models self-correct injected errors more often.

**What we learned:** The v1 placebo failure was a real methodology bug (cut
depth), not evidence against Premise 2 — fixing it flips the placebo to the
correct side of zero while leaving the main effect essentially unchanged,
which is the signature of "the confound was masking a real effect," not "the
effect was an artifact all along." The pooled DiD (+0.068, CI excludes 0) is
the first result in this plan that distinguishes on-chain fact corruption
from generic force-decode instability. **Caveat, and it's not small:**
position-matching this strictly means n is thin — 139 paired questions total,
and unevenly distributed (Nemotron-Nano-9B-v2 alone contributes 88 of 139,
~63%; gemma-4-E2B-it and gemma-4-E4B-it together contribute only 23). Most
individual (model, dataset) cells still have zero-crossing CIs; the pooled
result is doing the work, and it rests heavily on one model family. Treat
this as supportive, not conclusive, evidence for the causal claim until the
candidate pool grows (the 15-word position tolerance is the main lever —
widening it recovers pairs at the cost of a looser depth match; see the
plan's tolerance-sensitivity option that wasn't run this pass).
**Decision:** Tier 2 can now be cited as *supportive* (not yet definitive)
causal evidence for Premise 2, with the small-n and single-model-family
caveats stated above. Before treating this as final: (a) re-score at a wider
position tolerance (e.g. 25 words) and confirm the DiD conclusion is stable
across the choice (sensitivity check, no new generation needed — same
continuations, different pairing at select time); (b) if power remains a
concern, scale the candidate pool, which is still capped well under the
plan's 500/model target. Tier 1 (correlational, not yet started) and Tier 3
(blocked) are unaffected by this and can proceed independently per the
existing plan.

**Links**
- Settings: `experiment/exp20/settings.json`
- Corpus: `premise2/corpus/traces_manifest.jsonl`, `tier2_candidates.jsonl`,
  `tier2_prefixes.jsonl`, `tier2_continuations.jsonl`,
  `tier2_placebo_prefixes.jsonl`, `tier2_placebo_continuations.jsonl`
  (v1 versions preserved in `premise2/corpus/exp20_v1_backup/`)
- Reports: `premise2/reports/tier2_results.md`, `tier2_placebo_results.md`,
  `tier2_arm_comparison.md`, `tier2_position_and_recovery.md`
  (v1 reports preserved in `premise2/reports/exp20_v1_backup/`)
- Source spec: `premise2_claude_code_plan.md`
