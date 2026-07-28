---
id: E014
run_dir: experiment/exp21
condition: reasoning
model: TBD (synthetic-KG-tuned model)
datasets: synthetic KG QA (not yet generated)
n_examples: TBD
status: Blocked
---

# E014 — Tier 3: synthetic KG replication (BLOCKED)

**Owner:** Michelle Sheu
**Why:** Replicate Tier 2's causal force-decode result where the dependency
graph is fully known by construction (removing Tier 2's alignment guesswork),
and get a frequency-stratified breakdown (does corrupting a low- vs
high-pretraining-frequency fact flip the answer more or less often).
**Setup:** Would reuse the project's existing synthetic biography/KG QA
generation pipeline and an SFT'd model tuned on it, per
`premise2_claude_code_plan.md` Tier 3. **Blocked: no such pipeline exists in
this repo** (verified via repo-wide search — no synthetic biography/KG/SFT
scripts or data anywhere). Do not build a new one per the plan's explicit
instruction; revisit only once that pipeline and an SFT'd model exist.
**Expected result:** N/A — blocked.
**Result:** N/A — blocked.
**What we learned:** N/A — blocked.
**Decision:** Flag as blocked. Revisit when the synthetic biography/KG
generation pipeline + SFT'd model referenced in the plan become available.

**Links**
- Source spec: `premise2_claude_code_plan.md` (Tier 3)
- Report stub: `premise2/reports/tier3_results.md` (blocked)
