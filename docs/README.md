# Experiment index

Single lookup table linking a run directory (`experiment/expN/`) to its
canonical experiment ID (`E00X`, defined in `project_doc.md`'s Experiment
Log) and planning doc. Update this row whenever a run's status changes.

| Run dir | E-ID | Condition | Model | Datasets | Status | Doc |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| `experiment/exp1/` | E001 | direct | Qwen/Qwen3-4B | nq, triviaqa, popqa, hotpotqa, 2wikimultihopqa, musique, bamboogle (500/set; simpleqa skipped, see note) | Done | [exp1.md](experiments/exp1.md) |
| `experiment/exp2/` | E001 | reasoning | Qwen/Qwen3-4B | same as exp1 | Planned | [exp2.md](experiments/exp2.md) |
| `experiment/exp3/` | E004 | direct | Qwen/Qwen3.5-2B | 7 EM datasets + simpleqa_verified (n=500 per dataset, bamboogle=125) | Done | [exp3.md](experiments/exp3.md) |
| `experiment/exp4/` | E004 | reasoning | Qwen/Qwen3.5-2B | same as exp3 | Done | [exp4.md](experiments/exp4.md) |
| `experiment/exp5/` | E005 | direct | Qwen/Qwen3.5-4B | same as exp3 | Done | [exp5.md](experiments/exp5.md) |
| `experiment/exp6/` | E005 | reasoning | Qwen/Qwen3.5-4B | same as exp3 | Done | [exp6.md](experiments/exp6.md) |
| `experiment/exp7/` | E006 | direct | Qwen/Qwen3.5-9B | same as exp3 | Done | [exp7.md](experiments/exp7.md) |
| `experiment/exp8/` | E006 | reasoning | Qwen/Qwen3.5-9B | same as exp3 | Done | [exp8.md](experiments/exp8.md) |
| `experiment/exp9/` | E007 | direct | google/gemma-4-E4B-it | same as exp3 | Done | [exp9.md](experiments/exp9.md) |
| `experiment/exp10/` | E007 | reasoning | google/gemma-4-E4B-it | same as exp3 | Done | [exp10.md](experiments/exp10.md) |
| `experiment/exp11/` | E008 | direct | HuggingFaceTB/SmolLM3-3B | same as exp3 | Planned | [exp11.md](experiments/exp11.md) |
| `experiment/exp12/` | E008 | reasoning | HuggingFaceTB/SmolLM3-3B | same as exp3 | Planned | [exp12.md](experiments/exp12.md) |
| `experiment/exp13/` | E009 | direct | nvidia/NVIDIA-Nemotron-Nano-9B-v2 | same as exp3 | Done | [exp13.md](experiments/exp13.md) |
| `experiment/exp14/` | E009 | reasoning | nvidia/NVIDIA-Nemotron-Nano-9B-v2 | same as exp3 | Done | [exp14.md](experiments/exp14.md) |
| `experiment/exp15/` | E010 | direct | nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 | same as exp3 | Done | [exp15.md](experiments/exp15.md) |
| `experiment/exp16/` | E010 | reasoning | nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 | same as exp3 | Done | [exp16.md](experiments/exp16.md) |
| `experiment/exp17/` | E011 | direct | google/gemma-4-E2B-it | same as exp3 | Done | [exp17.md](experiments/exp17.md) |
| `experiment/exp18/` | E011 | reasoning | google/gemma-4-E2B-it | same as exp3 | Done | [exp18.md](experiments/exp18.md) |
| `experiment/misc/concise_reasoning_probe/` | -- (misc, not a formal exp) | reasoning | Qwen/Qwen3.5-4B | nq, popqa, hotpotqa (n=150 each) | Done | [README.md](../experiment/misc/concise_reasoning_probe/README.md) |
| `experiment/exp19/` | E012 | reasoning | 4 models (gemma-4-E4B/E2B, Nemotron-Nano-9B-v2, Nemotron-3-Nano-4B) | Premise 2 Tier 1 (SAFE observational) -- hotpotqa/2wikimultihopqa/musique/bamboogle | Planned | [exp19.md](experiments/exp19.md) |
| `experiment/exp20/` | E013 | reasoning | 4 models (gemma-4-E4B/E2B, Nemotron-Nano-9B-v2, Nemotron-3-Nano-4B) | Premise 2 Tier 2 (interventional prefix-corruption causal proof) -- hotpotqa/2wikimultihopqa/musique | Done (v2 -- placebo fixed, DiD +0.068 [+0.004,+0.135] excludes zero, small-n caveat, see exp20.md) | [exp20.md](experiments/exp20.md) |
| `experiment/exp21/` | E014 | reasoning | (synthetic KG, model TBD) | Premise 2 Tier 3 (synthetic KG replication) | Blocked | [exp21.md](experiments/exp21.md) |

**Note:** `simpleqa` is registered but disabled (`datasets_registry.py`) --
it needs an LLM-judge grader, not EM, and isn't on the FlashRAG hub. See
`evaluation/simpleqa_judge.py` for the stub/TODO.

To add an experiment: copy `experiment/exp1/` -> `experiment/expN/`, edit
`settings.json` (including `exp_id`), copy `docs/experiments/template.md` ->
`expN.md`, add a row here.
