# Experiment index

Single lookup table linking a run directory (`experiment/expN/`) to its
canonical experiment ID (`E00X`, defined in `project_doc.md`'s Experiment
Log) and planning doc. Update this row whenever a run's status changes.

| Run dir | E-ID | Condition | Model | Datasets | Status | Doc |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| `experiment/exp1/` | E001 | direct | Qwen/Qwen3-4B | nq, triviaqa, popqa, hotpotqa, 2wikimultihopqa, musique, bamboogle (500/set; simpleqa skipped, see note) | Done | [exp1.md](experiments/exp1.md) |
| `experiment/exp2/` | E001 | reasoning | Qwen/Qwen3-4B | same as exp1 | Planned | [exp2.md](experiments/exp2.md) |

**Note:** `simpleqa` is registered but disabled (`datasets_registry.py`) --
it needs an LLM-judge grader, not EM, and isn't on the FlashRAG hub. See
`evaluation/simpleqa_judge.py` for the stub/TODO.

To add an experiment: copy `experiment/exp1/` -> `experiment/expN/`, edit
`settings.json` (including `exp_id`), copy `docs/experiments/template.md` ->
`expN.md`, add a row here.
