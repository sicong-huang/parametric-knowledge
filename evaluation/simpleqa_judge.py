"""STUB -- SimpleQA is not on the FlashRAG hub and is not EM-scored.

Native eval (openai/simple-evals) grades each answer with an LLM into one of
three classes: CORRECT / INCORRECT / NOT_ATTEMPTED. This project's registry
(datasets_registry.py) marks "simpleqa" as enabled=False until this is built.

TODO (follow-up, not in this cut):
  - Load questions from the OpenAI simple-evals CSV (not FlashRAG).
  - Implement a 3-way LLM-judge grader (reuse the simple-evals grading
    prompt rather than inventing a new rubric).
  - Report SimpleQA accuracy separately from the EM-based datasets --
    it is a calibration benchmark, not directly comparable to normalized EM.
"""


def grade(question: str, gold: str, predicted: str) -> str:
    raise NotImplementedError(
        "SimpleQA LLM-judge grading not yet implemented; dataset is disabled "
        "in datasets_registry.py (enabled=False)."
    )
