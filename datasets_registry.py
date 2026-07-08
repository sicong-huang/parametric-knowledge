"""Single source of truth for the real-dataset eval suite.

All datasets share a unified schema from the FlashRAG hub
(RUC-NLPIR/FlashRAG_datasets on HuggingFace) -- the same source Search-R1
pulls from. Each record: {"id", "question", "golden_answers": [...], "metadata"?}.

Exact train-split counts vary by preprocessing (open vs full NQ, unfiltered vs
RC TriviaQA, MuSiQue-Ans vs full) -- verify against the specific FlashRAG file
loaded, not headline numbers in project_doc.md.
"""

FLASHRAG_REPO = "RUC-NLPIR/FlashRAG_datasets"
FLASHRAG_BASE_URL = f"https://huggingface.co/datasets/{FLASHRAG_REPO}/resolve/main"

DATASETS = {
    "nq": {
        "dir": "nq",
        "eval_split": "test",
        "metric": "em",
        "hop": "single",
        "enabled": True,
    },
    "triviaqa": {
        "dir": "triviaqa",
        "eval_split": "test",
        "metric": "em",
        "hop": "single",
        "enabled": True,
    },
    "popqa": {
        "dir": "popqa",
        "eval_split": "test",
        "metric": "em",
        "hop": "single",
        "enabled": True,
    },
    "simpleqa_verified": {
        # NOTE: not on the FlashRAG hub -- loaded from the HF hub directly
        # (google/simpleqa-verified). Not EM-scored: native eval is a GPT-4.1
        # LLM grader (3-way: correct / incorrect / not_attempted), per
        # openai/simple-evals grading convention. See evaluation/simpleqa_judge.py.
        "dir": None,
        "hf_repo": "google/simpleqa-verified",
        "eval_split": "eval",
        "metric": "llm_judge",
        "hop": "single",
        "enabled": True,
    },
    "hotpotqa": {
        "dir": "hotpotqa",
        "eval_split": "dev",  # no public test labels; dev is the eval split used
        "metric": "em",
        "hop": "multi",
        "enabled": True,
    },
    "2wikimultihopqa": {
        "dir": "2wikimultihopqa",
        "eval_split": "dev",
        "metric": "em",
        "hop": "multi",
        "enabled": True,
    },
    "musique": {
        "dir": "musique",
        "eval_split": "dev",
        "metric": "em",
        "hop": "multi",
        "enabled": True,
    },
    "bamboogle": {
        "dir": "bamboogle",
        "eval_split": "test",
        "metric": "em",
        "hop": "multi",
        "enabled": True,
    },
}


def enabled_datasets():
    return [name for name, cfg in DATASETS.items() if cfg.get("enabled")]


def flashrag_url(dataset: str) -> str:
    cfg = DATASETS[dataset]
    if cfg["dir"] is None:
        raise ValueError(f"{dataset} has no FlashRAG source (see registry note)")
    return f"{FLASHRAG_BASE_URL}/{cfg['dir']}/{cfg['eval_split']}.jsonl"
