"""Ex-Recall metric, ported from Ma & Hewitt's reference implementation
(MelodyHorsee/parametric-knowledge-access, scripts/Ex-Recall.py).

Two-stage metric: (1) our `predicted_answer` (evaluation/parse.py:extract_answer,
the <answer>-tag / post-</think> heuristic) already matches their
`extracted_prediction` stage; (2) a small LM "refines" that extraction down to a
single committed answer span (collapsing hedged/multi-candidate output), then a
word-boundary substring match against the gold aliases determines recall.

This is a co-primary metric alongside EM, not a replacement -- it exists to avoid
scoring 0 just because reasoning-ON output didn't emit a clean <answer> tag or
hedged across candidates, which is the confound documented in
eval_metrics_research.md.

normalize_answer here is intentionally a local, richer variant (handles unicode
dashes, underscores, curly apostrophes) than evaluation/metrics.py's normalize_answer
-- kept separate so it never perturbs the strict-EM comparability anchor.
"""
import os
import re
import string

EXTRACTOR_MODEL = os.environ.get("EXTRACTOR_MODEL", "gpt-5-mini-2025-08-07")

# Verbatim few-shot refine prompt from the reference Ex-Recall.py.
REFINE_PROMPT = """
You are given an answer that may contain one or multiple possibilities.
If it only contains one, just output it as is.
Otherwise, choose the answer that is stated with the most confidence, if there are multiple options.
DO NOT correct the answer, even if you think it's incorrect.
If the answer is just a question being repeated (not an actual answer), keep it exactly as is.
Examples:

A: While Leif Erikson reached North America earlier, Christopher Columbus is usually cited.
Refined Answer: Christopher Columbus

A: While some might think Saturn, the largest planet is Jupiter.
Refined Answer: Jupiter

A: It could be Paris, but some might mistakenly say Lyon.
Refined Answer: Paris

A: Leonardo da Vinci painted the Mona Lisa.
Refined Answer: Leonardo da Vinci painted the Mona Lisa.

A: Shanghai is the capital of China.
Refined Answer: Shanghai is the capital of China.

A: What is the capital of France?
Refined Answer: What is the capital of France?

Original Answer: {answer}
Refined Answer:
""".strip()


def normalize_answer(s: str) -> str:
    """Richer TriviaQA-derived normalization (from the reference Ex-Recall.py):
    unicode dashes -> space, underscores -> space, lower, strip punctuation
    (incl. curly apostrophes/backtick), drop articles, collapse whitespace."""

    def normalize_unicode_dashes(text):
        return re.sub(r"[‐-‒–—―−]", " ", text)

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def handle_punc(text):
        exclude = set(string.punctuation + "".join(["'", "’", "´", "`"]))
        return "".join(ch if ch not in exclude else " " for ch in text)

    def lower(text):
        return text.lower()

    def replace_underscore(text):
        return text.replace("_", " ")

    return white_space_fix(
        remove_articles(handle_punc(lower(normalize_unicode_dashes(replace_underscore(s)))))
    ).strip()


def recall_score(prediction: str, ground_truths: list[str]) -> bool:
    """Word-boundary substring match: True if any normalized gold appears as a
    whole-word (whitespace-flexible) span inside the normalized prediction."""
    pred = normalize_answer(prediction)
    for gt in ground_truths:
        gt_normalized = normalize_answer(gt)
        if not gt_normalized:
            continue
        gt_escaped = re.escape(gt_normalized)
        gt_pattern = gt_escaped.replace(r"\ ", r"\s+")
        pattern = r"\b" + gt_pattern + r"\b"
        if re.search(pattern, pred):
            return True
    return False


def refine_answer(predicted: str) -> str:
    """One LM call to collapse a hedged/multi-candidate extraction to a single
    committed answer span. Uses OPENAI_BASE_URL to allow a local vLLM server
    instead of paying per-call API cost at scale."""
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")
    if not api_key and not base_url:
        raise RuntimeError(
            "OPENAI_API_KEY not set; required for the Ex-Recall refine-answer LM call."
        )

    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)
    prompt = REFINE_PROMPT.format(answer=predicted)
    response = client.chat.completions.create(
        model=EXTRACTOR_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content.strip()


def score_ex_recall(records: list[dict]) -> dict:
    """records: list of {"golden_answers": [str], "predicted_answer": str}.

    Returns summary dict with mean `ex_recall` plus per-record refined answers
    and recall flags (caller merges these into the eval jsonl).
    """
    n = len(records)
    n_recalled = 0
    n_skipped = 0
    refined_answers = []
    recalled_flags = []
    skipped_flags = []

    for rec in records:
        predicted = rec.get("predicted_answer", "")
        golds = rec["golden_answers"]

        if not predicted:
            refined_answers.append("")
            recalled_flags.append(False)
            skipped_flags.append(True)
            n_skipped += 1
            continue

        refined = refine_answer(predicted)
        recalled = recall_score(refined, golds)

        refined_answers.append(refined)
        recalled_flags.append(recalled)
        skipped_flags.append(False)
        if recalled:
            n_recalled += 1

    return {
        "n": n,
        "n_recalled": n_recalled,
        "n_skipped": n_skipped,
        "ex_recall": n_recalled / n if n else 0.0,
        "refined_answers": refined_answers,
        "recalled": recalled_flags,
        "skipped": skipped_flags,
    }
