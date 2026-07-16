"""Parse model output: strip <think>...</think>, extract <answer>...</answer>
(Search-R1 prompt template)."""
import re

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
THINK_CONTENT_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def strip_think(text: str) -> str:
    return THINK_RE.sub("", text)


def extract_think(text: str) -> str:
    """Return the concatenated content of all <think>...</think> blocks
    (the reasoning trace), or "" if none are present (e.g. direct condition)."""
    matches = THINK_CONTENT_RE.findall(text)
    return "\n".join(m.strip() for m in matches)


def extract_answer(text: str) -> str:
    """Return content of the last <answer>...</answer> tag; fall back to the
    stripped full text (minus any <think> block) if no tag is found."""
    matches = ANSWER_RE.findall(text)
    if matches:
        return matches[-1].strip()
    return strip_think(text).strip()


def has_answer_tag(text: str) -> bool:
    """True if an <answer>...</answer> tag is present (extract_answer would NOT
    fall back). Extraction failure = not has_answer_tag(raw_output) -- a
    diagnostic for the format-vs-recall confound between conditions (see
    eval_metrics_research.md §5)."""
    return ANSWER_RE.search(text) is not None
