"""Parse model output: strip <think>...</think>, extract <answer>...</answer>
(Search-R1 prompt template)."""
import re

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def strip_think(text: str) -> str:
    return THINK_RE.sub("", text)


def extract_answer(text: str) -> str:
    """Return content of the last <answer>...</answer> tag; fall back to the
    stripped full text (minus any <think> block) if no tag is found."""
    matches = ANSWER_RE.findall(text)
    if matches:
        return matches[-1].strip()
    return strip_think(text).strip()
