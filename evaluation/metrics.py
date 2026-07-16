"""Answer-normalization metrics: standard SQuAD/Search-R1 normalized EM, plus
cover-EM (substring match) logged alongside it.

Default metric across this line of work is EM after normalization (Search-R1
uses it as both eval metric and RL reward). EM is strict -- penalizes
correct-but-rephrased answers -- so we also log cover-EM, which is more
faithful to this project's parametric-recall goal (a gold string appearing
anywhere in the prediction counts).
"""
import re
import string


def normalize_answer(s: str) -> str:
    """Lowercase, strip articles/punctuation, collapse whitespace."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = " ".join(s.split())
    return s


def exact_match(pred: str, golds: list[str]) -> int:
    norm_pred = normalize_answer(pred)
    return int(any(norm_pred == normalize_answer(g) for g in golds))


def cover_em(pred: str, golds: list[str]) -> int:
    norm_pred = normalize_answer(pred)
    return int(any(normalize_answer(g) in norm_pred for g in golds if normalize_answer(g)))
