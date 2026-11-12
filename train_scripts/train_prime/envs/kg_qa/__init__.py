"""kg_qa: single-turn synthetic knowledge-graph QA with a binary exact-match reward.

A verifiers v1 taskset for prime-rl RL. The model sees just the question (no system prompt,
the same prompt as SFT and eval.py) and gets reward 1.0 when its final answer equals the
gold answer after normalization (lowercase, no punctuation/articles, collapsed whitespace;
same as evaluation/metrics.py), else 0.0.

Direct vs reasoning is NOT a taskset setting. Each prime-rl train source sets it through the
agent's `chat_template_kwargs.enable_thinking`, and both sources load the same tasks file. The
final answer is the text after the last `</think>`. If the renderer already split the thinking
off into reasoning_content, the message content is the answer as-is. An unterminated `<think>`
counts as no answer, as in eval.py's strip_think.

Loaded by prime-rl as taskset id "kg_qa". run_rl.sh puts this directory's parent on PYTHONPATH,
so nothing is installed into prime-rl's environment.
"""

import json
import re
import string

import verifiers.v1 as vf


def normalize_answer(s: str) -> str:
    """Lowercase, strip punctuation and articles, collapse whitespace (= evaluation/metrics.py)."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def final_answer(text: str) -> str:
    """Text after the last </think>; '' for an unterminated think block (= eval.py strip_think)."""
    _, sep, answer = text.rpartition("</think>")
    if sep:
        return answer
    return "" if "<think>" in text else text


class KGQAData(vf.TaskData):
    answer: str
    """Gold answer string."""
    n_hop: str = ""
    pattern: str = ""


class KGQATask(vf.Task[KGQAData]):
    @vf.stop
    async def single_turn(self, trace: vf.Trace) -> bool:
        return trace.num_turns >= 1

    @vf.reward(weight=1.0)
    async def exact_match(self, trace: vf.Trace) -> float:
        pred = final_answer(trace.last_reply or "")
        return float(normalize_answer(pred) == normalize_answer(self.data.answer))


class KGQAConfig(vf.TasksetConfig):
    data_path: str = ""
    """JSONL written by prepare_rl_data.py: one row per question (qaid, question, answer, ...)."""


class KGQATaskset(vf.Taskset[KGQATask, KGQAConfig]):
    def load(self) -> list[KGQATask]:
        if not self.config.data_path:
            raise ValueError("kg_qa needs env.taskset.data_path")
        tasks = []
        with open(self.config.data_path) as f:
            for line in f:
                row = json.loads(line)
                tasks.append(
                    KGQATask(
                        KGQAData(
                            id=row["qaid"],
                            prompt=row["question"],
                            answer=row["answer"],
                            n_hop=row["n_hop"],
                            pattern=row["pattern"],
                        ),
                        self.config.task,
                    )
                )
        return tasks


__all__ = ["KGQATaskset"]
