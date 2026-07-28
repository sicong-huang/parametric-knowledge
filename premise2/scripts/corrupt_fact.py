"""Tier 2 Task 2.2: build the three-way corrupted prefix set.

For each candidate trace (premise2/corpus/tier2_candidates.jsonl) and each
matched on-chain hop fact, build three prefixes identical up to and including
the aligned sentence:
  (a) original      -- untouched
  (b) corrupted      -- minimal factual edit (swap entity/date/number only)
  (c) paraphrase     -- same fact, reworded, still correct

Uses the local gemma-4-31b judge server (OPENAI_BASE_URL, per user decision to
keep Tier-2 corruption edits off OpenAI spend -- see premise2_claude_code_plan.md
grader-split decision). Requires scripts/serve_judge.sh running.

Every corrupted sentence (b) is validated programmatically by diffing against
the original: reject edits that change more than the target span (loose token
heuristic: at most half the words may differ, and length must stay within 30%).

Usage:
    OPENAI_BASE_URL=http://localhost:8000/v1 OPENAI_API_KEY=EMPTY \\
    EXTRACTOR_MODEL=gemma-4-31b-it \\
    uv run python premise2/scripts/corrupt_fact.py
"""
import concurrent.futures
import json
import os
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

REPO_ROOT = pathlib.Path(__file__).parent.parent.parent
CANDIDATES_PATH = REPO_ROOT / "premise2" / "corpus" / "tier2_candidates.jsonl"
PREFIXES_OUT = REPO_ROOT / "premise2" / "corpus" / "tier2_prefixes.jsonl"

DEFAULT_MODEL = "gemma-4-31b-it"
CONCURRENCY = int(os.environ.get("PREMISE2_CONCURRENCY", "8"))

CORRUPT_PROMPT = """You will be given one sentence from a reasoning trace. Change ONLY one \
factual detail in it -- swap a single entity name, date, or number for a plausible but \
DIFFERENT and FACTUALLY WRONG value. Do not change anything else: keep the sentence \
structure, length, and all other words identical. Output only the edited sentence, \
nothing else.

Sentence: {sentence}
Edited sentence (one factual detail changed, everything else identical):"""

PARAPHRASE_PROMPT = """Reword the following sentence so it says the same true thing in \
different words. Keep it factually identical -- do not change any entity, date, or \
number. Output only the reworded sentence, nothing else.

Sentence: {sentence}
Reworded sentence (same facts, different wording):"""


def _client():
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")
    if not api_key and not base_url:
        raise RuntimeError(
            "OPENAI_API_KEY or OPENAI_BASE_URL must be set -- point this at the "
            "local judge server (bash scripts/serve_judge.sh) per the grader-split decision."
        )
    return OpenAI(api_key=api_key or "EMPTY", base_url=base_url)


def _call(client, model, prompt: str) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()


_WORD_RE = re.compile(r"\w+")


def validate_corruption(original: str, corrupted: str) -> bool:
    """Loose diff-based validator: reject edits that touch more than the
    target span. At most half the words may differ, and length must stay
    within 30% of the original (rejects the model rewriting the whole thing)."""
    if not corrupted or corrupted.strip() == original.strip():
        return False
    orig_words = _WORD_RE.findall(original.lower())
    corr_words = _WORD_RE.findall(corrupted.lower())
    if not orig_words or not corr_words:
        return False
    if abs(len(corr_words) - len(orig_words)) / len(orig_words) > 0.3:
        return False
    # word-level diff count via simple positional/set comparison
    orig_set = set(orig_words)
    corr_set = set(corr_words)
    diff = len(orig_set.symmetric_difference(corr_set))
    return diff <= max(2, len(orig_set) // 2)


def build_one_prefix_set(client, model, candidate, hop, sentence_key):
    """Returns list of {condition, prefix_text, sentence} rows for one hop fact.
    prefix_text = everything in trace_text up to and including the (possibly
    substituted) sentence, so force_decode.py can truncate the original trace
    at this point and splice in the continuation."""
    original_sentence = hop["sentence_span"]
    trace_text = candidate["_trace_text"]
    prefix_before = candidate["_prefix_before_sentence"][sentence_key]

    rows = [{"condition": "original", "prefix_text": prefix_before + original_sentence, "sentence": original_sentence}]

    corrupted_sentence = _call(client, model, CORRUPT_PROMPT.format(sentence=original_sentence))
    if validate_corruption(original_sentence, corrupted_sentence):
        rows.append({"condition": "corrupted", "prefix_text": prefix_before + corrupted_sentence, "sentence": corrupted_sentence})
    else:
        rows.append({"condition": "corrupted", "prefix_text": None, "sentence": corrupted_sentence, "rejected": True})

    paraphrase_sentence = _call(client, model, PARAPHRASE_PROMPT.format(sentence=original_sentence))
    rows.append({"condition": "paraphrase", "prefix_text": prefix_before + paraphrase_sentence, "sentence": paraphrase_sentence})

    return rows


def main():
    model = os.environ.get("EXTRACTOR_MODEL", DEFAULT_MODEL)
    client = _client()

    candidates = []
    with open(CANDIDATES_PATH) as f:
        for line in f:
            row = json.loads(line)
            if row["any_hop_matched"]:
                candidates.append(row)

    # Need trace_text (not just reasoning sentences) to build the true prefix
    # up to the target sentence -- pull from the manifest.
    trace_text_by_id = {}
    with open(REPO_ROOT / "premise2" / "corpus" / "traces_manifest.jsonl") as f:
        for line in f:
            r = json.loads(line)
            trace_text_by_id[r["trace_id"]] = r["reasoning"]

    jobs = []  # (candidate, hop, key) tuples to submit
    for cand in candidates:
        trace_text = trace_text_by_id.get(cand["trace_id"])
        if trace_text is None:
            continue
        cand["_trace_text"] = trace_text
        prefix_before_map = {}
        for hop in cand["hop_facts"]:
            if not hop["matched"]:
                continue
            sent = hop["sentence_span"]
            idx = trace_text.find(sent)
            if idx == -1:
                continue
            prefix_before_map[sent] = trace_text[:idx]
            jobs.append((cand, hop, sent))
        cand["_prefix_before_sentence"] = prefix_before_map

    print(f"[corrupt_fact] {len(jobs)} hop-fact jobs across {len(candidates)} candidates, model={model}")

    out_rows = []
    n_rejected = 0

    def worker(job):
        cand, hop, sent = job
        try:
            rows = build_one_prefix_set(client, model, cand, hop, sent)
        except Exception as e:
            return cand["trace_id"], hop["hop_index"], None, str(e)
        return cand["trace_id"], hop["hop_index"], rows, None

    with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        futures = [pool.submit(worker, job) for job in jobs]
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            trace_id, hop_index, rows, err = future.result()
            if err:
                print(f"  [warn] {trace_id} hop{hop_index}: {err}")
                continue
            for r in rows:
                if r["condition"] == "corrupted" and r.get("rejected"):
                    n_rejected += 1
                out_rows.append({
                    "trace_id": trace_id,
                    "hop_index": hop_index,
                    "condition": r["condition"],
                    "prefix_text": r["prefix_text"],
                    "sentence": r["sentence"],
                })
            if (i + 1) % 20 == 0:
                print(f"  ... {i + 1}/{len(jobs)}")

    PREFIXES_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(PREFIXES_OUT, "w") as f:
        for row in out_rows:
            f.write(json.dumps(row) + "\n")

    print(f"[done] {len(out_rows)} prefix rows -> {PREFIXES_OUT}")
    print(f"  corrupted edits rejected by diff-validator: {n_rejected}")


if __name__ == "__main__":
    main()
