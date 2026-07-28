"""Tier 2 Task 2.1: select target traces + align gold supporting facts to
sentences in the model's <think> block.

Filters premise2/corpus/traces_manifest.jsonl to multi-hop traces (hotpotqa,
2wikimultihopqa, musique -- bamboogle excluded, no gold supporting-fact
annotations) that were answered correctly (em_correct==1), and for each gold
supporting fact / decomposition hop, finds the best-matching sentence(s) in the
trace's <think> block via a normalized word-overlap heuristic. Unmatched hops
are flagged (on_chain candidates need an alignment; unmatched traces are kept,
not dropped, per the spec).

Also tags each trace's remaining (non-gold-matched) <think> sentences as
off-chain candidates, needed for the Task 2.5 placebo corruption.

Usage:
    uv run python premise2/scripts/select_targets.py
"""
import json
import pathlib
import re
import string
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

REPO_ROOT = pathlib.Path(__file__).parent.parent.parent
MANIFEST_PATH = REPO_ROOT / "premise2" / "corpus" / "traces_manifest.jsonl"
CANDIDATES_OUT = REPO_ROOT / "premise2" / "corpus" / "tier2_candidates.jsonl"

MULTIHOP_DATASETS = ["hotpotqa", "2wikimultihopqa", "musique"]
MATCH_THRESHOLD = 0.25  # jaccard word-overlap threshold to accept a sentence match
TARGET_PER_MODEL = 500

STOPWORDS = {
    "a", "an", "the", "is", "was", "were", "are", "be", "been", "of", "in",
    "on", "at", "to", "for", "and", "or", "his", "her", "its", "he", "she",
    "it", "they", "this", "that", "with", "as", "by", "from", "also",
}

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[a-z0-9]+")
# A period preceded by a lone digit/letter enumerator ("1.", "a.", "iv.") is a
# list marker, not a sentence boundary -- don't split there.
_ENUMERATOR_RE = re.compile(r"^\s*(?:[0-9]{1,3}|[a-zA-Z]|[ivxlcdm]{1,6})\.\s*$", re.IGNORECASE)
MIN_CONTENT_WORDS = 3  # fragments with fewer normalized content words are debris, not sentences


def split_sentences(text: str) -> list[str]:
    """Split into sentences, then re-merge/drop list-marker debris.

    _SENT_SPLIT_RE alone breaks on every period, so a model's numbered list
    ("1. **Identify...** 2. **Analyze...**") yields separate "1." / "2."
    fragments plus the real content sentences. We glue a bare enumerator
    fragment onto the sentence that follows it, and drop any fragment left
    with too few content words to be a real (or matchable) sentence.
    """
    text = text.strip()
    if not text:
        return []
    raw_sents = [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]

    merged: list[str] = []
    pending_marker = ""
    for s in raw_sents:
        if _ENUMERATOR_RE.match(s):
            pending_marker = s
            continue
        if pending_marker:
            s = f"{pending_marker} {s}"
            pending_marker = ""
        merged.append(s)

    return [s for s in merged if len(normalize_words(s)) >= MIN_CONTENT_WORDS]


def normalize_words(text: str) -> set[str]:
    words = _WORD_RE.findall(text.lower())
    return {w for w in words if w not in STOPWORDS}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def load_data_metadata(dataset: str) -> dict:
    """question_id -> metadata dict, from the local eval-split data file."""
    import datasets_registry

    cfg = datasets_registry.DATASETS[dataset]
    split_path = REPO_ROOT / "data" / dataset / f"{cfg['eval_split']}.jsonl"
    out = {}
    for line in open(split_path):
        rec = json.loads(line)
        out[rec["id"]] = rec.get("metadata", {})
    return out


def gold_facts_for_hotpot_2wiki(metadata: dict, dataset: str) -> list[dict]:
    """Returns [{hop_index, gold_fact_text}] using supporting_facts + context."""
    sf = metadata.get("supporting_facts")
    ctx = metadata.get("context")
    if not sf or not ctx:
        return []
    titles = sf["title"]
    sent_ids = sf["sent_id"]
    ctx_titles = ctx["title"]
    # hotpotqa uses "sentences", 2wikimultihopqa uses "content"
    ctx_sents = ctx.get("sentences") or ctx.get("content")
    if ctx_sents is None:
        return []

    facts = []
    for hop_index, (title, sent_id) in enumerate(zip(titles, sent_ids)):
        if title not in ctx_titles:
            facts.append({"hop_index": hop_index, "gold_fact_text": None, "title": title})
            continue
        doc_idx = ctx_titles.index(title)
        doc_sents = ctx_sents[doc_idx]
        if sent_id >= len(doc_sents):
            facts.append({"hop_index": hop_index, "gold_fact_text": None, "title": title})
            continue
        facts.append({
            "hop_index": hop_index,
            "gold_fact_text": doc_sents[sent_id].strip(),
            "title": title,
        })
    return facts


def gold_facts_for_musique(metadata: dict) -> list[dict]:
    """Returns [{hop_index, gold_fact_text}] using question_decomposition.
    gold_fact_text combines the sub-answer with its supporting sentence context
    (first sentence of the support paragraph mentioning the answer, else the
    paragraph's first sentence) so matching has enough lexical signal."""
    decomp = metadata.get("question_decomposition")
    if not decomp:
        return []
    facts = []
    for hop_index, hop in enumerate(decomp):
        answer = hop.get("answer", "")
        support = hop.get("support_paragraph") or {}
        para_text = support.get("paragraph_text", "")
        sentences = split_sentences(para_text)
        best_sent = None
        for s in sentences:
            if answer and answer.lower() in s.lower():
                best_sent = s
                break
        if best_sent is None and sentences:
            best_sent = sentences[0]
        gold_fact_text = f"{hop.get('question', '')} {answer} {best_sent or ''}".strip()
        facts.append({"hop_index": hop_index, "gold_fact_text": gold_fact_text, "answer": answer})
    return facts


def align_fact_to_trace(gold_fact_text: str, trace_sentences: list[str]) -> dict:
    """Best-matching trace sentence for one gold fact, via jaccard word overlap."""
    if not gold_fact_text:
        return {"sentence_span": None, "score": 0.0, "matched": False}
    gold_words = normalize_words(gold_fact_text)
    best_score = 0.0
    best_sent = None
    for sent in trace_sentences:
        score = jaccard(gold_words, normalize_words(sent))
        if score > best_score:
            best_score = score
            best_sent = sent
    matched = best_score >= MATCH_THRESHOLD
    return {"sentence_span": best_sent, "score": round(best_score, 3), "matched": matched}


# Lowered from MATCH_THRESHOLD (0.25): a sentence needs only *loosely* resemble
# any gold fact to be disqualified as an off-chain placebo target -- this is
# intentionally more permissive than the accept threshold so we catch near-miss
# on-chain restatements the aligner itself failed to match (e.g. a paraphrase
# that scored 0.20 against its gold fact: too low to accept as *the* on-chain
# sentence, but too relevant to safely corrupt as a placebo).
OFFCHAIN_DISQUALIFY_THRESHOLD = 0.15
OFFCHAIN_MIN_WORDS = 5  # placebo needs enough lexical content to corrupt/paraphrase
OFFCHAIN_QUESTION_MAX_JACCARD = 0.2  # too similar to the question -> likely load-bearing
OFFCHAIN_ANSWER_MAX_JACCARD = 0.2  # too similar to the gold answer -> likely load-bearing


def offchain_purity_score(sent: str, gold_facts: list[dict], question: str, gold_answers: list[str]) -> dict:
    """Positive check that a trace sentence is safe to use as an off-chain
    placebo target -- replaces pure set-negation (was: "not the sentence the
    aligner happened to pick"), which let near-miss on-chain restatements and
    splitter debris leak into off_chain_sentences (see exp20.md's Decision
    items (a)/(b): the dev_1140 Jack Russell sentence case).

    A sentence qualifies only if it: has enough content words, doesn't
    resemble ANY gold fact even loosely, and doesn't resemble the question or
    gold answer (a sentence that echoes the question/answer is plausibly
    load-bearing even if the aligner never picked it for a specific hop).
    """
    sent_words = normalize_words(sent)
    if len(sent_words) < OFFCHAIN_MIN_WORDS:
        return {"qualifies": False, "reason": "too_short", "max_gold_score": None}

    max_gold_score = 0.0
    for gf in gold_facts:
        score = jaccard(sent_words, normalize_words(gf.get("gold_fact_text") or ""))
        max_gold_score = max(max_gold_score, score)
    if max_gold_score >= OFFCHAIN_DISQUALIFY_THRESHOLD:
        return {"qualifies": False, "reason": "resembles_gold_fact", "max_gold_score": round(max_gold_score, 3)}

    q_score = jaccard(sent_words, normalize_words(question or ""))
    if q_score >= OFFCHAIN_QUESTION_MAX_JACCARD:
        return {"qualifies": False, "reason": "resembles_question", "max_gold_score": round(max_gold_score, 3)}

    a_score = max(
        (jaccard(sent_words, normalize_words(a)) for a in (gold_answers or [])),
        default=0.0,
    )
    if a_score >= OFFCHAIN_ANSWER_MAX_JACCARD:
        return {"qualifies": False, "reason": "resembles_answer", "max_gold_score": round(max_gold_score, 3)}

    return {"qualifies": True, "reason": None, "max_gold_score": round(max_gold_score, 3)}


def select_targets():
    data_metadata_cache = {ds: load_data_metadata(ds) for ds in MULTIHOP_DATASETS}

    per_model_count = {}
    candidates = []
    n_unmatched_traces = 0
    n_total_multihop_correct = 0
    n_offchain_considered = 0
    n_offchain_qualified = 0
    n_traces_no_offchain = 0

    with open(MANIFEST_PATH) as f:
        for line in f:
            row = json.loads(line)
            dataset = row["dataset"]
            if dataset not in MULTIHOP_DATASETS:
                continue
            if row["em_correct"] != 1:
                continue
            n_total_multihop_correct += 1

            metadata = data_metadata_cache[dataset].get(row["question_id"], {})
            if dataset == "musique":
                gold_facts = gold_facts_for_musique(metadata)
            else:
                gold_facts = gold_facts_for_hotpot_2wiki(metadata, dataset)
            if not gold_facts:
                continue

            trace_text = row["reasoning"]
            trace_sentences = split_sentences(trace_text)
            if not trace_sentences:
                continue

            # Word-offset of each sentence's start, for the placebo's
            # position-matching (Task 2.5 fix: match cut depth between the
            # on-chain and off-chain prefixes rather than always picking the
            # first qualifying off-chain sentence near the top of the trace).
            prefix_words_by_sentence = {}
            for sent in trace_sentences:
                idx = trace_text.find(sent)
                if idx != -1:
                    prefix_words_by_sentence.setdefault(sent, len(trace_text[:idx].split()))

            hop_facts = []
            any_matched = False
            matched_sentences = set()
            for gf in gold_facts:
                align = align_fact_to_trace(gf.get("gold_fact_text"), trace_sentences)
                hop_facts.append({
                    "hop_index": gf["hop_index"],
                    "sentence_span": align["sentence_span"],
                    "gold_fact_text": gf.get("gold_fact_text"),
                    "match_score": align["score"],
                    "on_chain": True,
                    "matched": align["matched"],
                    "prefix_words": prefix_words_by_sentence.get(align["sentence_span"]) if align["matched"] else None,
                })
                if align["matched"]:
                    any_matched = True
                    matched_sentences.add(align["sentence_span"])

            if not any_matched:
                n_unmatched_traces += 1
                # Keep the trace, flagged, rather than silently dropping (per spec).

            # Off-chain candidate sentences: positively verified as unrelated to
            # any gold fact / the question / the gold answer (not just "the
            # aligner didn't pick this one" -- see offchain_purity_score).
            off_chain_sentences = []
            for sent in trace_sentences:
                if sent in matched_sentences:
                    continue
                n_offchain_considered += 1
                purity = offchain_purity_score(sent, gold_facts, row["question"], row["gold_answer"])
                if purity["qualifies"]:
                    n_offchain_qualified += 1
                    off_chain_sentences.append({
                        "sentence": sent,
                        "prefix_words": prefix_words_by_sentence.get(sent),
                        "max_gold_score": purity["max_gold_score"],
                    })
            if not off_chain_sentences:
                n_traces_no_offchain += 1

            model = row["model"]
            per_model_count.setdefault(model, 0)
            if per_model_count[model] >= TARGET_PER_MODEL and any_matched:
                # still record unmatched-flag traces for auditability, but stop
                # accumulating past-target matched candidates for this model
                pass

            candidates.append({
                "trace_id": row["trace_id"],
                "model": model,
                "dataset": dataset,
                "question_id": row["question_id"],
                "any_hop_matched": any_matched,
                "hop_facts": hop_facts,
                "off_chain_sentences": off_chain_sentences,
            })
            if any_matched:
                per_model_count[model] += 1

    CANDIDATES_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(CANDIDATES_OUT, "w") as f:
        for c in candidates:
            f.write(json.dumps(c) + "\n")

    print(f"[done] {len(candidates)} candidate traces -> {CANDIDATES_OUT}")
    print(f"  multi-hop + em_correct traces considered: {n_total_multihop_correct}")
    print(f"  traces with zero matched hop fact (flagged, kept): {n_unmatched_traces}")
    print(
        f"  off-chain sentences: {n_offchain_qualified}/{n_offchain_considered} passed the "
        f"purity filter (threshold={OFFCHAIN_DISQUALIFY_THRESHOLD}); "
        f"{n_traces_no_offchain} traces have zero qualifying off-chain sentence"
    )
    print("\nUsable (>=1 matched hop) candidates per model:")
    for model, n in sorted(per_model_count.items()):
        print(f"  {model:45s} {n}")


if __name__ == "__main__":
    select_targets()
