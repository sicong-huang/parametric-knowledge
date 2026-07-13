# Stratified 1-hop QA

1-hop bio QA lives under `<run_dir>/stratified_1hop/` (parallel to
`stratified_2hop/`).

Unlike 2-hop, there are **no path geometry patterns** — each question is one
edge `(person --[relation]--> target)`. Sampling is over unique-answer
`(source, relation)` groups; train/test split is stratified by `relation_type`.

## Layout

```
stratified_1hop/
  qa_1_hop.jsonl
  qa_1_hop_direct.jsonl
  qa_1_hop_reasoning.jsonl
  qa_1_hop_sentence.jsonl
  qa_split/
    qa_1_hop_{train,test}.jsonl
    qa_1_hop_direct_{train,test}.jsonl
    qa_1_hop_reasoning_{train,test}.jsonl
    split_report.json
```

Legacy files may still exist under `run5/qa/` (**old 2-hop only**). Prefer this
folder for all 1-hop work.

## Commands

```bash
export QA_BASE_URL=http://localhost:8001/v1

# Generate questions
uv run syndata_archive/create_qa.py 1_hop \
  --dir syndata_archive/data/bio/run5 --sample 20000 --max-workers 32

# Styled answers
uv run syndata_archive/create_qa.py 1_hop_full_answer \
  --dir syndata_archive/data/bio/run5 --style direct
uv run syndata_archive/create_qa.py 1_hop_full_answer \
  --dir syndata_archive/data/bio/run5 --style reasoning --max-workers 32

# Edge-disjoint 70/30 train/test (same logic as 2-hop)
uv run syndata_archive/stratified_1hop/split_qa.py \
  --dir syndata_archive/data/bio/run5 --seed 42
```

## vs 2-hop

| | 1-hop (`stratified_1hop/`) | 2-hop (`stratified_2hop/`) |
|---|---|---|
| Structure | 1 edge | 2 edges, 3 patterns |
| Balance | by `relation_type` | 10k per pattern |
| Split | edge-disjoint 70/30 | edge-disjoint 70/30 |
| Typical drops | ~0 (unique triplets) | ~4–5% |
