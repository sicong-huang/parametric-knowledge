"""Score one dataset's generations: outputs/<dataset>.jsonl -> eval/<dataset>.jsonl
plus a summary dict {dataset, n, em, cover_em, avg_output_len}. With --judge and/or
--ex-recall, adds co-primary semantic metrics (judge_accuracy, ex_recall) alongside EM
on the EM-metric datasets, to guard against EM's verbosity/extraction confound between
reasoning-ON and direct conditions (see eval_metrics_research.md).
"""
import argparse
import json
import pathlib

from datasets_registry import DATASETS
from evaluation.ex_recall import score_ex_recall
from evaluation.metrics import cover_em, exact_match
from evaluation.parse import extract_think
from evaluation.simpleqa_judge import score_judge, score_simpleqa


def _score_file_em(
    outputs_path: pathlib.Path,
    eval_path: pathlib.Path,
    dataset: str,
    judge: bool = False,
    ex_recall: bool = False,
) -> dict:
    n = 0
    em_sum = 0
    cover_sum = 0
    len_sum = 0
    records = []
    per_record = []
    with open(outputs_path) as fin:
        for line in fin:
            rec = json.loads(line)
            pred = rec["predicted_answer"]
            golds = rec["golden_answers"]
            raw_output = rec.get("raw_output", pred)
            em = exact_match(pred, golds)
            cov = cover_em(pred, golds)
            out_len = len(raw_output.split())
            per_record.append({
                "id": rec["id"],
                "predicted_answer": pred,
                "golden_answers": golds,
                "em": em,
                "cover_em": cov,
                "raw_output": raw_output,
                "reasoning_trace": extract_think(raw_output),
            })
            records.append(rec)
            n += 1
            em_sum += em
            cover_sum += cov
            len_sum += out_len

    summary = {
        "dataset": dataset,
        "metric": "em",
        "n": n,
        "em": em_sum / n if n else 0.0,
        "cover_em": cover_sum / n if n else 0.0,
        "avg_output_len": len_sum / n if n else 0.0,
    }

    if judge:
        judge_result = score_judge(records)
        judge_grades = judge_result.pop("grades")
        for out_rec, g in zip(per_record, judge_grades):
            out_rec["judge_grade"] = g
        summary["judge_accuracy"] = judge_result["judge_accuracy"]
        summary["judge_n_correct"] = judge_result["n_correct"]
        summary["judge_n_incorrect"] = judge_result["n_incorrect"]
        summary["judge_n_not_attempted"] = judge_result["n_not_attempted"]

    if ex_recall:
        ex_result = score_ex_recall(records)
        refined = ex_result.pop("refined_answers")
        recalled = ex_result.pop("recalled")
        skipped = ex_result.pop("skipped")
        for out_rec, r, rec_flag, skip_flag in zip(per_record, refined, recalled, skipped):
            out_rec["refined_answer"] = r
            out_rec["recalled"] = rec_flag
            out_rec["skipped"] = skip_flag
        summary["ex_recall"] = ex_result["ex_recall"]

    with open(eval_path, "w") as fout:
        for out_rec in per_record:
            fout.write(json.dumps(out_rec) + "\n")

    return summary


def _score_file_llm_judge(outputs_path: pathlib.Path, eval_path: pathlib.Path, dataset: str) -> dict:
    records = [json.loads(line) for line in open(outputs_path)]
    result = score_simpleqa(records)
    grades = result.pop("grades")

    with open(eval_path, "w") as fout:
        for rec, grade in zip(records, grades):
            raw_output = rec.get("raw_output", rec["predicted_answer"])
            fout.write(json.dumps({
                "id": rec["id"],
                "predicted_answer": rec["predicted_answer"],
                "golden_answers": rec["golden_answers"],
                "grade": grade,
                "raw_output": raw_output,
                "reasoning_trace": extract_think(raw_output),
            }) + "\n")

    return {"dataset": dataset, "metric": "llm_judge", **result}


def score_file(
    outputs_path: pathlib.Path,
    eval_path: pathlib.Path,
    dataset: str,
    judge: bool = False,
    ex_recall: bool = False,
) -> dict:
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    metric = DATASETS[dataset]["metric"]
    if metric == "llm_judge":
        return _score_file_llm_judge(outputs_path, eval_path, dataset)
    return _score_file_em(outputs_path, eval_path, dataset, judge=judge, ex_recall=ex_recall)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--outputs", required=True, help="path to outputs/<dataset>.jsonl")
    ap.add_argument("--eval-out", required=True, help="path to eval/<dataset>.jsonl")
    ap.add_argument("--judge", action="store_true", help="also grade with the LLM judge (co-primary, EM datasets only)")
    ap.add_argument("--ex-recall", action="store_true", dest="ex_recall", help="also score Ex-Recall (co-primary, EM datasets only)")
    args = ap.parse_args()

    summary = score_file(
        pathlib.Path(args.outputs), pathlib.Path(args.eval_out), args.dataset,
        judge=args.judge, ex_recall=args.ex_recall,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
