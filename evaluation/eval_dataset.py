"""Score one dataset's generations: outputs/<dataset>.jsonl -> eval/<dataset>.jsonl
plus a summary dict {dataset, n, em, cover_em, avg_output_len}.
"""
import argparse
import json
import pathlib

from evaluation.metrics import cover_em, exact_match


def score_file(outputs_path: pathlib.Path, eval_path: pathlib.Path, dataset: str) -> dict:
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    em_sum = 0
    cover_sum = 0
    len_sum = 0
    with open(outputs_path) as fin, open(eval_path, "w") as fout:
        for line in fin:
            rec = json.loads(line)
            pred = rec["predicted_answer"]
            golds = rec["golden_answers"]
            em = exact_match(pred, golds)
            cov = cover_em(pred, golds)
            out_len = len(rec.get("raw_output", pred).split())
            fout.write(json.dumps({
                "id": rec["id"],
                "predicted_answer": pred,
                "golden_answers": golds,
                "em": em,
                "cover_em": cov,
            }) + "\n")
            n += 1
            em_sum += em
            cover_sum += cov
            len_sum += out_len

    summary = {
        "dataset": dataset,
        "n": n,
        "em": em_sum / n if n else 0.0,
        "cover_em": cover_sum / n if n else 0.0,
        "avg_output_len": len_sum / n if n else 0.0,
    }
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--outputs", required=True, help="path to outputs/<dataset>.jsonl")
    ap.add_argument("--eval-out", required=True, help="path to eval/<dataset>.jsonl")
    args = ap.parse_args()

    summary = score_file(pathlib.Path(args.outputs), pathlib.Path(args.eval_out), args.dataset)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
