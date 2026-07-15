"""One command: generate + evaluate every dataset listed in an experiment's
settings.json.

Usage:
    uv run python scripts/run_all.py --exp exp1
"""
import argparse
import json
import os
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from datasets_registry import DATASETS
from download_data import download_one
from evaluation.eval_dataset import score_file

REPO_ROOT = pathlib.Path(__file__).parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, help="experiment name, e.g. exp1")
    ap.add_argument(
        "--eval-only", action="store_true",
        help="skip download+generate; re-score existing outputs/<dataset>.jsonl "
             "(e.g. after switching judge_model/eval_base_url) without touching "
             "the raw generations",
    )
    args = ap.parse_args()

    exp_dir = REPO_ROOT / "experiment" / args.exp
    settings = json.loads((exp_dir / "settings.json").read_text())
    datasets = settings["datasets"]

    # Let settings.json drive the judge/ex_recall endpoint (e.g. a local MLX
    # server) instead of requiring OPENAI_BASE_URL/JUDGE_MODEL/EXTRACTOR_MODEL
    # to be set by hand. score_file() runs in-process, so env set here reaches
    # evaluation/simpleqa_judge.py and evaluation/ex_recall.py directly.
    if settings.get("eval_base_url"):
        os.environ["OPENAI_BASE_URL"] = settings["eval_base_url"]
        os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
    if settings.get("judge_model"):
        os.environ["JUDGE_MODEL"] = settings["judge_model"]
    if settings.get("extractor_model"):
        os.environ["EXTRACTOR_MODEL"] = settings["extractor_model"]

    summaries = []
    for name in datasets:
        cfg = DATASETS[name]
        if not cfg.get("enabled", True):
            print(f"[skip] {name}: disabled in datasets_registry.py ({cfg.get('metric')})")
            continue

        print(f"\n=== {name} ===")
        outputs_path = exp_dir / "outputs" / f"{name}.jsonl"
        eval_path = exp_dir / "eval" / f"{name}.jsonl"

        if args.eval_only:
            if not outputs_path.exists():
                print(f"[skip] {name}: --eval-only but {outputs_path} missing "
                      f"-- run without --eval-only first to generate it")
                continue
        else:
            download_one(name)

            print(f"[generate] {name}")
            subprocess.run(
                [sys.executable, str(REPO_ROOT / "scripts" / "generate.py"),
                 "--dataset", name, "--exp", args.exp],
                check=True,
            )

        print(f"[eval] {name}")
        summary = score_file(
            outputs_path, eval_path, name,
            judge=settings.get("judge", False),
            ex_recall=settings.get("ex_recall", False),
        )
        print(json.dumps(summary, indent=2))
        summaries.append(summary)

    summary_out = {
        "exp_id": settings.get("exp_id"),
        "model": settings["model"],
        "condition": settings["condition"],
        "n_examples": settings["n_examples"],
        "results": summaries,
    }
    (exp_dir / "summary.json").write_text(json.dumps(summary_out, indent=2))
    print(f"\n[done] summary -> {exp_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
