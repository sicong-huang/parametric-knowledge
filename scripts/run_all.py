"""One command: generate + evaluate every dataset listed in an experiment's
settings.json.

Usage:
    uv run python scripts/run_all.py --exp exp1
"""
import argparse
import json
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
    args = ap.parse_args()

    exp_dir = REPO_ROOT / "experiment" / args.exp
    settings = json.loads((exp_dir / "settings.json").read_text())
    datasets = settings["datasets"]

    summaries = []
    for name in datasets:
        cfg = DATASETS[name]
        if not cfg.get("enabled", True):
            print(f"[skip] {name}: disabled in datasets_registry.py ({cfg.get('metric')})")
            continue

        print(f"\n=== {name} ===")
        download_one(name)

        print(f"[generate] {name}")
        subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "generate.py"),
             "--dataset", name, "--exp", args.exp],
            check=True,
        )

        print(f"[eval] {name}")
        outputs_path = exp_dir / "outputs" / f"{name}.jsonl"
        eval_path = exp_dir / "eval" / f"{name}.jsonl"
        summary = score_file(outputs_path, eval_path, name)
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
