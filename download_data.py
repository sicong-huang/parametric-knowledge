"""Pull FlashRAG eval-split jsonl files into data/<dataset>/<split>.jsonl.

Usage:
    uv run python download_data.py                 # all enabled datasets
    uv run python download_data.py --dataset nq     # one dataset
"""
import argparse
import pathlib
import urllib.request

from datasets_registry import DATASETS, enabled_datasets, flashrag_url

DATA_DIR = pathlib.Path(__file__).parent / "data"


def download_one(dataset: str, force: bool = False) -> pathlib.Path:
    cfg = DATASETS[dataset]
    out_path = DATA_DIR / dataset / f"{cfg['eval_split']}.jsonl"
    if out_path.exists() and not force:
        print(f"[skip] {dataset}: {out_path} already present")
        return out_path
    url = flashrag_url(dataset)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[fetch] {dataset}: {url} -> {out_path}")
    urllib.request.urlretrieve(url, out_path)
    n_lines = sum(1 for _ in open(out_path))
    print(f"[done] {dataset}: {n_lines} lines")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None, help="single dataset name; default = all enabled")
    ap.add_argument("--force", action="store_true", help="re-download even if cached")
    args = ap.parse_args()

    targets = [args.dataset] if args.dataset else enabled_datasets()
    for name in targets:
        if DATASETS[name]["dir"] is None:
            print(f"[note] {name}: not on FlashRAG hub, needs manual source (see datasets_registry.py)")
            continue
        download_one(name, force=args.force)


if __name__ == "__main__":
    main()
