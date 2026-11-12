#!/usr/bin/env python3
"""Build the 4 nested SFT training sets + their prime-rl config overlays.

Configs (cumulative fraction of training *questions*; each set contains the previous one):
    c1_full    100%
    c2_10pct   10%
    c3_19pct   c2 + 10% of the remaining 90%           = 19.0%
    c4_43pct   c3 + 30% of the remaining 81%           = 43.3%

Sampling unit is the question (qaid), not the row: the direct and reasoning files share
the same qaids, so a sampled question contributes BOTH its direct and its reasoning row
(exactly what the full set does). This makes direct:reasoning exactly 1:1 in every
config. Sampling is stratified by (hop, pattern), so 1-hop:2-hop and the pattern mix
(P->NP, P,P->NP, ...) match the full set (~47:53 hop split). Nesting comes from one
fixed permutation per stratum cut at the cumulative fractions.

Writes, per config:
    <out-dir>/<cfg>/train.jsonl     prime-rl rows: prompt, completion, enable_thinking (+ meta)
    <out-dir>/<cfg>/qaids.txt       the sampled question ids (reproducibility / analysis)
    <out-dir>/manifest.json         counts per (hop, style) and per (hop, pattern), token + step stats
    <configs-dir>/<cfg>.toml        overlay: data.name, max_steps, warmup, run.name

max_steps: prime-rl has no `epochs` knob — its SFT dataset loops forever and packs
samples into rows of seq_len tokens, with [data].batch_size counting ROWS. So each
row is tokenized with prime-rl's own renderer (identical to training) and the greedy
packer is simulated to get rows/epoch, then max_steps = ceil(epochs * rows / batch).

Run inside the prime-rl environment (needs `renderers` + `transformers`):
    uv run --no-sync --project ~/prime-rl python train_scripts/train_prime/prepare_data.py \
        --qa-dir qa/train --tokenizer checkpoints/pretrain-qwen3.5-0.8b
"""

import argparse
import json
import math
import random
import tomllib
from collections import Counter, defaultdict
from pathlib import Path

HOPS = ("1_hop", "2_hop")
STYLES = ("direct", "reasoning")
HERE = Path(__file__).resolve().parent


def cumulative_fractions() -> dict[str, float]:
    f2 = 0.10
    f3 = f2 + 0.10 * (1 - f2)  # + 10% of the remaining 90%
    f4 = f3 + 0.30 * (1 - f3)  # + 30% of what remains after c3
    return {"c2_10pct": f2, "c3_19pct": f3, "c4_43pct": f4, "c1_full": 1.0}


def load_qa(qa_dir: Path) -> dict[tuple[str, str], dict[str, dict]]:
    """{(hop, style): {qaid: record}}, checking direct/reasoning cover the same qaids."""
    data = {}
    for hop in HOPS:
        for style in STYLES:
            src = qa_dir / f"qa_{hop}_{style}_train.jsonl"
            if not src.is_file():
                raise SystemExit(f"Missing QA file: {src}")
            with open(src) as f:
                records = [json.loads(line) for line in f]
            by_id = {r["qaid"]: r for r in records}
            if len(by_id) != len(records):
                raise SystemExit(f"Duplicate qaids in {src}")
            data[(hop, style)] = by_id
        if data[(hop, "direct")].keys() != data[(hop, "reasoning")].keys():
            raise SystemExit(f"{hop}: direct and reasoning files have different qaids; pairing assumption broken")
    return data


def sample_qaids(data, seed: int) -> dict[str, set[str]]:
    """Nested, (hop, pattern)-stratified qaid subsets for each config."""
    strata = defaultdict(list)
    for hop in HOPS:
        for qaid, rec in data[(hop, "direct")].items():
            strata[(hop, rec["pattern"])].append(qaid)

    rng = random.Random(seed)
    selected = {cfg: set() for cfg in cumulative_fractions()}
    for key in sorted(strata):
        ids = sorted(strata[key])
        rng.shuffle(ids)
        for cfg, frac in cumulative_fractions().items():
            selected[cfg].update(ids[: round(frac * len(ids))])
    return selected


def to_row(rec: dict, hop: str, style: str) -> dict:
    return {
        "prompt": [{"role": "user", "content": rec["question"]}],
        # full_answer is the answer (direct) or "<think>\n...\n</think>\nanswer" (reasoning);
        # the qwen3.5 renderer turns the inline think block into the same tokens as the
        # HF chat template (verified token-for-token), like TRL's prompt/completion rows.
        "completion": [{"role": "assistant", "content": rec["full_answer"]}],
        "enable_thinking": style == "reasoning",
        "qaid": rec["qaid"],
        "n_hop": hop,
        "style": style,
        "pattern": rec["pattern"],
    }


def token_lengths(rows: list[dict], tokenizer_path: str, renderer_name: str) -> list[int]:
    """Per-row training length exactly as prime-rl's SFTDataset._process computes it."""
    from pydantic import TypeAdapter
    from renderers import RendererConfig, merge_chat_template_kwargs
    from renderers.base import build_training_sample, create_renderer
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_path)
    base_cfg = TypeAdapter(RendererConfig).validate_python({"name": renderer_name})
    renderers = {}
    lengths = []
    for row in rows:
        et = row["enable_thinking"]
        if et not in renderers:
            renderers[et] = create_renderer(tok, merge_chat_template_kwargs(base_cfg, {"enable_thinking": et}))
        sample = build_training_sample(
            renderers[et],
            row["prompt"] + row["completion"],
            role_to_mask=None,
            tools=[],
            content_sft_roles=None,
            ensure_final_stop=True,
        )
        lengths.append(len(sample.token_ids) - 1)  # causal shift drops one token
    return lengths


def packed_rows(lengths: list[int], row_len: int, seed: int) -> int:
    """Rows/epoch from prime-rl's greedy CatDataset packer (shuffled order)."""
    order = list(range(len(lengths)))
    random.Random(seed).shuffle(order)
    rows, cur = 0, 0
    for i in order:
        n = lengths[i]
        if cur > 0 and cur + n > row_len:
            rows, cur = rows + 1, 0
        cur += n
        if cur >= row_len:
            rows, cur = rows + 1, 0
    return rows + (cur > 0)


def write_overlay(path: Path, cfg: str, data_dir: Path, frac: float, n_rows: int, max_steps: int,
                  warmup: int, epochs: float, samples_per_step: float, run_prefix: str) -> None:
    path.write_text(
        f"# GENERATED by prepare_data.py — do not edit by hand; rerun it instead.\n"
        f"# {cfg}: {frac:.1%} of training questions -> {n_rows} rows (direct + reasoning),\n"
        f"# {epochs:g} epochs ~= {max_steps} steps at ~{samples_per_step:.1f} samples/step.\n"
        f"max_steps = {max_steps}\n\n"
        f"[run]\nname = \"{run_prefix}-{cfg}\"\n\n"
        f"[data]\nname = \"{data_dir}\"\n\n"
        f"[scheduler]\nwarmup_steps = {warmup}\n"
    )


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--qa-dir", type=Path, default=Path("qa/train"))
    p.add_argument("--out-dir", type=Path, default=Path("data/sft_prime"))
    p.add_argument("--configs-dir", type=Path, default=HERE / "configs")
    p.add_argument("--base-config", type=Path, default=HERE / "configs" / "base.toml")
    p.add_argument("--tokenizer", default=None,
                   help="model/tokenizer path for token counts (default: [model].name in base.toml)")
    p.add_argument("--epochs", type=float, default=2.0, help="epochs per config (TRL recipe used 2)")
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--run-prefix", default="qwen3.5-0.8b", help="run.name = <prefix>-<cfg>")
    args = p.parse_args()

    base = tomllib.loads(args.base_config.read_text())
    row_len = base["data"]["seq_len"] * base["data"].get("micro_batch_size", 1)
    batch_rows = base["data"]["batch_size"]
    tokenizer = args.tokenizer or base["model"]["name"]
    renderer_name = base["renderer"]["name"]

    data = load_qa(args.qa_dir)
    selected = sample_qaids(data, args.seed)

    # Tokenize every row once (all configs are subsets of c1_full).
    all_rows = {(hop, style, qaid): to_row(rec, hop, style)
                for (hop, style), recs in data.items() for qaid, rec in recs.items()}
    keys = list(all_rows)
    print(f"Tokenizing {len(keys)} rows with renderer={renderer_name!r}, tokenizer={tokenizer}")
    lens = dict(zip(keys, token_lengths([all_rows[k] for k in keys], tokenizer, renderer_name)))
    if max(lens.values()) > row_len:
        raise SystemExit(f"Longest sample ({max(lens.values())} tok) exceeds seq_len {row_len}; raise [data].seq_len")

    manifest = {"qa_dir": str(args.qa_dir), "seed": args.seed, "epochs": args.epochs, "row_len": row_len,
                "batch_rows": batch_rows, "tokenizer": tokenizer, "renderer": renderer_name, "configs": {}}
    pattern_totals = Counter(f"{k[0]}/{all_rows[k]['pattern']}" for k in keys if k[1] == "direct")
    for cfg, frac in cumulative_fractions().items():
        cfg_keys = [k for k in keys if k[2] in selected[cfg]]
        rows = [all_rows[k] for k in cfg_keys]
        random.Random(args.seed).shuffle(rows)  # prime-rl reshuffles per epoch too; this just makes files readable

        out = args.out_dir / cfg
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "train.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        (out / "qaids.txt").write_text("\n".join(sorted(selected[cfg])) + "\n")

        cfg_lens = [lens[k] for k in cfg_keys]
        rows_per_epoch = packed_rows(cfg_lens, row_len, args.seed)
        max_steps = math.ceil(args.epochs * rows_per_epoch / batch_rows)
        warmup = max(1, round(args.warmup_ratio * max_steps))
        samples_per_step = args.epochs * len(rows) / max_steps
        write_overlay(args.configs_dir / f"{cfg}.toml", cfg, out.resolve(), frac, len(rows), max_steps,
                      warmup, args.epochs, samples_per_step, args.run_prefix)

        counts = Counter(f"{k[0]}/{k[1]}" for k in cfg_keys)
        # Per (hop, pattern) stratum: questions selected vs available. Each question has one
        # direct and one reasoning row, so count questions on the direct rows.
        pat = Counter(f"{k[0]}/{all_rows[k]['pattern']}" for k in cfg_keys if k[1] == "direct")
        by_pattern = {p: {"questions": n, "available": pattern_totals[p], "fraction": round(n / pattern_totals[p], 4)}
                      for p, n in sorted(pat.items())}
        manifest["configs"][cfg] = {
            "fraction_of_questions": round(frac, 4),
            "questions": len(selected[cfg]),
            "rows": len(rows),
            "rows_by_hop_style": dict(sorted(counts.items())),
            "share_by_hop_style": {k: round(v / len(rows), 4) for k, v in sorted(counts.items())},
            "questions_by_hop_pattern": by_pattern,
            "tokens": sum(cfg_lens),
            "packed_rows_per_epoch": rows_per_epoch,
            "max_steps": max_steps,
            "warmup_steps": warmup,
            "samples_per_step": round(samples_per_step, 1),
        }
        print(f"{cfg:9s} {frac:6.1%}  questions={len(selected[cfg]):6d}  rows={len(rows):6d}  "
              f"tokens={sum(cfg_lens):8d}  max_steps={max_steps:5d}  (~{samples_per_step:.1f} samples/step)  "
              f"{dict(sorted(counts.items()))}")

    # Nesting sanity check: c2 ⊂ c3 ⊂ c4 ⊂ c1.
    chain = ["c2_10pct", "c3_19pct", "c4_43pct", "c1_full"]
    assert all(selected[a] <= selected[b] for a, b in zip(chain, chain[1:])), "subsets are not nested"

    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {args.out_dir}/manifest.json and overlays in {args.configs_dir}")


if __name__ == "__main__":
    main()
