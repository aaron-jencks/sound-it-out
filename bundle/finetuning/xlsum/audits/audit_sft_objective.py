#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_from_disk


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit XLSUM SFT label alignment for autoregressive training.")
    p.add_argument(
        "--pack_dir",
        type=str,
        default="/fs/scratch/PAS2836/mugezhang/ml8x3_unified100k/xlsum_sft/packs/text_balanced_ctx2048_tgt256",
    )
    p.add_argument("--split", type=str, default="validation", choices=["train", "validation", "test"])
    p.add_argument("--index", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ds = load_from_disk(args.pack_dir)[args.split]
    ex = ds[args.index]

    input_ids = ex["input_ids"]
    labels = ex["labels"]
    prompt_len = int(ex["prompt_len"])
    target_len = int(ex["target_len"])

    labeled_positions = [i for i, label in enumerate(labels) if label != -100]
    shift_labels = labels[1:]
    shifted_positions = [i for i, label in enumerate(shift_labels) if label != -100]

    if not labeled_positions:
        raise RuntimeError("Example has no supervised tokens.")
    if labeled_positions[0] != prompt_len:
        raise RuntimeError(f"Expected first supervised position at prompt_len={prompt_len}, got {labeled_positions[0]}")
    if shifted_positions[0] != prompt_len - 1:
        raise RuntimeError(
            f"Expected first shifted supervised position at prompt_len-1={prompt_len - 1}, got {shifted_positions[0]}"
        )
    if len(shifted_positions) != target_len:
        raise RuntimeError(f"Expected {target_len} shifted supervised tokens, got {len(shifted_positions)}")

    report = {
        "pack_dir": str(Path(args.pack_dir)),
        "split": args.split,
        "index": args.index,
        "prompt_len": prompt_len,
        "target_len": target_len,
        "broken_objective_same_token_matches": sum(
            1 for i in labeled_positions if input_ids[i] == labels[i]
        ),
        "shifted_num_supervised_tokens": len(shifted_positions),
        "first_supervised_position": labeled_positions[0],
        "first_shifted_supervised_position": shifted_positions[0],
        "boundary_window": {
            "input_ids": input_ids[max(0, prompt_len - 2) : prompt_len + min(target_len, 4)],
            "labels": labels[max(0, prompt_len - 2) : prompt_len + min(target_len, 4)],
            "shift_input_ids": input_ids[max(0, prompt_len - 2) : prompt_len + min(target_len, 3)],
            "shift_labels": shift_labels[max(0, prompt_len - 2) : prompt_len + min(target_len, 3)],
        },
        "explanation": (
            "With the broken objective, supervised positions compare logits at index i against labels[i], "
            "which equals input_ids[i] for target tokens. With the corrected objective, logits at index i "
            "predict labels[i+1], so the first summary token is predicted from the last prompt position."
        ),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
