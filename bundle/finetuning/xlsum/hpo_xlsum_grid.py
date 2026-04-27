#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

try:
    import optuna
except ModuleNotFoundError:  # pragma: no cover - runtime dependency guard
    optuna = None  # type: ignore[assignment]

# Ensure the bundle root is importable regardless of invocation path.
BUNDLE_ROOT = Path(__file__).resolve().parent
if str(BUNDLE_ROOT) not in sys.path:
    sys.path.insert(0, str(BUNDLE_ROOT))

from modeling import SUPPORTED_SIZES, load_json, save_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optuna-backed fixed-grid HPO for XLSum multilingual SFT.")
    p.add_argument("--stage", type=int, required=True, choices=[1, 2])
    p.add_argument("--representation", type=str, required=True, choices=["text", "ipa_stripped", "romanized"])
    p.add_argument("--size", type=str, required=True, choices=list(SUPPORTED_SIZES))

    p.add_argument("--exp_base", type=str, default="/fs/scratch/PAS2836/mugezhang/ml8x3_unified100k")
    p.add_argument("--dataset_repo", type=str, default="mugezhang/xlsum_6lang_multirepr")
    p.add_argument("--mix_mode", type=str, default="balanced", choices=["balanced", "natural"])

    p.add_argument("--context_len", type=int, default=2048)
    p.add_argument("--target_max_tokens", type=int, default=256)
    p.add_argument("--num_beams", type=int, default=4)

    p.add_argument("--per_device_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=16)
    p.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])

    p.add_argument("--objective", type=str, default="rougeL", choices=["rougeL"])
    p.add_argument("--trials", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--lr_values", type=str, default="5e-6,1e-5,2e-5")
    p.add_argument("--wd_values", type=str, default="0.0,0.01,0.05")
    p.add_argument("--warmup_values", type=str, default="0.03,0.06")

    p.add_argument("--stage1_best_json", type=str, default=None, help="Best stage1 config path used to center stage2 grid.")
    p.add_argument("--run_final", action="store_true", help="After stage2, train final best + test evaluation.")

    p.add_argument("--output_root", type=str, default=None)
    p.add_argument("--dataset_cache_dir", type=str, default=None)
    return p.parse_args()


def parse_csv_floats(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def output_dir(args: argparse.Namespace) -> Path:
    root = Path(args.output_root) if args.output_root else Path(args.exp_base) / "xlsum_sft" / "hpo"
    d = root / f"stage{args.stage}" / f"{args.size}_{args.representation}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def choose_two_levels(levels: list[float], center: float) -> list[float]:
    levels = sorted(levels)
    idx = min(range(len(levels)), key=lambda i: abs(levels[i] - center))
    if len(levels) == 1:
        return [levels[0], levels[0]]
    if idx == 0:
        return [levels[0], levels[1]]
    if idx == len(levels) - 1:
        return [levels[-2], levels[-1]]
    # Prefer center + immediate lower for stable narrow grid.
    return [levels[idx - 1], levels[idx]]


def build_stage1_grid(args: argparse.Namespace) -> dict[str, list[float]]:
    return {
        "learning_rate": parse_csv_floats(args.lr_values),
        "weight_decay": parse_csv_floats(args.wd_values),
        "warmup_ratio": parse_csv_floats(args.warmup_values),
    }


def build_stage2_grid(args: argparse.Namespace) -> dict[str, list[float]]:
    stage1_best = None
    if args.stage1_best_json and Path(args.stage1_best_json).exists():
        stage1_best = load_json(args.stage1_best_json)

    if stage1_best is None:
        center_lr, center_wd, center_wr = 1e-5, 0.01, 0.06
    else:
        center_lr = float(stage1_best["best_params"]["learning_rate"])
        center_wd = float(stage1_best["best_params"]["weight_decay"])
        center_wr = float(stage1_best["best_params"]["warmup_ratio"])

    lr_levels = parse_csv_floats(args.lr_values)
    wd_levels = parse_csv_floats(args.wd_values)
    wr_levels = parse_csv_floats(args.warmup_values)

    return {
        "learning_rate": choose_two_levels(lr_levels, center_lr),
        "weight_decay": choose_two_levels(wd_levels, center_wd),
        "warmup_ratio": choose_two_levels(wr_levels, center_wr),
    }


def run_cmd(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")


def build_train_cmd(args: argparse.Namespace, trial_dir: Path, trial_id: int, lr: float, wd: float, wr: float) -> list[str]:
    num_train_epochs = 0.5 if args.stage == 1 else 3.0
    train_sample_frac = 0.3 if args.stage == 1 else 1.0

    return [
        sys.executable,
        "train_gpt_xlsum_sft_accelerate.py",
        "--dataset_repo",
        args.dataset_repo,
        "--representation",
        args.representation,
        "--size",
        args.size,
        "--mix_mode",
        args.mix_mode,
        "--context_len",
        str(args.context_len),
        "--target_max_tokens",
        str(args.target_max_tokens),
        "--exp_base",
        args.exp_base,
        "--build_packs_if_missing",
        "--num_train_epochs",
        str(num_train_epochs),
        "--train_sample_frac",
        str(train_sample_frac),
        "--per_device_batch_size",
        str(args.per_device_batch_size),
        "--gradient_accumulation_steps",
        str(args.gradient_accumulation_steps),
        "--learning_rate",
        str(lr),
        "--weight_decay",
        str(wd),
        "--warmup_ratio",
        str(wr),
        "--mixed_precision",
        args.mixed_precision,
        "--seed",
        str(args.seed + trial_id),
        "--run_name",
        f"trial_{trial_id:03d}",
        "--output_root",
        str(trial_dir.parent),
    ] + (["--dataset_cache_dir", args.dataset_cache_dir] if args.dataset_cache_dir else [])


def build_eval_cmd(args: argparse.Namespace, model_ckpt: Path, trial_dir: Path) -> list[str]:
    return [
        sys.executable,
        "evaluate_xlsum_generation.py",
        "--model_ckpt",
        str(model_ckpt),
        "--representation",
        args.representation,
        "--size",
        args.size,
        "--dataset_repo",
        args.dataset_repo,
        "--split",
        "validation",
        "--exp_base",
        args.exp_base,
        "--context_len",
        str(args.context_len),
        "--target_max_tokens",
        str(args.target_max_tokens),
        "--max_new_tokens",
        str(args.target_max_tokens),
        "--num_beams",
        str(args.num_beams),
        "--output_json",
        str(trial_dir / "validation_metrics.json"),
    ] + (["--dataset_cache_dir", args.dataset_cache_dir] if args.dataset_cache_dir else [])


def main() -> None:
    args = parse_args()
    if optuna is None:
        raise ModuleNotFoundError("Missing dependency `optuna`. Install it in your HPO environment: `pip install optuna`.")
    out = output_dir(args)

    if args.stage == 1:
        grid = build_stage1_grid(args)
    else:
        grid = build_stage2_grid(args)

    sampler = optuna.samplers.GridSampler(grid)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    trial_rows = []

    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_categorical("learning_rate", grid["learning_rate"])
        wd = trial.suggest_categorical("weight_decay", grid["weight_decay"])
        wr = trial.suggest_categorical("warmup_ratio", grid["warmup_ratio"])

        trial_dir = out / f"trial_{trial.number:03d}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        started = time.time()
        status = "ok"
        rougeL = float("-inf")
        bleu = float("nan")
        token_loss = float("nan")
        error_msg = ""

        try:
            train_cmd = build_train_cmd(args, trial_dir=trial_dir, trial_id=trial.number, lr=lr, wd=wd, wr=wr)
            run_cmd(train_cmd, trial_dir / "train.log")

            model_ckpt = trial_dir / "best.pt"
            if not model_ckpt.exists():
                model_ckpt = trial_dir / "last.pt"
            if not model_ckpt.exists():
                raise FileNotFoundError(f"No best.pt/last.pt found in {trial_dir}")

            eval_cmd = build_eval_cmd(args, model_ckpt=model_ckpt, trial_dir=trial_dir)
            run_cmd(eval_cmd, trial_dir / "eval.log")

            metrics = load_json(trial_dir / "validation_metrics.json")
            rougeL = float(metrics["macro"]["rougeL"])
            bleu = float(metrics["macro"]["bleu"])
            token_loss = float(metrics["macro"]["token_loss"])
        except Exception as e:  # noqa: BLE001
            status = "failed"
            error_msg = str(e)
            rougeL = float("-inf")

        elapsed = time.time() - started
        row = {
            "trial": trial.number,
            "stage": args.stage,
            "representation": args.representation,
            "size": args.size,
            "learning_rate": lr,
            "weight_decay": wd,
            "warmup_ratio": wr,
            "objective_rougeL": rougeL,
            "macro_bleu": bleu,
            "macro_token_loss": token_loss,
            "status": status,
            "elapsed_sec": elapsed,
            "error": error_msg,
            "trial_dir": str(trial_dir),
        }
        trial_rows.append(row)
        return rougeL

    # For stage2, enforce exactly requested trials when user asks; otherwise use full grid.
    requested = args.trials if args.stage == 2 else math.prod(len(v) for v in grid.values())
    n_trials = min(requested, math.prod(len(v) for v in grid.values()))
    study.optimize(objective, n_trials=n_trials)

    if not trial_rows:
        raise RuntimeError("No trials executed.")

    ok_rows = [r for r in trial_rows if r["status"] == "ok"]
    if not ok_rows:
        raise RuntimeError("All HPO trials failed.")

    best_row = max(ok_rows, key=lambda r: r["objective_rougeL"])
    best_params = {
        "learning_rate": best_row["learning_rate"],
        "weight_decay": best_row["weight_decay"],
        "warmup_ratio": best_row["warmup_ratio"],
    }

    summary = {
        "stage": args.stage,
        "representation": args.representation,
        "size": args.size,
        "objective": args.objective,
        "grid": grid,
        "num_trials_executed": len(trial_rows),
        "num_trials_success": len(ok_rows),
        "best_params": best_params,
        "best_objective_rougeL": best_row["objective_rougeL"],
        "best_trial": best_row,
        "trials": trial_rows,
    }

    save_json(out / "hpo_summary.json", summary)

    csv_path = out / "hpo_trials.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(trial_rows[0].keys()))
        writer.writeheader()
        writer.writerows(trial_rows)

    save_json(out / "best_config.json", {
        "stage": args.stage,
        "representation": args.representation,
        "size": args.size,
        "best_params": best_params,
        "best_trial_dir": best_row["trial_dir"],
        "best_objective_rougeL": best_row["objective_rougeL"],
    })

    print(json.dumps({
        "summary_json": str(out / "hpo_summary.json"),
        "best_config_json": str(out / "best_config.json"),
        "trials_csv": str(csv_path),
        "best_params": best_params,
        "best_macro_rougeL": best_row["objective_rougeL"],
    }, indent=2))

    if args.stage == 2 and args.run_final:
        final_dir = out / "final_best"
        final_dir.mkdir(parents=True, exist_ok=True)

        final_train_cmd = [
            sys.executable,
            "train_gpt_xlsum_sft_accelerate.py",
            "--dataset_repo",
            args.dataset_repo,
            "--representation",
            args.representation,
            "--size",
            args.size,
            "--mix_mode",
            args.mix_mode,
            "--context_len",
            str(args.context_len),
            "--target_max_tokens",
            str(args.target_max_tokens),
            "--exp_base",
            args.exp_base,
            "--build_packs_if_missing",
            "--num_train_epochs",
            "3.0",
            "--train_sample_frac",
            "1.0",
            "--per_device_batch_size",
            str(args.per_device_batch_size),
            "--gradient_accumulation_steps",
            str(args.gradient_accumulation_steps),
            "--learning_rate",
            str(best_params["learning_rate"]),
            "--weight_decay",
            str(best_params["weight_decay"]),
            "--warmup_ratio",
            str(best_params["warmup_ratio"]),
            "--mixed_precision",
            args.mixed_precision,
            "--seed",
            str(args.seed),
            "--run_name",
            "final_best",
            "--output_root",
            str(out),
        ] + (["--dataset_cache_dir", args.dataset_cache_dir] if args.dataset_cache_dir else [])

        run_cmd(final_train_cmd, final_dir / "train.log")

        final_ckpt = final_dir / "best.pt"
        if not final_ckpt.exists():
            final_ckpt = final_dir / "last.pt"

        final_eval_cmd = [
            sys.executable,
            "evaluate_xlsum_generation.py",
            "--model_ckpt",
            str(final_ckpt),
            "--representation",
            args.representation,
            "--size",
            args.size,
            "--dataset_repo",
            args.dataset_repo,
            "--split",
            "test",
            "--exp_base",
            args.exp_base,
            "--context_len",
            str(args.context_len),
            "--target_max_tokens",
            str(args.target_max_tokens),
            "--max_new_tokens",
            str(args.target_max_tokens),
            "--num_beams",
            str(args.num_beams),
            "--output_json",
            str(final_dir / "test_metrics.json"),
        ] + (["--dataset_cache_dir", args.dataset_cache_dir] if args.dataset_cache_dir else [])
        run_cmd(final_eval_cmd, final_dir / "eval.log")


if __name__ == "__main__":
    main()
