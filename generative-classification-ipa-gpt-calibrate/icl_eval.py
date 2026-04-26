#!/usr/bin/env python3
"""
icl_eval.py — k-shot ICL evaluator for IPA-GPT models.

Usage:
    python icl_eval.py --dataset sst2 --language en --representation text
    python icl_eval.py --dataset xnli --language en --representation ipa --k 0
"""

from __future__ import annotations

import argparse
import sys
import os
import numpy as np
import torch
import torch.nn.functional as F
import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

WORKSPACE = os.path.dirname(os.path.realpath(__file__))
CONFIG_PATH = os.path.join(WORKSPACE, "prompt_configs.yaml")

IPA_GPT_DIR = "/users/PAS2836/mugezhang/generative-classification-ipa-gpt"
sys.path.insert(0, IPA_GPT_DIR)

from ipa_gpt_runtime import load_model_and_tokenizer  # noqa: E402
from ipa_gpt_registry import MODEL_SPECS             # noqa: E402

HF_CACHE_DIR = "/fs/scratch/PAS2836/mugezhang/generative-classification-ipa-gpt-artifacts/cache"

# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

DATA_DIR = os.path.join(WORKSPACE, "data")


def load_sst2(split: str = "test", representation: str = "text"):
    """Load SST-2 from local stsa.binary{,.romanized,.ipa_stripped} files."""
    base = {
        "train": "stsa.binary.train",
        "test":  "stsa.binary.test",
        "dev":   "stsa.binary.dev",
    }[split]
    suffix = {
        "text":      "",
        "ipa":       ".ipa_stripped",
        "romanized": ".romanized",
    }[representation]
    path = os.path.join(DATA_DIR, "sst2", base + suffix)
    sentences, labels = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                labels.append(int(line[0]))
                sentences.append(line[2:].strip())
    return sentences, labels


def load_xnli(language: str, representation: str, split: str):
    """Load XNLI from HF. Returns examples as {'text1': premise, 'text2': hypothesis}."""
    from datasets import load_dataset

    hf_split = {"train": "train", "validation": "validation",
                 "test": "test", "dev": "validation"}[split]

    ds = load_dataset(
        "mugezhang/xnli_eval_multirepr",
        name=language,
        split=hf_split,
        cache_dir=HF_CACHE_DIR,
    )

    col_map = {
        "text":      ("premise",            "hypothesis"),
        "romanized": ("premise_romanized",   "hypothesis_romanized"),
        "ipa":       ("premise_ipa_stripped","hypothesis_ipa_stripped"),
    }
    p_col, h_col = col_map[representation]

    examples = [{"text1": row[p_col].strip(), "text2": row[h_col].strip()} for row in ds]
    labels   = [int(row["label"]) for row in ds]
    return examples, labels


def load_pawsx(language: str, representation: str, split: str):
    """Load PAWS-X from HF. Returns examples as {'text1': sentence1, 'text2': sentence2}."""
    from datasets import load_dataset

    hf_split = {"train": "train", "validation": "validation",
                 "test": "test", "dev": "validation"}[split]

    ds = load_dataset(
        "mugezhang/pawsx_eval_multirepr",
        name=language,
        split=hf_split,
        cache_dir=HF_CACHE_DIR,
    )

    col_map = {
        "text":      ("sentence1",            "sentence2"),
        "romanized": ("sentence1_romanized",   "sentence2_romanized"),
        "ipa":       ("sentence1_ipa_stripped","sentence2_ipa_stripped"),
    }
    s1_col, s2_col = col_map[representation]

    examples = [{"text1": row[s1_col].strip(), "text2": row[s2_col].strip()} for row in ds]
    labels   = [int(row["label"]) for row in ds]
    return examples, labels


DATASET_LOADERS = {
    "sst2":  lambda lang, rep, split: load_sst2(split, rep),
    "xnli":  load_xnli,
    "pawsx": load_pawsx,
}


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_prompt_config(dataset: str, language: str, representation: str) -> dict:
    """Load the prompt template + label dict for one (dataset, language, repr) cell."""
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    if dataset not in cfg["datasets"]:
        raise KeyError(f"Dataset '{dataset}' not found in {CONFIG_PATH}")
    ds_cfg = cfg["datasets"][dataset]

    task_format = ds_cfg.get("task_format", "classification")
    if task_format not in ("classification", "pair"):
        raise NotImplementedError(
            f"Task format '{task_format}' is not yet implemented in icl_eval.py. "
            "Supported: 'classification', 'pair'."
        )

    langs = ds_cfg.get("languages", {})
    if language not in langs:
        raise KeyError(f"Language '{language}' not in config for dataset '{dataset}'")
    reps = langs[language].get("representations", {})
    if representation not in reps:
        raise KeyError(
            f"Representation '{representation}' not in config for "
            f"dataset='{dataset}', language='{language}'"
        )
    rep_cfg = reps[representation]

    raw_label_dict = rep_cfg["label_dict"]
    label_dict = {int(k): v if isinstance(v, list) else [v]
                  for k, v in raw_label_dict.items()}

    # rep-level q_prefix/a_prefix override the dataset-level default if set
    q_prefix = rep_cfg.get("q_prefix") or ds_cfg.get("q_prefix")
    a_prefix = rep_cfg.get("a_prefix") or ds_cfg.get("a_prefix")

    return {
        "prompt_prefix": ds_cfg.get("prompt_prefix", ""),
        "q_prefix":      q_prefix,
        "a_prefix":      a_prefix,
        "label_dict":    label_dict,
        "task_format":   task_format,
    }


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _format_q(cfg: dict, example) -> str:
    """Format the question portion of the prompt for one example."""
    if isinstance(example, dict):
        return cfg["q_prefix"].format(**example)
    else:
        return cfg["q_prefix"] + example


def construct_prompt(
    cfg: dict,
    train_examples: list,
    train_labels: list[int],
    test_example,
) -> str:
    """Build a k-shot prompt for classification or pair tasks."""
    a = cfg["a_prefix"]
    assert a[-1] == " ", f"a_prefix must end with a space, got: {repr(a)}"

    prompt = cfg["prompt_prefix"]
    is_pair = isinstance(test_example, dict)

    for example, lab in zip(train_examples, train_labels):
        label_str = cfg["label_dict"][lab][0]
        q_text = _format_q(cfg, example)
        if is_pair:
            prompt += q_text + a + label_str + "\n\n"
        else:
            prompt += q_text + "\n" + a + label_str + "\n\n"

    q_text = _format_q(cfg, test_example)
    if is_pair:
        prompt += q_text + a[:-1]
    else:
        prompt += q_text + "\n" + a[:-1]

    return prompt


# ---------------------------------------------------------------------------
# NLL scoring
# ---------------------------------------------------------------------------

@torch.no_grad()
def score_labels_nll(
    model,
    tokenizer,
    context: str,
    label_words: list[str],
    device: torch.device,
) -> list[float]:
    """Per-token-average log-prob for each label word given context."""
    context_ids: list[int] = tokenizer.encode(context)
    scores = []

    for word in label_words:
        full_ids: list[int] = tokenizer.encode(context + " " + word)
        label_ids = full_ids[len(context_ids):]
        if not label_ids:
            scores.append(float("-inf"))
            continue

        input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
        attn_mask = torch.ones_like(input_ids)

        out = model(input_ids=input_ids, attention_mask=attn_mask)
        logits = out.logits[0].float()
        log_probs = F.log_softmax(logits, dim=-1)

        # logits[i] predicts token i+1, so label token j is at offset+j.
        offset = len(context_ids) - 1
        total = sum(log_probs[offset + j, lid].item() for j, lid in enumerate(label_ids))
        scores.append(total / len(label_ids))

    return scores


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def run_eval(
    model,
    tokenizer,
    train_examples: list,
    train_labels: list[int],
    test_examples: list,
    test_labels: list[int],
    cfg: dict,
    k: int,
    seed: int,
    subsample: int,
    device: torch.device,
) -> float:
    """Run k-shot ICL eval. Returns argmax accuracy."""
    rng = np.random.default_rng(seed)

    if subsample and subsample < len(test_examples):
        idx = rng.choice(len(test_examples), size=subsample, replace=False)
        test_examples = [test_examples[i] for i in idx]
        test_labels   = [test_labels[i]   for i in idx]

    label_words = [cfg["label_dict"][i][0] for i in sorted(cfg["label_dict"])]
    n = len(test_examples)
    all_scores = np.zeros((n, len(label_words)))

    for i, example in enumerate(test_examples):
        if k > 0:
            demo_idx   = rng.choice(len(train_examples), size=k, replace=False)
            k_examples = [train_examples[j] for j in demo_idx]
            k_labs     = [train_labels[j]   for j in demo_idx]
        else:
            k_examples, k_labs = [], []
        context = construct_prompt(cfg, k_examples, k_labs, example)
        all_scores[i] = score_labels_nll(model, tokenizer, context, label_words, device)

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{n}]", flush=True)

    preds = np.argmax(all_scores, axis=1)
    return float(np.mean(preds == np.array(test_labels)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="IPA-GPT ICL baseline evaluator")
    parser.add_argument("--dataset",        required=True, help="e.g. sst2, xnli")
    parser.add_argument("--language",       required=True, help="e.g. en, es, ru")
    parser.add_argument("--representation", required=True,
                        choices=["text", "romanized", "ipa"])
    parser.add_argument("--size",           default="medium",
                        choices=["small", "medium", "large"])
    parser.add_argument("--k",              nargs="+", type=int, default=[0])
    parser.add_argument("--subsample",      type=int, default=500)
    parser.add_argument("--seed",           type=int, default=42)
    parser.add_argument("--split",          default="test")
    parser.add_argument("--train_split",    default="train")
    args = parser.parse_args()

    model_key = f"{args.size}_{args.representation}"
    if model_key not in MODEL_SPECS:
        raise KeyError(
            f"Model key '{model_key}' not found in MODEL_SPECS. "
            f"Available: {list(MODEL_SPECS)}"
        )
    spec = MODEL_SPECS[model_key]

    print(f"Dataset      : {args.dataset} / {args.language} / {args.representation}")
    print(f"Model        : {model_key}  (size={args.size})")
    print(f"Checkpoint   : {spec.checkpoint_path}")
    print(f"k values     : {args.k}")
    print(f"Test examples: {args.subsample or 'all'}")
    print(f"Seed         : {args.seed}\n")

    cfg = load_prompt_config(args.dataset, args.language, args.representation)
    print("Task format  :", cfg["task_format"])

    label_words = [cfg["label_dict"][i][0] for i in sorted(cfg["label_dict"])]
    print("Label words  :", {i: w for i, w in enumerate(label_words)})

    print("\nLoading model and tokenizer...")
    tokenizer, model, device, _ = load_model_and_tokenizer(
        spec.checkpoint_path, spec.tokenizer_path, "bf16", False, None,
        max_seq_len=spec.max_seq_len, window_size_tokens=spec.window_size_tokens,
    )
    model.eval()

    print("\nLabel tokenization:")
    for word in label_words:
        ids = tokenizer.encode(" " + word)
        pieces = [tokenizer.decode([i]) for i in ids]
        flag = " ← multi-tok" if len(ids) > 1 else ""
        print(f"  ' {word}': {len(ids)} token(s) → {pieces}{flag}")

    print(f"\nLoading {args.dataset} ({args.language}, {args.representation})...")
    if args.dataset not in DATASET_LOADERS:
        raise NotImplementedError(
            f"No data loader for '{args.dataset}'. "
            f"Add it to DATASET_LOADERS in icl_eval.py."
        )
    loader = DATASET_LOADERS[args.dataset]

    max_k = max(args.k)
    if max_k > 0:
        train_examples, train_labels = loader(args.language, args.representation, args.train_split)
        print(f"  Train: {len(train_examples)}")
    else:
        train_examples, train_labels = [], []

    test_examples, test_labels = loader(args.language, args.representation, args.split)
    n_eval = min(args.subsample, len(test_examples)) if args.subsample else len(test_examples)
    print(f"  Test : {len(test_examples)} → evaluating {n_eval}")

    print(f"\n{'k':>4}  {'Accuracy':>10}")
    print("-" * 18)
    for k in args.k:
        print(f"\nRunning k={k}...", flush=True)
        acc = run_eval(
            model, tokenizer,
            train_examples, train_labels,
            test_examples,  test_labels,
            cfg=cfg, k=k, seed=args.seed,
            subsample=args.subsample, device=device,
        )
        print(f"{k:>4}  {acc:>10.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()