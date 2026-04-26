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


def load_xstorycloze(language: str, representation: str, split: str):
    """Load XStoryCloze from HF. Examples are {'context': 4 sentences, 'choices': [quiz1, quiz2]}."""
    from datasets import load_dataset

    # Note: evaluation split is named "eval", not "test", in this dataset.
    hf_split = {"train": "train", "eval": "eval",
                 "test": "eval", "validation": "eval"}[split]

    ds = load_dataset(
        "mugezhang/xstorycloze_eval_multirepr",
        name=language,
        split=hf_split,
        cache_dir=HF_CACHE_DIR,
    )

    sfx = {"text": "", "romanized": "_romanized", "ipa": "_ipa_stripped"}[representation]
    s_cols = [f"input_sentence_{i}{sfx}" for i in range(1, 5)]
    q_cols = [f"sentence_quiz{i}{sfx}" for i in range(1, 3)]

    examples, labels = [], []
    for row in ds:
        context = " ".join(row[c].strip() for c in s_cols)
        choices = [row[q_cols[0]].strip(), row[q_cols[1]].strip()]
        examples.append({"context": context, "choices": choices})
        labels.append(int(row["answer_right_ending"]) - 1)  # 1-indexed → 0-indexed
    return examples, labels


def load_xcopa(language: str, representation: str, split: str):
    """Load XCopa from HF. Examples are {'context': premise+connector, 'choices': [c1, c2]}."""
    from datasets import load_dataset

    hf_split = {"validation": "validation", "test": "test", "dev": "validation"}[split]

    ds = load_dataset(
        "mugezhang/xcopa_eval_multirepr",
        name=language,
        split=hf_split,
        cache_dir=HF_CACHE_DIR,
    )

    sfx = {"text": "", "romanized": "_romanized", "ipa": "_ipa_stripped"}[representation]
    p_col  = f"premise{sfx}"
    c1_col = f"choice1{sfx}"
    c2_col = f"choice2{sfx}"
    # connectors must match the surrounding representation
    connectors = {
        "text":      {"cause": "because", "effect": "so"},
        "romanized": {"cause": "because", "effect": "so"},
        "ipa":       {"cause": "bɪkʌz",   "effect": "soʊ"},
    }
    connector = connectors[representation]

    examples, labels = [], []
    for row in ds:
        ctx = f"{row[p_col].strip()} {connector[row['question']]}"
        choices = [row[c1_col].strip(), row[c2_col].strip()]
        examples.append({"context": ctx, "choices": choices})
        labels.append(int(row["label"]))  # already 0-indexed
    return examples, labels


DATASET_LOADERS = {
    "sst2":         lambda lang, rep, split: load_sst2(split, rep),
    "xnli":         load_xnli,
    "pawsx":        load_pawsx,
    "xstorycloze":  load_xstorycloze,
    "xcopa":        load_xcopa,
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
    if task_format not in ("classification", "pair", "mcq"):
        raise NotImplementedError(
            f"Task format '{task_format}' is not yet implemented in icl_eval.py. "
            "Supported: 'classification', 'pair', 'mcq'."
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

    # MCQ has per-example choices, no fixed label_dict.
    if task_format == "mcq":
        label_dict = None
    else:
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
) -> list[tuple[float, int]]:
    """Return (sum_log_prob, n_label_tokens) per label word given the context.

    Caller picks the reduction (sum or per-token avg).
    Empty label_ids → (-inf, 0).
    """
    context_ids: list[int] = tokenizer.encode(context)
    out_pairs: list[tuple[float, int]] = []

    for word in label_words:
        full_ids: list[int] = tokenizer.encode(context + " " + word)
        label_ids = full_ids[len(context_ids):]
        if not label_ids:
            out_pairs.append((float("-inf"), 0))
            continue

        input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
        attn_mask = torch.ones_like(input_ids)

        out = model(input_ids=input_ids, attention_mask=attn_mask)
        logits = out.logits[0].float()
        log_probs = F.log_softmax(logits, dim=-1)

        # logits[i] predicts token i+1, so label token j is at offset+j.
        offset = len(context_ids) - 1
        total = sum(log_probs[offset + j, lid].item() for j, lid in enumerate(label_ids))
        out_pairs.append((total, len(label_ids)))

    return out_pairs


def _reduce_scores(pairs: list[tuple[float, int]], mode: str) -> list[float]:
    """Reduce (sum_log_prob, n_tokens) pairs to scalar scores."""
    if mode == "avg":
        return [s / n if n > 0 else float("-inf") for s, n in pairs]
    if mode == "sum":
        return [s if n > 0 else float("-inf") for s, n in pairs]
    raise ValueError(f"unknown reduce mode: {mode}")


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def _print_confusion_matrix(
    label: str,
    preds: np.ndarray,
    truth: np.ndarray,
    n_classes: int,
    class_names: list[str] = None,
) -> None:
    """Debug helper: print a confusion matrix. Rows = true, cols = predicted."""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(truth, preds):
        cm[t, p] += 1
    width = max(6, max(len(n) for n in (class_names or [])) + 2 if class_names else 6)
    header = "true\\pred ".rjust(width)
    if class_names:
        header += " ".join(n.rjust(width) for n in class_names)
    else:
        header += " ".join(str(i).rjust(width) for i in range(n_classes))
    print(f"  [debug-cm] {label}")
    print(f"    {header}")
    for i in range(n_classes):
        row_name = (class_names[i] if class_names else str(i)).rjust(width)
        row = " ".join(str(cm[i, j]).rjust(width) for j in range(n_classes))
        marg = cm[i].sum()
        per_class = (cm[i, i] / marg) if marg > 0 else float("nan")
        print(f"    {row_name} {row}   (n={marg:>4d}, recall={per_class:.3f})")
    pred_marg = cm.sum(axis=0)
    print(f"    pred-marg ".rjust(width + 4) +
          " ".join(str(m).rjust(width) for m in pred_marg))


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
    debug_cm: bool = False,
) -> float:
    """Run k-shot ICL eval. Returns the primary-mode argmax accuracy.

    primary mode: avg for classification/pair, sum for mcq.
    """
    rng = np.random.default_rng(seed)

    # frozen subsample (same examples across k values)
    if subsample and subsample < len(test_examples):
        idx = rng.choice(len(test_examples), size=subsample, replace=False)
        test_examples = [test_examples[i] for i in idx]
        test_labels   = [test_labels[i]   for i in idx]

    is_mcq = cfg["task_format"] == "mcq"
    if not is_mcq:
        label_words = [cfg["label_dict"][i][0] for i in sorted(cfg["label_dict"])]
        n_choices = len(label_words)
    else:
        label_words = None
        n_choices = 2  # both MCQ tasks are binary

    primary_mode = "sum" if is_mcq else "avg"

    n = len(test_examples)
    sum_scores = np.zeros((n, n_choices))
    avg_scores = np.zeros((n, n_choices))

    for i, example in enumerate(test_examples):
        if is_mcq:
            context = cfg["prompt_prefix"] + example["context"]
            choices = example["choices"]
        else:
            if k > 0:
                demo_idx   = rng.choice(len(train_examples), size=k, replace=False)
                k_examples = [train_examples[j] for j in demo_idx]
                k_labs     = [train_labels[j]   for j in demo_idx]
            else:
                k_examples, k_labs = [], []
            context = construct_prompt(cfg, k_examples, k_labs, example)
            choices = label_words

        pairs = score_labels_nll(model, tokenizer, context, choices, device)
        sum_scores[i] = _reduce_scores(pairs, "sum")
        avg_scores[i] = _reduce_scores(pairs, "avg")

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{n}]", flush=True)

    labels_arr = np.array(test_labels)
    sum_preds  = np.argmax(sum_scores, axis=1)
    avg_preds  = np.argmax(avg_scores, axis=1)
    sum_acc    = float(np.mean(sum_preds == labels_arr))
    avg_acc    = float(np.mean(avg_preds == labels_arr))
    flip_rate  = float(np.mean(sum_preds != avg_preds))

    print(f"  reduce=sum acc={sum_acc:.4f}   reduce=avg acc={avg_acc:.4f}   "
          f"sum/avg disagree on {flip_rate:.2%} of examples")

    if debug_cm:
        if is_mcq:
            cm_names = ["choice0", "choice1"]
        else:
            cm_names = [cfg["label_dict"][i][0] for i in sorted(cfg["label_dict"])]
        _print_confusion_matrix(
            f"raw / reduce={primary_mode}",
            sum_preds if primary_mode == "sum" else avg_preds,
            labels_arr, n_choices, cm_names,
        )

    return sum_acc if primary_mode == "sum" else avg_acc


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
    parser.add_argument("--debug-cm",       action="store_true",
                        help="Print confusion matrix per run (debug only).")
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

    is_mcq = cfg["task_format"] == "mcq"
    if not is_mcq:
        label_words = [cfg["label_dict"][i][0] for i in sorted(cfg["label_dict"])]
        print("Label words  :", {i: w for i, w in enumerate(label_words)})
    else:
        label_words = None
        print("Label words  : (per-example choices — MCQ task)")

    print("\nLoading model and tokenizer...")
    tokenizer, model, device, _ = load_model_and_tokenizer(
        spec.checkpoint_path, spec.tokenizer_path, "bf16", False, None,
        max_seq_len=spec.max_seq_len, window_size_tokens=spec.window_size_tokens,
    )
    model.eval()

    if not is_mcq:
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
            debug_cm=args.debug_cm,
        )
        print(f"{k:>4}  {acc:>10.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()