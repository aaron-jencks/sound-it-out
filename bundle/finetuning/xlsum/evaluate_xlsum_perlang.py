#!/usr/bin/env python
"""Evaluate multilingual XLSum generation using per-language best checkpoints.

Instead of --model_ckpt (single checkpoint for all languages), this script accepts
--ckpt_dir containing best_{lang}.pt files produced by train_gpt_xlsum_sft_perlang.py.
Each language is evaluated with its own best checkpoint. Checkpoints are loaded and
unloaded one at a time to avoid holding all 6 in VRAM simultaneously.

Output format is identical to evaluate_xlsum_generation.py (per_language, macro, weighted).
"""
from __future__ import annotations

import argparse
import csv
import json
import time
import unicodedata
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parent
if sys.path and Path(sys.path[0]).resolve() == REPO_ROOT:
    sys.path.pop(0)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import sacrebleu
except ModuleNotFoundError:
    sacrebleu = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Unicode-aware ROUGE tokenizer (XL-Sum compatible) — unchanged from original.
# ---------------------------------------------------------------------------
from rouge_score import rouge_scorer as _rs, tokenizers as _rt

def _is_ws(c: str) -> bool:
    return c in (" ", "\t", "\n", "\r") or unicodedata.category(c) == "Zs"

def _is_ctrl(c: str) -> bool:
    if c in ("\t", "\n", "\r"):
        return False
    return unicodedata.category(c).startswith("C")

def _is_punc(c: str) -> bool:
    cp = ord(c)
    if (33 <= cp <= 47) or (58 <= cp <= 64) or (91 <= cp <= 96) or (123 <= cp <= 126):
        return True
    return unicodedata.category(c).startswith("P")


class _XlsumTokenizer(_rt.Tokenizer):
    def tokenize(self, text: str) -> list[str]:
        cleaned: list[str] = []
        for ch in text:
            if ord(ch) == 0 or ord(ch) == 0xFFFD or _is_ctrl(ch):
                continue
            cleaned.append(" " if _is_ws(ch) else ch)
        text = "".join(cleaned)

        tokens: list[str] = []
        for word in text.split():
            cur: list[str] = []
            for ch in word:
                if _is_punc(ch):
                    if cur:
                        tokens.append("".join(cur).lower())
                        cur = []
                else:
                    cur.append(ch)
            if cur:
                tokens.append("".join(cur).lower())

        return [t for t in tokens if t]


_XLSUM_SCORER = _rs.RougeScorer(
    ["rouge1", "rouge2", "rougeL"],
    use_stemmer=False,
    tokenizer=_XlsumTokenizer(),
)


def _compute_rouge(predictions: list[str], references: list[str]) -> dict:
    if not predictions:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    scores = [_XLSUM_SCORER.score(ref, pred) for pred, ref in zip(predictions, references)]
    return {
        "rouge1": sum(s["rouge1"].fmeasure for s in scores) / len(scores),
        "rouge2": sum(s["rouge2"].fmeasure for s in scores) / len(scores),
        "rougeL": sum(s["rougeL"].fmeasure for s in scores) / len(scores),
    }


import sys
from pathlib import Path as _Path
_BUNDLE_ROOT = _Path(__file__).resolve().parent
if str(_BUNDLE_ROOT) not in sys.path:
    sys.path.insert(0, str(_BUNDLE_ROOT))

from modeling import (
    LANGUAGES,
    SUPPORTED_SIZES,
    default_tokenizer_path,
    get_eos_token_id,
    get_rep_fields,
    load_model_from_training_checkpoint,
    prompt_template,
    save_json,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate XLSum generation using per-language best checkpoints."
    )
    # Primary input: directory containing best_{lang}.pt files.
    p.add_argument(
        "--ckpt_dir",
        type=str,
        required=True,
        help="Directory containing best_{lang}.pt files from train_gpt_xlsum_sft_perlang.py",
    )
    p.add_argument("--representation", type=str, required=True, choices=["text", "ipa_stripped", "romanized"])
    p.add_argument("--size", type=str, required=True, choices=list(SUPPORTED_SIZES))
    p.add_argument("--dataset_repo", type=str, default="mugezhang/xlsum_6lang_multirepr")
    p.add_argument("--dataset_cache_dir", type=str, default=None)
    p.add_argument("--split", type=str, default="test", choices=["validation", "test"])

    p.add_argument("--exp_base", type=str, default="/fs/scratch/PAS2836/mugezhang/ml8x3_unified100k")
    p.add_argument("--tokenizer_json", type=str, default=None)

    p.add_argument("--context_len", type=int, default=2048)
    p.add_argument("--target_max_tokens", type=int, default=256)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--num_beams", type=int, default=4)
    p.add_argument("--repetition_penalty", type=float, default=1.2)
    p.add_argument("--no_repeat_ngram_size", type=int, default=3)
    p.add_argument("--length_penalty", type=float, default=1.0)
    p.add_argument("--max_eval_samples_per_lang", type=int, default=0)

    p.add_argument("--batch_size_token_loss", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")

    p.add_argument("--output_json", type=str, required=True)
    p.add_argument("--output_csv", type=str, default=None)
    p.add_argument(
        "--output_preds_dir",
        type=str,
        default=None,
        help="If set, write per-language JSONL files (LANG.jsonl) with "
             "{language, source, reference, prediction} records into this directory.",
    )
    return p.parse_args()


def pad_batch(batch: list[dict], pad_token_id: int) -> dict[str, torch.Tensor]:
    bsz = len(batch)
    max_len = max(len(x["input_ids"]) for x in batch)

    input_ids = torch.full((bsz, max_len), pad_token_id, dtype=torch.long)
    labels = torch.full((bsz, max_len), -100, dtype=torch.long)

    for i, ex in enumerate(batch):
        n = len(ex["input_ids"])
        input_ids[i, :n] = torch.tensor(ex["input_ids"], dtype=torch.long)
        labels[i, :n] = torch.tensor(ex["labels"], dtype=torch.long)

    return {"input_ids": input_ids, "labels": labels}


def encode_teacher_forced(
    tokenizer: Tokenizer,
    lang: str,
    source: str,
    target: str,
    context_len: int,
    target_max_tokens: int,
    eos_id: int,
):
    prompt_ids = tokenizer.encode(prompt_template(lang, source)).ids
    target_ids = tokenizer.encode(target).ids[:target_max_tokens]
    if eos_id is not None:
        target_ids = target_ids + [int(eos_id)]
    if not target_ids:
        return None

    max_prompt = context_len - len(target_ids)
    if max_prompt <= 0:
        return None
    prompt_ids = prompt_ids[:max_prompt]

    input_ids = prompt_ids + target_ids
    labels = ([-100] * len(prompt_ids)) + target_ids
    return {"input_ids": input_ids, "labels": labels}


def beam_search_generate(
    model,
    input_ids: list[int],
    max_new_tokens: int,
    num_beams: int,
    eos_id: int,
    device: torch.device,
    valid_vocab_size: int,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    length_penalty: float,
) -> list[int]:
    beams: list[tuple[float, list[int]]] = [(0.0, list(input_ids))]

    def _apply_repetition_penalty(logits: torch.Tensor, generated_tokens: list[int]) -> torch.Tensor:
        if repetition_penalty <= 1.0 or not generated_tokens:
            return logits
        logits = logits.clone()
        for tid in set(generated_tokens):
            if 0 <= tid < logits.size(0):
                if logits[tid] < 0:
                    logits[tid] = logits[tid] * repetition_penalty
                else:
                    logits[tid] = logits[tid] / repetition_penalty
        return logits

    def _banned_tokens(generated_tokens: list[int]) -> set[int]:
        n = no_repeat_ngram_size
        if n <= 0:
            return set()
        if n == 1:
            return set(generated_tokens)
        if len(generated_tokens) < n - 1:
            return set()
        prefix_to_next: dict[tuple[int, ...], set[int]] = {}
        for i in range(len(generated_tokens) - n + 1):
            prefix = tuple(generated_tokens[i : i + n - 1])
            next_token = generated_tokens[i + n - 1]
            prefix_to_next.setdefault(prefix, set()).add(next_token)
        current_prefix = tuple(generated_tokens[-(n - 1) :])
        return prefix_to_next.get(current_prefix, set())

    def _rank_score(raw_score: float, seq: list[int]) -> float:
        gen_len = max(1, len(seq) - len(input_ids))
        if length_penalty == 0:
            return raw_score
        return raw_score / (gen_len ** length_penalty)

    for _ in range(max_new_tokens):
        candidates: list[tuple[float, float, list[int]]] = []
        ended = 0

        for score, seq in beams:
            if eos_id is not None and len(seq) > len(input_ids) and seq[-1] == eos_id:
                candidates.append((_rank_score(score, seq), score, seq))
                ended += 1
                continue

            model_input = torch.tensor(seq[-model.max_seq_len :], dtype=torch.long, device=device).unsqueeze(0)
            with torch.no_grad():
                logits = model(model_input)["logits"][0, -1]
            if logits.size(0) > valid_vocab_size:
                logits = logits.clone()
                logits[valid_vocab_size:] = -float("inf")

            generated_tokens = seq[len(input_ids) :]
            logits = _apply_repetition_penalty(logits, generated_tokens)
            banned = _banned_tokens(generated_tokens)
            if banned:
                logits = logits.clone()
                for tid in banned:
                    if 0 <= tid < logits.size(0):
                        logits[tid] = -float("inf")

            log_probs = F.log_softmax(logits, dim=-1)
            topk_log_probs, topk_ids = torch.topk(log_probs, k=num_beams, dim=-1)
            for lp, tid in zip(topk_log_probs.tolist(), topk_ids.tolist()):
                candidates.append((_rank_score(score + float(lp), seq + [int(tid)]), score + float(lp), seq + [int(tid)]))

        candidates.sort(key=lambda x: x[0], reverse=True)
        beams = [(raw_score, seq) for _, raw_score, seq in candidates[:num_beams]]

        if ended == len(beams):
            break

    return beams[0][1][len(input_ids) :]


def compute_token_loss(
    model, encoded_examples: list[dict], batch_size: int, pad_token_id: int, device: torch.device
) -> float:
    if not encoded_examples:
        return float("nan")

    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for i in range(0, len(encoded_examples), batch_size):
        batch = pad_batch(encoded_examples[i : i + batch_size], pad_token_id=pad_token_id)
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        with torch.no_grad():
            out = model(input_ids, labels=labels)
        total_loss += float(out["sum_loss"].item())
        total_tokens += int(out["num_tokens"].item())

    return total_loss / max(total_tokens, 1)


def resolve_tokenizer(args: argparse.Namespace, ckpt_dir: Path) -> Path:
    """Resolve tokenizer path from args, then checkpoint config, then default."""
    if args.tokenizer_json:
        return Path(args.tokenizer_json)
    # Try reading from the first available best_{lang}.pt config.
    for lang in LANGUAGES:
        ckpt_path = ckpt_dir / f"best_{lang}.pt"
        if ckpt_path.exists():
            try:
                ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
                cfg_path = ckpt.get("config", {}).get("tokenizer_json")
                if cfg_path and Path(cfg_path).is_file():
                    return Path(cfg_path)
            except Exception:
                pass
            break
    return default_tokenizer_path(args.exp_base, args.representation)


def main() -> None:
    args = parse_args()
    if sacrebleu is None:
        raise ModuleNotFoundError("Missing dependency `sacrebleu`. Install with: `pip install sacrebleu`.")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(args.ckpt_dir)

    # Verify all per-language checkpoints exist before starting.
    missing = [lang for lang in LANGUAGES if not (ckpt_dir / f"best_{lang}.pt").exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing per-language checkpoints in {ckpt_dir}: {missing}\n"
            f"Expected files: {[f'best_{l}.pt' for l in missing]}"
        )

    tokenizer_json = resolve_tokenizer(args, ckpt_dir)
    tokenizer = Tokenizer.from_file(str(tokenizer_json))
    eos_id = get_eos_token_id(tokenizer)
    source_field, target_field = get_rep_fields(args.representation)

    print(f"Tokenizer: {tokenizer_json}", flush=True)
    print(f"Checkpoint dir: {ckpt_dir}", flush=True)
    print(f"Device: {device}", flush=True)

    lang_rows = []

    for lang in LANGUAGES:
        ckpt_path = ckpt_dir / f"best_{lang}.pt"
        print(f"\n[{lang}] Loading {ckpt_path.name} ...")

        bundle = load_model_from_training_checkpoint(str(ckpt_path), map_location="cpu")
        model = bundle.model.to(device)
        model.eval()
        print(f"[{lang}] Model loaded.", flush=True)

        ds = load_dataset(args.dataset_repo, lang, split=args.split, cache_dir=args.dataset_cache_dir)
        if args.max_eval_samples_per_lang > 0:
            ds = ds.select(range(min(len(ds), args.max_eval_samples_per_lang)))
        n_total = len(ds)
        print(f"[{lang}] Generating {n_total} samples (split={args.split}) ...", flush=True)

        preds = []
        refs = []
        sources = []
        encoded_for_loss = []
        _lang_t0 = time.time()

        for _i, ex in enumerate(ds):
            source = str(ex[source_field])
            target = str(ex[target_field])

            prompt_ids = tokenizer.encode(prompt_template(lang, source)).ids
            max_prompt = max(1, args.context_len - args.max_new_tokens)
            prompt_ids = prompt_ids[:max_prompt]

            gen_ids = beam_search_generate(
                model=model,
                input_ids=prompt_ids,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
                eos_id=eos_id,
                device=device,
                valid_vocab_size=bundle.config["embed_vocab_size"],
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                length_penalty=args.length_penalty,
            )

            if eos_id is not None and eos_id in gen_ids:
                gen_ids = gen_ids[: gen_ids.index(eos_id)]

            pred_text = tokenizer.decode(gen_ids)
            preds.append(pred_text)
            refs.append(target)
            sources.append(source)

            encoded = encode_teacher_forced(
                tokenizer=tokenizer,
                lang=lang,
                source=source,
                target=target,
                context_len=args.context_len,
                target_max_tokens=args.target_max_tokens,
                eos_id=eos_id,
            )
            if encoded is not None:
                encoded_for_loss.append(encoded)

            if (_i + 1) % 50 == 0 or (_i + 1) == n_total:
                elapsed = time.time() - _lang_t0
                rate = (_i + 1) / elapsed if elapsed > 0 else 0
                eta = (n_total - _i - 1) / rate if rate > 0 else 0
                print(
                    f"[{lang}] {_i+1}/{n_total}  {rate:.1f} samples/s  "
                    f"elapsed={elapsed:.0f}s  eta={eta:.0f}s",
                    flush=True,
                )

        rouge_scores = _compute_rouge(preds, refs)
        bleu = sacrebleu.corpus_bleu(preds, [refs]).score
        token_loss = compute_token_loss(
            model=model,
            encoded_examples=encoded_for_loss,
            batch_size=args.batch_size_token_loss,
            pad_token_id=0,
            device=device,
        )

        lang_row = {
            "language": lang,
            "split": args.split,
            "num_examples": len(refs),
            "rouge1": float(rouge_scores["rouge1"]),
            "rouge2": float(rouge_scores["rouge2"]),
            "rougeL": float(rouge_scores["rougeL"]),
            "bleu": float(bleu),
            "token_loss": float(token_loss),
            "checkpoint": str(ckpt_path),
        }
        lang_rows.append(lang_row)
        print(
            f"[{lang}] rougeL={lang_row['rougeL']:.4f} bleu={lang_row['bleu']:.2f} "
            f"token_loss={lang_row['token_loss']:.4f}  "
            f"total_time={time.time()-_lang_t0:.0f}s",
            flush=True,
        )

        # Write per-language predictions JSONL (flushed before model unload so
        # partial results survive a job kill on later languages).
        if args.output_preds_dir:
            preds_dir = Path(args.output_preds_dir)
            preds_dir.mkdir(parents=True, exist_ok=True)
            preds_path = preds_dir / f"{lang}.jsonl"
            with preds_path.open("w", encoding="utf-8") as _f:
                for _src, _ref, _pred in zip(sources, refs, preds):
                    _f.write(json.dumps(
                        {"language": lang, "source": _src, "reference": _ref, "prediction": _pred},
                        ensure_ascii=False,
                    ) + "\n")
            print(f"[{lang}] Predictions saved -> {preds_path}", flush=True)

        # Unload model immediately to free VRAM before loading the next language.
        del model, bundle
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Aggregate metrics
    # -------------------------------------------------------------------------
    macro = {
        "rouge1": sum(r["rouge1"] for r in lang_rows) / len(lang_rows),
        "rouge2": sum(r["rouge2"] for r in lang_rows) / len(lang_rows),
        "rougeL": sum(r["rougeL"] for r in lang_rows) / len(lang_rows),
        "bleu": sum(r["bleu"] for r in lang_rows) / len(lang_rows),
        "token_loss": sum(r["token_loss"] for r in lang_rows) / len(lang_rows),
    }

    total_n = sum(r["num_examples"] for r in lang_rows)
    weighted = {
        k: sum(r[k] * r["num_examples"] for r in lang_rows) / max(total_n, 1)
        for k in ("rouge1", "rouge2", "rougeL", "bleu", "token_loss")
    }

    output = {
        "ckpt_dir": str(ckpt_dir),
        "representation": args.representation,
        "size": args.size,
        "split": args.split,
        "tokenizer_json": str(tokenizer_json),
        "num_beams": args.num_beams,
        "max_new_tokens": args.max_new_tokens,
        "max_eval_samples_per_lang": args.max_eval_samples_per_lang,
        "repetition_penalty": args.repetition_penalty,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
        "length_penalty": args.length_penalty,
        "per_language": lang_rows,
        "macro": macro,
        "weighted": weighted,
    }

    save_json(args.output_json, output)
    print("\nSaved JSON:", args.output_json, flush=True)

    csv_path = args.output_csv if args.output_csv else str(Path(args.output_json).with_suffix(".csv"))
    fieldnames = ["language", "split", "num_examples", "rouge1", "rouge2", "rougeL", "bleu", "token_loss", "checkpoint"]
    with Path(csv_path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in lang_rows:
            writer.writerow(row)
        writer.writerow({
            "language": "macro", "split": args.split, "num_examples": total_n,
            **macro, "checkpoint": "",
        })
        writer.writerow({
            "language": "weighted", "split": args.split, "num_examples": total_n,
            **weighted, "checkpoint": "",
        })

    print("Saved CSV:", csv_path, flush=True)
    print(json.dumps({"macro": macro, "weighted": weighted}, indent=2), flush=True)


if __name__ == "__main__":
    main()
