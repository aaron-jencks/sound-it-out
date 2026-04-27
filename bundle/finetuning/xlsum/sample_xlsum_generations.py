#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tokenizers import Tokenizer

_BUNDLE_ROOT = Path(__file__).resolve().parent
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
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sample actual XLSum generations from a training checkpoint.")
    p.add_argument("--model_ckpt", type=str, required=True, help="Path to SFT checkpoint (best.pt or last.pt)")
    p.add_argument("--representation", type=str, required=True, choices=["text", "ipa_stripped", "romanized"])
    p.add_argument("--size", type=str, required=True, choices=list(SUPPORTED_SIZES))
    p.add_argument("--dataset_repo", type=str, default="mugezhang/xlsum_6lang_multirepr")
    p.add_argument("--dataset_cache_dir", type=str, default=None)
    p.add_argument("--split", type=str, default="test", choices=["validation", "test"])
    p.add_argument("--exp_base", type=str, default="/fs/scratch/PAS2836/mugezhang/ml8x3_unified100k")
    p.add_argument("--tokenizer_json", type=str, default=None)
    p.add_argument("--context_len", type=int, default=2048)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--num_beams", type=int, default=4)
    p.add_argument("--repetition_penalty", type=float, default=1.2)
    p.add_argument("--no_repeat_ngram_size", type=int, default=3)
    p.add_argument("--length_penalty", type=float, default=1.0)
    p.add_argument("--samples_per_lang", type=int, default=1)
    p.add_argument("--languages", type=str, default="", help="Optional comma-separated subset of languages.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--output_jsonl", type=str, required=True)
    return p.parse_args()


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
                next_score = score + float(lp)
                next_seq = seq + [int(tid)]
                candidates.append((_rank_score(next_score, next_seq), next_score, next_seq))

        candidates.sort(key=lambda x: x[0], reverse=True)
        beams = [(raw_score, seq) for _, raw_score, seq in candidates[:num_beams]]
        if ended == len(beams):
            break

    best = beams[0][1]
    return best[len(input_ids) :]


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.device != "auto":
        device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")

    bundle = load_model_from_training_checkpoint(args.model_ckpt, map_location="cpu")
    model = bundle.model.to(device)
    model.eval()

    tokenizer_json = Path(args.tokenizer_json) if args.tokenizer_json else None
    if tokenizer_json is None:
        cfg_path = bundle.config.get("tokenizer_json")
        if cfg_path:
            tokenizer_json = Path(cfg_path)
    if tokenizer_json is None or not tokenizer_json.is_file():
        tokenizer_json = default_tokenizer_path(args.exp_base, args.representation)
    tokenizer = Tokenizer.from_file(str(tokenizer_json))
    eos_id = get_eos_token_id(tokenizer)

    source_field, target_field = get_rep_fields(args.representation)
    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    languages = [x.strip() for x in args.languages.split(",") if x.strip()] if args.languages else list(LANGUAGES)

    with out_path.open("w", encoding="utf-8") as f:
        for lang in languages:
            ds = load_dataset(args.dataset_repo, lang, split=args.split, cache_dir=args.dataset_cache_dir)
            ds = ds.shuffle(seed=args.seed).select(range(min(args.samples_per_lang, len(ds))))

            for idx, ex in enumerate(ds):
                source = str(ex[source_field])
                reference = str(ex[target_field])
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

                prediction = tokenizer.decode(gen_ids)
                row = {
                    "language": lang,
                    "split": args.split,
                    "sample_index": idx,
                    "representation": args.representation,
                    "model_ckpt": args.model_ckpt,
                    "source": source,
                    "reference": reference,
                    "prediction": prediction,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                print(f"[{lang} #{idx}] generated {len(prediction)} chars", flush=True)

    print(f"Saved samples to {out_path}")


if __name__ == "__main__":
    main()
