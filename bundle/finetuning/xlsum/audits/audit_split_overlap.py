#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from datasets import load_dataset

_BUNDLE_ROOT = Path(__file__).resolve().parent.parent
if str(_BUNDLE_ROOT) not in sys.path:
    sys.path.insert(0, str(_BUNDLE_ROOT))

from modeling import LANGUAGES, get_rep_fields


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit exact overlap between XLSUM train/validation/test splits.")
    p.add_argument("--dataset_repo", type=str, default="mugezhang/xlsum_6lang_multirepr")
    p.add_argument("--dataset_cache_dir", type=str, default=None)
    p.add_argument("--representation", type=str, default="text", choices=["text", "ipa_stripped", "romanized"])
    p.add_argument("--output_json", type=str, default=None)
    return p.parse_args()


def hash_example(language: str, source: str, target: str) -> str:
    payload = f"{language}\t{source}\t{target}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    args = parse_args()
    source_field, target_field = get_rep_fields(args.representation)

    per_language: dict[str, dict[str, int]] = {}
    totals = {
        "train_examples": 0,
        "validation_examples": 0,
        "test_examples": 0,
        "train_validation_overlap": 0,
        "train_test_overlap": 0,
        "validation_test_overlap": 0,
    }

    for language in LANGUAGES:
        train = load_dataset(args.dataset_repo, language, split="train", cache_dir=args.dataset_cache_dir)
        validation = load_dataset(args.dataset_repo, language, split="validation", cache_dir=args.dataset_cache_dir)
        test = load_dataset(args.dataset_repo, language, split="test", cache_dir=args.dataset_cache_dir)

        train_hashes = {hash_example(language, str(ex[source_field]), str(ex[target_field])) for ex in train}
        validation_hashes = {hash_example(language, str(ex[source_field]), str(ex[target_field])) for ex in validation}
        test_hashes = {hash_example(language, str(ex[source_field]), str(ex[target_field])) for ex in test}

        overlaps = {
            "train_examples": len(train_hashes),
            "validation_examples": len(validation_hashes),
            "test_examples": len(test_hashes),
            "train_validation_overlap": len(train_hashes & validation_hashes),
            "train_test_overlap": len(train_hashes & test_hashes),
            "validation_test_overlap": len(validation_hashes & test_hashes),
        }
        per_language[language] = overlaps
        for key, value in overlaps.items():
            totals[key] += value

    report = {
        "dataset_repo": args.dataset_repo,
        "representation": args.representation,
        "per_language": per_language,
        "totals": totals,
    }
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
