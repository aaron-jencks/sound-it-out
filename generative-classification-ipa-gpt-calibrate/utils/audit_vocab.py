#!/usr/bin/env python3
"""
audit_vocab.py — Verify that all label words and prompt scaffold tokens
are properly represented in each tokenizer (text, romanized, ipa).

CPU-only — no GPU needed. Runs in ~10 seconds.

Usage:
    python audit_vocab.py

Output:
    For each tokenizer × word: token count, decoded pieces.
    Words with byte-fallback tokens (e.g. '<0xC3>') are flagged as OOV.
"""
from __future__ import annotations

import sys

sys.path.insert(0, "/users/PAS2836/mugezhang/generative-classification-ipa-gpt")

from ipa_gpt_runtime import IPAGPTTokenizer  # noqa: E402
from ipa_gpt_registry import MODEL_SPECS     # noqa: E402

TOKENIZER_KEYS = ["medium_text", "medium_romanized", "medium_ipa"]

# ---------------------------------------------------------------------------
# Words to audit
# ---------------------------------------------------------------------------

# Scaffold words used in prompts (shared across all datasets)
SCAFFOLD_WORDS = [
    "Review:",
    "Sentiment:",
    "Premise:",
    "Hypothesis:",
    "Relationship:",
]

# Label words per representation
LABEL_WORDS: dict[str, list[str]] = {
    "text": [
        # SST-2 en
        "negative", "positive",
        # XNLI en
        "entailment", "neutral", "contradiction",
        # XNLI es
        "implica", "neutro", "contradicción",
        # XNLI ru
        "следование", "нейтрально", "противоречие",
    ],
    "romanized": [
        # SST-2 en
        "negative", "positive",
        # XNLI en
        "entailment", "neutral", "contradiction",
        # XNLI es
        "implica", "neutro", "contradiccion",   # uroman: no accent
        # XNLI ru
        "sledovaniye", "neytral'no", "protivorechiye",
    ],
    "ipa": [
        # SST-2 en
        "nɛɡətɪv", "pɑzᵻtɪv",
        # XNLI en
        "ɛnteɪlmənt", "nutɹəl", "kɑntɹədɪkʃən",
        # XNLI es
        "implika", "neutɾo", "kontɾaðikθjon",
        # XNLI ru
        "sɭʲidʌvɑnʲijɪ", "nʲijtrɑɭnʌ", "prʌtʲivorʲitʃʲijɪ",
    ],
}


def _has_byte_fallback(pieces: list[str]) -> bool:
    """Return True if any decoded piece is a raw byte token like '<0xC3>'."""
    return any(p.startswith("<0x") and p.endswith(">") for p in pieces)


def audit_tokenizer(key: str) -> None:
    repr_name = key.split("_", 1)[1]  # "text", "romanized", or "ipa"
    spec = MODEL_SPECS[key]
    tok = IPAGPTTokenizer(spec.tokenizer_path)
    print(f"\n{'=' * 60}")
    print(f"  {key}  ({spec.tokenizer_path})")
    print(f"{'=' * 60}")

    sections = [
        ("Scaffold words", SCAFFOLD_WORDS),
        ("Label words", LABEL_WORDS[repr_name]),
    ]
    for section_name, words in sections:
        print(f"\n  --- {section_name} ---")
        for w in words:
            ids    = tok.encode(" " + w)
            pieces = [tok.decode([i]) for i in ids]
            oov    = _has_byte_fallback(pieces)
            flag   = "  *** BYTE-FALLBACK (OOV)" if oov else ""
            multi  = "  (multi-tok)" if len(ids) > 1 else ""
            print(f"    ' {w}': {len(ids)} tok → {pieces}{multi}{flag}")


def main() -> None:
    for key in TOKENIZER_KEYS:
        audit_tokenizer(key)
    print("\nDone.")


if __name__ == "__main__":
    main()
