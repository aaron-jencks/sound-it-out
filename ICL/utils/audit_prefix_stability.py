#!/usr/bin/env python3
"""
audit_prefix_stability.py — Verify the BPE prefix-stability invariant
across every (dataset, language, representation) combination supported
by icl_eval.py.

Invariant: tokenizer.encode(context) must equal the prefix of
tokenizer.encode(context + " " + label_or_choice). If it doesn't, the
label-token boundary in score_labels_nll would silently shift and the
NLL would be computed on the wrong tokens.

CPU-only — does not load model weights, only tokenizers.

Usage:
    python utils/audit_prefix_stability.py
"""
from __future__ import annotations

import os
import sys
import random

# Lives under utils/; need the parent (calibrate/) on the path to import
# icl_eval, plus the ipa_gpt_runtime checkout.
_HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, "/users/PAS2836/mugezhang/generative-classification-ipa-gpt")

from ipa_gpt_runtime import IPAGPTTokenizer
from ipa_gpt_registry import MODEL_SPECS

from icl_eval import (
    DATASET_LOADERS,
    load_prompt_config,
    construct_prompt,
)

# (dataset, language) → split that the submit scripts use
COMBOS = [
    ("sst2",        "en",  "test"),
    ("xnli",        "en",  "test"),
    ("xnli",        "es",  "test"),
    ("xnli",        "ru",  "test"),
    ("pawsx",       "en",  "validation"),
    ("pawsx",       "es",  "validation"),
    ("xstorycloze", "en",  "eval"),
    ("xstorycloze", "es",  "eval"),
    ("xstorycloze", "ru",  "eval"),
    ("xstorycloze", "hi",  "eval"),
    ("xcopa",       "ta",  "test"),
]

REPRS = ("text", "romanized", "ipa")
SAMPLES_PER_COMBO = 50   # enough to catch boundary issues without loading whole splits

random.seed(0)
violations = []
total_checks = 0

for repr_name in REPRS:
    tok = IPAGPTTokenizer(MODEL_SPECS[f"medium_{repr_name}"].tokenizer_path)

    for dataset, language, split in COMBOS:
        try:
            cfg = load_prompt_config(dataset, language, repr_name)
        except KeyError:
            continue   # combo not registered for this repr (none today)

        loader = DATASET_LOADERS[dataset]
        examples, labels = loader(language, repr_name, split)

        # Sample
        idx = random.sample(range(len(examples)), min(SAMPLES_PER_COMBO, len(examples)))
        sample_examples = [examples[i] for i in idx]

        is_mcq = cfg["task_format"] == "mcq"

        for ex in sample_examples:
            if is_mcq:
                context = cfg["prompt_prefix"] + ex["context"]
                test_words = ex["choices"]
            else:
                context = construct_prompt(cfg, [], [], ex)
                test_words = [cfg["label_dict"][i][0]
                              for i in sorted(cfg["label_dict"])]

            ctx_ids = tok.encode(context)
            for w in test_words:
                full_ids = tok.encode(context + " " + w)
                total_checks += 1
                if full_ids[:len(ctx_ids)] != ctx_ids:
                    violations.append({
                        "dataset": dataset,
                        "lang": language,
                        "repr": repr_name,
                        "ctx_tail": ctx_ids[-3:],
                        "full_head_tail": full_ids[:len(ctx_ids)+1][-4:],
                        "ctx_end": context[-30:],
                        "word": w,
                    })

print(f"\nTotal prefix checks: {total_checks}")
print(f"Violations:          {len(violations)}")
if violations:
    print("\n--- VIOLATIONS ---")
    for v in violations[:20]:
        print(f"  {v['dataset']}/{v['lang']}/{v['repr']}")
        print(f"    word          : {v['word']!r}")
        print(f"    context tail  : ...{v['ctx_end']!r}")
        print(f"    ctx_ids tail  : {v['ctx_tail']}")
        print(f"    full_ids head : {v['full_head_tail']}")
    if len(violations) > 20:
        print(f"  ... and {len(violations) - 20} more")
    sys.exit(1)
else:
    print("\nAll combos pass prefix-stability invariant.")
