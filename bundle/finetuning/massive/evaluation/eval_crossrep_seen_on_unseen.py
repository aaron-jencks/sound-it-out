"""Eval cross-rep seen checkpoints (text pretrained -> romanized FT on seen langs) on unseen MASSIVE test sets."""
import sys
import os
import json
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, fbeta_score, accuracy_score
from datasets import load_from_disk
from accelerate import Accelerator
from tokenizers import Tokenizer
import argparse
import tempfile

sys.path.insert(0, '/users/PAS2836/krishnakb/ondemand/krishna_proj/modded-ipagpt-training')

BASE = "/fs/scratch/PAS2836/krishnakb/xl_multilingual_experiments_v2"
UNSEEN_DATA = f"{BASE}/unseen_processing/cache/massive_unseen_processed"
CROSSREP_BASE = f"{BASE}/cross_rep_seen"
UNSEEN_LANGS = ['bn-BD', 'ar-SA', 'fr-FR', 'el-GR', 'zh-CN']

# Both use romanized tokenizer and romanized text column
ROM_TOK = '/fs/scratch/PAS2836/mugezhang/ml8x3_unified100k/tokenizers/8lang_romanized/bpe-8lang-romanized-100k-tokenizer.json'

# Text pretrained models (needed for architecture init)
TEXT_PRETRAINED = {
    'small': '/fs/scratch/PAS2836/mugezhang/ml8x3_unified100k/models/text/small/000_6b22ff7e-1fed-4796-9152-8f8209a25399/best_state_step007200_val2.3869.pt',
    'medium': '/fs/scratch/PAS2836/mugezhang/ml8x3_unified100k/models/text/medium/000_ec803e5d-03c0-4811-9605-561498de5413/best_state_step115550_val1.8103.pt',
}

CONFIGS = [
    ('small', f'{CROSSREP_BASE}/small_text2rom/checkpoints/small-text2rom-seen-all-small-ckpt.pt',
     12, 6, 768),
    ('medium', f'{CROSSREP_BASE}/medium_text2rom/checkpoints/medium-text2rom-seen-all-medium-ckpt.pt',
     16, 8, 1024),
]


def load_and_eval(size, ckpt_path, n_layer, n_head, n_embd):
    if size == 'small':
        from model_small import GPTClassificationSmall as ModelClass
        from finetune_small import load_pretrained_model as load_fn
    else:
        from model import GPTClassification as ModelClass
        from finetune import load_pretrained_model as load_fn

    tok = Tokenizer.from_file(ROM_TOK)
    pad_id = tok.token_to_id("[PAD]") or 0
    vocab_size = tok.get_vocab_size()

    tmp = tempfile.mkdtemp()
    model = ModelClass(
        'cuda', ROM_TOK, None, tmp,
        num_classes=60, num_embed=n_embd, dropout=0.1,
        context_size=1024, batch_size=32, ipa=False,
        text_column=['romanized'], label_column='intent',
    )

    args = argparse.Namespace(
        pretrained_ckpt_path=TEXT_PRETRAINED[size],
        n_layer=n_layer, n_head=n_head, n_embd=n_embd,
        vocab_size=vocab_size, block_size=1024,
        device='cuda'
    )
    model = load_fn(args, model)

    ft_ckpt = torch.load(ckpt_path, map_location='cuda', weights_only=False)
    model.load_state_dict(ft_ckpt['model'], strict=True)
    model.eval()

    accelerator = Accelerator(mixed_precision='bf16')
    model = accelerator.prepare(model)

    results = {}
    for lang in UNSEEN_LANGS:
        ds = load_from_disk(f"{UNSEEN_DATA}/{lang}")
        test_ds = ds['test']

        all_ids, all_labels = [], []
        for ex in test_ds:
            text = ex['romanized'] or ""
            encoded = tok.encode(text)
            ids = encoded.ids[:1024]
            if len(ids) < 1024:
                ids = ids + [pad_id] * (1024 - len(ids))
            all_ids.append(ids)
            all_labels.append(ex['intent'])

        all_intents = sorted(set(
            list(ds['train']['intent']) + list(ds['test']['intent']) + list(ds['validation']['intent'])
        ))
        intent_to_id = {intent: i for i, intent in enumerate(all_intents)}
        label_ids = [intent_to_id[l] for l in all_labels]

        X = torch.tensor(all_ids, dtype=torch.long)
        Y = torch.tensor(label_ids, dtype=torch.long)
        loader = DataLoader(TensorDataset(X, Y), batch_size=32, shuffle=False)
        loader = accelerator.prepare(loader)

        all_preds, all_true = [], []
        with torch.no_grad():
            for batch_X, batch_Y in loader:
                attn_mask = (batch_X != pad_id).long()
                with accelerator.autocast():
                    logits, _ = model(batch_X, attention_mask=attn_mask, labels=batch_Y)
                all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
                all_true.extend(batch_Y.cpu().tolist())

        acc = accuracy_score(all_true, all_preds)
        f1 = f1_score(all_true, all_preds, average='macro', zero_division=0)
        f2 = fbeta_score(all_true, all_preds, beta=2, average='macro', zero_division=0)
        results[lang] = {'acc': acc, 'f1': f1, 'f2': f2}
        print(f"  [{lang}] acc={acc*100:.1f}%  f1={f1*100:.1f}%  f2={f2*100:.1f}%")

    return results


print("=" * 80)
print("CROSS-REP SEEN -> UNSEEN EVAL (text pretrained, romanized FT on seen, eval unseen)")
print("=" * 80)

all_results = {}
for size, ckpt, nl, nh, ne in CONFIGS:
    print(f"\n--- {size.upper()} text->rom (trained on seen, eval on unseen) ---")
    if not os.path.exists(ckpt):
        print(f"  MISSING: {ckpt}")
        continue
    results = load_and_eval(size, ckpt, nl, nh, ne)
    all_results[f"{size}_text2rom"] = results
    avg_acc = sum(r['acc'] for r in results.values()) / len(results)
    avg_f1 = sum(r['f1'] for r in results.values()) / len(results)
    print(f"  AVG: acc={avg_acc*100:.1f}%  f1={avg_f1*100:.1f}%")

results_dir = f"{BASE}/unseen_eval_results/crossrep_seen_on_unseen"
os.makedirs(results_dir, exist_ok=True)
results_file = os.path.join(results_dir, "crossrep_seen_eval_unseen_results.json")
with open(results_file, 'w') as f:
    json.dump(all_results, f, indent=2)
print(f"\nResults saved to: {results_file}")
