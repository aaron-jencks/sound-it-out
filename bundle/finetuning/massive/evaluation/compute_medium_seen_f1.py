"""Compute per-language Acc/F1/F2 for medium MASSIVE seen languages by running inference on GPU.
Saves results to CSV and per-language npz files."""
import sys
import os
import csv
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, fbeta_score, accuracy_score
from datasets import load_dataset
from tokenizers import Tokenizer

sys.path.insert(0, '/users/PAS2836/krishnakb/ondemand/krishna_proj/modded-ipagpt-training')

BASE = "/fs/scratch/PAS2836/krishnakb/xl_multilingual_experiments_v2"
SEEN_LANGS = ['en-US', 'es-ES', 'hi-IN', 'ur-PK', 'ru-RU', 'pl-PL', 'ta-IN', 'ml-IN']
LANG_SHORT = {'en-US': 'EN', 'es-ES': 'ES', 'hi-IN': 'HI', 'ur-PK': 'UR',
              'ru-RU': 'RU', 'pl-PL': 'PL', 'ta-IN': 'TA', 'ml-IN': 'ML'}

PRETRAINED = {
    'ipa': '/fs/scratch/PAS2836/mugezhang/ml8x3_unified100k/v2/models/ipa_stripped/medium/000_7b99b975-1c1e-477a-899a-6b4abc700d76/best_state_step171350_val2.6807.pt',
    'romanized': '/fs/scratch/PAS2836/mugezhang/ml8x3_unified100k/models/romanized/medium/000_6df95892-9904-4069-8a78-6494468907de/best_state_step116750_val2.8296.pt',
    'text': '/fs/scratch/PAS2836/mugezhang/ml8x3_unified100k/models/text/medium/000_ec803e5d-03c0-4811-9605-561498de5413/best_state_step115550_val1.8103.pt',
}

TOKENIZERS = {
    'ipa': '/fs/scratch/PAS2836/mugezhang/ml8x3_unified100k/tokenizers/8lang_ipa_stripped/bpe-8lang-ipa-stripped-100k-tokenizer.json',
    'romanized': '/fs/scratch/PAS2836/mugezhang/ml8x3_unified100k/tokenizers/8lang_romanized/bpe-8lang-romanized-100k-tokenizer.json',
    'text': '/fs/scratch/PAS2836/mugezhang/ml8x3_unified100k/tokenizers/8lang_text/bpe-8lang-text-100k-tokenizer.json',
}

TEXT_COLS = {'ipa': 'ipa_stripped', 'romanized': 'romanized', 'text': 'utt'}

from model import GPTClassification

CSV_PATH = '/users/PAS2836/krishnakb/ondemand/krishna_proj/modded-ipagpt-training/v2-multilingual-finetuning/massive_seen_results_medium_fixed.csv'

print("=" * 90)
print("MASSIVE SEEN LANGUAGES - MEDIUM MODEL (accuracy, F1-macro, F2-macro)")
print("=" * 90)

all_rows = []

for rep in ['ipa', 'romanized', 'text']:
    ckpt_path = f"{BASE}/medium_v2_{rep}_massive/checkpoints/medium-v2-{rep}-massive-medium-ckpt.pt"
    if not os.path.exists(ckpt_path):
        print(f"\n--- MEDIUM | {rep.upper()} --- MISSING checkpoint")
        continue

    print(f"\n--- MEDIUM | {rep.upper()} ---")

    tok = Tokenizer.from_file(TOKENIZERS[rep])
    pad_id = tok.token_to_id("[PAD]") or 0
    text_col = TEXT_COLS[rep]

    import tempfile
    tmp = tempfile.mkdtemp()
    model = GPTClassification(
        'cuda', TOKENIZERS[rep], None, tmp,
        num_classes=60, num_embed=1024, dropout=0.1,
        context_size=1024, batch_size=32, ipa=False,
        text_column=[text_col], label_column='intent',
    )

    import argparse
    args = argparse.Namespace(
        pretrained_ckpt_path=PRETRAINED[rep],
        n_layer=16, n_head=8, n_embd=1024,
        vocab_size=tok.get_vocab_size(), block_size=1024,
        device='cuda'
    )
    from finetune import load_pretrained_model
    model = load_pretrained_model(args, model)

    ft_ckpt = torch.load(ckpt_path, map_location='cuda', weights_only=False)
    model.load_state_dict(ft_ckpt['model'], strict=True)
    model.eval()

    row = {'Model': 'Medium', 'Rep': rep.upper() if rep != 'ipa' else 'IPA'}
    accs, f1s, f2s = [], [], []
    ckpt_dir = os.path.dirname(ckpt_path)

    for lang in SEEN_LANGS:
        short = LANG_SHORT[lang]
        print(f"  [{lang}] Loading test data...", end=" ", flush=True)
        ds = load_dataset('mugezhang/massive_ipa_romanized', lang, split='test',
                         cache_dir=f'/tmp/massive_seen_{rep}')

        all_ids, all_labels = [], []
        for ex in ds:
            text = ex[text_col] or ""
            encoded = tok.encode(text)
            ids = encoded.ids[:1024]
            if len(ids) < 1024:
                ids = ids + [pad_id] * (1024 - len(ids))
            all_ids.append(ids)
            all_labels.append(ex['intent'])

        X = torch.tensor(all_ids, dtype=torch.long)
        Y = torch.tensor(all_labels, dtype=torch.long)
        loader = DataLoader(TensorDataset(X, Y), batch_size=32, shuffle=False)

        all_preds, all_true = [], []
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            for batch_X, batch_Y in loader:
                batch_X = batch_X.cuda()
                batch_Y = batch_Y.cuda()
                attn_mask = (batch_X != pad_id).long()
                logits, _ = model(batch_X, attention_mask=attn_mask, labels=batch_Y)
                all_preds.extend(logits.argmax(dim=-1).tolist())
                all_true.extend(batch_Y.tolist())

        acc = accuracy_score(all_true, all_preds)
        f1 = f1_score(all_true, all_preds, average='macro', zero_division=0)
        f2 = fbeta_score(all_true, all_preds, beta=2, average='macro', zero_division=0)
        accs.append(acc)
        f1s.append(f1)
        f2s.append(f2)
        row[f'{short} Acc'] = round(acc * 100, 1)
        row[f'{short} F1'] = round(f1 * 100, 1)
        print(f"acc={acc:.4f}  f1={f1:.4f}  f2={f2:.4f}")

        # Save per-language npz
        npz_path = os.path.join(ckpt_dir, f"medium-v2-{rep}-massive-{lang}-test-preds.npz")
        np.savez(npz_path, labels=np.array(all_true), preds=np.array(all_preds))

    row['AVG Acc'] = round(np.mean(accs) * 100, 1)
    row['AVG F1'] = round(np.mean(f1s) * 100, 1)
    all_rows.append(row)
    print(f"  [AVG]    acc={np.mean(accs):.4f}  f1={np.mean(f1s):.4f}  f2={np.mean(f2s):.4f}")

# Write CSV
if all_rows:
    fieldnames = ['Model', 'Rep']
    for lang in SEEN_LANGS:
        s = LANG_SHORT[lang]
        fieldnames.extend([f'{s} Acc', f'{s} F1'])
    fieldnames.extend(['AVG Acc', 'AVG F1'])

    with open(CSV_PATH, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nResults saved to: {CSV_PATH}")
