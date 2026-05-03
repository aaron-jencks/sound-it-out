"""Compute Acc/F1/F2 on unseen MASSIVE test sets from unseen-finetuned checkpoints."""
import sys
import os
import json
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, fbeta_score, accuracy_score
from datasets import load_from_disk
from accelerate import Accelerator
from tokenizers import Tokenizer

sys.path.insert(0, '/users/PAS2836/krishnakb/ondemand/krishna_proj/modded-ipagpt-training')

BASE = "/fs/scratch/PAS2836/krishnakb/xl_multilingual_experiments_v2"
UNSEEN_DATA = f"{BASE}/unseen_processing/cache/massive_unseen_processed"
FT_BASE = f"{BASE}/unseen_ft_massive"
UNSEEN_LANGS = ['bn-BD', 'ar-SA', 'fr-FR', 'el-GR', 'zh-CN']

CONFIGS = [
    ('small', 'ipa',
     f'{FT_BASE}/small_v2_ipa/checkpoints/small-v2-ipa-massive-unseen-all-small-ckpt.pt',
     '/fs/scratch/PAS2836/mugezhang/ml8x3_unified100k/tokenizers/8lang_ipa_stripped/bpe-8lang-ipa-stripped-100k-tokenizer.json',
     'ipa_stripped', 12, 6, 768),
    ('small', 'romanized',
     f'{FT_BASE}/small_v2_romanized/checkpoints/small-v2-romanized-massive-unseen-all-small-ckpt.pt',
     '/fs/scratch/PAS2836/mugezhang/ml8x3_unified100k/tokenizers/8lang_romanized/bpe-8lang-romanized-100k-tokenizer.json',
     'romanized', 12, 6, 768),
    ('small', 'text',
     f'{FT_BASE}/small_v2_text/checkpoints/small-v2-text-massive-unseen-all-small-ckpt.pt',
     '/fs/scratch/PAS2836/mugezhang/ml8x3_unified100k/tokenizers/8lang_text/bpe-8lang-text-100k-tokenizer.json',
     'utt', 12, 6, 768),
    ('medium', 'ipa',
     f'{FT_BASE}/medium_v2_ipa/checkpoints/medium-v2-ipa-massive-unseen-all-medium-ckpt.pt',
     '/fs/scratch/PAS2836/mugezhang/ml8x3_unified100k/tokenizers/8lang_ipa_stripped/bpe-8lang-ipa-stripped-100k-tokenizer.json',
     'ipa_stripped', 16, 8, 1024),
]

PRETRAINED = {
    ('small', 'ipa'): '/fs/scratch/PAS2836/mugezhang/ml8x3_unified100k/v2/models/ipa_stripped/small/000_dc11bd0c-f53d-4b36-983d-89bd4ada6037/best_state_step009850_val3.5126.pt',
    ('small', 'romanized'): '/fs/scratch/PAS2836/mugezhang/ml8x3_unified100k/models/romanized/small/000_4e0f34dc-70fa-43bc-a302-44ddf4f99593/best_state_step008600_val3.5773.pt',
    ('small', 'text'): '/fs/scratch/PAS2836/mugezhang/ml8x3_unified100k/models/text/small/000_6b22ff7e-1fed-4796-9152-8f8209a25399/best_state_step007200_val2.3869.pt',
    ('medium', 'ipa'): '/fs/scratch/PAS2836/mugezhang/ml8x3_unified100k/v2/models/ipa_stripped/medium/000_7b99b975-1c1e-477a-899a-6b4abc700d76/best_state_step171350_val2.6807.pt',
}


def load_and_eval(size, rep, ckpt_path, tokenizer_path, text_col, n_layer, n_head, n_embd):
    if size == 'small':
        from model_small import GPTClassificationSmall as ModelClass
    else:
        from model import GPTClassification as ModelClass

    tok = Tokenizer.from_file(tokenizer_path)
    pad_id = tok.token_to_id("[PAD]") or 0

    import pathlib, tempfile, argparse
    tmp = tempfile.mkdtemp()
    model = ModelClass(
        'cuda', tokenizer_path, None, tmp,
        num_classes=60, num_embed=n_embd, dropout=0.1,
        context_size=1024, batch_size=32, ipa=False,
        text_column=[text_col], label_column='intent',
    )

    pretrained_path = PRETRAINED[(size, rep)]
    vocab_size = tok.get_vocab_size()
    args = argparse.Namespace(
        pretrained_ckpt_path=pretrained_path,
        n_layer=n_layer, n_head=n_head, n_embd=n_embd,
        vocab_size=vocab_size, block_size=1024,
        device='cuda'
    )
    if size == 'small':
        from finetune_small import load_pretrained_model as load_small
        model = load_small(args, model)
    else:
        from finetune import load_pretrained_model as load_medium
        model = load_medium(args, model)

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
            text = ex[text_col] or ""
            encoded = tok.encode(text)
            ids = encoded.ids[:1024]
            if len(ids) < 1024:
                ids = ids + [pad_id] * (1024 - len(ids))
            all_ids.append(ids)
            all_labels.append(ex['intent'])

        # Map intents to integers (same alphabetical ordering as training)
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
        print(f"  [{lang}] acc={acc:.4f}  f1={f1:.4f}  f2={f2:.4f}")

    return results


print("=" * 90)
print("MASSIVE UNSEEN FT - TEST RESULTS (accuracy, F1-macro, F2-macro)")
print("=" * 90)

all_results = {}
for size, rep, ckpt, tok_path, text_col, nl, nh, ne in CONFIGS:
    print(f"\n--- {size.upper()} | {rep.upper()} ---")
    if not os.path.exists(ckpt):
        print("  MISSING checkpoint")
        continue
    results = load_and_eval(size, rep, ckpt, tok_path, text_col, nl, nh, ne)
    all_results[f"{size}_{rep}"] = results

# Save
results_dir = f"{BASE}/unseen_eval_results/massive_ft"
os.makedirs(results_dir, exist_ok=True)
results_file = os.path.join(results_dir, "all_unseen_ft_massive_results.json")
with open(results_file, 'w') as f:
    json.dump(all_results, f, indent=2)

print("\n" + "=" * 90)
print("UNSEEN FT MASSIVE SUMMARY TABLE (percentages)")
print("=" * 90)
header = f"{'Config':<22}"
for lang in UNSEEN_LANGS:
    header += f" {lang} Acc  {lang} F1 "
print(header)
print("-" * len(header))
for key, res in all_results.items():
    row = f"{key:<22}"
    for lang in UNSEEN_LANGS:
        r = res.get(lang, {})
        row += f" {r.get('acc',0)*100:6.1f}  {r.get('f1',0)*100:6.1f} "
    print(row)

print(f"\nResults saved to: {results_file}")
