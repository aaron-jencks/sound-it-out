import argparse
import os
import pathlib
import signal
import sys
import time
from contextlib import nullcontext

from datasets import load_dataset, load_from_disk, concatenate_datasets
import inspect
import numpy as np
import torch
import wandb
import torch.nn as nn
from accelerate import Accelerator

from models.model_small import GPTBatchedSmall, GPTClassificationSmall, GPTFlexAttentionSmall, XNLIDataset

# ---------------------------------------------------------------------------
# Global flag for graceful SIGTERM handling (SLURM sends this before killing)
# ---------------------------------------------------------------------------
_SIGTERM_RECEIVED = False

def _sigterm_handler(signum, frame):
    global _SIGTERM_RECEIVED
    _SIGTERM_RECEIVED = True
    print(f"\n[SIGTERM] Received signal {signum} — will save resume checkpoint and exit after current step.")

signal.signal(signal.SIGTERM, _sigterm_handler)
signal.signal(signal.SIGUSR1, _sigterm_handler)  # SLURM sends SIGUSR1 on some clusters


def save_resume_checkpoint(model, optimizer, accelerator, global_step, epoch, batch_idx,
                           best_avg_val_loss, running_loss, args):
    """Save a checkpoint that allows training to be resumed exactly where it left off.

    best_avg_val_loss: float - best average val loss across all languages.
    """
    resume_path = pathlib.Path(args.out_dir) / f"{args.dataset}-resume-ckpt.pt"
    unwrapped_model = accelerator.unwrap_model(model)
    checkpoint = {
        'model': unwrapped_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'global_step': global_step,
        'epoch': epoch,
        'batch_idx': batch_idx,
        'best_avg_val_loss': best_avg_val_loss,
        'running_loss': running_loss,
        'args': vars(args),
        'is_resume_checkpoint': True,
    }
    torch.save(checkpoint, resume_path)
    print(f"[RESUME] Saved resume checkpoint to {resume_path} (epoch={epoch}, step={global_step}, batch={batch_idx})")
    return resume_path

def normalize_polish_columns(dataset):
    """Rename Polish columns to match standard XNLI format"""
    if 'sentence_A' in dataset.column_names:
        print("Renaming Polish columns...")
        rename_map = {
            'sentence_A': 'premise',
            'sentence_B': 'hypothesis',
            'sentence_A_ipa_stripped': 'premise_ipa_stripped',
            'sentence_B_ipa_stripped': 'hypothesis_ipa_stripped',
            'sentence_A_romanized': 'premise_romanized',
            'sentence_B_romanized': 'hypothesis_romanized',
        }
        # Only rename columns that exist
        rename_map = {k: v for k, v in rename_map.items() if k in dataset.column_names}
        dataset = dataset.rename_columns(rename_map)
        print(f" Renamed {len(rename_map)} columns")
    return dataset

# ===== Data Preparation Functions =====
def _load_one_repo(ds_repo: str, split: str, args):
    if args.from_disk:
        ds = load_from_disk(ds_repo)
    else:
        ds = load_dataset(ds_repo, cache_dir=str(args.hf_cache))
    if split not in ds:
        raise ValueError(f"{ds_repo} missing split '{split}'. Has {list(ds.keys())}")

    # NORMALIZE POLISH COLUMNS BEFORE RETURNING
    split_ds = ds[split]
    split_ds = normalize_polish_columns(split_ds)
    return split_ds

def _load_concat(ds_list, split: str, args):
    parts = [_load_one_repo(repo, split, args) for repo in ds_list]
    return concatenate_datasets(parts) if len(parts) > 1 else parts[0]

def prepare_datasets(args, model):
    """
    Five modes:
    1. Separate dataset repos: --train_datasets and --eval_datasets
    2. Same dataset, different configs: --parent_dataset with --train_config and --eval_config
    3. Same dataset, same config: --parent_dataset with --dataset (backward compatible)
    4. Same dataset, multiple train configs: --parent_dataset with --train_configs and --eval_config
    5. Mix of separate datasets + parent dataset with configs
    """

    # MODE 5: Mix of separate datasets + parent dataset with configs
    if args.train_datasets is not None and args.parent_dataset is not None and args.train_configs is not None:
        multi_lang = args.eval_langs is not None or args.eval_parent_langs is not None

        if args.eval_datasets is None and not multi_lang:
            raise ValueError("Must provide --eval_datasets (or --eval_langs/--eval_parent_langs) for MODE 5")

        print(f"\nMODE 5: Mixed multilingual training (separate repos + parent configs)")
        if multi_lang:
            print(f"  ** Multi-language evaluation mode **")
        print(f"Loading separate datasets: {args.train_datasets}")
        print(f"Loading from parent: {args.parent_dataset}")
        print(f"Parent configs: {args.train_configs}")

        train_datasets_list = []
        for repo in args.train_datasets:
            ds = _load_one_repo(repo, "train", args)
            train_datasets_list.append(ds)
            print(f"  Loaded {repo}: {len(ds):,} examples")

        for config in args.train_configs:
            ds = load_dataset(args.parent_dataset, config, cache_dir=str(args.hf_cache))
            train_datasets_list.append(ds["train"])
            print(f"  Loaded {args.parent_dataset} ({config}): {len(ds['train']):,} examples")

        train_dataset = concatenate_datasets(train_datasets_list)

        if multi_lang:
            # Multi-lang mode: use first eval language as dummy val for prepare_if_needed
            # (only needed for tokenizing train data + computing class weights)
            if args.eval_lang_datasets:
                dummy_val = _load_one_repo(args.eval_lang_datasets[0], "validation", args)
            else:
                dummy_ds = load_dataset(args.parent_dataset, args.eval_parent_langs[0],
                                        cache_dir=str(args.hf_cache))
                dummy_val = dummy_ds["validation"]
                dummy_val = normalize_polish_columns(dummy_val)
            print(f"Combined train size: {len(train_dataset):,} | Val: multi-lang (see below)\n")
            model.prepare_if_needed(train_dataset, dummy_val, args.force_tokenization)
        else:
            validation_dataset = _load_concat(args.eval_datasets, "validation", args)
            print(f"Combined train size: {len(train_dataset):,} | Val size: {len(validation_dataset):,}\n")
            model.prepare_if_needed(train_dataset, validation_dataset, args.force_tokenization)
        return

    # MODE 1: Separate dataset repos
    if args.train_datasets is not None or args.eval_datasets is not None:
        if args.train_datasets is None or args.eval_datasets is None:
            raise ValueError("Provide BOTH --train_datasets and --eval_datasets")

        train_dataset = _load_concat(args.train_datasets, "train", args)
        validation_dataset = _load_concat(args.eval_datasets, "validation", args)

        print(f"\nMODE 1: Separate dataset repos")
        print(f"Loaded TRAIN from: {args.train_datasets}")
        print(f"Loaded EVAL  from: {args.eval_datasets}")
        print(f"Train size: {len(train_dataset):,} | Val size: {len(validation_dataset):,}\n")

        model.prepare_if_needed(train_dataset, validation_dataset, args.force_tokenization)
        return

    # MODE 4: Same dataset, MULTIPLE train configs (check BEFORE Mode 2!)
    if args.train_configs is not None:
        multi_lang = args.eval_langs is not None or args.eval_parent_langs is not None
        if args.eval_config is None and not multi_lang:
            raise ValueError("Must provide --eval_config (or --eval_parent_langs) when using --train_configs")
        if args.parent_dataset is None:
            raise ValueError("Must provide --parent_dataset when using --train_configs")

        print(f"\nMODE 4: Same dataset, multiple train configs (COMBINED TRAINING)")
        print(f"Loading from: {args.parent_dataset}")
        print(f"Train configs: {args.train_configs}")
        if multi_lang:
            print(f"  ** Multi-language evaluation mode **")

        if args.from_disk:
            # Load from local disk paths: parent_dataset/config_name is a saved DatasetDict
            train_datasets_list = []
            for config in args.train_configs:
                local_path = pathlib.Path(args.parent_dataset) / config
                ds = load_from_disk(str(local_path))
                train_datasets_list.append(ds["train"])
                print(f"  Loaded {config} train (from disk): {len(ds['train']):,} examples")
            train_dataset = concatenate_datasets(train_datasets_list)

            if multi_lang:
                local_path = pathlib.Path(args.parent_dataset) / args.eval_parent_langs[0]
                dummy_ds = load_from_disk(str(local_path))
                dummy_val = dummy_ds["validation"]
                dummy_val = normalize_polish_columns(dummy_val)
                print(f"Combined train size: {len(train_dataset):,} | Val: multi-lang (see below)\n")
                model.prepare_if_needed(train_dataset, dummy_val, args.force_tokenization)
            else:
                local_path = pathlib.Path(args.parent_dataset) / args.eval_config
                eval_ds = load_from_disk(str(local_path))
                validation_dataset = eval_ds["validation"]
                print(f"Combined train size: {len(train_dataset):,} | Val size: {len(validation_dataset):,}\n")
                model.prepare_if_needed(train_dataset, validation_dataset, args.force_tokenization)
            return

        train_datasets_list = []
        for config in args.train_configs:
            ds = load_dataset(args.parent_dataset, config, cache_dir=str(args.hf_cache))
            train_datasets_list.append(ds["train"])
            print(f"  Loaded {config} train: {len(ds['train']):,} examples")

        train_dataset = concatenate_datasets(train_datasets_list)

        if multi_lang:
            # Multi-lang mode: use first eval language as dummy val for tokenization
            dummy_ds = load_dataset(args.parent_dataset, args.eval_parent_langs[0],
                                    cache_dir=str(args.hf_cache))
            dummy_val = dummy_ds["validation"]
            dummy_val = normalize_polish_columns(dummy_val)
            print(f"Combined train size: {len(train_dataset):,} | Val: multi-lang (see below)\n")
            model.prepare_if_needed(train_dataset, dummy_val, args.force_tokenization)
        else:
            eval_ds = load_dataset(args.parent_dataset, args.eval_config, cache_dir=str(args.hf_cache))
            validation_dataset = eval_ds["validation"]
            print(f"Combined train size: {len(train_dataset):,} | Val size: {len(validation_dataset):,}\n")
            model.prepare_if_needed(train_dataset, validation_dataset, args.force_tokenization)
        return

    # MODE 2: Same dataset, different configs
    if args.train_config is not None or args.eval_config is not None:
        if args.train_config is None or args.eval_config is None:
            raise ValueError("Provide BOTH --train_config and --eval_config")
        if args.parent_dataset is None:
            raise ValueError("Must provide --parent_dataset when using --train_config/--eval_config")

        print(f"\nMODE 2: Same dataset, different configs")
        print(f"Loading from: {args.parent_dataset}")
        print(f"Train config: {args.train_config}")
        print(f"Eval config:  {args.eval_config}")

        if args.from_disk:
            raise ValueError("Cannot use --from_disk with config-based loading")

        train_ds = load_dataset(args.parent_dataset, args.train_config, cache_dir=str(args.hf_cache))
        eval_ds = load_dataset(args.parent_dataset, args.eval_config, cache_dir=str(args.hf_cache))

        train_dataset = train_ds["train"]
        validation_dataset = eval_ds["validation"]

        # NORMALIZE POLISH COLUMNS
        train_dataset = normalize_polish_columns(train_dataset)
        validation_dataset = normalize_polish_columns(validation_dataset)

        print(f"Train size: {len(train_dataset):,} | Val size: {len(validation_dataset):,}\n")

        model.prepare_if_needed(train_dataset, validation_dataset, args.force_tokenization)
        return

    # MODE 3: Same dataset, same config (backward compatible)
    if args.no_subset:
        dataset = load_from_disk(args.parent_dataset) if args.from_disk else load_dataset(args.parent_dataset)
    else:
        if args.from_disk:
            raise Exception("can't load from disk with subset.")
        dataset = load_dataset(args.parent_dataset, args.dataset, cache_dir=str(args.hf_cache))

    print(f"\nMODE 3: Same dataset, same config (backward compatible)")
    print(f"Dataset: {args.parent_dataset}")
    print(f"Config: {args.dataset if not args.no_subset else 'None (no_subset)'}")
    print(f"Train size: {len(dataset['train']):,} | Val size: {len(dataset['validation']):,}\n")

    model.prepare_if_needed(dataset["train"], dataset["validation"], args.force_tokenization)

def prepare_multilang_eval(args, model):
    """Load and tokenize validation data for multiple evaluation languages.

    Returns dict of {lang: XNLIDataset}.
    """
    from models.model_small import XNLIDataset as _XNLIDataset

    eval_map = {}  # lang -> HF validation split

    # Languages with separate repos
    if args.eval_langs and args.eval_lang_datasets:
        assert len(args.eval_langs) == len(args.eval_lang_datasets), \
            "--eval_langs and --eval_lang_datasets must have the same length"
        for lang, repo in zip(args.eval_langs, args.eval_lang_datasets):
            eval_map[lang] = _load_one_repo(repo, "validation", args)
            print(f"  [multi-eval] Loaded {lang} val from {repo}: {len(eval_map[lang]):,}")

    # Languages from parent dataset configs
    if args.eval_parent_langs:
        assert args.parent_dataset is not None, \
            "--eval_parent_langs requires --parent_dataset"
        for lang in args.eval_parent_langs:
            if args.from_disk:
                local_path = pathlib.Path(args.parent_dataset) / lang
                ds = load_from_disk(str(local_path))
            else:
                ds = load_dataset(args.parent_dataset, lang, cache_dir=str(args.hf_cache))
            val_ds = ds["validation"]
            val_ds = normalize_polish_columns(val_ds)
            eval_map[lang] = val_ds
            print(f"  [multi-eval] Loaded {lang} val from {args.parent_dataset}({lang}): {len(val_ds):,}")

    # Tokenize each language's validation data into its own subdirectory
    val_datasets = {}
    for lang, val_ds in eval_map.items():
        lang_dir = model.data_dir / f"eval_{lang}"
        lang_dir.mkdir(parents=True, exist_ok=True)
        val_path = lang_dir / "val.bin"

        if not val_path.exists():
            print(f"  [multi-eval] Tokenizing {lang} validation data...")
            model._tokenize_and_save(val_ds, val_path)

        val_datasets[lang] = _XNLIDataset(val_path, model.context_window)

    return val_datasets


def prepare_multilang_test(args, model, accelerator):
    """Load and tokenize test data for multiple evaluation languages.

    Returns dict of {lang: DataLoader} (already accelerator-prepared).
    """
    from models.model_small import XNLIDataset as _XNLIDataset
    from torch.utils.data import DataLoader

    test_map = {}  # lang -> HF test split

    if args.eval_langs and args.eval_lang_datasets:
        for lang, repo in zip(args.eval_langs, args.eval_lang_datasets):
            test_map[lang] = _load_one_repo(repo, "test", args)

    if args.eval_parent_langs:
        for lang in args.eval_parent_langs:
            if args.from_disk:
                local_path = pathlib.Path(args.parent_dataset) / lang
                ds = load_from_disk(str(local_path))
            else:
                ds = load_dataset(args.parent_dataset, lang, cache_dir=str(args.hf_cache))
            test_ds = ds["test"]
            test_ds = normalize_polish_columns(test_ds)
            test_map[lang] = test_ds

    test_loaders = {}
    for lang, test_ds in test_map.items():
        lang_dir = model.data_dir / f"eval_{lang}"
        lang_dir.mkdir(parents=True, exist_ok=True)
        test_path = lang_dir / "test.bin"

        if not test_path.exists():
            print(f"  [multi-test] Tokenizing {lang} test data...")
            unwrapped = accelerator.unwrap_model(model)
            unwrapped._tokenize_and_save(test_ds, test_path)

        test_dataset = _XNLIDataset(test_path, model.context_window)
        loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True, drop_last=False)
        test_loaders[lang] = accelerator.prepare(loader)

    return test_loaders


def load_pretrained_model(args, model):
    """Load pretrained SMALL model from training script checkpoint and convert to batched version"""
    print(f"Loading pretrained SMALL model (12 layers, 6 heads, 768 dim) from {args.pretrained_ckpt_path}")
    checkpoint = torch.load(args.pretrained_ckpt_path, map_location=args.device, weights_only=False)

    if 'model' not in checkpoint:
        raise ValueError("Checkpoint does not contain 'model' key. Make sure this is a checkpoint from the training script.")

    state_dict = checkpoint['model']

    # Clean up any '_orig_mod.' prefix (from torch.compile)
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    # INFER vocab_size from checkpoint instead of using args
    actual_vocab_size = state_dict['embed.weight'].shape[0]

    has_scalars = 'scalars' in state_dict
    if has_scalars:
        actual_scalars_size = state_dict['scalars'].shape[0]

    print(f"Detected from checkpoint:")
    print(f"  - Vocab size: {actual_vocab_size} (you specified {args.vocab_size})")
    if has_scalars:
        print(f"  - Scalars size: {actual_scalars_size}")

    # Update args to match checkpoint
    args.vocab_size = actual_vocab_size

    # Load into temporary FlexAttention model structure
    print("Loading FlexAttention checkpoint...")
    pretrained_flex = GPTFlexAttentionSmall(
        vocab_size=actual_vocab_size,
        num_layers=args.n_layer,
        num_heads=args.n_head,
        model_dim=args.n_embd,
        max_seq_len=args.block_size
    ).to(args.device)

    if has_scalars:
        pretrained_flex.scalars = nn.Parameter(torch.zeros(actual_scalars_size, device=args.device))

    # Handle potential lm_head_w padding (like medium model)
    if 'lm_head_w' in state_dict:
        lm_head_vocab_size = state_dict['lm_head_w'].shape[0]
        if lm_head_vocab_size != actual_vocab_size:
            print(f"  - LM head vocab size: {lm_head_vocab_size} (padded)")
            pretrained_flex.lm_head_w = nn.Parameter(torch.zeros(lm_head_vocab_size, args.n_embd, device=args.device))

    result = pretrained_flex.load_state_dict(state_dict, strict=False)

    if result.missing_keys:
        print(f"Warning: Missing keys: {result.missing_keys}")
    if result.unexpected_keys:
        print(f"Warning: Unexpected keys: {result.unexpected_keys}")

    actual_n_embd = state_dict['embed.weight'].shape[1]
    if actual_n_embd != args.n_embd:
        raise ValueError(f"Checkpoint n_embd={actual_n_embd} != args.n_embd={args.n_embd}. Fix --n_embd.")

    # Convert to batched version (this copies all weights)
    print("Converting to batched model (enables batch_size > 1)...")
    pretrained_batched = GPTBatchedSmall.from_pretrained_gpt(pretrained_flex)
    pretrained_batched.to(args.device)

    # Attach pretrained model to classification wrapper
    model.pretrained_model = pretrained_batched
    model.to(args.device)

    print(f"Loaded checkpoint successfully!")
    print(f"  Model: 12 layers, 6 heads, 768 dimensions")
    print(f"  Vocab size: {actual_vocab_size}")
    print(f"  Batch size: {model.batch_size}")

    return model


def get_lr(it, learning_rate, warmup_iters, lr_decay_iters, min_lr):
    """Learning rate schedule: linear warmup then linear decay"""
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    return learning_rate - decay_ratio * (learning_rate - min_lr)


@torch.no_grad()
def evaluate(model, val_loader, accelerator):
    """Evaluate on the full validation set using the DataLoader"""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []

    for X, Y in val_loader:
        attention_mask = (X != model.pad_token_id).long()
        with accelerator.autocast():
            logits, loss = model(X, attention_mask=attention_mask, labels=Y)
        preds = logits.argmax(dim=-1)

        total_loss += loss.item() * X.size(0)
        total_correct += (preds == Y).sum().item()
        total_samples += X.size(0)
        all_preds.append(preds.cpu())
        all_labels.append(Y.cpu())

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    from sklearn.metrics import fbeta_score
    f2 = float(fbeta_score(all_labels, all_preds, beta=2, average='macro', zero_division=0))

    model.train()
    return {'val': avg_loss, 'val_accuracy': accuracy, 'val_f2': f2}


def finetune(model, train_loader, val_loaders, optimizer, accelerator,
             best_avg_val_loss, best_checkpoint_path, args,
             start_epoch=0, start_batch_idx=0, start_global_step=0, start_running_loss=0.0):
    """Main training loop using Accelerate + DataLoader, with resume + SIGTERM support.

    val_loaders: dict {lang: DataLoader}
    best_avg_val_loss: float - best average val loss across all languages
    best_checkpoint_path: Path - single best checkpoint
    """
    global _SIGTERM_RECEIVED

    if args.wandb_log:
        config = {
            "model_size": "small (12L, 6H, 768D)",
            "learning_rate": model.learning_rate,
            "weight_decay": model.weight_decay,
            "beta1": model.beta1,
            "beta2": model.beta2,
            "grad_clip": model.grad_clip,
            "dropout": model.dropout_rate,
            "warmup_iter_ratio": model.warmup_iter_ratio,
            "lr_decay_iter_ratio": model.lr_decay_iter_ratio,
            "min_lr": model.min_lr,
            "num_epochs": args.num_epochs,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "batch_size": model.batch_size,
        }
        wandb.init(project=args.wandb_project, name=f"{args.dataset}-small-{time.time()}", config=config)

    # Ensure all parameters can learn
    for param in model.pretrained_model.parameters():
        param.requires_grad = True

    # Calculate total steps for LR schedule
    steps_per_epoch = len(train_loader) // args.gradient_accumulation_steps
    total_steps = steps_per_epoch * args.num_epochs
    warmup_steps = int(model.warmup_iter_ratio * total_steps)
    lr_decay_steps = int(model.lr_decay_iter_ratio * total_steps)

    lang_names = sorted(val_loaders.keys())
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total steps: {total_steps}")
    print(f"Warmup steps: {warmup_steps}")
    print(f"LR decay steps: {lr_decay_steps}")
    print(f"Evaluating {len(lang_names)} language(s): {', '.join(lang_names)}")
    if start_global_step > 0:
        print(f"[RESUME] Resuming from epoch={start_epoch}, batch={start_batch_idx}, global_step={start_global_step}")

    global_step = start_global_step
    running_loss = start_running_loss

    for epoch in range(start_epoch, args.num_epochs):
        print(f"\n--- Epoch {epoch + 1}/{args.num_epochs} ---")
        epoch_start = time.time()

        for batch_idx, (X, Y) in enumerate(train_loader):
            # Skip batches we already processed (for resume)
            if epoch == start_epoch and batch_idx < start_batch_idx:
                continue

            # --- SIGTERM: save and exit ---
            if _SIGTERM_RECEIVED:
                print(f"[SIGTERM] Saving resume checkpoint before exit...")
                save_resume_checkpoint(model, optimizer, accelerator, global_step, epoch, batch_idx,
                                       best_avg_val_loss, running_loss, args)
                print(f"[SIGTERM] Exiting with code 99 (signals incomplete training).")
                sys.exit(99)

            # Learning rate scheduling
            lr = get_lr(global_step, model.learning_rate, warmup_steps, lr_decay_steps, model.min_lr) \
                if not args.no_decay_lr else model.learning_rate

            for param_group in optimizer.param_groups:
                if param_group.get('_is_pretrained', False):
                    param_group['lr'] = lr * 0.1
                else:
                    param_group['lr'] = lr

            # Forward pass with gradient accumulation
            with accelerator.accumulate(model):
                attention_mask = (X != model.pad_token_id).long()
                with accelerator.autocast():
                    logits, loss = model(X, attention_mask=attention_mask, labels=Y)

                accelerator.backward(loss)

                # Gradient clipping
                if model.grad_clip != 0.0:
                    accelerator.clip_grad_norm_(model.parameters(), model.grad_clip)

                optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item()

            # Check if this is an optimizer step (after accumulation)
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                global_step += 1

                # Evaluation
                if global_step % args.eval_interval == 0:
                    avg_train_loss = running_loss / args.eval_interval / args.gradient_accumulation_steps
                    running_loss = 0.0

                    print(f"\nstep {global_step}: train loss {avg_train_loss:.4f}")

                    # Get current model state once (for saving checkpoints)
                    unwrapped_model = accelerator.unwrap_model(model)
                    current_state = unwrapped_model.state_dict()

                    wandb_log = {
                        "step": global_step,
                        "epoch": epoch,
                        "train/loss": avg_train_loss,
                        "lr": lr,
                    }

                    total_val_loss = 0.0
                    total_val_acc = 0.0
                    total_val_f2 = 0.0
                    for lang in lang_names:
                        # Check SIGTERM between language evaluations
                        if _SIGTERM_RECEIVED:
                            print(f"\n[SIGTERM] Received during eval, saving checkpoint...")
                            save_resume_checkpoint(model, optimizer, accelerator, global_step, epoch, batch_idx,
                                                   best_avg_val_loss, running_loss, args)
                            print(f"[SIGTERM] Exiting with code 99.")
                            sys.exit(99)

                        losses = evaluate(model, val_loaders[lang], accelerator)
                        print(f"  [{lang}] val loss {losses['val']:.4f}, "
                              f"f2 {losses['val_f2']:.4f}, acc {losses['val_accuracy']:.4f}")

                        total_val_loss += losses['val']
                        total_val_acc += losses['val_accuracy']
                        total_val_f2 += losses['val_f2']

                        wandb_log[f"val/{lang}_loss"] = losses['val']
                        wandb_log[f"val/{lang}_f2"] = losses['val_f2']
                        wandb_log[f"val/{lang}_accuracy"] = losses['val_accuracy']

                    # Compute average across all languages
                    n_langs = len(lang_names)
                    avg_val_loss = total_val_loss / n_langs
                    avg_val_acc = total_val_acc / n_langs
                    avg_val_f2 = total_val_f2 / n_langs
                    print(f"  [AVG] val loss {avg_val_loss:.4f}, f2 {avg_val_f2:.4f}, acc {avg_val_acc:.4f}", end="")

                    # Save single best checkpoint based on average val loss
                    if avg_val_loss < best_avg_val_loss:
                        best_avg_val_loss = avg_val_loss
                        print(f"  ** NEW BEST **")
                        checkpoint = {
                            'model': current_state,
                            'optimizer': optimizer.state_dict(),
                            'global_step': global_step,
                            'epoch': epoch,
                            'val_loss': avg_val_loss,
                            'val_accuracy': avg_val_acc,
                            'val_f2': avg_val_f2,
                            'args': vars(args)
                        }
                        torch.save(checkpoint, best_checkpoint_path)
                    else:
                        print()

                    wandb_log["val/avg_loss"] = avg_val_loss
                    wandb_log["val/avg_f2"] = avg_val_f2
                    wandb_log["val/avg_accuracy"] = avg_val_acc

                    if args.wandb_log:
                        wandb.log(wandb_log)

        # Reset start_batch_idx after first resumed epoch completes
        start_batch_idx = 0
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch + 1} completed in {epoch_time:.1f}s ({epoch_time/60:.1f}m)")

    return best_avg_val_loss


def main():
    parser = argparse.ArgumentParser(description="Finetune a SMALL model (12L, 6H, 768D) with Accelerate + DataLoader")

    # Required
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (e.g., sst2)')
    parser.add_argument('--pretrained_ckpt_path', type=pathlib.Path, required=True,
                        help='Path to pretrained SMALL model checkpoint from training script')

    # Dataset settings
    parser.add_argument('--parent_dataset', type=str, default='nyu-mll/glue')
    parser.add_argument('--no_subset', action='store_true')
    parser.add_argument('--from_disk', action='store_true')
    parser.add_argument('--use_ipa', action='store_true')
    parser.add_argument('--force_tokenization', action='store_true')
    parser.add_argument('--text_column', type=str, nargs='+', default=['sentence'])
    parser.add_argument('--label_column', type=str, default='label',
                        help='Name of the label column in the dataset')
    parser.add_argument('--train_datasets', type=str, nargs='+', default=None)
    parser.add_argument('--eval_datasets', type=str, nargs='+', default=None)
    parser.add_argument('--train_config', type=str, default=None,
                    help='Config/subset for training (e.g., ta-IN). Use with --eval_config for cross-language.')
    parser.add_argument('--eval_config', type=str, default=None,
                    help='Config/subset for evaluation (e.g., ml-IN). Use with --train_config for cross-language.')
    parser.add_argument('--train_configs', type=str, nargs='+', default=None,
                    help='Multiple configs for training (e.g., ta-IN ml-IN). Use with --eval_config.')

    # Multi-language evaluation (evaluate all languages in a single training run)
    parser.add_argument('--eval_langs', type=str, nargs='+', default=None,
                    help='Language names for multi-lang eval from separate repos (parallel with --eval_lang_datasets)')
    parser.add_argument('--eval_lang_datasets', type=str, nargs='+', default=None,
                    help='Dataset repos for multi-lang eval (parallel with --eval_langs)')
    parser.add_argument('--eval_parent_langs', type=str, nargs='+', default=None,
                    help='Language names that use parent_dataset configs for eval (e.g., ta ml)')

    # Paths
    parser.add_argument('--hf_cache', type=pathlib.Path, default=pathlib.Path('./cache'))
    parser.add_argument('--out_dir', type=pathlib.Path, default=pathlib.Path('./checkpoints'))
    parser.add_argument('--tokenizer_dir', type=pathlib.Path, default=pathlib.Path('./tokenizers'))
    parser.add_argument('--data_dir', type=pathlib.Path, default=pathlib.Path('./datasets'))
    parser.add_argument('--tokenizer_name', type=str, default='gpt2')

    # Training settings
    parser.add_argument('--num_epochs', type=int, default=1,
                        help='Number of epochs (default: 1)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training (default: 16)')
    parser.add_argument('--eval_iters', type=int, default=10,
                        help='(unused, kept for CLI compatibility)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--no_decay_lr', action='store_true')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='Number of classes for classification (default: 2 for binary)')
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader workers (default: 4)')

    # Model architecture (SMALL model - must match training script)
    parser.add_argument('--n_layer', type=int, default=12, help='Number of layers (SMALL model: 12)')
    parser.add_argument('--n_head', type=int, default=6, help='Number of heads (SMALL model: 6)')
    parser.add_argument('--n_embd', type=int, default=768, help='Embedding dimension (SMALL model: 768)')
    parser.add_argument('--block_size', type=int, default=1024, help='Context size')
    parser.add_argument('--vocab_size', type=int, default=50257, help='Vocabulary size (must match pretrained model)')

    # Device settings
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dtype', type=str, default='bfloat16', choices=['float16', 'bfloat16', 'float32'])
    parser.add_argument('--no_cuda', action='store_true')

    # Logging
    parser.add_argument('--wandb_project', type=str, default='ipa_finetuning_small')
    parser.add_argument('--eval_interval', type=int, default=100,
                        help='Evaluate every N optimizer steps (default: 100)')
    parser.add_argument('--wandb_log', action='store_true')
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--save_preds', action='store_true',
                        help='Save predictions as .npz when using --eval_only')
    parser.add_argument('--resume_checkpoint', type=pathlib.Path, default=None,
                        help='Path to resume checkpoint (saved by SIGTERM handler or periodic save)')

    args = parser.parse_args()

    print("="*80)
    print("SMALL MODEL ACCELERATED FINE-TUNING")
    print("="*80)
    print(f"Model architecture:")
    print(f"  - Layers: {args.n_layer}")
    print(f"  - Heads: {args.n_head}")
    print(f"  - Dimensions: {args.n_embd}")
    print(f"\nOptimizations:")
    print(f"  - HF Accelerate (mixed precision, gradient accumulation)")
    print(f"  - DataLoader with {args.num_workers} workers, pin_memory, prefetch")
    print(f"  - Flash Attention via scaled_dot_product_attention")
    print(f"  - Fused AdamW optimizer")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"  - Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print("="*80)

    # Verify architecture matches small model
    if args.n_layer != 12 or args.n_head != 6 or args.n_embd != 768:
        raise ValueError(
            f"Small model must be 12L/6H/768D, got {args.n_layer}L/{args.n_head}H/{args.n_embd}D"
        )

    # --- Tokenizer resolution ---
    tokenizer_dir = pathlib.Path(args.tokenizer_dir)
    tokenizer_name_path = pathlib.Path(args.tokenizer_name)

    vocab_file = None

    if tokenizer_name_path.is_absolute():
        if tokenizer_name_path.exists():
            vocab_file = tokenizer_name_path
            print(f"Found tokenizer at: {vocab_file}")
        else:
            print(f"Warning: Absolute path provided but file not found: {tokenizer_name_path}")

    if vocab_file is None:
        possible_files = [
            tokenizer_dir / args.tokenizer_name,
            tokenizer_dir / f"{args.tokenizer_name}.json",
            tokenizer_dir / f"{args.tokenizer_name}-tokenizer.json",
            tokenizer_dir / f"{args.tokenizer_name}-vocab.json",
        ]

        for file_path in possible_files:
            if file_path.exists():
                vocab_file = file_path
                print(f"Found tokenizer at: {vocab_file}")
                break

    if vocab_file is None and args.tokenizer_name == 'gpt2':
        from tokenizers import Tokenizer
        try:
            print("Loading GPT-2 tokenizer from HuggingFace...")
            tokenizer = Tokenizer.from_pretrained("gpt2")
            vocab_file = tokenizer_dir / "gpt2-tokenizer.json"
            vocab_file.parent.mkdir(parents=True, exist_ok=True)
            tokenizer.save(str(vocab_file))
            print(f"Saved tokenizer to {vocab_file}")
        except Exception as e:
            print(f"Error: Could not load/save GPT-2 tokenizer: {e}")
            raise

    if vocab_file is None:
        error_msg = f"Could not find tokenizer file. Tried:\n"
        for f in possible_files:
            error_msg += f"  - {f}\n"
        error_msg += f"\nMake sure the tokenizer file exists or provide the full path using --tokenizer_name"
        raise FileNotFoundError(error_msg)

    # --- Initialize model ---
    model = GPTClassificationSmall(
        args.device,
        vocab_file,
        None,
        args.data_dir,
        num_classes=args.num_classes,
        num_embed=args.n_embd,
        dropout=args.dropout,
        context_size=args.block_size,
        batch_size=args.batch_size,
        ipa=args.use_ipa,
        text_column=args.text_column,
        label_column=args.label_column
    )

    if args.learning_rate is not None:
        print(f"Overriding learning rate: {model.learning_rate} -> {args.learning_rate}")
        model.learning_rate = args.learning_rate

    # --- Prepare data ---
    # For eval_only with cached data, skip HF dataset loading entirely
    if args.eval_only and model.val_data_path.exists() and model.train_data_path.exists():
        print(f"[eval_only] Using cached data: {model.train_data_path}, {model.val_data_path}")
        model.train_dataset = XNLIDataset(model.train_data_path, model.context_window)
        model.val_dataset = XNLIDataset(model.val_data_path, model.context_window)
        model._compute_class_weights()
    else:
        prepare_datasets(args, model)

    # --- Load pretrained weights ---
    torch.manual_seed(1337)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model = load_pretrained_model(args, model)

    # --- Detect multi-lang eval mode ---
    multi_lang = args.eval_langs is not None or args.eval_parent_langs is not None

    # --- Build DataLoaders ---
    if multi_lang:
        print("\n[MULTI-LANG] Preparing per-language evaluation datasets...")
        multilang_val_datasets = prepare_multilang_eval(args, model)
        # Build train loader only (val loaders built per-language below)
        from torch.utils.data import DataLoader
        train_loader = DataLoader(
            model.train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True, prefetch_factor=2,
            persistent_workers=True, drop_last=True,
        )
    else:
        train_loader, val_loader = model.build_dataloaders(
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )

    # --- Configure optimizer (fused AdamW with differential LR) ---
    device_type = 'cuda' if 'cuda' in args.device else 'cpu'

    pretrained_params = list(model.pretrained_model.parameters())
    pretrained_param_ids = set(id(p) for p in pretrained_params)
    classifier_params = [p for p in model.parameters() if id(p) not in pretrained_param_ids]

    pretrained_param_dict = {f"pretrained.{i}": p for i, p in enumerate(pretrained_params) if p.requires_grad}
    classifier_param_dict = {f"classifier.{i}": p for i, p in enumerate(classifier_params) if p.requires_grad}

    pretrained_decay = [p for n, p in pretrained_param_dict.items() if p.dim() >= 2]
    pretrained_nodecay = [p for n, p in pretrained_param_dict.items() if p.dim() < 2]
    classifier_decay = [p for n, p in classifier_param_dict.items() if p.dim() >= 2]
    classifier_nodecay = [p for n, p in classifier_param_dict.items() if p.dim() < 2]

    print(f"Pretrained decay: {len(pretrained_decay)} params, {sum(p.numel() for p in pretrained_decay):,} elements")
    print(f"Pretrained no-decay: {len(pretrained_nodecay)} params, {sum(p.numel() for p in pretrained_nodecay):,} elements")
    print(f"Classifier decay: {len(classifier_decay)} params, {sum(p.numel() for p in classifier_decay):,} elements")
    print(f"Classifier no-decay: {len(classifier_nodecay)} params, {sum(p.numel() for p in classifier_nodecay):,} elements")

    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == 'cuda'
    extra_args = dict(fused=True) if use_fused else dict()
    print(f"Using fused AdamW: {use_fused}")

    optim_groups = [
        {'params': pretrained_decay, 'weight_decay': model.weight_decay, 'lr': model.learning_rate * 0.1, '_is_pretrained': True},
        {'params': pretrained_nodecay, 'weight_decay': 0.0, 'lr': model.learning_rate * 0.1, '_is_pretrained': True},
        {'params': classifier_decay, 'weight_decay': model.weight_decay, '_is_pretrained': False},
        {'params': classifier_nodecay, 'weight_decay': 0.0, '_is_pretrained': False}
    ]

    optimizer = torch.optim.AdamW(optim_groups, lr=model.learning_rate, betas=(model.beta1, model.beta2), **extra_args)

    # --- Set up Accelerator ---
    mixed_precision_map = {'float16': 'fp16', 'bfloat16': 'bf16', 'float32': 'no'}
    accelerator = Accelerator(
        mixed_precision=mixed_precision_map[args.dtype],
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    if multi_lang:
        # Prepare model, optimizer, train_loader
        model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
        # Build and prepare per-language val loaders
        val_loaders = {}
        for lang, ds in multilang_val_datasets.items():
            loader = DataLoader(
                ds, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=True, drop_last=False,
            )
            val_loaders[lang] = accelerator.prepare(loader)
    else:
        model, optimizer, train_loader, val_loader = accelerator.prepare(
            model, optimizer, train_loader, val_loader
        )
        val_loaders = {"default": val_loader}

    # --- Training info ---
    steps_per_epoch = len(train_loader) // args.gradient_accumulation_steps
    total_steps = steps_per_epoch * args.num_epochs
    print("="*80)
    print("Training Configuration:")
    print(f"  Total training examples: {len(model.train_dataset):,}")
    if multi_lang:
        for lang in sorted(val_loaders.keys()):
            print(f"  Validation [{lang}]: {len(val_loaders[lang].dataset):,} examples")
    else:
        print(f"  Total validation examples: {len(model.val_dataset):,}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print(f"  Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Total optimizer steps: {total_steps}")
    print(f"  Evaluations: ~{total_steps // args.eval_interval}")
    print("="*80)

    # --- Check for previous best (single checkpoint based on avg val loss) ---
    args.out_dir = pathlib.Path(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    lang_names = sorted(val_loaders.keys())
    best_checkpoint_path = args.out_dir / f"{args.dataset}-small-ckpt.pt"
    best_avg_val_loss = float('inf')
    if best_checkpoint_path.exists():
        prev = torch.load(best_checkpoint_path, map_location=args.device, weights_only=False)
        best_avg_val_loss = prev.get('val_loss', float('inf'))
        print(f"Previous best avg val loss: {best_avg_val_loss:.4f}")

    if args.eval_only:
        # Load single best checkpoint and evaluate each language
        if best_checkpoint_path.exists():
            ft_ckpt = torch.load(best_checkpoint_path, map_location=args.device, weights_only=False)
            unwrapped = accelerator.unwrap_model(model)
            unwrapped.load_state_dict(ft_ckpt['model'], strict=True)
            print(f"Loaded best checkpoint (avg val_loss={ft_ckpt.get('val_loss', 'N/A'):.4f})")
            for lang in lang_names:
                losses = evaluate(model, val_loaders[lang], accelerator)
                print(f"  [{lang}] val loss {losses['val']:.4f}, f2 {losses['val_f2']:.4f}, acc {losses['val_accuracy']:.4f}")
        else:
            print("No checkpoint found!")
        return

    # --- Resume from checkpoint if provided ---
    resume_epoch = 0
    resume_batch_idx = 0
    resume_global_step = 0
    resume_running_loss = 0.0

    if args.resume_checkpoint is not None and args.resume_checkpoint.exists():
        print(f"\n[RESUME] Loading resume checkpoint from {args.resume_checkpoint}")
        resume_ckpt = torch.load(args.resume_checkpoint, map_location=args.device, weights_only=False)

        unwrapped = accelerator.unwrap_model(model)
        unwrapped.load_state_dict(resume_ckpt['model'], strict=True)
        optimizer.load_state_dict(resume_ckpt['optimizer'])

        resume_epoch = resume_ckpt.get('epoch', 0)
        resume_batch_idx = resume_ckpt.get('batch_idx', 0)
        resume_global_step = resume_ckpt.get('global_step', 0)
        resume_running_loss = resume_ckpt.get('running_loss', 0.0)

        # Restore best avg val loss from resume checkpoint
        saved_avg = resume_ckpt.get('best_avg_val_loss',
                                     resume_ckpt.get('best_val_losses', resume_ckpt.get('best_val_loss', float('inf'))))
        # Handle legacy dict format: compute average
        if isinstance(saved_avg, dict):
            saved_avg = sum(saved_avg.values()) / len(saved_avg) if saved_avg else float('inf')
        best_avg_val_loss = min(best_avg_val_loss, saved_avg)

        print(f"[RESUME] Restored: epoch={resume_epoch}, batch={resume_batch_idx}, "
              f"global_step={resume_global_step}, best_avg_val_loss={best_avg_val_loss:.4f}")

    # --- Fine-tune ---
    print(f"\nStarting fine-tuning for {args.num_epochs} epochs ({total_steps} optimizer steps)...")
    best_avg_val_loss = finetune(model, train_loader, val_loaders, optimizer, accelerator,
                               best_avg_val_loss, best_checkpoint_path, args,
                               start_epoch=resume_epoch, start_batch_idx=resume_batch_idx,
                               start_global_step=resume_global_step, start_running_loss=resume_running_loss)

    # --- Test set evaluation (single checkpoint, evaluate all languages) ---
    print("\n" + "="*80)
    print("EVALUATING ON TEST SET")
    print("="*80)

    if multi_lang:
        # Load single best checkpoint
        if best_checkpoint_path.exists():
            ft_ckpt = torch.load(best_checkpoint_path, map_location=args.device, weights_only=False)
            unwrapped = accelerator.unwrap_model(model)
            unwrapped.load_state_dict(ft_ckpt['model'], strict=True)
            print(f"Loaded best checkpoint (avg val_loss={ft_ckpt.get('val_loss', 'N/A'):.4f}, step={ft_ckpt.get('global_step', 'N/A')})")
        else:
            print("WARNING: No best checkpoint found, using final model state")

        # Build test loaders for all languages
        test_loaders = prepare_multilang_test(args, model, accelerator)

        all_test_results = {}
        for lang in lang_names:
            test_results = evaluate(model, test_loaders[lang], accelerator)
            all_test_results[lang] = test_results
            print(f"  [{lang}] TEST: loss={test_results['val']:.4f}, "
                  f"f2={test_results['val_f2']:.4f}, acc={test_results['val_accuracy']:.4f}")

            # Save per-language test results
            results_path = args.out_dir / f"{args.dataset}-{lang}-test-results.txt"
            results_path.write_text(
                f"test_loss={test_results['val']:.4f}\n"
                f"test_f2={test_results['val_f2']:.4f}\n"
                f"test_accuracy={test_results['val_accuracy']:.4f}\n"
            )

            # Save per-language test predictions
            model.eval()
            all_preds_list, all_labels_list = [], []
            with torch.no_grad():
                for X, Y in test_loaders[lang]:
                    attention_mask = (X != model.pad_token_id).long()
                    with accelerator.autocast():
                        logits, _ = model(X, attention_mask=attention_mask, labels=Y)
                    all_preds_list.append(logits.argmax(dim=-1).cpu())
                    all_labels_list.append(Y.cpu())
            npz_path = args.out_dir / f"{args.dataset}-{lang}-test-preds.npz"
            np.savez(npz_path,
                     labels=torch.cat(all_labels_list).numpy(),
                     preds=torch.cat(all_preds_list).numpy())

        # Compute average test results
        avg_test_acc = sum(r['val_accuracy'] for r in all_test_results.values()) / len(all_test_results)
        avg_test_f2 = sum(r['val_f2'] for r in all_test_results.values()) / len(all_test_results)

        # Write completion marker
        completion_marker = args.out_dir / f"{args.dataset}-COMPLETED"
        lines = [f"Training completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n"]
        lines.append(f"Best avg val loss: {best_avg_val_loss:.4f}\n")
        for lang in lang_names:
            tr = all_test_results[lang]
            lines.append(f"[{lang}] test_loss={tr['val']:.4f} test_f2={tr['val_f2']:.4f} "
                         f"test_acc={tr['val_accuracy']:.4f}\n")
        lines.append(f"[AVG] test_acc={avg_test_acc:.4f} test_f2={avg_test_f2:.4f}\n")
        completion_marker.write_text("".join(lines))

        print("="*80)
        print("Training complete! Results (single best checkpoint):")
        for lang in lang_names:
            tr = all_test_results[lang]
            print(f"  [{lang}] acc={tr['val_accuracy']:.4f} f2={tr['val_f2']:.4f}")
        print(f"  [AVG] acc={avg_test_acc:.4f} f2={avg_test_f2:.4f}")
        print(f"Checkpoint: {best_checkpoint_path}")
        print(f"Completion marker: {completion_marker}")
        print("="*80)

    else:
        # --- Single-lang test eval (legacy path) ---
        if best_checkpoint_path.exists():
            ft_ckpt = torch.load(best_checkpoint_path, map_location=args.device, weights_only=False)
            unwrapped = accelerator.unwrap_model(model)
            unwrapped.load_state_dict(ft_ckpt['model'], strict=True)
            print(f"Loaded best checkpoint (val_loss={ft_ckpt.get('val_loss', 'N/A'):.4f})")

        test_path = model.data_dir / "test.bin"
        if not test_path.exists():
            print("Tokenizing test data...")
            if args.eval_datasets is not None:
                test_data = _load_concat(args.eval_datasets, "test", args)
            elif args.eval_config is not None:
                test_ds = load_dataset(args.parent_dataset, args.eval_config, cache_dir=str(args.hf_cache))
                test_data = test_ds["test"]
                test_data = normalize_polish_columns(test_data)
            else:
                if args.no_subset:
                    test_ds = load_from_disk(args.parent_dataset) if args.from_disk else load_dataset(args.parent_dataset)
                else:
                    test_ds = load_dataset(args.parent_dataset, args.dataset, cache_dir=str(args.hf_cache))
                test_data = test_ds["test"]
                test_data = normalize_polish_columns(test_data)

            unwrapped = accelerator.unwrap_model(model)
            unwrapped._tokenize_and_save(test_data, test_path)

        test_dataset = XNLIDataset(test_path, model.context_window)
        from torch.utils.data import DataLoader as DL
        test_loader = DL(test_dataset, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers, pin_memory=True, drop_last=False)
        test_loader = accelerator.prepare(test_loader)

        test_results = evaluate(model, test_loader, accelerator)
        print(f"TEST RESULTS: loss={test_results['val']:.4f}, "
              f"f2={test_results['val_f2']:.4f}, accuracy={test_results['val_accuracy']:.4f}")

        results_path = pathlib.Path(args.out_dir) / f"{args.dataset}-test-results.txt"
        results_path.write_text(
            f"test_loss={test_results['val']:.4f}\n"
            f"test_f2={test_results['val_f2']:.4f}\n"
            f"test_accuracy={test_results['val_accuracy']:.4f}\n"
            f"test_samples={len(test_dataset)}\n"
            f"best_val_loss={best_val_loss:.4f}\n"
        )

        model.eval()
        all_preds_list, all_labels_list = [], []
        with torch.no_grad():
            for X, Y in test_loader:
                attention_mask = (X != model.pad_token_id).long()
                with accelerator.autocast():
                    logits, _ = model(X, attention_mask=attention_mask, labels=Y)
                all_preds_list.append(logits.argmax(dim=-1).cpu())
                all_labels_list.append(Y.cpu())
        npz_path = pathlib.Path(args.out_dir) / f"{args.dataset}-test-preds.npz"
        np.savez(npz_path,
                 labels=torch.cat(all_labels_list).numpy(),
                 preds=torch.cat(all_preds_list).numpy())

        completion_marker = pathlib.Path(args.out_dir) / f"{args.dataset}-COMPLETED"
        completion_marker.write_text(
            f"Training completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Best val loss: {best_val_loss:.4f}\n"
            f"Test loss: {test_results['val']:.4f}\n"
            f"Test F2: {test_results['val_f2']:.4f}\n"
            f"Test accuracy: {test_results['val_accuracy']:.4f}\n"
        )

        print("="*80)
        print("Training complete!")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Test accuracy: {test_results['val_accuracy']:.4f}")
        print(f"Test F2: {test_results['val_f2']:.4f}")
        print(f"Checkpoint: {best_checkpoint_path}")
        print(f"Completion marker: {completion_marker}")
        print("="*80)

if __name__ == "__main__":
    main()
