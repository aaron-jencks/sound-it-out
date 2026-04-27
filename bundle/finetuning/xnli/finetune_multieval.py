"""XNLI multilingual fine-tuning with per-language best-checkpoint saving."""

import argparse
import inspect
import os
import pathlib
import signal
import time
import uuid
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import wandb
from accelerate import Accelerator
from sklearn.metrics import fbeta_score
from torch.utils.data import DataLoader

from models import SUPPORTED_SIZES, load_model_module


torch.backends.cudnn.benchmark = True


STOP_REQUEST = {
    "requested": False,
    "signal_name": None,
}

DEFAULT_EVAL_LANGUAGES = (
    "english",
    "spanish",
    "russian",
    "polish",
    "tamil",
    "malayalam",
    "urdu",
    "hindi",
)

EXPECTED_MISSING_PREFIXES = ("classifier.", "classification_head.")
EXPECTED_UNEXPECTED_KEYS = {"scalars", "lm_head_w"}


def handle_stop_signal(signum, _frame):
    STOP_REQUEST["requested"] = True
    STOP_REQUEST["signal_name"] = signal.Signals(signum).name
    print(
        f"Received {STOP_REQUEST['signal_name']}; will save latest checkpoint at the next safe point."
    )


def install_stop_signal_handlers():
    for sig_name in ("SIGUSR1", "SIGTERM"):
        sig = getattr(signal, sig_name, None)
        if sig is not None:
            signal.signal(sig, handle_stop_signal)


def stop_requested_across_ranks(accelerator):
    flag = torch.tensor(
        [1 if STOP_REQUEST["requested"] else 0],
        dtype=torch.int64,
        device=accelerator.device,
    )
    if accelerator.num_processes > 1:
        from torch.distributed import all_reduce, ReduceOp

        all_reduce(flag, op=ReduceOp.MAX)
    return bool(flag.item())


def save_training_checkpoint(
    checkpoint_path,
    accelerator,
    model,
    optimizer,
    iter_num,
    args,
    checkpoint_kind,
    macro_metrics=None,
    language_metrics=None,
    best_macro_val_loss=None,
    best_language_val_loss=None,
    target_language=None,
):
    checkpoint_path = pathlib.Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    unwrapped = accelerator.unwrap_model(model)

    payload = {
        "model": unwrapped.state_dict(),
        "iter_num": iter_num,
        "args": vars(args),
        "checkpoint_kind": checkpoint_kind,
        "best_macro_val_loss": best_macro_val_loss,
        "best_language_val_loss": dict(best_language_val_loss or {}),
        "target_language": target_language,
    }

    if macro_metrics is not None:
        payload["macro_val_loss"] = macro_metrics["val"]
        payload["macro_val_accuracy"] = macro_metrics["val_accuracy"]
        payload["macro_val_f2"] = macro_metrics["val_f2"]
        payload.setdefault("val_loss", macro_metrics["val"])
        payload.setdefault("val_accuracy", macro_metrics["val_accuracy"])
        payload.setdefault("val_f2", macro_metrics["val_f2"])

    if language_metrics is not None:
        payload["language_metrics"] = {
            lang: {
                "val": metrics["val"],
                "val_accuracy": metrics["val_accuracy"],
                "val_f2": metrics["val_f2"],
            }
            for lang, metrics in language_metrics.items()
        }

    if (
        target_language is not None
        and language_metrics is not None
        and target_language in language_metrics
    ):
        metrics = language_metrics[target_language]
        payload["val_loss"] = metrics["val"]
        payload["val_accuracy"] = metrics["val_accuracy"]
        payload["val_f2"] = metrics["val_f2"]

    try:
        payload["optimizer"] = optimizer.state_dict()
    except Exception as exc:
        print(
            f"Warning: failed to serialize optimizer state for {checkpoint_kind} checkpoint: {exc}"
        )

    torch.save(payload, tmp_path)
    os.replace(tmp_path, checkpoint_path)


def configure_optimizers(model):
    pretrained_params = list(model.pretrained_model.parameters())
    pretrained_param_ids = set(id(p) for p in pretrained_params)
    classifier_params = [p for p in model.parameters() if id(p) not in pretrained_param_ids]

    pretrained_param_dict = {
        f"pretrained.{i}": p for i, p in enumerate(pretrained_params) if p.requires_grad
    }
    classifier_param_dict = {
        f"classifier.{i}": p for i, p in enumerate(classifier_params) if p.requires_grad
    }

    pretrained_decay = [p for p in pretrained_param_dict.values() if p.dim() >= 2]
    pretrained_nodecay = [p for p in pretrained_param_dict.values() if p.dim() < 2]
    classifier_decay = [p for p in classifier_param_dict.values() if p.dim() >= 2]
    classifier_nodecay = [p for p in classifier_param_dict.values() if p.dim() < 2]

    optim_groups = [
        {
            "params": pretrained_decay,
            "weight_decay": model.weight_decay,
            "lr": model.learning_rate * 0.1,
            "lr_scale": 0.1,
        },
        {
            "params": pretrained_nodecay,
            "weight_decay": 0.0,
            "lr": model.learning_rate * 0.1,
            "lr_scale": 0.1,
        },
        {
            "params": classifier_decay,
            "weight_decay": model.weight_decay,
            "lr_scale": 1.0,
        },
        {
            "params": classifier_nodecay,
            "weight_decay": 0.0,
            "lr_scale": 1.0,
        },
    ]

    print(
        f"Pretrained decay params:    {len(pretrained_decay)}, "
        f"{sum(p.numel() for p in pretrained_decay):,} params"
    )
    print(
        f"Pretrained no-decay params: {len(pretrained_nodecay)}, "
        f"{sum(p.numel() for p in pretrained_nodecay):,} params"
    )
    print(
        f"Classifier decay params:    {len(classifier_decay)}, "
        f"{sum(p.numel() for p in classifier_decay):,} params"
    )
    print(
        f"Classifier no-decay params: {len(classifier_nodecay)}, "
        f"{sum(p.numel() for p in classifier_nodecay):,} params"
    )

    device_type = "cuda" if "cuda" in str(model.device) else "cpu"
    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == "cuda"
    extra_args = {"fused": True} if use_fused else {}
    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=model.learning_rate,
        betas=(model.beta1, model.beta2),
        **extra_args,
    )
    print(f"Using torch.optim.AdamW: fused={use_fused}")
    return optimizer


def assert_load_keys_acceptable(load_result, args):
    """Hard-fail unless missing/unexpected keys match the small whitelist."""
    bad_missing = [
        k for k in load_result.missing_keys
        if not k.startswith(EXPECTED_MISSING_PREFIXES)
    ]
    bad_unexpected = [
        k for k in load_result.unexpected_keys
        if k not in EXPECTED_UNEXPECTED_KEYS
    ]
    if bad_missing or bad_unexpected:
        raise RuntimeError(
            "State-dict load reported unexpected key divergence:\n"
            f"  missing (not whitelisted): {bad_missing}\n"
            f"  unexpected (not whitelisted): {bad_unexpected}\n"
            "Refusing to proceed silently. Investigate the pretraining checkpoint vs FT-time model class."
        )
    if load_result.missing_keys:
        print(f"State-dict missing (whitelisted): {load_result.missing_keys}")
    if load_result.unexpected_keys:
        print(f"State-dict unexpected (whitelisted): {load_result.unexpected_keys}")


def load_pretrained_model(args, model, model_module):
    print(
        f"Loading pretrained {args.size.upper()} model "
        f"({args.n_layer}L, {args.n_head}H, {args.n_embd}D) from {args.pretrained_ckpt_path}"
    )
    checkpoint = torch.load(
        args.pretrained_ckpt_path, map_location=args.device, weights_only=False
    )
    if "model" not in checkpoint:
        raise ValueError("Checkpoint missing 'model' key.")

    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for key in list(state_dict.keys()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix):]] = state_dict.pop(key)

    embed_vocab_size = state_dict["embed.weight"].shape[0]
    lm_head_vocab_size = state_dict["lm_head_w"].shape[0]

    print(f"  Embed vocab size:   {embed_vocab_size}")
    print(f"  LM head vocab size: {lm_head_vocab_size}")

    has_scalars = "scalars" in state_dict
    if has_scalars:
        actual_scalars_size = state_dict["scalars"].shape[0]
        print(f"  Scalars size: {actual_scalars_size}")

    pretrained_flex = model_module.GPTFlexAttention(
        vocab_size=embed_vocab_size,
        num_layers=args.n_layer,
        num_heads=args.n_head,
        model_dim=args.n_embd,
        max_seq_len=args.block_size,
    ).to(args.device)

    if has_scalars:
        pretrained_flex.scalars = nn.Parameter(
            torch.zeros(actual_scalars_size, device=args.device)
        )
    pretrained_flex.lm_head_w = nn.Parameter(
        torch.zeros(lm_head_vocab_size, args.n_embd, device=args.device)
    )

    result = pretrained_flex.load_state_dict(state_dict, strict=False)
    assert_load_keys_acceptable(result, args)

    actual_n_embd = state_dict["embed.weight"].shape[1]
    if actual_n_embd != args.n_embd:
        raise ValueError(f"Checkpoint n_embd={actual_n_embd} != args.n_embd={args.n_embd}")

    print("Converting to batched model...")
    pretrained_batched = model_module.GPTBatched.from_pretrained_gpt(pretrained_flex)
    pretrained_batched.to(args.device)

    print(
        f"Loaded checkpoint: {args.n_layer}L, {args.n_head}H, {args.n_embd}D, "
        f"vocab={embed_vocab_size}"
    )
    model.pretrained_model = pretrained_batched
    model.to(args.device)

    if args.torch_compile:
        print("Compiling model with torch.compile...")
        model = torch.compile(model)
        print("Model compiled")
    else:
        print("torch.compile disabled (pass --torch_compile to enable)")
    return model


def resolve_tokenizer_file(args):
    tokenizer_dir = pathlib.Path(args.tokenizer_dir)
    tokenizer_name_path = pathlib.Path(args.tokenizer_name)
    vocab_file = None

    if tokenizer_name_path.is_absolute() and tokenizer_name_path.exists():
        vocab_file = tokenizer_name_path
        print(f"Found tokenizer at: {vocab_file}")

    possible_files = []
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

    if vocab_file is None:
        raise FileNotFoundError(
            f"Could not find tokenizer '{args.tokenizer_name}'. Tried: {possible_files}"
        )

    return vocab_file


def validate_runtime_inputs(args, vocab_file, train_bin_path, val_bin_map):
    ckpt_path = pathlib.Path(args.pretrained_ckpt_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {ckpt_path}")
    if not pathlib.Path(vocab_file).is_file():
        raise FileNotFoundError(f"Tokenizer file not found: {vocab_file}")
    if not pathlib.Path(train_bin_path).is_file():
        raise FileNotFoundError(f"Training bin not found: {train_bin_path}")
    missing = [
        f"{lang}={path}"
        for lang, path in val_bin_map.items()
        if not pathlib.Path(path).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Validation bins not found: {missing}")

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not os.access(out_dir, os.W_OK):
        raise PermissionError(f"Output directory is not writable: {out_dir}")


def log_startup_configuration(args, vocab_file, train_bin_path, val_bin_map, eval_languages):
    print("=" * 80)
    print("Resolved Runtime Inputs")
    print("=" * 80)
    print(f"  Size:                  {args.size}")
    print(f"  Pretrained checkpoint: {pathlib.Path(args.pretrained_ckpt_path).resolve()}")
    print(f"  Tokenizer file:        {pathlib.Path(vocab_file).resolve()}")
    print(f"  Training bin:          {pathlib.Path(train_bin_path).resolve()}")
    print(f"  Validation bins:       {len(val_bin_map)} languages")
    print(f"  Eval languages:        {', '.join(eval_languages)}")
    print(f"  Output directory:      {pathlib.Path(args.out_dir).resolve()}")
    print(f"  Layers/Heads/Dim:      {args.n_layer}/{args.n_head}/{args.n_embd}")
    print(f"  Block size:            {args.block_size}")
    print(f"  Batch size:            {args.batch_size}")
    print(f"  Grad accum:            {args.gradient_accumulation_steps}")
    print(f"  Effective batch size:  {args.batch_size * args.gradient_accumulation_steps}")
    print(f"  Learning rate arg:     {args.learning_rate if args.learning_rate is not None else 'model default'}")
    print(f"  LR schedule:           {'cosine' if args.cosine_lr else 'linear'}{' (no decay)' if args.no_decay_lr else ''}")
    print(f"  torch.compile:         {args.torch_compile}")
    print(f"  Class weights:         {args.use_class_weights}")
    print("=" * 80)


def configure_lr_schedule(model, total_iters):
    total_iters = max(int(total_iters), 1)
    warmup_iters = max(int(model.warmup_iter_ratio * total_iters), 0)
    lr_decay_iters = int(model.lr_decay_iter_ratio * total_iters)
    lr_decay_iters = max(lr_decay_iters, warmup_iters + 1)
    lr_decay_iters = min(lr_decay_iters, total_iters)
    model.warmup_iters = warmup_iters
    model.lr_decay_iters = lr_decay_iters
    return warmup_iters, lr_decay_iters


def maybe_override_max_iters(model, args, max_iters):
    if args.max_iters is None:
        return max_iters
    if args.max_iters <= 0:
        raise ValueError("--max_iters must be > 0 when provided")
    overridden = int(args.max_iters)
    warmup_iters, lr_decay_iters = configure_lr_schedule(model, overridden)
    print(f"Overriding max iterations: {max_iters:,} -> {overridden:,}")
    print(f"Adjusted LR schedule for override: warmup={warmup_iters}, decay={lr_decay_iters}")
    return overridden


def get_lr(learning_rate, min_lr, warmup_iters, lr_decay_iters, iteration, cosine: bool):
    if iteration < warmup_iters:
        return learning_rate * (iteration + 1) / (warmup_iters + 1)
    if lr_decay_iters <= warmup_iters:
        return learning_rate
    if iteration > lr_decay_iters:
        return min_lr
    decay_ratio = (iteration - warmup_iters) / (lr_decay_iters - warmup_iters)
    if cosine:
        coeff = 0.5 * (1.0 + np.cos(np.pi * decay_ratio))
        return min_lr + coeff * (learning_rate - min_lr)
    return learning_rate - decay_ratio * (learning_rate - min_lr)


def get_max_iters(model, gradient_accumulation_steps, num_epochs):
    tokens_per_iter = gradient_accumulation_steps * model.batch_size * model.context_window
    token_count = model.get_token_count()
    max_iters = int(np.ceil(token_count / tokens_per_iter)) * num_epochs
    configure_lr_schedule(model, max_iters)
    return max_iters


@torch.no_grad()
def estimate_loss(model, val_loader, accelerator, eval_iters, pad_token_id):
    model.eval()
    losses = []
    all_preds = []
    all_labels = []

    if eval_iters is None:
        for x, y in val_loader:
            x = x.to(accelerator.device)
            y = y.to(accelerator.device)
            attention_mask = (x != pad_token_id).long()
            logits, loss = model(x, attention_mask=attention_mask, labels=y)
            preds = logits.argmax(dim=-1)
            losses.append(loss.item())
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.cpu().numpy())
    else:
        val_iter = iter(val_loader)
        for _ in range(eval_iters):
            try:
                x, y = next(val_iter)
            except StopIteration:
                val_iter = iter(val_loader)
                x, y = next(val_iter)
            x = x.to(accelerator.device)
            y = y.to(accelerator.device)
            attention_mask = (x != pad_token_id).long()
            logits, loss = model(x, attention_mask=attention_mask, labels=y)
            preds = logits.argmax(dim=-1)
            losses.append(loss.item())
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.cpu().numpy())

    model.train()

    if not all_preds:
        return {"val": float("inf"), "val_accuracy": 0.0, "val_f2": 0.0}

    preds_cat = np.concatenate(all_preds)
    labels_cat = np.concatenate(all_labels)
    accuracy = float((preds_cat == labels_cat).mean())
    try:
        f2 = float(
            fbeta_score(labels_cat, preds_cat, beta=2, average="macro", zero_division=0)
        )
    except Exception:
        f2 = 0.0

    return {
        "val": float(np.mean(losses)),
        "val_accuracy": accuracy,
        "val_f2": f2,
    }


def parse_val_bins(raw_entries):
    flattened_entries = []
    for entry in raw_entries:
        if isinstance(entry, (list, tuple)):
            flattened_entries.extend(entry)
        else:
            flattened_entries.append(entry)

    val_bin_map = {}
    for entry in flattened_entries:
        if "=" not in entry:
            raise ValueError(
                f"Invalid --val_bin entry '{entry}'. Expected language=/path/to/bin"
            )
        language, path = entry.split("=", 1)
        language = language.strip()
        path = path.strip()
        if not language or not path:
            raise ValueError(
                f"Invalid --val_bin entry '{entry}'. Expected language=/path/to/bin"
            )
        if language in val_bin_map:
            raise ValueError(f"Duplicate validation bin for language '{language}'")
        val_bin_map[language] = pathlib.Path(path)
    return val_bin_map


def build_dataloader(dataset, batch_size, num_workers, shuffle, drop_last):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=num_workers > 0,
        drop_last=drop_last,
    )


def load_pretokenized_datasets(model, model_module, train_bin_path, val_bin_map, eval_languages):
    model.train_dataset = model_module.XNLIDataset(train_bin_path, model.context_window)
    val_datasets = {
        language: model_module.XNLIDataset(val_bin_map[language], model.context_window)
        for language in eval_languages
    }
    model.val_dataset = val_datasets[eval_languages[0]]
    return val_datasets


def evaluate_all_languages(model, val_loaders, eval_languages, accelerator, eval_iters, pad_token_id):
    language_metrics = {}
    for language in eval_languages:
        language_metrics[language] = estimate_loss(
            model, val_loaders[language], accelerator, eval_iters, pad_token_id
        )

    macro_metrics = {
        "val": float(np.mean([language_metrics[lang]["val"] for lang in eval_languages])),
        "val_accuracy": float(np.mean([language_metrics[lang]["val_accuracy"] for lang in eval_languages])),
        "val_f2": float(np.mean([language_metrics[lang]["val_f2"] for lang in eval_languages])),
    }
    return macro_metrics, language_metrics


def format_language_metrics(language_metrics, eval_languages):
    return " | ".join(
        f"{language[:2]} loss {language_metrics[language]['val']:.4f} "
        f"f2 {language_metrics[language]['val_f2']:.4f} "
        f"acc {language_metrics[language]['val_accuracy']:.4f}"
        for language in eval_languages
    )


def stable_wandb_run_id(args):
    """Stable run id so resume continues the original W&B run instead of forking."""
    key = f"{args.dataset}|{args.size}|{args.out_dir}"
    return f"xnli-{args.size}-" + uuid.uuid5(uuid.NAMESPACE_URL, key).hex[:16]


def finetune(
    model,
    train_loader,
    val_loaders,
    eval_languages,
    max_iters,
    optimizer,
    accelerator,
    best_macro_val_loss,
    best_language_val_loss,
    best_macro_checkpoint_path,
    best_language_checkpoint_paths,
    latest_checkpoint_path,
    args,
    learning_rate,
    min_lr,
    warmup_iters,
    lr_decay_iters,
    start_iter=0,
):
    if args.wandb_log and accelerator.is_main_process:
        wandb.init(
            project=args.wandb_project,
            name=f"{args.dataset}-{args.size}-multieval",
            id=stable_wandb_run_id(args),
            resume="allow",
            config={
                "size": args.size,
                "n_layer": args.n_layer,
                "n_head": args.n_head,
                "n_embd": args.n_embd,
                "learning_rate": learning_rate,
                "num_epochs": args.num_epochs,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "batch_size": args.batch_size,
                "eval_languages": list(eval_languages),
                "cosine_lr": args.cosine_lr,
                "no_decay_lr": args.no_decay_lr,
                "use_class_weights": args.use_class_weights,
            },
        )

    iter_num = start_iter
    train_iter = iter(train_loader)
    last_macro_metrics = None
    last_language_metrics = None

    while iter_num <= max_iters:
        lr = (
            get_lr(learning_rate, min_lr, warmup_iters, lr_decay_iters, iter_num, args.cosine_lr)
            if not args.no_decay_lr
            else learning_rate
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr * param_group.get("lr_scale", 1.0)

        if iter_num == 0 and args.eval_only:
            if accelerator.is_main_process:
                last_macro_metrics, last_language_metrics = evaluate_all_languages(
                    model, val_loaders, eval_languages, accelerator,
                    args.eval_iters, args.pad_token_id,
                )
                print(
                    f"step {iter_num}: macro val loss {last_macro_metrics['val']:.4f}, "
                    f"macro val f2 {last_macro_metrics['val_f2']:.4f}, "
                    f"macro val accuracy {last_macro_metrics['val_accuracy']:.4f}"
                )
                print(format_language_metrics(last_language_metrics, eval_languages))
            break

        total_loss = 0.0
        for _ in range(args.gradient_accumulation_steps):
            with accelerator.accumulate(model):
                try:
                    x, y = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    x, y = next(train_iter)

                x = x.to(accelerator.device)
                y = y.to(accelerator.device)
                attention_mask = (x != args.pad_token_id).long()

                _, loss = model(x, attention_mask=attention_mask, labels=y)
                loss = loss / args.gradient_accumulation_steps
                total_loss += loss.item()
                accelerator.backward(loss)

        if args.grad_clip != 0.0:
            accelerator.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if stop_requested_across_ranks(accelerator):
            if accelerator.is_main_process:
                print(
                    f"Saving latest checkpoint after stop signal at iter {iter_num}"
                )
                save_training_checkpoint(
                    latest_checkpoint_path, accelerator, model, optimizer,
                    iter_num, args,
                    checkpoint_kind="latest",
                    macro_metrics=last_macro_metrics,
                    language_metrics=last_language_metrics,
                    best_macro_val_loss=best_macro_val_loss,
                    best_language_val_loss=best_language_val_loss,
                )
            accelerator.wait_for_everyone()
            break

        if iter_num % args.eval_interval == 0:
            macro_metrics, language_metrics = evaluate_all_languages(
                model, val_loaders, eval_languages, accelerator,
                args.eval_iters, args.pad_token_id,
            )
            last_macro_metrics = macro_metrics
            last_language_metrics = language_metrics

            if accelerator.is_main_process:
                print(
                    f"step {iter_num}: train loss {total_loss:.4f}, "
                    f"macro val loss {macro_metrics['val']:.4f}, "
                    f"macro val f2 {macro_metrics['val_f2']:.4f}, "
                    f"macro val accuracy {macro_metrics['val_accuracy']:.4f}"
                )
                print(format_language_metrics(language_metrics, eval_languages))

                save_training_checkpoint(
                    latest_checkpoint_path, accelerator, model, optimizer,
                    iter_num, args,
                    checkpoint_kind="latest",
                    macro_metrics=macro_metrics,
                    language_metrics=language_metrics,
                    best_macro_val_loss=best_macro_val_loss,
                    best_language_val_loss=best_language_val_loss,
                )

                if macro_metrics["val"] < best_macro_val_loss:
                    best_macro_val_loss = macro_metrics["val"]
                    print(
                        f"New best macro validation loss: {best_macro_val_loss:.4f}, "
                        f"saving to {best_macro_checkpoint_path}"
                    )
                    save_training_checkpoint(
                        best_macro_checkpoint_path, accelerator, model, optimizer,
                        iter_num, args,
                        checkpoint_kind="best_macro",
                        macro_metrics=macro_metrics,
                        language_metrics=language_metrics,
                        best_macro_val_loss=best_macro_val_loss,
                        best_language_val_loss=best_language_val_loss,
                    )

                for language in eval_languages:
                    if language_metrics[language]["val"] < best_language_val_loss[language]:
                        best_language_val_loss[language] = language_metrics[language]["val"]
                        checkpoint_path = best_language_checkpoint_paths[language]
                        print(
                            f"New best {language} validation loss: "
                            f"{best_language_val_loss[language]:.4f}, "
                            f"saving to {checkpoint_path}"
                        )
                        save_training_checkpoint(
                            checkpoint_path, accelerator, model, optimizer,
                            iter_num, args,
                            checkpoint_kind="best_language",
                            macro_metrics=macro_metrics,
                            language_metrics=language_metrics,
                            best_macro_val_loss=best_macro_val_loss,
                            best_language_val_loss=best_language_val_loss,
                            target_language=language,
                        )

                if args.wandb_log:
                    payload = {
                        "iter": iter_num,
                        "train/loss": total_loss,
                        "macro/val_loss": macro_metrics["val"],
                        "macro/val_f2": macro_metrics["val_f2"],
                        "macro/val_accuracy": macro_metrics["val_accuracy"],
                        "lr": lr,
                        "best_macro_val_loss": best_macro_val_loss,
                    }
                    for language in eval_languages:
                        payload[f"{language}/val_loss"] = language_metrics[language]["val"]
                        payload[f"{language}/val_f2"] = language_metrics[language]["val_f2"]
                        payload[f"{language}/val_accuracy"] = language_metrics[language]["val_accuracy"]
                    wandb.log(payload)

        iter_num += 1

    if accelerator.is_main_process and not stop_requested_across_ranks(accelerator):
        save_training_checkpoint(
            latest_checkpoint_path, accelerator, model, optimizer,
            max(iter_num - 1, 0), args,
            checkpoint_kind="latest",
            macro_metrics=last_macro_metrics,
            language_metrics=last_language_metrics,
            best_macro_val_loss=best_macro_val_loss,
            best_language_val_loss=best_language_val_loss,
        )
    accelerator.wait_for_everyone()
    return best_macro_val_loss, best_language_val_loss


def main():
    parser = argparse.ArgumentParser(
        description="Finetune GPT model with multi-language XNLI validation and per-language best-checkpoint saving."
    )

    parser.add_argument("--size", type=str, required=True, choices=SUPPORTED_SIZES,
                        help="Model size to fine-tune (small / medium / large_standard / xl)")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--pretrained_ckpt_path", type=pathlib.Path, required=True)
    parser.add_argument("--train_bin", type=pathlib.Path, required=True)
    parser.add_argument("--val_bin", type=str, nargs="+", action="append", required=True,
                        help="Validation bin specs: language=/path/to/val.bin (repeatable)")
    parser.add_argument("--eval_languages", type=str, nargs="+",
                        default=list(DEFAULT_EVAL_LANGUAGES))

    parser.add_argument("--hf_cache", type=pathlib.Path, default=pathlib.Path("./cache"))
    parser.add_argument("--out_dir", type=pathlib.Path, default=pathlib.Path("./checkpoints"))
    parser.add_argument("--tokenizer_dir", type=pathlib.Path, default=pathlib.Path("./tokenizers"))
    parser.add_argument("--data_dir", type=pathlib.Path, default=pathlib.Path("./datasets"))
    parser.add_argument("--tokenizer_name", type=str, default="gpt2")

    parser.add_argument("--text_column", type=str, nargs="+", default=["sentence"])
    parser.add_argument("--label_column", type=str, default="label")
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--use_ipa", action="store_true")
    parser.add_argument("--use_class_weights", type=lambda v: str(v).lower() in ("1", "true", "yes"),
                        default=True, help="Apply per-class loss weights derived from train labels (default: true)")

    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--max_iters", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_iters", type=int, default=None,
                        help="Eval batches per language. Default: full val pass.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--no_decay_lr", action="store_true")
    parser.add_argument("--cosine_lr", action="store_true",
                        help="Use cosine LR decay instead of linear (default: linear)")
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pad_token_id", type=int, default=0)

    parser.add_argument("--n_layer", type=int, default=None)
    parser.add_argument("--n_head", type=int, default=None)
    parser.add_argument("--n_embd", type=int, default=None)
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--vocab_size", type=int, default=50257)

    parser.add_argument("--torch_compile", action="store_true",
                        help="Wrap model in torch.compile (off by default; off for xl/large_standard)")

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16", "float32"])

    parser.add_argument("--wandb_project", type=str, default="ipa_xnli_multieval")
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--wandb_log", action="store_true")
    parser.add_argument("--eval_only", action="store_true")

    args = parser.parse_args()
    install_stop_signal_handlers()

    model_module = load_model_module(args.size)
    config = model_module.MODEL_CONFIG
    if args.n_layer is None:
        args.n_layer = config["num_layers"]
    if args.n_head is None:
        args.n_head = config["num_heads"]
    if args.n_embd is None:
        args.n_embd = config["model_dim"]

    mixed_precision = (
        "bf16" if args.dtype == "bfloat16"
        else "fp16" if args.dtype == "float16"
        else "no"
    )
    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    print("=" * 80)
    print(f"GPT-{args.size.upper()} XNLI MULTIEVAL FINE-TUNING")
    print("=" * 80)
    print(f"  Layers: {args.n_layer} | Heads: {args.n_head} | Dim: {args.n_embd}")
    print(f"  Mixed precision: {mixed_precision}")
    print(
        f"  Batch size: {args.batch_size} | GAS: {args.gradient_accumulation_steps} | "
        f"Effective BS: {args.batch_size * args.gradient_accumulation_steps}"
    )
    print(f"  DataLoader workers: {args.num_workers}")
    print(f"  Eval interval: {args.eval_interval} | Eval iters: {args.eval_iters or 'full pass'}")
    print(f"  Eval languages: {', '.join(args.eval_languages)}")
    print("=" * 80)

    val_bin_map = parse_val_bins(args.val_bin)
    eval_languages = list(args.eval_languages)
    missing_languages = [lang for lang in eval_languages if lang not in val_bin_map]
    if missing_languages:
        raise ValueError(f"Missing --val_bin entries for: {missing_languages}")

    vocab_file = resolve_tokenizer_file(args)
    validate_runtime_inputs(args, vocab_file, args.train_bin, val_bin_map)
    log_startup_configuration(args, vocab_file, args.train_bin, val_bin_map, eval_languages)

    model = model_module.GPTClassification(
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
        label_column=args.label_column,
    )

    if args.learning_rate is not None:
        print(f"Overriding learning rate: {model.learning_rate} -> {args.learning_rate}")
        model.learning_rate = args.learning_rate

    val_datasets = load_pretokenized_datasets(
        model, model_module, args.train_bin, val_bin_map, eval_languages
    )
    if args.use_class_weights:
        model._compute_class_weights()

    train_loader = build_dataloader(
        model.train_dataset, args.batch_size, args.num_workers,
        shuffle=True, drop_last=True,
    )
    val_loaders = {
        language: build_dataloader(
            val_datasets[language], args.batch_size, args.num_workers,
            shuffle=False, drop_last=False,
        )
        for language in eval_languages
    }

    model = load_pretrained_model(args, model, model_module)

    pretrained_attr = (
        accelerator.unwrap_model(model).pretrained_model
        if hasattr(model, "_orig_mod")
        else model.pretrained_model
    )
    for param in pretrained_attr.parameters():
        param.requires_grad = True

    optimizer = configure_optimizers(model)

    max_iters = get_max_iters(model, args.gradient_accumulation_steps, args.num_epochs)
    max_iters = maybe_override_max_iters(model, args, max_iters)
    learning_rate = model.learning_rate
    min_lr = model.min_lr
    warmup_iters = model.warmup_iters
    lr_decay_iters = model.lr_decay_iters
    n_train = len(model.train_dataset)
    n_val_by_lang = {language: len(val_datasets[language]) for language in eval_languages}

    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    tokens_per_iter = args.gradient_accumulation_steps * args.batch_size * args.block_size
    print("=" * 80)
    print("Training Configuration:")
    print(f"  Train examples:    {n_train:,}")
    for language in eval_languages:
        print(f"  Val examples [{language}]: {n_val_by_lang[language]:,}")
    print(f"  Total iterations:  {max_iters:,}")
    print(f"  Evaluations:       ~{max_iters // args.eval_interval:,}")
    print(f"  Tokens/iter:       {tokens_per_iter:,}")
    print("=" * 80)

    best_macro_val_loss = float("inf")
    best_language_val_loss = {language: float("inf") for language in eval_languages}
    start_iter = 0

    best_macro_checkpoint_path = (
        pathlib.Path(args.out_dir) / f"{args.dataset}-{args.size}-best_macro.pt"
    )
    latest_checkpoint_path = (
        pathlib.Path(args.out_dir) / f"{args.dataset}-{args.size}-latest.pt"
    )
    best_language_checkpoint_paths = {
        language: pathlib.Path(args.out_dir) / f"{args.dataset}-{args.size}-best_{language}.pt"
        for language in eval_languages
    }

    resume_checkpoint_path = (
        latest_checkpoint_path if latest_checkpoint_path.exists() else best_macro_checkpoint_path
    )
    if resume_checkpoint_path.exists():
        print(f"Resuming from checkpoint: {resume_checkpoint_path}")
        previous = torch.load(
            resume_checkpoint_path, map_location=args.device, weights_only=False
        )
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.load_state_dict(previous["model"], strict=False)
        if "optimizer" in previous:
            optimizer.load_state_dict(previous["optimizer"])
        else:
            print("Warning: resume checkpoint does not contain optimizer state; continuing fresh.")
        best_macro_val_loss = previous.get(
            "best_macro_val_loss",
            previous.get("macro_val_loss", previous.get("val_loss", float("inf"))),
        )
        restored_best_language = previous.get("best_language_val_loss", {})
        for language in eval_languages:
            if language in restored_best_language:
                best_language_val_loss[language] = restored_best_language[language]
        start_iter = int(previous.get("iter_num", -1)) + 1
        checkpoint_kind = previous.get("checkpoint_kind", "unknown")
        print(f"Resumed {checkpoint_kind} checkpoint from iter {start_iter}")
        print(f"Previous best macro val loss: {best_macro_val_loss:.4f}")
    else:
        print(
            f"No existing checkpoint at {latest_checkpoint_path} or "
            f"{best_macro_checkpoint_path}; starting fresh."
        )

    print(f"Starting fine-tuning for {max_iters:,} iterations...")
    print(f"Warmup: {warmup_iters} | LR decay: {lr_decay_iters}")

    best_macro_val_loss, best_language_val_loss = finetune(
        model, train_loader, val_loaders, eval_languages, max_iters, optimizer,
        accelerator,
        best_macro_val_loss, best_language_val_loss,
        best_macro_checkpoint_path, best_language_checkpoint_paths,
        latest_checkpoint_path,
        args,
        learning_rate, min_lr, warmup_iters, lr_decay_iters,
        start_iter=start_iter,
    )

    print("=" * 80)
    print("Training complete!")
    print(f"Best macro validation loss: {best_macro_val_loss:.4f}")
    print(f"Macro checkpoint:  {best_macro_checkpoint_path}")
    print(f"Latest checkpoint: {latest_checkpoint_path}")
    for language in eval_languages:
        print(f"Best {language} checkpoint: {best_language_checkpoint_paths[language]}")
    print("=" * 80)


if __name__ == "__main__":
    main()
