import datetime as dt
import logging
import os
from pathlib import Path
from typing import Optional, Tuple

from datasets import Dataset, DatasetDict
import evaluate
import numpy as np
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, \
    Seq2SeqTrainer, Seq2SeqTrainingArguments, set_seed, Trainer
from transformers.utils import is_flash_attn_2_available
import wandb

from common import load_tokenizer, format_language_marker
from config import load_configs, TrainConfig, generate_argparse, DatasetFeatureConfig
from dataset import create_dataset, preprocess_dataset


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)  # Stop terminal vomit


def parse_args() -> TrainConfig:
    ap = generate_argparse('trains a p2g model')
    args = ap.parse_args()
    config = load_configs(args.configs, args.default_config)
    if config.cpus < 0:
        config.cpus = os.cpu_count()
    return config


def generate_wandb_run_name(ctx: TrainConfig) -> str:
    d_string = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"fine-tuning-{d_string}"


def setup_wandb(ctx: TrainConfig):
    wandb.init(
        project=ctx.wandb.project,
        name=generate_wandb_run_name(ctx),
        config=ctx.wandb.settings,
    )


def setup_tokenizer(ctx: TrainConfig, model: Optional[AutoModelForSeq2SeqLM],
        train_ds: Optional[Dataset], eval_ds: Optional[Dataset],
        train_ds_def: Optional[DatasetFeatureConfig] = None,
        eval_ds_def: Optional[DatasetFeatureConfig] = None
) -> AutoTokenizer:
    languages = []

    if train_ds is not None:
        long_names = train_ds.unique(train_ds_def.language_feature)
        for ln in long_names:
            if ln not in train_ds_def.language_map:
                raise ValueError(f"language feature {ln} not found in language map")
            languages.append(train_ds_def.language_map[ln])

    if eval_ds is not None:
        long_names = eval_ds.unique(eval_ds_def.language_feature)
        for ln in long_names:
            if ln not in eval_ds_def.language_map:
                raise ValueError(f"language feature {ln} not found in language map")
            languages.append(eval_ds_def.language_map[ln])

    tokenizer = load_tokenizer(ctx, model, list(set(languages)))

    return tokenizer


def generate_trainer(
        ctx: TrainConfig,
        train_ds: Optional[Dataset], eval_ds: Optional[Dataset],
        train_ds_def: Optional[DatasetFeatureConfig] = None,
        eval_ds_def: Optional[DatasetFeatureConfig] = None,
        model_checkpoint: Optional[Path] = None
) -> Tuple[Trainer, AutoTokenizer]:
    logger.info('setting up training pipeline')

    if train_ds is None and eval_ds is None:
        raise ValueError("train_ds and eval_ds cannot both be None")

    if train_ds is not None:
        if train_ds_def is None:
            raise ValueError("train_ds_def cannot be None if train_ds is not None")
        if train_ds_def.language_feature is None:
            raise ValueError("training dataset language feature cannot be None")

    if eval_ds is not None:
        if eval_ds_def is None:
            raise ValueError("eval_ds_def cannot be None if train_ds is not None")
        if eval_ds_def.language_feature is None:
            raise ValueError("evaluation dataset language feature cannot be None")

    if not is_flash_attn_2_available():
        model = AutoModelForSeq2SeqLM.from_pretrained(
            ctx.model.model_name if model_checkpoint is None else str(model_checkpoint),
            device_map="auto"
        )
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            ctx.model.model_name if model_checkpoint is None else str(model_checkpoint),
            device_map="auto",
            torch_dtype=torch.bfloat16 if ctx.model.supports_bf16 else torch.float16,
            attn_implementation="flash_attention_2"
        )

    if model_checkpoint is None:
        for k, v in ctx.model.generation.items():
            setattr(model.generation_config, k, v)

    # noinspection PyTypeChecker
    tokenizer = setup_tokenizer(ctx, model, train_ds, eval_ds, train_ds_def, eval_ds_def)

    if train_ds is not None:
        train_ds = preprocess_dataset(
            ctx, train_ds_def,
            train_ds, tokenizer
        )
    if eval_ds is not None:
        eval_ds = preprocess_dataset(
            ctx,
            eval_ds_def,
            eval_ds, tokenizer
        )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
    )

    bleu_metric = evaluate.load('sacrebleu')

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred

        if isinstance(predictions, tuple):
            predictions = predictions[0]

        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]

        # sacrebleu expects references as list[list[str]]
        decoded_labels = [[label] for label in decoded_labels]

        result = bleu_metric.compute(
            predictions=decoded_preds,
            references=decoded_labels,
        )

        prediction_lens = [
            np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions
        ]

        return {
            "bleu": round(result["score"], 4),
            "gen_len": round(float(np.mean(prediction_lens)), 4),
        }

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(ctx.model.checkpoint_prefix),
        load_best_model_at_end=True,
        predict_with_generate=True,
        report_to="wandb",
        seed=ctx.random_seed,
        data_seed=ctx.random_seed,
        **ctx.model.hyperparameters,
    )

    return Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    ), tokenizer


def train(ctx: TrainConfig):
    logger.info("creating dataset")
    train_ds, train_ds_config, eval_ds, eval_ds_config = create_dataset(ctx)

    trainer, _ = generate_trainer(ctx, train_ds, eval_ds, train_ds_config, eval_ds_config)

    if ctx.wandb.enabled:
        setup_wandb(ctx)

    trainer.train()
    trainer.save_model(str(ctx.model.checkpoint_prefix / 'best'))

    if ctx.wandb.enabled:
        wandb.finish()


def main():
    config = parse_args()
    set_seed(config.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    train(config)


if __name__ == '__main__':
    main()