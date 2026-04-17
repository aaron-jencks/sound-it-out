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
import wandb

from common import load_tokenizer
from config import load_configs, TrainConfig, generate_argparse, CoreDatasetConfig
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


def generate_trainer(
        ctx: TrainConfig,
        train_ds: Optional[Dataset], eval_ds: Optional[Dataset],
        eval_ds_def: Optional[CoreDatasetConfig] = None,
        model_checkpoint: Optional[Path] = None
) -> Tuple[Trainer, AutoTokenizer]:
    logger.info('setting up training pipeline')

    # noinspection PyTypeChecker
    tokenizer: AutoTokenizer = load_tokenizer(ctx)

    cache_prefix = ctx.dataset.hf_cache / ctx.dataset.output_dataset_name
    if train_ds is not None:
        train_ds = preprocess_dataset(
            ctx, ctx.dataset,
            train_ds, tokenizer,
            cache_prefix / 'tokens/tokenized_train.arrow'
        )
    if eval_ds is not None:
        eval_ds = preprocess_dataset(
            ctx,
            eval_ds_def if eval_ds_def is not None else ctx.dataset,
            eval_ds, tokenizer,
            cache_prefix / 'tokens/tokenized_eval.arrow'
        )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        ctx.model.model_name if model_checkpoint is None else str(model_checkpoint),
        device_map="auto"
    )

    if model_checkpoint is None:
        for k, v in ctx.model.generation.items():
            setattr(model.generation_config, k, v)

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
        eval_strategy="steps",
        eval_steps=20,
        save_strategy="steps",
        save_steps=40,
        save_total_limit=2,
        metric_for_best_model="bleu",
        load_best_model_at_end=True,
        logging_steps=10,
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
    train_ds, eval_ds = create_dataset(ctx)

    trainer, _ = generate_trainer(ctx, train_ds, eval_ds)

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