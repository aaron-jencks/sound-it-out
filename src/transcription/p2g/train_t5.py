from argparse import ArgumentParser
import datetime as dt
import logging
from pathlib import Path

import evaluate
import numpy as np
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments
import wandb

from config import load_configs, TrainConfig
from dataset import create_dataset


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> TrainConfig:
    ap = ArgumentParser('trains a p2g model')
    ap.add_argument('configs', type=Path, nargs='*', help='config files')
    ap.add_argument('--default-config', type=Path, default=Path('config/default.json'), help='default config file')
    args = ap.parse_args()
    return load_configs(args.configs, args.default_config)


def generate_wandb_run_name(ctx: TrainConfig) -> str:
    d_string = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"fine-tuning-{d_string}"


def train(ctx: TrainConfig):
    logger.info("creating dataset")
    ds = create_dataset(ctx)

    logger.info('setting up training pipeline')

    tokenizer = AutoTokenizer.from_pretrained(
        ctx.model.tokenizer_name,
        use_fast=True,
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        ctx.model.model_name,
        device_map="auto"
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

    wandb.init(
        project=ctx.wandb.project,
        name=generate_wandb_run_name(ctx),
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(ctx.model.checkpoint_prefix),
        eval_strategy="steps",
        eval_steps=0.01,
        save_strategy="steps",
        save_steps=0.01,
        save_total_limit=2,
        metric_for_best_model="bleu",
        load_best_model_at_end=True,
        weight_decay=0.01,
        logging_steps=0.01,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        num_train_epochs=3,
        report_to="wandb",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=ds['train'],
        eval_dataset=ds['test'],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(str(ctx.model.checkpoint_prefix / 'best'))

    wandb.finish()


def main():
    config = parse_args()
    train(config)


if __name__ == '__main__':
    main()