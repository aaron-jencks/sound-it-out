from argparse import ArgumentParser
import datetime as dt
import logging
import os
from pathlib import Path

import evaluate
import numpy as np
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, \
    Seq2SeqTrainer, Seq2SeqTrainingArguments, set_seed, GenerationConfig
import wandb

from config import load_configs, TrainConfig, generate_argparse
from dataset import create_dataset


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> TrainConfig:
    ap = generate_argparse('trains a p2g model')
    args = ap.parse_args()
    return load_configs(args.configs, args.default_config)


def generate_wandb_run_name(ctx: TrainConfig) -> str:
    d_string = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"fine-tuning-{d_string}"


def train(ctx: TrainConfig):
    logger.info("creating dataset")
    ds = create_dataset(ctx)

    logger.info('setting up training pipeline')

    # noinspection PyTypeChecker
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(
        ctx.model.tokenizer_name,
        use_fast=True,
    )

    def preprocess_function(examples):
        model_inputs = tokenizer(
            examples["ipa"],
            truncation=True,
            max_length=256,
        )

        labels = tokenizer(
            text_target=examples["text"],
            truncation=True,
            max_length=256,
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_train = ds['train'].map(
        preprocess_function,
        batched=True,
        remove_columns=ds['train'].column_names,
        num_proc=os.cpu_count(),
        cache_file_name=str(ctx.dataset.hf_cache / ctx.dataset.output_dataset_name / 'tokens/tokenized_train.arrow')
    )

    tokenized_eval = ds['test'].map(
        preprocess_function,
        batched=True,
        remove_columns=ds['test'].column_names,
        num_proc=os.cpu_count(),
        cache_file_name=str(ctx.dataset.hf_cache / ctx.dataset.output_dataset_name / 'tokens/tokenized_test.arrow')
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

    wandb.init(
        project=ctx.wandb.project,
        name=generate_wandb_run_name(ctx),
    )

    gen_config = {}
    for k, v in ctx.model.generation.items():
        gen_config[f'generation_{k}'] = v

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(ctx.model.checkpoint_prefix),
        eval_strategy="steps",
        eval_steps=0.01,
        save_strategy="steps",
        save_steps=0.01,
        save_total_limit=2,
        metric_for_best_model="bleu",
        load_best_model_at_end=True,
        logging_steps=0.01,
        predict_with_generate=True,
        **gen_config,
        report_to="wandb",
        seed=ctx.random_seed,
        data_seed=ctx.random_seed,
        **ctx.model.hyperparameters,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(str(ctx.model.checkpoint_prefix / 'best'))

    wandb.finish()


def main():
    config = parse_args()
    set_seed(config.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    train(config)


if __name__ == '__main__':
    main()