import datetime as dt
import logging
import os
from pathlib import Path
from typing import Optional, Tuple

from datasets import Dataset, DatasetDict
import evaluate
import numpy as np
import torch
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, set_seed, \
    Seq2SeqTrainer, Seq2SeqTrainingArguments, Trainer
from transformers.utils import is_flash_attn_2_available
from turbot5 import T5ForConditionalGeneration, T5Config
import wandb

from config import TrainConfig, DatasetFeatureConfig
from dataset import load_existing_dataset
from setup import setup_wandb, parse_args, setup_model, setup_tokenizer


logger = logging.getLogger(__file__)


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
        model = setup_model(ctx, model_checkpoint)
    else:
        try:
            model = setup_model(ctx, model_checkpoint, "flash_attention_2")
        except ValueError:
            logger.warning("failed to use flash attention, trying scaled dot product")
            try:
                model = setup_model(ctx, model_checkpoint, "sdpa")
            except ValueError:
                logger.warning("failed to use sdpa attention, using normal attention")
                model = setup_model(ctx, model_checkpoint)

    if model_checkpoint is None:
        for k, v in ctx.model.generation.items():
            setattr(model.generation_config, k, v)

    # noinspection PyTypeChecker
    tokenizer = setup_tokenizer(ctx, model, train_ds, eval_ds, train_ds_def, eval_ds_def)

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
    logger.info("loading existing dataset")
    train_ds, train_ds_config, eval_ds, eval_ds_config = load_existing_dataset(ctx)

    trainer, _ = generate_trainer(ctx, train_ds, eval_ds, train_ds_config, eval_ds_config)

    if ctx.wandb.enabled:
        setup_wandb(ctx)

    trainer.train()
    trainer.save_model(str(ctx.model.checkpoint_prefix / 'best'))

    if ctx.wandb.enabled:
        wandb.finish()


def main():
    config = parse_args("trains a p2g model")
    set_seed(config.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    train(config)


if __name__ == '__main__':
    main()
