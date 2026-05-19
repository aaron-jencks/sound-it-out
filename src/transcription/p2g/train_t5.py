import datetime as dt
import logging
from pathlib import Path
from typing import Optional, Tuple

from datasets import Dataset
import evaluate
import numpy as np
import torch
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, set_seed, \
    Seq2SeqTrainer, Seq2SeqTrainingArguments, Trainer
from transformers.utils import is_flash_attn_2_available
import wandb

from transcription.p2g.common import get_timestamp_string
from transcription.p2g.config import DatasetConfig, TrainConfig
from transcription.p2g.dataset_loading import load_existing_dataset
from transcription.p2g.setup import parse_args, setup_model, setup_tokenizer, setup_wandb


logger = logging.getLogger(__file__)


def get_checkpoint_path(ctx: TrainConfig, ts: dt.datetime) -> Path:
    dname = f"{get_timestamp_string(ts)}-{ctx.run_name}"
    return ctx.model.checkpoint_prefix / dname


def generate_trainer(
        ctx: TrainConfig,
        train_ds: Optional[Dataset], eval_ds: Optional[Dataset],
        train_ds_def: Optional[DatasetConfig] = None,
        eval_ds_def: Optional[DatasetConfig] = None,
        model_checkpoint: Optional[Path] = None,
        run_timestamp: Optional[dt.datetime] = None,
) -> Tuple[Trainer, AutoTokenizer, dt.datetime]:
    logger.info('setting up training pipeline')

    if run_timestamp is None:
        run_timestamp = dt.datetime.now()

    path_dt = get_timestamp_string(run_timestamp)
    logger.info(f"timestamp for this run will be: {path_dt}")

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
    chrf_metric = evaluate.load('chrf')
    rouge_metric = evaluate.load('rouge')

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

        chrf_score = chrf_metric.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            word_order=0,
            char_order=0,
            beta=2
        )

        rouge_score = rouge_metric.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True,
        )

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
            "chrf": round(chrf_score["score"], 4),
            "rouge1": round(rouge_score["rouge1"], 4),
            "rouge2": round(rouge_score["rouge2"], 4),
            "rougeL": round(rouge_score["rougeL"], 4),
            "rougeLsum": round(rouge_score["rougeLsum"], 4),
        }

    checkpoint_prefix = get_checkpoint_path(ctx, run_timestamp)
    logger.info(f"saving checkpoint to {checkpoint_prefix}")

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(checkpoint_prefix),
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
    ), tokenizer, run_timestamp


def train(ctx: TrainConfig):
    logger.info("loading existing dataset")
    train_ds, train_ds_config, eval_ds, eval_ds_config = load_existing_dataset(ctx)

    trainer, _, rts = generate_trainer(ctx, train_ds, eval_ds, train_ds_config, eval_ds_config)

    if ctx.wandb.enabled:
        setup_wandb(ctx)

    trainer.train()

    checkpoint_prefix = get_checkpoint_path(ctx, rts) / 'best'
    trainer.save_model(str(checkpoint_prefix))
    logger.info(f"saving best checkpoint to {checkpoint_prefix}")

    if ctx.wandb.enabled:
        wandb.finish()


def main():
    config = parse_args("trains a p2g model", schema=TrainConfig)
    set_seed(config.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    train(config)


if __name__ == '__main__':
    main()
