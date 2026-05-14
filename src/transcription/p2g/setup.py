import datetime as dt
import logging
import os
from pathlib import Path
from typing import Optional, Type, TypeVar

from datasets import Dataset
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import wandb

from common import load_tokenizer
from config import DatasetConfig, EvaluationConfig, TrainConfig, generate_argparse, load_configs


logger = logging.getLogger(__file__)
TConfig = TypeVar("TConfig", TrainConfig, EvaluationConfig)


def setup_logging(debug: bool = False):
    logging.basicConfig(level=logging.DEBUG if debug else logging.INFO)
    logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("http").setLevel(logging.WARNING)
    logging.getLogger("fsspec").setLevel(logging.WARNING)
    logging.getLogger("filelock").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("git").setLevel(logging.WARNING)


def parse_args(description: str, schema: Type[TConfig] = TrainConfig) -> TConfig:
    ap = generate_argparse(description)
    args = ap.parse_args()
    setup_logging(args.debug)
    config = load_configs(args.configs, args.default_config, schema=schema)
    if config.cpus < 0:
        config.cpus = os.cpu_count()
    return config


def generate_wandb_run_name(ctx: EvaluationConfig) -> str:
    d_string = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"fine-tuning-{d_string}"


def setup_wandb(ctx: EvaluationConfig):
    wandb.init(
        project=ctx.wandb.project,
        name=generate_wandb_run_name(ctx),
        config=ctx.wandb.settings,
    )


def setup_tokenizer(ctx: EvaluationConfig, model: Optional[AutoModelForSeq2SeqLM],
        train_ds: Optional[Dataset], eval_ds: Optional[Dataset],
        train_ds_def: Optional[DatasetConfig] = None,
        eval_ds_def: Optional[DatasetConfig] = None
) -> AutoTokenizer:
    languages = set()

    if train_ds_def is not None:
        languages.update(train_ds_def.language_map.values())

    if eval_ds_def is not None:
        languages.update(eval_ds_def.language_map.values())

    tokenizer = load_tokenizer(ctx, model, sorted(languages))

    return tokenizer


def setup_model(ctx: EvaluationConfig, model_checkpoint: Optional[Path], attn: Optional[str] = None) -> AutoModelForSeq2SeqLM:
    if attn is None:
        return AutoModelForSeq2SeqLM.from_pretrained(
            ctx.model.model_name if model_checkpoint is None else str(model_checkpoint),
            device_map="auto",
            torch_dtype=torch.bfloat16 if ctx.model.supports_bf16 else torch.float16,
        )
    return AutoModelForSeq2SeqLM.from_pretrained(
        ctx.model.model_name if model_checkpoint is None else str(model_checkpoint),
        device_map="auto",
        torch_dtype=torch.bfloat16 if ctx.model.supports_bf16 else torch.float16,
        attn_implementation=attn
    )
