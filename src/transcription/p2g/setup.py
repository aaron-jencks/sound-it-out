import datetime as dt
import logging
import os
from pathlib import Path
from typing import Optional

from datasets import Dataset
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from turbot5 import T5ForConditionalGeneration, T5Config
import wandb

from common import load_tokenizer
from config import load_configs, TrainConfig, generate_argparse, DatasetFeatureConfig


logger = logging.getLogger(__file__)


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


def parse_args(description: str) -> TrainConfig:
    ap = generate_argparse(description)
    args = ap.parse_args()
    setup_logging(args.debug)
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
    languages = set()

    if train_ds_def is not None:
        languages.update(train_ds_def.language_map.values())

    if eval_ds_def is not None:
        languages.update(eval_ds_def.language_map.values())

    tokenizer = load_tokenizer(ctx, model, sorted(languages))

    return tokenizer


def setup_model(ctx: TrainConfig, model_checkpoint: Optional[Path], attn: str = "basic", device: Optional[str] = None) -> T5ForConditionalGeneration:
    model = T5ForConditionalGeneration.from_pretrained(
        ctx.model.model_name if model_checkpoint is None else str(model_checkpoint),
        torch_dtype=torch.bfloat16 if ctx.model.supports_bf16 else torch.float16,
        attention_type=attn
    )
    if device is not None:
        return model.to(device)
    return model
