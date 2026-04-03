from argparse import ArgumentParser
import logging
from pathlib import Path
from typing import Dict, List

from cascade_config import CascadeConfig
from pydantic import BaseModel
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelConfig(BaseModel):
    model_name: str
    tokenizer_name: str
    hyperparameters: Dict
    checkpoint_prefix: Path


class DatasetConfig(BaseModel):
    name: str
    samples: int


class TrainConfig(BaseModel):
    model: ModelConfig
    dataset: DatasetConfig


def load_configs(files: List[Path], default_config: Path) -> TrainConfig:
    conf = CascadeConfig(validation_schema=TrainConfig.model_json_schema())
    conf.add_json(str(default_config))
    for file in files:
        conf.add_json(str(file))
    ddata = conf.parse()
    return TrainConfig.model_validate(ddata)


def parse_args() -> TrainConfig:
    ap = ArgumentParser('trains a p2g model')
    ap.add_argument('configs', type=Path, nargs='*', help='config files')
    ap.add_argument('--default-config', type=Path, default=Path('config/default.json'), help='default config file')
    args = ap.parse_args()
    return load_configs(args.configs, args.default_config)


def train(ctx: TrainConfig):
    logger.info('setting up training pipeline')

    tokenizer = AutoTokenizer.from_pretrained(
        ctx.model.tokenizer_name,
        use_fast=True,
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        ctx.model.model_name,
        device_map="auto"
    )


def main():
    config = parse_args()


if __name__ == '__main__':
    main()