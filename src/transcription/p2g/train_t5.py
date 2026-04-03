from argparse import ArgumentParser
import logging
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq

from config import load_configs, TrainConfig


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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