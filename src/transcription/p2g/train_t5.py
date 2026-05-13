import logging

import torch
from transformers import set_seed
import wandb

from config import TrainConfig
from dataset import create_dataset
from setup import setup_wandb, generate_trainer, parse_args


logger = logging.getLogger(__file__)


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