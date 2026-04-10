from transformers import AutoTokenizer

from config import TrainConfig


def load_tokenizer(ctx: TrainConfig) -> AutoTokenizer:
    # noinspection PyTypeChecker
    return AutoTokenizer.from_pretrained(
        ctx.model.tokenizer_name,
        use_fast=True,
    )
