from typing import List, Optional

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from config import TrainConfig


def format_language_marker(s: str) -> str:
    return f"<lang:{s}>"


def load_tokenizer(ctx: TrainConfig, model: Optional[AutoModelForSeq2SeqLM], extra_langs: List[str]) -> AutoTokenizer:
    # noinspection PyTypeChecker
    tok = AutoTokenizer.from_pretrained(
        ctx.model.tokenizer.name,
        use_fast=True,
    )
    tok.add_special_tokens({
        "additional_special_tokens": list(map(format_language_marker, extra_langs)),
    })
    if model is not None:
        model.resize_token_embeddings(len(tok))
    return tok
