import datetime
from typing import List, Optional

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from transcription.p2g.config import TokenizerConfig


def get_timestamp_string(dt: Optional[datetime.datetime] = None) -> str:
    return (dt if dt is not None else datetime.datetime.now()).strftime("%Y-%m-%d-%H-%M-%S")


def format_language_marker(s: str) -> str:
    return f"<lang:{s}>"


def load_tokenizer(ctx: TokenizerConfig, model: Optional[AutoModelForSeq2SeqLM], extra_langs: List[str]) -> AutoTokenizer:
    # noinspection PyTypeChecker
    tok = AutoTokenizer.from_pretrained(
        ctx.name,
        use_fast=True,
    )
    tok.add_special_tokens({
        "additional_special_tokens": list(map(format_language_marker, extra_langs)),
    })
    if model is not None:
        model.resize_token_embeddings(len(tok))
    return tok
