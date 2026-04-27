from ._dataset import XNLIDataset
from ._gpt_qkv_small import (
    GPTBatchedSmall as GPTBatched,
    GPTClassificationSmall as GPTClassification,
    GPTFlexAttentionSmall as GPTFlexAttention,
)

MODEL_CONFIG = {
    "size": "small",
    "num_layers": 12,
    "num_heads": 6,
    "model_dim": 768,
    "head_dim": 128,
}

__all__ = [
    "GPTBatched",
    "GPTClassification",
    "GPTFlexAttention",
    "XNLIDataset",
    "MODEL_CONFIG",
]
