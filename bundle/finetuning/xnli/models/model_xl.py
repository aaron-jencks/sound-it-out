from ._dataset import XNLIDataset
from ._gpt_qkvo import GPTBatched, GPTClassification, GPTFlexAttention

MODEL_CONFIG = {
    "size": "xl",
    "num_layers": 24,
    "num_heads": 12,
    "model_dim": 1536,
    "head_dim": 128,
}

__all__ = [
    "GPTBatched",
    "GPTClassification",
    "GPTFlexAttention",
    "XNLIDataset",
    "MODEL_CONFIG",
]
