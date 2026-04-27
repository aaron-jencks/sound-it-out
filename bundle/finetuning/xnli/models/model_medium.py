from ._dataset import XNLIDataset
from ._gpt_qkvo import GPTBatched, GPTClassification, GPTFlexAttention

MODEL_CONFIG = {
    "size": "medium",
    "num_layers": 16,
    "num_heads": 8,
    "model_dim": 1024,
    "head_dim": 128,
}

__all__ = [
    "GPTBatched",
    "GPTClassification",
    "GPTFlexAttention",
    "XNLIDataset",
    "MODEL_CONFIG",
]
