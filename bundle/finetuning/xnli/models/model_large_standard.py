from ._dataset import XNLIDataset
from ._gpt_qkvo import GPTBatched, GPTClassification, GPTFlexAttention

MODEL_CONFIG = {
    "size": "large_standard",
    "num_layers": 20,
    "num_heads": 10,
    "model_dim": 1280,
    "head_dim": 128,
}

__all__ = [
    "GPTBatched",
    "GPTClassification",
    "GPTFlexAttention",
    "XNLIDataset",
    "MODEL_CONFIG",
]
