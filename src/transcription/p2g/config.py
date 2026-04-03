from pathlib import Path
from typing import Dict, List

from cascade_config import CascadeConfig
from pydantic import BaseModel


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