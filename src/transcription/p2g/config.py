from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Optional, Type, TypeVar

from cascade_config import CascadeConfig
from pydantic import BaseModel


class ModelConfig(BaseModel):
    model_name: str
    tokenizer_name: str
    hyperparameters: Dict
    checkpoint_prefix: Path
    generation: Dict


class LanguageConfig(BaseModel):
    language: str
    sentence_separators: str


class DatasetConfig(BaseModel):
    name: str
    split: str
    subset: Optional[str]
    input_feature: str
    output_feature: str
    language_feature: str
    language_splits: List[str]


class DatasetBaseConfig(BaseModel):
    definitions: List[DatasetConfig]
    samples: int
    shuffle_buffer: int
    hf_cache: Path
    input_feature: str
    output_feature: str
    train_split_size: float
    output_dataset_name: str
    force_dataset_build: bool
    language_separators: Dict[str, str]


class WandbConfig(BaseModel):
    project: str


class TrainConfig(BaseModel):
    model: ModelConfig
    dataset: DatasetBaseConfig
    wandb: WandbConfig
    random_seed: int

CONFIG_TYPE = TypeVar('CONFIG_TYPE', bound=Type[BaseModel])

def load_configs(files: List[Path], default_config: Path, schema: CONFIG_TYPE = TrainConfig) -> CONFIG_TYPE:
    conf = CascadeConfig(validation_schema=schema.model_json_schema())
    conf.add_json(str(default_config))
    for file in files:
        conf.add_json(str(file))
    ddata = conf.parse()
    return schema.model_validate(ddata)


def generate_argparse(description: str = '') -> ArgumentParser:
    parser = ArgumentParser(description=description)
    ag = parser.add_argument_group('cascading config files')
    ag.add_argument('configs', type=Path, nargs='*', help='cascading config files to use')
    ag.add_argument('--default-config', type=Path, default=Path('config/default.json'), help='default config file')
    return parser
