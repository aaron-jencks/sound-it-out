from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Type, TypeVar, Union

from cascade_config import CascadeConfig
from pydantic import BaseModel


class GridSearchConfig(BaseModel):
    max_parallel_jobs: int
    polling_interval: float


class TokenizerConfig(BaseModel):
    name: str
    max_sequence_length: int


class ModelConfig(BaseModel):
    model_name: str
    tokenizer: TokenizerConfig
    hyperparameters: Dict
    checkpoint_prefix: Path
    generation: Dict
    supports_bf16: bool


class DatasetFeatureConfig(BaseModel):
    input_feature: str
    output_feature: str
    language_feature: Optional[str]


class NamedDatasetConfig(DatasetFeatureConfig):
    name: str


class DatasetConfig(NamedDatasetConfig):
    name: str
    split: str
    subset: Optional[str] = None
    languages: List[str]
    language_map: Dict[str, str]


class SplitRatioConfig(BaseModel):
    eval_ratio: float
    max_eval_size: int


class WandbConfig(BaseModel):
    enabled: bool
    project: str
    settings: Dict


class CoreConfig(BaseModel):
    random_seed: int
    cpus: int


class PreprocessingConfig(CoreConfig):
    tokenizer: TokenizerConfig
    output_dataset: NamedDatasetConfig
    input_datasets: List[DatasetConfig]
    samples: int
    shuffle_buffer: int
    hf_cache: Path
    splits: SplitRatioConfig


class EvaluationConfig(CoreConfig):
    model: ModelConfig
    evaluation_dataset: DatasetConfig
    results_prefix: Path
    wandb: WandbConfig


class TrainConfig(EvaluationConfig):
    train_dataset: DatasetConfig
    grid_search: GridSearchConfig


def load_configs(files: Optional[List[Path]], default_config: Path, schema: Type[BaseModel] = TrainConfig) -> BaseModel:
    conf = CascadeConfig(validation_schema=schema.model_json_schema())
    conf.add_json(str(default_config))
    if files is not None:
        for file in files:
            conf.add_json(str(file))
    ddata = conf.parse()
    return schema.model_validate(ddata)


def generate_argparse(description: str = '') -> ArgumentParser:
    parser = ArgumentParser(description=description)
    ag = parser.add_argument_group('cascading config files')
    ag.add_argument('configs', type=Path, nargs='*', help='cascading config files to use')
    ag.add_argument('--default-config', type=Path, default=Path('transcription/p2g/config/default_pre.json'), help='default config file')
    parser.add_argument('--debug', action='store_true', help='enable debug print statements')
    return parser
