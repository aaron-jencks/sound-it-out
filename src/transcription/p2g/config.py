from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Optional, Type, TypeVar, Union

from cascade_config import CascadeConfig
from pydantic import BaseModel


class GridSearchConfig(BaseModel):
    max_parallel_jobs: int
    polling_interval: float


class ModelConfig(BaseModel):
    model_name: str
    tokenizer_name: str
    hyperparameters: Dict
    checkpoint_prefix: Path
    generation: Dict


class LanguageConfig(BaseModel):
    language: str
    sentence_separators: str


class DatasetFeatureConfig(BaseModel):
    input_feature: str
    output_feature: str


class CoreDatasetConfig(DatasetFeatureConfig):
    name: str
    split: str
    subset: Optional[str]
    prediction_file: Optional[Path]


class ConstructedDatasetDefinitionConfig(CoreDatasetConfig):
    language_feature: Optional[str]
    language_splits: List[str]


class ConstructedDatasetConfig(DatasetFeatureConfig):
    definitions: List[ConstructedDatasetDefinitionConfig]
    samples: int
    shuffle_buffer: int
    hf_cache: Path
    train_split_size: float
    output_dataset_name: str
    force_dataset_build: bool
    language_separators: Dict[str, str]


class WandbConfig(BaseModel):
    enabled: bool
    project: str
    settings: Dict


class EvaluationConfig(BaseModel):
    datasets: Optional[List[CoreDatasetConfig]]
    result_prefix: Path


class TrainConfig(BaseModel):
    model: ModelConfig
    dataset: ConstructedDatasetConfig
    grid_search: GridSearchConfig
    evaluation: EvaluationConfig
    wandb: WandbConfig
    random_seed: int
    cpus: int

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
    ag.add_argument('--default-config', type=Path, default=Path('config/default.json'), help='default config file')
    return parser
