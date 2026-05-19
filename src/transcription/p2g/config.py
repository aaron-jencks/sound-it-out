from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional, Type, TypeVar, Union

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


class DatasetMetaConfig(BaseModel):
    name: str
    split: str
    subset: Optional[str] = None
    languages: List[str]
    language_map: Optional[Dict[str, str]] = None


class PartialDatasetFeatureConfig(BaseModel):
    output_feature: str
    language_feature: Optional[str]


class FullDatasetFeatureConfig(PartialDatasetFeatureConfig):
    input_feature: str


class NamedFullDatasetFeatureConfig(FullDatasetFeatureConfig):
    name: str


class DatasetConfig(DatasetMetaConfig, FullDatasetFeatureConfig):
    pass


class ConstructionInputDatasetConfig(DatasetMetaConfig, PartialDatasetFeatureConfig):
    streaming: bool = True


class RomanizationConfig(BaseModel):
    uroman_path: Path
    perl_path: str = "perl"


class TransformationConfig(BaseModel):
    type: Literal["phonemize", "romanize"]
    espeak_path: Optional[Path] = None
    romanization: Optional[RomanizationConfig] = None
    transform_worker_count: int = -1


class SplitRatioConfig(BaseModel):
    eval_ratio: float
    max_eval_size: int


class WandbConfig(BaseModel):
    enabled: bool
    project: str
    settings: Dict


class CoreConfig(BaseModel):
    hf_cache: Path
    random_seed: int
    cpus: int


class PreprocessingConfig(CoreConfig):
    tokenizer: TokenizerConfig
    output_dataset: NamedFullDatasetFeatureConfig
    input_datasets: List[ConstructionInputDatasetConfig]
    words_per_sample: int
    transform: TransformationConfig
    samples: int
    shuffle_buffer: int
    transform_batch_size: int
    splits: SplitRatioConfig


class EvaluationConfig(CoreConfig):
    model: ModelConfig
    evaluation_dataset: DatasetConfig
    results_prefix: Path
    wandb: WandbConfig


class TrainConfig(EvaluationConfig):
    train_dataset: DatasetConfig
    grid_search: Optional[GridSearchConfig] = None


def load_configs(files: Optional[List[Path]], default_config: Path, schema: Type[BaseModel] = TrainConfig) -> BaseModel:
    conf = CascadeConfig(validation_schema=schema.model_json_schema())
    conf.add_json(str(default_config))
    if files is not None:
        for file in files:
            conf.add_json(str(file))
    ddata = conf.parse()
    return schema.model_validate(ddata)


def generate_argparse(
        description: str = '',
        default_config: Path = Path("transcription/p2g/config/default_core.json"),
) -> ArgumentParser:
    parser = ArgumentParser(description=description)
    ag = parser.add_argument_group('cascading config files')
    ag.add_argument('configs', type=Path, nargs='*', help='cascading config files to use')
    ag.add_argument('--default-config', type=Path, default=default_config, help='default config file')
    parser.add_argument('--debug', action='store_true', help='enable debug print statements')
    return parser
