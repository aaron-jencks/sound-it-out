from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List

from cascade_config import CascadeConfig
from pydantic import BaseModel


class ModelConfig(BaseModel):
    model_name: str
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


def parse_args() -> TrainConfig:
    ap = ArgumentParser()
    ap.add_argument('configs', type=Path, nargs='*', help='config files')
    ap.add_argument('--default-config', type=Path, default=Path('config/default.json'), help='default config file')
    args = ap.parse_args()
    return load_configs(args.configs, args.default_config)


def main():
    config = parse_args()


if __name__ == '__main__':
    main()