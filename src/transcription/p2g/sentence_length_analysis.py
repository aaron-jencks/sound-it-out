import json
import logging
import multiprocessing as mp
import os
from pathlib import Path
from typing import List, Optional, Tuple

from datasets import Dataset
import numpy as np
from pydantic import BaseModel
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, set_seed

from config import load_configs, TrainConfig, generate_argparse, NamedSplitDatasetFeatureConfig
from dataset import create_dataset, preprocess_dataset
from train_t5 import setup_tokenizer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)  # Stop terminal vomit


def parse_args() -> Tuple[TrainConfig, Optional[Path]]:
    ap = generate_argparse('analyzes the p2g dataset to determine some statistics about the distribution of sentence lengths in bytes')
    ap.add_argument("--output-filename", type=Path, default=None, help="The location to store the statistics")
    args = ap.parse_args()
    config = load_configs(args.configs, args.default_config)
    if config.cpus < 0:
        config.cpus = os.cpu_count()
    return config, args.output_filename


class DatasetStats(BaseModel):
    records: int
    mean: float
    stddev: float
    min: float
    max: float
    median: float
    p75: float
    p90: float
    p95: float
    p99: float


def generate_statistics(l: List[int]) -> DatasetStats:
    med, p75, p90, p95, p99 = list(map(float, np.percentile(l, [50, 75, 90, 95, 99])))
    return DatasetStats(
        records=len(l),
        mean=float(np.mean(l)),
        stddev=float(np.std(l)),
        min=np.min(l),
        max=np.max(l),
        median=med,
        p75=p75,
        p90=p90,
        p95=p95,
        p99=p99,
    )


def get_lengths(example):
    return len(example["input_ids"]), len(example["labels"])


def analyze_dataset(
        ctx: TrainConfig, tokenizer: AutoTokenizer,
        ds_def: NamedSplitDatasetFeatureConfig, ds: Dataset
) -> Tuple[DatasetStats, DatasetStats]:
    pre_ds = preprocess_dataset(ctx, ds_def, ds, tokenizer)
    with mp.Pool(ctx.cpus) as pool:
        results = list(tqdm(
            pool.imap_unordered(get_lengths, pre_ds, chunksize=10000),
            total=len(pre_ds),
            desc="computing lengths"
        ))
        input_lengths, label_lengths = zip(*results)
    return generate_statistics(input_lengths), generate_statistics(label_lengths)


def main():
    config, output_path = parse_args()
    set_seed(config.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.info("creating dataset")
    train_ds, train_ds_config, eval_ds, eval_ds_config = create_dataset(config)

    logger.info("setting up tokenizer")
    tokenizer = setup_tokenizer(config, None, train_ds, eval_ds, train_ds_config, eval_ds_config)

    logger.info("analyzing train datasets")
    train_input_stats, train_label_stats = analyze_dataset(config, tokenizer, train_ds_config, train_ds)

    logger.info("analyzing eval datasets")
    eval_input_stats, eval_label_stats = analyze_dataset(config, tokenizer, eval_ds_config, eval_ds)

    print("Training Dataset:")
    print("Inputs:")
    print(train_input_stats)
    print("Labels:")
    print(train_label_stats)

    print("Eval Dataset:")
    print("Inputs:")
    print(eval_input_stats)
    print("Labels:")
    print(eval_label_stats)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "train": {
                "inputs": train_input_stats.model_dump(),
                "labels": train_label_stats.model_dump(),
            },
            "eval": {
                "inputs": eval_input_stats.model_dump(),
                "labels": eval_label_stats.model_dump(),
            }
        }
        with open(output_path, "w+") as f:
            json.dump(data, f, indent=2)


if __name__ == '__main__':
    main()