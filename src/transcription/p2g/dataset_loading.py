import os
import logging
from pathlib import Path
from typing import Optional, Tuple, Union

from datasets import Dataset, DatasetDict, IterableDataset, load_dataset, load_from_disk

from transcription.p2g.config import ConstructionInputDatasetConfig, DatasetConfig, TrainConfig


logger = logging.getLogger(__file__)
PREPROCESSED_FEATURES = ("input_ids", "attention_mask", "labels")


def configure_hf_cache(cache_loc: Path) -> Path:
    cache_root = cache_loc.resolve()
    datasets_cache = cache_root / "datasets"
    hub_cache = cache_root / "hub"

    datasets_cache.mkdir(parents=True, exist_ok=True)
    hub_cache.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(cache_root)
    os.environ["HF_DATASETS_CACHE"] = str(datasets_cache)
    os.environ["HF_HUB_CACHE"] = str(hub_cache)

    logger.info(
        "configured Hugging Face cache paths: HF_HOME=%s HF_DATASETS_CACHE=%s HF_HUB_CACHE=%s",
        cache_root,
        datasets_cache,
        hub_cache,
    )
    return datasets_cache


def load_hf_dataset(
        definition: Union[DatasetConfig, ConstructionInputDatasetConfig],
        streaming: bool = True,
        cache_loc: Optional[Path] = None,
) -> Union[IterableDataset, Dataset]:
    kwargs = {"streaming": streaming}
    if cache_loc is not None:
        kwargs["cache_dir"] = str(configure_hf_cache(cache_loc))
    if definition.subset is None:
        logger.info(f'loading {definition.name} from HF dataset.')
        dataset = load_dataset(
            definition.name, split=definition.split,
            **kwargs,
        )
    else:
        logger.info(f'loading {definition.name}/{definition.subset} from HF subset.')
        dataset = load_dataset(
            definition.name, definition.subset, split=definition.split,
            **kwargs,
        )
    return dataset


def validate_preprocessed_dataset(ds: Dataset, split_name: str, source: str) -> None:
    missing = [feature for feature in PREPROCESSED_FEATURES if feature not in ds.column_names]
    if missing:
        missing_features = ", ".join(missing)
        raise RuntimeError(
            f"{source} split '{split_name}' is not preprocessed; missing columns: {missing_features}"
        )


def load_saved_dataset(definition: DatasetConfig, prefix: Path) -> Dataset:
    dataset_path = prefix / definition.name
    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset artifact does not exist: {dataset_path}")

    output_ds = load_from_disk(str(dataset_path))
    if isinstance(output_ds, DatasetDict):
        if definition.split not in output_ds:
            available_splits = ", ".join(sorted(output_ds.keys()))
            raise RuntimeError(
                f"dataset artifact {dataset_path} does not contain split '{definition.split}'. "
                f"Available splits: {available_splits}"
            )
        ds = output_ds[definition.split]
    else:
        ds = output_ds

    validate_preprocessed_dataset(ds, definition.split, str(dataset_path))
    return ds


def load_existing_dataset(
        ctx: TrainConfig
) -> Tuple[Dataset, DatasetConfig, Dataset, DatasetConfig]:
    train_ds = load_saved_dataset(ctx.train_dataset, ctx.hf_cache)
    eval_ds = load_saved_dataset(ctx.evaluation_dataset, ctx.hf_cache)
    return train_ds, ctx.train_dataset, eval_ds, ctx.evaluation_dataset
