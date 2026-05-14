import logging
from pathlib import Path
from typing import Optional, Tuple, Union

from datasets import Dataset, DatasetDict, IterableDataset, load_dataset, load_from_disk

from config import DatasetConfig, TrainConfig


logger = logging.getLogger(__file__)
PREPROCESSED_FEATURES = ("input_ids", "attention_mask", "labels")


def load_hf_dataset(
        definition: DatasetConfig,
        streaming: bool = True,
        cache_loc: Optional[Path] = None,
) -> Union[IterableDataset, Dataset]:
    kwargs = {"streaming": streaming}
    if cache_loc is not None:
        kwargs["cache_dir"] = str(cache_loc)
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


def load_saved_dataset(definition: DatasetConfig) -> Dataset:
    dataset_path = Path(definition.name)
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
    train_ds = load_saved_dataset(ctx.train_dataset)
    eval_ds = load_saved_dataset(ctx.evaluation_dataset)
    return train_ds, ctx.train_dataset, eval_ds, ctx.evaluation_dataset
