from collections.abc import Mapping
from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Optional, Tuple, Union

from datasets import Dataset, DatasetDict, IterableDataset, load_dataset, load_from_disk
from pydantic import ValidationError
from transformers import PreTrainedTokenizer

from common import format_language_marker
from config import ConstructedDatasetConfig, CoreDatasetConfig, NamedSplitDatasetFeatureConfig, TokenizerConfig, TrainConfig
from setup import setup_tokenizer


logger = logging.getLogger(__file__)
PREPROCESSED_FEATURES = ("input_ids", "attention_mask", "labels")


def get_dataset_path(ctx: TrainConfig) -> Tuple[datetime, Path]:
    dt = ctx.dataset.last_date if ctx.dataset.last_date else datetime.now()
    dir_name = f"{dt.isoformat()}-{ctx.dataset.output_dataset_name}"
    output_path_name = ctx.dataset.hf_cache / dir_name
    return dt, output_path_name


def load_hf_dataset(definition: CoreDatasetConfig, cache_loc: Path, streaming: bool = True) -> Union[IterableDataset, Dataset]:
    if definition.subset is None:
        logger.info(f'loading {definition.name} from HF dataset.')
        dataset = load_dataset(
            definition.name, split=definition.split,
            cache_dir=str(cache_loc),
            streaming=streaming
        )
    else:
        logger.info(f'loading {definition.name}/{definition.subset} from HF subset.')
        dataset = load_dataset(
            definition.name, definition.subset, split=definition.split,
            cache_dir=str(cache_loc),
            streaming=streaming
        )
    return dataset


def load_metadata(output_dir: Path) -> Optional[ConstructedDatasetConfig]:
    if not output_dir.exists():
        logger.debug(f"cache directory: {output_dir} did not exist!")
        return None
    try:
        with open(output_dir / 'custom_metadata.json', 'r') as f:
            metadata = json.load(f)
        return ConstructedDatasetConfig.model_validate(metadata)
    except (FileNotFoundError, json.JSONDecodeError, ValidationError) as e:
        logger.debug(f"failed to read metadata from {output_dir}/custom_metadata.json: {e}")
        return None


def load_tokenizer_metadata(output_dir: Path) -> Optional[TokenizerConfig]:
    if not output_dir.exists():
        logger.debug(f"cache directory: {output_dir} did not exist!")
        return None
    try:
        with open(output_dir / 'tokenizer_metadata.json', 'r') as f:
            metadata = json.load(f)
        return TokenizerConfig.model_validate(metadata)
    except (FileNotFoundError, json.JSONDecodeError, ValidationError) as e:
        logger.debug(f"failed to read tokenizer metadata from {output_dir}/tokenizer_metadata.json: {e}")
        return None


def build_train_split_context(ctx: TrainConfig) -> NamedSplitDatasetFeatureConfig:
    return NamedSplitDatasetFeatureConfig(
        name=ctx.dataset.output_dataset_name,
        split="train",
        input_feature=ctx.dataset.input_feature,
        output_feature=ctx.dataset.output_feature,
        language_feature=ctx.dataset.language_feature,
        language_map=ctx.dataset.language_map,
    )


def validate_preprocessed_dataset(ds: Dataset, split_name: str, source: str) -> None:
    missing = [feature for feature in PREPROCESSED_FEATURES if feature not in ds.column_names]
    if missing:
        missing_features = ", ".join(missing)
        raise RuntimeError(
            f"{source} split '{split_name}' is not preprocessed; missing columns: {missing_features}"
        )


def preprocess_dataset(
        ctx: TrainConfig,
        ds_ctx: NamedSplitDatasetFeatureConfig,
        ds: Dataset,
        tokenizer: PreTrainedTokenizer
) -> Dataset:
    def preprocess_function(examples):
        inputs = [
            f"{format_language_marker(ds_ctx.language_map[lang])} {text}"
            for lang, text in zip(examples[ds_ctx.language_feature], examples[ds_ctx.input_feature])
        ]

        model_inputs = tokenizer(
            inputs,
            truncation=True,
            max_length=ctx.model.tokenizer.max_sequence_length,
        )

        outputs = [
            f"{format_language_marker(ds_ctx.language_map[lang])} {text}"
            for lang, text in zip(examples[ds_ctx.language_feature], examples[ds_ctx.output_feature])
        ]

        labels = tokenizer(
            outputs,
            truncation=True,
            max_length=ctx.model.tokenizer.max_sequence_length,
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    cache_file = ctx.dataset.hf_cache / ds_ctx.name / f"{ds_ctx.split}_tokens.arrow"

    return ds.map(
        preprocess_function,
        batched=True,
        remove_columns=ds.column_names,
        num_proc=ctx.cpus,
        cache_file_name=str(cache_file)
    )


def prepare_datasets(
        ctx: TrainConfig,
        train_ds: Dataset,
        train_ds_ctx: NamedSplitDatasetFeatureConfig,
        eval_ds: Dataset,
        eval_ds_ctx: NamedSplitDatasetFeatureConfig
) -> Tuple[Dataset, NamedSplitDatasetFeatureConfig, Dataset, NamedSplitDatasetFeatureConfig]:
    tokenizer = setup_tokenizer(ctx, None, train_ds, eval_ds, train_ds_ctx, eval_ds_ctx)
    logger.info("preprocessing train split")
    train_ds = preprocess_dataset(ctx, train_ds_ctx, train_ds, tokenizer)
    logger.info("preprocessing eval split")
    eval_ds = preprocess_dataset(ctx, eval_ds_ctx, eval_ds, tokenizer)
    return train_ds, train_ds_ctx, eval_ds, eval_ds_ctx


def load_existing_dataset(
        ctx: TrainConfig
) -> Tuple[Dataset, NamedSplitDatasetFeatureConfig, Dataset, NamedSplitDatasetFeatureConfig]:
    if ctx.dataset.last_date is None:
        raise RuntimeError("dataset.last_date must be set when loading an existing dataset artifact")

    _, output_path_name = get_dataset_path(ctx)
    if not output_path_name.exists():
        raise FileNotFoundError(f"dataset artifact does not exist: {output_path_name}")

    meta = load_metadata(output_path_name)
    if meta is None:
        raise RuntimeError(f"dataset metadata was missing or unreadable: {output_path_name / 'custom_metadata.json'}")
    if meta != ctx.dataset:
        raise RuntimeError(f"dataset metadata did not match current config: {output_path_name}")

    tokenizer_meta = load_tokenizer_metadata(output_path_name)
    if tokenizer_meta is None:
        raise RuntimeError(f"tokenizer metadata was missing or unreadable: {output_path_name / 'tokenizer_metadata.json'}")
    if tokenizer_meta != ctx.model.tokenizer:
        raise RuntimeError(f"tokenizer metadata did not match current config: {output_path_name}")

    output_ds = load_from_disk(str(output_path_name))
    train_ds_ctx = build_train_split_context(ctx)

    if ctx.evaluation.datasets is None or len(ctx.evaluation.datasets) == 0:
        if not isinstance(output_ds, DatasetDict) and not isinstance(output_ds, Mapping):
            raise RuntimeError(
                "expected dataset artifact to already contain named splits; "
                "load_existing_dataset no longer creates train/test splits"
            )
        if "train" not in output_ds or "test" not in output_ds:
            raise RuntimeError(
                "expected dataset artifact to contain 'train' and 'test' splits; "
                "load_existing_dataset no longer creates them automatically"
            )

        eval_ds_ctx = train_ds_ctx.model_copy(deep=True)
        eval_ds_ctx.split = "test"
        train_ds = output_ds["train"]
        eval_ds = output_ds["test"]
        validate_preprocessed_dataset(train_ds, "train", str(output_path_name))
        validate_preprocessed_dataset(eval_ds, "test", str(output_path_name))
        return train_ds, train_ds_ctx, eval_ds, eval_ds_ctx

    if len(ctx.evaluation.datasets) > 1:
        logger.warning("more than one evaluation script found, using the first one for training")

    if isinstance(output_ds, DatasetDict) or isinstance(output_ds, Mapping):
        if "train" not in output_ds:
            raise RuntimeError("expected dataset artifact to contain a 'train' split")
        train_ds = output_ds["train"]
    else:
        raise RuntimeError(
            "expected dataset artifact to contain a 'train' split; "
            "training no longer accepts an unsplit dataset artifact"
        )

    validate_preprocessed_dataset(train_ds, "train", str(output_path_name))

    eval_ds = load_hf_dataset(ctx.evaluation.datasets[0], ctx.dataset.hf_cache, streaming=False)
    validate_preprocessed_dataset(eval_ds, ctx.evaluation.datasets[0].split, ctx.evaluation.datasets[0].name)
    return train_ds, train_ds_ctx, eval_ds, ctx.evaluation.datasets[0]
