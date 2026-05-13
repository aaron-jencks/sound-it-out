from collections import Counter
from datetime import datetime
import json
import logging
from pathlib import Path
import re
import shutil
from typing import List, Optional, Tuple, Union
import uuid

from datasets import load_dataset, IterableDataset, Dataset, load_from_disk, ClassLabel
from pydantic import ValidationError
from transformers import PreTrainedTokenizer, set_seed

from common import format_language_marker
from config import TrainConfig, CoreDatasetConfig, NamedSplitDatasetFeatureConfig, ConstructedDatasetConfig, TokenizerConfig
from setup import parse_args, setup_tokenizer


logger = logging.getLogger(__file__)


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


def generate_punctuation_regex(separators: str) -> str:
    if len(separators) == 0:
        return ''
    elif len(separators) == 1:
        return f'[{separators[0]}]+'
    result = "(?:"
    valid_characters = f"[{separators}]+"
    result += f"{valid_characters}\\W+|\\W+{valid_characters})"
    return result


def determine_eval_size(ctx: TrainConfig, ds: Dataset) -> Union[int, float]:
    dslen = len(ds)
    if dslen * ctx.dataset.splits.eval_ratio > ctx.dataset.splits.max_eval_size:
        return ctx.dataset.splits.max_eval_size
    return ctx.dataset.splits.eval_ratio


def split_or_load_eval_dataset(ctx: TrainConfig, train_ds: Dataset, force: bool = False) -> Tuple[Dataset, NamedSplitDatasetFeatureConfig, Dataset, NamedSplitDatasetFeatureConfig]:
    train_ds_ctx = NamedSplitDatasetFeatureConfig(
        name=ctx.dataset.output_dataset_name,
        split="train",
        input_feature=ctx.dataset.input_feature,
        output_feature=ctx.dataset.output_feature,
        language_feature=ctx.dataset.language_feature,
        language_map=ctx.dataset.language_map,
    )
    if ctx.evaluation.datasets is None or len(ctx.evaluation.datasets) == 0:
        language_col = ctx.dataset.language_feature
        logger.info("generating train test splits")
        split_output_path_name = ctx.dataset.hf_cache / f"{ctx.dataset.output_dataset_name}_split"
        if not force and split_output_path_name.exists():
            logger.info("split dataset exists, loading from cache")
            output_ds = load_from_disk(str(split_output_path_name))
            train_ds = output_ds["train"]
            test_ds = output_ds["test"]
        else:
            if ctx.dataset.stratified:
                logger.warning("using stratified sampling for splitting, but FYI, this is VERYYY slow.")
                logger.info("generating numeric language column")

                if split_output_path_name.exists():
                    logger.info("removing stale cache entry")
                    shutil.rmtree(split_output_path_name)

                languages = sorted(set(train_ds[language_col]))
                language_to_id = {language: i for i, language in enumerate(languages)}
                temp_language_feature = str(uuid.uuid4())

                logger.info("encoding language column")
                train_ds = train_ds.map(
                    lambda batch: {
                        temp_language_feature: [language_to_id[language] for language in batch[language_col]]
                    },
                    batched=True,
                    num_proc=ctx.cpus,
                )

                logger.info("casting encoded language column")
                train_ds = train_ds.cast_column(
                    temp_language_feature,
                    ClassLabel(names=languages),
                )

                logger.info("generating train test splits")
                split_size = determine_eval_size(ctx, train_ds)
                logger.info(f"using test size of {split_size}")

                output_ds = train_ds.train_test_split(
                    seed=ctx.random_seed,
                    test_size=split_size,
                    stratify_by_column=temp_language_feature,
                )

                logger.info("removing old columns")
                output_ds["train"] = output_ds["train"].remove_columns(temp_language_feature)
                output_ds["test"] = output_ds["test"].remove_columns(temp_language_feature)
            else:
                logger.info("generating train test splits")
                split_size = determine_eval_size(ctx, train_ds)
                logger.info(f"using test size of {split_size}")

                output_ds = train_ds.train_test_split(
                    seed=ctx.random_seed,
                    test_size=split_size,
                )

            logger.info("saving dataset to disk")
            output_ds.save_to_disk(split_output_path_name)

            train_ds = output_ds["train"]
            test_ds = output_ds["test"]

        logger.info("generating eval context")
        test_ds_ctx = train_ds_ctx.model_copy(deep=True)
        test_ds_ctx.split = "test"
    else:
        if len(ctx.evaluation.datasets) > 1:
            logger.warning("more than one evaluation script found, using the first one for training")
        train_ds = train_ds
        test_ds = load_hf_dataset(ctx.evaluation.datasets[0], ctx.dataset.hf_cache, streaming=False)
        test_ds_ctx = ctx.evaluation.datasets[0]
    return train_ds, train_ds_ctx, test_ds, test_ds_ctx


def store_metadata(output_dir: Path, metadata: ConstructedDatasetConfig):
    with open(output_dir / 'custom_metadata.json', 'w+') as f:
        s = metadata.model_dump_json()
        f.write(s)


def store_tokenizer_metadata(output_dir: Path, metadata: TokenizerConfig):
    with open(output_dir / 'tokenizer_metadata.json', 'w+') as f:
        s = metadata.model_dump_json()
        f.write(s)


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


def persist_updated_config(ctx: TrainConfig):
    if ctx.dataset.update_config is None:
        return

    logger.info(f"writing updated dataset config to {ctx.dataset.update_config}")
    ctx.dataset.update_config.update_file.parent.mkdir(parents=True, exist_ok=True)
    ds = ctx.dataset.last_date.isoformat() if ctx.dataset.last_date else None
    output_dict = {
        "dataset": {
            "last_date": ds,
        }
    }
    if ctx.dataset.update_config.update_file.exists():
        with open(ctx.dataset.update_config.update_file, 'r') as f:
            output_dict = json.load(f)
            if "dataset" not in output_dict:
                output_dict["dataset"] = {
                    "last_date": ds,
                }
            else:
                output_dict["dataset"]["last_date"] = ds
    with open(ctx.dataset.update_config.update_file, 'w+') as f:
        json.dump(output_dict, f, indent=ctx.dataset.update_config.indent)


def assign_new_dataset_timestamp(ctx: TrainConfig) -> Tuple[datetime, Path]:
    ctx.dataset.last_date = datetime.now()
    return get_dataset_path(ctx)


def prepare_datasets(
        ctx: TrainConfig,
        train_ds: Dataset, train_ds_ctx: NamedSplitDatasetFeatureConfig,
        eval_ds: Dataset, eval_ds_ctx: NamedSplitDatasetFeatureConfig
) -> Tuple[Dataset, NamedSplitDatasetFeatureConfig, Dataset, NamedSplitDatasetFeatureConfig]:
    tokenizer = setup_tokenizer(ctx, None, train_ds, eval_ds, train_ds_ctx, eval_ds_ctx)
    logger.info("preprocessing train split")
    train_ds = preprocess_dataset(ctx, train_ds_ctx, train_ds, tokenizer)
    logger.info("preprocessing eval split")
    eval_ds = preprocess_dataset(ctx, eval_ds_ctx, eval_ds, tokenizer)
    return train_ds, train_ds_ctx, eval_ds, eval_ds_ctx


def load_existing_dataset(ctx: TrainConfig) -> Tuple[Dataset, NamedSplitDatasetFeatureConfig, Dataset, NamedSplitDatasetFeatureConfig]:
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
    train_ds, train_ds_ctx, eval_ds, eval_ds_ctx = split_or_load_eval_dataset(ctx, output_ds)
    return prepare_datasets(ctx, train_ds, train_ds_ctx, eval_ds, eval_ds_ctx)


def create_dataset(ctx: TrainConfig) -> Tuple[Dataset, NamedSplitDatasetFeatureConfig, Dataset, NamedSplitDatasetFeatureConfig]:
    logger.info("creating dataset")

    output_dt, output_path_name = get_dataset_path(ctx)
    if not ctx.dataset.force_dataset_build:
        if ctx.dataset.last_date is not None:
            logger.info(f'checking for cached dataset at {output_path_name}')
            meta = load_metadata(output_path_name)
            tokenizer_meta = load_tokenizer_metadata(output_path_name)
            logger.debug(f"found metadata: {meta}")
            logger.debug(f"found tokenizer metadata: {tokenizer_meta}")
            if meta is not None:
                logger.debug(f"existing metadata: {ctx.dataset}")
                logger.debug(f"existing tokenizer metadata: {ctx.model.tokenizer}")
                if meta == ctx.dataset and tokenizer_meta == ctx.model.tokenizer:
                    logger.info('cached dataset metadata matches')
                    output_ds = load_from_disk(str(output_path_name))
                    return split_or_load_eval_dataset(ctx, output_ds)
                logger.warning("cached dataset or tokenizer metadata did not match current config, rebuilding with a new timestamp")
            else:
                logger.warning("cached dataset metadata was missing or unreadable, rebuilding with a new timestamp")
        else:
            logger.info("no dataset timestamp configured, creating a new dataset")
    else:
        logger.info("force dataset build enabled, creating a new dataset")

    output_dt, output_path_name = assign_new_dataset_timestamp(ctx)
    logger.info(f"building dataset artifact for timestamp {output_dt.isoformat()}")
    logger.info(f"dataset will be written to {output_path_name}")

    # count how many samples we intend to have
    total_samples = 0
    languages = set()
    for ds in ctx.dataset.definitions:
        for lang in ds.language_splits:
            if lang in languages:
                continue
            total_samples += ctx.dataset.samples
            languages.add(lang)
    target_languages = sorted(languages)

    documents = 0
    samples = 0

    # process the datasets
    final_dataset = {
        ctx.dataset.input_feature: [],
        ctx.dataset.output_feature: [],
        ctx.dataset.language_feature: []
    }
    language_counts = {lang: 0 for lang in target_languages}
    language_documents_seen = {lang: 0 for lang in target_languages}
    language_documents_skipped = {lang: 0 for lang in target_languages}
    language_mismatch_skips = {lang: 0 for lang in target_languages}
    language_empty_sentence_skips = {lang: 0 for lang in target_languages}
    raw_language_counts = Counter()

    def check_language_status():
        for lang in target_languages:
            if language_counts[lang] < ctx.dataset.samples:
                logger.info(
                    f"language {lang} is waiting for {ctx.dataset.samples - language_counts[lang]} samples "
                    f"(seen_docs={language_documents_seen[lang]}, skipped_docs={language_documents_skipped[lang]}, "
                    f"mismatch_skips={language_mismatch_skips[lang]}, empty_sentence_skips={language_empty_sentence_skips[lang]})"
                )

    def check_if_languages_full(ds_language_splits: List[str]) -> bool:
        if all([(lang in language_counts and language_counts[lang] >= ctx.dataset.samples) for lang in
                ds_language_splits]):
            logger.info(f"all languages in dataset are full, moving on...")
            return True
        return False

    for ds_def in ctx.dataset.definitions:
        logger.info(f"processing dataset: {ds_def.name}")
        logger.info(f"processed {documents} documents and found {samples}/{total_samples} samples")
        check_language_status()

        if ds_def.language_feature is None and ds_def.subset is None:
            raise ValueError("cannot process dataset, subset and or language_feature must be defined")

        ds_language_splits = ds_def.language_splits
        if ds_def.language_feature is None:
            logger.info(f'no language feature found, using the subset "{ds_def.subset}" by default')
            if len(ds_def.language_splits) > 0:
                logger.warning(
                    f"single language configuration specified, but language splits supplied, they will be ignored")
            ds_language_splits = [ds_def.subset]

        if check_if_languages_full(ds_language_splits):
            continue
        ds = load_hf_dataset(ds_def, ctx.dataset.hf_cache).shuffle(seed=ctx.random_seed, buffer_size=ctx.dataset.shuffle_buffer)

        for doc in ds:
            if documents % 10000 == 0:
                logger.info(f"processed {documents} documents and found {samples}/{total_samples} samples")
                observed_languages = sum(1 for count in language_documents_seen.values() if count > 0)
                logger.info(f"there are {observed_languages}/{len(target_languages)} target languages observed so far")
                logger.info(f"raw language counts seen so far: {dict(raw_language_counts)}")
                check_language_status()
            # handle early stopping
            if check_if_languages_full(ds_language_splits):
                break
            if ds_def.language_feature is not None:
                lang = doc[ds_def.language_feature]
            else:
                lang = ds_def.subset
            raw_language_counts[lang] += 1
            if lang not in language_counts:
                documents += 1
                continue
            language_documents_seen[lang] += 1
            if language_counts[lang] >= ctx.dataset.samples:
                language_documents_skipped[lang] += 1
                documents += 1
                continue
            # Break down into individual sentences
            separator_pattern = generate_punctuation_regex(ctx.dataset.language_separators[lang])
            input_sentences = [s for s in re.split(separator_pattern, doc[ds_def.input_feature]) if len(s.strip()) > 0]
            output_sentences = [s for s in re.split(separator_pattern, doc[ds_def.output_feature]) if len(s.strip()) > 0]
            if len(input_sentences) != len(output_sentences):
                language_documents_skipped[lang] += 1
                language_mismatch_skips[lang] += 1
                documents += 1
                continue
            for i, o in zip(input_sentences, output_sentences):
                if len(i) == 0 or len(o) == 0:
                    language_empty_sentence_skips[lang] += 1
                    continue
                language_counts[lang] += 1
                final_dataset[ctx.dataset.input_feature].append(i)
                final_dataset[ctx.dataset.output_feature].append(o)
                final_dataset[ctx.dataset.language_feature].append(lang)
                samples += 1
                if language_counts[lang] >= ctx.dataset.samples:
                    logger.info("language is full, moving on...")
                    break
            documents += 1

    logger.info(f"processed {documents} documents and found {samples}/{total_samples} samples")
    logger.info(f"final raw language counts seen: {dict(raw_language_counts)}")
    missing_languages = {
        lang: ctx.dataset.samples - count
        for lang, count in language_counts.items()
        if count < ctx.dataset.samples
    }
    if missing_languages:
        details = ", ".join(f"{lang}={remaining}" for lang, remaining in sorted(missing_languages.items()))
        raise RuntimeError(
            "dataset source was exhausted before all language quotas were met; "
            f"remaining samples: {details}"
        )

    # Create an in-memory dataset
    output_ds = Dataset.from_dict(final_dataset)
    output_ds.save_to_disk(str(output_path_name))
    persist_updated_config(ctx)
    store_metadata(output_path_name, ctx.dataset)
    store_tokenizer_metadata(output_path_name, ctx.model.tokenizer)
    train_ds, train_ds_ctx, eval_ds, eval_ds_ctx = split_or_load_eval_dataset(ctx, output_ds, force=True)
    return prepare_datasets(ctx, train_ds, train_ds_ctx, eval_ds, eval_ds_ctx)


def preprocess_dataset(
        ctx: TrainConfig, ds_ctx: NamedSplitDatasetFeatureConfig,
        ds: Dataset, tokenizer: PreTrainedTokenizer
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


if __name__ == "__main__":
    config = parse_args("generates the dataset necessary for training")
    logger.info("setting seed")
    set_seed(config.random_seed)
    logger.info("loading/creating dataset")
    create_dataset(config)
    logger.info(f"dataset setup complete, dataset written to: ")
