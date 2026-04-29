from collections import Counter
import json
import logging
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple, Union

from datasets import load_dataset, IterableDataset, Dataset, load_from_disk, ClassLabel, Value
from pydantic import ValidationError
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from common import format_language_marker
from config import TrainConfig, CoreDatasetConfig, DatasetFeatureConfig, NamedSplitDatasetFeatureConfig, \
    ConstructedDatasetConfig

logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


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


def split_or_load_eval_dataset(ctx: TrainConfig, train_ds: Dataset) -> Tuple[Dataset, NamedSplitDatasetFeatureConfig, Dataset, NamedSplitDatasetFeatureConfig]:
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

        languages = sorted(set(train_ds[language_col]))

        train_ds = train_ds.cast_column(
            language_col,
            ClassLabel(names=languages),
        )

        split_size = determine_eval_size(ctx, train_ds)
        logger.info(f"using test size of {split_size}")

        output_ds = train_ds.train_test_split(
            seed=ctx.random_seed,
            test_size=split_size,
            stratify_by_column=language_col,
        )

        label_feature = train_ds.features[language_col]
        new_features = output_ds["train"].features.copy()
        new_features[language_col] = Value("string")

        def decode_language(examples):
            examples[language_col] = label_feature.int2str(examples[language_col])
            return examples

        output_ds = output_ds.map(decode_language, batched=True, num_proc=ctx.cpus, features=new_features)

        logger.info(output_ds["train"][0][language_col])
        logger.info(output_ds["test"][0][language_col])  # or "test", depending on split name
        logger.info(output_ds["train"].features[language_col])

        train_ds = output_ds["train"]
        test_ds = output_ds["test"]
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


def load_metadata(output_dir: Path) -> Optional[ConstructedDatasetConfig]:
    if not output_dir.exists():
        return None
    try:
        with open(output_dir / 'custom_metadata.json', 'r') as f:
            metadata = json.load(f)
        return ConstructedDatasetConfig.model_validate_json(metadata)
    except (FileNotFoundError, json.JSONDecodeError, ValidationError) as _:
        return None


def create_dataset(ctx: TrainConfig) -> Tuple[Dataset, NamedSplitDatasetFeatureConfig, Dataset, NamedSplitDatasetFeatureConfig]:
    logger.info("creating dataset")

    output_path_name = ctx.dataset.hf_cache / ctx.dataset.output_dataset_name
    if not ctx.dataset.force_dataset_build:
        logger.info('checking for cached dataset')
        meta = load_metadata(output_path_name)
        if meta is not None:
            if meta == ctx.dataset:
                logger.info('cached dataset metadata matches')
                output_ds = load_from_disk(str(output_path_name))
                return split_or_load_eval_dataset(ctx, output_ds)
        logger.info("dataset cache doesn't exist or isn't usable, creating new dataset")

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
    store_metadata(output_path_name, ctx.dataset)
    return split_or_load_eval_dataset(ctx, output_ds)


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
