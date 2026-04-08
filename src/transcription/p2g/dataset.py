import json
import logging
import re
from pathlib import Path
from collections import Counter

from datasets import load_dataset, IterableDataset, DatasetDict, Dataset, load_from_disk
from tqdm import tqdm

from config import TrainConfig, DatasetConfig


logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def load_streaming_dataset(definition: DatasetConfig, cache_loc: Path) -> IterableDataset:
    if definition.subset is None:
        logger.info(f'loading {definition.name} from HF dataset.')
        dataset = load_dataset(
            definition.name, split=definition.split,
            cache_dir=str(cache_loc),
            streaming=True
        )
    else:
        logger.info(f'loading {definition.name}/{definition.subset} from HF subset.')
        dataset = load_dataset(
            definition.name, definition.subset, split=definition.split,
            cache_dir=str(cache_loc),
            streaming=True
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


def create_dataset(ctx: TrainConfig) -> DatasetDict:
    logger.info("creating dataset")

    output_path_name = ctx.dataset.hf_cache / ctx.dataset.output_dataset_name
    if not ctx.dataset.force_dataset_build:
        logger.info('checking for cached dataset')
        if output_path_name.exists():
            logger.info('cached dataset exists')
            with open(output_path_name / 'custom_metadata.json', 'r') as f:
                metadata = json.load(f)
            if metadata['seed'] == ctx.random_seed and metadata['samples'] == ctx.dataset.samples:
                logger.info('cached dataset random seed and sample count matches')
                return load_from_disk(str(output_path_name))
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
        'language': []
    }
    language_counts = {lang: 0 for lang in target_languages}
    language_documents_seen = {lang: 0 for lang in target_languages}
    language_documents_skipped = {lang: 0 for lang in target_languages}
    language_mismatch_skips = {lang: 0 for lang in target_languages}
    language_empty_sentence_skips = {lang: 0 for lang in target_languages}
    raw_language_counts = Counter()
    for ds_def in ctx.dataset.definitions:
        logger.info(f"processing dataset: {ds_def.name}")
        logger.info(f"processed {documents} documents and found {samples}/{total_samples} samples")
        for lang in target_languages:
            if language_counts[lang] < ctx.dataset.samples:
                logger.info(
                    f"language {lang} is waiting for {ctx.dataset.samples - language_counts[lang]} samples "
                    f"(seen_docs={language_documents_seen[lang]}, skipped_docs={language_documents_skipped[lang]}, "
                    f"mismatch_skips={language_mismatch_skips[lang]}, empty_sentence_skips={language_empty_sentence_skips[lang]})"
                )
        if all([(lang in language_counts and language_counts[lang] >= ctx.dataset.samples) for lang in
                ds_def.language_splits]):
            logger.info(f"all languages in dataset are full, moving on...")
            continue
        ds = load_streaming_dataset(ds_def, ctx.dataset.hf_cache).shuffle(seed=ctx.random_seed, buffer_size=ctx.dataset.shuffle_buffer)
        for doc in ds:
            if documents % 10000 == 0:
                logger.info(f"processed {documents} documents and found {samples}/{total_samples} samples")
                observed_languages = sum(1 for count in language_documents_seen.values() if count > 0)
                logger.info(f"there are {observed_languages}/{len(target_languages)} target languages observed so far")
                logger.info(f"raw language counts seen so far: {dict(raw_language_counts)}")
                for lang in target_languages:
                    if language_counts[lang] < ctx.dataset.samples:
                        logger.info(
                            f"language {lang} is waiting for {ctx.dataset.samples - language_counts[lang]} samples "
                            f"(seen_docs={language_documents_seen[lang]}, skipped_docs={language_documents_skipped[lang]}, "
                            f"mismatch_skips={language_mismatch_skips[lang]}, empty_sentence_skips={language_empty_sentence_skips[lang]})"
                        )
            # handle early stopping
            if all([(lang in language_counts and language_counts[lang] >= ctx.dataset.samples) for lang in
                    ds_def.language_splits]):
                logger.info(f"all languages in dataset are full, moving on...")
                break
            lang = doc[ds_def.language_feature]
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
                # logger.warning(f"sentences for {lang} didn't match after splitting "
                #                f"({len(input_sentences)} vs {len(output_sentences)})")
                language_documents_skipped[lang] += 1
                language_mismatch_skips[lang] += 1
                documents += 1
                continue
            for i, o in zip(input_sentences, output_sentences):
                if len(i) == 0 or len(o) == 0:
                    # logger.warning(f"skipping empty sentences: ({len(i)} vs {len(o)})")
                    language_empty_sentence_skips[lang] += 1
                    continue
                language_counts[lang] += 1
                final_dataset[ctx.dataset.input_feature].append(i)
                final_dataset[ctx.dataset.output_feature].append(o)
                final_dataset['language'].append(lang)
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
    output_ds = Dataset.from_dict(final_dataset).train_test_split(seed=ctx.random_seed, train_size=ctx.dataset.train_split_size)
    output_ds.save_to_disk(str(output_path_name))
    with open(output_path_name / 'custom_metadata.json', 'w+') as f:
        json.dump({'seed': ctx.random_seed, 'samples': ctx.dataset.samples}, f)
    return output_ds
