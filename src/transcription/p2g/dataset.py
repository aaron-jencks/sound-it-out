import json
import logging
from pathlib import Path

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


def create_dataset(ctx: TrainConfig) -> DatasetDict:
    logger.info("creating dataset")

    output_path_name = ctx.dataset.hf_cache / ctx.dataset.output_dataset_name
    if not ctx.dataset.force_dataset_build:
        logger.info('checking for cached dataset')
        if output_path_name.exists():
            logger.info('cached dataset exists')
            with open(output_path_name / 'custom_metadata.json', 'r') as f:
                metadata = json.load(f)
            if metadata['seed'] == ctx.random_seed:
                logger.info('cached dataset random seed matches')
                return load_from_disk(ctx.dataset.output_dataset_name)
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

    pbar = tqdm(total=total_samples, desc="collecting samples from languages")

    # process the datasets
    final_dataset = {
        ctx.dataset.input_feature: [],
        ctx.dataset.output_feature: [],
        'language': []
    }
    language_counts = {}
    for ds_def in ctx.dataset.definitions:
        logger.info(f"processing dataset: {ds_def.name}")
        if all([(lang in language_counts and language_counts[lang] >= ctx.dataset.samples) for lang in
                ds_def.language_splits]):
            logger.info(f"all languages in dataset are full, moving on...")
            continue
        ds = load_streaming_dataset(ds_def, ctx.dataset.hf_cache).shuffle(seed=ctx.random_seed, buffer_size=ctx.dataset.shuffle_buffer)
        for doc in ds:
            # handle early stopping
            if all([(lang in language_counts and language_counts[lang] >= ctx.dataset.samples) for lang in
                    ds_def.language_splits]):
                logger.info(f"all languages in dataset are full, moving on...")
                break
            lang = doc[ds_def.language_feature]
            if lang not in language_counts:
                language_counts[lang] = 0
            if language_counts[lang] >= ctx.dataset.samples:
                continue
            language_counts[lang] += 1
            final_dataset[ctx.dataset.input_feature].append(doc[ds_def.input_feature])
            final_dataset[ctx.dataset.output_feature].append(doc[ds_def.output_feature])
            final_dataset['language'].append(lang)
            pbar.update(1)

    pbar.close()

    # Create an in-memory dataset
    output_ds = Dataset.from_dict(final_dataset).train_test_split(seed=ctx.random_seed, train_size=ctx.dataset.train_split_size)
