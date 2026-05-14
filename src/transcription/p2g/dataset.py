from collections import Counter
from datetime import datetime
import logging
from pathlib import Path
import sys
from typing import Callable, Tuple

from datasets import Dataset, DatasetDict
from transformers import set_seed

from config import ConstructionInputDatasetConfig, PreprocessingConfig
from dataset_loading import load_hf_dataset
from setup import parse_args


logger = logging.getLogger(__file__)
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
from g2p import phonemize_romanize


TransformFn = Callable[[str, str], str]


def get_dataset_path(ctx: PreprocessingConfig) -> Tuple[datetime, Path]:
    dt = datetime.now()
    dir_name = f"{dt.isoformat()}-{ctx.output_dataset.name}"
    output_path_name = ctx.hf_cache / dir_name
    return dt, output_path_name


def determine_eval_size(ctx: PreprocessingConfig, ds: Dataset) -> int | float:
    dslen = len(ds)
    if dslen * ctx.splits.eval_ratio > ctx.splits.max_eval_size:
        return ctx.splits.max_eval_size
    return ctx.splits.eval_ratio


def resolve_language(definition: ConstructionInputDatasetConfig, source_language: str) -> str:
    if definition.language_map is None:
        return source_language
    return definition.language_map.get(source_language, source_language)


def target_languages(ctx: PreprocessingConfig) -> list[str]:
    resolved = set()
    for definition in ctx.input_datasets:
        for language in definition.languages:
            resolved.add(resolve_language(definition, language))
    return sorted(resolved)


def get_source_language(definition: ConstructionInputDatasetConfig, document: dict) -> str:
    if definition.language_feature is not None:
        return document[definition.language_feature]
    if definition.subset is not None:
        return definition.subset
    raise ValueError("input dataset must define language_feature or subset")


def extract_word_chunk(text: str, words_per_sample: int) -> Tuple[str, bool]:
    words = [word for word in text.split(" ") if len(word) > 0]
    if len(words) == 0:
        return "", False
    short_document = len(words) < words_per_sample
    return " ".join(words[:words_per_sample]), short_document


def build_transform(ctx: PreprocessingConfig) -> TransformFn:
    if ctx.transform.espeak_path is not None:
        phonemize_romanize.CustomEspeakBackend.set_library(str(ctx.transform.espeak_path))
        phonemize_romanize.BACKENDS.clear()

    if ctx.transform.type == "phonemize":
        def transform(text: str, language: str) -> str:
            return phonemize_romanize.strip_ipa(
                phonemize_romanize.phonemize_batch([text], language)[0]
            )

        return transform

    if ctx.transform.type == "romanize":
        if ctx.transform.romanization is None:
            raise ValueError("romanization config is required when transform.type is 'romanize'")

        def transform(text: str, language: str) -> str:
            del language
            return phonemize_romanize.uromanize_batch(
                [text],
                str(ctx.transform.romanization.uroman_path),
                ctx.transform.romanization.perl_path,
            )[0]

        return transform

    raise ValueError(f"unsupported transform type: {ctx.transform.type}")


def split_dataset(ctx: PreprocessingConfig, ds: Dataset) -> DatasetDict:
    split_size = determine_eval_size(ctx, ds)
    logger.info(f"using test size of {split_size}")
    return ds.train_test_split(
        seed=ctx.random_seed,
        test_size=split_size,
    )


def create_dataset(ctx: PreprocessingConfig) -> Path:
    logger.info("creating dataset")
    if ctx.output_dataset.language_feature is None:
        raise ValueError("output_dataset.language_feature cannot be None for construction")
    output_dt, output_path_name = get_dataset_path(ctx)
    logger.info(f"building dataset artifact for timestamp {output_dt.isoformat()}")
    logger.info(f"dataset will be written to {output_path_name}")

    transform = build_transform(ctx)
    languages = target_languages(ctx)
    total_samples = len(languages) * ctx.samples

    final_dataset = {
        ctx.output_dataset.input_feature: [],
        ctx.output_dataset.output_feature: [],
        ctx.output_dataset.language_feature: [],
    }
    language_counts = {language: 0 for language in languages}
    language_documents_seen = {language: 0 for language in languages}
    language_documents_skipped = {language: 0 for language in languages}
    language_short_documents = {language: 0 for language in languages}
    raw_language_counts = Counter()

    documents = 0
    samples = 0

    def check_language_status():
        for language in languages:
            if language_counts[language] < ctx.samples:
                logger.info(
                    f"language {language} is waiting for {ctx.samples - language_counts[language]} samples "
                    f"(seen_docs={language_documents_seen[language]}, "
                    f"skipped_docs={language_documents_skipped[language]}, "
                    f"short_docs={language_short_documents[language]})"
                )

    def dataset_languages_full(dataset_languages: list[str]) -> bool:
        return all(language_counts.get(language, 0) >= ctx.samples for language in dataset_languages)

    for definition in ctx.input_datasets:
        dataset_target_languages = [resolve_language(definition, language) for language in definition.languages]
        logger.info(f"processing dataset: {definition.name}")
        logger.info(f"processed {documents} documents and found {samples}/{total_samples} samples")
        check_language_status()

        if dataset_languages_full(dataset_target_languages):
            logger.info("all languages in dataset are full, moving on...")
            continue

        ds = load_hf_dataset(definition, streaming=True, cache_loc=ctx.hf_cache).shuffle(
            seed=ctx.random_seed,
            buffer_size=ctx.shuffle_buffer,
        )

        for document in ds:
            if documents % 10000 == 0:
                logger.info(f"processed {documents} documents and found {samples}/{total_samples} samples")
                observed_languages = sum(1 for count in language_documents_seen.values() if count > 0)
                logger.info(f"there are {observed_languages}/{len(languages)} target languages observed so far")
                logger.info(f"raw language counts seen so far: {dict(raw_language_counts)}")
                check_language_status()

            if dataset_languages_full(dataset_target_languages):
                logger.info("all languages in dataset are full, moving on...")
                break

            source_language = get_source_language(definition, document)
            raw_language_counts[source_language] += 1
            resolved_language = resolve_language(definition, source_language)

            if resolved_language not in language_counts:
                documents += 1
                continue

            language_documents_seen[resolved_language] += 1
            if language_counts[resolved_language] >= ctx.samples:
                language_documents_skipped[resolved_language] += 1
                documents += 1
                continue

            source_text = document[definition.output_feature]
            if source_text is None:
                language_documents_skipped[resolved_language] += 1
                documents += 1
                continue

            chunk, short_document = extract_word_chunk(str(source_text), ctx.words_per_sample)
            if len(chunk) == 0:
                language_documents_skipped[resolved_language] += 1
                documents += 1
                continue

            if short_document:
                language_short_documents[resolved_language] += 1

            try:
                transformed_chunk = transform(chunk, resolved_language)
            except Exception as exc:
                raise RuntimeError(
                    f"failed to transform document {documents} from dataset {definition.name} "
                    f"for language {resolved_language}: {exc}"
                ) from exc

            final_dataset[ctx.output_dataset.input_feature].append(transformed_chunk)
            final_dataset[ctx.output_dataset.output_feature].append(chunk)
            final_dataset[ctx.output_dataset.language_feature].append(resolved_language)
            language_counts[resolved_language] += 1
            samples += 1
            documents += 1

    logger.info(f"processed {documents} documents and found {samples}/{total_samples} samples")
    logger.info(f"final raw language counts seen: {dict(raw_language_counts)}")
    missing_languages = {
        language: ctx.samples - count
        for language, count in language_counts.items()
        if count < ctx.samples
    }
    if missing_languages:
        details = ", ".join(
            f"{language}={remaining}" for language, remaining in sorted(missing_languages.items())
        )
        raise RuntimeError(
            "dataset source was exhausted before all language quotas were met; "
            f"remaining samples: {details}"
        )

    output_ds = Dataset.from_dict(final_dataset)
    split_ds = split_dataset(ctx, output_ds)
    split_ds.save_to_disk(str(output_path_name))
    logger.info(f"dataset setup complete, dataset written to: {output_path_name}")
    print(output_path_name)
    return output_path_name


def main():
    config = parse_args("constructs a dataset", schema=PreprocessingConfig)
    logger.info("setting seed")
    set_seed(config.random_seed)
    logger.info("loading/creating dataset")
    create_dataset(config)


if __name__ == "__main__":
    main()
