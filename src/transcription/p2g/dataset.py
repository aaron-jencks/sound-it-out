from collections import Counter
from datetime import datetime
import logging
from pathlib import Path
import time
from typing import Optional, Tuple

from datasets import Dataset, DatasetDict
from transformers import set_seed

from transcription.p2g.config import ConstructionInputDatasetConfig, PreprocessingConfig
from transcription.p2g.dataset_loading import load_hf_dataset
from transcription.p2g.setup import parse_args
from transcription.p2g.transform_batching import LanguageBatchCollector
from transcription.p2g.transform_pool import (
    CompletedTransformBatch,
    QueuedTransformBatch,
    TransformPoolSupervisor,
)


logger = logging.getLogger(__file__)


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


def build_transform_pool(ctx: PreprocessingConfig) -> TransformPoolSupervisor:
    if ctx.transform.type == "phonemize":
        return TransformPoolSupervisor(
            transform_type="phonemize",
            worker_count=ctx.cpus,
            timeout_seconds=30,
            espeak_path=ctx.transform.espeak_path,
        )

    if ctx.transform.type == "romanize":
        if ctx.transform.romanization is None:
            raise ValueError("romanization config is required when transform.type is 'romanize'")
        return TransformPoolSupervisor(
            transform_type="romanize",
            worker_count=ctx.cpus,
            timeout_seconds=30,
            uroman_path=ctx.transform.romanization.uroman_path,
            perl_path=ctx.transform.romanization.perl_path,
        )

    raise ValueError(f"unsupported transform type: {ctx.transform.type}")


def split_dataset(ctx: PreprocessingConfig, ds: Dataset) -> DatasetDict:
    split_size = determine_eval_size(ctx, ds)
    logger.info(f"using test size of {split_size}")
    return ds.train_test_split(
        seed=ctx.random_seed,
        test_size=split_size,
    )


def format_duration(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def create_dataset(ctx: PreprocessingConfig) -> Path:
    logger.info("creating dataset")
    if ctx.output_dataset.language_feature is None:
        raise ValueError("output_dataset.language_feature cannot be None for construction")
    output_dt, output_path_name = get_dataset_path(ctx)
    logger.info(f"building dataset artifact for timestamp {output_dt.isoformat()}")
    logger.info(f"dataset will be written to {output_path_name}")

    transform_pool = build_transform_pool(ctx)
    batch_collector = LanguageBatchCollector(ctx.transform_batch_size)
    languages = target_languages(ctx)
    total_samples = len(languages) * ctx.samples

    final_dataset = {
        ctx.output_dataset.input_feature: [],
        ctx.output_dataset.output_feature: [],
        ctx.output_dataset.language_feature: [],
    }
    language_counts = {language: 0 for language in languages}
    language_inflight_counts = {language: 0 for language in languages}
    language_documents_seen = {language: 0 for language in languages}
    language_documents_skipped = {language: 0 for language in languages}
    language_short_documents = {language: 0 for language in languages}
    raw_language_counts = Counter()
    transform_failure_counts = Counter()

    documents = 0
    processed_samples = 0
    start_time = time.monotonic()

    def log_progress() -> None:
        elapsed_seconds = time.monotonic() - start_time
        base = (
            f"processed {documents} documents and found {processed_samples}/{total_samples} samples, "
            f"elapsed time {format_duration(elapsed_seconds)} "
        )
        if processed_samples == 0:
            logger.info(f"{base}estimated remaining time unknown")
            return

        remaining_samples = total_samples - processed_samples
        estimated_remaining_seconds = elapsed_seconds * remaining_samples / processed_samples
        logger.info(
            f"{base}estimated remaining time {format_duration(estimated_remaining_seconds)}"
        )

    def check_language_status():
        for language in languages:
            if language_counts[language] < ctx.samples:
                logger.info(
                    f"language {language} is waiting for {ctx.samples - language_counts[language]} samples "
                    f"(buff_count={batch_collector.count(language)}, "
                    f"inflight_count={language_inflight_counts[language]}, "
                    f"seen_docs={language_documents_seen[language]}, "
                    f"skipped_docs={language_documents_skipped[language]}, "
                    f"short_docs={language_short_documents[language]})"
                )

    def dataset_languages_full(dataset_languages: list[str]) -> bool:
        return all(language_counts.get(language, 0) >= ctx.samples for language in dataset_languages)

    def effective_language_count(language: str) -> int:
        return language_counts[language] + language_inflight_counts[language]

    def merge_completed_batches(completed_batches: list[CompletedTransformBatch]) -> None:
        nonlocal processed_samples

        for batch in completed_batches:
            language_inflight_counts[batch.language] -= len(batch.texts)
            transform_failure_counts.update(batch.event_counts)

            for text, transformed_text in zip(batch.texts, batch.outputs):
                if transformed_text is None:
                    language_documents_skipped[batch.language] += 1
                    continue

                final_dataset[ctx.output_dataset.input_feature].append(transformed_text)
                final_dataset[ctx.output_dataset.output_feature].append(text)
                final_dataset[ctx.output_dataset.language_feature].append(batch.language)
                language_counts[batch.language] += 1
                processed_samples += 1

    def flush_language(language: str, texts: Optional[list[str]] = None) -> None:
        if texts is None:
            texts = batch_collector.pop(language)
        if not texts:
            return

        language_inflight_counts[language] += len(texts)
        completed_batches = transform_pool.pump(QueuedTransformBatch(language=language, texts=texts))
        merge_completed_batches(completed_batches)

    def flush_languages(languages_to_flush: list[str]) -> None:
        for language, texts in batch_collector.pop_all(languages_to_flush):
            flush_language(language, texts)

    def drain_completed_batches() -> None:
        merge_completed_batches(transform_pool.pump())

    def drain_all_inflight_batches() -> None:
        while transform_pool.has_inflight_batches():
            merge_completed_batches(transform_pool.wait_for_completion())

    try:
        for definition in ctx.input_datasets:
            dataset_target_languages = [resolve_language(definition, language) for language in definition.languages]
            logger.info(f"processing dataset: {definition.name}")
            log_progress()
            check_language_status()

            if dataset_languages_full(dataset_target_languages):
                logger.info("all languages in dataset are full, moving on...")
                continue

            ds = load_hf_dataset(
                definition,
                streaming=definition.streaming,
                cache_loc=ctx.hf_cache,
            )
            if definition.streaming:
                ds = ds.shuffle(
                    seed=ctx.random_seed,
                    buffer_size=ctx.shuffle_buffer,
                )
            else:
                ds = ds.shuffle(seed=ctx.random_seed)

            for document in ds:
                drain_completed_batches()
                if documents % (ctx.transform_batch_size * len(languages)) == 0:
                    log_progress()
                    observed_languages = sum(1 for count in language_documents_seen.values() if count > 0)
                    logger.info(f"there are {observed_languages}/{len(languages)} target languages observed so far")
                    logger.info(f"raw language counts seen so far: {dict(raw_language_counts)}")
                    if transform_failure_counts:
                        logger.info(f"transform failures so far: {dict(transform_failure_counts)}")
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
                if effective_language_count(resolved_language) >= ctx.samples:
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

                batch_collector.add(resolved_language, chunk)
                remaining_quota = ctx.samples - effective_language_count(resolved_language)
                if batch_collector.should_flush(resolved_language, remaining_quota):
                    flush_language(resolved_language)
                documents += 1

            flush_languages(dataset_target_languages)
        flush_languages(languages)
        drain_all_inflight_batches()
    finally:
        transform_pool.close()

    log_progress()
    logger.info(f"final raw language counts seen: {dict(raw_language_counts)}")
    if transform_failure_counts:
        logger.info(f"final transform failure counts: {dict(transform_failure_counts)}")
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
    config = parse_args(
        "constructs a dataset",
        default_config=Path("transcription/p2g/config/default_pre.json"),
        schema=PreprocessingConfig,
    )
    logger.info("setting seed")
    set_seed(config.random_seed)
    logger.info("loading/creating dataset")
    create_dataset(config)


if __name__ == "__main__":
    main()
