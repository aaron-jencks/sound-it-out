import logging
import os
from typing import Callable, Dict, List, Optional, Tuple
import unicodedata as ud

from datasets import DatasetDict, Dataset
from phonemizer.backend import EspeakBackend
from phonemizer.custom_espeak import CustomEspeakBackend

from romanization import UromanWrapper

logger = logging.getLogger(__name__)


# We do it this way because finetuning has multiple columns to transcribe
ROW_EXTRACTOR = Callable[[Dict[str, List[str]]], List[List[str]]]
ROW_PACKER = Callable[[List[List[str]]], Dict[str, List[str]]]
ROW_TRANSCRIBER = Callable[[str], str]


def transcribe_dataset(
        dataset: DatasetDict,
        extractor: ROW_EXTRACTOR, packer: ROW_PACKER,
        phone: EspeakBackend,
        cpus: int = os.cpu_count()
) -> DatasetDict:
    logger.info(f'transcribing dataset...')

    def process(examples, idxs):
        # extract the list of examples for each column
        columns = extractor(examples)
        result_columns = []

        # transcribe each column
        for column in columns:
            result_column = []
            for example, idx in zip(column, idxs):
                try:
                    phonemes = phone.phonemize([example])[0]
                    result_column.append(phonemes)
                except Exception as e:
                    raise Exception(f'unable to process row {idx}: "{example}": {e}')
            result_columns.append(result_column)

        # repack each column into a dict
        result = packer(result_columns)

        return result

    phonemized = dataset.map(
        process,
        with_indices=True,
        desc="phonemizing the splits",
        batched=True,
        num_proc=cpus,
    )

    return phonemized


def customized_transcribe_dataset(
        dataset: DatasetDict,
        extractor: ROW_EXTRACTOR, packer: ROW_PACKER,
        transcribers: List[ROW_TRANSCRIBER]
) -> DatasetDict:
    logger.info(f'transcribing dataset...')

    def process(examples, idxs):
        # if idx < 1540000:
        #     return {'phonemes': ''}

        # extract the list of examples for each column
        columns = extractor(examples)
        result_columns = []

        # transcribe each column
        for column, idx in zip(columns, idxs):
            result_column = []
            for example in column:
                try:
                    for transcriber in transcribers:
                        phonemes = transcriber(example)
                        result_column.append(phonemes)
                except Exception as e:
                    raise Exception(f'unable to process row {idx}: "{example}"')
            result_columns.append(result_column)

        # repack each column into a dict
        result = packer(result_columns)

        return result

    phonemized = dataset.map(
        process,
        with_indices=True,
        desc="phonemizing the splits",
        batched=True,
        num_proc=1,
    )

    return phonemized


def _strip_ipa(text: str) -> str:
    if text is None:
        return text
    text = text.replace("ˈ", "").replace("ˌ", "").replace("ː", "").replace("ˑ", "")
    return "".join(ch for ch in text if ud.category(ch) != "Mn")


# Now we have multiple columns we're returning at once
ROW_PACKER_ROMAN = Callable[[List[Tuple[List[str], List[str], List[str]]]], Dict[str, List[str]]]


def transcribe_dataset_w_romanization(
        dataset: DatasetDict,
        extractor: ROW_EXTRACTOR, packer: ROW_PACKER_ROMAN,
        phone: EspeakBackend,
        rom: UromanWrapper,
        cpus: int = os.cpu_count()
) -> DatasetDict:
    logger.info(f'transcribing dataset...')

    def process(examples, idxs):
        # extract the list of examples for each column
        columns = extractor(examples)
        result_columns = []

        # transcribe each column
        for column in columns:
            result_column_ipa = []
            result_column_ipa_stripped = []
            result_column_roman = []
            for example, idx in zip(column, idxs):
                try:
                    phonemes = phone.phonemize([example])[0]
                    romes = rom.romanize_string(example)
                    result_column_ipa.append(phonemes)
                    result_column_ipa_stripped.append(_strip_ipa(phonemes))
                    result_column_roman.append(romes)
                except Exception as e:
                    raise Exception(f'unable to process row {idx}: "{example}": {e}')
            result_columns.append((result_column_ipa, result_column_ipa_stripped, result_column_roman))

        # repack each column into a dict
        result = packer(result_columns)

        return result

    phonemized = dataset.map(
        process,
        with_indices=True,
        desc="phonemizing the splits",
        batched=True,
        num_proc=cpus,
    )

    return phonemized


def create_transcribed_dataset_name(full_dataset_name: str, prefix: str = '') -> str:
    name = full_dataset_name
    if '/' in full_dataset_name:
        name = full_dataset_name.split('/')[-1]

    fname = f'{name}-ipa'
    if len(prefix) > 0:
        fname = f'{name}-{prefix}-ipa'
    return fname


def create_default_transcriber(
        espeak: str = '',
        language: str = 'en-us', punctuation: str = r"[^'a-zA-Z0-9\s]",
        marker_token_vocab: str = ',`~', rtl_preserve: bool = False,
        custom_regex: Optional[List[str]] = None,
) -> CustomEspeakBackend:
    slogger = logging.getLogger()
    slogger.handlers.clear()
    slogger.addHandler(logging.NullHandler())

    logger.info('setting up espeak...')

    regex = [
        r'[^\s]*\d+[^\s]*'
    ]
    if custom_regex is not None:
        regex += custom_regex

    if len(espeak) > 0:
        logger.info('setting espeak library to {}'.format(espeak))
        CustomEspeakBackend.set_library(espeak)
    phone = CustomEspeakBackend(
        language,
        punctuation,  # handle arabic numerals
        regex,
        preserve_punctuation=True, with_stress=True, logger=slogger,
        token_vocab=marker_token_vocab,
        rtl_preserve=rtl_preserve,
    )
    return phone


def create_default_transcriber_no_punctuation(
        espeak: str = '',
        language: str = 'en-us'
) -> EspeakBackend:
    slogger = logging.getLogger()
    slogger.handlers.clear()
    slogger.addHandler(logging.NullHandler())

    logger.info('setting up espeak...')

    if len(espeak) > 0:
        logger.info('setting espeak library to {}'.format(espeak))
        EspeakBackend.set_library(espeak)
    phone = EspeakBackend(
        language, preserve_punctuation=True, with_stress=True, logger=slogger
    )
    return phone


def normalize_document(document: str, normalization_style: str = 'NFD') -> str:
    return ud.normalize(normalization_style, document)


def normalize_dataset(dataset: Dataset, feature: str, normalization_style: str = 'NFD', cpus: int = os.cpu_count()) -> Dataset:
    def normalize(examples, idxs):
        column = examples[feature]
        result_columns = {
            f'{feature}_normalized': [],
        }

        for example, idx in zip(column, idxs):
            try:
                normalized = normalize_document(example, normalization_style)
                result_columns[f'{feature}_normalized'].append(normalized)
            except Exception as e:
                raise Exception(f'unable to process row {idx}: "{example}"')

        return result_columns

    return dataset.map(
        normalize,
        with_indices=True,
        desc=f"normalizing dataset feature {feature}",
        batched=True,
        num_proc=cpus,
    )