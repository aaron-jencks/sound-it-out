import logging
import os
import pathlib
from typing import List, Optional, Union

from datasets import load_dataset as ld
from datasets import load_from_disk, DatasetDict, Dataset

logger = logging.getLogger(__name__)


def load_csv_dataset(fname: Union[pathlib.Path, List[pathlib.Path]], delimiter: str = ',', columns: Optional[List[str]] = None) -> Dataset:
    fnames = str(fname) if isinstance(fname, pathlib.Path) else list(map(str, fname))
    return ld(
        'csv',
        data_files=fnames,
        delimiter=delimiter,
    ) if columns is None else ld(
        'csv',
        data_files=fnames,
        delimiter=delimiter,
        column_names=columns,
    )


def load_hf_dataset(
        name: str = 'openwebtext', subset: str = '',
        cache_loc: pathlib.Path = pathlib.Path('./cache/huggingface'),
        procs: int = os.cpu_count(), split: bool = True,
) -> DatasetDict:
    if len(subset) == 0:
        logger.info(f'Loading {name} from HF dataset.')
        dataset = ld(name, num_proc=procs, trust_remote_code=True, cache_dir=str(cache_loc))
    else:
        logger.info(f'Loading {name}/{subset} from HF subset.')
        dataset = ld(name, subset, num_proc=procs, trust_remote_code=True, cache_dir=str(cache_loc))
    if split:
        logger.info(f'Splitting {name}/{subset} into train and val sets.')
        # owt by default only contains the 'train' split, so create a test split
        split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
        split_dataset['val'] = split_dataset.pop('test')  # rename the test split to val
        return split_dataset
    return dataset


def load_dataset(
        name: str = 'openwebtext', subset: str = '',
        cache_loc: pathlib.Path = pathlib.Path('./cache/huggingface'),
        procs: int = os.cpu_count(),
        from_disk: bool = False,
        split: bool = True,
) -> DatasetDict:
    if not from_disk:
        return load_hf_dataset(name, subset, cache_loc, procs, split)
    return load_from_disk(str(cache_loc / name))


def load_streaming_dataset(
        name: str = 'openwebtext', subset: str = '', split: str = 'train',
        cache_loc: pathlib.Path = pathlib.Path('./cache/huggingface')
) -> DatasetDict:
    if len(subset) == 0:
        logger.info(f'Loading {name} from HF dataset.')
        dataset = ld(
            name, split=split,
            trust_remote_code=True, cache_dir=str(cache_loc),
            streaming=True
        )
    else:
        logger.info(f'Loading {name}/{subset} from HF subset.')
        dataset = ld(
            name, subset, split=split,
            trust_remote_code=True, cache_dir=str(cache_loc),
            streaming=True
        )
    return dataset
