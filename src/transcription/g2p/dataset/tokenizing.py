import os
import pathlib

from datasets import DatasetDict
import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm


eod_token = r'<|endoftext|>'


def tokenize(dataset: DatasetDict, output_prefix: pathlib.Path, tokenizer: Tokenizer, procs: int = os.cpu_count()):
    # we now want to tokenize the dataset. first define the encoding function (gpt2 bpe)
    def process(example):
        text = example["text"] + eod_token  # add the end of text token, e.g. 50256 for gpt2 bpe
        encoding = tokenizer.encode(text)  # encode_ordinary ignores any special tokens
        # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
        out = {'ids': encoding.ids, 'readable': ','.join(encoding.tokens), 'len': len(encoding.ids)}
        return out

    tokenized = dataset.map(
        process,
        remove_columns=['text'],
        desc="tokenizing the splits",
        num_proc=procs,
    )

    # concatenate all the ids in each dataset into one large file we can use for training
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = output_prefix / f'{split}.bin'
        dtype = np.uint16  # (can do since enc.max_token_value == 50256 is < 2**16)
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            # Batch together samples for faster write
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            # Write into mmap
            arr[idx: idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()