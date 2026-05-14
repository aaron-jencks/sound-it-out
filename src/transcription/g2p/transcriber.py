import argparse
import logging
import pathlib
import os
from typing import Dict, List

from datasets import Value, Features

from dataset.loading import load_dataset
from dataset.transcription import transcribe_dataset, create_transcribed_dataset_name, create_default_transcriber

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='transcribes the openwebtext dataset into ipa')
    ap.add_argument('-d', '--dataset', type=str, default='openwebtext', help='dataset to use')
    ap.add_argument('-s', '--subset', type=str, default='', help='the subdataset to use')
    ap.add_argument('-e', '--espeak', type=str, default='', help='the location of the espeak library')
    ap.add_argument('--cache', type=pathlib.Path, default=pathlib.Path('./cache'),
                    help='the location of the cache folder for huggingface')
    ap.add_argument('--cpus', type=int, default=os.cpu_count(), help='number of cpus to use')
    ap.add_argument('--username', type=str, default='iggy12345', help='the huggingface username to use for uploading')
    ap.add_argument('--target-dataset', type=str, default='openwebtext-ipa', help='the target dataset to upload to')
    args = ap.parse_args()

    logger.info(f'loading dataset "{args.dataset}"...')

    dataset = load_dataset(args.dataset, args.subset, cache_loc=args.cache, procs=args.cpus)

    phone = create_default_transcriber(args.espeak)

    def simple_text_extractor(row: Dict[str, List[str]]) -> List[List[str]]:
        return [row['text']]

    def simple_phoneme_packer(phonemes: List[List[str]]) -> Dict[str, List[str]]:
        return {
            'phonemes': phonemes[0]
        }

    phonemized = transcribe_dataset(
        dataset,
        simple_text_extractor, simple_phoneme_packer,
        phone,
        args.cpus
    )

    # Cast new field properly
    new_features = phonemized['train'].features.copy()
    new_features['phonemes'] = Value("string")
    phonemized = phonemized.cast(Features(new_features))

    logger.info(f'saving dataset "{args.dataset}"...')

    # fname = create_transcribed_dataset_name(args.dataset, args.prefix)
    # phonemized.save_to_disk(args.cache / fname)

    phonemized.push_to_hub(f'{args.username}/{args.target_dataset}')
