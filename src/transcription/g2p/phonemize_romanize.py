#!/usr/bin/env python3
import argparse
import logging
import os
import subprocess
import unicodedata

from datasets import load_dataset

from phonemizer.backend import EspeakBackend
from .custom_espeak import CustomEspeakBackend

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)
logging.getLogger("phonemizer").setLevel(logging.ERROR)

MASSIVE_LANGS = {
    "en-US": "en-us",
    "es-ES": "es",
    "ru-RU": "ru",
    "pl-PL": "pl",
    "ta-IN": "ta",
    "ml-IN": "ml",
    "hi-IN": "hi",
    "ur-PK": "ur",
}

INDICXNLI_LANGS = {
    "ml": "ml",
    "ta": "ta",
}

EXTRA_PAIR_DATASETS = [
    {
        "source": "iggy12345/xnli-en-ipa",
        "repo_id": "mugezhang/xnli-en-ipa_ipa_romanized",
        "field_a": "premise",
        "field_b": "hypothesis",
        "out_a": "premise",
        "out_b": "hypothesis",
        "espeak_lang": "en-us",
    },
    {
        "source": "iggy12345/xnli-es-ipa",
        "repo_id": "mugezhang/xnli-es-ipa_ipa_romanized",
        "field_a": "premise",
        "field_b": "hypothesis",
        "out_a": "premise",
        "out_b": "hypothesis",
        "espeak_lang": "es",
    },
    {
        "source": "krishnAbadikelA/hindi-xnli-ipa",
        "repo_id": "mugezhang/hindi-xnli-ipa_ipa_romanized",
        "field_a": "premise",
        "field_b": "hypothesis",
        "out_a": "premise",
        "out_b": "hypothesis",
        "espeak_lang": "hi",
    },
    {
        "source": "krishnAbadikelA/urdu-xnli-ipa",
        "repo_id": "mugezhang/urdu-xnli-ipa_ipa_romanized",
        "field_a": "premise",
        "field_b": "hypothesis",
        "out_a": "premise",
        "out_b": "hypothesis",
        "espeak_lang": "ur",
    },
    {
        "source": "iggy12345/russian-xnli-ipa-rosetta",
        "repo_id": "mugezhang/russian-xnli-ipa-rosetta_ipa_romanized",
        "field_a": "premise",
        "field_b": "hypothesis",
        "out_a": "premise",
        "out_b": "hypothesis",
        "espeak_lang": "ru",
    },
    {
        "source": "iggy12345/cdsc-e-ipa",
        "repo_id": "mugezhang/cdsc-e-ipa_ipa_romanized",
        "field_a": "sentence_A",
        "field_b": "sentence_B",
        "out_a": "sentence_A",
        "out_b": "sentence_B",
        "espeak_lang": "pl",
    },
]

BACKENDS = {}


def create_phonemizer(language: str) -> EspeakBackend:
    return CustomEspeakBackend(
        language,
        preserve_punctuation=True,
        with_stress=True,
        logger=logging.getLogger("phonemizer"),
    )


def get_phonemizer(language: str) -> EspeakBackend:
    backend = BACKENDS.get(language)
    if backend is None:
        backend = create_phonemizer(language)
        BACKENDS[language] = backend
    return backend


def strip_ipa(text: str) -> str:
    if text is None:
        return text
    text = text.replace("ˈ", "").replace("ˌ", "").replace("ː", "").replace("ˑ", "")
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")


def phonemize_batch(texts: list[str], language: str) -> list[str]:
    backend = get_phonemizer(language)
    return backend.phonemize(texts)


def uromanize_batch(texts: list[str], uroman_path: str, perl_path: str) -> list[str]:
    escaped = [(t or "").replace("\n", "\\n") for t in texts]
    stdin = "\n".join(escaped) + "\n"
    result = subprocess.run(
        [perl_path, uroman_path],
        input=stdin,
        text=True,
        capture_output=True,
        check=True,
    )
    out_lines = result.stdout.splitlines()
    if len(out_lines) != len(escaped):
        raise RuntimeError(
            f"uroman output line mismatch: expected {len(escaped)}, got {len(out_lines)}"
        )
    return [line.replace("\\n", "\n") for line in out_lines]


def process_massive(repo_id: str, cache_dir: str, uroman_path: str, perl_path: str, num_proc: int, batch_size: int):
    for idx, (config_name, espeak_lang) in enumerate(MASSIVE_LANGS.items()):
        logger.info("Loading MASSIVE config %s", config_name)
        ds = load_dataset(
            "AmazonScience/massive",
            name=config_name,
            cache_dir=cache_dir,
            download_mode="reuse_dataset_if_exists",
        )

        def map_fn(batch, espeak_lang=espeak_lang):
            utts = [u or "" for u in batch["utt"]]
            ipa = phonemize_batch(utts, espeak_lang)
            ipa_stripped = [strip_ipa(x) for x in ipa]
            romanized = uromanize_batch(utts, uroman_path, perl_path)
            return {
                "ipa_stripped": ipa_stripped,
                "romanized": romanized,
            }

        logger.info("Phonemizing + romanizing MASSIVE %s", config_name)
        ds = ds.map(
            map_fn,
            batched=True,
            batch_size=batch_size,
            num_proc=num_proc,
            desc=f"massive-{config_name}",
        )
        logger.info("Pushing MASSIVE %s to %s", config_name, repo_id)
        ds.push_to_hub(
            repo_id,
            config_name=config_name,
            set_default=(idx == 0),
            token=os.environ.get("HF_TOKEN"),
            max_shard_size="1GB",
        )


def process_indicxnli(repo_id: str, cache_dir: str, uroman_path: str, perl_path: str, num_proc: int, batch_size: int):
    for idx, (config_name, espeak_lang) in enumerate(INDICXNLI_LANGS.items()):
        logger.info("Loading IndicXNLI config %s", config_name)
        ds = load_dataset(
            "Divyanshu/indicxnli",
            name=config_name,
            cache_dir=cache_dir,
            download_mode="reuse_dataset_if_exists",
        )

        def map_fn(batch, espeak_lang=espeak_lang):
            premise = [p or "" for p in batch["premise"]]
            hypothesis = [h or "" for h in batch["hypothesis"]]

            premise_ipa = phonemize_batch(premise, espeak_lang)
            hypothesis_ipa = phonemize_batch(hypothesis, espeak_lang)

            premise_ipa_stripped = [strip_ipa(x) for x in premise_ipa]
            hypothesis_ipa_stripped = [strip_ipa(x) for x in hypothesis_ipa]

            premise_romanized = uromanize_batch(premise, uroman_path, perl_path)
            hypothesis_romanized = uromanize_batch(hypothesis, uroman_path, perl_path)

            return {
                "premise_ipa_stripped": premise_ipa_stripped,
                "hypothesis_ipa_stripped": hypothesis_ipa_stripped,
                "premise_romanized": premise_romanized,
                "hypothesis_romanized": hypothesis_romanized,
            }

        logger.info("Phonemizing + romanizing IndicXNLI %s", config_name)
        ds = ds.map(
            map_fn,
            batched=True,
            batch_size=batch_size,
            num_proc=num_proc,
            desc=f"indicxnli-{config_name}",
        )
        logger.info("Pushing IndicXNLI %s to %s", config_name, repo_id)
        ds.push_to_hub(
            repo_id,
            config_name=config_name,
            set_default=(idx == 0),
            token=os.environ.get("HF_TOKEN"),
            max_shard_size="1GB",
        )


def process_pair_dataset(
    dataset_id: str,
    repo_id: str,
    field_a: str,
    field_b: str,
    out_a: str,
    out_b: str,
    espeak_lang: str,
    cache_dir: str,
    uroman_path: str,
    perl_path: str,
    num_proc: int,
    batch_size: int,
):
    logger.info("Loading dataset %s", dataset_id)
    ds = load_dataset(
        dataset_id,
        cache_dir=cache_dir,
        download_mode="reuse_dataset_if_exists",
    )

    def map_fn(batch):
        a_text = [t or "" for t in batch[field_a]]
        b_text = [t or "" for t in batch[field_b]]

        a_ipa = phonemize_batch(a_text, espeak_lang)
        b_ipa = phonemize_batch(b_text, espeak_lang)

        a_ipa_stripped = [strip_ipa(x) for x in a_ipa]
        b_ipa_stripped = [strip_ipa(x) for x in b_ipa]

        a_romanized = uromanize_batch(a_text, uroman_path, perl_path)
        b_romanized = uromanize_batch(b_text, uroman_path, perl_path)

        return {
            f"{out_a}_ipa_stripped": a_ipa_stripped,
            f"{out_b}_ipa_stripped": b_ipa_stripped,
            f"{out_a}_romanized": a_romanized,
            f"{out_b}_romanized": b_romanized,
        }

    logger.info("Phonemizing + romanizing %s", dataset_id)
    ds = ds.map(
        map_fn,
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        desc=f"pair-{dataset_id}",
    )
    logger.info("Pushing %s to %s", dataset_id, repo_id)
    ds.push_to_hub(
        repo_id,
        token=os.environ.get("HF_TOKEN"),
        max_shard_size="1GB",
    )


def process_extra_pair_datasets(cache_dir: str, uroman_path: str, perl_path: str, num_proc: int, batch_size: int):
    for entry in EXTRA_PAIR_DATASETS:
        process_pair_dataset(
            dataset_id=entry["source"],
            repo_id=entry["repo_id"],
            field_a=entry["field_a"],
            field_b=entry["field_b"],
            out_a=entry["out_a"],
            out_b=entry["out_b"],
            espeak_lang=entry["espeak_lang"],
            cache_dir=cache_dir,
            uroman_path=uroman_path,
            perl_path=perl_path,
            num_proc=num_proc,
            batch_size=batch_size,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--uroman-path", required=True)
    ap.add_argument("--perl-path", default="perl")
    ap.add_argument("--num-proc", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--skip-massive", action="store_true")
    ap.add_argument("--skip-indicxnli", action="store_true")
    ap.add_argument("--skip-extra", action="store_true")
    args = ap.parse_args()

    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN is not set in the environment.")

    if not args.skip_massive:
        process_massive(
            repo_id="mugezhang/massive_ipa_romanized",
            cache_dir=args.cache_dir,
            uroman_path=args.uroman_path,
            perl_path=args.perl_path,
            num_proc=args.num_proc,
            batch_size=args.batch_size,
        )

    if not args.skip_indicxnli:
        process_indicxnli(
            repo_id="mugezhang/indicxnli_ipa_romanized",
            cache_dir=args.cache_dir,
            uroman_path=args.uroman_path,
            perl_path=args.perl_path,
            num_proc=args.num_proc,
            batch_size=args.batch_size,
        )

    if not args.skip_extra:
        process_extra_pair_datasets(
            cache_dir=args.cache_dir,
            uroman_path=args.uroman_path,
            perl_path=args.perl_path,
            num_proc=args.num_proc,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()
