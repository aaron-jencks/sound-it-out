import logging
from pathlib import Path

from datasets import DatasetDict
from transformers import set_seed

from transcription.p2g.common import format_language_marker
from transcription.p2g.config import TokenizingConfig
from transcription.p2g.dataset import resolve_language
from transcription.p2g.dataset_loading import load_saved_dataset
from transcription.p2g.setup import parse_args, setup_tokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)


def tokenize_existing_dataset(config: TokenizingConfig):
    ds = load_saved_dataset(config.target_dataset, config.hf_cache, preprocessed=False)
    columns = (ds["train"] if isinstance(ds, DatasetDict) else ds).column_names

    tokenizer = setup_tokenizer(config.tokenizer, None, ds, None, config.target_dataset)
    def tokenize(batch):
        input_features = []
        output_features = []
        for lang, feat_in, feat_out in zip(
            batch[config.target_dataset.language_feature],
            batch[config.target_dataset.input_feature],
            batch[config.target_dataset.output_feature]
        ):
            marker = format_language_marker(resolve_language(config.target_dataset, lang))
            input_features.append(f"{marker}{feat_in}")
            output_features.append(f"{marker}{feat_out}")

        model_inputs = tokenizer(
            input_features,
            truncation=True,
            padding=False,
            max_length=config.tokenizer.max_sequence_length,
        )
        labels = tokenizer(
            output_features,
            truncation=True,
            padding=False,
            max_length=config.tokenizer.max_sequence_length,
        )

        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    tokenized_ds = ds.map(tokenize, batched=True, remove_columns=columns, num_proc=config.cpus)
    output_path_name = config.hf_cache / f"{config.target_dataset.name}-tokenized"
    tokenized_ds.save_to_disk(output_path_name)
    logger.info(f"saved tokenized dataset to {output_path_name}")


def main():
    config = parse_args(
        "tokenizes an existing dataset",
        default_config=Path("transcription/p2g/config/default_core.json"),
        schema=TokenizingConfig,
    )
    logger.info("setting seed")
    set_seed(config.random_seed)
    logger.info("tokenizing dataset")
    tokenize_existing_dataset(config)


if __name__ == "__main__":
    main()