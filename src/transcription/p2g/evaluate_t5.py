import os
from pathlib import Path
from typing import Dict, Optional

from datasets import Dataset
import torch
from tqdm import tqdm

from config import generate_argparse, load_configs, TrainConfig, CoreDatasetConfig, DatasetFeatureConfig
from dataset_loading import load_existing_dataset, load_hf_dataset, validate_preprocessed_dataset
from setup import setup_logging, setup_tokenizer
from train_t5 import generate_trainer


def generate_predictions(trainer, tokenizer, tds, batch_size=8):
    model = trainer.model
    model.eval()
    collator = trainer.data_collator

    preds = []

    for start in tqdm(range(0, len(tds), batch_size), desc="processing batches"):
        features = [tds[i] for i in range(start, min(start + batch_size, len(tds)))]
        batch = collator(features)

        batch = {
            k: v.to(model.device)
            for k, v in batch.items()
            if v is not None and k in ("input_ids", "attention_mask")
        }

        with torch.no_grad():
            generated = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )

        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        preds.extend(decoded)

    return preds


def evaluate_dataset(
        ctx: TrainConfig,
        ds_def: DatasetFeatureConfig,
        ds: Dataset,
        checkpoint: Path,
) -> Dict[str, float]:
    validate_preprocessed_dataset(ds, ds_def.split, ds_def.name)
    processed_ds = ds
    trainer, tokenizer = generate_trainer(ctx, None, processed_ds, None, ds_def, checkpoint)

    if ds_def.prediction_file is not None:
        decoded_predictions = generate_predictions(trainer, tokenizer, processed_ds)
        result_fname = ctx.evaluation.results_prefix / ds_def.prediction_file
        result_fname.parent.mkdir(parents=True, exist_ok=True)
        lines = '\n'.join(decoded_predictions)
        with open(result_fname, 'w+') as f:
            f.write(lines)
    return trainer.evaluate()


def main():
    ap = generate_argparse()
    ag = ap.add_argument_group('evaluation')
    ag.add_argument('--checkpoint', type=str, required=True, help='The checkpoint to be evaluated.')
    args = ap.parse_args()
    conf = load_configs(args.configs, args.default_config)

    setup_logging()

    if conf.cpus < 0:
        conf.cpus = os.cpu_count()

    if conf.evaluation.datasets is None:
        _, _, ds, ds_def = load_existing_dataset(conf)
        print(evaluate_dataset(conf, ds_def, ds, args.checkpoint))
        return

    for ds_def in conf.evaluation.datasets:
        ds = load_hf_dataset(ds_def, conf.dataset.hf_cache, False)
        if ds_def.subset is None:
            print(f"{ds_def.name}({ds_def.split})")
        else:
            print(f"{ds_def.name}:{ds_def.subset}({ds_def.split})")
        print(evaluate_dataset(conf, ds_def, ds, args.checkpoint))


if __name__ == "__main__":
    main()
