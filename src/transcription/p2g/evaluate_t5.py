import os
from pathlib import Path
from typing import Dict, Optional

from datasets import Dataset
import torch

from config import generate_argparse, load_configs, TrainConfig, CoreDatasetConfig, DatasetFeatureConfig
from dataset import create_dataset, load_hf_dataset
from train_t5 import generate_trainer


def tokenize_dataset(ds_def: DatasetFeatureConfig, ds, tokenizer):
    def preprocess(batch):
        model_inputs = tokenizer(
            batch[ds_def.input_feature]
        )
        return model_inputs
    tds = ds.map(preprocess, batched=True, remove_columns=ds.column_names)
    return tds


def generate_predictions(trainer, tokenizer, tds, batch_size=64):
    model = trainer.model
    model.eval()
    collator = trainer.data_collator

    preds = []

    for start in range(0, len(tds), batch_size):
        features = [tds[i] for i in range(start, min(start + batch_size, len(tds)))]
        batch = collator(features)
        batch = {k: v.to(model.device) for k, v in batch.items()}

        with torch.no_grad():
            generated = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                **trainer.model.generation_config.to_diff_dict()
            )

        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        preds.extend(decoded)

    return preds


def evaluate_dataset(ctx: TrainConfig, ds_def: Optional[CoreDatasetConfig], ds: Dataset, checkpoint: Path) -> Dict[str, float]:
    trainer, tokenizer = generate_trainer(ctx, None, ds, ds_def, checkpoint)
    if ds_def is not None and ds_def.prediction_file is not None:
        tds = tokenize_dataset(ds_def if ds_def is not None else ctx.dataset, ds, tokenizer)
        decoded_predictions = generate_predictions(trainer, tokenizer, tds)
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

    if conf.cpus < 0:
        conf.cpus = os.cpu_count()

    if conf.evaluation.datasets is None:
        _, ds = create_dataset(conf)
        print(evaluate_dataset(conf, None, ds, args.checkpoint))
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
