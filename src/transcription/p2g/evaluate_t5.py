import os
from pathlib import Path
from typing import Dict, Optional

from datasets import Dataset

from config import generate_argparse, load_configs, TrainConfig, CoreDatasetConfig
from dataset import create_dataset, load_hf_dataset
from train_t5 import generate_trainer


def evaluate_dataset(ctx: TrainConfig, ds_def: Optional[CoreDatasetConfig], ds: Dataset, checkpoint: Path) -> Dict[str, float]:
    trainer, tokenizer = generate_trainer(ctx, None, ds, ds_def, checkpoint)
    if ds_def.prediction_file is not None:
        predictions = trainer.predict(ds)
        decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        result_fname = ctx.evaluation.result_prefix / ds_def.prediction_file
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
