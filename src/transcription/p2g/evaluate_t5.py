from pathlib import Path
from typing import Dict

from datasets import Dataset

from config import generate_argparse, load_configs, TrainConfig
from dataset import create_dataset, load_hf_dataset
from train_t5 import generate_trainer


def evaluate_dataset(ctx: TrainConfig, ds: Dataset, checkpoint: Path) -> Dict[str, float]:
    trainer = generate_trainer(ctx, None, ds, checkpoint)
    return trainer.evaluate()


def main():
    ap = generate_argparse()
    ag = ap.add_argument_group('evaluation')
    ag.add_argument('--checkpoint', type=str, required=True, help='The checkpoint to be evaluated.')
    args = ap.parse_args()
    conf = load_configs(args.configs, args.default_config)

    if conf.evaluation_datasets is None:
        _, ds = create_dataset(conf)
        print(evaluate_dataset(conf, ds, args.checkpoint))
        return

    for ds_def in conf.evaluation_datasets:
        ds = load_hf_dataset(ds_def, conf.dataset.hf_cache, False)
        if ds_def.subset is None:
            print(f"{ds_def.name}({ds_def.split})")
        else:
            print(f"{ds_def.name}:{ds_def.subset}({ds_def.split})")
        print(evaluate_dataset(conf, ds, args.checkpoint))


if __name__ == "__main__":
    main()
