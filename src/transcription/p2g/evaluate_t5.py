from config import generate_argparse, load_configs
from dataset import create_dataset, load_hf_dataset
from train_t5 import generate_trainer, setup_wandb


def main():
    ap = generate_argparse()
    ag = ap.add_argument_group('evaluation')
    ag.add_argument('--checkpoint', type=str, required=True, help='The checkpoint to be evaluated.')
    args = ap.parse_args()
    conf = load_configs(args.configs, args.default_config)

    if conf.evaluation_dataset is None:
        _, ds = create_dataset(conf)
    else:
        ds = load_hf_dataset(conf.evaluation_dataset, conf.dataset.hf_cache, False)

    trainer = generate_trainer(conf, None, ds, args.checkpoint)

    metrics = trainer.evaluate()
    print(metrics)


if __name__ == "__main__":
    main()
