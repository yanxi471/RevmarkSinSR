import argparse

from omegaconf import OmegaConf

from trainer_lr_storage import TrainerLRStoragePredictor


def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./saved_logs",
        help="Folder to save checkpoints and the training log",
    )
    parser.add_argument(
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="Resume from an lr_storage predictor checkpoint",
    )
    parser.add_argument(
        "--cfg_path",
        type=str,
        default="./configs/SinSR_wm_phaseE_predictor.yaml",
        help="Config yaml path",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=15,
        help="Number of diffusion steps for the frozen cover generator",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_parser()
    configs = OmegaConf.load(args.cfg_path)
    configs.diffusion.params.steps = args.steps

    for key in vars(args):
        if key in ['cfg_path', 'save_dir', 'resume']:
            configs[key] = getattr(args, key)

    trainer = TrainerLRStoragePredictor(configs)
    trainer.train()
