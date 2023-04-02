import subprocess
from argparse import ArgumentParser


def current_git_hash() -> str:
    return (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .decode("ascii")
        .strip()
    )

def initialize_training_arg_parser(ap: ArgumentParser):
    """
    initialize_training_arg_parser(parser)

    Description:
        This function declares the arguments that are allowed to be passed to a training script for the adaptive CLBF
        experiments.

        TODO: Add multi-GPU at some point?
    """

    # Random Seed Setups
    # ==================
    ap.add_argument(
        '--pt_random_seed', type=int, default=361,
        help='Integer used as PyTorch\'s random seed (default: 361)',
    ),
    ap.add_argument(
        '--np_random_seed', type=int, default=30,
        help='Integer used as Numpy\'s random seed (default: 30)'
    )
    # ap.add_argument('--gpus', type=int, default=1)
    ap.add_argument('--max_epochs', type=int, default=6)
    ap.add_argument(
        '--checkpoint_path', type=str, default=None,
        help='Path to checkpoint to load from (default: None)',
    )
    ap.add_argument(
        '--include_oracle_loss', type=bool, default=False,
        help='Whether to use the oracle loss in training(default: False)',
    )
    ap.add_argument(
        '--barrier', type=bool, default=False,
        help='Whether to use the barrier loss in training(default: False)',
    )
    ap.add_argument(
        '--safe_level', type=float, default=0.5,
        help='Safe level for the CLBF (default: 0.5)',
    )
    ap.add_argument(
        '--clf_lambda', type=float, default=0.01,
        help='Lambda for the CLF (default: 0.01)',
    )
    ap.add_argument(
        '--gradient_clip_val', type=float, default=0.5,
        help='Gradient clipping value (default: 0.5)',
    )
    # Including Certain Losses
    ap.add_argument(
        '--use_estim_err_loss', type=bool, default=False,
        help='Whether to use the estimation error loss in training(default: False)',
    )
