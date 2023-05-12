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
    # Loading Data
    ap.add_argument(
        '--checkpoint_path', type=str, default=None,
        help='Path to checkpoint to load from (default: None)',
    )
    ap.add_argument(
        '--saved_Vnn_subpath', type=str, default=None,
        help='Path to saved Vnn to load from (default: None)',
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
        '--gradient_clip_val', type=float, default=float('Inf'),
        help='Gradient clipping value (default: Inf)',
    )
    ap.add_argument(
        '--clf_relaxation_penalty', type=float, default=1e2,
        help='Penalty for the relaxation of the CLF (default: 1e2)',
    )
    ap.add_argument(
        '--max_iters_cvxpylayer', type=int, default=int(5e7),
        help='Maximum number of iterations for cvxpylayers (default: 5e7)',
    )

    # Number of Epochs to devote to X
    ap.add_argument(
        '--learn_shape_epochs', type=int, default=20,
        help='Number of epochs to devote to learning the shape (default: 20)',
    )

    # Including Certain Losses
    ap.add_argument(
        '--include_estimation_error_loss', type=bool, default=False,
        help='Whether to use the estimation error loss in training (default: False)',
    )
    ap.add_argument(
        '--include_oracle_loss', type=bool, default=False,
        help='Whether to use the oracle loss in training(default: False)',
    )

    # GPU / Multi-GPU Setups
    # This doesn't currently work. I get this error whenever I try to use more than 1 GPU:
    #   AttributeError: 'EpisodicDataModuleAdaptive' object has no attribute 'validation_data'
    # which indicates that the data module is not being properly distributed across the GPUs. This might require a
    # specific type of dataloader or sampler to be used in order to overcome this. Consider this one:
    #   https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler
    ap.add_argument(
        '--number_of_gpus', type=int, default=0,
        help='Number of GPUs to use (default: 1). TODO: Test how well this works? (See comment in utils.py for more details)',
    )
    ap.add_argument(
        '--num_cpu_cores', type=int, default=10,
        help='Number of CPU cores to use (default: 10)',
    )
