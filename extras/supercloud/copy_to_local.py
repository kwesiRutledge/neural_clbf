"""
copy_to_local.py
Description:

"""

import os
from argparse import ArgumentParser

# =========
# Functions
# =========

def add_ssh_key():
    """
    add_ssh_key()
    Description:
        Add ssh key to ssh agent
    """

    os.system("eval $(ssh-agent)")
    os.system("DISPLAY=1 SSH_ASKPASS=\"./x.sh\" ssh-add ~/.ssh/id_rsa_supercloud < /dev/null ")

def copy_directory_using_scp(args):
    """
    copy_file_using_scp(args)
    Description:
        Copy file using scp
    """

    # Get arguments

    # Create source directory name
    source_directory = "krutledge@txe1-login.mit.edu:~/neural_clbf/neural_clbf/training/adaptive_w_observed_parameters/logs/"
    if args.system == "ps":
        source_directory += "pusher_slider_sticking_force_input/"
    elif args.system == "scalar":
        source_directory += "scalar_demo_capa2_system/"
    elif args.system == "ls":
        source_directory += "load_sharing_manipulator/"
    else:
        raise ValueError("System {} not recognized.".format(args.system))

    # Add commit to directory name
    source_directory += "commit_{}/".format(args.commit)

    # Add destination directory name
    destination_directory = "./data/"

    os.system("scp -r {} {}".format(source_directory, destination_directory))


if __name__ == "__main__":
    # Input Processing
    parser = ArgumentParser(
        description="This script copies a specific commit's set of datasets into the data folder of the local project.",
    )
    parser.add_argument(
        '--system', type=str, default="ps",
        help='The system whose data we want to copy. Options are: ps, ls, tt, tt2, scalar. Default is ps.',
    )
    parser.add_argument(
        '--commit', type=str, default="435228a",
        help='The commit whose data we want to copy. Default is 435228a.',
    )
    args = parser.parse_args()

    # Copy data
    add_ssh_key() # Add ssh-keys to identify.
    copy_directory_using_scp(args) # Copy data.
