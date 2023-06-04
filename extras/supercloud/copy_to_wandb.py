"""
copy_to_wandb.py
Description:
    Uploads a wandb dataset from supercloud onto the cloud.
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

    return_code = os.system("scp -r {} {}".format(source_directory, destination_directory))
    if return_code != 0:
        raise RuntimeError("scp command failed.")

def copy_jobid_stdout_for_processing(args):
    """
    copy_jobid_stdout_for_processing(args)
    Description:
        Copy jobid stdout for processing
    """

    # Create source directory name
    source_directory = "krutledge@txe1-login.mit.edu:~/neural_clbf/extras/supercloud/outputs/"
    source_filename = ""
    if args.system == "ps":
        source_filename += "pusher-slider-"
    elif args.system == "scalar":
        source_filename += "scalar-"
    elif args.system == "ls":
        source_filename += "loaded-manipulator-"
    else:
        raise ValueError("System {} not recognized.".format(args.system))

    # Add Job ID to source name
    source_filename = source_filename + "{}.stdout".format(args.job_id)

    # Copy file to stdouts directory
    destination_directory = "./data/stdouts/"
    return_code = os.system("scp -r {} {}".format(source_directory + source_filename, destination_directory))
    if return_code != 0:
        raise RuntimeError("scp command failed.")

    # Create name of file on local machine
    local_filename = "./data/stdouts/{}".format(source_filename)

    return local_filename


def find_logfile_location_on_supercloud(local_stdout_filename):
    """
    find_logfile_location_on_supercloud(local_stdout_filename)
    Description:
        Extracts the location of the logfile on supercloud from the local stdout file.
    """
    # Constants

    # Open local file
    f = os.open(local_stdout_filename, "r")
    for line in f.readlines():
        if line.startswith("wandb: Find logs at: "):
            return line.split("wandb: Find logs at: ")[1].strip()


def copy_logfile_from_supercloud(supercloud_logfilename):
    """
    copy_logfile_from_supercloud(supercloud_logfilename)
    Description:
        Copies the logfile from supercloud to the local data directory.
    """

    # Create source directory name
    source_directory = "krutledge@txe1-login.mit.edu:~/neural_clbf/neural_clbf/training/adaptive_w_observed_parameters/wandb/"

    # Extract offline_run name from supercloud_logfilename
    offline_run = logfile_declared_name_to_offline_run_name(supercloud_logfilename)
    source_directory += "{}/".format(offline_run)
    print(supercloud_logfilename)
    print(source_directory)

    # Copy file to wandb directory
    destination_directory = "./data/wandb/"
    return_code = os.system("scp -r {} {}".format(source_directory, destination_directory))
    if return_code != 0:
        raise RuntimeError("scp command failed.")

def logfile_declared_name_to_offline_run_name(logfile_declared_name):
    """
    logfile_declared_name_to_offline_run_name(logfile_declared_name)
    Description:
        Converts the logfile declared name to the offline run name.
    """
    return logfile_declared_name.split("/")[-2]

def upload_logfile_to_wandb(supercloud_logfilename):
    """
    upload_logfile_to_wandb(supercloud_logfilename)
    Description:
        Uploads the logfile to wandb.
    """

    # Create source directory name
    local_logfile = "{}/data/wandb/{}".format(
        os.getcwd(),
        logfile_declared_name_to_offline_run_name(supercloud_logfilename),
    )

    return_code = os.system("wandb sync {}".format(local_logfile))
    if return_code != 0:
        raise RuntimeError("wandb command failed.")

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
    parser.add_argument(
        '--job_id', type=int, default=0,
        help='The job id of the job whose data we want to copy. Default is 0.',
    )
    args = parser.parse_args()

    # Copy data
    add_ssh_key() # Add ssh-keys to identify.
    local_stdout_filename = copy_jobid_stdout_for_processing(args) # Copy stdout data to local machine.
    copy_logfile_from_supercloud(local_stdout_filename)
    upload_logfile_to_wandb(local_stdout_filename)
