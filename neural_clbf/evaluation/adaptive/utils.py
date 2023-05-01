"""
utils.py
Description:
    Extra files that can be shared across multiple evaluation scripts.
"""

import numpy as np
import matplotlib.pyplot as plt

def load_yaml_trajectory_data(yaml_filename: str):
    # Constants

    # Load data
    trajopt_file_data = yaml.load(open(yaml_filename, "r"), Loader=yaml.FullLoader)

    # Extract the trajectories
    U_trajopt = torch.zeros(
        (trajopt_file_data["num_x0s"], trajopt_file_data["horizon"], dynamics_model.n_controls)
    )
    X_trajopt = torch.zeros(
        (trajopt_file_data["num_x0s"], trajopt_file_data["horizon"] + 1, dynamics_model.n_dims)
    )
    for ic_index in range(trajopt_file_data["num_x0s"]):
        U_trajopt[ic_index, :, :] = torch.tensor(trajopt_file_data["U" + str(ic_index)])
        X_trajopt[ic_index, :, :] = torch.tensor(trajopt_file_data["X" + str(ic_index)])

    return U_trajopt, X_trajopt