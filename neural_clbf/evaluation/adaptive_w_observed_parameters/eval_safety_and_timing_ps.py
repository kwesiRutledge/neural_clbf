"""
eval_pssfi_safety_case_study.py
Description:
    This script will evaluate the safety of the Pusher Slider System with Force Input (PSSFI) system using the CLBF
    and other comparable methods.
"""

import yaml
import jax
import jax.numpy as jnp


# Constants
yaml_data_file = 'data/trajax2_data_Apr-06-2023-13:16:09.yml'


with open(yaml_data_file, 'r') as f:
    config = yaml.safe_load(f)

    print(config)


