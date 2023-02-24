#!/bin/bash

# Assumes that this is run from the root of the repository

# Create a new conda environment
conda create --name neural_clbf python=3.9
conda activate neural_clbf

# Install with pip
pip install -e .
pip install -r requirements.txt