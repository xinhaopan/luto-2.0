#!/bin/bash

# Add the Conda binaries to the PATH
source ~/.bashrc

# Create a new Conda environment, and install the required packages
mamba env create -f xinhao_env.yml -c conda-forge

