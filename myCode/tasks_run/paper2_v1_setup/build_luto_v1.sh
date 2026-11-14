#!/bin/bash
#PBS -N build_luto_v1
#PBS -P jk53
#PBS -q copyq
#PBS -l ncpus=1
#PBS -l mem=16GB
#PBS -l walltime=03:00:00
#PBS -l storage=gdata/jk53
#PBS -l wd
# Environment for paper2_v1 (current LUTO2 requirements.yml without CPLEX, docplex and cuOpt, which the model does not use yet)
set -e
source $HOME/apps/miniforge3/etc/profile.d/conda.sh
B=/g/data/jk53/LUTO_XH/envs
export CONDA_PKGS_DIRS=$B/pkgs
$HOME/apps/miniforge3/bin/mamba env create -y -p $B/luto_v1 -f $B/luto_v1_requirements.yml
$B/luto_v1/bin/python -c "import numpy, pandas, xarray, gurobipy, sys; print(sys.version.split()[0], numpy.__version__, pandas.__version__, xarray.__version__, gurobipy.gurobi.version())"
echo BUILD_DONE
