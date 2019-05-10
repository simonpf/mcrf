#!/usr/bin/env bash
#SBATCH -A C3SE2019-1-15 -p vera
#SBATCH -n 128
#SBATCH -c 2
#SBATCH -J cloud_simulations
#SBATCH -t 0-02:00:00

cd ${HOME}/src/crac
source ops/setup_vera.sh
export OMP_NUM_THREADS=1

mpiexec -n 128 python scripts/liras/forward_simulations.py A 3000 3800 ${HOME}/src/liras/data/forward_simulations_a.nc
mpiexec -n 128 python scripts/liras/forward_simulations.py B 2200 3000 ${HOME}/src/liras/data/forward_simulations_b.nc
