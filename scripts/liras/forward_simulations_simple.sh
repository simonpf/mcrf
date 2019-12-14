#!/usr/bin/env bash
#SBATCH -A C3SE2019-1-15 -p vera
#SBATCH -n 128
#SBATCH -c 2
#SBATCH -J cloud_simulations
#SBATCH -t 0-04:00:00

cd ${HOME}/src/crac
source ops/setup_vera.sh
export OMP_NUM_THREADS=1

mpiexec -n 128 python scripts/liras/forward_simulations_simple.py A 2800 3600 SHAPE ${SNIC_NOBACKUP}/src/liras/data/forward_simulations_SHAPE_a.nc lcpr mwi mwi_full ici
mpiexec -n 128 python scripts/liras/forward_simulations_simple.py B 2200 3000 SHAPE ${SNIC_NOBACKUP}/src/liras/data/forward_simulations_SHAPE_b.nc lcpr mwi mwi_full ici
