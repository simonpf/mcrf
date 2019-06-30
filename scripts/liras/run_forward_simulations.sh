#!/usr/bin/env bash
#SBATCH -A C3SE2019-1-15 -p vera
#SBATCH -n 128
#SBATCH -c 4
#SBATCH -J cloud_simulations
#SBATCH -t 0-04:00:00

cd ${HOME}/src/crac
source ops/setup_vera.sh
export OMP_NUM_THREADS=1

mpiexec -n 128 python scripts/liras/forward_simulations.py A 2800 3600 ${SNIC_NOBACKUP}/src/liras/data/forward_simulations_a.nc lcpr mwi mwi_full ici
mpiexec -n 128 python scripts/liras/forward_simulations.py B 2200 3000 ${SNIC_NOBACKUP}/src/liras/data/forward_simulations_b.nc lcpr mwi mwi_full ici
mpiexec -n 128 python scripts/liras/forward_simulations.py A 2800 3600 ${SNIC_NOBACKUP}/src/liras/data/forward_simulations_a_hamp.nc hamp_space mwi mwi_full ici
mpiexec -n 128 python scripts/liras/forward_simulations.py B 2200 3000 ${SNIC_NOBACKUP}/src/liras/data/forward_simulations_b_hamp.nc hamp_space mwi mwi_full ici
