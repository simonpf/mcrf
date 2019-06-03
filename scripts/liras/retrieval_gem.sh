#!/usr/bin/env bash
#SBATCH -A C3SE2019-1-15 -p vera
#SBATCH -n 128
#SBATCH -c 2
#SBATCH -J clouds
#SBATCH -t 0-10:00:00

cd ${HOME}/src/crac
source ops/setup_vera.sh
export OMP_NUM_THREADS=1

s="SCENE"

mpiexec -n 128 python scripts/liras/retrieval_gem.py ${s} 3000 data/forward_simulations_${s}_noise.nc data/retrieval_${s}_gem.nc
