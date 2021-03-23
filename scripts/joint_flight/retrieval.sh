#!/usr/bin/env bash
#SBATCH -A C3SE2021-1-12 -p vera
#SBATCH -n 256
#SBATCH -c 2
#SBATCH -J clouds
#SBATCH -t 0-20:00:00
#SBATCH --mail-type END

export OMP_NUM_THREADS=1
cd ${HOME}/src/crac
s="SHAPE"

source ops/setup_vera.sh
mpiexec -n 256 -output-filename combined python scripts/joint_flight/retrieval.py 0 1441 ${s}

