#!/usr/bin/env bash
#SBATCH -A C3SE2021-1-12 -p vera
#SBATCH -n 256
#SBATCH -c 2
#SBATCH -J clouds
#SBATCH -t 0-10:00:00
#SBATCH --mail-type END
#SBATCH --mail-user simon.pfreundschuh@chalmers.se

export JOINT_FLIGHT_PATH=${HOME}/src/joint_flight
export LIRAS_PATH=${HOME}/src/joint_flight
export OMP_NUM_THREADS=1
cd ${HOME}/src/crac

source ops/setup_vera.sh
mpiexec -n 1 -output-filename combined python scripts/faam_combined/retrieval.py 198 200  LargePlateAggregate.xml 
