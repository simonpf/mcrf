#!/usr/bin/env bash
#SBATCH -A C3SE2020-1-7 -p vera
#SBATCH -n 256
#SBATCH -c 2
#SBATCH -J clouds
#SBATCH -t 0-10:00:00
#SBATCH --mail-type END
#SBATCH --mail-user simon.pfreundschuh@chalmers.se

cd ${HOME}/src/crac
source ops/setup_vera.sh
export OMP_NUM_THREADS=1

s="SCENE"
s1="SHAPE1"
s2="SHAPE2"

s="a"
mpiexec -n 256 -output-filename combined python scripts/liras/retrieval.py ${s} 3000 ${s1} ${s2} data/forward_simulations_${s}_noise.nc data/retrieval_${s}_${s1}_${s2}.nc

# Reference run
#mpiexec -n 256 --output-filename scripts/liras/reference_retrieval python scripts/liras/retrieval.py ${s} 2800 ${s1} ${s2} data/forward_simulations_${s}_noise.nc data/retrieval_reference_${s}_${s1}_${s2}.nc --reference

#s="b"
#mpiexec -n 256 --output-filename log python scripts/liras/retrieval.py ${s} 2200 ${s1} ${s2} data/forward_simulations_${s}_noise.nc data/retrieval_${s}_${s1}_${s2}.nc
#
# Reference run
#mpiexec -n 256 python scripts/liras/retrieval.py ${s} 2200 ${s1} ${s2} data/forward_simulations_${s}_noise.nc data/retrieval_reference_${s}_${s1}_${s2}.nc --reference
