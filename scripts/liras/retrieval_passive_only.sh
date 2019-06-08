#!/usr/bin/env bash
#SBATCH -A C3SE2019-1-15 -p vera
#SBATCH -n 256
#SBATCH -c 2
#SBATCH -J clouds
#SBATCH -t 0-10:00:00

cd ${HOME}/src/crac
source ops/setup_vera.sh
export OMP_NUM_THREADS=1

s="SCENE"
s1="SHAPE1"
s2="SHAPE2"

#mpiexec -n 128 python scripts/liras/retrieval_passive_only.py ${s} 3000 ${s1} data/forward_simulations_${s}_noise.nc data/retrieval_${s}_${s1}_po.nc
#mpiexec -n 256 python scripts/liras/retrieval_ice_passive_only.py ${s} 3000 ${s1} data/forward_simulations_${s}_noise.nc data/retrieval_ice_${s}_${s1}_po.nc
#mpiexec -n 256 python scripts/liras/retrieval_snow_passive_only.py ${s} 3000 ${s1} data/forward_simulations_${s}_noise.nc data/retrieval_snow_${s}_${s1}_po.nc
mpiexec -n 256 python scripts/liras/retrieval_ice_snow_passive_only.py ${s} 3000 ${s1} ${s2} data/forward_simulations_${s}_noise.nc data/retrieval_ice_snow_${s}_${s1}_${s2}_po.nc
