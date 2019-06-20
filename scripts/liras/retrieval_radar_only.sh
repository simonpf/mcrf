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
s1="SHAPE1"
s2="SHAPE2"

#mpiexec -n 128 python scripts/liras/retrieval_radar_only.py ${s} 3000 ${s1} ${s2} data/forward_simulations_${s}_noise.nc data/retrieval_${s}_${s1}_${s2}_ro.nc
#mpiexec -n 128 python scripts/liras/retrieval_ice_radar_only.py ${s} 3000 ${s1} data/forward_simulations_ice_${s}_noise.nc data/retrieval_ice_${s}_${s1}_ro.nc
#mpiexec -n 128 python scripts/liras/retrieval_ice_radar_only.py ${s} 3000 ${s1} data/forward_simulations_snow_${s}_noise.nc data/retrieval_snow_${s}_${s1}_ro.nc
#mpiexec -n 128 python scripts/liras/retrieval_ice_snow_radar_only.py ${s} 3000 ${s1} ${s2} data/forward_simulations_ice_snow_${s}_noise.nc data/retrieval_ice_snow_${s}_${s1}_${s2}_ro.nc

s="b"
mpiexec -n 128 python scripts/liras/retrieval_radar_only.py ${s} 2200 ${s1} ${s2} data/forward_simulations_${s}_noise.nc data/retrieval_${s}_${s1}_${s2}_ro.nc
