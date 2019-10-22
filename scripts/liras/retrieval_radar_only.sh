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

#s="a"
#mpiexec -n 128 python scripts/liras/retrieval_radar_only.py ${s} 2800 ${s1} ${s2} data/forward_simulations_${s}_noise.nc data/retrieval_${s}_${s1}_${s2}_ro.nc

# reference run
#mpiexec -n 128 python scripts/liras/retrieval_radar_only.py ${s} 2800 ${s1} ${s2} data/forward_simulations_${s}_noise.nc data/retrieval_reference_${s}_${s1}_${s2}_ro.nc --reference
#
s="b"
mpiexec -n 128 python scripts/liras/retrieval_radar_only.py ${s} 2200 ${s1} ${s2} data/forward_simulations_${s}_noise.nc data/retrieval_${s}_${s1}_${s2}_ro.nc
#
#mpiexec -n 128 python scripts/liras/retrieval_radar_only.py ${s} 2200 ${s1} ${s2} data/forward_simulations_${s}_noise.nc data/retrieval_reference_${s}_${s1}_${s2}_ro.nc --reference
