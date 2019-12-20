#! /bin/bash

source ~/.bashrc

function replace_variables () {
    sed "s=SHAPE=$1=;" ${HOME}/src/crac/scripts/liras/forward_simulations_simple.sh
}

LIRAS_PATH=${HOME}/src/liras

replace_variables 8-ColumnAggregate | sbatch
replace_variables IceSphere | sbatch
replace_variables LargePlateAggregate | sbatch
replace_variables LargeColumnAggregate None | sbatch
replace_variables SectorSnowflake | sbatch
