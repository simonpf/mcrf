#! /bin/bash

source ~/.bashrc

function replace_variables () {
    sed "s=SHAPE=$1=;" ${HOME}/src/crac/scripts/liras/forward_simulations_simple.sh
}

LIRAS_PATH=${HOME}/src/liras

replace_variables 8-ColumnAggregate | sbatch
replace_variables LargePlateAggregate | sbatch
replace_variables GemSnow None | sbatch
replace_variables GemCloudIce | sbatch
replace_variables PlateType1 | sbatch
replace_variables GemGraupel | sbatch
