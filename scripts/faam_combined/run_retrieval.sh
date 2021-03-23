#! /bin/bash

source ~/.bashrc

function replace_variables () {
    sed "s=SHAPE=$1=;s=FLIGHT=$2=;" ${HOME}/src/crac/scripts/faam_combined/retrieval.sh
}

LIRAS_PATH=${HOME}/src/liras

replace_variables LargePlateAggregate 161 | sbatch
#replace_variables LargePlateAggregate 159 | sbatch
#replace_variables 8-ColumnAggregate 159 | sbatch
#replace_variables 8-ColumnAggregate 161 | sbatch
#replace_variables SectorSnowflake | sbatch
#replace_variables 6-BulletRosette | sbatch
#replace_variables 8-ColumnAggregate | sbatch
#replace_variables PlateType1 | sbatch
#replace_variables IconCloudIce | sbatch
#replace_variables ColumnType1 | sbatch
#replace_variables SectorSnowflake | sbatch
