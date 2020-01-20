#! /bin/bash

source ~/.bashrc

function replace_variables () {
    sed "s=SCENE=$1=;s=SHAPE1=$2=;s=SHAPE2=$3=;" ${HOME}/src/crac/scripts/liras/retrieval.sh
}

LIRAS_PATH=${HOME}/src/liras

replace_variables a 8-ColumnAggregate None | sbatch
replace_variables a LargePlateAggregate None | sbatch
replace_variables a LargeColumnAggregate None | sbatch
replace_variables a IceSphere None | sbatch
replace_variables a SectorSnowflake None | sbatch
#replace_variables a 8-ColumnAggregate LargePlateAggregate | sbatch
#replace_variables a GemCloudIce GemSnow | sbatch

#replace_variables a 8-ColumnAggregate LargePlateAggregate | sbatch
#replace_variables a IconSnow LargePlateAggregate | sbatch
#replace_variables a PlateType1 LargePlateAggregate | sbatch
#replace_variables a 8-ColumnAggregate LargeBlockAggregate | sbatch
#replace_variables a IconSnow LargeBlockAggregate | sbatch
#replace_variables a PlateType1 LargeBlockAggregate | sbatch
#replace_variables a GemCloudIce GemSnow | sbatch
