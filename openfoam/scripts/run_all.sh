#!/bin/bash
# Build the SA-AI library and run all flat-plate cases (serial per case,
# cases in parallel -- each is only 25.6k cells).
set -e
OFROOT=/local_data/qiqi/openfoam-sa-ai/OpenFOAM-v2412
SRC=/home/qiqi/flexcompute/sa-ai/openfoam/src
CASES=/local_data/qiqi/openfoam-sa-ai/cases

source $OFROOT/etc/bashrc || true

( cd $SRC && wmake libso )

for cd in $CASES/flatplate_Tu*; do
    (
        cd $cd
        [ -d constant/polyMesh ] || blockMesh > log.blockMesh 2>&1
        simpleFoam > log.simpleFoam 2>&1
        echo "$(basename $cd): exit $?"
    ) &
done
wait
echo ALL_DONE
