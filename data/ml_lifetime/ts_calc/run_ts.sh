#!/bin/bash
# TS calculations for 3 representative organolithiums
# Mechanism: RLi + THF → R-H + lithium enolate (THF cleavage)
# Method: M06-2X/def2-SVP NEB-TS

export ORCA_DIR=/Users/zhaowenyuan/Downloads/orca_6_1_1_macosx_arm64_openblas_openmpi411
export PATH=$ORCA_DIR:$PATH
export DYLD_LIBRARY_PATH=$ORCA_DIR:$DYLD_LIBRARY_PATH

WORKDIR=/Users/zhaowenyuan/Projects/FlowFigTabMiner/data/ml_lifetime/ts_calc

cd $WORKDIR

for mol in phli oxiranyl oco2me; do
    echo "=== Running $mol ==="
    $ORCA_DIR/orca ${mol}_ts.inp > ${mol}_ts.out 2>&1
    echo "Exit: $?"
    grep "FINAL SINGLE POINT ENERGY" ${mol}_ts.out | tail -1
    echo ""
done
