#!/bin/bash
# M06-2X/def2-SVP + CPCM(THF) for oxiranylLi reactant and TS
# Compare with gas-phase results to quantify solvation contribution

export ORCA_DIR=/Users/zhaowenyuan/Downloads/orca_6_1_1_macosx_arm64_openblas_openmpi411
export PATH=$ORCA_DIR:$PATH
export DYLD_LIBRARY_PATH=$ORCA_DIR:$DYLD_LIBRARY_PATH

WORKDIR=/Users/zhaowenyuan/Projects/FlowFigTabMiner/data/ml_lifetime/ts_calc
cd $WORKDIR

echo "=== Reactant + CPCM(THF) ==="
$ORCA_DIR/orca oxiranyl_opt_cpcm.inp > oxiranyl_opt_cpcm.out 2>&1
echo "Exit: $?"
grep "FINAL SINGLE POINT ENERGY" oxiranyl_opt_cpcm.out | tail -1

echo ""
echo "=== TS + CPCM(THF) ==="
$ORCA_DIR/orca oxiranyl_optts_cpcm.inp > oxiranyl_optts_cpcm.out 2>&1
echo "Exit: $?"
grep "FINAL SINGLE POINT ENERGY" oxiranyl_optts_cpcm.out | tail -1

echo ""
echo "=== Done ==="
