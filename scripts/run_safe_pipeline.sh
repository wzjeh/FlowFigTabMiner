#!/bin/bash
# Master pipeline script to run steps 1-4 safely avoiding Torch/Paddle conflicts

# Path to venv python
PYTHON_EXEC="./flowfigtabminer/bin/python3"
INPUT_PDF="data/input/example.pdf"

if [ ! -f "$PYTHON_EXEC" ]; then
    echo "Error: Virtual environment python not found at $PYTHON_EXEC"
    exit 1
fi

echo "=================================================="
echo " Starting Safe Pipeline for $INPUT_PDF"
echo "=================================================="

# 1. Run Step 1 (TF-ID Detection) - Uses Transformers/Torch
echo "[Master] Running Step 1: TF-ID Detection..."
$PYTHON_EXEC scripts/step1_tfid.py "$INPUT_PDF"
if [ $? -ne 0 ]; then
    echo "[Master] Step 1 Failed. Exiting."
    exit 1
fi

echo "--------------------------------------------------"
echo "[Master] Step 1 Complete. Sleeping 2s to clear memory..."
sleep 2
echo "--------------------------------------------------"

# 2. Run Steps 2-4 (Extraction & Assembly) - Uses YOLO/Paddle
echo "[Master] Running Steps 2-4: Extraction & Assembly..."
$PYTHON_EXEC scripts/run_steps2_to_4.py "$INPUT_PDF"
if [ $? -ne 0 ]; then
    echo "[Master] Steps 2-4 Failed. Exiting."
    exit 1
fi

echo "=================================================="
echo " Pipeline Complete Successfully!"
echo "=================================================="
