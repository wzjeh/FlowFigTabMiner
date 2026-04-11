#!/bin/bash
# Run full FlowFigTabMiner pipeline on 6 unprocessed main papers
# for data/ml_lifetime clean dataset construction

set -e
cd /Users/zhaowenyuan/Projects/FlowFigTabMiner
source flowfigtabminer/bin/activate

PAPERS_DIR="data/ml_lifetime/papers"

PAPERS=(
    "CHLiIF-carbenoid_2020_Nagaki.pdf"
    "Nagaki et al. 2007 - Integrated micro flow synthesis based on sequential Br-Li exchange reactions of p-, m-, and o-dibromobenzenes.pdf"
    "Nagaki et al. 2009 - Generation and reactions of α-silyloxiranyllithium in a microreactor.pdf"
    "Nagaki et al. 2009 - Generations and reactions ofN-(t-butylsulfonyl)aziridinyllithiums using microreactors.pdf"
    "Nagaki et al. 2009 - Nitro-substituted aryl lithium compounds in microreactor synthesis - switch between kinetic and thermodynamic control.pdf"
    "Nagaki et al. 2009 - Synthesis of unsymmetrically substituted biaryls via sequential lithiation of dibromobiaryls using integrated microflow systems.pdf"
)

echo "=== Running full pipeline on ${#PAPERS[@]} papers ==="
echo ""

for i in "${!PAPERS[@]}"; do
    PDF="${PAPERS_DIR}/${PAPERS[$i]}"
    echo "[$((i+1))/${#PAPERS[@]}] ${PAPERS[$i]}"
    if [ ! -f "$PDF" ]; then
        echo "  ERROR: PDF not found!"
        continue
    fi
    python src/pipeline/main.py "$PDF" 2>&1 | tail -20
    echo ""
    echo "---"
    echo ""
done

echo "=== All done ==="
