#!/usr/bin/env bash
# Step 1 for an idea: train its seeds, average them, and compare with the current checkpoint
# on the validation sets only. Test-only sets are never touched here (scripts/run.sh does that,
# once an idea has passed step 1).
#   PY=.venv/Scripts/python bash scripts/run_idea.sh n            # configs/skinnet/skinnet-idea-n.yaml
#   REF=skinnet-c3 SEEDS="0 1 2" PY=python bash scripts/run_idea.sh m   # against an older checkpoint
# Safe to re-run: seeds with a finished summary.json are skipped.
set -uo pipefail
PY=${PY:-python}
LETTER=${1:?usage: run_idea.sh <letter>}
EXP=${EXP:-skinnet-idea-$LETTER}   # EXP=panderm-idea-a: a PanDerm-track idea (configs/panderm/panderm-idea-a.yaml), with REF=panderm-c1
case $EXP in panderm*) CFG=configs/panderm/$EXP.yaml ;; *) CFG=configs/skinnet/$EXP.yaml ;; esac
REF=${REF:-skinnet-c4}
SEEDS=${SEEDS:-"0 1 2"}
VAL=${VAL:-"val midas_val"}   # VAL="val midas_val ddi_val": DDI validation in step 1 (section 26)
LOGS=runs/$EXP/logs
mkdir -p "$LOGS"
export PYTHONUNBUFFERED=1

for s in $SEEDS; do
  if [ -f "runs/$EXP/s$s/summary.json" ]; then echo "[$(date +%H:%M)] $EXP seed $s already done"; continue; fi
  echo "[$(date +%H:%M)] train $EXP seed $s"
  $PY -m skincancer.train --config "$CFG" --seed "$s" > "$LOGS/s$s.log" 2>&1 \
    || { echo "  FAILED: see $LOGS/s$s.log"; exit 1; }
  echo "[$(date +%H:%M)] $EXP seed $s done: $(tail -n 1 "$LOGS/s$s.log")"
done
$PY -m skincancer.derive ensemble --exp "$EXP" --splits test ak_side $VAL > "$LOGS/derive.log" 2>&1
$PY -m skincancer.compare --a "$REF" --b "$EXP" --splits $VAL | tee "$LOGS/compare_val.log"
echo "[$(date +%H:%M)] $EXP step 1 done"
