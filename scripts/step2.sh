#!/usr/bin/env bash
# Step 2 for an idea that passed step 1 (DECISIONS.md, sections 29 and 30): every test-only set
# scored once per seed and for the seed average (runs/final_ledger.csv), then the paired
# comparison with the reference on every test set, and the metrics.
#   bash scripts/step2.sh skinnet-idea-<x> [skinnet-c6]
#   MF=1 EXTRA=derm1m_test bash scripts/step2.sh <idea> skinnet-c6   # also derm1m_test and MF alone (section 29)
# EXTRA is scored for the reference too, once (skipped if the ledger already lists it).
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/Scripts/python}
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
EXP=${1:?usage: step2.sh <idea> [reference]}
REF=${2:-skinnet-c6}
RUN=runs/$EXP; RREF=runs/$REF   # skincancer/paths.py
EXTRA=${EXTRA:-}
log() { echo "[$(date +%H:%M)] $*" >> runs/queue.log; }
EXT="ext_hiba ext_mednode ext_mcrsl ext_derm7pt ext_asan ext_isic2019"
FINAL="ddi_hold dermacon_test dermacon_other midas_test $EXT $EXTRA"
L=$RUN/logs/final.log
mkdir -p "$RUN/logs" "$RREF/logs"
log "$EXP step 2 start"
# A reference older than a test-only set (e.g. the base model and the outside sets) is scored on it
# first, once per seed and for its average, like any model (the ledger records the look).
MISSING=""
for sp in $FINAL; do [ -f "$RREF/avg/preds/$sp.csv" ] || MISSING="$MISSING $sp"; done
if [ -n "$MISSING" ]; then
  for s in 0 1 2; do
    $PY -m skincancer.predict --run $RREF/s$s --splits $MISSING --final >> $RREF/logs/final_missing.log 2>&1 || log "$REF predict s$s FAILED"
  done
  $PY -m skincancer.derive ensemble --exp $REF --splits $MISSING --final >> $RREF/logs/final_missing.log 2>&1 || log "$REF ensemble FAILED"
  log "$REF scored for the first time on$MISSING (the ledger records it)"
fi
for s in 0 1 2; do
  $PY -m skincancer.predict --run $RUN/s$s --splits $FINAL --final >> $L 2>&1 || log "$EXP predict s$s FAILED"
  if [ -n "$EXTRA" ] && ! grep -q "^$RREF/s$s,$EXTRA," runs/final_ledger.csv; then
    $PY -m skincancer.predict --run $RREF/s$s --splits $EXTRA --final >> $RREF/logs/final_extra.log 2>&1 || log "$REF predict s$s FAILED"
  fi
done
$PY -m skincancer.derive ensemble --exp $EXP --splits $FINAL --final >> $L 2>&1 || log "$EXP ensemble FAILED"
if [ -n "$EXTRA" ] && ! grep -q "^$RREF/avg,$EXTRA," runs/final_ledger.csv; then
  $PY -m skincancer.derive ensemble --exp $REF --splits $EXTRA --final >> $RREF/logs/final_extra.log 2>&1 || log "$REF ensemble FAILED"
fi
$PY -m skincancer.compare --a $REF --b $EXP --splits test midas_test ddi_hold dermacon_test $EXT $EXTRA \
  > $RUN/logs/compare_final.log 2>&1 || log "$EXP compare FAILED"
if [ -n "${MF:-}" ]; then
  $PY -m skincancer.compare --a $REF --b $EXP --splits ddi_hold test --positives "mycosis.fungoides" \
    > $RUN/logs/compare_final_mf.log 2>&1 || log "$EXP MF compare FAILED"
fi
$PY -m skincancer.evaluate --exp $EXP > $RUN/logs/evaluate.log 2>&1 || log "$EXP evaluate FAILED"
log "$EXP step 2 done"
