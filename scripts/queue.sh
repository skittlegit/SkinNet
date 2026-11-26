#!/usr/bin/env bash
# The training queue: ideas one after another on the one GPU. Each goes through step 1 (train
# its seeds, average them, compare on the validation sets), the step 1 rule, and step 2 if it
# passed (DECISIONS.md). The queue runs detached from the terminal, so it survives closing it,
# logs to runs/queue.log and skips finished seeds: after a stop, the same command carries on,
# and a stopped seed resumes from its last.pt. Nothing is deleted or adopted; that is the user's call.
#
#   bash scripts/queue.sh start z aa            # <model>-idea-z, then -aa, against the current checkpoint
#   bash scripts/queue.sh start z@skinnet-c4    # against another reference
#   bash scripts/queue.sh start z:guard         # another step 1 rule: gain (default) | guard | unseen
#   bash scripts/queue.sh start --wait a        # start once the other repository's queue has finished
#   bash scripts/queue.sh status                # running or not, which seed and epoch, time left, last results
#   bash scripts/queue.sh stop                  # stop this repository's queue and its training now
# An item is letter[@reference][:rule][/seeds]. With /seeds (e.g. a/0) only those seeds train: a first
# look, with no step 1 verdict and no test sets, which need all three.
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)
PY=${PY:-.venv/Scripts/python}
PREFIX=$([ -f configs/panderm.yaml ] && echo panderm || echo skinnet)
DEFAULT_REF=$(sed -n 's/^REF=\${REF:-\(.*\)}$/\1/p' scripts/run_idea.sh)
LOG=runs/queue.log
PIDF=runs/queue.pid
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
log() { echo "[$(date +%H:%M)] $*" >> "$LOG"; }

queue_pids() {   # live queue runners in this and the neighbouring repositories (../*/runs/queue.pid)
  local f q
  for f in "$ROOT"/../*/runs/queue.pid; do
    [ -f "$f" ] && q=$(cat "$f") && kill -0 "$q" 2>/dev/null && echo "$q"
  done
}

mine() {         # this repository's runner, if it is alive
  [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null && cat "$PIDF"
}

parse() {        # item -> "letter reference rule seeds"
  local l=$1 rule=gain ref=$DEFAULT_REF seeds=all
  case $l in */*) seeds=${l##*/}; l=${l%/*};; esac
  case $l in *:*) rule=${l##*:}; l=${l%:*};; esac
  case $l in *@*) ref=${l#*@}; l=${l%@*};; esac
  echo "$l $ref $rule $seeds"
}

check_items() {
  local bad=0 l ref rule seeds
  for item in "$@"; do
    read -r l ref rule seeds <<< "$(parse "$item")"
    [[ $seeds == all || $seeds =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo "seeds $seeds: give them as 0 or 0,1"; bad=1; }
    [ -f "configs/$PREFIX-idea-$l.yaml" ] || { echo "no configs/$PREFIX-idea-$l.yaml"; bad=1; }
    [ -d "runs/$ref/avg" ] || { echo "reference $ref has no runs/$ref/avg"; bad=1; }
    case $rule in gain|guard|unseen) ;; *) echo "unknown rule $rule (gain | guard | unseen)"; bad=1;; esac
  done
  return $bad
}

start() {
  local wait=""
  [ "${1:-}" = "--wait" ] && { wait=--wait; shift; }
  [ $# -gt 0 ] || { echo "usage: queue.sh start [--wait] <letter>[@reference][:rule] ..."; exit 2; }
  check_items "$@" || exit 2
  [ -n "$(mine)" ] && { echo "this repository's queue is already running (pid $(mine)); stop it first"; exit 1; }
  if [ -z "$wait" ] && [ -n "$(queue_pids)" ]; then
    echo "another queue is running (pid $(queue_pids | tr '\n' ' ')): pass --wait to start after it"; exit 1
  fi
  mkdir -p runs
  local before
  before=$(wc -l < "$LOG" 2>/dev/null || echo 0)
  printf 'cd %q && PY=%q exec bash scripts/queue.sh run %s %s\n' "$ROOT" "$PY" "$wait" "$*" > runs/queue_launch.sh
  local gitbash
  gitbash=$(cygpath -w /)\\bin\\bash.exe
  powershell.exe -NoProfile -Command "Start-Process -WindowStyle Hidden -FilePath '$gitbash' -ArgumentList '$(cygpath -m "$ROOT")/runs/queue_launch.sh'"
  for _ in 1 2 3 4 5 6 7 8 9 10; do [ -n "$(mine)" ] && break; sleep 1; done
  if [ -n "$(mine)" ]; then echo "queue started (pid $(mine)): $wait $*; log in $LOG"
  elif [ -n "$(tail -n +$((before + 1)) "$LOG" 2>/dev/null | grep "queue done")" ]; then
    echo "the queue ran and has already finished (nothing left to train?):"; tail -n +$((before + 1)) "$LOG"
  else echo "the queue did not start: see $LOG"; exit 1; fi
}

run() {
  echo $$ > "$PIDF"
  trap 'rm -f "$PIDF"' EXIT
  if [ "${1:-}" = "--wait" ]; then
    shift
    log "queue waiting for the other queue to finish: $*"
    # (no grep -q in a pipe here: with pipefail, an early exit would end the wait while the other runs)
    while [ -n "$(queue_pids | grep -vx $$)" ]; do sleep 60; done
  fi
  log "queue start: $*"
  local l ref rule seeds e v rc s
  for item in "$@"; do
    read -r l ref rule seeds <<< "$(parse "$item")"
    e=$PREFIX-idea-$l
    if [ "$seeds" != all ]; then   # a first look: these seeds only
      mkdir -p "runs/$e/logs"
      for s in ${seeds//,/ }; do
        if [ -f "runs/$e/s$s/summary.json" ]; then log "$e seed $s already done"; continue; fi
        log "train $e seed $s (seeds $seeds only: no step 1 verdict, no test sets)"
        if $PY -m skincancer.train --config "configs/$e.yaml" --seed "$s" > "runs/$e/logs/s$s.log" 2>&1; then
          log "$e seed $s done: $(tail -n 1 "runs/$e/logs/s$s.log")"
        else log "$e seed $s FAILED: see runs/$e/logs/s$s.log"; fi
      done
      continue
    fi
    log "$e start (against $ref, step 1 rule: $rule)"
    if REF=$ref PY=$PY bash scripts/run_idea.sh "$l" >> "$LOG" 2>&1; then
      v=$($PY scripts/step1_gate.py "runs/$e/logs/compare_val.log" "$rule" 2>&1); rc=$?
      log "$e step 1 $v"
      if [ $rc -eq 0 ]; then
        PY=$PY bash scripts/step2.sh "$e" "$ref" && log "$e step 2 written; keeping it is the user's call"
      fi
    else log "$e FAILED"; fi
  done
  log "queue done"
}

stop() {
  local q
  q=$(mine) || true
  [ -n "$q" ] || { echo "this repository's queue is not running"; return 0; }
  # Everything the queue started shares its process group: the shell scripts, python and, under
  # python, its data-loading workers. taskkill /T ends a Windows process with all its children.
  local wins
  wins=$(ps | awk -v g="$q" 'NR > 1 && $3 == g {print $4}')
  kill "$q" 2>/dev/null
  for w in $wins; do taskkill //PID "$w" //T //F > /dev/null 2>&1; done
  rm -f "$PIDF"
  log "queue stopped by hand; the same start command carries on (a stopped seed resumes from last.pt)"
  echo "queue stopped (pid $q)"
}

case ${1:-} in
  start) shift; start "$@";;
  run) shift; run "$@";;
  stop) stop;;
  status) if q=$(mine); then echo "queue running (pid $q)"; $PY scripts/status.py
          else echo "queue not running$([ -n "$(queue_pids)" ] && echo "; the other repository's queue is")"
               $PY scripts/status.py stopped; fi;;
  *) sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'; exit 2;;
esac
