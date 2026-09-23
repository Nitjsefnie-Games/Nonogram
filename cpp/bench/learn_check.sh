#!/usr/bin/env bash
# Learned clauses are implied: run the stats build with learning on and
# LEARN_CHECK=1 over the small buckets. Each run checks every clause it learnt
# against every solution of the puzzle (a separate enumeration without
# learning; skipped past 10,000 solutions) and prints "learn-check: clause
# violated" for any clause some solution falsifies. Fails on any such line,
# on a run that dies (a stats check aborts), or on a run with no check line.
# Usage (from cpp/, after `make stats`): [SOLVER=...] bench/learn_check.sh [dir ...]
#   default dirs: ../nonograms/easy_small ../nonograms/trivial
set -u
SOLVER=${SOLVER:-./solver-stats}
[ -x "$SOLVER" ] || { echo "learn_check.sh: $SOLVER missing (make stats)" >&2; exit 2; }
if [ "$#" -eq 0 ]; then set -- ../nonograms/easy_small ../nonograms/trivial; fi

puzzles=0 checked=0 skipped=0 clauses=0 bad=0
out=$(mktemp)
trap 'rm -f "$out"' EXIT
for dir in "$@"; do
  for p in "$dir"/*; do
    [ -f "$p" ] || continue
    puzzles=$((puzzles + 1))
    if ! LEARN=1 LEARN_CHECK=1 "$SOLVER" "$p" > /dev/null 2> "$out"; then
      echo "FAIL $p: solver exited nonzero"; sed -n '1,5p' "$out"; bad=$((bad + 1)); continue
    fi
    if grep -q "clause violated" "$out"; then
      echo "FAIL $p:"; grep "clause violated" "$out"; bad=$((bad + 1)); continue
    fi
    line=$(grep "^learn-check: " "$out")
    case "$line" in
      *"clauses checked against"*)
        checked=$((checked + 1))
        clauses=$((clauses + $(echo "$line" | sed 's/^learn-check: \([0-9]*\) clauses.*/\1/'))) ;;
      *skipped*) skipped=$((skipped + 1)) ;;
      *) echo "FAIL $p: no learn-check line"; bad=$((bad + 1)) ;;
    esac
  done
done
echo "learn_check: $puzzles puzzles, $checked checked ($clauses learnt clauses), $skipped skipped (> 10000 solutions), $bad failed"
[ "$bad" -eq 0 ]
