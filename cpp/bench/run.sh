#!/usr/bin/env bash
# Run a command on the shielded core 5 inside the bench cgroup.
# Usage: bench/run.sh <cmd> [args...]
set -u
CG=/sys/fs/cgroup/bench
exec bash -c '
  echo $BASHPID > '"$CG"'/cgroup.procs 2>/dev/null
  exec taskset -c 5 nice -n -20 "$@"
' _ "$@"
