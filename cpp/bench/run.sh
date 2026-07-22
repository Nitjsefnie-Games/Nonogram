#!/usr/bin/env bash
# Run a command on the shielded core inside the bench cgroup.
# The core is read back from the cgroup itself, so this tracks whatever
# shield.sh reserved instead of assuming a fixed core.
# Usage: bench/run.sh <cmd> [args...]
set -u
CG=/sys/fs/cgroup/bench
CORE=$(cat "$CG/cpuset.cpus" 2>/dev/null) || {
  echo "run.sh: $CG missing — run bench/shield.sh first" >&2
  exit 1
}
exec bash -c '
  echo $BASHPID > '"$CG"'/cgroup.procs 2>/dev/null
  exec taskset -c '"$CORE"' nice -n -20 "$@"
' _ "$@"
