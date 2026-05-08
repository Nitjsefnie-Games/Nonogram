#!/bin/bash
# Dismantle the solver's exclusive cpuset partition.
# Usage: sudo ./scripts/cpuset_teardown.sh
set -euo pipefail

CG=/sys/fs/cgroup/nonogram-solver

if [[ $EUID -ne 0 ]]; then
    echo "Must run as root" >&2
    exit 1
fi

if [[ ! -d $CG ]]; then
    echo "(no cgroup at $CG; nothing to do)"
    exit 0
fi

# evict any remaining tasks back to the root cgroup
if [[ -s $CG/cgroup.procs ]]; then
    while read -r pid; do
        [[ -n $pid ]] && echo "$pid" > /sys/fs/cgroup/cgroup.procs 2>/dev/null || true
    done < $CG/cgroup.procs
fi

echo member > $CG/cpuset.cpus.partition
rmdir $CG

echo "Torn down. Root effective cpus = $(cat /sys/fs/cgroup/cpuset.cpus.effective)"
