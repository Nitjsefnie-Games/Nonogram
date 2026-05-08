#!/bin/bash
# Create an exclusive cpuset partition for the solver.
# Usage: sudo ./scripts/cpuset_setup.sh [core]   (default core = highest)
set -euo pipefail

CG=/sys/fs/cgroup/nonogram-solver

if [[ $EUID -ne 0 ]]; then
    echo "Must run as root (writes to /sys/fs/cgroup/)" >&2
    exit 1
fi

if [[ ! -d /sys/fs/cgroup/cpuset.cpus.effective ]] && [[ ! -e /sys/fs/cgroup/cpuset.cpus.effective ]]; then
    if ! grep -q cpuset /sys/fs/cgroup/cgroup.controllers 2>/dev/null; then
        echo "cpuset controller not available in cgroup v2 root" >&2
        exit 1
    fi
fi

if [[ $# -ge 1 ]]; then
    CORE=$1
else
    LAST=$(tr ',' '\n' < /sys/fs/cgroup/cpuset.cpus.effective | tail -1)
    CORE=${LAST##*-}
fi

if [[ -d $CG ]]; then
    echo "Cgroup $CG already exists. Tearing down first."
    "$(dirname "$0")/cpuset_teardown.sh"
fi

mkdir $CG
echo "$CORE" > $CG/cpuset.cpus
echo root > $CG/cpuset.cpus.partition

echo "Exclusive partition ready:"
echo "  $CG/cpuset.cpus.effective = $(cat $CG/cpuset.cpus.effective)"
echo "  $CG/cpuset.cpus.partition = $(cat $CG/cpuset.cpus.partition)"
echo "  /sys/fs/cgroup/cpuset.cpus.effective (root, others) = $(cat /sys/fs/cgroup/cpuset.cpus.effective)"
echo
echo "Run categorize.py normally; it will auto-join $CG."
