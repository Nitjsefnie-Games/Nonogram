#!/usr/bin/env bash
# Reverse shield.sh: dissolve the bench partition and give every cgroup all cores.
set -u
CG=/sys/fs/cgroup
# Every online core, e.g. "0-11". Was hardcoded 0-5, which on a 12-core box
# confined every top-level cgroup and IRQ to half the machine on "unshield".
ALL=$(cat /sys/devices/system/cpu/online)

if [ -d "$CG/bench" ]; then
  echo member > "$CG/bench/cpuset.cpus.partition" 2>/dev/null
  # move any stragglers back to root before removing
  if [ -f "$CG/bench/cgroup.procs" ]; then
    while read -r pid; do echo "$pid" > "$CG/cgroup.procs" 2>/dev/null; done < "$CG/bench/cgroup.procs"
  fi
  rmdir "$CG/bench" 2>/dev/null && echo "[unshield] removed bench cgroup"
fi

echo "[unshield] restoring top-level cgroups to ${ALL}"
for d in "$CG"/*/; do
  [ -f "$d/cpuset.cpus" ] || continue
  echo "$ALL" > "$d/cpuset.cpus" 2>/dev/null
  # empty string = inherit; prefer clearing so systemd manages it
  : > "$d/cpuset.cpus" 2>/dev/null || true
done

echo "[unshield] restoring IRQ affinity to ${ALL}"
for f in /proc/irq/*/smp_affinity_list; do
  [ -w "$f" ] || continue
  echo "$ALL" > "$f" 2>/dev/null
done
echo "[unshield] done"
