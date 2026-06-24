#!/usr/bin/env bash
# Shield CPU core 5 for benchmarking. Reversible via unshield.sh.
# Restricts every top-level cgroup to cores 0-4, steers movable IRQs off 5,
# then makes /sys/fs/cgroup/bench an exclusive partition owning core 5.
set -u
CG=/sys/fs/cgroup
CORE=5
OTHERS=0-4

echo "[shield] restricting top-level cgroups to ${OTHERS}"
for d in "$CG"/*/; do
  name=$(basename "$d")
  [ "$name" = "bench" ] && continue
  [ -f "$d/cpuset.cpus" ] || continue
  echo "$OTHERS" > "$d/cpuset.cpus" 2>/dev/null \
    && echo "  ok   $name" || echo "  skip $name"
done

echo "[shield] steering movable IRQs off core ${CORE}"
moved=0
for f in /proc/irq/*/smp_affinity_list; do
  [ -w "$f" ] || continue
  cur=$(cat "$f" 2>/dev/null) || continue
  case ",$cur," in *",$CORE,"*|*"-"*) ;; esac
  echo "$OTHERS" > "$f" 2>/dev/null && moved=$((moved+1))
done
echo "  steered $moved IRQs (managed/per-cpu IRQs refuse and are left as-is)"

echo "[shield] forming exclusive partition for core ${CORE}"
mkdir -p "$CG/bench"
echo 0 > "$CG/bench/cpuset.mems"
echo "$CORE" > "$CG/bench/cpuset.cpus"
if echo isolated > "$CG/bench/cpuset.cpus.partition" 2>/dev/null; then
  echo "  partition: isolated"
elif echo root > "$CG/bench/cpuset.cpus.partition" 2>/dev/null; then
  echo "  partition: root"
else
  echo "  partition: member (exclusive via sibling restriction only)"
fi
echo "[shield] state: $(cat "$CG/bench/cpuset.cpus.partition") cpus=$(cat "$CG/bench/cpuset.cpus.effective")"
echo "[shield] threads still on core ${CORE}:"
ps -eLo psr,comm | awk -v c="$CORE" '$1==c {print "  "$2}' | sort | uniq -c
