#!/usr/bin/env python3
"""Sum puzzle solve_time headers under a folder, grouped by (solver, cpu).

Walks DIR recursively, parses every `# key=value` header block in each file
(blocks delimited by `# solver=` lines), and aggregates `solve_time` per
(solver, cpu) pair.

Usage:
  python3 scripts/sum_times.py nonograms/
  python3 scripts/sum_times.py nonograms/ --solver 1.0.0
  python3 scripts/sum_times.py nonograms/ --solver 1.0.0 --cpu "AMD EPYC 7H12 64-Core Processor"
"""
import argparse
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass

HEADER = re.compile(r"^# (\w+)=(.*)$")


@dataclass
class Aggregate:
    sum: float = 0.0
    n: int = 0
    max: float = 0.0
    max_path: str = ""
    min: float = float("inf")


def parse_blocks(path):
    """Yield dicts of header keys/values for each block in path."""
    blocks = []
    current = None
    try:
        with open(path) as f:
            for line in f:
                line = line.rstrip("\n")
                if not line.startswith("#"):
                    break
                m = HEADER.match(line)
                if not m:
                    continue
                key, value = m.group(1), m.group(2).strip()
                if key == "solver":
                    if current:
                        blocks.append(current)
                    current = {"solver": value}
                elif current is not None:
                    current[key] = value
    except OSError as exc:
        print(f"warning: {path}: {exc}", file=sys.stderr)
        return []
    if current:
        blocks.append(current)
    return blocks


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", help="Folder to scan recursively")
    ap.add_argument("--solver", help="Only count blocks matching this solver version")
    ap.add_argument("--cpu", help="Only count blocks matching this CPU model")
    args = ap.parse_args()

    sums: defaultdict[tuple[str, str], Aggregate] = defaultdict(Aggregate)
    files_seen = 0
    for dirpath, _, filenames in os.walk(args.dir):
        for fname in filenames:
            files_seen += 1
            path = os.path.join(dirpath, fname)
            for block in parse_blocks(path):
                if "solve_time" not in block or "solver" not in block or "cpu" not in block:
                    continue
                if args.solver and block["solver"] != args.solver:
                    continue
                if args.cpu and block["cpu"] != args.cpu:
                    continue
                try:
                    t = float(block["solve_time"])
                except ValueError:
                    continue
                key = (block["solver"], block["cpu"])
                agg = sums[key]
                agg.sum += t
                agg.n += 1
                if t > agg.max:
                    agg.max = t
                    agg.max_path = path
                if t < agg.min:
                    agg.min = t

    if not sums:
        print(f"No matching headers found ({files_seen} files scanned).", file=sys.stderr)
        sys.exit(1)

    rows = sorted(sums.items(), key=lambda kv: kv[1].sum / kv[1].n)
    w_solver = max(len(s) for (s, _), _ in rows)
    w_cpu = max(len(c) for (_, c), _ in rows)

    print(f"{'solver':<{w_solver}}  {'cpu':<{w_cpu}}    n       sum         mean        min          max  (max file)")
    print("-" * (w_solver + w_cpu + 70))
    for (solver, cpu), agg in rows:
        mean = agg.sum / agg.n
        print(f"{solver:<{w_solver}}  {cpu:<{w_cpu}}  {agg.n:>4}  {agg.sum:>10.3f}s  {mean:>9.4f}s  {agg.min:>9.4f}s  {agg.max:>10.3f}s  {agg.max_path}")
    print(f"\n({files_seen} files scanned)")


if __name__ == "__main__":
    main()
