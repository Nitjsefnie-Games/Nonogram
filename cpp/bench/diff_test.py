#!/usr/bin/env python3
"""Differential test: run two solver binaries over many puzzles and assert
identical (solution_count, strategy). Used to validate behavior-preserving
optimizations against a reference binary.

Usage: diff_test.py <ref_binary> <new_binary> [max_seconds_per_puzzle]
"""
import os, re, subprocess, sys, glob, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NONO = os.path.join(os.path.dirname(ROOT), "nonograms")
CG = "/sys/fs/cgroup/bench"

def parse(out):
    n = strat = None
    m = re.search(r"Found ([\d,]+) solution", out)
    if m: n = int(m.group(1).replace(",", ""))
    m = re.search(r"Strategy: (\w+)", out)
    if m: strat = m.group(1)
    return n, strat

def run(binary, path, timeout):
    cmd = ["bash", "-c",
           f'echo $BASHPID > {CG}/cgroup.procs 2>/dev/null; exec taskset -c 5 "$0" "$1"',
           binary, path]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return ("TIMEOUT", None)
    return parse(r.stdout)

def collect(timeout):
    paths = []
    # fast, broad coverage
    for cat in ("trivial", "easy_small", "easy_medium", "easy_large", "medium"):
        for f in sorted(glob.glob(os.path.join(NONO, cat, "*"))):
            paths.append(os.path.relpath(f, NONO))
    return paths

def main():
    ref, new = sys.argv[1], sys.argv[2]
    timeout = float(sys.argv[3]) if len(sys.argv) > 3 else 15.0
    paths = collect(timeout)
    print(f"comparing {len(paths)} puzzles, timeout {timeout}s each")
    mism = 0
    checked = 0
    skipped = 0
    t0 = time.time()
    for i, p in enumerate(paths):
        rn = run(ref, os.path.join(NONO, p), timeout)
        if rn[0] == "TIMEOUT":
            skipped += 1
            continue
        nn = run(new, os.path.join(NONO, p), timeout)
        checked += 1
        if rn != nn:
            mism += 1
            print(f"  MISMATCH {p}: ref={rn} new={nn}")
            if mism >= 20:
                print("  too many mismatches, aborting"); break
        if (i + 1) % 200 == 0:
            print(f"  ...{i+1}/{len(paths)} checked={checked} mism={mism} ({time.time()-t0:.0f}s)")
    print(f"DONE checked={checked} skipped(timeout)={skipped} mismatches={mism}")
    sys.exit(1 if mism else 0)

if __name__ == "__main__":
    main()
