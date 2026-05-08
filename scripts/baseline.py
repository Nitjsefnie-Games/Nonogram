#!/usr/bin/env python3
"""Solve every puzzle in DIR and either save a (count, hash) baseline or
diff the current solver's output against a saved baseline. Used as a
correctness gate when changing the line DP kernel.

Usage:
  python3 scripts/baseline.py --save nonograms baseline.jsonl --exclude partially_solved
  python3 scripts/baseline.py        nonograms baseline.jsonl --exclude partially_solved
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from puzzle_io import load_clues, clues_valid
from search import solve


def hash_pic(pic):
    return hashlib.sha256(pic.pixels.tobytes()).hexdigest()[:16]


def solve_puzzle(path):
    rows, cols = load_clues(path)
    if not clues_valid(rows, cols):
        return None
    solutions = []
    for pic in solve(rows, cols):
        solutions.append(hash_pic(pic))
    solutions.sort()
    composite = hashlib.sha256("|".join(solutions).encode()).hexdigest()[:16]
    return {"n": len(solutions), "hash": composite}


def walk_puzzles(root, exclude):
    for dirpath, dirnames, filenames in os.walk(root):
        parts = Path(dirpath).parts
        if any(p in exclude for p in parts):
            dirnames[:] = []
            continue
        dirnames[:] = sorted(d for d in dirnames if d not in exclude)
        for fname in sorted(filenames):
            yield os.path.join(dirpath, fname)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir")
    ap.add_argument("baseline_file")
    ap.add_argument("--save", action="store_true",
                    help="Solve every puzzle and write the baseline (default mode is check)")
    ap.add_argument("--exclude", action="append", default=[], metavar="FOLDER",
                    help="Folder name to skip (exact match, any depth). Repeatable.")
    args = ap.parse_args()
    excludes = set(args.exclude)

    if args.save:
        n_written = 0
        with open(args.baseline_file, "w") as f:
            for path in walk_puzzles(args.dir, excludes):
                rel = os.path.relpath(path, args.dir)
                result = solve_puzzle(path)
                if result is None:
                    print(f"  SKIP {rel}: invalid clues", flush=True)
                    continue
                f.write(json.dumps({"id": rel, **result}) + "\n")
                f.flush()
                n_written += 1
                print(f"  {rel}: n={result['n']} hash={result['hash']}", flush=True)
        print(f"Saved {n_written} entries to {args.baseline_file}")
        return

    baseline = {}
    with open(args.baseline_file) as f:
        for line in f:
            obj = json.loads(line)
            baseline[obj["id"]] = obj

    mismatches = []
    n_checked = 0
    for path in walk_puzzles(args.dir, excludes):
        rel = os.path.relpath(path, args.dir)
        if rel not in baseline:
            continue
        current = solve_puzzle(path)
        if current is None:
            print(f"  SKIP {rel}: invalid clues now", flush=True)
            continue
        expected = baseline[rel]
        n_checked += 1
        if current["n"] != expected["n"] or current["hash"] != expected["hash"]:
            mismatches.append((rel, expected, current))
            print(f"  MISMATCH {rel}: expected n={expected['n']} hash={expected['hash']}; got n={current['n']} hash={current['hash']}", flush=True)
        else:
            print(f"  ok {rel}: n={current['n']}", flush=True)

    if mismatches:
        print(f"\n{len(mismatches)} mismatch(es) out of {n_checked} checked.")
        sys.exit(1)
    print(f"\nAll {n_checked} puzzles match baseline.")


if __name__ == "__main__":
    main()
