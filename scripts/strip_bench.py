#!/usr/bin/env python3
"""Remove header blocks matching a given (solver, cpu) pair from puzzle files.

A "block" is a contiguous run of `#` lines starting with `# solver=...` and
ending at the next `# solver=...` or the first non-`#` line. A block matches
when its `# solver=` value equals --solver AND its `# cpu=` value equals --cpu.

Usage:
  python3 scripts/strip_bench.py nonograms/ \\
    --solver 1.0.0 --cpu "AMD EPYC 7H12 64-Core Processor" \\
    --exclude partially_solved
"""
import argparse
import os
from pathlib import Path


def parse_blocks(text):
    lines = text.splitlines(keepends=True)
    i = 0
    while i < len(lines) and lines[i].startswith("#"):
        i += 1
    header_lines = lines[:i]
    body = "".join(lines[i:])

    blocks = []
    current = []
    for line in header_lines:
        if line.startswith("# solver=") and current:
            blocks.append("".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        blocks.append("".join(current))
    return blocks, body


def block_matches(block, solver, cpu):
    has_solver = has_cpu = False
    for line in block.splitlines():
        if line.startswith("# solver="):
            if line[len("# solver="):].strip() == solver:
                has_solver = True
        elif line.startswith("# cpu="):
            if line[len("# cpu="):].strip() == cpu:
                has_cpu = True
    return has_solver and has_cpu


def strip_file(path, solver, cpu):
    with open(path) as f:
        text = f.read()
    blocks, body = parse_blocks(text)
    if not blocks:
        return False
    kept = [b for b in blocks if not block_matches(b, solver, cpu)]
    if len(kept) == len(blocks):
        return False
    with open(path, "w") as f:
        f.write("".join(kept) + body)
    return True


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("dir", help="Folder to walk recursively")
    ap.add_argument("--solver", required=True, help="Solver version to strip")
    ap.add_argument("--cpu", required=True, help="CPU model to strip")
    ap.add_argument("--exclude", action="append", default=[], metavar="FOLDER",
                    help="Folder name (exact, any depth) to skip. Repeatable.")
    args = ap.parse_args()
    excludes = set(args.exclude)

    stripped = scanned = 0
    for dirpath, dirnames, filenames in os.walk(args.dir):
        parts = Path(dirpath).parts
        if any(part in excludes for part in parts):
            dirnames[:] = []
            continue
        dirnames[:] = sorted(d for d in dirnames if d not in excludes)
        for fname in sorted(filenames):
            path = os.path.join(dirpath, fname)
            scanned += 1
            if strip_file(path, args.solver, args.cpu):
                stripped += 1

    print(f"Scanned {scanned} files, stripped a matching block from {stripped}.")


if __name__ == "__main__":
    main()
