#!/usr/bin/env python3
"""Convert a puzzle from this repo's format to Steve Simpson's .non format,
which pbnsolve and other survey solvers read. Usage: bench/to_non.py puzzle > out.non"""
import sys

def load(path):
    rows, cols = [], []
    cur = rows
    for line in open(path):
        line = line.rstrip("\n")
        if line.startswith("#"):
            continue
        if line.strip() == "---":
            cur = cols
            continue
        cur.append(line.strip())
    return rows, cols

def to_non(rows, cols):
    out = [f"width {len(cols)}", f"height {len(rows)}", "", "rows"]
    out += [(",".join(r.split()) if r else "0") for r in rows]
    out += ["", "columns"]
    out += [(",".join(c.split()) if c else "0") for c in cols]
    return "\n".join(out) + "\n"

if __name__ == "__main__":
    sys.stdout.write(to_non(*load(sys.argv[1])))
