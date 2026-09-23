#!/usr/bin/env python3
"""Speed diff of the working tree's corpus headers against HEAD, for the
commit message of a rebench: one `git diff -U0 -- nonograms` parse (never
one `git show` per file), pairing `-# solve_time=` with `+# solve_time=`
per file. Prints the paired total and the largest gains and losses.

Usage: speeddiff.py [N]   (N = movers to list, default 12); run from the repo root.
"""
import subprocess
import sys

diff = subprocess.run(["git", "diff", "-U0", "--", "nonograms"], capture_output=True, text=True, check=True).stdout
old, new, cur = {}, {}, None
for line in diff.splitlines():
    if line.startswith("+++ b/"):
        cur = line[6:]
    elif cur and line.startswith("-# solve_time="):
        old[cur] = float(line.split("=", 1)[1])
    elif cur and line.startswith("+# solve_time="):
        new[cur] = float(line.split("=", 1)[1])

paired = sorted((f for f in new if f in old), key=lambda f: old[f] - new[f], reverse=True)
so, sn = sum(old[f] for f in paired), sum(new[f] for f in paired)
print(f"{len(paired)} re-headed files: {so:.1f} -> {sn:.1f} s ({(sn / so - 1) * 100:+.1f}%)" if so else "no paired files")
n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
print("largest gains:")
for f in paired[:n]:
    print(f"  {f}: {old[f]:.4g} -> {new[f]:.4g} s")
print("largest losses:")
for f in paired[-n:][::-1]:
    if new[f] > old[f]:
        print(f"  {f}: {old[f]:.4g} -> {new[f]:.4g} s")
