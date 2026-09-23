#!/usr/bin/env python3
"""Branch-node and probe counts of the stats build over the whole corpus.

Wall time is noisy and the two corpus-dominating puzzles run with probing
shut off, so a change to the branch heuristic is judged here instead: the
`solver-stats` binary (make stats) prints `branch-nodes=` and `probes=` with
DEBUG_CACHE_STATS=1, and those counts are deterministic. Every puzzle (minus
--exclude categories) is solved to full enumeration; solution count and
strategy are checked against the golden header like bench/corpus.py.

Usage:
  bench/nodes.py ./solver-stats -o bench/results/<tag>-nodes.jsonl [--env K=V ...]
  bench/nodes.py --compare a.jsonl b.jsonl [--top 15]

--env passes extra environment variables to the solver (the experimental
branch-score switches live behind such variables in the stats build).
"""
import argparse
import json
import os
import re
import subprocess
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))      # cpp/
NONO = os.path.join(os.path.dirname(ROOT), "nonograms")
# Folders without a finished count (no golden header) and the fully counted
# puzzles that take minutes each; --exclude adds to these, it does not
# replace them (a run that replaced them walked into in_progress and sat
# on an unfinishable puzzle for an hour).
DEFAULT_EXCLUDE = ("partially_solved", "in_progress", "extreme", "insane")


def golden(path):
    n = strat = None
    with open(path) as f:
        for line in f:
            if not line.startswith("#"):
                break
            m = re.match(r"#\s*n_solutions=(\d+)", line)
            if m: n = int(m.group(1))
            m = re.match(r"#\s*strategy=(\w+)", line)
            if m: strat = m.group(1)
    return n, strat


def parse(out, err):
    n = strat = None
    m = re.search(r"Found ([\d,]+) solution", out)
    if m: n = int(m.group(1).replace(",", ""))
    m = re.search(r"Strategy: (\w+)", out)
    if m: strat = m.group(1)
    nodes = probes = None
    m = re.search(r"branch-nodes=(\d+)", err)
    if m: nodes = int(m.group(1))
    m = re.search(r"probes=(\d+)", err)
    if m: probes = int(m.group(1))
    return n, strat, nodes, probes


def puzzles(exclude):
    out = []
    for cat in sorted(os.listdir(NONO)):
        d = os.path.join(NONO, cat)
        if not os.path.isdir(d) or cat in exclude:
            continue
        for name in sorted(os.listdir(d), key=lambda s: (len(s), s)):
            out.append((cat, name, os.path.join(d, name)))
    return out


def run(binary, out_path, exclude, timeout, extra_env):
    env = dict(os.environ)
    env["DEBUG_CACHE_STATS"] = "1"
    env.update(extra_env)
    items = puzzles(exclude)
    print(f"{len(items)} puzzles, binary {binary}, env {extra_env}")
    t0 = time.time()
    rows = []
    mismatches = 0
    with open(out_path, "w") as fo:
        for i, (cat, name, path) in enumerate(items, 1):
            try:
                r = subprocess.run([binary, path], capture_output=True, text=True,
                                   timeout=timeout, env=env)
                n, strat, nodes, probes = parse(r.stdout, r.stderr)
                status = "ok"
            except subprocess.TimeoutExpired:
                n = strat = nodes = probes = None
                status = "timeout"
            gn, gs = golden(path)
            ok = status == "ok" and (gn is None or gn == n) and (gs is None or gs == strat)
            if not ok:
                mismatches += 1
                print(f"  MISMATCH {cat}/{name}: got n={n} strat={strat} ({status}), golden n={gn} strat={gs}")
            row = {"puzzle": f"{cat}/{name}", "nodes": nodes, "probes": probes,
                   "n": n, "strategy": strat, "status": status, "ok": ok}
            rows.append(row)
            fo.write(json.dumps(row) + "\n")
            if i % 250 == 0:
                print(f"  ...{i}/{len(items)} ({time.time() - t0:.0f}s)")
    tn = sum(r["nodes"] or 0 for r in rows)
    tp = sum(r["probes"] or 0 for r in rows)
    print(f"TOTAL nodes={tn} probes={tp} mismatches={mismatches} wall {time.time() - t0:.0f}s")


def load(path):
    with open(path) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    return {r["puzzle"]: r for r in rows}


def compare(a_path, b_path, top):
    a, b = load(a_path), load(b_path)
    common = [p for p in a if p in b]
    ta = sum(a[p]["nodes"] or 0 for p in common)
    tb = sum(b[p]["nodes"] or 0 for p in common)
    pa = sum(a[p]["probes"] or 0 for p in common)
    pb = sum(b[p]["probes"] or 0 for p in common)
    changed = [(p, a[p]["nodes"] or 0, b[p]["nodes"] or 0) for p in common
               if (a[p]["nodes"] or 0) != (b[p]["nodes"] or 0)]
    better = sum(1 for _, x, y in changed if y < x)
    worse = sum(1 for _, x, y in changed if y > x)
    print(f"{len(common)} puzzles: nodes {ta} -> {tb} ({(tb - ta) / ta * 100:+.1f}%), "
          f"probes {pa} -> {pb} ({(pb - pa) / pa * 100:+.1f}%)")
    print(f"changed {len(changed)}: {better} fewer nodes, {worse} more")
    changed.sort(key=lambda t: -abs(t[2] - t[1]))
    for p, x, y in changed[:top]:
        print(f"  {p:28s} {x:>10d} -> {y:>10d} ({y - x:+d})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("binary", nargs="?")
    ap.add_argument("-o", "--out")
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--exclude", default="", help="extra categories to skip, comma-separated, on top of " + ",".join(DEFAULT_EXCLUDE) + "; --include-all runs everything")
    ap.add_argument("--include-all", action="store_true", help="run every category, the defaults included")
    ap.add_argument("--env", action="append", default=[], metavar="K=V")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()
    if args.compare:
        compare(args.compare[0], args.compare[1], args.top)
        return
    if not args.binary or not args.out:
        ap.error("binary and -o are required unless --compare is given")
    extra = dict(kv.split("=", 1) for kv in args.env)
    exclude = tuple(c for c in args.exclude.split(",") if c)
    if not args.include_all:
        exclude = DEFAULT_EXCLUDE + exclude
    run(args.binary, args.out, exclude, args.timeout, extra)


if __name__ == "__main__":
    main()
