#!/usr/bin/env python3
"""Time the C++ solver over the whole corpus on the shielded core.

Every puzzle (minus --exclude categories) is solved to full enumeration once,
on the core bench/shield.sh reserved, with a per-puzzle timeout. Each result
is checked against the golden `# n_solutions=` / `# strategy=` header, so
the run is also a corpus-wide correctness gate. Results go to a JSONL file
and a per-category summary is printed; --compare diffs against an earlier
results file.

Usage:
  bench/corpus.py ./solver -o bench/results/<tag>.jsonl [--timeout 900]
  bench/corpus.py --compare bench/results/a.jsonl bench/results/b.jsonl
"""
import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))      # cpp/
NONO = os.path.join(os.path.dirname(ROOT), "nonograms")
CG = "/sys/fs/cgroup/bench"
DEFAULT_EXCLUDE = ("partially_solved", "in_progress")


def shielded_core():
    try:
        with open(os.path.join(CG, "cpuset.cpus")) as f:
            return f.read().strip()
    except OSError:
        sys.exit(f"{CG} missing -- run bench/shield.sh first")


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


def parse(out):
    n = strat = t = None
    m = re.search(r"Found ([\d,]+) solution", out)
    if m: n = int(m.group(1).replace(",", ""))
    m = re.search(r"Strategy: (\w+)", out)
    if m: strat = m.group(1)
    m = re.search(r"Time: ([\d.]+)s", out)
    if m: t = float(m.group(1))
    return n, strat, t


def run_one(binary, path, core, timeout):
    cmd = ["bash", "-c",
           f'echo $BASHPID > {CG}/cgroup.procs 2>/dev/null; exec taskset -c {core} nice -n -20 "$0" "$1"',
           binary, path]
    t0 = time.perf_counter()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"timeout": True, "wall": time.perf_counter() - t0}
    n, strat, t = parse(r.stdout)
    return {"timeout": False, "n": n, "strategy": strat, "time": t,
            "wall": time.perf_counter() - t0, "rc": r.returncode}


def collect(exclude):
    paths = []
    for cat in sorted(os.listdir(NONO)):
        if cat in exclude:
            continue
        d = os.path.join(NONO, cat)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            p = os.path.join(d, f)
            if os.path.isfile(p):
                paths.append((f"{cat}/{f}", p))
    return paths


def summarize(rows):
    by_cat = {}
    for r in rows:
        by_cat.setdefault(r["id"].split("/")[0], []).append(r)
    print(f"{'category':16s} {'n':>5s} {'timeouts':>8s} {'mismatch':>8s} {'sum':>10s} {'median':>9s} {'p90':>9s} {'max':>10s}  max puzzle")
    tot_n = tot_sum = tot_to = tot_mm = 0
    all_max = (0.0, "")
    for cat, rs in sorted(by_cat.items(), key=lambda kv: -sum(x.get("time") or 0 for x in kv[1])):
        times = [x["time"] for x in rs if not x["timeout"] and x.get("time") is not None]
        to = sum(1 for x in rs if x["timeout"])
        mm = sum(1 for x in rs if x.get("mismatch"))
        mx = max(((x["time"], x["id"]) for x in rs if not x["timeout"] and x.get("time") is not None), default=(0.0, ""))
        if mx[0] > all_max[0]:
            all_max = mx
        s = sum(times)
        med = statistics.median(times) if times else 0.0
        p90 = sorted(times)[int(0.9 * (len(times) - 1))] if times else 0.0
        print(f"{cat:16s} {len(rs):5d} {to:8d} {mm:8d} {s:10.3f} {med:9.4f} {p90:9.4f} {mx[0]:10.3f}  {mx[1]}")
        tot_n += len(rs); tot_sum += s; tot_to += to; tot_mm += mm
    print(f"{'TOTAL':16s} {tot_n:5d} {tot_to:8d} {tot_mm:8d} {tot_sum:10.3f} {'':9s} {'':9s} {all_max[0]:10.3f}  {all_max[1]}")
    return tot_to == 0 and tot_mm == 0


def compare(a_path, b_path):
    a = {json.loads(l)["id"]: json.loads(l) for l in open(a_path) if l.strip()}
    b = {json.loads(l)["id"]: json.loads(l) for l in open(b_path) if l.strip()}
    common = [i for i in a if i in b and not a[i]["timeout"] and not b[i]["timeout"]
              and a[i].get("time") is not None and b[i].get("time") is not None]
    sa = sum(a[i]["time"] for i in common)
    sb = sum(b[i]["time"] for i in common)
    print(f"{len(common)} puzzles timed in both: {os.path.basename(a_path)} {sa:.3f}s -> {os.path.basename(b_path)} {sb:.3f}s ({100*(sb/sa-1):+.1f}%)")
    diff = [i for i in common if (a[i].get("n"), a[i].get("strategy")) != (b[i].get("n"), b[i].get("strategy"))]
    print(f"count/strategy differences: {len(diff)}")
    for i in diff[:20]:
        print(f"  {i}: {a[i].get('n')}/{a[i].get('strategy')} -> {b[i].get('n')}/{b[i].get('strategy')}")
    ratios = sorted(((b[i]["time"] / a[i]["time"], i) for i in common if a[i]["time"] >= 0.01), key=lambda x: x[0])
    if ratios:
        print("largest speedups (time >= 10ms):")
        for r, i in ratios[:5]:
            print(f"  {i:24s} {a[i]['time']:.3f}s -> {b[i]['time']:.3f}s (x{1/r:.2f})")
        print("largest slowdowns (time >= 10ms):")
        for r, i in ratios[-5:][::-1]:
            print(f"  {i:24s} {a[i]['time']:.3f}s -> {b[i]['time']:.3f}s (x{1/r:.2f})")
    return len(diff) == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("binary", nargs="?")
    ap.add_argument("-o", "--out")
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--exclude", default=",".join(DEFAULT_EXCLUDE))
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    args = ap.parse_args()

    if args.compare:
        sys.exit(0 if compare(*args.compare) else 1)
    if not args.binary:
        ap.error("binary required unless --compare")

    core = shielded_core()
    paths = collect(set(args.exclude.split(",")) if args.exclude else set())
    print(f"{len(paths)} puzzles, core {core}, timeout {args.timeout}s, binary {args.binary}")
    out = open(args.out, "w") if args.out else None
    rows = []
    t0 = time.time()
    for i, (pid, path) in enumerate(paths):
        r = run_one(args.binary, path, core, args.timeout)
        gn, gs = golden(path)
        r["id"] = pid
        r["golden_n"] = gn
        r["golden_strategy"] = gs
        r["mismatch"] = (not r["timeout"]) and (r.get("n") != gn or r.get("strategy") != gs)
        rows.append(r)
        if out:
            out.write(json.dumps(r) + "\n"); out.flush()
        if r["timeout"]:
            print(f"  TIMEOUT  {pid}")
        elif r["mismatch"]:
            print(f"  MISMATCH {pid}: n={r.get('n')} (gold {gn}) strat={r.get('strategy')} (gold {gs})")
        if (i + 1) % 250 == 0:
            print(f"  ...{i+1}/{len(paths)} ({time.time()-t0:.0f}s)", flush=True)
    ok = summarize(rows)
    print(f"wall {time.time()-t0:.0f}s")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
