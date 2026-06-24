#!/usr/bin/env python3
"""Correctness gate + timing harness for the C++ nonogram solver.

Runs the solver on the shielded core (via bench/run.sh) to completion and
compares exact solution count + strategy against the golden values stored in
each puzzle's header (# n_solutions=, # strategy=).

Modes:
  gate   -- correctness only: run every gate puzzle to completion, assert
            solution count + strategy match golden. Exit 1 on any mismatch.
  bench  -- timing: run each bench puzzle REPS times, report best/median.
  both   -- gate then bench (default).
"""
import os, re, subprocess, sys, time, statistics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))      # cpp/
NONO = os.path.join(os.path.dirname(ROOT), "nonograms")
SOLVER = os.path.join(ROOT, "solver")
RUN = os.path.join(ROOT, "bench", "run.sh")

# Gate: run to completion, verify exact count + strategy. Keep total < ~30s.
GATE = [
    "easy_large/3250", "easy_large/3162", "easy_large/3094", "easy_large/2606",
    "easy_large/2040", "easy_large/2403", "easy_large/3371", "easy_large/1837",
    "easy_large/3379", "easy_large/4758", "easy_large/803", "easy_large/6727",
    "medium/108", "medium/3163",
]
# Bench: timing targets. (path, extra_args)
BENCH = [
    ("medium/108", []),       # enumeration, 564k solutions
    ("medium/3163", []),      # single-solution deep backtrack
    ("easy_large/6727", []),  # small enumeration
    ("easy_large/803", []),   # contra-heavy single solution
]

def golden(path):
    n = strat = None
    with open(os.path.join(NONO, path)) as f:
        for line in f:
            if not line.startswith("#"):
                break
            m = re.match(r"#\s*n_solutions=(\d+)", line)
            if m: n = int(m.group(1))
            m = re.match(r"#\s*strategy=(\w+)", line)
            if m: strat = m.group(1)
    return n, strat

def run(path, extra=(), isolated=True):
    cmd = ([RUN] if isolated else []) + [SOLVER, os.path.join(NONO, path)] + list(extra)
    t0 = time.perf_counter()
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    dt = time.perf_counter() - t0
    n = strat = None
    m = re.search(r"Found ([\d,]+) solution", out)
    if m: n = int(m.group(1).replace(",", ""))
    m = re.search(r"Strategy: (\w+)", out)
    if m: strat = m.group(1)
    mt = re.search(r"Time: ([\d.]+)s", out)
    solve_t = float(mt.group(1)) if mt else dt
    return n, strat, solve_t

def do_gate():
    print("=== GATE (correctness) ===")
    ok = True
    for p in GATE:
        gn, gs = golden(p)
        n, s, t = run(p)
        good = (n == gn and s == gs)
        ok = ok and good
        print(f"  [{'OK ' if good else 'FAIL'}] {p:22s} n={n} (gold {gn}) strat={s} (gold {gs})  {t:.3f}s")
    print("GATE:", "PASS" if ok else "FAIL")
    return ok

def do_bench(reps=3):
    print(f"=== BENCH (timing, reps={reps}, shielded core 5) ===")
    results = {}
    for p, extra in BENCH:
        ts = []
        for _ in range(reps):
            _, _, t = run(p, extra)
            ts.append(t)
        best, med = min(ts), statistics.median(ts)
        results[p] = best
        print(f"  {p:22s} best={best:.4f}s median={med:.4f}s  (all={[f'{x:.3f}' for x in ts]})")
    total = sum(results.values())
    print(f"BENCH total(best): {total:.4f}s")
    return results

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    rc = 0
    if mode in ("gate", "both"):
        if not do_gate(): rc = 1
    if mode in ("bench", "both"):
        do_bench(reps)
    sys.exit(rc)
