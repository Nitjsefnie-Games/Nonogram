import time
import sys
from os import listdir
from os.path import isfile, join

from puzzle_io import load_clues, clues_valid
from search import solve


def solve_folder(loc):
    start = time.time()
    for file in sorted(join(loc, f) for f in listdir(loc) if isfile(join(loc, f))):
        solve_file(file)
    print(f"\nAll from {loc}: {time.time() - start}\n")


def solve_file(location, number=-1):
    rows, cols = load_clues(location)
    if not clues_valid(rows, cols):
        print(f"Invalid clues: {location}")
        return -1
    start = time.time()
    i = 0
    for pic in solve(rows, cols):
        i += 1
        print(f"{i}", end=' ')
        if i == number:
            break
    print(f"{location}: {time.time() - start}, found {i} solutions")
    return i


def benchmark(rows, cols, runs=5, max_solutions=None):
    print(f"Benchmarking {runs} runs...")
    times = []
    counts = []
    pic = None
    for i in range(runs):
        start = time.perf_counter()
        count = 0
        for pic in solve(rows, cols):
            count += 1
            if max_solutions is not None and count >= max_solutions:
                break
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        counts.append(count)
        rate = count / elapsed if elapsed > 0 else 0
        print(f"  Run {i + 1}: {elapsed:.4f}s ({count:,} solutions, {rate:,.0f}/s)")

    avg = sum(times) / len(times)
    min_t = min(times)
    max_t = max(times)
    print(f"\nAverage: {avg:.4f}s")
    print(f"Min: {min_t:.4f}s, Max: {max_t:.4f}s")
    if counts and all(c == counts[0] for c in counts):
        print(f"Solutions: {counts[0]:,}")
    else:
        print(f"Solutions per run: {counts}")
    return pic


def main():
    if len(sys.argv) < 2:
        print("Usage: python solver.py <puzzle_file> [--benchmark] [--runs N] [--print] [--max N]")
        print("\nPuzzle file format:")
        print("  Row clues (one per line, space-separated numbers)")
        print("  ---")
        print("  Column clues (one per line, space-separated numbers)")
        print("\nOptions:")
        print("  --benchmark  Run multiple times; reports time, solution count, rate")
        print("               (combine with --max N to cap each run)")
        print("  --runs N     Number of benchmark runs (default 5)")
        print("  --print      Print progress and grid during solving")
        print("  --max N      Stop after finding N solutions")
        sys.exit(1)

    filename = sys.argv[1]
    do_benchmark = '--benchmark' in sys.argv
    print_progress = '--print' in sys.argv

    runs = 5
    if '--runs' in sys.argv:
        idx = sys.argv.index('--runs')
        if idx + 1 < len(sys.argv):
            runs = int(sys.argv[idx + 1])

    max_solutions = None
    if '--max' in sys.argv:
        idx = sys.argv.index('--max')
        if idx + 1 < len(sys.argv):
            max_solutions = int(sys.argv[idx + 1])

    rows, cols = load_clues(filename)

    if not clues_valid(rows, cols):
        print(f"Invalid clues: {filename}")
        sys.exit(1)

    if do_benchmark:
        pic = benchmark(rows, cols, runs, max_solutions=max_solutions)
    else:
        print(f"Puzzle size: {len(rows)} rows x {len(cols)} cols")
        print("Solving...")

        start = time.perf_counter()
        solution_count = 0
        print_count_threshold = 10
        print_count = 0
        for pic in solve(rows, cols, print_progress=print_progress):
            solution_count += 1
            if print_progress:
                elapsed = time.perf_counter() - start
                rate = solution_count / elapsed if elapsed > 0 else 0
                print(f"\n=== Solution {solution_count} found ({rate:,.0f}/s, elapsed {elapsed:.1f}s) ===")
                print(pic)
            elif solution_count % print_count_threshold == 0:
                elapsed = time.perf_counter() - start
                rate = solution_count / elapsed if elapsed > 0 else 0
                print(f"{solution_count:,} ({rate:,.0f}/s, {elapsed:.1f}s) ", end='', flush=True)
                print_count_threshold = int(print_count_threshold * 1.1)
                print_count += 1
                if print_count == 10:
                    print()
                    print_count = 0
            if max_solutions is not None and solution_count >= max_solutions:
                break
        print()
        elapsed = time.perf_counter() - start

        print(f"\nTime: {elapsed:.4f}s")
        rate = solution_count / elapsed if elapsed > 0 else 0
        print(f"Found {solution_count:,} solution(s) ({rate:,.0f}/s)")


if __name__ == "__main__":
    main()