import time
import sys
from os import listdir
from os.path import isfile, join

from categorize import load_clues, clues_valid, solve


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


def benchmark(rows, cols, runs=5):
    print(f"Benchmarking {runs} runs...")
    times = []
    pic = None
    for i in range(runs):
        start = time.perf_counter()
        for pic in solve(rows, cols):
            break
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"  Run {i + 1}: {elapsed:.4f}s")

    avg = sum(times) / len(times)
    min_t = min(times)
    max_t = max(times)
    print(f"\nAverage: {avg:.4f}s")
    print(f"Min: {min_t:.4f}s, Max: {max_t:.4f}s")
    return pic


def main():
    if len(sys.argv) < 2:
        print("Usage: python solver.py <puzzle_file> [--benchmark] [--runs N] [--print]")
        print("\nPuzzle file format:")
        print("  Row clues (one per line, space-separated numbers)")
        print("  ---")
        print("  Column clues (one per line, space-separated numbers)")
        print("\nOptions:")
        print("  --benchmark  Run multiple times and report statistics")
        print("  --runs N     Number of benchmark runs (default 5)")
        print("  --print      Print progress and grid during solving")
        sys.exit(1)

    filename = sys.argv[1]
    do_benchmark = '--benchmark' in sys.argv
    print_progress = '--print' in sys.argv

    runs = 5
    if '--runs' in sys.argv:
        idx = sys.argv.index('--runs')
        if idx + 1 < len(sys.argv):
            runs = int(sys.argv[idx + 1])

    rows, cols = load_clues(filename)

    if not clues_valid(rows, cols):
        print(f"Invalid clues: {filename}")
        sys.exit(1)

    if do_benchmark:
        pic = benchmark(rows, cols, runs)
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
                print(f"\n=== Solution {solution_count} found ===")
                print(pic)
            elif solution_count % print_count_threshold == 0:
                print(f"{solution_count} ", end='', flush=True)
                print_count_threshold = int(print_count_threshold * 1.1)
                print_count += 1
                if print_count == 10:
                    print()
                    print_count = 0
        print()
        elapsed = time.perf_counter() - start

        print(f"\nTime: {elapsed:.4f}s")
        print(f"Found {solution_count} solution(s)")


if __name__ == "__main__":
    main()