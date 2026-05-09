import re
import subprocess
import time
import os
from datetime import datetime
from functools import partial
from pathlib import Path

from cpuinfo import get_cpu_info

from picture import SolveStrategy
from puzzle_io import clues_valid, load_clues
from search import solve_with_strategy, __version__ as SOLVER_VERSION
from webpbn import fetch_webpbn

print = partial(print, flush=True)

PROGRESS_FILE = "webpbn_progress.txt"
IN_PROGRESS_DIR = "nonograms/in_progress"
SOLVER_CGROUP = "/sys/fs/cgroup/nonogram-solver"
CPU_MODEL = get_cpu_info().get("brand_raw", "unknown")
ISOLATED_CORE = None


def join_solver_cgroup():
    """Move this process into the exclusive cpuset cgroup if scripts/cpuset_setup.sh
    has been run. Returns the effective cpu list, or None if unavailable."""
    global ISOLATED_CORE
    if not os.path.isdir(SOLVER_CGROUP):
        return None
    try:
        with open(f"{SOLVER_CGROUP}/cgroup.procs", "w") as f:
            f.write(str(os.getpid()))
        with open(f"{SOLVER_CGROUP}/cpuset.cpus.effective") as f:
            ISOLATED_CORE = f.read().strip()
        return ISOLATED_CORE
    except OSError:
        return None


def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            try:
                return int(f.read().strip())
            except ValueError:
                return 0
    return 0


def save_progress(puzzle_id):
    with open(PROGRESS_FILE, 'w') as f:
        f.write(str(puzzle_id))


def save_in_progress(puzzle_id, clue_text):
    os.makedirs(IN_PROGRESS_DIR, exist_ok=True)
    filepath = f"{IN_PROGRESS_DIR}/{puzzle_id}"
    with open(filepath, "w") as f:
        f.write(clue_text)


def clear_in_progress(puzzle_id):
    filepath = f"{IN_PROGRESS_DIR}/{puzzle_id}"
    if os.path.exists(filepath):
        os.remove(filepath)


def get_time_category(solve_time, rows, cols):
    if solve_time >= 21600:
        return "insane"
    if solve_time >= 600:
        return "extreme"
    if solve_time >= 30:
        return "hard"
    if solve_time >= 5:
        return "medium"

    height = len(rows)
    width = len(cols)
    total_cells = height * width

    if total_cells <= 100:
        return "trivial"
    if total_cells <= 225:
        return "easy_small"
    if total_cells <= 400:
        return "easy_medium"
    return "easy_large"


def make_header_block(strategy, n_solutions, solve_time):
    lines = [
        f"# solver={SOLVER_VERSION}",
        f"# cpu={CPU_MODEL}",
    ]
    if ISOLATED_CORE:
        lines.append(f"# isolated_core={ISOLATED_CORE}")
    lines.append(f"# strategy={strategy}")
    lines.append(f"# n_solutions={n_solutions}")
    lines.append(f"# solve_time={solve_time:.5f}")
    return "\n".join(lines) + "\n"


def parse_header_blocks(text):
    """Split a puzzle file's leading comment lines into chronological blocks.

    Each block starts with `# solver=` and continues until the next `# solver=`
    or the first non-`#` line. Returns (list_of_block_strings, body_string).
    Files without a `# solver=` line return all leading `#` lines as one
    "legacy" block which will never match current.
    """
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


def header_matches_current(block):
    """True iff the block reports our SOLVER_VERSION and our CPU_MODEL."""
    has_solver = has_cpu = False
    for line in block.splitlines():
        if line.startswith("# solver="):
            if line[len("# solver="):].strip() != SOLVER_VERSION:
                return False
            has_solver = True
        elif line.startswith("# cpu="):
            if line[len("# cpu="):].strip() != CPU_MODEL:
                return False
            has_cpu = True
    return has_solver and has_cpu


def categorize_and_save(puzzle_id, clue_text, rows, cols, solution_count, strategy, solve_time):
    time_cat = get_time_category(solve_time, rows, cols)
    strat_cat = strategy.value

    folder = f"nonograms/{time_cat}"
    os.makedirs(folder, exist_ok=True)

    filepath = f"{folder}/{puzzle_id}"
    with open(filepath, "w") as f:
        f.write(make_header_block(strat_cat, solution_count, solve_time))
        f.write(clue_text)

    return f"{time_cat}/{strat_cat}"


def solve_puzzle_text(clue_text):
    lines = clue_text.strip().split('\n')

    cols = []
    rows = []
    for line in lines:
        if not line:
            cols.append([])
        elif line[0] == '#':
            continue
        elif line == '---':
            rows, cols = cols, rows
        else:
            cols.append([int(x) for x in line.split()])

    if not clues_valid(rows, cols):
        return None, None, None, None, None

    start = time.perf_counter()
    solution_count, strategy = solve_with_strategy(rows, cols)
    elapsed = time.perf_counter() - start

    return solution_count, strategy, elapsed, rows, cols


_STRATEGY_NAME_TO_ENUM = {
    "basic": SolveStrategy.BASIC,
    "contra": SolveStrategy.CONTRA,
    "backtrack": SolveStrategy.BACKTRACK,
}


def _solve_via_external(cmd, puzzle_path):
    """Run an external solver binary on `puzzle_path` and parse its stdout.

    Expects (case-sensitive) the lines emitted by cpp/main.cpp:
        Time: <X.XXXXX>s
        Found <N[,with,commas]> solution(s) (...)
        Strategy: basic|contra|backtrack

    Returns (n_solutions, strategy_enum, elapsed). Raises RuntimeError on
    non-zero exit or unparseable output.
    """
    proc = subprocess.run(
        [cmd, puzzle_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"external solver {cmd!r} failed on {puzzle_path!r} "
            f"(exit {proc.returncode}): {proc.stderr.strip()}"
        )

    out = proc.stdout

    m_found = re.search(r"^Found\s+([\d,]+)\s+solution\(s\)", out, re.MULTILINE)
    m_time = re.search(r"^Time:\s+([0-9]*\.?[0-9]+)s", out, re.MULTILINE)
    m_strat = re.search(r"^Strategy:\s+(\S+)", out, re.MULTILINE)

    if not (m_found and m_time and m_strat):
        raise RuntimeError(
            f"external solver {cmd!r}: could not parse output for {puzzle_path!r}.\n"
            f"stdout: {out!r}"
        )

    n_solutions = int(m_found.group(1).replace(",", ""))
    elapsed = float(m_time.group(1))
    strat_name = m_strat.group(1).strip()
    if strat_name not in _STRATEGY_NAME_TO_ENUM:
        raise RuntimeError(
            f"external solver {cmd!r}: unknown strategy {strat_name!r}"
        )
    strategy = _STRATEGY_NAME_TO_ENUM[strat_name]
    return n_solutions, strategy, elapsed


def rebench_file(path, solver_cmd=None):
    try:
        rows, cols = load_clues(path)
    except Exception as exc:
        print(f"  {path}: parse error ({exc})")
        return False, path, None
    if not clues_valid(rows, cols):
        print(f"  {path}: invalid clues")
        return False, path, None

    if solver_cmd is None:
        start = time.perf_counter()
        n_solutions, strategy = solve_with_strategy(rows, cols)
        elapsed = time.perf_counter() - start
    else:
        n_solutions, strategy, elapsed = _solve_via_external(solver_cmd, path)
    strat_cat = strategy.value

    with open(path) as f:
        full = f.read()
    blocks, body = parse_header_blocks(full)
    new_block = make_header_block(strat_cat, n_solutions, elapsed)
    if blocks and header_matches_current(blocks[0]):
        blocks[0] = new_block
    else:
        blocks.insert(0, new_block)
    new_content = "".join(blocks) + body

    time_cat = get_time_category(elapsed, rows, cols)
    fname = os.path.basename(path)
    new_path = os.path.join("nonograms", time_cat, fname)
    moved = os.path.realpath(new_path) != os.path.realpath(path)

    if moved:
        os.makedirs(os.path.dirname(new_path), exist_ok=True)
    with open(new_path, "w") as f:
        f.write(new_content)
    if moved and os.path.exists(path):
        os.remove(path)
    return True, new_path, elapsed


def rebench_folder(root, excludes, solver_cmd=None):
    print(f"Rebenching {root}")
    if excludes:
        print(f"Excluded folders (exact-name match): {sorted(excludes)}")
    print("-" * 60)

    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        parts = Path(dirpath).parts
        if any(part in excludes for part in parts):
            dirnames[:] = []
            continue
        dirnames[:] = sorted(d for d in dirnames if d not in excludes)
        for fname in sorted(filenames):
            files.append(os.path.join(dirpath, fname))

    rebenched = moved = 0
    for path in files:
        ok, new_path, elapsed = rebench_file(path, solver_cmd=solver_cmd)
        if not ok:
            continue
        rebenched += 1
        if new_path != path:
            moved += 1
            print(f"  {path}  ->  {new_path}  ({elapsed:.4f}s)")
        else:
            print(f"  {path}  ({elapsed:.4f}s)")

    print("-" * 60)
    print(f"Rebenched {rebenched} puzzles, moved {moved} between buckets.")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Fetch and categorize webpbn puzzles')
    parser.add_argument('--start', type=int, default=None,
                        help='Starting puzzle ID (overrides progress file)')
    parser.add_argument('--max-not-found', type=int, default=1000,
                        help='Stop after N consecutive not-found puzzles')
    parser.add_argument('--max-puzzles', type=int, default=None,
                        help='Maximum number of puzzles to process')
    parser.add_argument('--timeout', type=float, default=None,
                        help='Maximum solve time per puzzle in seconds (skip if exceeded)')
    parser.add_argument('--rebench', metavar='DIR',
                        help='Re-solve every puzzle file under DIR with the current solver instead of fetching from webpbn')
    parser.add_argument('--exclude', action='append', default=[], metavar='FOLDER',
                        help='Folder name (exact match, any depth) to skip during --rebench. Repeatable.')
    parser.add_argument('--solver-cmd', metavar='PATH', default=None,
                        help='External solver binary to use during --rebench. Each '
                             'puzzle file path is passed as the sole argument; the '
                             'solver must print "Found N solution(s)", "Time: Xs", '
                             'and "Strategy: <name>" lines on stdout. '
                             'Ignored outside --rebench mode.')
    args = parser.parse_args()

    joined = join_solver_cgroup()
    if joined:
        print(f"Pinned to exclusive cpuset core(s): {joined}")
    else:
        print(f"(no exclusive cpuset; run scripts/cpuset_setup.sh as root for stable timings)")

    if args.rebench:
        rebench_folder(args.rebench, set(args.exclude), solver_cmd=args.solver_cmd)
        return

    if args.start is not None:
        current_id = args.start
    else:
        current_id = load_progress() + 1

    not_found_streak = 0
    puzzles_processed = 0

    print(f"Starting from puzzle #{current_id}")
    print(f"Progress will be saved to {PROGRESS_FILE}")
    print("-" * 60)

    while not_found_streak < args.max_not_found:
        if args.max_puzzles and puzzles_processed >= args.max_puzzles:
            print(f"\nReached max puzzles limit ({args.max_puzzles})")
            break

        print(f"#{current_id}: ", end='', flush=True)

        clue_text = fetch_webpbn(current_id)

        if clue_text is None:
            print("not found / colored")
            not_found_streak += 1
            current_id += 1
            save_progress(current_id - 1)
            continue

        not_found_streak = 0

        save_in_progress(current_id, clue_text)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-1]
        print(f"found [{timestamp}], solving... ", end='', flush=True)

        solution_count, strategy, solve_time, rows, cols = solve_puzzle_text(clue_text)

        if solution_count is None:
            print("invalid clues")
            clear_in_progress(current_id)
            current_id += 1
            save_progress(current_id - 1)
            continue

        if args.timeout and solve_time > args.timeout:
            print(f"timeout ({solve_time:.1f}s > {args.timeout}s)")
            current_id += 1
            save_progress(current_id - 1)
            puzzles_processed += 1
            continue

        category = categorize_and_save(current_id, clue_text, rows, cols, solution_count, strategy, solve_time)
        clear_in_progress(current_id)

        print(f"{solution_count} solutions, {solve_time:.2f}s -> {category}")

        current_id += 1
        save_progress(current_id - 1)
        puzzles_processed += 1

    print("-" * 60)
    print(f"Finished. Processed {puzzles_processed} puzzles.")
    print(f"Last puzzle ID: {current_id - 1}")


if __name__ == "__main__":
    main()
