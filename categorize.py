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
PARTIALLY_SOLVED_DIR = "nonograms/partially_solved"
SOLVER_CGROUP = "/sys/fs/cgroup/nonogram-solver"
CPU_MODEL = get_cpu_info().get("brand_raw", "unknown")
ISOLATED_CORE = None
KEEP_OLD_HEADERS = False  # --keep-old-headers: retain previous header blocks in the file


def join_solver_cgroup():
    """Move this process into the exclusive cpuset cgroup if scripts/cpuset_setup.sh
    has been run. Returns the effective cpu list, or None if unavailable."""
    global ISOLATED_CORE
    # Already inside an exclusive cpuset partition (cpp/bench/run.sh puts a
    # command into /sys/fs/cgroup/bench, the core bench/shield.sh
    # reserved): record that core and leave the process where it is.
    try:
        with open("/proc/self/cgroup") as f:
            own = next((l.split(":", 2)[2].strip() for l in f if l.startswith("0::")), "")
        own_dir = "/sys/fs/cgroup" + own
        with open(f"{own_dir}/cpuset.cpus.partition") as f:
            partition = f.read().strip()
        if partition in ("root", "isolated"):
            with open(f"{own_dir}/cpuset.cpus.effective") as f:
                ISOLATED_CORE = f.read().strip()
            return ISOLATED_CORE
    except (OSError, StopIteration):
        pass
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


class ExternalSolverTimeout(Exception):
    """Raised when the external solver exceeds the configured timeout."""
    found_solution = False  # set from the solver's partial output


def _has_a_solution(cmd, puzzle_path, timeout_s):
    """True if `cmd --anytime --max 1` prints a Found line within timeout_s."""
    try:
        proc = subprocess.run(
            [cmd, puzzle_path, "--anytime", "--max", "1"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return False
    return re.search(r"^Found\s+[1-9]", proc.stdout, re.MULTILINE) is not None


def _solve_via_external(cmd, puzzle_path, timeout_s=None):
    """Run an external solver binary on `puzzle_path` and parse its stdout.

    Expects (case-sensitive) the lines emitted by cpp/main.cpp:
        Time: <X.XXXXX>s
        Found <N[,with,commas]> solution(s) (...)
        Strategy: basic|contra|backtrack

    Returns (n_solutions, strategy_enum, elapsed). Raises RuntimeError on
    non-zero exit or unparseable output. Raises ExternalSolverTimeout if
    the subprocess exceeds `timeout_s` seconds (when not None).
    """
    try:
        proc = subprocess.run(
            [cmd, puzzle_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        partial = exc.stdout or b""
        if isinstance(partial, bytes):
            partial = partial.decode(errors="replace")
        err = ExternalSolverTimeout(
            f"external solver {cmd!r} exceeded {timeout_s}s on {puzzle_path!r}"
        )
        # An enumerating run announces the first solution and then progress
        # batches as "<count> (<rate>/s, ...)"; any such entry means at
        # least one solution was found before the cut-off. A count-mode run
        # (the default) prints no solutions, so ask for one directly: a
        # short --anytime --max 1 run that finds one settles it.
        err.found_solution = re.search(r"(^|\s)[\d,]+ \(", partial) is not None
        if not err.found_solution:
            err.found_solution = _has_a_solution(cmd, puzzle_path, min(60.0, timeout_s or 60.0))
        raise err
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


def solve_puzzle_text_external(clue_text, puzzle_id, solver_cmd, timeout_s=None):
    """Same shape as `solve_puzzle_text`, but runs the external solver on the
    in-progress file at `nonograms/in_progress/<puzzle_id>`.

    Returns (solution_count, strategy, elapsed, rows, cols), or
    (None, None, None, None, None) if the parsed clues are invalid.

    Assumes `save_in_progress(puzzle_id, clue_text)` has already been called
    by the caller — we read the rows/cols from clue_text but invoke the
    external solver on the saved file path.

    Propagates `ExternalSolverTimeout` from `_solve_via_external` so the
    caller can distinguish timeout from other failures.
    """
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

    puzzle_path = f"{IN_PROGRESS_DIR}/{puzzle_id}"
    n_solutions, strategy, elapsed = _solve_via_external(
        solver_cmd, puzzle_path, timeout_s=timeout_s
    )
    return n_solutions, strategy, elapsed, rows, cols


def rebench_file(path, solver_cmd=None, repeat=1):
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
        # Best of `repeat` runs: the fastest time is the one least disturbed
        # by whatever else the machine was doing; the count must not vary.
        n_solutions, strategy, elapsed = _solve_via_external(solver_cmd, path)
        for _ in range(repeat - 1):
            n2, s2, t2 = _solve_via_external(solver_cmd, path)
            if n2 != n_solutions or s2 != strategy:
                raise RuntimeError(f"{path}: runs disagree ({n_solutions} {strategy.value} vs {n2} {s2.value})")
            elapsed = min(elapsed, t2)
    return place_with_header(path, rows, cols, n_solutions, strategy, elapsed)


def record_file(path, log_path):
    """Write the golden header from a finished solver run's stdout (a
    journal dump with the Found / Time / Strategy lines) instead of solving
    again; a count that took an hour in a service does not need a second
    hour to be recorded."""
    try:
        rows, cols = load_clues(path)
    except Exception as exc:
        print(f"  {path}: parse error ({exc})")
        return False, path, None
    with open(log_path) as f:
        out = f.read()
    # The journal of a reused unit name holds every run, and a run stopped
    # by a signal prints Time: and Strategy: without a Found line; so the
    # record is the LAST Found line and the Time / Strategy lines of that
    # run (the ones between the Time: line just before it and the next
    # run's start).
    founds = list(re.finditer(r"^Found\s+([\d,]+)\s+solution\(s\)", out, re.MULTILINE))
    m_found = founds[-1] if founds else None
    m_time = m_strat = None
    if m_found:
        times = list(re.finditer(r"^Time:\s+([0-9]*\.?[0-9]+)s", out[:m_found.start()], re.MULTILINE))
        m_time = times[-1] if times else None
        m_strat = re.search(r"^Strategy:\s+(\S+)", out[m_found.end():], re.MULTILINE)
    if not (m_found and m_time and m_strat) or m_strat.group(1) not in _STRATEGY_NAME_TO_ENUM:
        print(f"  {log_path}: no complete Found / Time / Strategy lines")
        return False, path, None
    n_solutions = int(m_found.group(1).replace(",", ""))
    elapsed = float(m_time.group(1))
    strategy = _STRATEGY_NAME_TO_ENUM[m_strat.group(1)]
    ok, new_path, elapsed = place_with_header(path, rows, cols, n_solutions, strategy, elapsed)
    print(f"  {path}  ->  {new_path}  ({elapsed:.4f}s, {n_solutions} solutions, recorded)")
    return ok, new_path, elapsed


def strip_old_headers(path):
    """Keep only the newest header block of a puzzle file (the history is
    in git). Returns True when the file changed."""
    with open(path) as f:
        full = f.read()
    blocks, body = parse_header_blocks(full)
    if len(blocks) <= 1:
        return False
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        f.write(blocks[0] + body)
    os.replace(tmp_path, path)
    return True


def place_with_header(path, rows, cols, n_solutions, strategy, elapsed):
    """Prepend the current header block and move the file to its time bucket."""
    strat_cat = strategy.value

    with open(path) as f:
        full = f.read()
    blocks, body = parse_header_blocks(full)
    new_block = make_header_block(strat_cat, n_solutions, elapsed)
    if KEEP_OLD_HEADERS:
        if blocks and header_matches_current(blocks[0]):
            blocks[0] = new_block
        else:
            blocks.insert(0, new_block)
    else:
        # Only the current block: the history is in git, not in the file.
        blocks = [new_block]
    new_content = "".join(blocks) + body

    time_cat = get_time_category(elapsed, rows, cols)
    fname = os.path.basename(path)
    new_path = os.path.join("nonograms", time_cat, fname)
    moved = os.path.realpath(new_path) != os.path.realpath(path)

    if moved:
        os.makedirs(os.path.dirname(new_path), exist_ok=True)
    # Written to a temp file and renamed into place: a run killed between
    # open() and write() left easy_large/2955 empty (2026-09-21), and the
    # next rebench filed the empty clue set under trivial.
    tmp_path = new_path + ".tmp"
    with open(tmp_path, "w") as f:
        f.write(new_content)
    os.replace(tmp_path, new_path)
    if moved and os.path.exists(path):
        os.remove(path)
    return True, new_path, elapsed


def previous_solve_time(path):
    """solve_time of the file's newest header block, or None without one."""
    try:
        with open(path) as f:
            for line in f:
                if not line.startswith("#"):
                    return None
                m = re.match(r"# solve_time=([0-9.]+)", line)
                if m:
                    return float(m.group(1))
    except OSError:
        return None
    return None


def rebench_folder(root, excludes, solver_cmd=None, repeat=1):
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
    # Shortest previous solve first, so results land in ascending order and
    # a run cut short has done the cheap ones; files without a header last.
    before = {p: previous_solve_time(p) for p in files}
    files.sort(key=lambda p: (before[p] is None, before[p] or 0.0, p))

    rebenched = moved = 0
    paired = []  # (path, previous time, new time) for files that had a header
    for path in files:
        ok, new_path, elapsed = rebench_file(path, solver_cmd=solver_cmd, repeat=repeat)
        if not ok:
            continue
        rebenched += 1
        old = before[path]
        was = f"{old:.4f}s -> " if old is not None else ""
        if new_path != path:
            moved += 1
            print(f"  {path}  ->  {new_path}  ({was}{elapsed:.4f}s)")
        else:
            print(f"  {path}  ({was}{elapsed:.4f}s)")
        if old is not None:
            paired.append((path, old, elapsed))

    print("-" * 60)
    print(f"Rebenched {rebenched} puzzles, moved {moved} between buckets.")
    report_speed_change(paired)


def report_speed_change(paired, movers=12):
    """The rebench's speed change against the headers it replaced: the sum
    of the previous and the new solve times over the files that had one,
    and the largest gains and losses, for the rebench commit's message."""
    if not paired:
        print("No previous solve times to compare against.")
        return
    old_sum = sum(old for _, old, _ in paired)
    new_sum = sum(new for _, _, new in paired)
    change = f" ({(new_sum / old_sum - 1) * 100:+.1f}%)" if old_sum else ""
    print(f"{len(paired)} re-headed files with a previous time: {old_sum:.1f} -> {new_sum:.1f} s{change}")
    by_gain = sorted(paired, key=lambda t: t[1] - t[2], reverse=True)
    print("largest gains:")
    for path, old, new in by_gain[:movers]:
        if new < old:
            print(f"  {path}: {old:.4g} -> {new:.4g} s")
    print("largest losses:")
    for path, old, new in by_gain[-movers:][::-1]:
        if new > old:
            print(f"  {path}: {old:.4g} -> {new:.4g} s")


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
    parser.add_argument('--record', nargs=2, metavar=('PUZZLE', 'LOG'),
                        help='Write the golden header for PUZZLE from LOG, a finished solver run\'s stdout '
                             '(journalctl -o cat dump with the Found / Time / Strategy lines), and move '
                             'the file to its time bucket; no re-solve')
    parser.add_argument('--keep-old-headers', action='store_true',
                        help='Retain previous header blocks in a re-headed file (default: only the current block)')
    parser.add_argument('--strip-old-headers', nargs='+', metavar='PUZZLE',
                        help='Rewrite the named puzzle files keeping only their newest header block')
    parser.add_argument('--repeat', type=int, default=1, metavar='N',
                        help='--rebench with --solver-cmd: solve each puzzle N times and keep the '
                             'fastest time (the counts must agree)')
    parser.add_argument('--rebench', metavar='DIR',
                        help='Re-solve every puzzle file under DIR with the current solver instead of fetching from webpbn')
    parser.add_argument('--exclude', action='append', default=[], metavar='FOLDER',
                        help='Folder name (exact match, any depth) to skip during --rebench. Repeatable.')
    parser.add_argument('--solver-cmd', metavar='PATH', default=None,
                        help='External solver binary to use for solving. Each '
                             'puzzle file path is passed as the sole argument; the '
                             'solver must print "Found N solution(s)", "Time: Xs", '
                             'and "Strategy: <name>" lines on stdout. '
                             'Works in both --rebench mode and harvest (default) mode. '
                             'In harvest mode, the saved in_progress file is used as '
                             'the puzzle path passed to the solver.')
    parser.add_argument('--solver-version', metavar='LABEL', default=None,
                        help='Label written as "# solver=" in the header blocks. Defaults to '
                             'the Python solver version, or "cpp-<short sha of the last commit '
                             'that touched cpp/*.cpp, cpp/*.hpp, cpp/Makefile or cpp/external>" '
                             'when --solver-cmd is given, so blocks from the two solvers never '
                             'replace each other.')
    args = parser.parse_args()

    global SOLVER_VERSION
    if args.solver_version:
        SOLVER_VERSION = args.solver_version
    elif args.solver_cmd:
        # The last commit that touched what the binary is built from, not
        # HEAD: a corpus, docs, bench or ignore-file commit does not change
        # what the solver does.
        sha = subprocess.run(['git', 'log', '-1', '--format=%h', '--', 'cpp/*.cpp', 'cpp/*.hpp', 'cpp/Makefile', 'cpp/external'],
                             capture_output=True, text=True).stdout.strip()
        SOLVER_VERSION = f"cpp-{sha or 'unknown'}"
    print(f"Header solver label: {SOLVER_VERSION}")

    joined = join_solver_cgroup()
    if joined:
        print(f"Pinned to exclusive cpuset core(s): {joined}")
    else:
        print(f"(no exclusive cpuset; run scripts/cpuset_setup.sh as root for stable timings)")

    global KEEP_OLD_HEADERS
    KEEP_OLD_HEADERS = args.keep_old_headers

    if args.strip_old_headers:
        changed = sum(strip_old_headers(p) for p in args.strip_old_headers)
        print(f"stripped old header blocks from {changed} of {len(args.strip_old_headers)} files")
        return

    if args.record:
        record_file(args.record[0], args.record[1])
        return

    if args.rebench:
        rebench_folder(args.rebench, set(args.exclude), solver_cmd=args.solver_cmd, repeat=args.repeat)
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

        if args.solver_cmd:
            try:
                solution_count, strategy, solve_time, rows, cols = (
                    solve_puzzle_text_external(
                        clue_text, current_id, args.solver_cmd, timeout_s=args.timeout
                    )
                )
            except ExternalSolverTimeout as exc:
                if getattr(exc, "found_solution", False):
                    # Solutions exist but the enumeration did not finish:
                    # a partially solved puzzle, kept without a header.
                    os.makedirs(PARTIALLY_SOLVED_DIR, exist_ok=True)
                    os.replace(f"{IN_PROGRESS_DIR}/{current_id}", f"{PARTIALLY_SOLVED_DIR}/{current_id}")
                    print(f"timeout (>{args.timeout}s) with solutions -> partially_solved")
                else:
                    # No solution within the timeout: left in in_progress.
                    print(f"timeout (>{args.timeout}s), no solution yet -> in_progress")
                current_id += 1
                save_progress(current_id - 1)
                puzzles_processed += 1
                continue
        else:
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
