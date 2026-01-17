import time
import os
from sys import setrecursionlimit

import requests
import xml.etree.ElementTree as ET
from io import StringIO
from enum import Enum
from datetime import datetime
import numpy as np
from numba import njit

EMPTY = 0
FULL = 1
UNKNOWN = 2

setrecursionlimit(100000)


def check_line(clue, size):
    if len(clue) == 1:
        return clue[0] <= size
    return sum(clue) + len(clue) - 1 <= size


def load_clues(filename):
    with open(filename) as f:
        content = [x.strip() for x in f]
    cols = []
    rows = []
    for line in content:
        if not line:
            cols.append([])
        elif line[0] == '#':
            continue
        elif line == '---':
            rows, cols = cols, rows
        else:
            cols.append([int(x) for x in line.split()])
    return rows, cols


def clues_valid(rows, cols):
    return (sum(map(sum, rows)) == sum(map(sum, cols))
            and all(check_line(r, len(cols)) for r in rows)
            and all(check_line(c, len(rows)) for c in cols))


@njit(cache=True)
def solve_line_batch(line, states):
    n = len(line)
    len_states = len(states)

    forward = np.zeros((n + 1, len_states), dtype=np.int64)
    forward[0, 0] = 1

    for pos in range(n):
        val = line[pos]
        for state in range(len_states):
            count = forward[pos, state]
            if count == 0:
                continue

            cur_state_val = states[state]
            next_state = state + 1
            next_state_val = states[next_state] if next_state < len_states else -1

            if val == UNKNOWN:
                if cur_state_val == EMPTY:
                    forward[pos + 1, state] += count
                if next_state_val != -1:
                    forward[pos + 1, next_state] += count
            elif val == EMPTY:
                if cur_state_val == EMPTY:
                    forward[pos + 1, state] += count
                if next_state_val == EMPTY:
                    forward[pos + 1, next_state] += count
            else:
                if next_state_val == FULL:
                    forward[pos + 1, next_state] += count

    total = np.int64(0)
    if len_states >= 1:
        total += forward[n, len_states - 1]
    if len_states >= 2:
        total += forward[n, len_states - 2]

    if total == 0:
        return np.full(n, UNKNOWN, dtype=np.int32), np.int64(0)

    backward = np.zeros((n + 1, len_states), dtype=np.int64)
    if len_states >= 1:
        backward[n, len_states - 1] = 1
    if len_states >= 2:
        backward[n, len_states - 2] = 1

    for pos in range(n - 1, -1, -1):
        val = line[pos]
        for state in range(len_states):
            cur_state_val = states[state]
            next_state = state + 1
            next_state_val = states[next_state] if next_state < len_states else -1

            if val == UNKNOWN:
                if cur_state_val == EMPTY:
                    backward[pos, state] += backward[pos + 1, state]
                if next_state_val != -1:
                    backward[pos, state] += backward[pos + 1, next_state]
            elif val == EMPTY:
                if cur_state_val == EMPTY:
                    backward[pos, state] += backward[pos + 1, state]
                if next_state_val == EMPTY:
                    backward[pos, state] += backward[pos + 1, next_state]
            else:
                if next_state_val == FULL:
                    backward[pos, state] += backward[pos + 1, next_state]

    result = np.full(n, UNKNOWN, dtype=np.int32)

    for pos in range(n):
        if line[pos] != UNKNOWN:
            result[pos] = line[pos]
            continue

        can_empty = np.int64(0)
        can_full = np.int64(0)

        for state in range(len_states):
            fwd = forward[pos, state]
            if fwd == 0:
                continue

            cur_state_val = states[state]
            next_state = state + 1
            next_state_val = states[next_state] if next_state < len_states else -1

            if cur_state_val == EMPTY:
                can_empty += fwd * backward[pos + 1, state]
            if next_state_val == EMPTY:
                can_empty += fwd * backward[pos + 1, next_state]

            if next_state_val == FULL:
                can_full += fwd * backward[pos + 1, next_state]

        if can_empty > 0 and can_full == 0:
            result[pos] = EMPTY
        elif can_full > 0 and can_empty == 0:
            result[pos] = FULL

    return result, total


@njit(cache=True)
def check_line_valid(line, states):
    n = len(line)
    len_states = len(states)

    forward = np.zeros((n + 1, len_states), dtype=np.int64)
    forward[0, 0] = 1

    for pos in range(n):
        val = line[pos]
        for state in range(len_states):
            count = forward[pos, state]
            if count == 0:
                continue

            cur_state_val = states[state]
            next_state = state + 1
            next_state_val = states[next_state] if next_state < len_states else -1

            if val == UNKNOWN:
                if cur_state_val == EMPTY:
                    forward[pos + 1, state] += count
                if next_state_val != -1:
                    forward[pos + 1, next_state] += count
            elif val == EMPTY:
                if cur_state_val == EMPTY:
                    forward[pos + 1, state] += count
                if next_state_val == EMPTY:
                    forward[pos + 1, next_state] += count
            else:
                if next_state_val == FULL:
                    forward[pos + 1, next_state] += count

    total = np.int64(0)
    if len_states >= 1:
        total += forward[n, len_states - 1]
    if len_states >= 2:
        total += forward[n, len_states - 2]

    return total > 0


def states_pregen(clue):
    states = [0]
    for nr in clue:
        states.extend([1] * nr)
        states.append(0)
    return np.array(states, dtype=np.int32)


class Picture:
    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.pixels = np.full((height, width), UNKNOWN, dtype=np.int32)
        self.rows_to_solve = np.ones(height, dtype=np.bool_)
        self.cols_to_solve = np.ones(width, dtype=np.bool_)
        self.solved_rows = set()
        self.solved_cols = set()

    def get_pixel(self, row, col):
        return self.pixels[row, col]

    def set_pixel(self, row, col, val):
        self.pixels[row, col] = val

    def get_row_view(self, row):
        return self.pixels[row]

    def get_col_view(self, col):
        return self.pixels[:, col]

    def get_row(self, row):
        return self.pixels[row].copy()

    def get_col(self, col):
        return self.pixels[:, col].copy()

    def is_solved(self):
        return not np.any(self.pixels == UNKNOWN)

    def copy(self):
        new_pic = Picture.__new__(Picture)
        new_pic.height = self.height
        new_pic.width = self.width
        new_pic.pixels = self.pixels.copy()
        new_pic.rows_to_solve = self.rows_to_solve.copy()
        new_pic.cols_to_solve = self.cols_to_solve.copy()
        new_pic.solved_rows = self.solved_rows.copy()
        new_pic.solved_cols = self.solved_cols.copy()
        return new_pic

    def __str__(self):
        chars = {EMPTY: '.', FULL: '#', UNKNOWN: '?'}
        return '\n'.join(''.join(chars[c] for c in row) for row in self.pixels)


class SolveStrategy(Enum):
    BASIC = "basic"
    CONTRA = "contra"
    BACKTRACK = "backtrack"


class SolveState:
    def __init__(self, print_progress=False):
        self.print_progress = print_progress
        self.depth = 0
        self.progress_bits = 0  # binary representation of progress
        self.solutions_found = 0
        self.used_contradiction = False
        self.used_backtrack = False

    def enter_backtrack(self):
        self.depth += 1
        # Append a 0 at current depth (no action needed, bit is already 0)

    def first_branch_failed(self):
        # Set bit at current depth to 1
        bit_pos = self.depth - 1
        if not (self.progress_bits & (1 << bit_pos)):
            # Bit is 0, set it to 1
            self.progress_bits |= (1 << bit_pos)
        else:
            # Bit is already 1, need to carry: clear this bit and increment above
            self._carry_from(bit_pos)

    def _carry_from(self, bit_pos):
        # Clear bit at bit_pos and propagate carry upward
        self.progress_bits &= ~(1 << bit_pos)
        if bit_pos > 0:
            parent_pos = bit_pos - 1
            if not (self.progress_bits & (1 << parent_pos)):
                self.progress_bits |= (1 << parent_pos)
            else:
                self._carry_from(parent_pos)

    def exit_backtrack(self):
        # Clear any bits at current depth and below when exiting
        bit_pos = self.depth - 1
        # Clear this bit position
        self.progress_bits &= ~(1 << bit_pos)
        self.depth -= 1

    @property
    def progress(self):
        # Calculate progress percentage from bits
        result = 0.0
        for k in range(64):  # max 64 depth levels
            if self.progress_bits & (1 << k):
                result += 100.0 * (0.5 ** (k + 1))
        return result

    def solution_found(self):
        self.solutions_found += 1

    def mark_contradiction(self):
        self.used_contradiction = True

    def mark_backtrack(self):
        self.used_backtrack = True

    def get_strategy(self):
        if self.used_backtrack:
            return SolveStrategy.BACKTRACK
        elif self.used_contradiction:
            return SolveStrategy.CONTRA
        else:
            return SolveStrategy.BASIC

    def print_state(self, pic):
        if self.print_progress:
            print(f"\n=== Progress: {self.progress:.2f}% | Depth: {self.depth} | Solutions: {self.solutions_found} ===")
            print(pic)
            print()


def solve(rows, cols, print_progress=False):
    pic = Picture(len(rows), len(cols))
    mapped_rows = [(i, states_pregen(clue)) for i, clue in enumerate(rows)]
    mapped_cols = [(i, states_pregen(clue)) for i, clue in enumerate(cols)]
    state = SolveState(print_progress)
    yield from solve_real(mapped_rows, mapped_cols, pic, state)


def solve_with_strategy(rows, cols, print_progress=False):
    pic = Picture(len(rows), len(cols))
    mapped_rows = [(i, states_pregen(clue)) for i, clue in enumerate(rows)]
    mapped_cols = [(i, states_pregen(clue)) for i, clue in enumerate(cols)]
    state = SolveState(print_progress)
    solution_count = sum(1 for _ in solve_real(mapped_rows, mapped_cols, pic, state))
    return solution_count, state.get_strategy()


def solve_one_batch(clue, index, is_col, pic):
    if is_col:
        line = pic.get_col(index)
    else:
        line = pic.get_row_view(index)

    result_line, total = solve_line_batch(line, clue)

    if total == 0:
        return False, None, None, None

    has_unknown = False
    for i in range(len(line)):
        if line[i] == UNKNOWN:
            has_unknown = True
            break

    if not has_unknown:
        if is_col:
            pic.solved_cols.add(index)
        else:
            pic.solved_rows.add(index)
        return True, None, None, None

    n = len(line)
    determined_positions = []
    determined_values = []

    for i in range(n):
        if line[i] == UNKNOWN and result_line[i] != UNKNOWN:
            determined_positions.append(i)
            determined_values.append(result_line[i])

    if not determined_positions:
        return True, None, None, None

    return True, np.array(determined_positions, dtype=np.int32), \
        np.array(determined_values, dtype=np.int32), index


def write_intersection_vectorized(positions, values, line_index, pic, is_row):
    if positions is None:
        return

    if is_row:
        row = line_index
        for i in range(len(positions)):
            col = positions[i]
            if pic.pixels[row, col] == UNKNOWN:
                pic.pixels[row, col] = values[i]
                pic.cols_to_solve[col] = True
    else:
        col = line_index
        for i in range(len(positions)):
            row = positions[i]
            if pic.pixels[row, col] == UNKNOWN:
                pic.pixels[row, col] = values[i]
                pic.rows_to_solve[row] = True


def solve_real(mapped_rows, mapped_cols, pic, state):
    if not solve_check(pic, mapped_rows, mapped_cols):
        return

    if pic.is_solved():
        state.solution_found()
        yield pic
        return

    while np.any(pic.rows_to_solve) or np.any(pic.cols_to_solve):
        if not solve_lines(mapped_rows, pic, is_row=True):
            return
        if not solve_lines(mapped_cols, pic, is_row=False):
            return
        if not solve_check(pic, mapped_rows, mapped_cols):
            return

    if pic.is_solved():
        state.solution_found()
        yield pic
        return

    state.print_state(pic)

    yield from solve_backtrack(mapped_rows, mapped_cols, pic, state)


def solve_lines(mapped, pic, is_row):
    for index, clue in mapped:
        should_solve = pic.rows_to_solve[index] if is_row else pic.cols_to_solve[index]
        if not should_solve:
            continue
        if is_row:
            pic.rows_to_solve[index] = False
        else:
            pic.cols_to_solve[index] = False

        success, positions, values, line_idx = solve_one_batch(clue, index, not is_row, pic)
        if not success:
            return False
        write_intersection_vectorized(positions, values, index, pic, is_row)
    return True


def get_neighbor_scores(pic):
    filled = (pic.pixels != UNKNOWN).astype(np.int32)
    padded = np.pad(filled, 1, constant_values=1)
    scores = (
            padded[:-2, 1:-1] +
            padded[2:, 1:-1] +
            padded[1:-1, :-2] +
            padded[1:-1, 2:]
    )
    return scores


def count_solved_pixels(pic):
    return np.count_nonzero(pic.pixels != UNKNOWN)


def probe_cell(row, col, val, mapped_rows, mapped_cols, pic):
    pic2 = pic.copy()
    pic2.set_pixel(row, col, val)
    pic2.rows_to_solve[row] = True
    pic2.cols_to_solve[col] = True

    if not solve_check(pic2, mapped_rows, mapped_cols):
        return False, 0

    while np.any(pic2.rows_to_solve) or np.any(pic2.cols_to_solve):
        if not solve_lines(mapped_rows, pic2, is_row=True):
            return False, 0
        if not solve_lines(mapped_cols, pic2, is_row=False):
            return False, 0
        if not solve_check(pic2, mapped_rows, mapped_cols):
            return False, 0

    return True, count_solved_pixels(pic2)


def solve_backtrack(mapped_rows, mapped_cols, pic, state):
    scores = get_neighbor_scores(pic)
    unknown_mask = (pic.pixels == UNKNOWN)
    unknown_coords = np.argwhere(unknown_mask)

    if len(unknown_coords) == 0:
        return

    unknown_scores = scores[unknown_mask]
    order = np.argsort(-unknown_scores)
    sorted_coords = unknown_coords[order]

    best_cell = None
    best_pixels = -1
    best_first_val = FULL

    for coord in sorted_coords:
        row, col = coord[0], coord[1]

        full_ok, full_pixels = probe_cell(row, col, FULL, mapped_rows, mapped_cols, pic)
        empty_ok, empty_pixels = probe_cell(row, col, EMPTY, mapped_rows, mapped_cols, pic)

        if not full_ok and not empty_ok:
            return

        if full_ok and not empty_ok:
            state.mark_contradiction()
            pic.set_pixel(row, col, FULL)
            pic.rows_to_solve[row] = True
            pic.cols_to_solve[col] = True
            yield from solve_real(mapped_rows, mapped_cols, pic, state)
            return

        if empty_ok and not full_ok:
            state.mark_contradiction()
            pic.set_pixel(row, col, EMPTY)
            pic.rows_to_solve[row] = True
            pic.cols_to_solve[col] = True
            yield from solve_real(mapped_rows, mapped_cols, pic, state)
            return

        max_pixels = max(full_pixels, empty_pixels)
        if max_pixels > best_pixels:
            best_pixels = max_pixels
            best_cell = (row, col)
            best_first_val = FULL if full_pixels >= empty_pixels else EMPTY

    if best_cell is None:
        return

    state.mark_backtrack()

    row, col = best_cell
    first_val = best_first_val
    second_val = EMPTY if first_val == FULL else FULL

    state.enter_backtrack()

    pic2 = pic.copy()
    pic2.set_pixel(row, col, first_val)
    pic2.rows_to_solve[row] = True
    pic2.cols_to_solve[col] = True

    found_solution = False
    for solution in solve_real(mapped_rows, mapped_cols, pic2, state):
        found_solution = True
        yield solution

    if not found_solution:
        state.first_branch_failed()

    pic2 = pic.copy()
    pic2.set_pixel(row, col, second_val)
    pic2.rows_to_solve[row] = True
    pic2.cols_to_solve[col] = True
    yield from solve_real(mapped_rows, mapped_cols, pic2, state)

    state.exit_backtrack()


def solve_check(pic, mapped_rows, mapped_cols):
    for i, clue in mapped_rows:
        if i in pic.solved_rows:
            continue
        line = pic.get_row_view(i)
        if not check_line_valid(line, clue):
            return False
        if UNKNOWN not in line:
            pic.solved_rows.add(i)

    for i, clue in mapped_cols:
        if i in pic.solved_cols:
            continue
        line = pic.get_col(i)
        if not check_line_valid(line, clue):
            return False
        if UNKNOWN not in line:
            pic.solved_cols.add(i)

    return True


def fetch_webpbn(num):
    data = {
        "go": 1,
        "id": num,
        "xml_clue": "on",
        "fmt": "xml",
        "xml_soln": "on"
    }
    url = "https://webpbn.com/export.cgi/webpbn%06i.sgriddler" % num
    try:
        text = requests.post(url, data, timeout=30).text
        return parse_clues(text)
    except Exception:
        return None


def parse_clues(xml_data):
    try:
        root = ET.parse(StringIO(xml_data)).getroot()
    except ET.ParseError:
        return None

    for color in root.findall(".//color"):
        if color.get('name') not in {'black', 'white'}:
            return None

    rows_formatted = ""
    columns_formatted = ""

    for clue in root.findall(".//clues"):
        formatted_clue = ""
        for line in clue.findall('./line'):
            counts = [count.text for count in line.findall('./count')]
            if counts:
                formatted_clue += ' '.join(counts) + '\n'
            else:
                formatted_clue += '0\n'

        if clue.get('type') == 'columns':
            columns_formatted += formatted_clue
        elif clue.get('type') == 'rows':
            rows_formatted += formatted_clue

    if not rows_formatted or not columns_formatted:
        return None

    return rows_formatted.strip() + "\n---\n" + columns_formatted.strip()


PROGRESS_FILE = "webpbn_progress.txt"
IN_PROGRESS_DIR = "nonograms/in_progress"


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


def categorize_and_save(puzzle_id, clue_text, rows, cols, solution_count, strategy, solve_time):
    time_cat = get_time_category(solve_time, rows, cols)
    strat_cat = strategy.value

    folder = f"nonograms/{time_cat}/{strat_cat}"
    os.makedirs(folder, exist_ok=True)

    filepath = f"{folder}/{puzzle_id}"
    with open(filepath, "w") as f:
        f.write(f"# {solve_time:.5f}\n")
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
    args = parser.parse_args()

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