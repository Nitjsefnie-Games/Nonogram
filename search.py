from sys import setrecursionlimit

import numpy as np

from lines import EMPTY, FULL, UNKNOWN, solve_line_batch, check_line_valid, states_pregen
from picture import Picture, SolveState

__version__ = "1.1.0"

setrecursionlimit(100000)


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

    return True, count_solved_pixels(pic2)


def solve_backtrack(mapped_rows, mapped_cols, pic, state):
    scores = get_neighbor_scores(pic)
    unknown_mask = (pic.pixels == UNKNOWN)
    unknown_coords = np.argwhere(unknown_mask)

    if len(unknown_coords) == 0:
        return

    unknown_scores = scores[unknown_mask]

    if state.skip_probing:
        best_idx = int(np.argmax(unknown_scores))
        best_cell = (int(unknown_coords[best_idx][0]), int(unknown_coords[best_idx][1]))
        best_first_val = FULL
    else:
        order = np.argsort(-unknown_scores)
        sorted_coords = unknown_coords[order]

        best_cell = None
        best_pixels = -1
        best_first_val = FULL

        for coord in sorted_coords:
            row, col = coord[0], coord[1]

            full_ok, full_pixels = probe_cell(row, col, FULL, mapped_rows, mapped_cols, pic)
            empty_ok, empty_pixels = probe_cell(row, col, EMPTY, mapped_rows, mapped_cols, pic)

            state.record_probe((not full_ok) or (not empty_ok))

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
