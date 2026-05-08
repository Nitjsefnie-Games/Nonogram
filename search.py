from sys import setrecursionlimit

import numpy as np

from lines import EMPTY, FULL, UNKNOWN, solve_line_batch, check_line_valid, states_pregen
from picture import Picture, SolveState

__version__ = "1.6.0"

setrecursionlimit(100000)

_line_cache = {}


def _reset_line_cache():
    _line_cache.clear()


def solve(rows, cols, print_progress=False):
    _reset_line_cache()
    pic = Picture(len(rows), len(cols))
    mapped_rows = [states_pregen(clue) for clue in rows]
    mapped_cols = [states_pregen(clue) for clue in cols]
    state = SolveState(print_progress)
    yield from solve_real(mapped_rows, mapped_cols, pic, state)


def solve_with_strategy(rows, cols, print_progress=False):
    _reset_line_cache()
    pic = Picture(len(rows), len(cols))
    mapped_rows = [states_pregen(clue) for clue in rows]
    mapped_cols = [states_pregen(clue) for clue in cols]
    state = SolveState(print_progress)
    solution_count = sum(1 for _ in solve_real(mapped_rows, mapped_cols, pic, state))
    return solution_count, state.get_strategy()


def solve_one_batch(clue, index, is_col, pic):
    if is_col:
        line = pic.get_col(index)
    else:
        line = pic.get_row_view(index)

    key = (line.tobytes(), id(clue))
    cached = _line_cache.get(key)
    if cached is None:
        positions, values, total, fully_solved = solve_line_batch(line, clue)
        _line_cache[key] = (positions.copy(), values.copy(), total, fully_solved)
    else:
        positions, values, total, fully_solved = cached

    if total == 0:
        return False, None, None, None

    if fully_solved:
        if is_col:
            pic.solved_cols.add(index)
        else:
            pic.solved_rows.add(index)

    if positions.size == 0:
        return True, None, None, None

    return True, positions, values, index


def write_intersection_vectorized(positions, values, line_index, pic, is_row):
    if positions is None:
        return

    if is_row:
        row = line_index
        for i in range(len(positions)):
            col = positions[i]
            if pic.pixels[row, col] == UNKNOWN:
                pic.pixels[row, col] = values[i]
                pic.unknown_count -= 1
                pic.mark_col_dirty(col)
    else:
        col = line_index
        for i in range(len(positions)):
            row = positions[i]
            if pic.pixels[row, col] == UNKNOWN:
                pic.pixels[row, col] = values[i]
                pic.unknown_count -= 1
                pic.mark_row_dirty(row)


def solve_real(mapped_rows, mapped_cols, pic, state):
    if not solve_check(pic, mapped_rows, mapped_cols):
        return

    if pic.is_solved():
        state.solution_found()
        yield pic
        return

    while pic.has_dirty():
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
    queue = pic.row_queue if is_row else pic.col_queue
    dirty = pic.row_dirty if is_row else pic.col_dirty
    while queue:
        index = queue.popleft()
        dirty[index] = False
        success, positions, values, _ = solve_one_batch(mapped[index], index, not is_row, pic)
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
    pic2.mark_row_dirty(row)
    pic2.mark_col_dirty(col)

    if not solve_check(pic2, mapped_rows, mapped_cols):
        return False, 0

    while pic2.has_dirty():
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
                pic.mark_row_dirty(row)
                pic.mark_col_dirty(col)
                yield from solve_real(mapped_rows, mapped_cols, pic, state)
                return

            if empty_ok and not full_ok:
                state.mark_contradiction()
                pic.set_pixel(row, col, EMPTY)
                pic.mark_row_dirty(row)
                pic.mark_col_dirty(col)
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
    pic2.mark_row_dirty(row)
    pic2.mark_col_dirty(col)

    found_solution = False
    for solution in solve_real(mapped_rows, mapped_cols, pic2, state):
        found_solution = True
        yield solution

    if not found_solution:
        state.first_branch_failed()

    pic2 = pic.copy()
    pic2.set_pixel(row, col, second_val)
    pic2.mark_row_dirty(row)
    pic2.mark_col_dirty(col)
    yield from solve_real(mapped_rows, mapped_cols, pic2, state)

    state.exit_backtrack()


def solve_check(pic, mapped_rows, mapped_cols):
    for i in range(len(mapped_rows)):
        if i in pic.solved_rows:
            continue
        line = pic.get_row_view(i)
        if not check_line_valid(line, mapped_rows[i]):
            return False
        if UNKNOWN not in line:
            pic.solved_rows.add(i)

    for i in range(len(mapped_cols)):
        if i in pic.solved_cols:
            continue
        line = pic.get_col(i)
        if not check_line_valid(line, mapped_cols[i]):
            return False
        if UNKNOWN not in line:
            pic.solved_cols.add(i)

    return True
