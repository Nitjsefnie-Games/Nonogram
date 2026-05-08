import numpy as np
from numba import njit

EMPTY = 0
FULL = 1
UNKNOWN = 2


def check_line(clue, size):
    if len(clue) == 1:
        return clue[0] <= size
    return sum(clue) + len(clue) - 1 <= size


def states_pregen(clue):
    states = [0]
    for nr in clue:
        states.extend([1] * nr)
        states.append(0)
    return np.array(states, dtype=np.int32)


@njit(cache=True)
def solve_line_batch(line, states):
    n = len(line)
    len_states = len(states)

    forward = np.zeros((n + 1, len_states), dtype=np.bool_)
    forward[0, 0] = True

    for pos in range(n):
        val = line[pos]
        for state in range(len_states):
            if not forward[pos, state]:
                continue

            cur_state_val = states[state]
            next_state = state + 1
            next_state_val = states[next_state] if next_state < len_states else -1

            if val == UNKNOWN:
                if cur_state_val == EMPTY:
                    forward[pos + 1, state] = True
                if next_state_val != -1:
                    forward[pos + 1, next_state] = True
            elif val == EMPTY:
                if cur_state_val == EMPTY:
                    forward[pos + 1, state] = True
                if next_state_val == EMPTY:
                    forward[pos + 1, next_state] = True
            else:
                if next_state_val == FULL:
                    forward[pos + 1, next_state] = True

    reachable = False
    if len_states >= 1 and forward[n, len_states - 1]:
        reachable = True
    if len_states >= 2 and forward[n, len_states - 2]:
        reachable = True

    if not reachable:
        return (np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.int32),
                np.int64(0),
                False)

    backward = np.zeros((n + 1, len_states), dtype=np.bool_)
    if len_states >= 1:
        backward[n, len_states - 1] = True
    if len_states >= 2:
        backward[n, len_states - 2] = True

    for pos in range(n - 1, -1, -1):
        val = line[pos]
        for state in range(len_states):
            cur_state_val = states[state]
            next_state = state + 1
            next_state_val = states[next_state] if next_state < len_states else -1

            if val == UNKNOWN:
                if cur_state_val == EMPTY:
                    backward[pos, state] |= backward[pos + 1, state]
                if next_state_val != -1:
                    backward[pos, state] |= backward[pos + 1, next_state]
            elif val == EMPTY:
                if cur_state_val == EMPTY:
                    backward[pos, state] |= backward[pos + 1, state]
                if next_state_val == EMPTY:
                    backward[pos, state] |= backward[pos + 1, next_state]
            else:
                if next_state_val == FULL:
                    backward[pos, state] |= backward[pos + 1, next_state]

    positions = np.empty(n, dtype=np.int32)
    values = np.empty(n, dtype=np.int32)
    n_changed = 0
    n_unknown_total = 0

    for pos in range(n):
        if line[pos] != UNKNOWN:
            continue
        n_unknown_total += 1

        can_empty = False
        can_full = False

        for state in range(len_states):
            if not forward[pos, state]:
                continue

            cur_state_val = states[state]
            next_state = state + 1
            next_state_val = states[next_state] if next_state < len_states else -1

            if cur_state_val == EMPTY and backward[pos + 1, state]:
                can_empty = True
            if next_state_val == EMPTY and backward[pos + 1, next_state]:
                can_empty = True
            if next_state_val == FULL and backward[pos + 1, next_state]:
                can_full = True

            if can_empty and can_full:
                break

        if can_empty and not can_full:
            positions[n_changed] = pos
            values[n_changed] = EMPTY
            n_changed += 1
        elif can_full and not can_empty:
            positions[n_changed] = pos
            values[n_changed] = FULL
            n_changed += 1

    fully_solved = (n_unknown_total == n_changed)
    return positions[:n_changed], values[:n_changed], np.int64(1), fully_solved


@njit(cache=True)
def check_line_valid(line, states):
    n = len(line)
    len_states = len(states)

    forward = np.zeros((n + 1, len_states), dtype=np.bool_)
    forward[0, 0] = True

    for pos in range(n):
        val = line[pos]
        for state in range(len_states):
            if not forward[pos, state]:
                continue

            cur_state_val = states[state]
            next_state = state + 1
            next_state_val = states[next_state] if next_state < len_states else -1

            if val == UNKNOWN:
                if cur_state_val == EMPTY:
                    forward[pos + 1, state] = True
                if next_state_val != -1:
                    forward[pos + 1, next_state] = True
            elif val == EMPTY:
                if cur_state_val == EMPTY:
                    forward[pos + 1, state] = True
                if next_state_val == EMPTY:
                    forward[pos + 1, next_state] = True
            else:
                if next_state_val == FULL:
                    forward[pos + 1, next_state] = True

    if len_states >= 1 and forward[n, len_states - 1]:
        return True
    if len_states >= 2 and forward[n, len_states - 2]:
        return True
    return False
