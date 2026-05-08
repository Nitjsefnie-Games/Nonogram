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
