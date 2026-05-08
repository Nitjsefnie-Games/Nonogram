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


# Bitset DP. The DP state at position p in the line is a bitmap over
# `len_states` (the logical state-machine positions: separators + block
# cells). forward[p] / backward[p] is a uint64 array of n_words words,
# where bit s of word w (= s // 64, s % 64) is set iff state s is
# reachable at position p.
#
# Transitions per cell at position p with value v:
#   v = UNKNOWN: stay in EMPTY-valued states; advance to any state.
#   v = EMPTY:   stay in EMPTY-valued states; advance into EMPTY states.
#   v = FULL:    no stay; advance into FULL states only.
#
# In bitset form (single word):
#   stay_bits      = current & empty_mask
#   advance_to_X   = (current << 1) & X_mask
# where X is one of {state_valid_mask (all valid states), empty_mask,
# full_mask} depending on the cell value.
#
# Multi-word shifts use a carry chain across consecutive words.


@njit(cache=True)
def _shift_left_1(src, dst, n_words):
    carry = np.uint64(0)
    for w in range(n_words):
        next_carry = src[w] >> np.uint64(63)
        dst[w] = (src[w] << np.uint64(1)) | carry
        carry = next_carry


@njit(cache=True)
def _shift_right_1(src, dst, n_words):
    carry = np.uint64(0)
    for w in range(n_words - 1, -1, -1):
        next_carry = (src[w] & np.uint64(1)) << np.uint64(63)
        dst[w] = (src[w] >> np.uint64(1)) | carry
        carry = next_carry


@njit(cache=True)
def _build_masks(states, len_states, n_words):
    state_valid = np.zeros(n_words, dtype=np.uint64)
    empty_mask = np.zeros(n_words, dtype=np.uint64)
    full_mask = np.zeros(n_words, dtype=np.uint64)
    for s in range(len_states):
        bit = np.uint64(1) << np.uint64(s % 64)
        w = s // 64
        state_valid[w] |= bit
        if states[s] == EMPTY:
            empty_mask[w] |= bit
        else:
            full_mask[w] |= bit
    return state_valid, empty_mask, full_mask


@njit(cache=True)
def solve_line_batch(line, states):
    n = len(line)
    len_states = len(states)
    n_words = (len_states + 63) // 64

    state_valid, empty_mask, full_mask = _build_masks(states, len_states, n_words)

    forward = np.zeros((n + 1, n_words), dtype=np.uint64)
    forward[0, 0] = np.uint64(1)

    cur = np.zeros(n_words, dtype=np.uint64)
    shifted = np.zeros(n_words, dtype=np.uint64)

    for p in range(n):
        for w in range(n_words):
            cur[w] = forward[p, w]

        cell = line[p]
        if cell == UNKNOWN:
            _shift_left_1(cur, shifted, n_words)
            for w in range(n_words):
                forward[p + 1, w] = (cur[w] & empty_mask[w]) | (shifted[w] & state_valid[w])
        elif cell == EMPTY:
            _shift_left_1(cur, shifted, n_words)
            for w in range(n_words):
                forward[p + 1, w] = (cur[w] & empty_mask[w]) | (shifted[w] & empty_mask[w])
        else:
            _shift_left_1(cur, shifted, n_words)
            for w in range(n_words):
                forward[p + 1, w] = shifted[w] & full_mask[w]

    # Accept: bit (len_states - 1) and (optionally) bit (len_states - 2)
    accept_w1 = (len_states - 1) // 64
    accept_b1 = np.uint64(1) << np.uint64((len_states - 1) % 64)
    reachable = (forward[n, accept_w1] & accept_b1) != 0
    if not reachable and len_states >= 2:
        accept_w2 = (len_states - 2) // 64
        accept_b2 = np.uint64(1) << np.uint64((len_states - 2) % 64)
        reachable = (forward[n, accept_w2] & accept_b2) != 0

    if not reachable:
        return (np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.int32),
                np.int64(0),
                False)

    # Backward DP. Initial state: bits len_states-1 and len_states-2 set.
    backward = np.zeros((n + 1, n_words), dtype=np.uint64)
    aw1 = (len_states - 1) // 64
    backward[n, aw1] |= np.uint64(1) << np.uint64((len_states - 1) % 64)
    if len_states >= 2:
        aw2 = (len_states - 2) // 64
        backward[n, aw2] |= np.uint64(1) << np.uint64((len_states - 2) % 64)

    masked = np.zeros(n_words, dtype=np.uint64)

    for p in range(n - 1, -1, -1):
        for w in range(n_words):
            cur[w] = backward[p + 1, w]

        cell = line[p]
        if cell == UNKNOWN:
            _shift_right_1(cur, shifted, n_words)
            for w in range(n_words):
                backward[p, w] = (cur[w] & empty_mask[w]) | shifted[w]
        elif cell == EMPTY:
            for w in range(n_words):
                masked[w] = cur[w] & empty_mask[w]
            _shift_right_1(masked, shifted, n_words)
            for w in range(n_words):
                backward[p, w] = masked[w] | shifted[w]
        else:
            for w in range(n_words):
                masked[w] = cur[w] & full_mask[w]
            _shift_right_1(masked, shifted, n_words)
            for w in range(n_words):
                backward[p, w] = shifted[w]

    # Determine each unknown cell.
    positions = np.empty(n, dtype=np.int32)
    values = np.empty(n, dtype=np.int32)
    n_changed = 0
    n_unknown_total = 0

    bw_empty = np.zeros(n_words, dtype=np.uint64)
    bw_full = np.zeros(n_words, dtype=np.uint64)
    bw_empty_shifted = np.zeros(n_words, dtype=np.uint64)
    bw_full_shifted = np.zeros(n_words, dtype=np.uint64)
    can_empty_buf = np.zeros(n_words, dtype=np.uint64)
    can_full_buf = np.zeros(n_words, dtype=np.uint64)

    for p in range(n):
        if line[p] != UNKNOWN:
            continue
        n_unknown_total += 1

        for w in range(n_words):
            bw_empty[w] = backward[p + 1, w] & empty_mask[w]
            bw_full[w] = backward[p + 1, w] & full_mask[w]
        _shift_right_1(bw_empty, bw_empty_shifted, n_words)
        _shift_right_1(bw_full, bw_full_shifted, n_words)

        can_empty = False
        can_full = False
        for w in range(n_words):
            f = forward[p, w]
            can_empty_buf[w] = f & (bw_empty[w] | bw_empty_shifted[w])
            can_full_buf[w] = f & bw_full_shifted[w]
            if can_empty_buf[w] != 0:
                can_empty = True
            if can_full_buf[w] != 0:
                can_full = True

        if can_empty and not can_full:
            positions[n_changed] = p
            values[n_changed] = EMPTY
            n_changed += 1
        elif can_full and not can_empty:
            positions[n_changed] = p
            values[n_changed] = FULL
            n_changed += 1

    fully_solved = (n_unknown_total == n_changed)
    return positions[:n_changed], values[:n_changed], np.int64(1), fully_solved


@njit(cache=True)
def check_line_valid(line, states):
    n = len(line)
    len_states = len(states)
    n_words = (len_states + 63) // 64

    state_valid, empty_mask, full_mask = _build_masks(states, len_states, n_words)

    forward = np.zeros((n + 1, n_words), dtype=np.uint64)
    forward[0, 0] = np.uint64(1)

    cur = np.zeros(n_words, dtype=np.uint64)
    shifted = np.zeros(n_words, dtype=np.uint64)

    for p in range(n):
        for w in range(n_words):
            cur[w] = forward[p, w]

        cell = line[p]
        if cell == UNKNOWN:
            _shift_left_1(cur, shifted, n_words)
            for w in range(n_words):
                forward[p + 1, w] = (cur[w] & empty_mask[w]) | (shifted[w] & state_valid[w])
        elif cell == EMPTY:
            _shift_left_1(cur, shifted, n_words)
            for w in range(n_words):
                forward[p + 1, w] = (cur[w] & empty_mask[w]) | (shifted[w] & empty_mask[w])
        else:
            _shift_left_1(cur, shifted, n_words)
            for w in range(n_words):
                forward[p + 1, w] = shifted[w] & full_mask[w]

    accept_w1 = (len_states - 1) // 64
    accept_b1 = np.uint64(1) << np.uint64((len_states - 1) % 64)
    if (forward[n, accept_w1] & accept_b1) != 0:
        return True
    if len_states >= 2:
        accept_w2 = (len_states - 2) // 64
        accept_b2 = np.uint64(1) << np.uint64((len_states - 2) % 64)
        if (forward[n, accept_w2] & accept_b2) != 0:
            return True
    return False
