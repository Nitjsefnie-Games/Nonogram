import os
from collections import deque
from enum import Enum

import numpy as np

from lines import EMPTY, FULL, UNKNOWN


class Picture:
    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.pixels = np.full((height, width), UNKNOWN, dtype=np.int32)
        self.row_dirty = np.ones(height, dtype=np.bool_)
        self.col_dirty = np.ones(width, dtype=np.bool_)
        self.row_queue = deque(range(height))
        self.col_queue = deque(range(width))
        self.solved_rows = set()
        self.solved_cols = set()
        self.unknown_count = height * width

    def mark_row_dirty(self, idx):
        if not self.row_dirty[idx]:
            self.row_dirty[idx] = True
            self.row_queue.append(idx)

    def mark_col_dirty(self, idx):
        if not self.col_dirty[idx]:
            self.col_dirty[idx] = True
            self.col_queue.append(idx)

    def has_dirty(self):
        return bool(self.row_queue) or bool(self.col_queue)

    def get_pixel(self, row, col):
        return self.pixels[row, col]

    def set_pixel(self, row, col, val):
        old = self.pixels[row, col]
        self.pixels[row, col] = val
        if old == UNKNOWN and val != UNKNOWN:
            self.unknown_count -= 1

    def get_row_view(self, row):
        return self.pixels[row]

    def get_col_view(self, col):
        return self.pixels[:, col]

    def get_row(self, row):
        return self.pixels[row].copy()

    def get_col(self, col):
        return self.pixels[:, col].copy()

    def is_solved(self):
        return self.unknown_count == 0

    def copy(self):
        new_pic = Picture.__new__(Picture)
        new_pic.height = self.height
        new_pic.width = self.width
        new_pic.pixels = self.pixels.copy()
        new_pic.row_dirty = self.row_dirty.copy()
        new_pic.col_dirty = self.col_dirty.copy()
        new_pic.row_queue = deque(self.row_queue)
        new_pic.col_queue = deque(self.col_queue)
        new_pic.solved_rows = self.solved_rows.copy()
        new_pic.solved_cols = self.solved_cols.copy()
        new_pic.unknown_count = self.unknown_count
        return new_pic

    def __str__(self):
        chars = {EMPTY: '.', FULL: '#', UNKNOWN: '?'}
        return '\n'.join(''.join(chars[c] for c in row) for row in self.pixels)


class SolveStrategy(Enum):
    BASIC = "basic"
    CONTRA = "contra"
    BACKTRACK = "backtrack"


class SolveState:
    PROBING_WINDOW = 100        # how many recent probes feed the yield estimate
    PROBING_THRESHOLD = 0.01    # disable probing once yield drops below this
    PROBING_MIN_SOLUTIONS = int(os.environ.get('PROBING_MIN_SOLUTIONS', '2'))

    def __init__(self, print_progress=False):
        self.print_progress = print_progress
        self.depth = 0
        self.progress_bits = 0  # binary representation of progress
        self.solutions_found = 0
        self.used_contradiction = False
        self.used_backtrack = False
        self.skip_probing = False
        self._probe_outcomes = deque(maxlen=self.PROBING_WINDOW)

    def record_probe(self, found_contradiction):
        if self.skip_probing:
            return
        # Don't let the yield-window disable probing until we've already found
        # multiple solutions — that's our evidence the puzzle is multi-solution
        # and enumeration (not first-solution search) is the dominant cost.
        # Probing is what makes hard, loose puzzles tractable to begin with;
        # a low yield near the root of a hard search is normal and not a
        # signal that probing is wasted.
        if self.solutions_found < self.PROBING_MIN_SOLUTIONS:
            return
        self._probe_outcomes.append(1 if found_contradiction else 0)
        if len(self._probe_outcomes) == self.PROBING_WINDOW:
            yield_rate = sum(self._probe_outcomes) / self.PROBING_WINDOW
            if yield_rate < self.PROBING_THRESHOLD:
                self.skip_probing = True

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
