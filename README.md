# Nonogram

A black-and-white nonogram (a.k.a. picross / griddler) solver written in Python,
plus a harvester that pulls puzzles from [webpbn.com](https://webpbn.com) and
categorizes them by difficulty and the strategy needed to solve them.

The solver hot path is JIT-compiled with [numba](https://numba.pydata.org/) and
runs forward/backward DP over each line; it falls back to two-valued probing
(contradiction search) and finally to depth-first backtracking when local
constraint propagation stalls.

## Layout

```
.
├── lines.py              # line DP kernels (numba) + EMPTY/FULL/UNKNOWN
├── picture.py            # Picture grid, SolveState, SolveStrategy
├── search.py             # solve loop, contradiction probing, backtrack
├── puzzle_io.py          # puzzle-file parser + clue validation
├── webpbn.py             # webpbn.com fetcher (library + CLI)
├── solver.py             # CLI: solve a single puzzle file
├── categorize.py         # CLI: harvest webpbn → solve → categorize
├── webpbn_progress.txt   # last webpbn ID processed by `categorize.py`
└── nonograms/            # categorized puzzle corpus
    ├── trivial/<id>          # strategy stored in `# strategy=...` header
    ├── easy_small/<id>
    ├── easy_medium/<id>
    ├── easy_large/<id>
    ├── medium/<id>
    ├── hard/<id>
    ├── extreme/<id>
    ├── insane/<id>
    └── partially_solved/
```

Module dependencies (top-down, no cycles):

```
categorize.py     solver.py
        │             │
        └─→ search.py ←┘
            │   │
            │   └─→ picture.py ──→ lines.py
            │                         ↑
            └────────────→ puzzle_io.py
        ↓
    webpbn.py
```

Each puzzle file is plain text:

```
# <optional comment, e.g. solve time in seconds>
2
2 1
1 1
3
---
1 1
2 2
1 1
2
```

Row clues come first, then `---`, then column clues. Empty lines = empty
clue (`0`). Lines starting with `#` are ignored.

## Install

Requires Python 3.9+ and:

```
pip install -r requirements.txt
```

## Solve a single puzzle

```
python solver.py nonograms/trivial/basic/1
python solver.py path/to/puzzle --print          # stream progress + grids
python solver.py path/to/puzzle --max 1          # stop after the first solution
python solver.py path/to/puzzle --benchmark      # measure first-solution time
python solver.py path/to/puzzle --benchmark --runs 10
```

By default the solver enumerates **all** solutions (so the time printed for
puzzles with many solutions includes the full enumeration). Pass `--max N` to
stop after the Nth one. Use `--print` to watch the grid fill in; `.` = empty,
`#` = filled, `?` = unknown.

## Fetch from webpbn

Pull a single black-and-white puzzle by ID:

```
python webpbn.py 1                       # print clues to stdout
python webpbn.py 1 -o nonograms/foo      # save to file
```

## Harvest from webpbn

`categorize.py` walks puzzle IDs on webpbn.com, downloads each one, solves it,
and files it under `nonograms/<difficulty>/<strategy>/<id>`.

```
python categorize.py                       # resume from webpbn_progress.txt
python categorize.py --start 1             # start over
python categorize.py --max-puzzles 100     # cap this run
python categorize.py --max-not-found 50    # stop after N consecutive misses
python categorize.py --timeout 60          # skip puzzles slower than 60s
```

Only black-and-white puzzles are accepted; colored ones are skipped. While a
puzzle is being solved the clues are written to `nonograms/in_progress/<id>`
so a crash leaves no orphaned ID.

### Difficulty categories

Bucketed by wall-clock solve time, falling back to grid area for the fast
puzzles:

| Category       | Rule                                       |
|----------------|--------------------------------------------|
| `insane`       | ≥ 6 h                                      |
| `extreme`      | ≥ 10 min                                   |
| `hard`         | ≥ 30 s                                     |
| `medium`       | ≥ 5 s                                      |
| `easy_large`   | < 5 s and > 400 cells                      |
| `easy_medium`  | < 5 s and 226–400 cells                    |
| `easy_small`   | < 5 s and 101–225 cells                    |
| `trivial`      | < 5 s and ≤ 100 cells                      |

### Strategy categories

Recorded in each puzzle file as a `# strategy=<name>` header (first line).
Set by which technique the solver actually had to reach for:

- `basic` — solved by line-level constraint propagation alone.
- `contra` — needed contradiction probing (try a value, propagate, reject if it kills the puzzle).
- `backtrack` — needed full DFS backtracking with copy-on-branch.

## Solver internals

- `lines.solve_line_batch` (numba) — forward DP from `(0, start_state)` and
  backward DP from the two accepting end states gives, for each cell, whether
  `EMPTY` and/or `FULL` is still reachable. Cells reachable by exactly one
  value are pinned. Returns `(determined_line, total_arrangement_count)`;
  `total == 0` means the line is unsatisfiable.
- `picture.Picture` — `numpy` int32 grid with per-row / per-col dirty bitmaps
  so only lines whose cells changed get re-solved.
- `search.solve_real` — drains the dirty queues, then if anything is still
  `UNKNOWN` enters `solve_backtrack`.
- `search.solve_backtrack` — orders unknown cells by neighbour density,
  probes both values:
  - if neither is consistent, the branch dies;
  - if exactly one is, that value is forced (contradiction);
  - otherwise picks the cell whose better-branch fills the most cells and
    recurses, larger branch first.
- `picture.SolveState.progress_bits` — packs the DFS progress into a 64-bit
  integer (`first_branch_failed` carries upward), so `--print` shows a
  percentage of the search tree explored.

## Notes

- First run pays a numba compilation cost; subsequent runs hit the on-disk
  cache (`__pycache__/`).
- `search.py` calls `setrecursionlimit(100000)` at import time; deep backtrack
  chains rely on it.
