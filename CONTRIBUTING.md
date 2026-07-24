# Contributing to Nonogram

Issues and pull requests are welcome — especially if the solver got a
puzzle wrong or hung on one you expected it to finish. This is a solver, so
the two failure modes that matter are **wrong** (a returned grid that does
not satisfy the clues, or a missed solution) and **slow** (a puzzle that
should fall to propagation but falls to backtracking). A failing puzzle
file attached to the issue is worth more than a description of it.

## LLM and agent contributions are welcome

You may use an LLM or a coding agent to write your contribution. There is
no penalty, no separate review queue, and no expectation that you rewrite
its output by hand. Much of this repo was built that way.

Two conditions, and they are about honesty rather than provenance:

1. **Disclose the model** with a trailer on each commit it authored:

   ```
   Co-Authored-By: <Model Name> <noreply@example.com>
   ```

   e.g. `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`. One
   primary-author trailer per commit.

2. **Do not submit claims you have not verified.** Performance claims here
   are especially easy to get wrong: numba compiles on first call, so an
   unwarmed timing measures the compiler, not the kernel. If your PR says
   something is faster, paste the measurement and say how you warmed it.
   "Should be faster" is not evidence.

If a maintainer's reply reads like it was drafted by an agent, it probably
was. That is fine in both directions.

## The constraints

- **Correctness beats speed, always.** A faster solver that returns a grid
  violating its clues is not a faster solver. `puzzle_io.py` validates
  clues on load; keep that path honest.
- **The line kernels in `lines.py` are numba-jitted.** They must stay in
  the subset numba can compile in `nopython` mode — no Python objects, no
  dicts of mixed types, no exceptions carrying payloads in the hot path. If
  a change makes numba fall back to object mode, it is a large silent
  regression, not a style issue.
- **The solve ladder is deliberate**: line DP propagation first, then
  two-valued probing (contradiction search), then depth-first backtracking.
  Moving work down the ladder to make one puzzle faster usually makes the
  corpus slower. Show the corpus, not the one puzzle.
- **`cpp/` mirrors the Python solver.** If you change the algorithm on one
  side, say in the PR whether the other side needs the same change.

## Getting it running

Requires **Python 3.9+**:

```
pip install -r requirements.txt

python solver.py nonograms/trivial/basic/1
python solver.py path/to/puzzle --print       # stream progress + grids
python solver.py path/to/puzzle --benchmark   # measure first-solution time
```

The C++ port builds separately:

```
make -C cpp
```

## Tests and benchmarks

There is no pytest suite; the corpus under `nonograms/` is the test set and
`baseline.jsonl` is the reference. Before and after a change that touches
the solver:

```
python scripts/baseline.py          # regenerate timings against the corpus
python scripts/sum_times.py         # aggregate
```

`scripts/cpuset_setup.sh` / `cpuset_teardown.sh` pin the benchmark to
isolated cores — use them if you are reporting timings, because unpinned
numbers on a loaded machine are noise and will be treated as such.

A PR that adds a real regression suite (a handful of puzzles with known
solutions, asserted end to end) is welcome on its own.

## House style

- **Python** — numpy arrays over Python lists in anything the solver
  touches. `EMPTY` / `FULL` / `UNKNOWN` from `lines.py`, never bare
  integers.
- One responsibility per module: DP kernels in `lines.py`, grid state in
  `picture.py`, search control in `search.py`, parsing in `puzzle_io.py`.
- There is no linter or formatter config. Match the surrounding file.

## Pull requests

Small and single-purpose. Include what changed and why, the corpus result
before and after, and — for the solver — at least one puzzle that
demonstrates the difference. A bug report with a reproducing puzzle file is
worth as much as a patch and is often easier to review.
