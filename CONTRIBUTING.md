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

### The C++ solver

`cpp/` has its own gate and benchmark protocol; a performance claim about
the C++ solver should come with these numbers, measured this way:

```
cd cpp
bench/shield.sh                      # reserve one core (undo: bench/unshield.sh)
make                                 # the binary that ships: plain -O3 (see below on PGO)
python3 bench/harness.py gate        # solution count + strategy vs golden
python3 bench/harness.py gate-anytime   # same puzzles with --anytime (count only)
python3 bench/harness.py bench 3     # default-mode timing suite, best of 3
bench/run.sh ./solver ../nonograms/partially_solved/pikachu --anytime --max 300000
```

The last line is the headline anytime benchmark (time to 300k solutions
on a puzzle that never finishes). `make pgo` still builds a
profile-guided binary, but since count mode became the default it
measures slower than the plain build on that mode (medium/7382 15.1 s
against 10.3 s, easy_medium/12130 2.8 against 1.9, 10810 explores 2.6x
less in 60 s; 2026-09-18), whichever training set was tried, so the
plain build ships; re-measure before shipping PGO again. `./solver <puzzle> --balance 6` switches
the branch score to `6*min - max` of the two probe fills, which shrinks
exhaustive trees on hard unique puzzles several-fold (11-Dom 217k -> 28k
nodes) but changes where a `--max N` run on a many-solution puzzle stops;
it is off by default, and `bench/survey/` holds the puzzles that show
both effects. With it, `bench/survey/22336 --max 2` (Gettys, 99x59, which
no probing solver in the webpbn survey finishes) reaches a second solution
in 11 minutes on a loaded core; the default order was stopped at 15. Compare binaries **interleaved** on the
shielded core, several rounds each, and read best-of-N and medians; on a
shared machine a single pair of runs is noise. For a behavior-preserving
change, `bench/diff_test.py <old> <new>` runs the whole corpus through
both binaries and fails on any difference in solution count or strategy.

Two counters are far less noisy than wall time and settle most decisions:

```
perf stat -e instructions:u,branch-misses:u ./solver <puzzle> --anytime --max 100000
```

Instructions retired repeat to well under 0.1% run to run. Cycles and
wall time still have the last word, because most of the anytime hot path
is memory latency that no instruction count sees. Small changes also move
code layout enough to swing plain `-O3` builds by a few percent, so treat
a wall difference under ~5% as noise unless instructions or node counts
move with it.

`make stats` builds `solver-stats`, which with `DEBUG_CACHE_STATS=1` prints
line-cache and probe statistics to stderr (lookups, misses, deductions per
entry, lookups per probe, branch nodes). `LINE_CACHE_LEGACY=1` forces the
string-keyed cache used for lines longer than 128 cells, for differential
testing of the packed-key one. Neither affects results.

A change to the search itself (branch heuristic, probing policy) is judged
on branch-node counts, which are deterministic, rather than on wall time:

```
make stats
bench/nodes.py ./solver-stats -o bench/results/<tag>-nodes.jsonl
bench/nodes.py --compare bench/results/a-nodes.jsonl bench/results/b-nodes.jsonl
```

`nodes.py` skips `partially_solved/` and `in_progress/` (no finished
count to check), `extreme/` (fully counted puzzles that take minutes
to hours each, 3867 ten minutes, 30254 forty) and `insane/` (six hours
and up, 7785 twenty-two) by default; `--exclude`
adds categories on top of those and `--include-all` runs everything.

Two things to know when reading those numbers. The corpus's two largest
trees (`easy_large/6689`, `easy_large/3929`) run with probing shut off after the
first few nodes, so the branch heuristic barely touches them; the unique-
solution puzzles that do exercise it are `easy_large/6574`, `easy_large/5281`
and `easy_medium/8098` (9-Dom). And the survey puzzles
in `bench/survey/` have many solutions, so a `--max 2` run there measures
how soon the branch order finds a second solution, not the tree size.
`bench/nodes.py --env NAME=value` passes the stats build's experiment
knobs through; each is documented next to its definition in `search.cpp`
with the numbers that kept it out of the shipped build:

| knob | what it changes |
|---|---|
| `BRANCH_K=k` | branch score `k*min - max` instead of min with the balanced tie-break |
| `ANYTIME_TB=1/2` | anytime-mode tie-break on the max toward larger / smaller min |
| `BRANCH_MAX=1` | the anytime order (larger probe fill first) in every mode: clears dead and sparse subtrees fast (pikachu 47% of the space in 300 s against 10%) but exhaustive trees grow, corpus +24% nodes and 9x probes (9-Dom 3k -> 114k nodes), so not shipped |
| `FIRST_VAL=1` | explore the branch value whose probe settled fewer cells first |
| `NO_SKIP=1` | never latch the probing shut-off |
| `SMALL_NOPROBE=n` | branch without probing at non-root nodes with at most n unknown cells (shipped: 47; 0 disables) |
| `PROBE_WINDOW`, `PROBE_THRESH` | the shut-off's yield window and threshold |
| `DEAD_WINDOW`, `DEAD_FRAC` | the dead-work watchdog's window and fraction |
| `EARLY_SOLVE=1` | anytime: stop the probe pass at a probe that completes the grid |
| `NO_PROBE_SKIP=1` | probe every cell both ways even when an earlier probe of the same pass already bounds the fill below the best branch score (the skip is on in the shipped min-balanced and anytime-max orders; corpus probes -14.5%, counts and strategy labels unchanged) |
| `PROBE_MEMO=1` | answer a probe from its memo while none of the lines it solved (the lines of the cells it settled) has had a committed change since: same outcome and fill, tree bit-for-bit the same (corpus gate 0 mismatches, probes -19.9%). Instructions 3867 (150k nodes) -5.3%, 7382 -3.8%, 12130 -0.4%, but cycles on a quiet core (medians of 7) 3867 +0.3%, 7382 +3.6%, 12130 +3.0%: it answers the short cache-hot probes and pays memo-line and stamp loads plus a 32-byte store per probe. Not shipped |
| `DEBUG_IMPL=1/2` | implication graph with contrapositive edges: count / act |
| `NO_STATE_CACHE=1` | count mode without the region state cache (the count of a region's state, keyed by its line keys, reused when another branch order reaches it; it also serves the region split, so a split-off region seen before is a hit at its search's root) |
| `STATE_CACHE_PROBE_ONLY=1` | key and look up every node but never take a hit: the instruction delta against `NO_STATE_CACHE=1` is the cache's own cost |
| `STATE_CACHE_YIELD=<x>` | the state cache's per-node-size gate: a size bucket stops using the cache while the cycles its hits save fall below x times the cycles its lookups cost (default 1). The gate and the table's eviction read cycle counters, so a long run's tree is not bit-for-bit reproducible; on the corpus neither fires and two `nodes.py` runs agree on every puzzle (checked 2026-09-18), and `0` makes the gate inert for a strictly deterministic tree |
| `DEBUG_CACHE_STATS=1` | the counters themselves, including the latched dead/live subtree histogram and the state cache's lookups, hits and evictions |

Measured on the partially solved class (explored fraction and count after
60-300 s on one core, build 6b570c1b3) and not shipped, each a search
policy the stats build no longer carries:

- **Row-major / column-major branch order** (the row-by-row DP the state
  cache would memoize): without probing pikachu counts 1,836 solutions in
  65 s against 39M, with probing 2.65M; first-cell order walks into dead
  subtrees the most-constrained order avoids. A Python model of that DP
  holds 1.2M distinct column-state tuples after 13 of easy_medium/108's
  20 rows, more states than the puzzle has solutions.
- **Column cuts in the region split** (treating a column as independent
  segments at every boundary where its clue automaton has one viable
  state, so rows above and below multiply): exact, fires on 60% of
  pikachu's splits, but only where a plain split was near; 7785 431M
  against 860M counted in 60 s, 16900 4.2e9 against 12.7e9.
- **Probing only the K most constrained cells**: K=16 and 64 count
  nothing on pikachu and 16900 in 65 s and a fifth on 7785; the full
  pass's forced-cell detection is what keeps the search out of dead
  subtrees.
- **Trusting every probe bound** (a cell proven consistent by an earlier
  probe is never probed, the bound stands in for its fill): pikachu 9.7M
  against 39M, 7382 13.9 s against 4.8 s; recording bounds on every
  consistent probe costs more than the skips save.
- **Support-keyed nogoods** (a dead subtree's zero keyed on the lines its
  search consulted rather than the whole state): every dead subtree above
  32 nodes on 3867, 9892 and 12548 consulted 90-100% of the region's
  lines, so nothing would generalise.
- **Block restore of the picture after a probe** instead of walking the
  trail: probes settle 8-13 cells on average (3867 9.8, 7382 12.8, pikachu
  13, 23210 8.2), fewer than a block copy of the keys is worth.

Three things hold in every build. `STATE_CACHE_MB` sets the state cache's
budget (default 512; it starts small and doubles up to that). A count-mode
run prints `progress: explored X% after Ys` once a minute, so a service's
journal shows where a long run is. And SIGTERM or SIGINT stops the search
at its next branch node instead of killing the process: the run then
prints `Stopped by signal after Xs; explored Y% of the search space` and
no `Found` line, so a `timeout 300 ./solver puzzle` still says how far it
got, and two binaries can be compared on a puzzle that neither finishes by
the fraction each reaches in the same time.

**`--learn`** (count mode; off by default). The search learns a 1UIP
clause over cell literals (`cell*2 + (value==FULL)`) from every
contradiction the solve trail commits to and from every contradicting
probe. The clauses live in a two-watched-literal store (`cpp/clauses.hpp`)
and propagate inside the fixpoint loop alongside line propagation (a clause
pass after each line drain); a line's deductions are explained lazily, by
re-running the line automaton on the content it saw (`explain_deduction` /
`explain_conflict` in `lines.cpp`). Backtracking stays chronological, and
the DFS, probing, region split and state cache are untouched. A probe's
contradiction clause becomes the reason of the forced commit of the other
value.

Phase 1's limits, each with its reason:

- **Chronological backtracking**: there is no backjump, so a learned clause
  prunes the tree but never redirects the search.
- **No clause propagation inside region searches**: the state cache keys a
  region node on its lines' DFA residuals, which do not determine the known
  cells' values, so any clause action inside a region could cache a count
  under a key that does not determine it. The design's claim that a clause
  spanning two regions can never become unit inside one was false; it was
  found in review and fixed by disabling the pass there, at a measured cost
  under learning of 7382 129,308 -> 134,958 nodes and 23210 10,482,310 ->
  10,535,389.
- **No re-assertion after a revert**: a non-unit learned clause that is unit
  again after a chronological revert is not re-asserted until one of its
  watched literals is re-assigned, which only misses a propagation.
- **Unit clauses** are re-forced from the store's unit list on every pass,
  because a one-literal clause has no second literal to watch; their cells
  sit at level 0.
- **Probe clauses that miss the pixel**: 11.3% of probe-learned clauses on
  7382 (3,033 of 26,749) end on a forced probe-trail literal rather than
  the probe pixel, so those commits carry no reason; that is sound, and
  resolving past the 1UIP to the pixel would fix it (recorded, not built).
- **The tree does not only shrink**: learned clauses change which cell the
  min-balanced heuristic branches on, so the corpus loses nodes overall but
  easy_medium/8424 (56,306 -> 84,799), easy_large/12534 (7,297 -> 15,525),
  easy_large/32291 (+11) and easy_large/11820 (+1) gain. The probing-yield
  watchdog was measured as the cause and rejected: feeding it clause
  conflicts or clause-forced cells changed nothing, and disabling it under
  learning fixed three of those but grew five others and raised corpus
  probes from 179.6 million to 9.26 billion.

| knob | what it changes |
|---|---|
| `--learn` | the mode itself, in count mode |
| `LEARN=1` | stats build only: the same as `--learn`, so `bench/nodes.py --env LEARN=1` gates the mode |
| `LEARN_MAX_CLAUSES=n` | the clause store's size (default 50000; every build): `reduce()` between subtrees deletes the worst half by lbd, then activity, and never a clause that is a reason or one propagation has not examined yet |
| `LEARN_CHECK=1` | stats build: records every learned clause and checks each against every solution of a puzzle with at most 10,000 solutions, enumerated by a second, learning-off solve. `bench/learn_check.sh` runs it over the small buckets and fails on any `clause violated` line: 0 violations over easy_small, easy_medium, medium and easy_large |
| `DEBUG_CACHE_STATS=1` | adds a `cache-stats: learn conflicts= clauses= avg-len= forced-by-clauses= clause-conflicts= deleted= probe-conflicts-learned= probe-uip-elsewhere=` line |

The corpus gate under learning, `bench/nodes.py ./solver-stats --env
STATE_CACHE_YIELD=0 --env LEARN=1 --timeout 3600` on build 42636e3c7, totals
96,627,580 nodes and 74,235,479 probes with `mismatches=1`: the one mismatch
is hard/3867's timeout, and every other puzzle counts exactly. Excluding
3867 that is nodes -26.4% and probes -58.7% against the reference
(208,538,669 nodes, 712,729,214 probes over 9,867 puzzles); 3867 itself is
unverified under learning (stopped at 74.96% after 72 minutes). The gate
needs `--timeout 3600` because 23210 takes 1,651 s and 38332 1,189 s under
learning (both exact). The off path costs `MAX_NODES=150000 ./solver
../nonograms/hard/3867` 6,258,221,545 -> 6,283,610,134 instructions
(+0.41%), with the tree bit-identical and the off gate at 0 mismatches.
Instructions with `--learn` (release build, one core):

| run | off | `--learn` | ratio |
|---|---|---|---|
| hard/3867, `MAX_NODES=150000` | 6,283,610,134 | 188,610,523,766 | 30.0x |
| easy_large/7382 | 29,818,408,184 | 285,879,083,746 | 9.59x |
| easy_medium/12130 | 8,380,855,300 | 72,406,952,389 | 8.64x |
| hard/23210 | 394,964,416,746 | 5,877,227,717,081 | 14.9x |

Within 3867's budget the clauses never force a cell, because every conflict
there is inside a region search. The partially solved sweep, 300 s on one
core, explored fraction / solutions counted:

| puzzle | baseline (6b570c1b3) | committed-conflict learning | plus probe learning |
|---|---|---|---|
| 9892 | 50.037% / 722 million | 50.026% / 847,057 | 50.048% / 248,857,346 |
| 12548 | 0.000016% / 2.2 million | 0.0000207% / 2,192,128 | 0.0000767% / 30,481,024 |
| 19080 | ~0% / 61 billion | ~0% / 998,944,030 | ~0% / 990,753,501 |
| 20058 | ~0% / 1.6 billion | ~0% / 250,294,287 | ~0% / 229,666,805 |
| 13480 | 10.55% / 13.3 billion | 10.547% / 351,219,980 | 8.594% / 79,012,143 |

`--learn` stays a flag and does not ship as the default. Every corpus
puzzle measured pays 9 to 30 times the instructions, the 300 s sweep gains
no explored fraction that matters (12548's rises from 0.000016% to
0.0000767% and its count from 2.2 million to 30.5 million, while the counts
of the other four fall by a factor of 3 on 9892 and of 7 to 170 on the
rest), and four corpus puzzles gain nodes, two of them by half or more. The tree reduction (-26%
nodes, -59% probes) shows that the clauses prune; the instruction counts
show that phase 1 pays for them with an average learned clause of 70
literals and a line-automaton rerun per explained trail entry. The levers,
in the order the reviews ranked them: recursive minimisation of learned
clauses (design §3.4), sharing one line-content build across the entries of
a line during analysis, resolving probe clauses past the 1UIP to the pixel,
skipping the analysis inside region searches (sound, not tree-neutral), and
an empty-store early-out in the clause pass. Phase 2 (backjumping,
activity-based decisions) has no signal behind it yet. The design and its
post-implementation corrections are in
`cpp/bench/clause-learning-design-2026-09-23.md`.

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
