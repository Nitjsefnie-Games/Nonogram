# Clause learning in the count-mode search — design

Audience: the successor session that plans and dispatches the implementation,
and its subagents. Agent-facing Markdown; not published.

## 1. Problem and goal

`cpp/solver` counts every solution of a nonogram by DFS with per-node probing,
a min-balanced branch heuristic, region splitting (independent components
multiply) and a state cache keyed by line residuals. On the puzzles still in
`nonograms/partially_solved/` the tree is astronomically large and, on the
few-solution ones (9892, 19080, 20058, 13480, 2647, 5903, 30509), dominated by
the same conflicts rediscovered in sibling subtrees. Measured this session
(2026-09-22/23, commit 8fb68f694 and CONTRIBUTING's "Measured … and not
shipped" list): every cheap search-policy change is a loss, and the only
technique in the literature that finishes the hard uniqueness instances
(12548, 22336) is clause learning (Copris / Lazyfd: block-position model,
order encoding, CDCL). A generic exact counter (GANAK 2.7.0 on our CNF
encoding, `cpp/bench/cnf_encode.py`) counts the corpus exactly but is 30x-7000x
slower on it because of fixed preprocessing, and needs two-hour budgets on the
stuck puzzles; its result on pikachu and 9892 is the last input to this
design (see §8).

Goal: a `--learn` mode of the existing solver that adds conflict-driven
clause learning to the count-mode search, keeps every count exact (corpus gate
0 mismatches), and is judged by explored fraction and count after 300 s on the
partially solved class plus instructions on the corpus. Shipping it as the
default is a later decision on those numbers.

## 2. What stays

Everything not named in §3: line solver and its caches, packed keys, probing
with bounds, min-balanced branching, the small-node no-probe path, region
split, the state cache with residual keys, progress and signal-stop output,
`categorize.py --record`. Learned clauses are implied by the puzzle, so:

- a region split stays exact (every combination of the regions' solutions is a
  puzzle solution and satisfies every learned clause; a clause spanning two
  regions can never become unit inside one of them because the other region's
  literals are unassigned there);
- a state-cache entry stays exact (the count of a state does not depend on
  which implied clauses are present);
- a probe stays a valid lookahead (clause propagation inside it only adds
  forced cells and earlier contradictions).

## 3. What changes

### 3.1 Reasons on the trail

Every cell set on the solve() trail (and the probe trail) records why:

```
enum ReasonKind : uint8_t { DECISION, LINE, CLAUSE, PROBE_FORCED };
struct Reason { ReasonKind kind; uint32_t id; };   // id: line (row r / col H+c), clause index
```

plus a per-cell trail index `cell_trail_pos[cell]` (UINT32_MAX when unknown),
maintained by set/unset. `write_intersection` passes the line it is applying;
the branch set is a DECISION; a forced commit from a probe pair is
PROBE_FORCED with the learned clause's id (§3.4); a clause propagation is
CLAUSE.

### 3.2 Line explanations, computed lazily

`lines.cpp` gains

```
// Smallest-found subset of the line's known cells that alone forces cell
// `pos` to `val` (explain_deduction) or makes the line unsatisfiable
// (explain_conflict). Greedy: start from all known cells, try dropping each
// (set it UNKNOWN), keep the drop when the deduction / unsat still holds.
// O(known * n) sweeps of the automaton; only called during conflict analysis.
int explain_deduction(const int8_t* line, size_t n, const LineSpec&, int pos, int8_t val, int* out_positions);
int explain_conflict (const int8_t* line, size_t n, const LineSpec&, int* out_positions);
```

The line content *at the time of the deduction* is reconstructed from the
trail: a cell of the line counts as known iff `cell_trail_pos[cell] < the
deduced cell's trail index` (the probe trail is separate; a probe's
explanation runs against the probe-time content the same way with the probe
trail's indices). The explanation is a set of (cell, value) literals; the
reason clause for a LINE deduction of `c = v` is `c=v ∨ ⋁ (cell ≠ its value)`.

Explanations are not cached in the line cache in phase 1 (they are computed
per conflict, thousands per second at most, ~10 µs each).

### 3.3 Clause store and unit propagation

New `cpp/clauses.hpp/.cpp`:

- literal encoding `lit = cell * 2 + (val == FULL)`, negation `lit ^ 1`;
- clauses in a flat arena, two watched literals per clause, watch lists per
  literal;
- `propagate(trail, from_index)`: unit propagation over cells set since
  `from_index`, sets forced cells through `Picture::set_known` + dirty marks +
  trail push with reason CLAUSE, returns the conflicting clause or none;
- undo is by the existing trail revert (watches are lazy, nothing to undo);
- database limit: keep at most `LEARN_MAX_CLAUSES` (default 50,000); when
  exceeded, delete half by LBD then activity, never a clause that is a reason
  on the current trail.

`propagate_t` becomes: drain rows, drain columns, clause-propagate the cells
set since the last clause pass, repeat until no queue and no new cells.

### 3.4 Conflict analysis

At every contradiction in count mode with `--learn`: a line returning unsat
(LINE conflict, explained by `explain_conflict`), or a clause conflict. Build
the implication graph from the trail (reasons resolved lazily via §3.2),
resolve to the first unique implication point of the current decision level,
minimize (recursive minimization over reasons), add the learned clause, bump
activities.

Levels: a decision level is opened by each branch set in `solve_backtrack`;
region searches nest inside the level of the split node; the root
propagation is level 0.

Probes: a probe pair with one contradicting value is a conflict at the
probe's level; the analysis yields a clause `¬X=v ∨ …` and the forced commit
`X=¬v` takes that clause as its PROBE_FORCED reason. A probe that contradicts
is therefore also a learning opportunity (this is Wu et al.'s FP2
contrapositive in clause form, general).

### 3.5 Backtracking (phase 1: chronological)

Phase 1 keeps the DFS exactly as it is: on a conflict the current subtree is
dead (count 0), the clause is learned, and the search returns to the parent.
The learned clauses act through propagation only. The tree can only shrink
(same decisions, more forced cells, earlier contradictions), so every count
stays exact by construction, and `bench/nodes.py` must report 0 mismatches
and, per puzzle, nodes ≤ the reference.

Phase 2 (only if phase 1 shows a large but insufficient gain): backjump to the
assertion level, discarding the partial counts of the levels jumped over
(their finished subtrees are in the state cache), the way sharpSAT does.

### 3.6 Decision heuristic

Unchanged in phase 1 (min-balanced probing, latched scan). Phase 2 option:
when probing is latched off, branch by conflict activity (VSIDS) instead of
the most-constrained cell.

## 4. Flags and knobs

- `--learn` (main.cpp): enables §3; default off.
- `LEARN_MAX_CLAUSES`, `LEARN_MIN_LBD_KEEP` env knobs (stats build documents
  them in CONTRIBUTING's table).
- `DEBUG_CACHE_STATS=1` prints: conflicts analysed, clauses learned / deleted,
  average clause length, cells forced by clauses, probe conflicts learned.

## 5. Correctness gates (every task)

1. `bench/nodes.py ./solver-stats -o … --env STATE_CACHE_YIELD=0` with
   `--learn` on: 0 mismatches over the corpus; `--compare` against the
   reference: no puzzle with more nodes.
2. Unit tests for `explain_deduction` / `explain_conflict`: for random lines,
   the returned subset alone reproduces the deduction / unsat, and dropping any
   single member of it does not (local minimality).
3. Unit tests for the clause store: propagation and conflict detection on
   hand-built clause sets, watch invariants after trail reverts.
4. Learned clauses are implied: in the stats build, `LEARN_CHECK=1` re-derives
   each learned clause by a bounded search (or at least checks it against
   every solution found) on small puzzles.

## 6. Performance metrics

- The 300 s sweep on 9892, 12548, 19080, 20058, 13480 (core 10, build
  6b570c1b3 baseline in the 2026-09-22 sweep table in this session's
  handoff): explored fraction and count.
- Corpus: `perf stat -e instructions:u` on the 3867 150k-node budget, 7382,
  12130, 23210, `--learn` on and off; `--learn` is only shipped as default if
  no corpus puzzle loses more than noise.

## 7. Task breakdown for the plan

1. Trail reasons and `cell_trail_pos` (search.cpp): no behaviour change; gate.
2. `explain_deduction` / `explain_conflict` in lines.cpp with unit tests.
3. Clause store with 2-watched-literal propagation and tests (clauses.cpp).
4. Clause propagation in the fixpoint loop and in probes; reverts; gate with
   an empty clause store (identical tree).
5. Conflict analysis at line and clause conflicts, learned-clause DB,
   `--learn` flag, stats; gate; sweep measurement.
6. Probe-pair conflicts as learning (PROBE_FORCED reasons); gate; sweep.
7. CONTRIBUTING: the mode, its knobs, its numbers.

Each task is one commit on `master` after its gate, per the repository's
practice (see CONTRIBUTING "The C++ solver").

## 8. Open input: the GANAK verdict

Two-hour GANAK runs on pikachu and 9892 were started 2026-09-23 ~01:40 CEST
(`/tmp/claude-0/-root-Nonogram/625aa262-ae05-4e18-800c-b94f7b534fae/scratchpad/ganak-*-2h.log`).
If GANAK counts either, the tree under learning plus component caching is
orders of magnitude smaller than ours and this design proceeds as written. If
both time out in the counting phase, phase 1 is still worth its measurement
(it can only shrink the tree) but phase 2 should not be started without a
new signal.

## Post-implementation corrections (2026-09-24)

- §2, region split: "a clause spanning two regions can never become unit
  inside one of them" is false. The state cache keys a region node on its
  lines' DFA residuals, which do not determine the known cells' values, so a
  clause action inside a region search can cache a count under a key that
  does not determine it. Fixed by running no clause pass inside region
  searches (commit c207b8a8a); cost under learning 7382 129,308 -> 134,958
  nodes, 23210 10,482,310 -> 10,535,389.
- §3.5, "the tree can only shrink (same decisions, more forced cells)" is
  false: the min-balanced heuristic reads the picture, so clause-forced cells
  change which cell it branches on. The corpus loses 26.4% of its nodes
  (3867 excluded) but easy_medium/8424 (56,306 -> 84,799), easy_large/12534
  (7,297 -> 15,525), easy_large/32291 (+11) and easy_large/11820 (+1) grow.
  The probing-yield watchdog was ruled out as the cause on the
  committed-conflict build (1b7389cae, where the growers were 8424, 12534
  and 12130): feeding it clause conflicts or clause-forced cells changed
  nothing, and disabling it under learning fixed those three but grew five
  others and raised that build's corpus probes from 179.6 million to 9.26
  billion.
- §3.4 as the plan built it: unit learned clauses were to be watched with
  their literal duplicated (`lits[1] = lits[0]`). The store instead keeps
  unit clauses on a unit list that every propagate call re-forces; their
  cells sit at level 0.
- §3.4's recursive minimisation and §4's `LEARN_MIN_LBD_KEEP` were never
  built: the plan omitted them. Minimisation is a listed lever in
  CONTRIBUTING; the knob does not exist.
- §5 gate 1: "no puzzle with more nodes" is a measurement, not a gate (see
  §3.5 above). `mismatches=0` is the exactness gate.

The numbers and the shipping verdict (`--learn` stays a flag) are in
CONTRIBUTING.md, "The C++ solver".
