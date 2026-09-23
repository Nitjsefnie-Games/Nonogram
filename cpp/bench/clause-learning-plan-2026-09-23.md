# Clause Learning (phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `--learn` mode to `cpp/solver`'s count-mode search that learns a clause from every contradiction and propagates learned clauses alongside line propagation, with every count still exact.

**Architecture:** Reasons and decision levels are recorded on the existing trails; a line's deductions are explained lazily by re-running the line automaton on the content it saw; a two-watched-literal clause store propagates inside the existing fixpoint loop and inside probes; a 1UIP analysis at every contradiction adds a clause. Backtracking stays chronological (the DFS, region split and state cache are untouched), so the tree can only shrink and the corpus gate must report 0 mismatches and no puzzle with more nodes.

**Tech Stack:** C++17, `g++ -O3 -march=native -flto`, the existing `cpp/Makefile`; tests as a `make test` target building `cpp/tests/*.cpp`; gates `cpp/bench/nodes.py` and `perf stat`.

**Spec:** `cpp/bench/clause-learning-design-2026-09-23.md` (read it first; §2 says what must not change, §5 the gates, §6 the metrics).

## Global Constraints

- Every commit lands on `master`, one task per commit, message in the repository's shape (subject `solver: …`, body with the numbers), trailer `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`; stage explicit paths.
- `.gitignore` is deny-by-default: a new file needs its path named back (`cpp/tests/` is not yet allowed; Task 2 adds `!/cpp/tests/`, `/cpp/tests/*`, `!/cpp/tests/*.cpp`).
- The shipped build must not change when `--learn` is off: every new hook is behind `g_learn` (a `const bool` read once) or a `constexpr false` in the default build; report `perf stat -e instructions:u` on `MAX_NODES=150000 ./solver ../nonograms/hard/3867` before and after each task (reference 6.25G; drift beyond 1% needs a folded hook).
- Exactness gate after Tasks 1, 4, 5, 6: `python3 bench/nodes.py ./solver-stats -o /tmp/<tag>-nodes.jsonl --env STATE_CACHE_YIELD=0` (with `--env LEARN=1` from Task 5 on, see Task 5 for how the env flag maps to `--learn`), then `python3 bench/nodes.py --compare <reference>.jsonl <tag>.jsonl`: `mismatches=0`, and no puzzle with more nodes except hard/3867, hard/23210, hard/30532, hard/38332 by a few thousand (their state cache evicts by measured cycles). The reference is the same command on a stats build of `master` before Task 1; keep its jsonl.
- Never run more than one solver at a time on core 11 (a count unit may be running there); use cores 3-10 with `taskset -c N`.
- No other solver of ours may be running while timing anything.
- One action per shell command; long runs go to the background and their output files are read afterwards.

---

## File structure

- `cpp/search.cpp` (existing, 3,000 lines): trail reasons and levels, clause propagation in the fixpoint, conflict analysis at the three contradiction sites, `--learn` plumbing, stats. Modified in Tasks 1, 4, 5, 6.
- `cpp/lines.cpp` / `cpp/lines.hpp` (existing): `explain_deduction`, `explain_conflict`. Task 2.
- `cpp/clauses.hpp` / `cpp/clauses.cpp` (new): the clause store, watches, propagation, deletion. Task 3.
- `cpp/tests/test_explain.cpp`, `cpp/tests/test_clauses.cpp` (new): unit tests; `make test` in `cpp/Makefile`. Tasks 2, 3.
- `cpp/main.cpp`: `--learn` flag. Task 5.
- `CONTRIBUTING.md`: the mode and its numbers. Task 7.

Trail entries stay `int` packed `(row << 16) | col` in `Trail::changed_cell_indices`; parallel vectors carry the reason and level so the hot revert loop over the packed ints is unchanged.

---

### Task 1: Reasons and levels on the trails

**Files:**
- Modify: `cpp/search.cpp` — `struct Trail` (search: `struct Trail {`), `write_intersection_impl`, `probe_cell`, `revert_branch`, `ProbeGuard::~ProbeGuard`, the branch loop and forced commit in `solve_backtrack`, `solve()` init.

**Interfaces:**
- Produces:
  ```cpp
  enum ReasonKind : std::uint8_t { R_DECISION = 0, R_LINE = 1, R_CLAUSE = 2, R_PROBE = 3 };
  struct Reason { std::uint8_t kind; std::uint8_t is_col; std::uint16_t level; std::uint32_t id; };
  // Trail gains:
  std::vector<Reason> reasons;        // parallel to changed_cell_indices
  std::vector<std::uint32_t> cell_pos; // H*W: index into changed_cell_indices, UINT32_MAX if not on this trail
  std::uint16_t level = 0;            // current decision level (solve trail: branch depth incl. splits; probe trail: solve level + 1)
  void push(int packed, Reason r);    // push_back on both vectors, cell_pos[cell] = index
  ```
  `R_LINE` id = line index (`is_col` says which array); `R_CLAUSE` / `R_PROBE` id = clause id (Task 3/6); `R_DECISION` id unused.
- Consumes: nothing new.

- [ ] **Step 1: Add the types and fields**

In `search.cpp` just above `struct Trail`:

```cpp
enum ReasonKind : std::uint8_t { R_DECISION = 0, R_LINE = 1, R_CLAUSE = 2, R_PROBE = 3 };
struct Reason { std::uint8_t kind; std::uint8_t is_col; std::uint16_t level; std::uint32_t id; };
```

In `struct Trail` add the three fields and:

```cpp
    int W = 0;  // for cell_pos indexing
    void push(int packed, Reason r) {
        cell_pos[static_cast<std::size_t>(trail_row(packed)) * W + trail_col(packed)] = static_cast<std::uint32_t>(changed_cell_indices.size());
        changed_cell_indices.push_back(packed);
        reasons.push_back(r);
    }
```

`trail_pack/trail_row/trail_col` are defined right after `Trail`; move those three inline functions above `struct Trail`.

- [ ] **Step 2: Route every push through `push`**

Sites (grep `changed_cell_indices.push_back`): `write_intersection_impl` (reason `{R_LINE, is_row ? 0 : 1, trail.level, line_index}`), `probe_cell`'s probe pixel (`{R_DECISION, 0, trail.level, 0}`), the forced commit in the probing pass (`{R_PROBE, 0, trail.level, 0}` for now; Task 6 fills the id), the branch pixel (`{R_DECISION, 0, trail.level, 0}`), the `g_debug_impl >= 2` commit (`{R_DECISION, 0, trail.level, 0}`).

Levels: in the branch loop of `solve_backtrack`, `++trail.level` before `pic.set_known(row, col, val)` and `--trail.level` after `revert_branch`; in `probe_cell`, the probe trail's `level` is set to `solve_level + 1` at entry — pass the solve trail's level in: add a `std::uint16_t solve_level` parameter to `probe_cell` (all three callers are in `solve_backtrack` and `estimate_dive`; the dive passes 0).

- [ ] **Step 3: Reverts clear `cell_pos` and pop `reasons`**

In `revert_branch` both loops: after `pic.unset(r, c)` add `trail.cell_pos[r * trail.W + c] = UINT32_MAX;`; after `changed_cell_indices.resize(mark)` add `trail.reasons.resize(mark)`. Same in `ProbeGuard::~ProbeGuard` (it clears the probe trail's entries; the probe trail is a `static thread_local Trail` in `probe_cell`: size its `cell_pos` to H*W and set `W` on first use, `if (trail.cell_pos.size() != H*W) trail.cell_pos.assign(H*W, UINT32_MAX)`).

- [ ] **Step 4: Initialise in `solve()`**

Next to `trail.row_unknown.assign(...)`: `trail.W = W; trail.cell_pos.assign(H * W, UINT32_MAX); trail.reasons.reserve(H * W); trail.level = 0;`.

- [ ] **Step 5: Build, check exactness and cost**

```
make -C /root/Nonogram/cpp solver stats
taskset -c 5 ./solver ../nonograms/easy_medium/108      # Found 564,468
taskset -c 5 ./solver ../nonograms/easy_large/7382       # Found 13,265,283
MAX_NODES=150000 taskset -c 5 perf stat -e instructions:u ./solver ../nonograms/hard/3867   # "Found 6", instructions within 1% of 6.25G
```
If the instruction count rose more than 1%, the `push` helper is not inlined or `cell_pos` writes miss: check `-fopt-info-inline` on `push`.

- [ ] **Step 6: Gate**

```
taskset -c 4 python3 bench/nodes.py ./solver-stats -o /tmp/claude-0/-root-Nonogram/<session>/scratchpad/t1-nodes.jsonl --env STATE_CACHE_YIELD=0
python3 bench/nodes.py --compare <reference>.jsonl /tmp/…/t1-nodes.jsonl
```
Expected: `mismatches=0`, nodes equal except the four cycle-evicting hard puzzles.

- [ ] **Step 7: Commit**

```
git add cpp/search.cpp
git commit -m "solver: every trail entry carries its reason and decision level"
```

---

### Task 2: Line explanations

**Files:**
- Modify: `cpp/lines.hpp`, `cpp/lines.cpp`, `cpp/Makefile`, `.gitignore`
- Create: `cpp/tests/test_explain.cpp`

**Interfaces:**
- Produces:
  ```cpp
  // Positions (ascending) of a subset of the line's known cells that alone
  // forces cell `pos` to `val`; returns the count. `line` must currently
  // force pos = val (pos is UNKNOWN in it). out has room for n ints.
  int explain_deduction(const std::int8_t* line, std::size_t n, const LineSpec& spec, int pos, std::int8_t val, int* out);
  // Positions of a subset of the known cells that alone makes the line
  // unsatisfiable; `line` must be unsatisfiable. Returns the count.
  int explain_conflict(const std::int8_t* line, std::size_t n, const LineSpec& spec, int* out);
  ```
- Consumes: `solve_line_batch`, `LineSpec`, `deduce_pos/deduce_val`.

- [ ] **Step 1: Write the failing tests**

`cpp/tests/test_explain.cpp`:

```cpp
#include "../lines.hpp"
#include "../types.hpp"
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

static bool forces(const std::vector<std::int8_t>& line, const LineSpec& spec, int pos, std::int8_t val) {
    LineSolveResult r;
    solve_line_batch(line.data(), line.size(), spec, r, true);
    if (r.total == 0) return false;
    for (int d : r.deductions) if (deduce_pos(d) == pos && deduce_val(d) == val) return true;
    return false;
}
static bool unsat(const std::vector<std::int8_t>& line, const LineSpec& spec) {
    LineSolveResult r;
    solve_line_batch(line.data(), line.size(), spec, r, true);
    return r.total == 0;
}

int main() {
    std::mt19937 rng(7);
    int checked = 0, conflicts = 0;
    for (int iter = 0; iter < 3000; ++iter) {
        const int n = 5 + static_cast<int>(rng() % 26);
        // random clue that fits
        std::vector<int> clue;
        int used = 0;
        while (true) {
            int b = 1 + static_cast<int>(rng() % 4);
            if (used + b + (clue.empty() ? 0 : 1) > n) break;
            used += b + (clue.empty() ? 0 : 1);
            clue.push_back(b);
            if (rng() % 3 == 0) break;
        }
        LineSpec spec = make_line_spec(clue);
        // random partial content from a random solution, then some cells flipped
        std::vector<std::int8_t> line(n, UNKNOWN);
        for (int i = 0; i < n; ++i) if (rng() % 3 == 0) line[i] = (rng() % 2) ? FULL : EMPTY;
        if (unsat(line, spec)) {
            std::vector<int> out(n);
            const int k = explain_conflict(line.data(), n, spec, out.data());
            std::vector<std::int8_t> sub(n, UNKNOWN);
            for (int i = 0; i < k; ++i) sub[out[i]] = line[out[i]];
            if (!unsat(sub, spec)) { std::printf("FAIL conflict subset not unsat\n"); return 1; }
            for (int i = 0; i < k; ++i) {  // local minimality
                std::vector<std::int8_t> s2 = sub; s2[out[i]] = UNKNOWN;
                if (unsat(s2, spec)) { std::printf("FAIL conflict subset not minimal\n"); return 1; }
            }
            ++conflicts;
            continue;
        }
        LineSolveResult r;
        solve_line_batch(line.data(), n, spec, r, true);
        for (int d : r.deductions) {
            const int pos = deduce_pos(d); const std::int8_t val = deduce_val(d);
            std::vector<int> out(n);
            const int k = explain_deduction(line.data(), n, spec, pos, val, out.data());
            std::vector<std::int8_t> sub(n, UNKNOWN);
            for (int i = 0; i < k; ++i) sub[out[i]] = line[out[i]];
            if (!forces(sub, spec, pos, val)) { std::printf("FAIL deduction subset does not force\n"); return 1; }
            for (int i = 0; i < k; ++i) {
                std::vector<std::int8_t> s2 = sub; s2[out[i]] = UNKNOWN;
                if (forces(s2, spec, pos, val)) { std::printf("FAIL deduction subset not minimal\n"); return 1; }
            }
            ++checked;
        }
    }
    std::printf("ok: %d deductions, %d conflicts explained\n", checked, conflicts);
    return 0;
}
```

Add to `cpp/Makefile`:

```make
TESTS = $(wildcard tests/*.cpp)
test: $(TESTS:tests/%.cpp=tests/%)
	for t in $(TESTS:tests/%.cpp=tests/%); do ./$$t || exit 1; done
tests/%: tests/%.cpp lines.cpp picture.cpp $(wildcard *.hpp)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -I. -o $@ $< lines.cpp picture.cpp
.PHONY: test
```
(Task 3 adds `clauses.cpp` to that link line.) Add to `.gitignore` under the C++ block: `!/cpp/tests/`, `/cpp/tests/*`, `!/cpp/tests/*.cpp`, and check with `git check-ignore -v cpp/tests/test_explain.cpp` (must print nothing) and `git status --porcelain` (only the intended files).

- [ ] **Step 2: Run it to see it fail**

`make -C /root/Nonogram/cpp test` — expected: link error, `explain_deduction` undefined.

- [ ] **Step 3: Implement**

`lines.hpp`: the two declarations from Interfaces. `lines.cpp`:

```cpp
namespace {
bool line_forces(std::int8_t* buf, std::size_t n, const LineSpec& spec, int pos, std::int8_t val) {
    static thread_local LineSolveResult r;
    solve_line_batch(buf, n, spec, r, true);
    if (r.total == 0) return false;
    for (int d : r.deductions) if (deduce_pos(d) == pos && deduce_val(d) == val) return true;
    return false;
}
bool line_unsat(std::int8_t* buf, std::size_t n, const LineSpec& spec) {
    static thread_local LineSolveResult r;
    solve_line_batch(buf, n, spec, r, true);
    return r.total == 0;
}
// Greedy: drop the known cells farthest from `pos` first (the ones least
// likely to matter), keeping a drop when the property still holds.
template <class Holds>
int explain_greedy(const std::int8_t* line, std::size_t n, int pos, int* out, Holds holds) {
    static thread_local std::vector<std::int8_t> buf;
    buf.assign(line, line + n);
    std::vector<int> known;
    for (int i = 0; i < static_cast<int>(n); ++i) if (buf[i] != UNKNOWN && i != pos) known.push_back(i);
    std::sort(known.begin(), known.end(), [pos](int a, int b) { return std::abs(a - pos) > std::abs(b - pos); });
    for (int i : known) {
        const std::int8_t saved = buf[i];
        buf[i] = UNKNOWN;
        if (!holds(buf.data())) buf[i] = saved;
    }
    int k = 0;
    for (int i = 0; i < static_cast<int>(n); ++i) if (buf[i] != UNKNOWN && i != pos) out[k++] = i;
    return k;
}
}  // namespace

int explain_deduction(const std::int8_t* line, std::size_t n, const LineSpec& spec, int pos, std::int8_t val, int* out) {
    return explain_greedy(line, n, pos, out, [&](std::int8_t* b) { return line_forces(b, n, spec, pos, val); });
}
int explain_conflict(const std::int8_t* line, std::size_t n, const LineSpec& spec, int* out) {
    return explain_greedy(line, n, -1, out, [&](std::int8_t* b) { return line_unsat(b, n, spec); });
}
```
For `explain_conflict` the sort key with `pos = -1` orders by position descending; that is fine (any order is sound; a smarter order is a later measurement).

- [ ] **Step 4: Run the tests**

`make -C /root/Nonogram/cpp test` — expected `ok: N deductions, M conflicts explained` with both counts in the thousands.

- [ ] **Step 5: Commit**

```
git add cpp/lines.hpp cpp/lines.cpp cpp/Makefile cpp/tests/test_explain.cpp .gitignore
git commit -m "lines: explain a deduction or a contradiction by a minimal subset of the known cells"
```

---

### Task 3: Clause store with two watched literals

**Files:**
- Create: `cpp/clauses.hpp`, `cpp/clauses.cpp`, `cpp/tests/test_clauses.cpp`
- Modify: `cpp/Makefile` (`SRCS += clauses.cpp`, the test link line)

**Interfaces:**
- Produces:
  ```cpp
  // Literal: cell * 2 + (value == FULL). neg(l) = l ^ 1. A literal is TRUE
  // when the cell is known with that value, FALSE when known with the
  // other, else unassigned.
  class ClauseStore {
  public:
      void init(int n_cells, std::size_t max_clauses);
      int  add(const int* lits, int n, int lbd);   // returns the clause id, -1 if refused (n < 1). lits[0] and lits[1] become the watches.
      int  size() const;                            // live clauses
      // Propagate: `assigned(l)` tells a literal's state (1 true, -1 false, 0 free);
      // `force(l, clause_id)` must make literal l true (set the cell, mark lines
      // dirty, push the trail entry with reason R_CLAUSE id) and return false if
      // the cell was already known with the other value. Visits, in order, the
      // literals of `newly_true` (the trail's cells since the last call, as
      // literals). Returns the conflicting clause id, or -1.
      template <class Assigned, class Force>
      int propagate(const int* newly_true, int n_new, Assigned assigned, Force force);
      const int* lits(int id, int* n) const;
      int  lbd(int id) const;
      void bump(int id);                            // activity for deletion
      void lock(int id, bool locked);               // a reason on a trail is never deleted
      // Delete half of the unlocked clauses with the highest lbd (ties: lowest activity)
      // when size() > max_clauses; watches are rebuilt for the survivors. Ids of survivors are stable.
      void reduce();
  };
  ```
- Consumes: nothing from the solver (kept independent so the test needs no Picture).

Representation: `std::vector<int> arena` (clauses as `[n, lit0, lit1, ...]` at offset `start[id]`), `std::vector<std::vector<int>> watches` indexed by literal (list of clause ids watching that literal), `std::vector<int> lbd_`, `std::vector<float> act_`, `std::vector<char> locked_, dead_`, a free list of dead ids. Watch scheme: a clause watches `lits[0]` and `lits[1]`; when the watched literal `w` becomes FALSE, for each clause in `watches[w]`: make sure the false literal is at index 1; if `lits[0]` is true, keep; else find a non-false literal at index ≥ 2, swap it into index 1, move the watch (remove from `watches[w]`, add to `watches[new]`); if none: if `lits[0]` is false → conflict (return id); else force `lits[0]` with reason id.

- [ ] **Step 1: Write the failing test**

`cpp/tests/test_clauses.cpp` builds `ClauseStore` over 6 cells with a simple assignment array `val[6]` (−1 unknown, 0 EMPTY, 1 FULL), an `assigned` lambda, and a `force` lambda that sets `val` and appends the literal to a `trail` vector, then checks:

```cpp
// (a) unit propagation chain: clauses {1,2}, {~2,3}, {~3,4}: assert ~1 -> propagate -> 2,3,4 true, no conflict.
// (b) conflict: add {~4} ... propagate -> returns that clause's id.
// (c) after "revert" (val back to -1 for the chain), propagating the same literal again yields the same result (watches self-repair).
// (d) reduce(): add 20 clauses with lbd 2..21, lock two of them, reduce() with max_clauses = 10 -> size() <= 10, locked ones survive, lits(id) of survivors unchanged.
```
Print `ok` and return 0 on success, 1 with a message on the first failure.

- [ ] **Step 2: Run it to see it fail** — `make -C /root/Nonogram/cpp test`: compile error, no `clauses.hpp`.

- [ ] **Step 3: Implement `clauses.hpp` / `clauses.cpp`** per the representation above. `propagate` is a template in the header. Keep `reduce()` out of line.

- [ ] **Step 4: Run the tests** — both test binaries print `ok`.

- [ ] **Step 5: Commit**

```
git add cpp/clauses.hpp cpp/clauses.cpp cpp/tests/test_clauses.cpp cpp/Makefile
git commit -m "solver: a clause store with two watched literals (not yet wired in)"
```

---

### Task 4: Clause propagation in the fixpoint loop and in probes

**Files:**
- Modify: `cpp/search.cpp` — `propagate_t`, `probe_cell`, `revert_branch`, `ProbeGuard`, `solve()`; `cpp/search.hpp` (`solve()` gains `bool learn = false` after `out_count`); `cpp/main.cpp` (`--learn` sets it; usage text).

**Interfaces:**
- Consumes: `ClauseStore` (Task 3), `Trail::push` (Task 1).
- Produces: `const bool g_learn` semantics: `solve(..., learn=true)` sets a namespace-scope `bool g_learn_mode` read by the hooks; the stats build also honours `LEARN=1` in the environment so `bench/nodes.py --env LEARN=1` gates the mode. `Trail` gains `std::size_t clause_next = 0` (first trail index not yet clause-propagated). A global `ClauseStore g_clauses` and `int g_conflict_clause = -1`, `int g_conflict_line = -1; bool g_conflict_line_is_col`.

- [ ] **Step 1: Record which line failed**

In `solve_lines`, where `!r.success` returns false, first set `g_conflict_line = index; g_conflict_line_is_col = !is_row; g_conflict_clause = -1;`.

- [ ] **Step 2: Clause pass in `propagate_t`**

```cpp
template <bool FAST, int KW>
bool propagate_t(..., Trail& trail) {
    for (;;) {
        while (pic.has_dirty()) {
            if (!solve_lines<FAST, KW>(mapped_rows, pic, true, trail)) return false;
            if (!solve_lines<FAST, KW>(mapped_cols, pic, false, trail)) return false;
        }
        if (!g_learn_mode || trail.clause_next >= trail.changed_cell_indices.size()) return true;
        if (!clause_pass(pic, trail)) return false;   // sets g_conflict_clause
        // clause_pass may have set cells (lines dirty) -> loop
    }
}
```
`clause_pass` builds the literal list from `trail.changed_cell_indices[clause_next..)` (literal = (r*W+c)*2 + (pixel == FULL)), advances `clause_next`, calls `g_clauses.propagate` with `assigned` reading `pic.pixels` and `force` doing `pic.set_known(r, c, v); pic.mark_row_dirty(r); pic.mark_col_dirty(c); trail.push(trail_pack(r, c), {R_CLAUSE, 0, trail.level, id})` (returning false if the cell is known with the other value, which the store reports as a conflict). Cells forced by clauses are also new trail entries, so the next iteration of the outer loop propagates them.

- [ ] **Step 3: Reverts**

`revert_branch`: after resizing, `trail.clause_next = std::min(trail.clause_next, mark)`. `ProbeGuard::~ProbeGuard`: `trail.clause_next = 0` (the probe trail is emptied). Also, in `probe_cell` at entry, the probe trail's `clause_next` is 0 and the solve trail's `clause_next` must equal its size (the node is at a fixpoint) — assert in the stats build.

- [ ] **Step 4: Probes see the solve trail's clauses**

Clause propagation inside a probe runs on the probe trail only (literals set by the probe); the watch structure is global, so the probe's forced cells are found normally. Nothing else to do; note it in a comment.

- [ ] **Step 5: `--learn` plumbing**

`search.hpp`: add `bool learn = false` parameter; `solve()` sets `g_learn_mode = learn || (stats build: getenv("LEARN"))`, and when on: `g_clauses.init(H * W, max_clauses)` with `max_clauses` from `LEARN_MAX_CLAUSES` (default 50000). `main.cpp`: parse `--learn`, pass it, add one usage line: `--learn  Learn a clause from every contradiction (count mode)`.

- [ ] **Step 6: Build, verify identical behaviour, gate**

With no clauses ever added, `--learn` must give the same tree: `DEBUG_CACHE_STATS=1 ./solver-stats ../nonograms/easy_large/7382 --learn` and without: same `branch-nodes=` and count. Instructions on the 3867 budget without `--learn` within 1% of 6.25G. Gate (`--env LEARN=1`): 0 mismatches, nodes unchanged.

- [ ] **Step 7: Commit** — `git add cpp/search.cpp cpp/search.hpp cpp/main.cpp cpp/Makefile` then `git commit -m "solver: --learn wires clause propagation into the fixpoint loop (no clauses learned yet)"`.

---

### Task 5: Conflict analysis at committed contradictions

**Files:**
- Modify: `cpp/search.cpp` — new functions `explain_entry`, `analyze_conflict`, calls at the two committed contradiction sites (`solve_real`'s failed propagate after a branch set; the forced commit's failed propagate in the probing pass), stats print.

**Interfaces:**
- Consumes: Tasks 1-4.
- Produces:
  ```cpp
  // The antecedent literals of trail entry `idx` of `trail` as literals that
  // are TRUE on the trail (their negations form the reason clause with the
  // entry's own literal). R_LINE: explain_deduction on the line's content
  // at the time (cells of the line with cell_pos < idx on this trail, plus,
  // for the probe trail, every cell of the solve trail). R_CLAUSE / R_PROBE:
  // the clause's other literals. R_DECISION: none.
  void explain_entry(const Trail& trail, const Trail* solve_trail, std::size_t idx, const Picture& pic, ..., std::vector<int>& out_true_lits);
  // 1UIP analysis of the current conflict (g_conflict_line or g_conflict_clause)
  // at the top of `trail`; adds the learned clause to g_clauses and returns its
  // id, or -1 if the conflict is at level 0 (nothing to learn).
  int analyze_conflict(Trail& trail, const Trail* solve_trail, Picture& pic, const std::vector<const LineSpec*>& rows, const std::vector<const LineSpec*>& cols);
  ```

- [ ] **Step 1: `explain_entry`**

For `R_LINE`: build the line's content as of index `idx`: for each cell of the line, known iff its `cell_pos` on this trail `< idx` (for the probe trail also every cell known on the solve trail, i.e. `solve_trail->cell_pos[cell] != UINT32_MAX`), else UNKNOWN; then `explain_deduction(content, n, spec, pos, val, tmp)` and emit the literal `cell*2 + (content[p] == FULL)` for each returned position. For `R_CLAUSE`/`R_PROBE`: every literal of the clause other than the entry's own literal, negated (they are false on the trail; emit them as the true literals `lit ^ 1`).

- [ ] **Step 2: `analyze_conflict`**

```cpp
// conflict literals (all TRUE on the trail):
//   line conflict: explain_conflict on the line's current content -> literals
//   clause conflict: the clause's literals, each negated (all false)
// then standard 1UIP:
std::vector<int> learnt;  int counter = 0;  seen[] per cell (generation-stamped)
mark(lit): cell = lit >> 1; entry = position of cell on the relevant trail; level = reasons[entry].level;
   if level == current level: if !seen: seen, ++counter
   else if level > 0: if !seen: seen, learnt.push_back(lit ^ 1)    // negation of a true literal at a lower level
for each conflict literal: mark(it)
walk the trail from the top down (probe trail first, then the solve trail when in a probe):
   entry e; if !seen[cell(e)] continue;
   --counter; if counter == 0: uip = literal of e; break;
   explain_entry(e) -> antecedents; for each: mark(it)
learnt.insert(learnt.begin(), uip ^ 1);
lbd = number of distinct levels among learnt[1..]
id = g_clauses.add(learnt.data(), learnt.size(), lbd + 1)
```
Unit clauses (learnt.size() == 1) are added too (the store must accept n == 1 and propagate them at the next pass: a unit clause is watched with its single literal duplicated; handle in Task 3's `add` by allowing `lits[1] = lits[0]`). If `current level == 0` return -1 without adding. After `add`, if `g_clauses.size() > max`, call `reduce()` **only at a branch node between subtrees** (the branch loop, after `revert_branch`), never mid-propagation.

- [ ] **Step 3: Call sites**

In `solve_real`: `if (!propagate(...)) { if (g_learn_mode) analyze_conflict(trail, nullptr, ...); state.result = 0; return true; }`. In the probing pass's forced commit: `if (!propagate(...)) { if (g_learn_mode) analyze_conflict(...); return true; }`. Both are conflicts of the solve trail at the node's level.

- [ ] **Step 4: Stats**

Counters `g_stat_conflicts`, `g_stat_learnt`, `g_stat_learnt_lits`, `g_stat_clause_forced`, `g_stat_clause_conflicts`, printed in the `DEBUG_CACHE_STATS` block as `cache-stats: learn conflicts=… clauses=… avg-len=… forced-by-clauses=… clause-conflicts=…`.

- [ ] **Step 5: Verify implied clauses (stats build)**

`LEARN_CHECK=1`: after `solve()` returns on a puzzle with ≤ 10,000 solutions, re-run a plain count with every learned clause asserted as hard constraints… simpler and sufficient: in the stats build keep every solution's grid when `LEARN_CHECK=1` and a `--print`-style callback is possible, and at the end check that every learned clause is satisfied by every solution; do it on `easy_medium/108` (564,468 solutions is too many to store: use `nonograms/easy_small/*` and `trivial/*`, which have few) — add a `bench/learn_check.sh` that runs the stats build over `easy_small` with `LEARN_CHECK=1 LEARN=1` and fails on any `clause violated` line.

- [ ] **Step 6: Gate and measure**

Gate with `--env LEARN=1`: 0 mismatches; `--compare` against the reference: no puzzle with more nodes; report the total node and probe deltas. Then the sweep: for each of 9892 12548 19080 20058 13480: `taskset -c 10 timeout 300 ./solver ../nonograms/partially_solved/<p> --learn > /tmp/…/learn-<p>.log 2>&1`, read the last `progress:` line, compare with the baseline table (9892 50.037% / 722M; 12548 0.000016% / 2.2M; 19080 ~0% / 61G; 20058 ~0% / 1.6G; 13480 10.55% / 13.3G).

- [ ] **Step 7: Commit** — `solver: --learn learns a 1UIP clause from every committed contradiction` with the gate and sweep numbers in the body.

---

### Task 6: Probe-pair conflicts as learning

**Files:**
- Modify: `cpp/search.cpp` — `probe_cell` (return the conflict), the probing pass (analysis in the probe context, `R_PROBE` reason id).

- [ ] **Step 1:** When a probe's propagate fails and `g_learn_mode`, call `analyze_conflict(probe_trail, &solve_trail, ...)` before the guard reverts (the guard is a local RAII object: run the analysis inside `probe_cell` right after the failed propagate, store the returned clause id in a new `ProbeResult::learnt_clause` field, -1 otherwise).

- [ ] **Step 2:** In the probing pass, the forced commit uses `{R_PROBE, 0, trail.level, static_cast<std::uint32_t>(learnt_id)}` when the contradicting probe learned a clause; if the analysis returned -1 (level-0 conflict) keep `R_DECISION`-like handling: reason `{R_PROBE, 0, level, UINT32_MAX}` and `explain_entry` treats `UINT32_MAX` as "no antecedents" (sound: the cell is then treated like a decision, which only makes learned clauses longer, never wrong).

- [ ] **Step 3:** When both probes of a cell contradict, both clauses are learned and the node returns dead as before.

- [ ] **Step 4:** Gate (`--env LEARN=1`, 0 mismatches, no puzzle with more nodes) and the same sweep as Task 5; report both.

- [ ] **Step 5: Commit** — `solver: --learn learns from every contradicting probe, and forced cells carry that clause as their reason`.

---

### Task 7: CONTRIBUTING

- [ ] Add `--learn` to "The C++ solver": what it does, the phase-1 limits (chronological backtracking, unchanged decisions), the knobs `LEARN_MAX_CLAUSES` / `LEARN=1` (stats build) / `LEARN_CHECK=1`, the gate numbers from Task 6, the sweep table before/after, and the corpus instruction deltas on the 3867 budget, 7382, 12130, 23210 with `--learn` on. State plainly whether it ships as the default (only if no corpus puzzle loses more than noise) or stays a flag.
- [ ] Commit: `git add CONTRIBUTING.md` — `docs: the --learn mode, its knobs and its numbers`.

---

## Self-review notes

- Spec §3.1-§3.5 map to Tasks 1-6; §3.6 (VSIDS) is phase 2 and intentionally absent; §4 knobs are in Tasks 4/5/7; §5 gates are in every task's last steps; §6 metrics in Tasks 5-7.
- Names used across tasks: `Trail::push`, `Reason{kind,is_col,level,id}`, `R_DECISION/R_LINE/R_CLAUSE/R_PROBE`, `explain_deduction/explain_conflict`, `ClauseStore::{init,add,propagate,lits,lbd,bump,lock,reduce,size}`, `g_learn_mode`, `g_clauses`, `g_conflict_line/_is_col`, `g_conflict_clause`, `Trail::clause_next`, `explain_entry`, `analyze_conflict`, `ProbeResult::learnt_clause`.
- Locking: `lock(id, true)` when a clause becomes a reason (`force` in the clause pass, the `R_PROBE` commit); `lock(id, false)` in the revert loops when popping an entry whose reason is that clause (both `revert_branch` and the probe guard walk the reasons for this; the packed-int loop stays, the reasons vector is read alongside).
