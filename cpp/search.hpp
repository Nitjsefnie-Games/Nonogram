#pragma once
#include <functional>
#include <string>
#include <vector>
#include "picture.hpp"

// Strategy classification mirroring picture.SolveStrategy in Python.
// Priority: BACKTRACK > CONTRA > BASIC.
enum class Strategy {
    BASIC,
    CONTRA,
    BACKTRACK,
};

// Run the solver. For each solution found, on_solution(pic) is called.
// Returning true from the callback continues the search; returning false
// aborts immediately and solve() returns.
//
// If out_strategy is non-null, *out_strategy is written before solve() returns,
// reflecting which technique was needed during the search.
// keep_probing: "anytime" mode. When true, the adaptive probing-shutoff is
// disabled so lookahead pruning stays active for the whole search. This makes
// otherwise-walling deep enumerations (e.g. pikachu) keep finding solutions,
// at the cost of per-node probing overhead on easy puzzles. Default false
// preserves the original adaptive behavior exactly.
void solve(const std::vector<std::vector<int>>& rows,
           const std::vector<std::vector<int>>& cols,
           std::function<bool(const Picture&)> on_solution,
           Strategy* out_strategy = nullptr,
           bool keep_probing = false,
           double balance_k = 0.0,
           bool count_mode = false,
           std::string* out_count = nullptr);
// count_mode: count solutions instead of visiting them. on_solution is never
// called; the exact total is written to *out_count in decimal. After
// propagation at every branch node the unknown cells are split into regions
// that share no row or column, each region is counted on its own with the
// same search, and the counts multiply (every full solution is one choice
// per region), so solution-rich puzzles whose ambiguity is spread over
// separate patches are counted without visiting each combination.
// balance_k > 0 scores a branch cell by balance_k * min(f, e) - max(f, e)
// (f, e = cells settled by probing it FULL / EMPTY) instead of min(f, e)
// with ties toward the smaller max. Exhaustive searches on hard unique
// puzzles get much smaller trees (K=6: 11-Dom 217k -> 28k nodes, 12-Dom
// from over 15 minutes to 115s); on many-solution puzzles a --max N run
// stops at a different point of a different tree, sooner or later.
// 0 keeps the default order. Anytime mode ignores it.

// Fraction (0..1) of the search space the running or finished solve() has
// completed: every branch node halves the space, and completed subtrees are
// summed. solutions_so_far / explored_fraction() is a running estimate of
// the total count, exact in the fraction and extrapolating a uniform
// solution density over the rest. Valid from inside the on_solution
// callback and after solve() returns.
double explored_fraction();

// Knuth-style estimate of the TOTAL number of solutions, without enumerating.
// Performs n_dives random weighted root-to-leaf dives (Knuth 1975): at each
// branching node it counts the viable child values, multiplies a running weight
// by that count, and descends into a uniformly-random viable child. The mean of
// the per-dive weights (weight if the dive reaches a full solution, 0 if it
// dead-ends) is an unbiased estimate of the solution count. High variance, so
// it gives an order of magnitude. Writes diagnostics (hit rate, max weight) to
// stderr. Returns the mean estimate.
double estimate_solutions(const std::vector<std::vector<int>>& rows,
                          const std::vector<std::vector<int>>& cols,
                          long n_dives,
                          unsigned long seed);
