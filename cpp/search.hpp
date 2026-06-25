#pragma once
#include <functional>
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
           bool keep_probing = false);

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
