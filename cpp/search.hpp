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
void solve(const std::vector<std::vector<int>>& rows,
           const std::vector<std::vector<int>>& cols,
           std::function<bool(const Picture&)> on_solution,
           Strategy* out_strategy = nullptr);
