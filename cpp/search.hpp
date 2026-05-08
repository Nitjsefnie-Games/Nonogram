#pragma once
#include <functional>
#include <vector>
#include "picture.hpp"

// Run the solver. For each solution found, on_solution(pic) is called.
// Returning true from the callback continues the search; returning false
// aborts immediately and solve() returns.
void solve(const std::vector<std::vector<int>>& rows,
           const std::vector<std::vector<int>>& cols,
           std::function<bool(const Picture&)> on_solution);
