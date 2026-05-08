#pragma once
#include <functional>
#include <vector>
#include "picture.hpp"

void solve(const std::vector<std::vector<int>>& rows,
           const std::vector<std::vector<int>>& cols,
           std::function<bool(const Picture&)> on_solution);
