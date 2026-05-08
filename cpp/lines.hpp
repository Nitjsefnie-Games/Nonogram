#pragma once
#include <cstdint>
#include <vector>

struct LineSolveResult {
    std::vector<int> positions;
    std::vector<std::int8_t> values;
    int total;       // 0 = unsat, 1 = sat
    bool fully_solved;
};

LineSolveResult solve_line_batch(const std::vector<std::int8_t>& line,
                                 const std::vector<std::int8_t>& states);

bool check_line_valid(const std::vector<std::int8_t>& line,
                      const std::vector<std::int8_t>& states);

std::vector<std::int8_t> states_pregen(const std::vector<int>& clue);
