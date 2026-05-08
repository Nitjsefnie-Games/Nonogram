#include "lines.hpp"

LineSolveResult solve_line_batch(const std::vector<std::int8_t>& line,
                                 const std::vector<std::int8_t>& states) {
    (void)line;
    (void)states;
    LineSolveResult result;
    result.total = 0;
    result.fully_solved = false;
    return result;
}

bool check_line_valid(const std::vector<std::int8_t>& line,
                      const std::vector<std::int8_t>& states) {
    (void)line;
    (void)states;
    return false;
}

std::vector<std::int8_t> states_pregen(const std::vector<int>& clue) {
    (void)clue;
    return {};
}
