#pragma once
#include <cstdint>
#include <vector>

struct LineSolveResult {
    std::vector<int> positions;
    std::vector<std::int8_t> values;
    int total;       // 0 = unsat, 1 = sat
    bool fully_solved;
};

// Precomputed, puzzle-lifetime line specification: the DFA states for a clue
// plus the bit-masks derived from them. Masks depend only on the clue, so they
// are built once per row/col instead of on every line-solve call.
struct LineSpec {
    std::vector<std::int8_t> states;
    std::vector<std::uint64_t> state_valid;
    std::vector<std::uint64_t> empty_mask;
    std::vector<std::uint64_t> full_mask;
    std::size_t len_states = 0;
    std::size_t n_words = 0;
};

LineSpec make_line_spec(const std::vector<int>& clue);

LineSolveResult solve_line_batch(const std::vector<std::int8_t>& line,
                                 const LineSpec& spec);

bool check_line_valid(const std::vector<std::int8_t>& line,
                      const LineSpec& spec);
