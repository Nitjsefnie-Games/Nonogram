#pragma once
#include <cstdint>
#include <vector>

// A deduced cell is packed into a single int: the cell index in the low bits,
// the determined value (EMPTY=0 / FULL=1) in bit 30. Keeps the per-entry payload
// to one vector instead of two (one fewer allocation, smaller cache entries).
constexpr int DEDUCE_VAL_SHIFT = 30;
constexpr int DEDUCE_POS_MASK = (1 << DEDUCE_VAL_SHIFT) - 1;
inline int deduce_pack(int pos, std::int8_t val) { return pos | (static_cast<int>(val) << DEDUCE_VAL_SHIFT); }
inline int deduce_pos(int enc) { return enc & DEDUCE_POS_MASK; }
inline std::int8_t deduce_val(int enc) { return static_cast<std::int8_t>(enc >> DEDUCE_VAL_SHIFT); }

struct LineSolveResult {
    std::vector<int> deductions;  // packed (pos, val); see deduce_pack
    int total;       // 0 = unsat, 1 = sat
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

LineSolveResult solve_line_batch(const std::int8_t* line, std::size_t n,
                                 const LineSpec& spec);
