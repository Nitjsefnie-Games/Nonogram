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
    int id = -1;  // index in the solve()'s spec pool; identifies the clue in cache keys
};

LineSpec make_line_spec(const std::vector<int>& clue);

// Solves one line into `out` (cleared first; its capacity is reused across
// calls so a miss performs no heap allocation).
// has_unknown = false promises the line has no UNKNOWN cell; then only the
// forward validity check runs (there can be no deductions).
void solve_line_batch(const std::int8_t* line, std::size_t n,
                      const LineSpec& spec, LineSolveResult& out,
                      bool has_unknown = true);

// Neither is re-entrant (static thread-local scratch): never call one from inside its own predicate.
// Positions (ascending) of a subset of the line's known cells that alone
// forces cell `pos` to `val`; returns the count. `line` must currently
// force pos = val (pos is UNKNOWN in it). out has room for n ints.
int explain_deduction(const std::int8_t* line, std::size_t n, const LineSpec& spec, int pos, std::int8_t val, int* out);
// Positions of a subset of the known cells that alone makes the line
// unsatisfiable; `line` must be unsatisfiable. Returns the count.
int explain_conflict(const std::int8_t* line, std::size_t n, const LineSpec& spec, int* out);
