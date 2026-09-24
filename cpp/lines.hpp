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
    // The per-cell-value transition tables of the DFA sweeps, by value
    // (EMPTY, FULL, UNKNOWN) and state word, for clues of up to four
    // words: stay = {em, 0, em}, step = {em, fm, sv}. Built here once so
    // the residual key's sweeps (millions of calls per count) read them
    // instead of assembling them on the stack per call.
    static constexpr std::size_t kTableWords = 4;
    std::uint64_t stay[3][kTableWords] = {};
    std::uint64_t step[3][kTableWords] = {};
};

LineSpec make_line_spec(const std::vector<int>& clue);

// A line's last sweeps, so its next solve resumes them past the cells that
// did not change (see solve_line_batch_1w): fwd is valid for positions
// [0, fv] and bwd for [bv, n] under the key words kept here. The arrays
// are per line, n + 1 words each, owned by the caller.
struct LineSweepMemo {
    std::uint64_t* fwd = nullptr;
    std::uint64_t* bwd = nullptr;
    std::uint64_t key[4] = {};
    std::uint32_t fv = 0, bv = 0;
    bool valid = false;
};

// Solves one line into `out` (cleared first; its capacity is reused across
// calls so a miss performs no heap allocation).
// has_unknown = false promises the line has no UNKNOWN cell; then only the
// forward validity check runs (there can be no deductions).
void solve_line_batch(const std::int8_t* line, std::size_t n,
                      const LineSpec& spec, LineSolveResult& out,
                      bool has_unknown = true);
// Cell p is line[p * stride], so a column is read in place; key is the
// line's packed key (2 bits per cell, digit 2 = UNKNOWN, one word per 32
// cells), which supplies the unknown mask without a pass over the cells.
// memo, when given with a key, resumes the one-word path's sweeps from the
// line's previous solve.
void solve_line_batch(const std::int8_t* line, std::size_t stride, std::size_t n,
                      const std::uint64_t* key,
                      const LineSpec& spec, LineSolveResult& out,
                      bool has_unknown, LineSweepMemo* memo = nullptr);
