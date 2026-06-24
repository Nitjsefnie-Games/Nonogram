#include "lines.hpp"
#include "types.hpp"

#include <cstdint>
#include <cstddef>
#include <vector>

namespace {

inline void shift_left_1(const std::uint64_t* src, std::uint64_t* dst, std::size_t n_words) {
    std::uint64_t carry = 0;
    for (std::size_t w = 0; w < n_words; ++w) {
        std::uint64_t next_carry = src[w] >> 63;
        dst[w] = (src[w] << 1) | carry;
        carry = next_carry;
    }
}

inline void shift_right_1(const std::uint64_t* src, std::uint64_t* dst, std::size_t n_words) {
    std::uint64_t carry = 0;
    // Iterate from high word down to 0; use w+1 then decrement to avoid
    // unsigned underflow on the loop variable.
    for (std::size_t w = n_words; w-- > 0; ) {
        std::uint64_t next_carry = (src[w] & 1ULL) << 63;
        dst[w] = (src[w] >> 1) | carry;
        carry = next_carry;
    }
}

// Reusable per-thread scratch buffers for the line DP. The solver is
// single-threaded per process; thread_local keeps it correct if that ever
// changes. Buffers grow monotonically and are reused across calls so the hot
// path performs no heap allocation.
struct LineScratch {
    std::vector<std::uint64_t> forward;
    std::vector<std::uint64_t> backward;
    std::vector<std::uint64_t> cur;
    std::vector<std::uint64_t> shifted;
    std::vector<std::uint64_t> masked;
    std::vector<std::uint64_t> bw_empty;
    std::vector<std::uint64_t> bw_full;
    std::vector<std::uint64_t> bw_empty_shifted;
    std::vector<std::uint64_t> bw_full_shifted;

    void ensure(std::size_t n_rows, std::size_t n_words) {
        if (forward.size() < n_rows * n_words) forward.resize(n_rows * n_words);
        if (backward.size() < n_rows * n_words) backward.resize(n_rows * n_words);
        if (cur.size() < n_words) {
            cur.resize(n_words);
            shifted.resize(n_words);
            masked.resize(n_words);
            bw_empty.resize(n_words);
            bw_full.resize(n_words);
            bw_empty_shifted.resize(n_words);
            bw_full_shifted.resize(n_words);
        }
    }
};

thread_local LineScratch g_scratch;

} // namespace

LineSpec make_line_spec(const std::vector<int>& clue) {
    LineSpec spec;
    spec.states.push_back(EMPTY);
    for (int nr : clue) {
        for (int i = 0; i < nr; ++i) {
            spec.states.push_back(FULL);
        }
        spec.states.push_back(EMPTY);
    }
    spec.len_states = spec.states.size();
    spec.n_words = (spec.len_states + 63) / 64;

    spec.state_valid.assign(spec.n_words, 0);
    spec.empty_mask.assign(spec.n_words, 0);
    spec.full_mask.assign(spec.n_words, 0);
    for (std::size_t s = 0; s < spec.len_states; ++s) {
        std::uint64_t bit = 1ULL << (s % 64);
        std::size_t w = s / 64;
        spec.state_valid[w] |= bit;
        if (spec.states[s] == EMPTY) {
            spec.empty_mask[w] |= bit;
        } else {
            spec.full_mask[w] |= bit;
        }
    }
    return spec;
}

// Specialized single-word path (len_states <= 64). Avoids the per-word loops
// and carry-propagating shifts of the general routine; shifts are plain <<1/>>1.
// Behavior is identical to the general path for n_words == 1.
namespace {
LineSolveResult solve_line_batch_1w(const std::int8_t* line, std::size_t n,
                                    const LineSpec& spec) {
    const std::size_t len_states = spec.len_states;
    const std::uint64_t sv = spec.state_valid[0];
    const std::uint64_t em = spec.empty_mask[0];
    const std::uint64_t fm = spec.full_mask[0];

    LineSolveResult result;
    result.total = 0;
    result.fully_solved = false;

    g_scratch.ensure(n + 1, 1);
    std::uint64_t* fwd = g_scratch.forward.data();
    std::uint64_t* bwd = g_scratch.backward.data();

    fwd[0] = 1ULL;
    for (std::size_t p = 0; p < n; ++p) {
        const std::uint64_t cur = fwd[p];
        const std::uint64_t sh = cur << 1;
        const std::int8_t cell = line[p];
        std::uint64_t nxt;
        if (cell == UNKNOWN)      nxt = (cur & em) | (sh & sv);
        else if (cell == EMPTY)   nxt = (cur & em) | (sh & em);
        else                      nxt = sh & fm;
        fwd[p + 1] = nxt;
    }

    std::uint64_t accept = 1ULL << (len_states - 1);
    if (len_states >= 2) accept |= 1ULL << (len_states - 2);
    if ((fwd[n] & accept) == 0) {
        return result;  // unsat
    }

    bwd[n] = accept;
    for (std::size_t pi = n; pi-- > 0; ) {
        const std::uint64_t cur = bwd[pi + 1];
        const std::int8_t cell = line[pi];
        std::uint64_t bp;
        if (cell == UNKNOWN)      { bp = (cur & em) | (cur >> 1); }
        else if (cell == EMPTY)   { const std::uint64_t m = cur & em; bp = m | (m >> 1); }
        else                      { bp = (cur & fm) >> 1; }
        bwd[pi] = bp;
    }

    result.deductions.reserve(n);
    int n_changed = 0, n_unknown_total = 0;
    for (std::size_t p = 0; p < n; ++p) {
        if (line[p] != UNKNOWN) continue;
        ++n_unknown_total;
        const std::uint64_t bw = bwd[p + 1];
        const std::uint64_t f = fwd[p];
        const std::uint64_t bw_empty = bw & em;
        const std::uint64_t bw_full = bw & fm;
        const bool can_empty = (f & (bw_empty | (bw_empty >> 1))) != 0;
        const bool can_full = (f & (bw_full >> 1)) != 0;
        if (can_empty && !can_full) {
            result.deductions.push_back(deduce_pack(static_cast<int>(p), EMPTY));
            ++n_changed;
        } else if (can_full && !can_empty) {
            result.deductions.push_back(deduce_pack(static_cast<int>(p), FULL));
            ++n_changed;
        }
    }
    result.total = 1;
    result.fully_solved = (n_unknown_total == n_changed);
    return result;
}
}  // namespace

LineSolveResult solve_line_batch(const std::int8_t* line, std::size_t n,
                                 const LineSpec& spec) {
    const std::size_t len_states = spec.len_states;
    const std::size_t n_words = spec.n_words;

    LineSolveResult result;
    result.total = 0;
    result.fully_solved = false;

    if (len_states == 0 || n_words == 0) {
        // Degenerate: no states means no clue at all (empty puzzle line).
        return result;
    }

    if (n_words == 1) {
        return solve_line_batch_1w(line, n, spec);
    }

    const std::uint64_t* state_valid = spec.state_valid.data();
    const std::uint64_t* empty_mask = spec.empty_mask.data();
    const std::uint64_t* full_mask = spec.full_mask.data();

    g_scratch.ensure(n + 1, n_words);
    std::uint64_t* forward = g_scratch.forward.data();
    std::uint64_t* cur = g_scratch.cur.data();
    std::uint64_t* shifted = g_scratch.shifted.data();

    // forward[0] = seed; rows 1..n are fully overwritten by the loop, so only
    // the seed row needs explicit zeroing.
    for (std::size_t w = 0; w < n_words; ++w) forward[w] = 0;
    forward[0] = 1ULL; // bit 0 of word 0 of forward[0]

    for (std::size_t p = 0; p < n; ++p) {
        std::uint64_t* fwd_p = forward + p * n_words;
        std::uint64_t* fwd_next = forward + (p + 1) * n_words;
        for (std::size_t w = 0; w < n_words; ++w) {
            cur[w] = fwd_p[w];
        }

        std::int8_t cell = line[p];
        if (cell == UNKNOWN) {
            shift_left_1(cur, shifted, n_words);
            for (std::size_t w = 0; w < n_words; ++w) {
                fwd_next[w] = (cur[w] & empty_mask[w]) | (shifted[w] & state_valid[w]);
            }
        } else if (cell == EMPTY) {
            shift_left_1(cur, shifted, n_words);
            for (std::size_t w = 0; w < n_words; ++w) {
                fwd_next[w] = (cur[w] & empty_mask[w]) | (shifted[w] & empty_mask[w]);
            }
        } else { // FULL
            shift_left_1(cur, shifted, n_words);
            for (std::size_t w = 0; w < n_words; ++w) {
                fwd_next[w] = shifted[w] & full_mask[w];
            }
        }
    }

    // Accept check: bit (len_states - 1) or bit (len_states - 2) of forward[n].
    const std::uint64_t* fwd_n = forward + n * n_words;
    std::size_t accept_w1 = (len_states - 1) / 64;
    std::uint64_t accept_b1 = 1ULL << ((len_states - 1) % 64);
    bool reachable = (fwd_n[accept_w1] & accept_b1) != 0;
    if (!reachable && len_states >= 2) {
        std::size_t accept_w2 = (len_states - 2) / 64;
        std::uint64_t accept_b2 = 1ULL << ((len_states - 2) % 64);
        reachable = (fwd_n[accept_w2] & accept_b2) != 0;
    }

    if (!reachable) {
        return result; // total=0, fully_solved=false, empty diff
    }

    // Backward DP. Initial state (row n): bits len_states-1 and len_states-2 set.
    std::uint64_t* backward = g_scratch.backward.data();
    std::uint64_t* masked = g_scratch.masked.data();
    {
        std::uint64_t* bw_n = backward + n * n_words;
        for (std::size_t w = 0; w < n_words; ++w) bw_n[w] = 0;
        bw_n[(len_states - 1) / 64] |= 1ULL << ((len_states - 1) % 64);
        if (len_states >= 2) {
            bw_n[(len_states - 2) / 64] |= 1ULL << ((len_states - 2) % 64);
        }
    }

    for (std::size_t pi = n; pi-- > 0; ) {
        const std::size_t p = pi;
        std::uint64_t* bw_p = backward + p * n_words;
        std::uint64_t* bw_next = backward + (p + 1) * n_words;
        for (std::size_t w = 0; w < n_words; ++w) {
            cur[w] = bw_next[w];
        }

        std::int8_t cell = line[p];
        if (cell == UNKNOWN) {
            shift_right_1(cur, shifted, n_words);
            for (std::size_t w = 0; w < n_words; ++w) {
                bw_p[w] = (cur[w] & empty_mask[w]) | shifted[w];
            }
        } else if (cell == EMPTY) {
            for (std::size_t w = 0; w < n_words; ++w) {
                masked[w] = cur[w] & empty_mask[w];
            }
            shift_right_1(masked, shifted, n_words);
            for (std::size_t w = 0; w < n_words; ++w) {
                bw_p[w] = masked[w] | shifted[w];
            }
        } else { // FULL
            for (std::size_t w = 0; w < n_words; ++w) {
                masked[w] = cur[w] & full_mask[w];
            }
            shift_right_1(masked, shifted, n_words);
            for (std::size_t w = 0; w < n_words; ++w) {
                bw_p[w] = shifted[w];
            }
        }
    }

    // Determine each unknown cell.
    result.deductions.reserve(n);

    std::uint64_t* bw_empty = g_scratch.bw_empty.data();
    std::uint64_t* bw_full = g_scratch.bw_full.data();
    std::uint64_t* bw_empty_shifted = g_scratch.bw_empty_shifted.data();
    std::uint64_t* bw_full_shifted = g_scratch.bw_full_shifted.data();

    int n_changed = 0;
    int n_unknown_total = 0;

    for (std::size_t p = 0; p < n; ++p) {
        if (line[p] != UNKNOWN) {
            continue;
        }
        ++n_unknown_total;

        const std::uint64_t* bw_pp1 = backward + (p + 1) * n_words;
        const std::uint64_t* fwd_p = forward + p * n_words;

        for (std::size_t w = 0; w < n_words; ++w) {
            bw_empty[w] = bw_pp1[w] & empty_mask[w];
            bw_full[w] = bw_pp1[w] & full_mask[w];
        }
        shift_right_1(bw_empty, bw_empty_shifted, n_words);
        shift_right_1(bw_full, bw_full_shifted, n_words);

        bool can_empty = false;
        bool can_full = false;
        for (std::size_t w = 0; w < n_words; ++w) {
            std::uint64_t f = fwd_p[w];
            std::uint64_t ce = f & (bw_empty[w] | bw_empty_shifted[w]);
            std::uint64_t cf = f & bw_full_shifted[w];
            if (ce != 0) can_empty = true;
            if (cf != 0) can_full = true;
        }

        if (can_empty && !can_full) {
            result.deductions.push_back(deduce_pack(static_cast<int>(p), EMPTY));
            ++n_changed;
        } else if (can_full && !can_empty) {
            result.deductions.push_back(deduce_pack(static_cast<int>(p), FULL));
            ++n_changed;
        }
    }

    result.total = 1;
    result.fully_solved = (n_unknown_total == n_changed);
    return result;
}

