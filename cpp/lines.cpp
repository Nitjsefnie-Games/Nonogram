#include "lines.hpp"
#include "types.hpp"

#include <algorithm>
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
    std::vector<int> ded;  // deduction candidates of the fused sweeps, one per cell

    // n ints of uninitialized-by-contract scratch for the deduction sweep.
    int* ded_buf(std::size_t n) {
        if (ded.size() < n) ded.resize(n);
        return ded.data();
    }

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
void solve_line_batch_1w(const std::int8_t* line, std::size_t n,
                         const LineSpec& spec, LineSolveResult& result,
                         bool has_unknown) {
    const std::size_t len_states = spec.len_states;
    const std::uint64_t sv = spec.state_valid[0];
    const std::uint64_t em = spec.empty_mask[0];
    const std::uint64_t fm = spec.full_mask[0];

    result.deductions.clear();
    result.total = 0;

    g_scratch.ensure(n + 1, 1);
    std::uint64_t* fwd = g_scratch.forward.data();

    // Per-cell-value mask tables make both DP sweeps branch-free: the cell
    // values along a line are data-dependent and mispredict badly otherwise.
    //   forward:  next = (cur & stay[v]) | ((cur << 1) & step[v])
    //   backward: prev = (cur & stay[v]) | ((cur & bstep[v]) >> 1)
    // with stay = {em, 0, em}, step = {em, fm, sv}, bstep = {em, fm, sv}
    // for v = EMPTY, FULL, UNKNOWN (UNKNOWN steps into any valid state).
    const std::uint64_t stay[3] = {em, 0, em};
    const std::uint64_t step[3] = {em, fm, sv};

    fwd[0] = 1ULL;
    for (std::size_t p = 0; p < n; ++p) {
        const std::uint64_t cur = fwd[p];
        const int v = line[p];
        fwd[p + 1] = (cur & stay[v]) | ((cur << 1) & step[v]);
    }

    std::uint64_t accept = 1ULL << (len_states - 1);
    if (len_states >= 2) accept |= 1ULL << (len_states - 2);
    if ((fwd[n] & accept) == 0) {
        return;  // unsat
    }
    if (!has_unknown) {
        result.total = 1;  // nothing to deduce
        return;
    }

    // Backward sweep fused with the deduction pass: at position p the sweep
    // holds bwd[p+1] (cur) and fwd[p] is already stored, which is all a
    // deduction needs, so the third pass over the line goes away. The
    // backward state itself only needs to be kept in a register.
    //
    // Deductions are collected branch-free: every cell writes a candidate
    // into the next slot of a scratch buffer and the cursor advances only
    // for an UNKNOWN cell that is determined (exactly one of can_empty /
    // can_full). The sweep runs high-to-low, so they land in descending
    // position order and are copied out reversed into the ascending order
    // consumers expect. (Sizing the result vector to n first zero-filled n
    // ints per line solve: 4% of a 90 x 69 count run in memset.)
    int* out = g_scratch.ded_buf(n);
    std::size_t k = 0;
    std::uint64_t cur = accept;
    for (std::size_t pi = n; pi-- > 0; ) {
        const int v = line[pi];
        const std::uint64_t f = fwd[pi];
        const std::uint64_t bw_empty = cur & em;
        const std::uint64_t bw_full = cur & fm;
        const int can_empty = (f & (bw_empty | (bw_empty >> 1))) != 0;
        const int can_full = (f & (bw_full >> 1)) != 0;
        // FULL iff can_full (when determined, exactly one is set).
        out[k] = deduce_pack(static_cast<int>(pi), static_cast<std::int8_t>(can_full));
        k += static_cast<std::size_t>((v == UNKNOWN) & (can_empty ^ can_full));
        cur = (cur & stay[v]) | ((cur & step[v]) >> 1);
    }
    result.deductions.resize(k);
    int* d = result.deductions.data();
    for (std::size_t i = 0; i < k; ++i) d[i] = out[k - 1 - i];
    result.total = 1;
}

// Two-word states (65 to 128 DFA states): the one-word routine on 128-bit
// values held as a (lo, hi) pair, with the shifts carrying between the
// words. Same tables, same fused backward sweep, same deductions as the
// general routine below, which this replaces for n_words == 2 (the general
// loops over words with a runtime count, three passes, and pushes each
// deduction; on a 90 x 69 count run it was 14% of the instructions).
void solve_line_batch_2w(const std::int8_t* line, std::size_t n,
                         const LineSpec& spec, LineSolveResult& result,
                         bool has_unknown) {
    const std::size_t len_states = spec.len_states;
    const std::uint64_t sv0 = spec.state_valid[0], sv1 = spec.state_valid[1];
    const std::uint64_t em0 = spec.empty_mask[0], em1 = spec.empty_mask[1];
    const std::uint64_t fm0 = spec.full_mask[0], fm1 = spec.full_mask[1];

    result.deductions.clear();
    result.total = 0;

    g_scratch.ensure(n + 1, 2);
    std::uint64_t* fwd = g_scratch.forward.data();

    const std::uint64_t stay0[3] = {em0, 0, em0}, stay1[3] = {em1, 0, em1};
    const std::uint64_t step0[3] = {em0, fm0, sv0}, step1[3] = {em1, fm1, sv1};

    fwd[0] = 1ULL;
    fwd[1] = 0;
    for (std::size_t p = 0; p < n; ++p) {
        const std::uint64_t lo = fwd[2 * p], hi = fwd[2 * p + 1];
        const int v = line[p];
        fwd[2 * p + 2] = (lo & stay0[v]) | ((lo << 1) & step0[v]);
        fwd[2 * p + 3] = (hi & stay1[v]) | (((hi << 1) | (lo >> 63)) & step1[v]);
    }

    // Accept states len_states-1 and len_states-2; len_states >= 65 here, so
    // both are in the high word unless len_states == 65 (state 63 is in the
    // low word).
    std::uint64_t acc0 = 0, acc1 = 0;
    {
        const std::size_t s1 = len_states - 1, s2 = len_states - 2;
        if (s1 >= 64) acc1 |= 1ULL << (s1 - 64); else acc0 |= 1ULL << s1;
        if (s2 >= 64) acc1 |= 1ULL << (s2 - 64); else acc0 |= 1ULL << s2;
    }
    if (((fwd[2 * n] & acc0) | (fwd[2 * n + 1] & acc1)) == 0) {
        return;  // unsat
    }
    if (!has_unknown) {
        result.total = 1;
        return;
    }

    int* out = g_scratch.ded_buf(n);
    std::size_t k = 0;
    std::uint64_t lo = acc0, hi = acc1;
    for (std::size_t pi = n; pi-- > 0; ) {
        const int v = line[pi];
        const std::uint64_t f0 = fwd[2 * pi], f1 = fwd[2 * pi + 1];
        const std::uint64_t be0 = lo & em0, be1 = hi & em1;
        const std::uint64_t bf0 = lo & fm0, bf1 = hi & fm1;
        // 128-bit >> 1 of (be0, be1) and (bf0, bf1).
        const std::uint64_t bes0 = (be0 >> 1) | (be1 << 63), bes1 = be1 >> 1;
        const std::uint64_t bfs0 = (bf0 >> 1) | (bf1 << 63), bfs1 = bf1 >> 1;
        const int can_empty = ((f0 & (be0 | bes0)) | (f1 & (be1 | bes1))) != 0;
        const int can_full = ((f0 & bfs0) | (f1 & bfs1)) != 0;
        out[k] = deduce_pack(static_cast<int>(pi), static_cast<std::int8_t>(can_full));
        k += static_cast<std::size_t>((v == UNKNOWN) & (can_empty ^ can_full));
        const std::uint64_t s0 = lo & step0[v], s1 = hi & step1[v];
        lo = (lo & stay0[v]) | ((s0 >> 1) | (s1 << 63));
        hi = (hi & stay1[v]) | (s1 >> 1);
    }
    result.deductions.resize(k);
    int* d = result.deductions.data();
    for (std::size_t i = 0; i < k; ++i) d[i] = out[k - 1 - i];
    result.total = 1;
}
}  // namespace

void solve_line_batch(const std::int8_t* line, std::size_t n,
                      const LineSpec& spec, LineSolveResult& result,
                      bool has_unknown) {
    const std::size_t len_states = spec.len_states;
    const std::size_t n_words = spec.n_words;

    result.deductions.clear();
    result.total = 0;

    if (len_states == 0 || n_words == 0) {
        // Degenerate: no states means no clue at all (empty puzzle line).
        return;
    }

    if (n_words == 1) {
        solve_line_batch_1w(line, n, spec, result, has_unknown);
        return;
    }
    if (n_words == 2) {
        solve_line_batch_2w(line, n, spec, result, has_unknown);
        return;
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
        return; // total=0, empty deductions
    }
    if (!has_unknown) {
        result.total = 1;  // nothing to deduce
        return;
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

    std::uint64_t* bw_empty = g_scratch.bw_empty.data();
    std::uint64_t* bw_full = g_scratch.bw_full.data();
    std::uint64_t* bw_empty_shifted = g_scratch.bw_empty_shifted.data();
    std::uint64_t* bw_full_shifted = g_scratch.bw_full_shifted.data();

    for (std::size_t p = 0; p < n; ++p) {
        if (line[p] != UNKNOWN) {
            continue;
        }

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
        } else if (can_full && !can_empty) {
            result.deductions.push_back(deduce_pack(static_cast<int>(p), FULL));
        }
    }

    result.total = 1;
}

