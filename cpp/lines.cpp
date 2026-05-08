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

void build_masks(const std::vector<std::int8_t>& states,
                 std::size_t len_states,
                 std::size_t n_words,
                 std::vector<std::uint64_t>& state_valid,
                 std::vector<std::uint64_t>& empty_mask,
                 std::vector<std::uint64_t>& full_mask) {
    state_valid.assign(n_words, 0);
    empty_mask.assign(n_words, 0);
    full_mask.assign(n_words, 0);
    for (std::size_t s = 0; s < len_states; ++s) {
        std::uint64_t bit = 1ULL << (s % 64);
        std::size_t w = s / 64;
        state_valid[w] |= bit;
        if (states[s] == EMPTY) {
            empty_mask[w] |= bit;
        } else {
            full_mask[w] |= bit;
        }
    }
}

} // namespace

LineSolveResult solve_line_batch(const std::vector<std::int8_t>& line,
                                 const std::vector<std::int8_t>& states) {
    const std::size_t n = line.size();
    const std::size_t len_states = states.size();
    const std::size_t n_words = (len_states + 63) / 64;

    LineSolveResult result;
    result.total = 0;
    result.fully_solved = false;

    if (len_states == 0 || n_words == 0) {
        // Degenerate: no states means no clue at all (empty puzzle line).
        // Treat as unsat to match Python behavior on this path (won't occur
        // normally because states_pregen always emits at least [0]).
        return result;
    }

    std::vector<std::uint64_t> state_valid, empty_mask, full_mask;
    build_masks(states, len_states, n_words, state_valid, empty_mask, full_mask);

    // forward[p] is row p of an (n+1) x n_words array, flat row-major.
    std::vector<std::uint64_t> forward((n + 1) * n_words, 0);
    forward[0] = 1ULL; // bit 0 of word 0 of forward[0]

    std::vector<std::uint64_t> cur(n_words, 0);
    std::vector<std::uint64_t> shifted(n_words, 0);

    for (std::size_t p = 0; p < n; ++p) {
        std::uint64_t* fwd_p = forward.data() + p * n_words;
        std::uint64_t* fwd_next = forward.data() + (p + 1) * n_words;
        for (std::size_t w = 0; w < n_words; ++w) {
            cur[w] = fwd_p[w];
        }

        std::int8_t cell = line[p];
        if (cell == UNKNOWN) {
            shift_left_1(cur.data(), shifted.data(), n_words);
            for (std::size_t w = 0; w < n_words; ++w) {
                fwd_next[w] = (cur[w] & empty_mask[w]) | (shifted[w] & state_valid[w]);
            }
        } else if (cell == EMPTY) {
            shift_left_1(cur.data(), shifted.data(), n_words);
            for (std::size_t w = 0; w < n_words; ++w) {
                fwd_next[w] = (cur[w] & empty_mask[w]) | (shifted[w] & empty_mask[w]);
            }
        } else { // FULL
            shift_left_1(cur.data(), shifted.data(), n_words);
            for (std::size_t w = 0; w < n_words; ++w) {
                fwd_next[w] = shifted[w] & full_mask[w];
            }
        }
    }

    // Accept check: bit (len_states - 1) or bit (len_states - 2) of forward[n].
    const std::uint64_t* fwd_n = forward.data() + n * n_words;
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

    // Backward DP. Initial state: bits len_states-1 and len_states-2 set.
    std::vector<std::uint64_t> backward((n + 1) * n_words, 0);
    {
        std::uint64_t* bw_n = backward.data() + n * n_words;
        std::size_t aw1 = (len_states - 1) / 64;
        bw_n[aw1] |= 1ULL << ((len_states - 1) % 64);
        if (len_states >= 2) {
            std::size_t aw2 = (len_states - 2) / 64;
            bw_n[aw2] |= 1ULL << ((len_states - 2) % 64);
        }
    }

    std::vector<std::uint64_t> masked(n_words, 0);

    for (std::size_t pi = n; pi-- > 0; ) {
        const std::size_t p = pi;
        std::uint64_t* bw_p = backward.data() + p * n_words;
        std::uint64_t* bw_next = backward.data() + (p + 1) * n_words;
        for (std::size_t w = 0; w < n_words; ++w) {
            cur[w] = bw_next[w];
        }

        std::int8_t cell = line[p];
        if (cell == UNKNOWN) {
            shift_right_1(cur.data(), shifted.data(), n_words);
            for (std::size_t w = 0; w < n_words; ++w) {
                bw_p[w] = (cur[w] & empty_mask[w]) | shifted[w];
            }
        } else if (cell == EMPTY) {
            for (std::size_t w = 0; w < n_words; ++w) {
                masked[w] = cur[w] & empty_mask[w];
            }
            shift_right_1(masked.data(), shifted.data(), n_words);
            for (std::size_t w = 0; w < n_words; ++w) {
                bw_p[w] = masked[w] | shifted[w];
            }
        } else { // FULL
            for (std::size_t w = 0; w < n_words; ++w) {
                masked[w] = cur[w] & full_mask[w];
            }
            shift_right_1(masked.data(), shifted.data(), n_words);
            for (std::size_t w = 0; w < n_words; ++w) {
                bw_p[w] = shifted[w];
            }
        }
    }

    // Determine each unknown cell.
    result.positions.reserve(n);
    result.values.reserve(n);

    std::vector<std::uint64_t> bw_empty(n_words, 0);
    std::vector<std::uint64_t> bw_full(n_words, 0);
    std::vector<std::uint64_t> bw_empty_shifted(n_words, 0);
    std::vector<std::uint64_t> bw_full_shifted(n_words, 0);

    int n_changed = 0;
    int n_unknown_total = 0;

    for (std::size_t p = 0; p < n; ++p) {
        if (line[p] != UNKNOWN) {
            continue;
        }
        ++n_unknown_total;

        const std::uint64_t* bw_pp1 = backward.data() + (p + 1) * n_words;
        const std::uint64_t* fwd_p = forward.data() + p * n_words;

        for (std::size_t w = 0; w < n_words; ++w) {
            bw_empty[w] = bw_pp1[w] & empty_mask[w];
            bw_full[w] = bw_pp1[w] & full_mask[w];
        }
        shift_right_1(bw_empty.data(), bw_empty_shifted.data(), n_words);
        shift_right_1(bw_full.data(), bw_full_shifted.data(), n_words);

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
            result.positions.push_back(static_cast<int>(p));
            result.values.push_back(EMPTY);
            ++n_changed;
        } else if (can_full && !can_empty) {
            result.positions.push_back(static_cast<int>(p));
            result.values.push_back(FULL);
            ++n_changed;
        }
    }

    result.total = 1;
    result.fully_solved = (n_unknown_total == n_changed);
    return result;
}

bool check_line_valid(const std::vector<std::int8_t>& line,
                      const std::vector<std::int8_t>& states) {
    const std::size_t n = line.size();
    const std::size_t len_states = states.size();
    const std::size_t n_words = (len_states + 63) / 64;

    if (len_states == 0 || n_words == 0) {
        return false;
    }

    std::vector<std::uint64_t> state_valid, empty_mask, full_mask;
    build_masks(states, len_states, n_words, state_valid, empty_mask, full_mask);

    std::vector<std::uint64_t> forward((n + 1) * n_words, 0);
    forward[0] = 1ULL;

    std::vector<std::uint64_t> cur(n_words, 0);
    std::vector<std::uint64_t> shifted(n_words, 0);

    for (std::size_t p = 0; p < n; ++p) {
        std::uint64_t* fwd_p = forward.data() + p * n_words;
        std::uint64_t* fwd_next = forward.data() + (p + 1) * n_words;
        for (std::size_t w = 0; w < n_words; ++w) {
            cur[w] = fwd_p[w];
        }

        std::int8_t cell = line[p];
        if (cell == UNKNOWN) {
            shift_left_1(cur.data(), shifted.data(), n_words);
            for (std::size_t w = 0; w < n_words; ++w) {
                fwd_next[w] = (cur[w] & empty_mask[w]) | (shifted[w] & state_valid[w]);
            }
        } else if (cell == EMPTY) {
            shift_left_1(cur.data(), shifted.data(), n_words);
            for (std::size_t w = 0; w < n_words; ++w) {
                fwd_next[w] = (cur[w] & empty_mask[w]) | (shifted[w] & empty_mask[w]);
            }
        } else { // FULL
            shift_left_1(cur.data(), shifted.data(), n_words);
            for (std::size_t w = 0; w < n_words; ++w) {
                fwd_next[w] = shifted[w] & full_mask[w];
            }
        }
    }

    const std::uint64_t* fwd_n = forward.data() + n * n_words;
    std::size_t accept_w1 = (len_states - 1) / 64;
    std::uint64_t accept_b1 = 1ULL << ((len_states - 1) % 64);
    if ((fwd_n[accept_w1] & accept_b1) != 0) {
        return true;
    }
    if (len_states >= 2) {
        std::size_t accept_w2 = (len_states - 2) / 64;
        std::uint64_t accept_b2 = 1ULL << ((len_states - 2) % 64);
        if ((fwd_n[accept_w2] & accept_b2) != 0) {
            return true;
        }
    }
    return false;
}

std::vector<std::int8_t> states_pregen(const std::vector<int>& clue) {
    std::vector<std::int8_t> states;
    states.push_back(EMPTY);
    for (int nr : clue) {
        for (int i = 0; i < nr; ++i) {
            states.push_back(FULL);
        }
        states.push_back(EMPTY);
    }
    return states;
}
