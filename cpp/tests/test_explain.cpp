#include "../lines.hpp"
#include "../types.hpp"
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

static bool forces(const std::vector<std::int8_t>& line, const LineSpec& spec, int pos, std::int8_t val) {
    LineSolveResult r;
    solve_line_batch(line.data(), line.size(), spec, r, true);
    if (r.total == 0) return false;
    for (int d : r.deductions) if (deduce_pos(d) == pos && deduce_val(d) == val) return true;
    return false;
}
static bool unsat(const std::vector<std::int8_t>& line, const LineSpec& spec) {
    LineSolveResult r;
    solve_line_batch(line.data(), line.size(), spec, r, true);
    return r.total == 0;
}

// Explains the line's conflict (if unsat) or every deduction it makes, and
// checks each explanation: the subset alone reproduces the conflict /
// deduction, and dropping any single member of it does not (local
// minimality). Counts into `checked` / `conflicts`; false on a failure.
static bool check_line(const std::vector<std::int8_t>& line, const LineSpec& spec, int& checked, int& conflicts) {
    const int n = static_cast<int>(line.size());
    if (unsat(line, spec)) {
        std::vector<int> out(n);
        const int k = explain_conflict(line.data(), n, spec, out.data());
        std::vector<std::int8_t> sub(n, UNKNOWN);
        for (int i = 0; i < k; ++i) sub[out[i]] = line[out[i]];
        if (!unsat(sub, spec)) { std::printf("FAIL conflict subset not unsat\n"); return false; }
        for (int i = 0; i < k; ++i) {  // local minimality
            std::vector<std::int8_t> s2 = sub; s2[out[i]] = UNKNOWN;
            if (unsat(s2, spec)) { std::printf("FAIL conflict subset not minimal\n"); return false; }
        }
        ++conflicts;
        return true;
    }
    LineSolveResult r;
    solve_line_batch(line.data(), n, spec, r, true);
    for (int d : r.deductions) {
        const int pos = deduce_pos(d); const std::int8_t val = deduce_val(d);
        std::vector<int> out(n);
        const int k = explain_deduction(line.data(), n, spec, pos, val, out.data());
        std::vector<std::int8_t> sub(n, UNKNOWN);
        for (int i = 0; i < k; ++i) sub[out[i]] = line[out[i]];
        if (!forces(sub, spec, pos, val)) { std::printf("FAIL deduction subset does not force\n"); return false; }
        for (int i = 0; i < k; ++i) {
            std::vector<std::int8_t> s2 = sub; s2[out[i]] = UNKNOWN;
            if (forces(s2, spec, pos, val)) { std::printf("FAIL deduction subset not minimal\n"); return false; }
        }
        ++checked;
    }
    return true;
}

int main() {
    std::mt19937 rng(7);
    int checked = 0, conflicts = 0;
    // Short lines (1-word automaton): a random clue that fits, with random
    // cells set independently of it (each known with probability 1/3, FULL or
    // EMPTY at random), so many lines contradict.
    for (int iter = 0; iter < 3000; ++iter) {
        const int n = 5 + static_cast<int>(rng() % 26);
        std::vector<int> clue;
        int used = 0;
        while (true) {
            int b = 1 + static_cast<int>(rng() % 4);
            if (used + b + (clue.empty() ? 0 : 1) > n) break;
            used += b + (clue.empty() ? 0 : 1);
            clue.push_back(b);
            if (rng() % 3 == 0) break;
        }
        LineSpec spec = make_line_spec(clue);
        std::vector<std::int8_t> line(n, UNKNOWN);
        for (int i = 0; i < n; ++i) if (rng() % 3 == 0) line[i] = (rng() % 2) ? FULL : EMPTY;
        if (!check_line(line, spec, checked, conflicts)) return 1;
    }
    // Long lines with dense clues (blocks up to 8, filled until the line is
    // nearly full). The automaton has 1 + sum(blocks) + #blocks states, at
    // most n + 2: most 40-80-cell lines stay within one word (64 states) and
    // some reach two (65-128 states); the 130-160-cell lines need three
    // words, the general routine. The asserts below keep both of those
    // covered. The content is a third of the
    // cells of a random solution, and in a quarter of the lines one of the
    // revealed cells flipped, so both deductions and conflicts come up.
    int by_words[4] = {0, 0, 0, 0};  // lines per automaton width: 1, 2, 3+ words
    for (int iter = 0; iter < 400; ++iter) {
        const int n = iter < 350 ? 40 + static_cast<int>(rng() % 41)     // 40-80 cells
                                 : 130 + static_cast<int>(rng() % 31);   // 130-160 cells
        std::vector<int> clue;
        int used = 0;
        while (true) {
            int b = 1 + static_cast<int>(rng() % 8);
            if (used + b + (clue.empty() ? 0 : 1) > n) break;
            used += b + (clue.empty() ? 0 : 1);
            clue.push_back(b);
        }
        if (clue.empty()) continue;
        LineSpec spec = make_line_spec(clue);
        ++by_words[spec.n_words >= 3 ? 3 : spec.n_words];
        // A random solution: the slack n - used spread over the k + 1 gaps.
        std::vector<int> extra(clue.size() + 1, 0);
        for (int s = 0; s < n - used; ++s) ++extra[rng() % extra.size()];
        std::vector<std::int8_t> sol;
        for (std::size_t j = 0; j < clue.size(); ++j) {
            sol.insert(sol.end(), extra[j] + (j == 0 ? 0 : 1), EMPTY);
            sol.insert(sol.end(), clue[j], FULL);
        }
        sol.insert(sol.end(), extra.back(), EMPTY);
        std::vector<std::int8_t> line(n, UNKNOWN);
        std::vector<int> revealed;
        for (int i = 0; i < n; ++i) if (rng() % 3 == 0) { line[i] = sol[i]; revealed.push_back(i); }
        if (!revealed.empty() && rng() % 4 == 0) {
            const int i = revealed[rng() % revealed.size()];
            line[i] = line[i] == FULL ? EMPTY : FULL;
        }
        if (!check_line(line, spec, checked, conflicts)) return 1;
    }
    if (by_words[2] == 0 || by_words[3] == 0) {
        std::printf("FAIL long lines did not reach the 2-word and general paths (%d 2-word, %d 3+-word)\n",
                    by_words[2], by_words[3]);
        return 1;
    }
    std::printf("ok: %d deductions, %d conflicts explained (long lines: %d 1-word, %d 2-word, %d 3+-word)\n",
                checked, conflicts, by_words[1], by_words[2], by_words[3]);
    return 0;
}
