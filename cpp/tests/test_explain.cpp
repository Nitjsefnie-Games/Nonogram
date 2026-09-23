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

int main() {
    std::mt19937 rng(7);
    int checked = 0, conflicts = 0;
    for (int iter = 0; iter < 3000; ++iter) {
        const int n = 5 + static_cast<int>(rng() % 26);
        // random clue that fits
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
        // random partial content from a random solution, then some cells flipped
        std::vector<std::int8_t> line(n, UNKNOWN);
        for (int i = 0; i < n; ++i) if (rng() % 3 == 0) line[i] = (rng() % 2) ? FULL : EMPTY;
        if (unsat(line, spec)) {
            std::vector<int> out(n);
            const int k = explain_conflict(line.data(), n, spec, out.data());
            std::vector<std::int8_t> sub(n, UNKNOWN);
            for (int i = 0; i < k; ++i) sub[out[i]] = line[out[i]];
            if (!unsat(sub, spec)) { std::printf("FAIL conflict subset not unsat\n"); return 1; }
            for (int i = 0; i < k; ++i) {  // local minimality
                std::vector<std::int8_t> s2 = sub; s2[out[i]] = UNKNOWN;
                if (unsat(s2, spec)) { std::printf("FAIL conflict subset not minimal\n"); return 1; }
            }
            ++conflicts;
            continue;
        }
        LineSolveResult r;
        solve_line_batch(line.data(), n, spec, r, true);
        for (int d : r.deductions) {
            const int pos = deduce_pos(d); const std::int8_t val = deduce_val(d);
            std::vector<int> out(n);
            const int k = explain_deduction(line.data(), n, spec, pos, val, out.data());
            std::vector<std::int8_t> sub(n, UNKNOWN);
            for (int i = 0; i < k; ++i) sub[out[i]] = line[out[i]];
            if (!forces(sub, spec, pos, val)) { std::printf("FAIL deduction subset does not force\n"); return 1; }
            for (int i = 0; i < k; ++i) {
                std::vector<std::int8_t> s2 = sub; s2[out[i]] = UNKNOWN;
                if (forces(s2, spec, pos, val)) { std::printf("FAIL deduction subset not minimal\n"); return 1; }
            }
            ++checked;
        }
    }
    std::printf("ok: %d deductions, %d conflicts explained\n", checked, conflicts);
    return 0;
}
