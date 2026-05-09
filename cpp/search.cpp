#include "search.hpp"

#include "lines.hpp"
#include "picture.hpp"
#include "types.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// Memoization cache for solve_line_batch.
// Key = (line bytes, pointer to states vector). Pointer plays the role of
// Python's `id(clue)`: states vectors live in mapped_rows/mapped_cols owned by
// solve() for the lifetime of a single puzzle, so addresses are stable.
// Cleared at the entry of solve().
// ---------------------------------------------------------------------------

struct LineKey {
    std::string line_bytes;
    const std::vector<std::int8_t>* states_ptr;

    bool operator==(const LineKey& o) const noexcept {
        return states_ptr == o.states_ptr && line_bytes == o.line_bytes;
    }
};

struct LineKeyHash {
    std::size_t operator()(const LineKey& k) const noexcept {
        std::size_t h1 = std::hash<std::string>{}(k.line_bytes);
        std::size_t h2 = std::hash<const void*>{}(static_cast<const void*>(k.states_ptr));
        // Mix
        return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
    }
};

using LineCache = std::unordered_map<LineKey, LineSolveResult, LineKeyHash>;

LineCache& line_cache() {
    static LineCache cache;
    return cache;
}

void reset_line_cache() {
    line_cache().clear();
}

// Convert a vector<int8_t> to a std::string of the same byte content for use
// as a hashmap key.
std::string line_to_bytes(const std::vector<std::int8_t>& line) {
    return std::string(reinterpret_cast<const char*>(line.data()), line.size());
}

// ---------------------------------------------------------------------------
// SolveState — mirrors picture.py SolveState (without print_state).
// ---------------------------------------------------------------------------

struct SolveState {
    static constexpr std::size_t PROBING_WINDOW = 100;
    static constexpr double PROBING_THRESHOLD = 0.01;

    int probing_min_solutions;

    int depth = 0;
    std::uint64_t progress_bits = 0;
    int solutions_found = 0;
    bool used_contradiction = false;
    bool used_backtrack = false;
    bool skip_probing = false;
    std::deque<int> probe_outcomes;

    SolveState() {
        const char* env = std::getenv("PROBING_MIN_SOLUTIONS");
        probing_min_solutions = 2;
        if (env != nullptr) {
            try {
                probing_min_solutions = std::stoi(env);
            } catch (...) {
                probing_min_solutions = 2;
            }
        }
    }

    void record_probe(bool found_contradiction) {
        if (skip_probing) return;
        // Mirror picture.py: don't let yield-window disable probing until
        // we've already found multiple solutions.
        if (solutions_found < probing_min_solutions) return;
        probe_outcomes.push_back(found_contradiction ? 1 : 0);
        if (probe_outcomes.size() > PROBING_WINDOW) {
            probe_outcomes.pop_front();
        }
        if (probe_outcomes.size() == PROBING_WINDOW) {
            int sum = 0;
            for (int v : probe_outcomes) sum += v;
            double yield_rate = static_cast<double>(sum) / static_cast<double>(PROBING_WINDOW);
            if (yield_rate < PROBING_THRESHOLD) {
                skip_probing = true;
            }
        }
    }

    void enter_backtrack() {
        depth += 1;
    }

    void first_branch_failed() {
        int bit_pos = depth - 1;
        if (bit_pos < 0) return;
        if (!(progress_bits & (1ULL << bit_pos))) {
            progress_bits |= (1ULL << bit_pos);
        } else {
            carry_from(bit_pos);
        }
    }

    void carry_from(int bit_pos) {
        progress_bits &= ~(1ULL << bit_pos);
        if (bit_pos > 0) {
            int parent_pos = bit_pos - 1;
            if (!(progress_bits & (1ULL << parent_pos))) {
                progress_bits |= (1ULL << parent_pos);
            } else {
                carry_from(parent_pos);
            }
        }
    }

    void exit_backtrack() {
        int bit_pos = depth - 1;
        if (bit_pos >= 0) {
            progress_bits &= ~(1ULL << bit_pos);
        }
        depth -= 1;
    }

    void solution_found() { solutions_found += 1; }
    void mark_contradiction() { used_contradiction = true; }
    void mark_backtrack() { used_backtrack = true; }
};

// ---------------------------------------------------------------------------
// Forward declarations.
// ---------------------------------------------------------------------------

using OnSolution = std::function<bool(const Picture&)>;

bool solve_real(const std::vector<std::vector<std::int8_t>>& mapped_rows,
                const std::vector<std::vector<std::int8_t>>& mapped_cols,
                Picture& pic,
                SolveState& state,
                const OnSolution& on_solution);

bool solve_backtrack(const std::vector<std::vector<std::int8_t>>& mapped_rows,
                     const std::vector<std::vector<std::int8_t>>& mapped_cols,
                     Picture& pic,
                     SolveState& state,
                     const OnSolution& on_solution);

// ---------------------------------------------------------------------------
// solve_check: validate every not-yet-solved row/col; mark fully-known lines
// as solved. Returns false on contradiction.
// ---------------------------------------------------------------------------

bool solve_check(Picture& pic,
                 const std::vector<std::vector<std::int8_t>>& mapped_rows,
                 const std::vector<std::vector<std::int8_t>>& mapped_cols) {
    const int H = pic.height();
    const int W = pic.width();

    for (int i = 0; i < H; ++i) {
        if (pic.solved_rows.count(i)) continue;
        std::vector<std::int8_t> line = pic.get_row(i);
        if (!check_line_valid(line, mapped_rows[i])) {
            return false;
        }
        bool has_unknown = false;
        for (std::int8_t v : line) {
            if (v == UNKNOWN) { has_unknown = true; break; }
        }
        if (!has_unknown) {
            pic.solved_rows.insert(i);
        }
    }

    for (int j = 0; j < W; ++j) {
        if (pic.solved_cols.count(j)) continue;
        std::vector<std::int8_t> line = pic.get_col(j);
        if (!check_line_valid(line, mapped_cols[j])) {
            return false;
        }
        bool has_unknown = false;
        for (std::int8_t v : line) {
            if (v == UNKNOWN) { has_unknown = true; break; }
        }
        if (!has_unknown) {
            pic.solved_cols.insert(j);
        }
    }

    return true;
}

// ---------------------------------------------------------------------------
// solve_one_batch: returns (success, positions, values). On success==false,
// positions/values empty (caller must check success first).
// fully_solved bookkeeping: insert into pic.solved_{rows,cols} on the fly.
// ---------------------------------------------------------------------------

struct BatchResult {
    bool success;
    const std::vector<int>* positions; // pointer into cache entry; valid until next cache mutation
    const std::vector<std::int8_t>* values;
};

BatchResult solve_one_batch(const std::vector<std::int8_t>& states,
                            int index,
                            bool is_col,
                            Picture& pic) {
    std::vector<std::int8_t> line = is_col ? pic.get_col(index) : pic.get_row(index);

    LineKey key{line_to_bytes(line), &states};
    auto& cache = line_cache();
    auto it = cache.find(key);
    const LineSolveResult* result_ptr = nullptr;
    if (it == cache.end()) {
        LineSolveResult res = solve_line_batch(line, states);
        auto ins = cache.emplace(std::move(key), std::move(res));
        result_ptr = &ins.first->second;
    } else {
        result_ptr = &it->second;
    }

    if (result_ptr->total == 0) {
        return BatchResult{false, nullptr, nullptr};
    }

    if (result_ptr->fully_solved) {
        if (is_col) {
            pic.solved_cols.insert(index);
        } else {
            pic.solved_rows.insert(index);
        }
    }

    if (result_ptr->positions.empty()) {
        return BatchResult{true, nullptr, nullptr};
    }

    return BatchResult{true, &result_ptr->positions, &result_ptr->values};
}

// ---------------------------------------------------------------------------
// write_intersection: apply the deduced positions/values to pic, marking
// the cross-direction dirty for any cells that go from UNKNOWN to a value.
// ---------------------------------------------------------------------------

void write_intersection(const std::vector<int>& positions,
                        const std::vector<std::int8_t>& values,
                        int line_index,
                        Picture& pic,
                        bool is_row) {
    if (is_row) {
        const int row = line_index;
        const int W = pic.width();
        std::int8_t* px = pic.pixels.data() + static_cast<std::size_t>(row) * static_cast<std::size_t>(W);
        for (std::size_t i = 0; i < positions.size(); ++i) {
            const int col = positions[i];
            if (px[col] == UNKNOWN) {
                px[col] = values[i];
                pic.unknown_count -= 1;
                pic.mark_col_dirty(col);
            }
        }
    } else {
        const int col = line_index;
        const int W = pic.width();
        for (std::size_t i = 0; i < positions.size(); ++i) {
            const int row = positions[i];
            std::int8_t& cell = pic.pixels[static_cast<std::size_t>(row) * static_cast<std::size_t>(W) + static_cast<std::size_t>(col)];
            if (cell == UNKNOWN) {
                cell = values[i];
                pic.unknown_count -= 1;
                pic.mark_row_dirty(row);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// solve_lines: drain the appropriate dirty queue; on contradiction returns
// false, on success returns true.
// ---------------------------------------------------------------------------

bool solve_lines(const std::vector<std::vector<std::int8_t>>& mapped,
                 Picture& pic,
                 bool is_row) {
    auto& queue = is_row ? pic.row_queue : pic.col_queue;
    auto& dirty = is_row ? pic.row_dirty : pic.col_dirty;
    while (!queue.empty()) {
        const int index = queue.front();
        queue.pop_front();
        dirty[index] = 0;
        BatchResult r = solve_one_batch(mapped[index], index, !is_row, pic);
        if (!r.success) {
            return false;
        }
        if (r.positions != nullptr) {
            write_intersection(*r.positions, *r.values, index, pic, is_row);
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Neighbour scores: for each cell, count of (top, bottom, left, right)
// neighbours that are filled (non-UNKNOWN). Out-of-bounds counts as 1.
// ---------------------------------------------------------------------------

std::vector<int> get_neighbor_scores(const Picture& pic) {
    const int H = pic.height();
    const int W = pic.width();
    std::vector<int> scores(static_cast<std::size_t>(H) * static_cast<std::size_t>(W), 0);
    const std::int8_t* px = pic.pixels.data();
    for (int r = 0; r < H; ++r) {
        for (int c = 0; c < W; ++c) {
            int s = 0;
            // top
            if (r == 0) s += 1;
            else s += (px[(r - 1) * W + c] != UNKNOWN) ? 1 : 0;
            // bottom
            if (r == H - 1) s += 1;
            else s += (px[(r + 1) * W + c] != UNKNOWN) ? 1 : 0;
            // left
            if (c == 0) s += 1;
            else s += (px[r * W + (c - 1)] != UNKNOWN) ? 1 : 0;
            // right
            if (c == W - 1) s += 1;
            else s += (px[r * W + (c + 1)] != UNKNOWN) ? 1 : 0;
            scores[r * W + c] = s;
        }
    }
    return scores;
}

int count_solved_pixels(const Picture& pic) {
    return pic.height() * pic.width() - pic.unknown_count;
}

// ---------------------------------------------------------------------------
// probe_cell: try setting (row, col) to val on a copy of pic, drain queues.
// Returns (ok, pixels_filled). ok=false on contradiction, pixels_filled=0.
// ---------------------------------------------------------------------------

struct ProbeResult {
    bool ok;
    int pixels_filled;
};

ProbeResult probe_cell(int row,
                       int col,
                       std::int8_t val,
                       const std::vector<std::vector<std::int8_t>>& mapped_rows,
                       const std::vector<std::vector<std::int8_t>>& mapped_cols,
                       const Picture& pic) {
    Picture pic2 = pic.copy();
    pic2.set_pixel(row, col, val);
    pic2.mark_row_dirty(row);
    pic2.mark_col_dirty(col);

    if (!solve_check(pic2, mapped_rows, mapped_cols)) {
        return ProbeResult{false, 0};
    }

    while (pic2.has_dirty()) {
        if (!solve_lines(mapped_rows, pic2, true)) {
            return ProbeResult{false, 0};
        }
        if (!solve_lines(mapped_cols, pic2, false)) {
            return ProbeResult{false, 0};
        }
    }

    return ProbeResult{true, count_solved_pixels(pic2)};
}

// ---------------------------------------------------------------------------
// solve_backtrack
// ---------------------------------------------------------------------------

bool solve_backtrack(const std::vector<std::vector<std::int8_t>>& mapped_rows,
                     const std::vector<std::vector<std::int8_t>>& mapped_cols,
                     Picture& pic,
                     SolveState& state,
                     const OnSolution& on_solution) {
    const int H = pic.height();
    const int W = pic.width();
    std::vector<int> scores = get_neighbor_scores(pic);

    // Collect unknown coords in row-major order (matches np.argwhere order).
    std::vector<std::pair<int, int>> unknown_coords;
    unknown_coords.reserve(static_cast<std::size_t>(pic.unknown_count));
    for (int r = 0; r < H; ++r) {
        for (int c = 0; c < W; ++c) {
            if (pic.pixels[r * W + c] == UNKNOWN) {
                unknown_coords.emplace_back(r, c);
            }
        }
    }

    if (unknown_coords.empty()) {
        return true;
    }

    int best_row = -1;
    int best_col = -1;
    std::int8_t best_first_val = FULL;

    if (state.skip_probing) {
        // Pick the cell with highest neighbor score.
        int best_idx = 0;
        int best_score = -1;
        for (std::size_t i = 0; i < unknown_coords.size(); ++i) {
            int r = unknown_coords[i].first;
            int c = unknown_coords[i].second;
            int s = scores[r * W + c];
            if (s > best_score) {
                best_score = s;
                best_idx = static_cast<int>(i);
            }
        }
        best_row = unknown_coords[best_idx].first;
        best_col = unknown_coords[best_idx].second;
        best_first_val = FULL;
    } else {
        // Sort by descending neighbor score (stable to mirror np.argsort which
        // is stable for equal keys).
        std::vector<int> order(unknown_coords.size());
        for (std::size_t i = 0; i < order.size(); ++i) order[i] = static_cast<int>(i);
        std::stable_sort(order.begin(), order.end(),
            [&](int a, int b) {
                int ra = unknown_coords[a].first, ca = unknown_coords[a].second;
                int rb = unknown_coords[b].first, cb = unknown_coords[b].second;
                return scores[ra * W + ca] > scores[rb * W + cb];
            });

        int best_pixels = -1;
        bool have_best = false;

        for (int idx : order) {
            int row = unknown_coords[idx].first;
            int col = unknown_coords[idx].second;

            ProbeResult full_res = probe_cell(row, col, FULL, mapped_rows, mapped_cols, pic);
            ProbeResult empty_res = probe_cell(row, col, EMPTY, mapped_rows, mapped_cols, pic);

            state.record_probe((!full_res.ok) || (!empty_res.ok));

            if (!full_res.ok && !empty_res.ok) {
                return true; // dead branch
            }

            if (full_res.ok && !empty_res.ok) {
                state.mark_contradiction();
                pic.set_pixel(row, col, FULL);
                pic.mark_row_dirty(row);
                pic.mark_col_dirty(col);
                return solve_real(mapped_rows, mapped_cols, pic, state, on_solution);
            }

            if (empty_res.ok && !full_res.ok) {
                state.mark_contradiction();
                pic.set_pixel(row, col, EMPTY);
                pic.mark_row_dirty(row);
                pic.mark_col_dirty(col);
                return solve_real(mapped_rows, mapped_cols, pic, state, on_solution);
            }

            int max_pixels = std::max(full_res.pixels_filled, empty_res.pixels_filled);
            if (max_pixels > best_pixels) {
                best_pixels = max_pixels;
                best_row = row;
                best_col = col;
                best_first_val = (full_res.pixels_filled >= empty_res.pixels_filled) ? FULL : EMPTY;
                have_best = true;
            }
        }

        if (!have_best) {
            return true;
        }
    }

    state.mark_backtrack();

    const int row = best_row;
    const int col = best_col;
    const std::int8_t first_val = best_first_val;
    const std::int8_t second_val = (first_val == FULL) ? EMPTY : FULL;

    state.enter_backtrack();

    // Branch 1
    {
        Picture pic2 = pic.copy();
        pic2.set_pixel(row, col, first_val);
        pic2.mark_row_dirty(row);
        pic2.mark_col_dirty(col);

        bool found_solution = false;
        OnSolution wrapped = [&](const Picture& p) {
            found_solution = true;
            return on_solution(p);
        };
        if (!solve_real(mapped_rows, mapped_cols, pic2, state, wrapped)) {
            // Stop signal — propagate. (We deliberately do NOT call exit_backtrack
            // because the caller is aborting.)
            return false;
        }
        if (!found_solution) {
            state.first_branch_failed();
        }
    }

    // Branch 2
    {
        Picture pic2 = pic.copy();
        pic2.set_pixel(row, col, second_val);
        pic2.mark_row_dirty(row);
        pic2.mark_col_dirty(col);
        if (!solve_real(mapped_rows, mapped_cols, pic2, state, on_solution)) {
            return false;
        }
    }

    state.exit_backtrack();
    return true;
}

// ---------------------------------------------------------------------------
// solve_real
// ---------------------------------------------------------------------------

bool solve_real(const std::vector<std::vector<std::int8_t>>& mapped_rows,
                const std::vector<std::vector<std::int8_t>>& mapped_cols,
                Picture& pic,
                SolveState& state,
                const OnSolution& on_solution) {
    if (!solve_check(pic, mapped_rows, mapped_cols)) {
        return true;
    }

    if (pic.is_solved()) {
        state.solution_found();
        return on_solution(pic);
    }

    while (pic.has_dirty()) {
        if (!solve_lines(mapped_rows, pic, true)) {
            return true;
        }
        if (!solve_lines(mapped_cols, pic, false)) {
            return true;
        }
    }

    if (pic.is_solved()) {
        state.solution_found();
        return on_solution(pic);
    }

    return solve_backtrack(mapped_rows, mapped_cols, pic, state, on_solution);
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

void solve(const std::vector<std::vector<int>>& rows,
           const std::vector<std::vector<int>>& cols,
           std::function<bool(const Picture&)> on_solution,
           Strategy* out_strategy) {
    reset_line_cache();

    const int H = static_cast<int>(rows.size());
    const int W = static_cast<int>(cols.size());

    Picture pic(H, W);

    std::vector<std::vector<std::int8_t>> mapped_rows;
    mapped_rows.reserve(rows.size());
    for (const auto& clue : rows) {
        mapped_rows.push_back(states_pregen(clue));
    }

    std::vector<std::vector<std::int8_t>> mapped_cols;
    mapped_cols.reserve(cols.size());
    for (const auto& clue : cols) {
        mapped_cols.push_back(states_pregen(clue));
    }

    SolveState state;
    (void)solve_real(mapped_rows, mapped_cols, pic, state, on_solution);

    if (out_strategy != nullptr) {
        // Priority: BACKTRACK > CONTRA > BASIC. Mirrors SolveState.get_strategy()
        // in picture.py.
        if (state.used_backtrack) {
            *out_strategy = Strategy::BACKTRACK;
        } else if (state.used_contradiction) {
            *out_strategy = Strategy::CONTRA;
        } else {
            *out_strategy = Strategy::BASIC;
        }
    }
}
