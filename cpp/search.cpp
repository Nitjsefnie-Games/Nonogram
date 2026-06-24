#include "search.hpp"

#include "lines.hpp"
#include "picture.hpp"
#include "types.hpp"

// Vendored third-party hash map: ankerl::unordered_dense (MIT, v4.8.1).
// See cpp/external/ankerl/unordered_dense.h for license/copyright header.
// Used for the line-batch memoization cache in this file.
#include "external/ankerl/unordered_dense.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <list>
#include <string>
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
    const LineSpec* states_ptr;

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

struct LineCacheEntry {
    LineKey key;
    LineSolveResult value;
};

class LRULineCache {
public:
    std::size_t max_entries_ = 1'000'000;  // default; set by reset_line_cache

    void set_max_entries(std::size_t n) { max_entries_ = std::max<std::size_t>(1, n); }

    void clear() {
        entries_.clear();
        index_.clear();
    }

    LineSolveResult* find_and_promote(const LineKey& key) {
        auto it = index_.find(key);
        if (it == index_.end()) return nullptr;
        // Move to front (most recently used).
        entries_.splice(entries_.begin(), entries_, it->second);
        return &it->second->value;
    }

    LineSolveResult* insert(LineKey key, LineSolveResult value) {
        if (index_.size() >= max_entries_) {
            // Evict LRU (back of list).
            index_.erase(entries_.back().key);
            entries_.pop_back();
        }
        entries_.push_front(LineCacheEntry{key, std::move(value)});
        auto list_it = entries_.begin();
        index_.emplace(std::move(key), list_it);
        return &list_it->value;
    }

private:
    std::list<LineCacheEntry> entries_;
    ankerl::unordered_dense::map<LineKey, std::list<LineCacheEntry>::iterator, LineKeyHash> index_;
};

LRULineCache& line_cache() {
    static LRULineCache cache;
    return cache;
}

void reset_line_cache(int max_line_len, std::size_t budget_bytes) {
    auto& c = line_cache();
    c.clear();
    std::size_t per_entry = 295 + 2 * static_cast<std::size_t>(max_line_len);
    c.set_max_entries(std::max<std::size_t>(1, budget_bytes / per_entry));
}

// Convert a vector<int8_t> to a std::string of the same byte content for use
// as a hashmap key.
std::string line_to_bytes(const std::vector<std::int8_t>& line) {
    return std::string(reinterpret_cast<const char*>(line.data()), line.size());
}

// ---------------------------------------------------------------------------
// Trail — records cell-index mutations so probe_cell and solve_backtrack can
// undo them in place (avoiding pic.copy()). Threaded through all propagation
// paths. A single Trail is owned by solve(); each backtrack branch uses
// trail.size() as a mark, applies its branch pixel + propagation, then
// truncates the trail (reverting cells to UNKNOWN) before the next branch.
// probe_cell uses a private local Trail because its mutations are reverted
// before returning to its caller (solve_backtrack), so they don't need to
// commingle with the outer trail.
// ---------------------------------------------------------------------------

struct Trail {
    std::vector<int> changed_cell_indices;
};

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

bool solve_real(const std::vector<LineSpec>& mapped_rows,
                const std::vector<LineSpec>& mapped_cols,
                Picture& pic,
                SolveState& state,
                const OnSolution& on_solution,
                Trail& trail);

bool solve_backtrack(const std::vector<LineSpec>& mapped_rows,
                     const std::vector<LineSpec>& mapped_cols,
                     Picture& pic,
                     SolveState& state,
                     const OnSolution& on_solution,
                     Trail& trail);

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

BatchResult solve_one_batch(const LineSpec& spec,
                            int index,
                            bool is_col,
                            Picture& pic) {
    std::vector<std::int8_t> line = is_col ? pic.get_col(index) : pic.get_row(index);

    LineKey key{line_to_bytes(line), &spec};
    auto& cache = line_cache();
    const LineSolveResult* result_ptr = cache.find_and_promote(key);
    if (result_ptr == nullptr) {
        LineSolveResult res = solve_line_batch(line, spec);
        result_ptr = cache.insert(std::move(key), std::move(res));
    }

    if (result_ptr->total == 0) {
        return BatchResult{false, nullptr, nullptr};
    }

    if (result_ptr->positions.empty()) {
        return BatchResult{true, nullptr, nullptr};
    }

    return BatchResult{true, &result_ptr->positions, &result_ptr->values};
}

// ---------------------------------------------------------------------------
// write_intersection: apply the deduced positions/values to pic, marking
// the cross-direction dirty for any cells that go from UNKNOWN to a value.
//
// Every pixel transition UNKNOWN -> value is recorded into trail (linear cell
// index r*W + c). On revert, the caller truncates the trail and walks each
// recorded index back to UNKNOWN.
// ---------------------------------------------------------------------------

void write_intersection(const std::vector<int>& positions,
                        const std::vector<std::int8_t>& values,
                        int line_index,
                        Picture& pic,
                        bool is_row,
                        Trail& trail) {
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
                trail.changed_cell_indices.push_back(row * W + col);
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
                trail.changed_cell_indices.push_back(row * W + col);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// solve_lines: drain the appropriate dirty queue; on contradiction returns
// false, on success returns true.
// ---------------------------------------------------------------------------

bool solve_lines(const std::vector<LineSpec>& mapped,
                 Picture& pic,
                 bool is_row,
                 Trail& trail) {
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
            write_intersection(*r.positions, *r.values, index, pic, is_row, trail);
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
// probe_cell: try setting (row, col) to val, drain queues. Returns
// (ok, pixels_filled). ok=false on contradiction, pixels_filled=0.
//
// Mutates pic in place under a trail and reverts every change before
// returning, regardless of which path is taken (RAII guard). Avoids the
// per-probe pic.copy() that dominated the hot path.
//
// Precondition: at entry, both queues are empty and all dirty bits are 0.
// (Holds because solve_real drains both queues before calling solve_backtrack,
// which calls probe_cell without touching the picture.)
// ---------------------------------------------------------------------------

struct ProbeResult {
    bool ok;
    int pixels_filled;
};

// RAII guard that snapshots a small amount of Picture state at construction
// and restores pic to its entry state on destruction by walking the trail.
struct ProbeGuard {
    Picture& pic;
    Trail& trail;
    int saved_unknown_count;

    ProbeGuard(Picture& p, Trail& t)
        : pic(p),
          trail(t),
          saved_unknown_count(p.unknown_count) {
        // Precondition: queues empty, dirty all zero.
        assert(p.row_queue.empty() && p.col_queue.empty());
    }

    ~ProbeGuard() {
        // Walk trail in reverse and restore each cell to UNKNOWN. We bypass
        // Picture::set_pixel because it only adjusts unknown_count for the
        // UNKNOWN -> value direction.
        std::int8_t* px = pic.pixels.data();
        for (auto it = trail.changed_cell_indices.rbegin();
             it != trail.changed_cell_indices.rend(); ++it) {
            px[*it] = UNKNOWN;
        }
        pic.unknown_count = saved_unknown_count;

        // Drain any pending queue entries and clear their dirty flags.
        // (Anything in the queue has dirty[i] = 1 by construction; clearing
        // dirty for popped indices fully restores the empty/zero invariant.)
        while (!pic.row_queue.empty()) {
            int i = pic.row_queue.front();
            pic.row_queue.pop_front();
            pic.row_dirty[i] = 0;
        }
        while (!pic.col_queue.empty()) {
            int j = pic.col_queue.front();
            pic.col_queue.pop_front();
            pic.col_dirty[j] = 0;
        }
    }
};

ProbeResult probe_cell(int row,
                       int col,
                       std::int8_t val,
                       const std::vector<LineSpec>& mapped_rows,
                       const std::vector<LineSpec>& mapped_cols,
                       Picture& pic) {
    Trail trail;
    // Reserve enough headroom that most probes don't reallocate. Linear in
    // the number of unknowns.
    trail.changed_cell_indices.reserve(static_cast<std::size_t>(pic.unknown_count));

    ProbeGuard guard(pic, trail);

    // Apply the probe pixel directly (bypass set_pixel; record on trail).
    const int W = pic.width();
    const int idx = row * W + col;
    pic.pixels[idx] = val;
    pic.unknown_count -= 1;
    trail.changed_cell_indices.push_back(idx);

    pic.mark_row_dirty(row);
    pic.mark_col_dirty(col);

    // No separate full-board validation pass: only the row+col just set can
    // be inconsistent (the board was consistent at probe entry), and both are
    // dirty, so solve_lines below re-solves them and reports any contradiction
    // via solve_line_batch's total==0. solve_check was a redundant O(H+W)
    // re-validation of every line on every probe.
    while (pic.has_dirty()) {
        if (!solve_lines(mapped_rows, pic, true, trail)) {
            return ProbeResult{false, 0};
        }
        if (!solve_lines(mapped_cols, pic, false, trail)) {
            return ProbeResult{false, 0};
        }
    }

    return ProbeResult{true, count_solved_pixels(pic)};
}

// ---------------------------------------------------------------------------
// solve_backtrack
// ---------------------------------------------------------------------------

// Helper: revert pic to the state captured at branch entry. Walks the trail
// from its current size back to `mark`, restoring each cell to UNKNOWN.
// Restores unknown_count and the solved-row/col sets from snapshots. Drains
// any leftover queue entries (clearing their dirty bits) so the picture
// is in a clean queues-empty / dirty-zero state, matching branch entry.
void revert_branch(Picture& pic,
                   Trail& trail,
                   std::size_t mark,
                   int saved_unknown_count) {
    std::int8_t* px = pic.pixels.data();
    while (trail.changed_cell_indices.size() > mark) {
        int idx = trail.changed_cell_indices.back();
        trail.changed_cell_indices.pop_back();
        px[idx] = UNKNOWN;
    }
    pic.unknown_count = saved_unknown_count;
    while (!pic.row_queue.empty()) {
        int i = pic.row_queue.front();
        pic.row_queue.pop_front();
        pic.row_dirty[i] = 0;
    }
    while (!pic.col_queue.empty()) {
        int j = pic.col_queue.front();
        pic.col_queue.pop_front();
        pic.col_dirty[j] = 0;
    }
}

bool solve_backtrack(const std::vector<LineSpec>& mapped_rows,
                     const std::vector<LineSpec>& mapped_cols,
                     Picture& pic,
                     SolveState& state,
                     const OnSolution& on_solution,
                     Trail& trail) {
    const int H = pic.height();
    const int W = pic.width();
    std::vector<int> scores = get_neighbor_scores(pic);

    // Per-row / per-col UNKNOWN counts, used as the "most constrained line"
    // signal in the composite sort key (lower => more constrained line).
    std::vector<int> unknowns_in_row(static_cast<std::size_t>(H), 0);
    std::vector<int> unknowns_in_col(static_cast<std::size_t>(W), 0);

    // Collect unknown coords in row-major order (matches np.argwhere order).
    std::vector<std::pair<int, int>> unknown_coords;
    unknown_coords.reserve(static_cast<std::size_t>(pic.unknown_count));
    for (int r = 0; r < H; ++r) {
        for (int c = 0; c < W; ++c) {
            if (pic.pixels[r * W + c] == UNKNOWN) {
                unknown_coords.emplace_back(r, c);
                unknowns_in_row[r] += 1;
                unknowns_in_col[c] += 1;
            }
        }
    }

    if (unknown_coords.empty()) {
        return true;
    }

    int best_row = -1;
    int best_col = -1;
    std::int8_t best_first_val = FULL;

    // Composite key: ascending line_constraint, then descending neighbor.
    // Returned via a pair so std::less ordering matches the desired ordering
    // (we negate neighbor so larger neighbor compares smaller, i.e. wins ties).
    auto composite_key = [&](int r, int c) {
        int line_constraint = unknowns_in_row[r] + unknowns_in_col[c];
        int neighbor = scores[r * W + c];
        return std::pair<int, int>(line_constraint, -neighbor);
    };

    if (state.skip_probing) {
        // Pick the cell with the smallest composite key (most-constrained
        // line; tie-broken by highest neighbor score). Equivalent to
        // sorted_coords[0] under the same key.
        int best_idx = 0;
        std::pair<int, int> best_key = composite_key(unknown_coords[0].first,
                                                     unknown_coords[0].second);
        for (std::size_t i = 1; i < unknown_coords.size(); ++i) {
            std::pair<int, int> k = composite_key(unknown_coords[i].first,
                                                  unknown_coords[i].second);
            if (k < best_key) {
                best_key = k;
                best_idx = static_cast<int>(i);
            }
        }
        best_row = unknown_coords[best_idx].first;
        best_col = unknown_coords[best_idx].second;
        best_first_val = FULL;
    } else {
        // Sort by composite key ascending: ascending line_constraint, then
        // descending neighbor (stable to keep deterministic order on full ties,
        // mirroring np.argsort which is stable).
        std::vector<int> order(unknown_coords.size());
        for (std::size_t i = 0; i < order.size(); ++i) order[i] = static_cast<int>(i);
        std::stable_sort(order.begin(), order.end(),
            [&](int a, int b) {
                int ra = unknown_coords[a].first, ca = unknown_coords[a].second;
                int rb = unknown_coords[b].first, cb = unknown_coords[b].second;
                return composite_key(ra, ca) < composite_key(rb, cb);
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
                // Forced commit: record the pixel on the trail so a parent
                // backtrack frame can revert it if its own branch fails.
                const int idx_f = row * W + col;
                pic.pixels[idx_f] = FULL;
                pic.unknown_count -= 1;
                trail.changed_cell_indices.push_back(idx_f);
                pic.mark_row_dirty(row);
                pic.mark_col_dirty(col);
                return solve_real(mapped_rows, mapped_cols, pic, state, on_solution, trail);
            }

            if (empty_res.ok && !full_res.ok) {
                state.mark_contradiction();
                const int idx_e = row * W + col;
                pic.pixels[idx_e] = EMPTY;
                pic.unknown_count -= 1;
                trail.changed_cell_indices.push_back(idx_e);
                pic.mark_row_dirty(row);
                pic.mark_col_dirty(col);
                return solve_real(mapped_rows, mapped_cols, pic, state, on_solution, trail);
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

    // Iterative two-branch loop, no pic.copy(). Each branch records a trail
    // mark + small snapshot, applies the branch pixel, recurses, then on
    // normal return reverts pic to the entry state for the next branch.
    //
    // On stop signal (solve_real returns false) we propagate immediately
    // WITHOUT reverting or calling exit_backtrack — the caller is aborting,
    // pic state is no longer observed.
    //
    // exit_backtrack is invoked exactly once, after both branches complete.
    bool found_solution_in_first = false;
    for (int branch = 0; branch < 2; ++branch) {
        const std::int8_t val = (branch == 0) ? first_val : second_val;

        const std::size_t mark = trail.changed_cell_indices.size();
        const int saved_unknown_count = pic.unknown_count;

        // Apply the branch pixel directly (bypass set_pixel; record on trail).
        const int br_idx = row * W + col;
        pic.pixels[br_idx] = val;
        pic.unknown_count -= 1;
        trail.changed_cell_indices.push_back(br_idx);
        pic.mark_row_dirty(row);
        pic.mark_col_dirty(col);

        if (branch == 0) {
            OnSolution wrapped = [&](const Picture& p) {
                found_solution_in_first = true;
                return on_solution(p);
            };
            if (!solve_real(mapped_rows, mapped_cols, pic, state, wrapped, trail)) {
                return false;
            }
            revert_branch(pic, trail, mark, saved_unknown_count);
            if (!found_solution_in_first) {
                state.first_branch_failed();
            }
        } else {
            if (!solve_real(mapped_rows, mapped_cols, pic, state, on_solution, trail)) {
                return false;
            }
            revert_branch(pic, trail, mark, saved_unknown_count);
        }
    }

    state.exit_backtrack();
    return true;
}

// ---------------------------------------------------------------------------
// solve_real
// ---------------------------------------------------------------------------

bool solve_real(const std::vector<LineSpec>& mapped_rows,
                const std::vector<LineSpec>& mapped_cols,
                Picture& pic,
                SolveState& state,
                const OnSolution& on_solution,
                Trail& trail) {
    // Drain all dirty lines first (this both propagates and validates every
    // changed line via solve_line_batch's total==0). The board was consistent
    // on entry except for the freshly-dirtied lines, so this fully validates
    // it — replacing the old redundant solve_check full-board pass. The
    // is_solved() callback must come AFTER this drain so an invalid completing
    // assignment is rejected (solve_lines returns false) rather than accepted.
    while (pic.has_dirty()) {
        if (!solve_lines(mapped_rows, pic, true, trail)) {
            return true;
        }
        if (!solve_lines(mapped_cols, pic, false, trail)) {
            return true;
        }
    }

    if (pic.is_solved()) {
        state.solution_found();
        return on_solution(pic);
    }

    return solve_backtrack(mapped_rows, mapped_cols, pic, state, on_solution, trail);
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

void solve(const std::vector<std::vector<int>>& rows,
           const std::vector<std::vector<int>>& cols,
           std::function<bool(const Picture&)> on_solution,
           Strategy* out_strategy) {
    std::size_t budget = 1024ULL * 1024ULL * 1024ULL;  // 1 GB default
    const char* env = std::getenv("LINE_CACHE_BUDGET_MB");
    if (env != nullptr) {
        try {
            budget = static_cast<std::size_t>(std::stoull(env)) * 1024ULL * 1024ULL;
        } catch (...) {}
    }
    int max_line = static_cast<int>(std::max(rows.size(), cols.size()));
    reset_line_cache(max_line, budget);

    const int H = static_cast<int>(rows.size());
    const int W = static_cast<int>(cols.size());

    Picture pic(H, W);

    std::vector<LineSpec> mapped_rows;
    mapped_rows.reserve(rows.size());
    for (const auto& clue : rows) {
        mapped_rows.push_back(make_line_spec(clue));
    }

    std::vector<LineSpec> mapped_cols;
    mapped_cols.reserve(cols.size());
    for (const auto& clue : cols) {
        mapped_cols.push_back(make_line_spec(clue));
    }

    SolveState state;
    Trail trail;
    // Reserve enough headroom that the trail rarely reallocates.
    trail.changed_cell_indices.reserve(static_cast<std::size_t>(H) * static_cast<std::size_t>(W));
    (void)solve_real(mapped_rows, mapped_cols, pic, state, on_solution, trail);

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
