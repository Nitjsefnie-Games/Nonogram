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
#include <map>
#include <string>
#include <string_view>
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

// Owning key, stored in the cache. line_bytes holds the line's cell bytes.
struct LineKey {
    std::string line_bytes;
    const LineSpec* states_ptr;

    bool operator==(const LineKey& o) const noexcept {
        return states_ptr == o.states_ptr && line_bytes == o.line_bytes;
    }
};

// Borrowed lookup key: a view over the current line buffer, so cache HITS need
// no std::string allocation. Heterogeneous (transparent) lookup matches it
// against stored owning LineKeys.
struct LineKeyView {
    std::string_view line_bytes;
    const LineSpec* states_ptr;
};

inline std::size_t mix_line_key(std::string_view bytes, const LineSpec* p) noexcept {
    namespace wy = ankerl::unordered_dense::detail::wyhash;
    std::uint64_t h = wy::hash(bytes.data(), bytes.size());
    return wy::hash(h ^ reinterpret_cast<std::uintptr_t>(p));
}

struct LineKeyHash {
    using is_transparent = void;
    using is_avalanching = void;  // wyhash output is already well-mixed
    std::size_t operator()(const LineKey& k) const noexcept {
        return mix_line_key(k.line_bytes, k.states_ptr);
    }
    std::size_t operator()(const LineKeyView& k) const noexcept {
        return mix_line_key(k.line_bytes, k.states_ptr);
    }
};

struct LineKeyEq {
    using is_transparent = void;
    bool operator()(const LineKey& a, const LineKey& b) const noexcept {
        return a.states_ptr == b.states_ptr && a.line_bytes == b.line_bytes;
    }
    bool operator()(const LineKey& a, const LineKeyView& b) const noexcept {
        return a.states_ptr == b.states_ptr && std::string_view(a.line_bytes) == b.line_bytes;
    }
    bool operator()(const LineKeyView& a, const LineKey& b) const noexcept {
        return a.states_ptr == b.states_ptr && a.line_bytes == std::string_view(b.line_bytes);
    }
};

// Plain memoization map. The line cache only ever evicts on puzzles whose
// distinct-line working set exceeds the budget cap (rare, huge puzzles); the
// common case never fills it. So instead of an LRU (which stored every key
// twice and added a list-node indirection per hit), use a flat map directly:
// one key copy per miss, one fewer pointer chase per hit, smaller entries.
// Eviction for the OOM-safety case is a generational full clear at capacity.
class LineCache {
public:
    std::size_t max_entries_ = 1'000'000;  // default; set by reset_line_cache

    void set_max_entries(std::size_t n) { max_entries_ = std::max<std::size_t>(1, n); }
    void set_reserve_enabled(bool e) { reserve_enabled_ = e; }

    void clear() { map_.clear(); reserve_at_ = kReserveTrigger; }

    LineSolveResult* find_and_promote(const LineKeyView& key) {
        auto it = map_.find(key);
        return it == map_.end() ? nullptr : &it->second;
    }

    LineSolveResult* insert(LineKey key, LineSolveResult value) {
        if (map_.size() >= max_entries_) {
            map_.clear();  // OOM-safe generational eviction (cold path)
            reserve_at_ = kReserveTrigger;
        }
        // Lazy reserve-ahead: only once a workload proves large (so small/medium
        // puzzles never pay the upfront allocation) reserve well ahead of need,
        // eliminating the repeated ankerl rehash-moves that dominate the insert
        // path on big enumerations. Enabled only in anytime mode so the default
        // path's timings (and over-reserve on medium enumerations) are untouched.
        if (reserve_enabled_ && map_.size() >= reserve_at_ && reserve_at_ < max_entries_) {
            reserve_at_ = std::min(max_entries_, reserve_at_ * 8);
            map_.reserve(reserve_at_);
        }
        auto res = map_.emplace(std::move(key), std::move(value));
        return &res.first->second;
    }

private:
    static constexpr std::size_t kReserveTrigger = 1u << 18;  // 262144
    std::size_t reserve_at_ = kReserveTrigger;
    bool reserve_enabled_ = false;
    ankerl::unordered_dense::map<LineKey, LineSolveResult, LineKeyHash, LineKeyEq> map_;
};

LineCache& line_cache() {
    static LineCache cache;
    return cache;
}

void reset_line_cache(int max_line_len, std::size_t budget_bytes, bool reserve_enabled) {
    auto& c = line_cache();
    c.clear();
    std::size_t per_entry = 295 + 2 * static_cast<std::size_t>(max_line_len);
    c.set_max_entries(std::max<std::size_t>(1, budget_bytes / per_entry));
    c.set_reserve_enabled(reserve_enabled);
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

    int solutions_found = 0;
    bool used_contradiction = false;
    bool used_backtrack = false;
    bool skip_probing = false;
    bool keep_probing = false;  // anytime mode: never latch skip_probing
    std::deque<int> probe_outcomes;

    // Benchmark hook: if MAX_NODES is set (>0), abort the search after that many
    // backtrack nodes. Lets pikachu-class never-terminating puzzles be timed
    // over a deterministic, fixed amount of work. 0 = unlimited (normal).
    std::uint64_t node_limit = 0;
    std::uint64_t nodes = 0;

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
        const char* nenv = std::getenv("MAX_NODES");
        if (nenv != nullptr) {
            try {
                node_limit = std::stoull(nenv);
            } catch (...) {
                node_limit = 0;
            }
        }
    }

    void record_probe(bool found_contradiction) {
        // In anytime mode skip_probing can never latch, so the whole yield
        // window is dead work on every probe -- skip it entirely.
        if (keep_probing || skip_probing) return;
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
            if (yield_rate < PROBING_THRESHOLD && !keep_probing) {
                skip_probing = true;
            }
        }
    }

    void solution_found() { solutions_found += 1; }
    void mark_contradiction() { used_contradiction = true; }
    void mark_backtrack() { used_backtrack = true; }
};

// ---------------------------------------------------------------------------
// Forward declarations.
// ---------------------------------------------------------------------------

using OnSolution = std::function<bool(const Picture&)>;

bool solve_real(const std::vector<const LineSpec*>& mapped_rows,
                const std::vector<const LineSpec*>& mapped_cols,
                Picture& pic,
                SolveState& state,
                const OnSolution& on_solution,
                Trail& trail);

bool solve_backtrack(const std::vector<const LineSpec*>& mapped_rows,
                     const std::vector<const LineSpec*>& mapped_cols,
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
    const std::vector<int>* deductions; // packed (pos,val); points into cache entry
};

BatchResult solve_one_batch(const LineSpec& spec,
                            int index,
                            bool is_col,
                            Picture& pic) {
    // Obtain the line cells. Rows are contiguous in pic.pixels, so we point
    // straight at them (no copy). Columns are strided, so gather into a
    // reusable thread_local buffer.
    const int W = pic.width();
    const std::int8_t* px = pic.pixels.data();
    const std::int8_t* line;
    std::size_t line_n;
    if (is_col) {
        static thread_local std::vector<std::int8_t> col_buf;
        const int H = pic.height();
        col_buf.resize(static_cast<std::size_t>(H));
        // Hoist the thread_local base pointer into a local; otherwise the
        // compiler reloads col_buf.data() from TLS (fs:) on every iteration.
        std::int8_t* cb = col_buf.data();
        const std::int8_t* src = px + index;
        for (int r = 0; r < H; ++r) {
            cb[r] = src[static_cast<std::size_t>(r) * W];
        }
        line = cb;
        line_n = static_cast<std::size_t>(H);
    } else {
        line = px + static_cast<std::size_t>(index) * W;
        line_n = static_cast<std::size_t>(W);
    }

    std::string_view view_bytes(reinterpret_cast<const char*>(line), line_n);
    LineKeyView vkey{view_bytes, &spec};
    auto& cache = line_cache();
    const LineSolveResult* result_ptr = cache.find_and_promote(vkey);
    if (result_ptr == nullptr) {
        LineSolveResult res = solve_line_batch(line, line_n, spec);
        LineKey okey{std::string(view_bytes), &spec};
        result_ptr = cache.insert(std::move(okey), std::move(res));
    }

    if (result_ptr->total == 0) {
        return BatchResult{false, nullptr};
    }

    if (result_ptr->deductions.empty()) {
        return BatchResult{true, nullptr};
    }

    return BatchResult{true, &result_ptr->deductions};
}

// ---------------------------------------------------------------------------
// write_intersection: apply the deduced positions/values to pic, marking
// the cross-direction dirty for any cells that go from UNKNOWN to a value.
//
// Every pixel transition UNKNOWN -> value is recorded into trail (linear cell
// index r*W + c). On revert, the caller truncates the trail and walks each
// recorded index back to UNKNOWN.
// ---------------------------------------------------------------------------

void write_intersection(const std::vector<int>& deductions,
                        int line_index,
                        Picture& pic,
                        bool is_row,
                        Trail& trail) {
    if (is_row) {
        const int row = line_index;
        const int W = pic.width();
        std::int8_t* px = pic.pixels.data() + static_cast<std::size_t>(row) * static_cast<std::size_t>(W);
        for (int enc : deductions) {
            const int col = deduce_pos(enc);
            if (px[col] == UNKNOWN) {
                px[col] = deduce_val(enc);
                pic.unknown_count -= 1;
                pic.mark_col_dirty(col);
                trail.changed_cell_indices.push_back(row * W + col);
            }
        }
    } else {
        const int col = line_index;
        const int W = pic.width();
        for (int enc : deductions) {
            const int row = deduce_pos(enc);
            std::int8_t& cell = pic.pixels[static_cast<std::size_t>(row) * static_cast<std::size_t>(W) + static_cast<std::size_t>(col)];
            if (cell == UNKNOWN) {
                cell = deduce_val(enc);
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

bool solve_lines(const std::vector<const LineSpec*>& mapped,
                 Picture& pic,
                 bool is_row,
                 Trail& trail) {
    auto& queue = is_row ? pic.row_queue : pic.col_queue;
    auto& dirty = is_row ? pic.row_dirty : pic.col_dirty;
    while (!queue.empty()) {
        const int index = queue.front();
        queue.pop_front();
        dirty[index] = 0;
        BatchResult r = solve_one_batch(*mapped[index], index, !is_row, pic);
        if (!r.success) {
            return false;
        }
        if (r.deductions != nullptr) {
            write_intersection(*r.deductions, index, pic, is_row, trail);
        }
    }
    return true;
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
                       const std::vector<const LineSpec*>& mapped_rows,
                       const std::vector<const LineSpec*>& mapped_cols,
                       Picture& pic) {
    // Reusable per-probe trail. probe_cell never nests (it calls only
    // solve_lines, which never probes), and the previous probe's ProbeGuard
    // already reverted every cell it touched, so we just clear the index buffer
    // and reuse its capacity — no per-probe heap allocation.
    static thread_local Trail trail;
    trail.changed_cell_indices.clear();

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

bool solve_backtrack(const std::vector<const LineSpec*>& mapped_rows,
                     const std::vector<const LineSpec*>& mapped_cols,
                     Picture& pic,
                     SolveState& state,
                     const OnSolution& on_solution,
                     Trail& trail) {
    // Benchmark abort: stop after node_limit backtrack nodes (return false
    // propagates as a stop signal, exactly like the --max callback).
    if (state.node_limit && ++state.nodes > state.node_limit) {
        return false;
    }

    const int H = pic.height();
    const int W = pic.width();
    const std::int8_t* px = pic.pixels.data();

    // Reusable scratch buffers. Safe to share across recursive solve_backtrack
    // frames: every frame fully consumes these during cell-selection, BEFORE
    // any recursive solve_real call (forced-commit returns immediately; the
    // branch loop only touches plain locals). Avoids 4 heap allocations per
    // backtrack node.
    static thread_local std::vector<int> unknowns_in_row;
    static thread_local std::vector<int> unknowns_in_col;
    unknowns_in_row.assign(static_cast<std::size_t>(H), 0);
    unknowns_in_col.assign(static_cast<std::size_t>(W), 0);

    // Per-row / per-col UNKNOWN counts, used as the "most constrained line"
    // signal in the composite sort key (lower => more constrained line).
    int n_unknown = 0;
    for (int r = 0; r < H; ++r) {
        const std::int8_t* row_px = px + static_cast<std::size_t>(r) * W;
        for (int c = 0; c < W; ++c) {
            if (row_px[c] == UNKNOWN) {
                unknowns_in_row[r] += 1;
                unknowns_in_col[c] += 1;
                ++n_unknown;
            }
        }
    }

    if (n_unknown == 0) {
        return true;
    }

    int best_row = -1;
    int best_col = -1;
    std::int8_t best_first_val = FULL;

    // Neighbor score: count of filled (non-UNKNOWN) orthogonal neighbours,
    // out-of-bounds counting as 1. Computed on demand only for the unknown
    // cells we actually score (mirrors the old get_neighbor_scores values).
    auto neighbor_score = [&](int r, int c) -> int {
        int s = 0;
        s += (r == 0)     ? 1 : (px[(r - 1) * W + c] != UNKNOWN);
        s += (r == H - 1) ? 1 : (px[(r + 1) * W + c] != UNKNOWN);
        s += (c == 0)     ? 1 : (px[r * W + (c - 1)] != UNKNOWN);
        s += (c == W - 1) ? 1 : (px[r * W + (c + 1)] != UNKNOWN);
        return s;
    };

    // Composite key: ascending line_constraint, then descending neighbor.
    // Returned via a pair so std::less ordering matches the desired ordering
    // (we negate neighbor so larger neighbor compares smaller, i.e. wins ties).
    auto composite_key = [&](int r, int c) {
        int line_constraint = unknowns_in_row[r] + unknowns_in_col[c];
        return std::pair<int, int>(line_constraint, -neighbor_score(r, c));
    };

    if (state.skip_probing) {
        // Pick the unknown cell with the smallest composite key (most-
        // constrained line; tie-broken by highest neighbor score). Single
        // row-major scan, no coords/scores arrays materialized.
        std::pair<int, int> best_key;
        bool have = false;
        for (int r = 0; r < H; ++r) {
            const std::int8_t* row_px = px + static_cast<std::size_t>(r) * W;
            for (int c = 0; c < W; ++c) {
                if (row_px[c] != UNKNOWN) continue;
                std::pair<int, int> k = composite_key(r, c);
                if (!have || k < best_key) {
                    best_key = k;
                    best_row = r;
                    best_col = c;
                    have = true;
                }
            }
        }
        best_first_val = FULL;
    } else {
        // Collect unknown coords in row-major order (matches np.argwhere order)
        // with a single-int sort key encoding the composite ordering: ascending
        // line_constraint, then descending neighbor (0..4). enc = lc*5 + (4-nb)
        // is order-equivalent to the (lc, -nb) pair, and its range is tiny, so a
        // stable counting sort (O(n), no merge buffer / comparator indirection)
        // replaces the O(n log n) stable_sort while producing the identical order.
        static thread_local std::vector<std::pair<int, int>> unknown_coords;
        static thread_local std::vector<int> enc;
        unknown_coords.clear();
        enc.clear();
        int max_enc = 0;
        for (int r = 0; r < H; ++r) {
            const std::int8_t* row_px = px + static_cast<std::size_t>(r) * W;
            for (int c = 0; c < W; ++c) {
                if (row_px[c] != UNKNOWN) continue;
                unknown_coords.emplace_back(r, c);
                int lc = unknowns_in_row[r] + unknowns_in_col[c];
                int e = lc * 5 + (4 - neighbor_score(r, c));
                enc.push_back(e);
                if (e > max_enc) max_enc = e;
            }
        }

        const std::size_t nu = unknown_coords.size();
        static thread_local std::vector<int> order;
        static thread_local std::vector<int> bucket;
        order.resize(nu);
        bucket.assign(static_cast<std::size_t>(max_enc) + 2, 0);
        for (std::size_t i = 0; i < nu; ++i) bucket[enc[i] + 1]++;
        for (int k = 1; k <= max_enc + 1; ++k) bucket[k] += bucket[k - 1];
        for (std::size_t i = 0; i < nu; ++i) order[bucket[enc[i]]++] = static_cast<int>(i);

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

    // Iterative two-branch loop, no pic.copy(). Each branch records a trail
    // mark + small snapshot, applies the branch pixel, recurses, then on
    // normal return reverts pic to the entry state for the next branch.
    // On stop signal (solve_real returns false) we propagate immediately
    // without reverting — the caller is aborting, pic state is no longer
    // observed.
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

        if (!solve_real(mapped_rows, mapped_cols, pic, state, on_solution, trail)) {
            return false;
        }
        revert_branch(pic, trail, mark, saved_unknown_count);
    }

    return true;
}

// ---------------------------------------------------------------------------
// solve_real
// ---------------------------------------------------------------------------

bool solve_real(const std::vector<const LineSpec*>& mapped_rows,
                const std::vector<const LineSpec*>& mapped_cols,
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
           Strategy* out_strategy,
           bool keep_probing) {
    std::size_t budget = 1024ULL * 1024ULL * 1024ULL;  // 1 GB default
    const char* env = std::getenv("LINE_CACHE_BUDGET_MB");
    if (env != nullptr) {
        try {
            budget = static_cast<std::size_t>(std::stoull(env)) * 1024ULL * 1024ULL;
        } catch (...) {}
    }
    int max_line = static_cast<int>(std::max(rows.size(), cols.size()));
    const bool anytime = keep_probing || (std::getenv("ANYTIME") != nullptr);
    reset_line_cache(max_line, budget, anytime);

    const int H = static_cast<int>(rows.size());
    const int W = static_cast<int>(cols.size());

    Picture pic(H, W);

    // Dedupe LineSpecs by clue. A line solve depends only on (content, clue),
    // not on which row/col it is, so same-clue lines can share a LineSpec and
    // therefore share line-cache entries -> smaller working set, fewer misses.
    // spec_pool is reserved to its upper bound so its element addresses (used as
    // stable cache-key identities) never move.
    std::vector<LineSpec> spec_pool;
    spec_pool.reserve(rows.size() + cols.size());
    std::map<std::vector<int>, const LineSpec*> clue_to_spec;
    auto get_spec = [&](const std::vector<int>& clue) -> const LineSpec* {
        auto it = clue_to_spec.find(clue);
        if (it != clue_to_spec.end()) return it->second;
        spec_pool.push_back(make_line_spec(clue));
        const LineSpec* p = &spec_pool.back();
        clue_to_spec.emplace(clue, p);
        return p;
    };

    std::vector<const LineSpec*> mapped_rows;
    mapped_rows.reserve(rows.size());
    for (const auto& clue : rows) mapped_rows.push_back(get_spec(clue));

    std::vector<const LineSpec*> mapped_cols;
    mapped_cols.reserve(cols.size());
    for (const auto& clue : cols) mapped_cols.push_back(get_spec(clue));

    SolveState state;
    state.keep_probing = anytime;
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
