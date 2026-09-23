#include "search.hpp"

#include "clauses.hpp"
#include "lines.hpp"
#include "picture.hpp"
#include "types.hpp"

// Vendored third-party hash map: ankerl::unordered_dense (MIT, v4.8.1).
// See cpp/external/ankerl/unordered_dense.h for license/copyright header.
// Used for the line-batch memoization cache in this file.
#include "external/ankerl/unordered_dense.h"

#include <sys/mman.h>
#include <x86intrin.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <climits>
#include <cmath>
#include <csignal>
#include <cstring>
#include <deque>
#include <functional>
#include <list>
#include <map>
#include <random>
#include <string>
#include <string_view>
#include <type_traits>
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

// Namespace-scope instance (not a function-local static): the hot path calls
// into it on every line solve and a function-local static costs an init-guard
// check per call.
LineCache g_legacy_cache;

// ---------------------------------------------------------------------------
// Fast line cache: lines of at most kMaxFastCells cells.
//
// Keyed on Picture's incrementally-maintained 2-bit packed line keys plus a
// 16-bit tag (spec id, row/col), so a lookup hashes key_words machine words
// instead of the line's bytes and never gathers a column. Open addressing
// with linear probing over fixed-size slots that hold the key, the tag and the
// deductions INLINE (one byte each: pos | val << 7), so a hit touches exactly
// one cache line. The previous map paid three to four dependent misses per
// lookup (bucket, value entry, heap key string, deductions vector).
//
// Slots are 32 bytes for keys of up to 2 words (lines <= 64 cells), 64 bytes
// beyond. On a deep enumeration the hot table is far larger than L3, so the
// slot size sets the working set: with 32-byte slots pikachu's ~1.8M-entry
// table is ~58 MB instead of ~115 MB. Measured on pikachu, 97% of entries
// carry <= 12 deductions, the inline room a 32-byte slot leaves for a 2-word
// key; the rest spill to an arena and pay one extra miss on a hit.
// The table is mapped with MADV_HUGEPAGE so lookups do not also miss the TLB.
//
// Slot layout (byte offsets, kw = key words, S = slot bytes):
//   [0, 8kw)        key words
//   [8kw, 8kw+2)    tag: bits 0-9 spec id * 2 + orientation, bits 10-15 a
//                   hash fingerprint, stored complemented so that a zero
//                   word marks a free slot and a fresh zero-filled table
//                   needs no clearing pass (a tag of 0xFFFF, id 511 with
//                   fingerprint 63, is indistinguishable from free: its
//                   entry is simply lost, as it was with 0xFFFF as the
//                   free marker). Many entries share a spec, so the id
//                   alone let most same-spec probe neighbours through to the
//                   key compare; the fingerprint rejects 63/64 of them.
//   [8kw+2]         deduction count
//   [8kw+3]         flags: bit0 = line satisfiable, bit1 = deductions spilled
//   [8kw+4, S)      deductions inline, or (spilled) uint32 offset into arena_
// ---------------------------------------------------------------------------

// Solution counts: products of region counts pass 2^64 on real puzzles.
__extension__ typedef unsigned __int128 u128;

constexpr int kMaxFastCells = 128;   // 7-bit positions in the deduction bytes
constexpr std::uint16_t kEmptyStored = 0;  // complemented tag of a free slot
constexpr std::uint8_t kFlagSat = 1;
constexpr std::uint8_t kFlagSpilled = 2;
constexpr std::size_t kHugePage = 2u << 20;

// Zero-filled buffer for a hash table. Below one huge page it is ordinary
// aligned heap memory cleared here. From one huge page up it is a fresh
// anonymous mapping trimmed to a 2 MB boundary, advised for transparent
// huge pages and prefaulted (MADV_POPULATE_WRITE): the kernel zeroes the
// pages once as it faults them in, and the table needs no clearing pass
// of its own -- the second zeroing of every grown table was 4% of a 90 x
// 69 count run and 1.8% of easy_medium/108. Prefaulting keeps the faults
// out of the random-access lookups (page-by-page faulting on a mapping
// touched at random got 4 KB pages and 4.6x the kernel time).
struct SlotBuf {
    unsigned char* p = nullptr;
    std::size_t bytes = 0;
    bool mapped = false;
    SlotBuf() = default;
    SlotBuf(const SlotBuf&) = delete;
    SlotBuf& operator=(const SlotBuf&) = delete;
    ~SlotBuf() { release(); }
    void release() {
        if (p != nullptr) {
            if (mapped) munmap(p, bytes); else std::free(p);
        }
        p = nullptr;
        bytes = 0;
        mapped = false;
    }
    void allocate(std::size_t n) {
        release();
        if (n < kHugePage) {
            // A 2 MB alignment on a small table faults in a whole zeroed
            // hugepage, ~0.3 ms of startup on puzzles that solve in 0.1 ms.
            void* mem = nullptr;
            if (posix_memalign(&mem, 64, n) != 0) throw std::bad_alloc();
            std::memset(mem, 0, n);
            p = static_cast<unsigned char*>(mem);
            bytes = n;
            return;
        }
        const std::size_t len = (n + kHugePage - 1) / kHugePage * kHugePage;
        void* raw = mmap(nullptr, len + kHugePage, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (raw == MAP_FAILED) throw std::bad_alloc();
        const std::uintptr_t lo = reinterpret_cast<std::uintptr_t>(raw);
        const std::uintptr_t start = (lo + kHugePage - 1) / kHugePage * kHugePage;
        if (start > lo) munmap(raw, start - lo);
        const std::uintptr_t end = start + len;
        if (end < lo + len + kHugePage) munmap(reinterpret_cast<void*>(end), lo + len + kHugePage - end);
        p = reinterpret_cast<unsigned char*>(start);
        bytes = len;
        mapped = true;
        madvise(p, len, MADV_HUGEPAGE);        // advisory; failure is harmless
        madvise(p, len, MADV_POPULATE_WRITE);  // prefault; on failure pages fault in on first touch
    }
    void swap(SlotBuf& o) { std::swap(p, o.p); std::swap(bytes, o.bytes); std::swap(mapped, o.mapped); }
};

// DEBUG_CACHE_STATS=1: count line-cache lookups / misses and histogram the
// deduction count of inserted entries, printed to stderr when solve() returns.
// Inert when unset.
// (Needs a `make stats` build, -DNONOGRAM_STATS. In the default build the
// flag is a constexpr false so every counter folds away: the run-time checks
// alone measured 3% of instructions on the anytime hot path.)
#ifdef NONOGRAM_STATS
const bool g_debug_stats = std::getenv("DEBUG_CACHE_STATS") != nullptr;
#else
constexpr bool g_debug_stats = false;
#endif
std::uint64_t g_stat_lookups = 0;
std::uint64_t g_stat_misses = 0;
std::uint64_t g_stat_probes = 0;
std::uint64_t g_stat_ded_hist[kMaxFastCells + 1] = {};
std::uint64_t g_stat_unsat = 0, g_stat_noded = 0, g_stat_ded = 0;  // lookup outcomes
std::uint64_t g_stat_nodes = 0;  // solve_backtrack entries that branched
std::uint64_t g_stat_probe_lookups_hist[64] = {};  // lookups per probe (capped)
std::uint64_t g_stat_probe_cur = 0;
std::uint64_t g_stat_probe_ok = 0;
// State cache by the node's unknown-cell count (bucket = log2): lookups,
// hits, and the branch nodes the hits saved (sum of 2^work).
std::uint64_t g_stat_sc_lookups[24] = {}, g_stat_sc_hits[24] = {}, g_stat_sc_saved[24] = {}, g_stat_sc_gated[24] = {};
// Cycles spent in subtrees rooted at the topmost node with <= 31 / <= 15
// unknown cells (what a direct small-region counter could replace), and
// in the whole solve.
std::uint64_t g_stat_small31_cycles = 0, g_stat_small15_cycles = 0, g_stat_small31_roots = 0, g_stat_small15_roots = 0;
int g_stat_small_depth31 = 0, g_stat_small_depth15 = 0;
struct SmallTimer {
    std::uint64_t t0 = 0;
    int which = 0;  // 0 none, 1 = <=31 root, 2 = <=15 root (and 31 if also topmost)
    bool root31 = false, root15 = false;
    SmallTimer(int n_unknown) {
        if (!g_debug_stats) return;
        if (n_unknown <= 31 && g_stat_small_depth31++ == 0) { root31 = true; ++g_stat_small31_roots; }
        if (n_unknown <= 15 && g_stat_small_depth15++ == 0) { root15 = true; ++g_stat_small15_roots; }
        if (root31 || root15) t0 = __rdtsc();
        which = (n_unknown <= 31) + (n_unknown <= 15);
    }
    ~SmallTimer() {
        if (!g_debug_stats) return;
        if (which >= 1) --g_stat_small_depth31;
        if (which >= 2) --g_stat_small_depth15;
        if (root31 || root15) {
            const std::uint64_t dt = __rdtsc() - t0;
            if (root31) g_stat_small31_cycles += dt;
            if (root15) g_stat_small15_cycles += dt;
        }
    }
};

// Below this many unknown cells a node (other than the root, which keeps
// probing so the strategy label of a puzzle solved there is unchanged)
// branches by the cheap latched scan instead of probing every cell: a
// node with 31 cells pays 62 probes, each a line solve plus propagation,
// to pick one of them, and on the partially solved puzzles the subtrees
// under such nodes were 40-63% of the run. Measured (count mode, state
// cache on), at 31: 3867 reaches in 60 s what took 300 s (its 812-cell
// region 12.5% -> 49.9% explored), 7290 0.14% -> 0.79%, 10810 unchanged;
// corpus branch nodes +9.3% for probes -27% (7382 532k -> 635k nodes),
// counts and labels unchanged. 47 against 31, 60 s: 7785 +21%, 9798
// +19%, 2712 / 7290 / 10810 / 5903 level, corpus nodes +1.7%, 7382 12.7
// -> 10.8 s, 12130 2.8 -> 1.9 s. 63 grows the corpus 37% and loses 7290
// 4.4x; 16 is half the gain.
constexpr int kSmallNoProbe = 47;

// BRANCH_K=<k> (stats build only): score branch cells by k*min - max instead
// of the shipped (min, then smaller max) order, for exploring the balance
// penalty. Values of 4-8 halve 9-Dom's tree again and cut webpbn 12548 13x,
// but 18297 grows 7-18x and k=3 grows medium/3929 by 50%, so it is not
// shipped. Unset = the shipped order.
#ifdef NONOGRAM_STATS
const bool g_branch_k_set = std::getenv("BRANCH_K") != nullptr;
const double g_branch_k = g_branch_k_set ? std::atof(std::getenv("BRANCH_K")) : 0.0;
#else
constexpr bool g_branch_k_set = false;
constexpr double g_branch_k = 0.0;
#endif

// DEBUG_IMPL=<n> (stats build only): implication-graph experiment. Every
// successful probe (p=c) that settles q=v yields the implication p_c -> q_v
// and its contrapositive q_!v -> p_!c (Wu et al. 2013's FP2 insight;
// direct implications are already transitively closed by propagation, so
// only paths that use a contrapositive edge can add anything). A literal
// that reaches its own negation is false. n=1 counts what that would force
// at each branch node; n=2 commits those cells and restarts the node.
// Measured (with the balanced tie-break): 9-Dom 10 forced cells over 3,232
// nodes (tree 3,232 -> 3,231), extreme/6574 none over 154 nodes,
// easy_large/5281 3 cells and a slightly larger tree, so it is not shipped.
#ifdef NONOGRAM_STATS
const int g_debug_impl = std::getenv("DEBUG_IMPL") ? std::atoi(std::getenv("DEBUG_IMPL")) : 0;
// ANYTIME_TB=1 breaks anytime-mode ties on the max toward the larger min,
// =2 toward the smaller min (the shipped order is first-found).
const int g_anytime_tb = std::getenv("ANYTIME_TB") ? std::atoi(std::getenv("ANYTIME_TB")) : 0;
// BRANCH_MAX=1 scores branch cells by the larger probe fill (the anytime
// order) in every mode. Count mode then clears dead and sparse subtrees
// far faster (300 s: pikachu 47% of the space against 10%, 7785 50%
// against 8%, 13480 50% against 11% with zero solutions in that half)
// but exhaustive trees grow: corpus +24% branch nodes and 9x probes,
// 9-Dom 3,232 -> 114,021 nodes, 6574 154 -> 102,529. Not shipped.
const bool g_branch_max = std::getenv("BRANCH_MAX") != nullptr;
// FIRST_VAL=1 explores the branch value whose probe settled FEWER cells
// first (the shipped order explores the one that settled more).
const bool g_first_val_low = std::getenv("FIRST_VAL") != nullptr;
// NO_SKIP=1 never latches the adaptive probing shut-off in default mode.
const bool g_no_skip = std::getenv("NO_SKIP") != nullptr;
// SMALL_NOPROBE=<n> overrides the small-node probing cut-off (see
// kSmallNoProbe; 0 disables it).
const int g_small_noprobe = std::getenv("SMALL_NOPROBE") ? std::atoi(std::getenv("SMALL_NOPROBE")) : kSmallNoProbe;
// EARLY_SOLVE=1 (anytime mode): stop the probe pass at the first probe that
// completes the grid and branch on that cell. Measured on pikachu --anytime
// --max 300000: probes 23.1M -> 21.7M, wall within noise (7.10/7.54/7.01s
// vs 7.11/6.75/6.85s), so not shipped.
const bool g_early_solve = std::getenv("EARLY_SOLVE") != nullptr;
// PROBE_WINDOW / PROBE_THRESH override the shut-off window and yield threshold.
const std::size_t g_probe_window = std::getenv("PROBE_WINDOW") ? std::strtoull(std::getenv("PROBE_WINDOW"), nullptr, 10) : 100;
const double g_probe_thresh = std::getenv("PROBE_THRESH") ? std::atof(std::getenv("PROBE_THRESH")) : 0.01;
// DEAD_WINDOW / DEAD_FRAC override the dead-work watchdog's window and fraction.
const std::uint64_t g_dead_window = std::getenv("DEAD_WINDOW") ? std::strtoull(std::getenv("DEAD_WINDOW"), nullptr, 10) : 4096;
const double g_dead_frac = std::getenv("DEAD_FRAC") ? std::atof(std::getenv("DEAD_FRAC")) : 0.9;
// NO_STATE_CACHE=1 (count mode) turns the region state cache off, for
// measuring what it saves (see SolveState::state_cache).
const bool g_no_state_cache = std::getenv("NO_STATE_CACHE") != nullptr;
// STATE_CACHE_PROBE_ONLY=1 keys and looks up every node but never takes a
// hit, so the tree is the no-cache tree and the instruction delta against
// NO_STATE_CACHE=1 is the cache's own cost.
const bool g_state_cache_probe_only = std::getenv("STATE_CACHE_PROBE_ONLY") != nullptr;
// NO_PROBE_SKIP=1 probes every cell both ways even when an earlier probe
// of the pass bounds the result (see ProbeBounds), for measuring the skip.
const bool g_no_probe_skip = std::getenv("NO_PROBE_SKIP") != nullptr;
// STATE_CACHE_YIELD=<x> overrides the per-bucket gate's threshold (nodes
// saved per lookup below which a node-size bucket stops using the cache;
// 0 never gates).
const double g_state_cache_yield = std::getenv("STATE_CACHE_YIELD") ? std::atof(std::getenv("STATE_CACHE_YIELD")) : 0.5;
#else
constexpr double g_state_cache_yield = 0.5;
constexpr bool g_no_state_cache = false;
constexpr bool g_state_cache_probe_only = false;
constexpr bool g_no_probe_skip = false;
constexpr std::uint64_t g_dead_window = 4096;
constexpr double g_dead_frac = 0.9;
constexpr int g_debug_impl = 0;
constexpr int g_anytime_tb = 0;
constexpr bool g_branch_max = false;
constexpr bool g_first_val_low = false;
constexpr bool g_no_skip = false;
constexpr int g_small_noprobe = kSmallNoProbe;
constexpr bool g_early_solve = false;
constexpr std::size_t g_probe_window = 100;
constexpr double g_probe_thresh = 0.01;
#endif
std::uint64_t g_stat_probe_pairs = 0, g_stat_probe_hits = 0;  // probe pairs, and those with a contradiction

// Probe memo. A probe (cell, value) at one node solved a set of lines --
// the rows and columns of the cells it settled, the probe cell included,
// since a line is solved exactly when a cell of it changed. If none of
// those lines has had a committed change since (a set or a revert on the
// solve() trail; a probe's own reverted changes do not count), replaying
// the probe now solves the same contents in the same order, reaches the
// same fixpoint with the same settled cells (all still unknown), and meets
// the same contradiction if it met one. So the memo answers the probe:
// same outcome, same fill, hence the same branch choice, forced cells and
// tree, bit for bit. Each line carries the generation of its last
// committed change; a memo carries the generation it was taken at and is
// valid while no line it touched has a later stamp.
// Measured (count mode, corpus gate 0 mismatches, nodes identical): it
// answers 25-29% of the probes on hard/3867, easy_large/7382 and
// hard/23210 and cuts the corpus's probes 19.9%, instructions 3867 (150k
// nodes) -5.3%, 7382 -3.8%, 12130 -0.4% -- but the probes it answers are
// the short, cache-hot ones, and the memo line and stamp loads plus a
// 32-byte store per probe cost more cycles than those saved: quiet core,
// medians of 7, 3867 +0.3%, 7382 +3.6%, 12130 +3.0%. Storing memos only
// for probes of <= 12 cells is no better. Not shipped: PROBE_MEMO=1 in
// the stats build.
// A cell's two memos are one 64-byte line: 32-byte entries, three words of
// lines (H + W <= 192), a 32-bit generation. The generation wraps after
// 4 billion committed changes; then every memo is dropped and the stamps
// restart, so a stale memo can never read as valid (a wrong answer would
// force a cell wrongly and lose solutions).
struct ProbeMemo {
    std::uint32_t gen = 0;          // 0 = none
    std::uint16_t fill = 0;         // cells the probe settled, itself included
    std::uint8_t ok = 0;            // consistent
    std::uint8_t pad = 0;
    std::uint64_t touched[3] = {};  // lines: rows, then columns at H + c
};
struct alignas(64) ProbeMemoPair { ProbeMemo v[2]; };  // [0] EMPTY, [1] FULL
static_assert(sizeof(ProbeMemoPair) == 64, "a cell's memo pair must be one cache line");
constexpr int kMemoWords = 3;
#ifdef NONOGRAM_STATS
const bool g_probe_memo = std::getenv("PROBE_MEMO") != nullptr;
bool g_memo_enabled = false;              // this puzzle fits the memo: fast mode, H + W <= 192
#else
constexpr bool g_probe_memo = false;
constexpr bool g_memo_enabled = false;    // every memo hook folds away
#endif
int g_memo_H = 0, g_memo_words = 0;
std::vector<ProbeMemoPair> g_memo;        // by row * W + col
std::vector<std::uint32_t> g_line_stamp;  // per line, the generation of its last committed change
std::uint32_t g_gen = 1;
std::uint64_t g_stat_memo_checked = 0, g_stat_memo_valid = 0;
inline void memo_next_gen() {
    if (++g_gen == 0) {  // wrapped: forget everything
        std::fill(g_memo.begin(), g_memo.end(), ProbeMemoPair{});
        std::fill(g_line_stamp.begin(), g_line_stamp.end(), 0u);
        g_gen = 1;
    }
}
// Every memo stops being valid: a probe answer recorded before a clause was
// learnt may be one the clause would now change (the stamps see only line
// changes). Stamps every line with a generation no memo has yet.
inline void memo_invalidate_all() {
    memo_next_gen();
    std::fill(g_line_stamp.begin(), g_line_stamp.end(), g_gen);
}
inline void stamp_cell(int row, int col) {
    if (g_memo_enabled) {
        g_line_stamp[static_cast<std::size_t>(row)] = g_gen;
        g_line_stamp[static_cast<std::size_t>(g_memo_H + col)] = g_gen;
    }
}
// Fraction of the search space completed so far; see SolveState::branch_depth.
double g_explored_mass = 0.0;
volatile std::sig_atomic_t g_stop_requested = 0;
std::string g_stop_path;  // SolveState::path_string() when the stop was taken
// Periodic progress line (see set_progress_interval).
double g_progress_interval = 0.0;
std::chrono::steady_clock::time_point g_progress_start, g_progress_next;

// Solutions in the finished subtrees so far: every open node's finished
// children summed (a node adds each child's count as it returns and takes
// its own total back when it returns to its parent, so nothing is counted
// twice; a region split's product joins only once the split is complete).
u128 g_counted_so_far = 0;
bool g_first_solution_printed = false;

std::string u128_str(u128 v) {
    std::string s;
    do { s.insert(s.begin(), static_cast<char>('0' + static_cast<int>(v % 10))); v /= 10; } while (v != 0);
    for (int i = static_cast<int>(s.size()) - 3; i > 0; i -= 3) s.insert(static_cast<std::size_t>(i), ",");
    return s;
}

std::string eta_str(double seconds) {
    // A fraction below the print precision extrapolates to 1e60 seconds
    // and more; past a million years the number carries no information
    // (and "%.1fd" of it overflowed the buffer, dropping the unit), so
    // print "inf". The comparison chain also sends NaN there.
    constexpr double year = 365.25 * 86400.0;
    char buf[64];
    if (seconds < 120.0) std::snprintf(buf, sizeof buf, "%.0fs", seconds);
    else if (seconds < 7200.0) std::snprintf(buf, sizeof buf, "%.1fmin", seconds / 60.0);
    else if (seconds < 172800.0) std::snprintf(buf, sizeof buf, "%.1fh", seconds / 3600.0);
    else if (seconds < 2.0 * year) std::snprintf(buf, sizeof buf, "%.1fd", seconds / 86400.0);
    else if (seconds < 1e6 * year) std::snprintf(buf, sizeof buf, "%.3gy", seconds / year);
    else std::snprintf(buf, sizeof buf, "inf");
    return buf;
}

// "progress: explored X% after Ys, counted N so far, est~E total, eta~T":
// the same extrapolation the enumerating mode prints (count over the
// explored fraction, time over the fraction), with the same caveat that
// the fraction moves in jumps.
void print_progress_line(double elapsed) {
    const double frac = g_explored_mass;
    std::printf("progress: explored %.8f%% after %.0fs, counted %s so far", frac * 100.0, elapsed, u128_str(g_counted_so_far).c_str());
    if (frac > 0.0 && frac < 1.0) {
        std::printf(", est~%.3g total, eta~%s", static_cast<double>(g_counted_so_far) / frac, eta_str(elapsed / frac - elapsed).c_str());
    }
    std::printf("\n");
    std::fflush(stdout);
}

void maybe_print_progress() {
    const auto now = std::chrono::steady_clock::now();
    if (now < g_progress_next) return;
    print_progress_line(std::chrono::duration<double>(now - g_progress_start).count());
    g_progress_next = now + std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(g_progress_interval));
}
std::vector<double> g_half_pow;  // g_half_pow[d] = 2^-d, d up to the cell count + 1
// Leaf mass is scaled by 1/k inside a k-region split (each region search
// sums to the split node's whole share on its own), so the fraction reads
// right mid-search too.
double g_mass_scale = 1.0;

// Clues of the puzzle being solved (stats build: search-area report).
const std::vector<std::vector<int>>* g_row_clues = nullptr;
const std::vector<std::vector<int>>* g_col_clues = nullptr;

// Number of placements of `clue` on a line of n cells read at stride
// `step` from `cells`, consistent with its known cells. Plain DP over
// (position, next block); double because the count can exceed 2^63.
double count_line_completions(const std::int8_t* cells, std::size_t step, int n, const std::vector<int>& clue) {
    const int k = static_cast<int>(clue.size());
    auto at = [&](int i) { return cells[static_cast<std::size_t>(i) * step]; };
    // no_full[i] = no FULL cell in i..n-1
    std::vector<char> no_full(static_cast<std::size_t>(n) + 1, 1);
    for (int i = n - 1; i >= 0; --i) no_full[i] = no_full[i + 1] && at(i) != FULL;
    // f[i][j]: ways to fill cells i..n-1 with blocks j..k-1
    std::vector<double> f(static_cast<std::size_t>(n + 2) * (k + 1), 0.0);
    auto F = [&](int i, int j) -> double& { return f[static_cast<std::size_t>(i) * (k + 1) + j]; };
    for (int i = n; i >= 0; --i) {
        for (int j = k; j >= 0; --j) {
            if (j == k) { F(i, j) = no_full[i] ? 1.0 : 0.0; continue; }
            if (i >= n) { F(i, j) = 0.0; continue; }
            double v = 0.0;
            if (at(i) != FULL) v += F(i + 1, j);
            const int b = clue[j], end = i + b;
            if (end <= n) {
                bool ok = true;
                for (int t = i; t < end && ok; ++t) ok = at(t) != EMPTY;
                if (ok) {
                    if (end == n) v += (j + 1 == k) ? 1.0 : 0.0;
                    else if (at(end) != FULL) v += F(end + 1, j + 1);
                }
            }
            F(i, j) = v;
        }
    }
    return F(0, 0);
}
// Dead-subtree histogram for latched (no-probing) branch nodes: bucket i
// holds subtrees of 2^i .. 2^(i+1)-1 nodes that found no solution.
std::uint64_t g_stat_dead_hist[40] = {};
std::uint64_t g_stat_live_hist[40] = {};
std::uint64_t g_stat_all_nodes = 0;
std::uint64_t g_stat_impl_nodes = 0, g_stat_impl_nodes_forced = 0, g_stat_impl_forced = 0, g_stat_impl_dead = 0;
std::uint64_t g_stat_impl_edges = 0;
// Per-node probe record: for each successful probe, a header (packed cell,
// value) followed by its settled cells (packed cell, value), all as
// (trail_pack << 1) | (val == FULL).
std::vector<int> g_impl_rec;
std::vector<int> g_impl_off;

inline std::uint16_t load_u16(const unsigned char* p) { std::uint16_t v; std::memcpy(&v, p, 2); return v; }
inline std::uint32_t load_u32(const unsigned char* p) { std::uint32_t v; std::memcpy(&v, p, 4); return v; }
inline void store_u16(unsigned char* p, std::uint16_t v) { std::memcpy(p, &v, 2); }
inline void store_u32(unsigned char* p, std::uint32_t v) { std::memcpy(p, &v, 4); }

class FastLineCache {
public:
    void set_grow_ahead(bool e) { grow_ahead_ = e; }

    void init(int key_words, std::size_t budget_bytes) {
        kw_ = key_words;
        hdr_ = 8 * static_cast<std::size_t>(kw_);
        slot_shift_ = (kw_ <= 2) ? 5 : 6;
        inline_cap_ = (std::size_t{1} << slot_shift_) - hdr_ - 4;
        std::size_t want = std::max<std::size_t>(budget_bytes >> slot_shift_, kMinSlots);
        max_slots_ = kMinSlots;
        while (max_slots_ * 2 <= want) max_slots_ *= 2;
        alloc(kMinSlots);
        arena_.clear();
    }

    int key_words() const { return kw_; }
    std::size_t header_offset() const { return hdr_; }
    const std::uint8_t* arena() const { return arena_.data(); }

    // The complemented tag word a slot stores for (tag, hash h).
    static std::uint16_t full_tag(std::uint16_t tag, std::uint64_t h) {
        return static_cast<std::uint16_t>(~(tag | ((h >> 58) << 10)));
    }

    // On a miss, remembers the empty slot it stopped at so the insert that
    // follows need not hash and walk the probe sequence a second time.
    // KW is the key's word count when the caller knows it at compile time
    // (propagate() dispatches on it once per propagation), 0 for the runtime kw_:
    // with a runtime count the hash and the compare were loops, or a call
    // when moved out of line, on every lookup.
    template <int KW>
    unsigned char* find(const std::uint64_t* key, std::uint16_t tag) {
        return find<KW>(key, tag, hash<KW>(key, tag));
    }
    // With the key's hash already computed (solve_lines hashes the next
    // queued line ahead of time to prefetch its slot).
    void prefetch(std::uint64_t h) const { __builtin_prefetch(slot(h & mask_)); }
    template <int KW>
    unsigned char* find(const std::uint64_t* key, std::uint16_t tag, std::uint64_t h) {
        std::size_t idx = h & mask_;
        tag = full_tag(tag, h);
        for (;;) {
            unsigned char* s = slot(idx);
            const std::uint16_t t = load_u16(s + hdr_);
            if (t == kEmptyStored) { miss_slot_ = s; miss_tag_ = tag; return nullptr; }
            if (t == tag && keys_equal<KW>(s, key)) return s;
            idx = (idx + 1) & mask_;
        }
    }

    // Inserts a fresh entry (precondition: find() returned nullptr) and returns
    // its slot. May grow or generationally clear the table.
    unsigned char* insert(const std::uint64_t* key, std::uint16_t tag, const LineSolveResult& res) {
        unsigned char* s;
        std::uint16_t ftag;
        if ((count_ + 1) * 3 > nslots_) {  // load factor <= 1/3
            if (nslots_ >= max_slots_) {
                clear_slots();  // OOM-safe generational eviction (cold path)
            } else {
                grow();
            }
            const std::uint64_t h = hash(key, tag);
            s = free_slot(h);
            ftag = full_tag(tag, h);
        } else {
            s = miss_slot_;  // precondition: find(key, tag) just missed
            ftag = miss_tag_;
        }
        std::memcpy(s, key, hdr_);
        store_u16(s + hdr_, ftag);
        const std::size_t n = res.deductions.size();
        if (g_debug_stats) ++g_stat_ded_hist[n];
        s[hdr_ + 2] = static_cast<std::uint8_t>(n);
        std::uint8_t flags = res.total ? kFlagSat : 0;
        std::uint8_t* out;
        if (n <= inline_cap_) {
            out = s + hdr_ + 4;
        } else {
            flags |= kFlagSpilled;
            store_u32(s + hdr_ + 4, static_cast<std::uint32_t>(arena_.size()));
            arena_.resize(arena_.size() + n);
            out = arena_.data() + (arena_.size() - n);
        }
        s[hdr_ + 3] = flags;
        for (std::size_t i = 0; i < n; ++i) {
            const int enc = res.deductions[i];
            out[i] = static_cast<std::uint8_t>(deduce_pos(enc) | (deduce_val(enc) << 7));
        }
        ++count_;
        return s;
    }

    // KW = 0 reads the word count from kw_ (insert and grow, off the hot
    // path); a positive KW is a compile-time count and the same value.
    template <int KW = 0>
    std::uint64_t hash(const std::uint64_t* key, std::uint16_t tag) const {
        namespace wy = ankerl::unordered_dense::detail::wyhash;
        const int kw = KW > 0 ? KW : kw_;
        // One 64x64->128 multiply-fold covers the common <= 2-word key; the
        // tag is folded in by a cheap multiply so it perturbs every bit.
        const std::uint64_t t = static_cast<std::uint64_t>(tag) * 0x9E3779B97F4A7C15ULL;
        if (kw <= 2) {
            const std::uint64_t k1 = (kw == 2) ? key[1] : 0;
            return wy::mix(key[0] ^ t, k1 ^ 0xE7037ED1A0B428DBULL);
        }
        std::uint64_t h = wy::mix(t, 0xA0761D6478BD642FULL);
        for (int w = 0; w < kw; ++w) h = wy::mix(h ^ key[w], 0xE7037ED1A0B428DBULL);
        return h;
    }

    template <int KW = 0>
    bool keys_equal(const unsigned char* s, const std::uint64_t* key) const {
        const int kw = KW > 0 ? KW : kw_;
        std::uint64_t diff = 0;
        for (int w = 0; w < kw; ++w) {
            std::uint64_t v;
            std::memcpy(&v, s + 8 * static_cast<std::size_t>(w), 8);
            diff |= v ^ key[w];
        }
        return diff == 0;
    }

private:
    static constexpr std::size_t kMinSlots = 1u << 12;  // 128 KB at 32-byte slots

    unsigned char* slot(std::size_t idx) const { return buf_.p + (idx << slot_shift_); }

    unsigned char* free_slot(std::uint64_t h) {
        std::size_t idx = h & mask_;
        while (load_u16(slot(idx) + hdr_) != kEmptyStored) idx = (idx + 1) & mask_;
        return slot(idx);
    }

    void alloc(std::size_t n) {
        buf_.allocate(n << slot_shift_);  // zero-filled: every slot free
        nslots_ = n;
        mask_ = n - 1;
        count_ = 0;
    }

    void clear_slots() {
        std::memset(buf_.p, 0, buf_.bytes);
        count_ = 0;
        arena_.clear();
    }

    void grow() {
        SlotBuf old;
        old.swap(buf_);
        const std::size_t old_n = nslots_;
        const std::size_t slot_bytes = std::size_t{1} << slot_shift_;
        // Every doubling re-walks every entry (a DRAM miss each), so once a
        // workload has proven large -- and in anytime mode, where it will
        // keep growing -- jump 8x and skip two of the three intermediate
        // rehashes. Capped by the budget-derived max_slots_.
        std::size_t factor = (grow_ahead_ && old_n >= kGrowAheadFrom) ? 8 : 2;
        while (factor > 2 && old_n * factor > max_slots_) factor /= 2;
        alloc(old_n * factor);
        for (std::size_t i = 0; i < old_n; ++i) {
            const unsigned char* s = old.p + (i << slot_shift_);
            const std::uint16_t t = load_u16(s + hdr_);
            if (t == kEmptyStored) continue;
            std::uint64_t key[kMaxFastCells / 32];
            std::memcpy(key, s, hdr_);
            std::memcpy(free_slot(hash(key, static_cast<std::uint16_t>(~t & 0x3FF))), s, slot_bytes);
            ++count_;
        }
    }

    static constexpr std::size_t kGrowAheadFrom = 1u << 20;  // slots
    bool grow_ahead_ = false;
    unsigned char* miss_slot_ = nullptr;
    std::uint16_t miss_tag_ = 0;
    int kw_ = 1;
    std::size_t hdr_ = 8;
    unsigned slot_shift_ = 5;
    std::size_t inline_cap_ = 20;
    std::size_t nslots_ = 0;
    std::size_t mask_ = 0;
    std::size_t count_ = 0;
    std::size_t max_slots_ = kMinSlots;
    SlotBuf buf_;
    std::vector<std::uint8_t> arena_;
};

FastLineCache g_fast_cache;
bool g_fast_mode = false;
// Orientation bit folded into the tag. A line solve depends only on (cells,
// clue, length); the packed key cannot tell trailing EMPTY cells from cells
// beyond the line's end, so rows and columns must not share entries when
// their lengths differ. On square puzzles they can, so the bit is dropped
// there to keep the legacy cache's sharing.
std::uint16_t g_tag_col_bit = 1;
// High bit of every 2-bit cell in a packed key: set exactly for UNKNOWN cells.
constexpr std::uint64_t kUnknownBits = 0xAAAAAAAAAAAAAAAAULL;
// Tag per row / per column (spec id * 2 + orientation bit), precomputed in
// solve() so the hit path reads one uint16 instead of dereferencing the
// LineSpec for its id -- a dependent load ahead of the hash.
std::vector<std::uint16_t> g_row_tags;
std::vector<std::uint16_t> g_col_tags;

// LINE_CACHE_LEGACY=1 forces the string-keyed cache (the fallback for lines
// longer than kMaxFastCells) on every puzzle; used to differential-test the two.
void reset_line_cache(int height, int width, std::size_t budget_bytes, bool reserve_enabled) {
    const int max_line_len = std::max(height, width);
    g_fast_mode = max_line_len <= kMaxFastCells && std::getenv("LINE_CACHE_LEGACY") == nullptr;
    if (g_fast_mode) {
        g_fast_cache.init((max_line_len + 31) / 32, budget_bytes);
        g_fast_cache.set_grow_ahead(reserve_enabled);
        g_tag_col_bit = (height != width) ? 1 : 0;
    }
    auto& c = g_legacy_cache;
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
// Probes use their own Trail (a reused one per LEARN instantiation of
// probe_cell_t) because their mutations are reverted before returning to
// the caller (solve_backtrack), so they don't need to commingle with the
// outer trail.
// ---------------------------------------------------------------------------

// Trail entries carry (row, col) rather than a linear index so a revert can
// update the packed line keys without a division to recover the row.
inline int trail_pack(int row, int col) { return (row << 16) | col; }
inline int trail_row(int e) { return e >> 16; }
inline int trail_col(int e) { return e & 0xFFFF; }

// Why a trail entry was set, for conflict analysis (--learn): a decision
// (branch or probe pixel), a line's deduction (id = line index, is_col says
// which array), a learnt clause's unit (id = clause id), or a probe-forced
// commit (id = its clause, UINT32_MAX while it has none: no antecedents).
// level is the decision level it was set at; a cell a unit clause forces is
// true in every solution and is recorded at level 0 wherever it is forced.
enum ReasonKind : std::uint8_t { R_DECISION = 0, R_LINE = 1, R_CLAUSE = 2, R_PROBE = 3 };
struct Reason { std::uint8_t kind; std::uint8_t is_col; std::uint16_t level; std::uint32_t id; };

// Clause-learning mode, set once per solve(). Reasons, cell_pos and levels
// are recorded only when it is on: the propagation drain, probe_cell (with
// its guard) and revert_branch are instantiated on a compile-time LEARN
// chosen once per propagation / probe / revert, so the off path does none
// of that work (recording unconditionally cost hard/3867 +9.4%
// instructions for 17.5M pushes); the per-node sites use Trail::record.
bool g_learn_mode = false;

// The learnt clauses (--learn), shared by the solve trail and the probe
// trails: the watch lists are global, so a probe's clause pass finds the
// clauses its cells make unit like any other. Initialised by solve() only
// when learning is on.
ClauseStore g_clauses;
// The last contradiction a propagation reported (learning only): a clause
// id, or else the line (index, and whether it is a column) whose solve
// found no filling. Conflict analysis starts from it.
int g_conflict_clause = -1;
int g_conflict_line = -1;
bool g_conflict_line_is_col = false;
// Learning statistics (DEBUG_CACHE_STATS): contradictions analysed, clauses
// learnt and their total length, clauses deleted by reduce, cells forced by
// clauses and clause conflicts (solve and probe trails alike); clauses learnt
// from contradicting probes, and those of them whose UIP was not the probe
// pixel (no reason for the forced commit, see probe_cell_t).
std::uint64_t g_stat_conflicts = 0, g_stat_learnt = 0, g_stat_learnt_lits = 0, g_stat_learnt_deleted = 0;
std::uint64_t g_stat_clause_forced = 0, g_stat_clause_conflicts = 0;
std::uint64_t g_stat_probe_learnt = 0, g_stat_probe_uip_elsewhere = 0;
// LEARN_CHECK=1 (stats build, with learning on): every clause learnt is kept
// here, deleted ones too, and checked against every solution when solve()
// returns (see learn_check).
#ifdef NONOGRAM_STATS
const bool g_learn_check = std::getenv("LEARN_CHECK") != nullptr;
#else
constexpr bool g_learn_check = false;
#endif
std::vector<std::vector<int>> g_check_clauses;
bool g_learn_check_inner = false;  // the check's own enumeration runs solve() without learning

// No clause pass inside a region search (learning only). g_region_depth
// counts the region searches active on the solve trail (a split's
// per-region recursive call until it returns; nested splits stack), and
// propagate_t skips clause propagation while it is positive, on the probe
// trail too. A region's nodes cache their counts under keys built from the
// region's lines, and a key is each line's DFA residual, so neither a cell
// off the region nor a known cell on one of its lines is determined by the
// key. A learnt clause is implied, so a force or conflict of one inside the
// region is right for the split's product: it can happen only when the
// outside has no completion in this context. But the region's nodes would
// then cache counts pruned by that outside under keys a later context with
// a feasible outside can hit. With no clause pass, a region's count is a
// function of its key by construction. A line conflict inside a region
// still learns (the derivation is sound anywhere: the entries below the
// region root may carry R_CLAUSE reasons from the top level); the fresh and
// unit clauses wait until the region search returns, and the entries it
// added are reverted then, so clause_next (never advanced inside) resumes
// at the split node's fixpoint. Top-level nodes need no such rule: with no
// outside, every completion of the state is a puzzle solution, and an
// implied clause prunes none. Written only when learning.
int g_region_depth = 0;

// An invariant the stats build checks (assert is compiled out of both
// builds by -DNDEBUG); nothing in the release build.
#ifdef NONOGRAM_STATS
#define STATS_CHECK(cond) \
    do { \
        if (!(cond)) { \
            std::fprintf(stderr, "stats check failed: %s (%s:%d)\n", #cond, __FILE__, __LINE__); \
            std::abort(); \
        } \
    } while (0)
#else
#define STATS_CHECK(cond) do {} while (0)
#endif

struct Trail {
    std::vector<int> changed_cell_indices;  // packed (row, col); see trail_pack
    std::vector<Reason> reasons;            // parallel to changed_cell_indices
    // Per cell (row * W + col): its index in changed_cell_indices, or
    // UINT32_MAX when the cell is not on this trail.
    std::vector<std::uint32_t> cell_pos;
    int W = 0;  // for cell_pos indexing
    // Current decision level: on the solve() trail the branch depth
    // (region searches nest inside their split node's level); on the probe
    // trail the solve trail's level + 1.
    std::uint16_t level = 0;
    // Learning only: the first changed_cell_indices entry the clause pass
    // has not seen yet (see clause_pass). A revert lowers it to the mark.
    std::size_t clause_next = 0;
    void push(int packed, Reason r) {
        cell_pos[static_cast<std::size_t>(trail_row(packed)) * W + trail_col(packed)] = static_cast<std::uint32_t>(changed_cell_indices.size());
        changed_cell_indices.push_back(packed);
        reasons.push_back(r);
    }
    // The entry with its reason when learning, the bare entry otherwise
    // (the sites outside the per-cell loops; see g_learn_mode).
    void record(int packed, Reason r) {
        if (g_learn_mode) push(packed, r);
        else changed_cell_indices.push_back(packed);
    }
    // Per-row / per-col UNKNOWN counts at the last branch node, maintained
    // from this trail alone: the node entry subtracts every entry pushed
    // since `counted` (cells settled by branch pixels, commits and real
    // propagation), and revert_branch adds back the counted entries it pops.
    // Probes use their own Trail and revert every cell before returning,
    // so the probe hot path pays nothing (counting inside Picture's
    // set_known/unset cost pikachu --anytime 3.3% of instructions).
    // Only the solve() trail carries these; probe trails leave them empty.
    std::vector<int> row_unknown;
    std::vector<int> col_unknown;
    std::size_t counted = 0;
    // Rows bucketed by their unknown count as bitsets (bucket k = rows with
    // exactly k unknowns, row_words words each), and a histogram of column
    // counts, so the latched branch-cell scan visits only rows that can
    // hold the minimum key instead of testing every row.
    std::vector<std::uint64_t> row_bucket;
    std::vector<int> col_hist;
    int row_words = 0;
    // Lines whose packed key changed since the state cache last read their
    // hash (count mode only; see SolveState::state_cache). settle and
    // unsettle are exactly the net cell changes between two branch nodes,
    // so the flags are set there, not per cell change inside Picture:
    // probes change and revert thousands of cells per node.
    bool track_hash = false;
    bool stamps = false;  // the solve() trail: its cell changes stamp their lines (probe memo)
    std::vector<std::uint8_t> row_hash_dirty, col_hash_dirty;
    std::vector<int> dirty_rows, dirty_cols;
    void mark_hash_dirty(int r, int c) {
        if (!row_hash_dirty[static_cast<std::size_t>(r)]) { row_hash_dirty[static_cast<std::size_t>(r)] = 1; dirty_rows.push_back(r); }
        if (!col_hash_dirty[static_cast<std::size_t>(c)]) { col_hash_dirty[static_cast<std::size_t>(c)] = 1; dirty_cols.push_back(c); }
    }

    void settle(int r, int c) {
        if (track_hash) mark_hash_dirty(r, c);
        const int old = row_unknown[r]--;
        std::uint64_t* b = row_bucket.data() + static_cast<std::size_t>(r >> 6);
        const std::uint64_t bit = 1ULL << (r & 63);
        b[static_cast<std::size_t>(old) * row_words] &= ~bit;
        b[static_cast<std::size_t>(old - 1) * row_words] |= bit;
        const int oc = col_unknown[c]--;
        --col_hist[oc];
        ++col_hist[oc - 1];
    }
    void unsettle(int r, int c) {
        if (track_hash) mark_hash_dirty(r, c);
        const int old = row_unknown[r]++;
        std::uint64_t* b = row_bucket.data() + static_cast<std::size_t>(r >> 6);
        const std::uint64_t bit = 1ULL << (r & 63);
        b[static_cast<std::size_t>(old) * row_words] &= ~bit;
        b[static_cast<std::size_t>(old + 1) * row_words] |= bit;
        const int oc = col_unknown[c]++;
        --col_hist[oc];
        ++col_hist[oc + 1];
    }
};

// Zero-filled array on 2 MB-aligned memory advised for transparent huge
// pages: the state table is accessed at random, so with 4 KB pages every
// lookup was also a TLB miss.
template <class T>
class HugeArray {
public:
    HugeArray() = default;
    ~HugeArray() { release(); }
    HugeArray(const HugeArray&) = delete;
    HugeArray& operator=(const HugeArray&) = delete;
    // Zero-filled by SlotBuf (fresh prefaulted huge pages, no memset pass).
    void assign(std::size_t n) {
        buf_.allocate(n * sizeof(T));
        data_ = reinterpret_cast<T*>(buf_.p);
        n_ = n;
    }
    void swap(HugeArray& o) noexcept { buf_.swap(o.buf_); std::swap(data_, o.data_); std::swap(n_, o.n_); }
    T& operator[](std::size_t i) { return data_[i]; }
    const T& operator[](std::size_t i) const { return data_[i]; }
    const T* begin() const { return data_; }
    const T* end() const { return data_ + n_; }

private:
    void release() { buf_.release(); data_ = nullptr; n_ = 0; }
    SlotBuf buf_;
    T* data_ = nullptr;
    std::size_t n_ = 0;
};

// Fixed-budget table of region counts by state (see SolveState::state_cache).
// Open addressing over 32-byte slots: a 120-bit key (two words of a wyhash
// chain over the region's line keys; the top byte of the second word holds
// the entry's work, log2 of the branch nodes its subtree cost), then the
// count. The key is a hash, not the state: with n entries the chance of a
// false hit is about n^2 / 2^121, under 1e-20 at a billion entries. A
// lookup probes a window of kWindow slots; an insert into a full window
// evicts the entry with the least work, since an entry's value is the
// search it saves. The budget is STATE_CACHE_MB (default 512).
class StateTable {
public:
    struct Key { std::uint64_t a, b; };
    static constexpr int kWindow = 8;
    static constexpr std::uint64_t kWorkMask = 0xFFULL << 56;

    // Starts at kMinSlots and doubles at half load up to the budget, so a
    // puzzle with a few hundred nodes never touches (or zeroes) the budget.
    void init(std::size_t budget_bytes) {
        std::size_t want = std::max<std::size_t>(budget_bytes / sizeof(Slot), kMinSlots);
        max_slots_ = kMinSlots;
        while (max_slots_ * 2 <= want) max_slots_ *= 2;
        nslots_ = kMinSlots;
        slots_.assign(nslots_);
        mask_ = nslots_ - 1;
        count_ = 0;
    }
    std::size_t size() const { return count_; }
    std::size_t slots() const { return nslots_; }

    // Empty slots have a == 0 && b == 0; a real key with both words zero is
    // nudged so it can never read as empty.
    static Key normalize(std::uint64_t a, std::uint64_t b) {
        b &= ~kWorkMask;
        if (a == 0 && b == 0) b = 1;
        return Key{a, b};
    }

    // Two candidate windows, one per key word; an entry lives in whichever
    // was emptier when it was inserted. A window's slots are contiguous, so
    // a lookup touches at most two runs of four cache lines.
    // The occupied slots of a window are always a prefix of it (an insert
    // takes the first free slot; an eviction happens only in a full
    // window), so a scan stops at the first empty slot.
    // work_lg, when given, receives the hit entry's work byte (log2 of the
    // branch nodes its subtree cost when it was stored).
    const u128* find(Key k, int* work_lg = nullptr) const {
        const std::size_t ia = bucket(k.a);
        const std::size_t ib = bucket(k.b);
        for (int p = 0; p < kWindow; ++p) {
            const Slot& s = slots_[ia + static_cast<std::size_t>(p)];
            if (s.a == k.a && (s.b & ~kWorkMask) == k.b) { if (work_lg) *work_lg = static_cast<int>(s.b >> 56); return &s.count; }
            if (s.a == 0 && s.b == 0) break;
        }
        for (int p = 0; p < kWindow; ++p) {
            const Slot& s = slots_[ib + static_cast<std::size_t>(p)];
            if (s.a == k.a && (s.b & ~kWorkMask) == k.b) { if (work_lg) *work_lg = static_cast<int>(s.b >> 56); return &s.count; }
            if (s.a == 0 && s.b == 0) break;
        }
        return nullptr;
    }
    void prefetch(Key k) const {
        __builtin_prefetch(&slots_[bucket(k.a)]);
        __builtin_prefetch(&slots_[bucket(k.b)]);
    }
    // Windows are aligned, non-overlapping buckets of kWindow slots, so the
    // occupied-prefix property holds for each.
    std::size_t bucket(std::uint64_t h) const {
        return static_cast<std::size_t>(h) & mask_ & ~static_cast<std::size_t>(kWindow - 1);
    }

    // Returns true when an occupied slot was overwritten.
    bool insert(Key k, u128 count, std::uint64_t work_nodes) {
        if ((count_ + 1) * 2 > nslots_ && nslots_ < max_slots_) grow();
        return place(k.a, k.b | (static_cast<std::uint64_t>(encode_work(work_nodes)) << 56), count);
    }
    // The work byte is log2 of the node count with two fraction bits
    // (byte = 4*lg + next two mantissa bits), so decode_work is within 20%.
    static int encode_work(std::uint64_t n) {
        if (n < 4) return static_cast<int>(n);  // 0..3 exact: byte/4 == 0 plus the fraction
        const int lg = 63 - __builtin_clzll(n);
        const int frac = static_cast<int>((n >> (lg - 2)) & 3);
        return std::min(255, lg * 4 + frac);
    }
    static std::uint64_t decode_work(int byte) {
        const int lg = byte >> 2, frac = byte & 3;
        if (lg == 0) return static_cast<std::uint64_t>(frac);
        return (4ULL + static_cast<std::uint64_t>(frac)) << (lg - 2);
    }

private:
    struct Slot { std::uint64_t a, b; u128 count; };
    static constexpr std::size_t kMinSlots = 4096;

    // Doubles while small; from 64k slots (2 MB, one huge page) grows 4x,
    // since every growth re-places every entry (two window scans each) and
    // the re-placing was 2/3 of StateTable::insert on easy_medium/108.
    // Capped by the budget-derived max_slots_, which is a power of two.
    void grow() {
        HugeArray<Slot> old;
        old.swap(slots_);
        std::size_t factor = (nslots_ >= (std::size_t{1} << 16)) ? 4 : 2;
        while (factor > 2 && nslots_ * factor > max_slots_) factor /= 2;
        nslots_ *= factor;
        mask_ = nslots_ - 1;
        slots_.assign(nslots_);
        count_ = 0;
        for (const Slot& s : old) {
            if (s.a == 0 && s.b == 0) continue;
            place(s.a, s.b, s.count);
        }
    }

    // b carries the work byte. Returns true when an occupied slot was overwritten.
    bool place(std::uint64_t a, std::uint64_t b, u128 count) {
        const Key k{a, b & ~kWorkMask};
        const std::size_t base[2] = {bucket(k.a), bucket(k.b)};
        // The emptier window takes the entry; on a tie the first. The
        // occupied slots are a prefix, so the first empty one ends the scan.
        std::size_t free_at[2];
        int free_n[2] = {0, 0};
        for (int w = 0; w < 2; ++w) {
            for (int p = 0; p < kWindow; ++p) {
                const std::size_t j = base[w] + static_cast<std::size_t>(p);
                const Slot& s = slots_[j];
                if (s.a == 0 && s.b == 0) {
                    free_at[w] = j;
                    free_n[w] = kWindow - p;
                    break;
                }
            }
        }
        if (free_n[0] + free_n[1] > 0) {
            const int w = (free_n[1] > free_n[0]) ? 1 : 0;
            Slot& s = slots_[free_at[w]];
            s.a = k.a; s.b = b; s.count = count;
            ++count_;
            return false;
        }
        // Both windows full: evict the entry with the least work. Ranked
        // only here, not on every slot of every insert.
        std::size_t victim = base[0];
        std::uint64_t victim_work = ~0ULL;
        for (int w = 0; w < 2; ++w) {
            for (int p = 0; p < kWindow; ++p) {
                const std::size_t j = base[w] + static_cast<std::size_t>(p);
                const std::uint64_t wk = slots_[j].b >> 56;
                if (wk < victim_work) { victim_work = wk; victim = j; }
            }
        }
        Slot& s = slots_[victim];
        s.a = k.a; s.b = b; s.count = count;
        return true;
    }

    HugeArray<Slot> slots_;
    std::size_t nslots_ = 0, mask_ = 0, count_ = 0, max_slots_ = 0;
};

// Start of the two wyhash chains of one line's 128-bit hash for the state
// cache: depends on the line's tag (index plus a row/column bit) alone, so
// it is computed once per line (kept in SolveState); residual_key below
// folds the line's residual constraint onto it.
inline StateTable::Key line_seed(std::uint64_t tag) {
    namespace wy = ankerl::unordered_dense::detail::wyhash;
    return StateTable::Key{wy::mix(0x243F6A8885A308D3ULL ^ tag, 0xE7037ED1A0B428DBULL),
                           wy::mix(0x13198A2E03707344ULL ^ tag, 0x8EBC6AF09C88C6E3ULL)};
}

// What residual_key last computed for a line, so the next computation
// resumes the sweeps instead of restarting them: descending the tree,
// the known prefix and suffix only grow, so the previous state sets are
// the prefixes of the new sweeps whenever the cells they covered are
// unchanged (checked on the packed key words). A revert shrinks them and
// the sweeps restart from the ends.
struct ResidualMemo {
    std::uint32_t fu = 0, lu = 0;  // prefix length; one past the last unknown
    std::uint64_t key[kMaxFastCells / 32] = {};
    std::uint64_t fwd[4] = {}, bwd[4] = {};
    bool valid = false;
};

// ---------------------------------------------------------------------------
// SolveState — mirrors picture.py SolveState (without print_state).
// ---------------------------------------------------------------------------

struct SolveState {

    int probing_min_solutions;

    int solutions_found = 0;
    bool used_contradiction = false;
    bool used_backtrack = false;
    // Progress: the search space is split in half at every branch node, so
    // a subtree at branch depth d holds 2^-d of it; explored_mass sums the
    // completed subtrees. solutions / explored_mass is a running estimate
    // of the total count (exact fraction, uniform-density extrapolation).
    int branch_depth = 0;
    // The path from the root to the node being searched: '0' / '1' for the
    // first / second branch value, 'a' + i for the i-th region of a split.
    // Two runs of the same tree (same policy) stopped at different times
    // compare by this string lexicographically, a prefix ranking below its
    // extensions; the explored fraction cannot separate them once both sit
    // inside one big subtree. Kept as a preallocated buffer plus length
    // (solve() sizes it for the deepest possible path) and read into a
    // string only at a stop: a std::string push/pop per node was 3.5% of
    // the branch node on easy_medium/108.
    std::vector<char> path_buf;
    std::size_t path_len = 0;
    void path_push(char c) { path_buf[path_len++] = c; }
    void path_pop() { --path_len; }
    std::string path_string() const { return std::string(path_buf.data(), path_len); }
    // Count mode (solve(..., count_mode=true)): no solution callbacks; every
    // solve_real / solve_backtrack call leaves the number of solutions of
    // the subtree it just searched in `result`. After propagation the
    // unknown cells are split into independent regions (rows and columns
    // joined by an unknown cell), each region is counted with the search
    // restricted to its rows via `region_row`, and the counts multiply:
    // every full solution is one choice per region, so this is exact.
    bool count_mode = false;
    u128 result = 0;
    std::vector<char> region_row;   // empty = whole grid
    std::vector<std::uint64_t> region_bits;  // the same rows as a bitset (trail.row_words words)
    std::uint64_t regions_split = 0, region_calls = 0;
    // Count of the whole current region by state, consulted at every
    // count-mode branch node: the state is the region's rows with unknowns
    // (their packed keys) and the columns those rows have unknowns in, so
    // two branch orders that settle the same cells reach the same key and
    // the second is not searched. The split above only separates regions
    // that share no line; this catches the recurrence inside one region.
    // Measured before it shipped (stats build, count mode, bench/nodes.py,
    // 0 mismatches): corpus branch nodes 5,121,344 -> 1,571,730 (-69%),
    // probes -30%; hard/7382 2,800,356 -> 532,363 nodes, medium/12130
    // 1,235,610 -> 598,608, easy_medium/108 587,907 -> 133,835,
    // easy_large/6689 36,944 -> 5,187; the one puzzle with more nodes is
    // easy_large/3929 (+5.9k of 15k). PGO wall: 7382 22.2 -> 16.3 s,
    // 12130 6.8 -> 5.7 s, 108 0.40 -> 0.23 s, 6689 0.12 -> 0.04 s; 4774
    // 0.30 -> 0.36 s and 3929 0.10 -> 0.15 s (cheap nodes, few lookups
    // per gate window). On the partially solved class, explored fraction
    // after 300 s, cache on / off: 10810 32x, 7290 25x, 10088 5.8x, 9798
    // 4.8x, 5903 2x, 7785 1.6x, 5485 1.07x, 2712 / 2647 / 3867 / pikachu
    // slightly ahead, 12548 / 13480 / 9892 identical (both inside one
    // subtree the fraction cannot resolve); none behind.
    StateTable state_cache;
    // Per-line contributions to the state key (the line's hash while it has
    // unknowns, zero otherwise) and their running combination over the
    // whole grid, both brought up to date from the trail's dirty lists at
    // every branch node.
    std::vector<StateTable::Key> row_hash, col_hash;
    std::vector<StateTable::Key> row_seed, col_seed;  // line_seed per row / column
    std::vector<ResidualMemo> row_memo, col_memo;     // residual_key's resume state per line
    StateTable::Key grid_key{0, 0};
    std::uint64_t state_lookups = 0, state_hits = 0, state_evictions = 0;
    // Adaptive gate by node size (bucket = log2 of the unknown cells):
    // hits come almost only from small nodes, and which sizes pay differs
    // by puzzle (10810: the 16-63-cell buckets took 2.2M lookups for 363
    // hits while the 4-15-cell ones hit 30%; on 3867 every bucket up to 31
    // cells pays). Every kScWindow lookups in a bucket, the cycles its hits
    // saved (the entries' work bytes) are compared with the cycles the
    // lookups cost; below sc_yield times that the bucket stops looking up
    // and storing for kScResample nodes, then samples again.
    // STATE_CACHE_YIELD overrides the factor (default 1: break-even).
    static constexpr int kScBuckets = 24;
    static constexpr std::uint32_t kScWindow = 16384;
    static constexpr std::uint32_t kScResample = 16 * kScWindow;
    std::uint32_t sc_lookups[kScBuckets] = {}, sc_skipped[kScBuckets] = {};
    std::uint64_t sc_saved[kScBuckets] = {}, sc_lookup_cycles[kScBuckets] = {};
    bool sc_off[kScBuckets] = {};
    double sc_yield = 1.0;
    bool skip_probing = false;
    // Dead-work watchdog for the latched (no-probing) mode. The yield window
    // shuts probing off when few probes find contradictions, but that is
    // the wrong signal for a puzzle whose tree is mostly dead subtrees that
    // line solving alone discovers late: medium/3929 latches after a dry
    // stretch, then spends 89% of its 556k latched nodes in dead subtrees,
    // while probing throughout gives a 36k-node tree. Solution-dense trees
    // (hard/6689 2.2M solutions, medium/108 564k) are the ones where the
    // shut-off pays, and there almost every subtree is live. So while
    // latched, count completed branch subtrees as dead or live, and when a
    // window of 4096 is over 90% dead, turn probing back on for good.
    // The window and fraction are conservative on purpose: easy_small/4774
    // (195k solutions, 11% dead subtrees overall) un-latches at 50%/1024 or
    // 75%/1024 and loses 4x; at 90%/4096 it stays latched while 3929 still
    // goes 1.63s -> 0.27s (556k -> 116k nodes).
    std::uint64_t latched_dead = 0, latched_live = 0;
    bool probing_pinned = false;  // watchdog fired: never latch again

    void note_latched_subtree(bool dead) {
        if (dead) ++latched_dead; else ++latched_live;
        if (latched_dead + latched_live < g_dead_window) return;
        if (static_cast<double>(latched_dead) > g_dead_frac * static_cast<double>(latched_dead + latched_live)) {
            skip_probing = false;
            probing_pinned = true;
            probe_outcomes.clear();
            probe_outcome_sum = 0;
        }
        latched_dead = latched_live = 0;
    }
    bool keep_probing = false;  // anytime mode: never latch skip_probing
    double balance_k = 0.0;     // see solve(): K*min - max branch score when > 0
    std::deque<int> probe_outcomes;
    int probe_outcome_sum = 0;  // sum of probe_outcomes

    // Benchmark hook: if MAX_NODES is set (>0), abort the search after that many
    // backtrack nodes. Lets pikachu-class never-terminating puzzles be timed
    // over a deterministic, fixed amount of work. 0 = unlimited (normal).
    std::uint64_t node_limit = 0;
    std::uint64_t nodes = 0;
    std::uint64_t branch_nodes = 0;  // every branch node; the state cache's work measure

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
        if (g_debug_stats) { ++g_stat_probe_pairs; g_stat_probe_hits += found_contradiction; }
        // In anytime mode skip_probing can never latch, so the whole yield
        // window is dead work on every probe -- skip it entirely.
        if (keep_probing || skip_probing || probing_pinned || g_no_skip) return;
        // Mirror picture.py: don't let yield-window disable probing until
        // we've already found multiple solutions.
        if (solutions_found < probing_min_solutions) return;
        // probe_outcome_sum is the window's running sum (re-summing the
        // window on every probe pair was 100 iterations per pair).
        probe_outcomes.push_back(found_contradiction ? 1 : 0);
        probe_outcome_sum += found_contradiction ? 1 : 0;
        if (probe_outcomes.size() > g_probe_window) {
            probe_outcome_sum -= probe_outcomes.front();
            probe_outcomes.pop_front();
        }
        if (probe_outcomes.size() == g_probe_window) {
            double yield_rate = static_cast<double>(probe_outcome_sum) / static_cast<double>(g_probe_window);
            if (yield_rate < g_probe_thresh && !keep_probing) {
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

// 16 bytes so it comes back in two registers instead of through memory.
// n8 >= 0: fast path, `ded` points at n8 bytes of pos | val << 7 (n8 may be
// 0 with ded null). n8 == kLegacy: `ded` is a const std::vector<int>* of
// packed (pos,val) ints (null when empty).
struct BatchResult {
    static constexpr int kLegacy = -1;
    const void* ded;
    int n8;
    bool success;
};

// Materialize the line's cells: rows are contiguous in pic.pixels (no copy);
// columns are strided, so gather into a reusable thread_local buffer.
inline const std::int8_t* line_cells(int index, bool is_col, const Picture& pic, std::size_t& n) {
    const int W = pic.width();
    const std::int8_t* px = pic.pixels.data();
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
        n = static_cast<std::size_t>(H);
        return cb;
    }
    n = static_cast<std::size_t>(W);
    return px + static_cast<std::size_t>(index) * W;
}

// The state cache's hash of one line (see SolveState::state_cache): the
// line's residual constraint on its unknown cells, not its content. A
// region's count depends on a line only through which fillings of its
// unknown cells the clue admits, and that is fixed by the clue DFA's
// state set after the known prefix (the cells before the first unknown),
// the cells from the first to the last unknown, and the state set that
// accepts the known suffix. Two contents that differ only in how the
// prefix or suffix was filled -- the same rows completed in a different
// order, or a different way -- hash the same, so the second is a cache
// hit; keyed on the content they were distinct states. Measured on the
// content key's misses (stats build): 3867 at 150k nodes 52% would have
// hit, 23210 62%, medium/7382 85%.
// words: the line's packed key (2 bits per cell, kUnknownBits marks the
// unknown cells); cells: its bytes. The middle is hashed from the key
// words masked to [first unknown, last unknown]; the two sweeps run on
// the cells, one word of DFA states when the clue fits (2 words on a
// carry pair, more words in a loop).
// Mask of the digits of cells < p (prefix) / >= p (suffix) in key word w.
inline std::uint64_t prefix_digits(int w, std::size_t p) {
    const std::size_t lo = 32 * static_cast<std::size_t>(w);
    if (p <= lo) return 0;
    if (p >= lo + 32) return ~0ULL;
    return (1ULL << (2 * (p - lo))) - 1;
}
inline std::uint64_t suffix_digits(int w, std::size_t p) {
    const std::size_t lo = 32 * static_cast<std::size_t>(w);
    if (p <= lo) return ~0ULL;
    if (p >= lo + 32) return 0;
    return ~0ULL << (2 * (p - lo));
}

// fwd holds the forward state set after cells [0, fwd_from) on entry and
// after [0, fu) on exit; bwd the backward set before cells [bwd_from, n)
// on entry and before [lu, n) on exit.
template <int NW>
inline void residual_sweeps(const LineSpec& spec, const std::int8_t* cells, std::size_t n, std::size_t fu, std::size_t lu,
                            std::uint64_t* fwd, std::uint64_t* bwd, std::size_t fwd_from, std::size_t bwd_from) {
    // Per-cell-value mask tables, as in the line solver: the sweeps are
    // then branch-free on the (data-dependent) cell values.
    //   forward:  next = (cur & stay[v]) | ((cur << 1) & step[v])
    //   backward: prev = (cur & stay[v]) | ((cur & step[v]) >> 1)
    std::uint64_t stay[3][NW], step[3][NW];
    for (int w = 0; w < NW; ++w) {
        const std::uint64_t em = spec.empty_mask[static_cast<std::size_t>(w)];
        stay[EMPTY][w] = em; stay[FULL][w] = 0; stay[UNKNOWN][w] = em;
        step[EMPTY][w] = em; step[FULL][w] = spec.full_mask[static_cast<std::size_t>(w)]; step[UNKNOWN][w] = spec.state_valid[static_cast<std::size_t>(w)];
    }
    if (fwd_from == 0) {
        for (int w = 0; w < NW; ++w) fwd[w] = 0;
        fwd[0] = 1;
    }
    for (std::size_t i = fwd_from; i < fu; ++i) {
        const int v = cells[i];
        std::uint64_t carry = 0;
        for (int w = 0; w < NW; ++w) {
            const std::uint64_t cur = fwd[w];
            const std::uint64_t sh = (cur << 1) | carry;
            carry = cur >> 63;
            fwd[w] = (cur & stay[v][w]) | (sh & step[v][w]);
        }
    }
    if (bwd_from == n) {
        for (int w = 0; w < NW; ++w) bwd[w] = 0;
        bwd[(spec.len_states - 1) / 64] |= 1ULL << ((spec.len_states - 1) % 64);
        if (spec.len_states >= 2) bwd[(spec.len_states - 2) / 64] |= 1ULL << ((spec.len_states - 2) % 64);
    }
    for (std::size_t i = bwd_from; i > lu; ) {
        const int v = cells[--i];
        std::uint64_t carry = 0;
        for (int w = NW - 1; w >= 0; --w) {
            const std::uint64_t cur = bwd[w];
            const std::uint64_t masked = cur & step[v][w];
            const std::uint64_t sh = (masked >> 1) | carry;
            carry = masked << 63;
            bwd[w] = (cur & stay[v][w]) | sh;
        }
    }
}

inline StateTable::Key residual_key(StateTable::Key seed, const LineSpec& spec, const std::int8_t* cells, std::size_t n,
                                    const std::uint64_t* words, int kw, ResidualMemo& memo) {
    namespace wy = ankerl::unordered_dense::detail::wyhash;
    // First and last unknown cell from the packed key.
    std::size_t fu = 0, lu = 0;
    for (int w = 0; w < kw; ++w) {
        const std::uint64_t m = words[w] & kUnknownBits;
        if (m) { fu = 32 * static_cast<std::size_t>(w) + (static_cast<std::size_t>(__builtin_ctzll(m)) >> 1); break; }
    }
    for (int w = kw - 1; w >= 0; --w) {
        const std::uint64_t m = words[w] & kUnknownBits;
        if (m) { lu = 32 * static_cast<std::size_t>(w) + (63 - static_cast<std::size_t>(__builtin_clzll(m))) / 2; break; }
    }
    const std::size_t lu1 = lu + 1;
    // Resume the sweeps from the memo where the cells it covered are unchanged.
    std::size_t fwd_from = 0, bwd_from = n;
    std::uint64_t* fwd = memo.fwd;
    std::uint64_t* bwd = memo.bwd;
    if (memo.valid) {
        if (memo.fu <= fu) {
            std::uint64_t diff = 0;
            for (int w = 0; w < kw; ++w) diff |= (words[w] ^ memo.key[w]) & prefix_digits(w, memo.fu);
            if (diff == 0) fwd_from = memo.fu;
        }
        if (memo.lu >= lu1) {
            std::uint64_t diff = 0;
            for (int w = 0; w < kw; ++w) diff |= (words[w] ^ memo.key[w]) & suffix_digits(w, memo.lu);
            if (diff == 0) bwd_from = memo.lu;
        }
    }
    const int nw = static_cast<int>(spec.n_words);
    if (nw == 1) residual_sweeps<1>(spec, cells, n, fu, lu1, fwd, bwd, fwd_from, bwd_from);
    else if (nw == 2) residual_sweeps<2>(spec, cells, n, fu, lu1, fwd, bwd, fwd_from, bwd_from);
    else if (nw == 3) residual_sweeps<3>(spec, cells, n, fu, lu1, fwd, bwd, fwd_from, bwd_from);
    else residual_sweeps<4>(spec, cells, n, fu, lu1, fwd, bwd, fwd_from, bwd_from);
    memo.fu = static_cast<std::uint32_t>(fu);
    memo.lu = static_cast<std::uint32_t>(lu1);
    for (int w = 0; w < kw; ++w) memo.key[w] = words[w];
    memo.valid = true;
    std::uint64_t ha = seed.a ^ (fu | (lu << 8)), hb = seed.b + (fu | (lu << 8));
    for (int w = 0; w < nw; ++w) {
        ha = wy::mix(ha ^ fwd[w], 0xE7037ED1A0B428DBULL); hb = wy::mix(hb ^ fwd[w], 0x8EBC6AF09C88C6E3ULL);
        ha = wy::mix(ha ^ bwd[w], 0xE7037ED1A0B428DBULL); hb = wy::mix(hb ^ bwd[w], 0x8EBC6AF09C88C6E3ULL);
    }
    // The cells from fu to lu: the key words with the digits outside zeroed.
    for (int w = 0; w < kw; ++w) {
        const std::size_t lo = 32 * static_cast<std::size_t>(w), hi = lo + 31;
        std::uint64_t mask = ~0ULL;
        if (fu > lo) mask &= (fu > hi) ? 0 : (~0ULL << (2 * (fu - lo)));
        if (lu < hi) mask &= (lu < lo) ? 0 : (~0ULL >> (2 * (hi - lu)));
        const std::uint64_t v = words[w] & mask;
        ha = wy::mix(ha ^ v, 0xE7037ED1A0B428DBULL);
        hb = wy::mix(hb ^ v, 0x8EBC6AF09C88C6E3ULL);
    }
    return StateTable::Key{ha, hb};
}

BatchResult solve_one_batch_legacy(const LineSpec& spec,
                                   int index,
                                   bool is_col,
                                   Picture& pic) {
    std::size_t line_n;
    const std::int8_t* line = line_cells(index, is_col, pic, line_n);

    std::string_view view_bytes(reinterpret_cast<const char*>(line), line_n);
    LineKeyView vkey{view_bytes, &spec};
    auto& cache = g_legacy_cache;
    const LineSolveResult* result_ptr = cache.find_and_promote(vkey);
    if (result_ptr == nullptr) {
        LineSolveResult res;
        solve_line_batch(line, line_n, spec, res, true);
        LineKey okey{std::string(view_bytes), &spec};
        result_ptr = cache.insert(std::move(okey), std::move(res));
    }

    if (result_ptr->total == 0) {
        return BatchResult{nullptr, BatchResult::kLegacy, false};
    }

    if (result_ptr->deductions.empty()) {
        return BatchResult{nullptr, BatchResult::kLegacy, true};
    }

    return BatchResult{&result_ptr->deductions, BatchResult::kLegacy, true};
}

// Miss path, kept out of line so the hit path below stays small enough to
// inline into solve_lines' drain loop.
[[gnu::noinline]] unsigned char* solve_one_batch_miss(const LineSpec& spec,
                                                      int index,
                                                      bool is_col,
                                                      Picture& pic,
                                                      const std::uint64_t* key,
                                                      std::uint16_t tag) {
    if (g_debug_stats) ++g_stat_misses;
    std::size_t line_n;
    const std::int8_t* line = line_cells(index, is_col, pic, line_n);
    static thread_local LineSolveResult res;  // capacity reused across misses
    // UNKNOWN is digit 2 (0b10) in the packed key, so the line has an
    // unknown cell iff some cell's high bit is set. A fully known line only
    // needs the forward validity check, not the backward sweep.
    std::uint64_t any = 0;
    for (int w = 0; w < pic.key_words; ++w) any |= key[w];
    const bool has_unknown = (any & kUnknownBits) != 0;
    solve_line_batch(line, line_n, spec, res, has_unknown);
    return g_fast_cache.insert(key, tag, res);
}

// KW: the puzzle's key word count when known at compile time (1..4), else 0.
// h: hash of the line's (key, tag), computed by the caller.
template <bool FAST, int KW>
inline BatchResult solve_one_batch(const std::vector<const LineSpec*>& mapped,
                                   int index,
                                   bool is_col,
                                   Picture& pic,
                                   std::uint64_t h) {
    if (!FAST) {
        return solve_one_batch_legacy(*mapped[index], index, is_col, pic);
    }

    const int kw = KW > 0 ? KW : pic.key_words;
    const std::uint64_t* key =
        (is_col ? pic.col_keys.data() : pic.row_keys.data()) + static_cast<std::size_t>(index) * kw;
    const std::uint16_t tag = (is_col ? g_col_tags : g_row_tags)[static_cast<std::size_t>(index)];

    if (g_debug_stats) ++g_stat_lookups;
    unsigned char* s = g_fast_cache.find<KW>(key, tag, h);
    if (s == nullptr) {
        s = solve_one_batch_miss(*mapped[index], index, is_col, pic, key, tag);
    }

    const std::size_t hdr = g_fast_cache.header_offset();
    const std::uint8_t flags = s[hdr + 3];
    if (g_debug_stats) ++g_stat_probe_cur;
    if (!(flags & kFlagSat)) {
        if (g_debug_stats) ++g_stat_unsat;
        return BatchResult{nullptr, 0, false};
    }
    const int n = s[hdr + 2];
    if (g_debug_stats) { if (n == 0) ++g_stat_noded; else ++g_stat_ded; }
    const std::uint8_t* ded = (flags & kFlagSpilled)
        ? g_fast_cache.arena() + load_u32(s + hdr + 4)
        : s + hdr + 4;
    return BatchResult{ded, n, true};
}

// ---------------------------------------------------------------------------
// write_intersection: apply the deduced positions/values to pic, marking
// the cross-direction dirty for any cells that go from UNKNOWN to a value.
//
// Every pixel transition UNKNOWN -> value is recorded into trail (the packed
// (row, col) entry, see trail_pack). On revert, the caller truncates the
// trail and walks each recorded entry back to UNKNOWN.
// ---------------------------------------------------------------------------

template <bool LEARN, typename Iter, typename Pos, typename Val>
inline void write_intersection_impl(Iter first, Iter last, Pos pos_of, Val val_of,
                                    int line_index, Picture& pic, bool is_row, Trail& trail) {
    // No UNKNOWN re-check: the cache entry was computed for exactly the
    // line's current content (that is its key), and the line DP only emits
    // deductions for cells that are UNKNOWN in that content.
    [[maybe_unused]] const Reason reason{R_LINE, static_cast<std::uint8_t>(is_row ? 0 : 1), trail.level, static_cast<std::uint32_t>(line_index)};
    if (is_row) {
        const int row = line_index;
        for (; first != last; ++first) {
            const int col = pos_of(*first);
            pic.set_known(row, col, val_of(*first));
            pic.mark_col_dirty(col);
            if constexpr (LEARN) trail.push(trail_pack(row, col), reason);
            else trail.changed_cell_indices.push_back(trail_pack(row, col));
            if (g_memo_enabled && trail.stamps) stamp_cell(row, col);
        }
    } else {
        const int col = line_index;
        for (; first != last; ++first) {
            const int row = pos_of(*first);
            pic.set_known(row, col, val_of(*first));
            pic.mark_row_dirty(row);
            if constexpr (LEARN) trail.push(trail_pack(row, col), reason);
            else trail.changed_cell_indices.push_back(trail_pack(row, col));
            if (g_memo_enabled && trail.stamps) stamp_cell(row, col);
        }
    }
}

template <bool LEARN>
__attribute__((flatten)) void write_intersection(const BatchResult& r, int line_index, Picture& pic, bool is_row, Trail& trail) {
    if (r.n8 != BatchResult::kLegacy) {
        const std::uint8_t* d = static_cast<const std::uint8_t*>(r.ded);
        write_intersection_impl<LEARN>(d, d + r.n8,
                                       [](std::uint8_t b) { return static_cast<int>(b & 0x7F); },
                                       [](std::uint8_t b) { return static_cast<std::int8_t>(b >> 7); },
                                       line_index, pic, is_row, trail);
    } else {
        const auto* v = static_cast<const std::vector<int>*>(r.ded);
        write_intersection_impl<LEARN>(v->begin(), v->end(),
                                       [](int enc) { return deduce_pos(enc); },
                                       [](int enc) { return deduce_val(enc); },
                                       line_index, pic, is_row, trail);
    }
}

// ---------------------------------------------------------------------------
// solve_lines: drain the appropriate dirty queue; on contradiction returns
// false, on success returns true.
// ---------------------------------------------------------------------------

// FAST: the fast line cache (vs the legacy string-keyed one); KW: the key
// word count when it is one of the fast cache's 1..4, else 0 (runtime).
template <bool FAST, int KW, bool LEARN>
inline bool solve_lines(const std::vector<const LineSpec*>& mapped,
                        Picture& pic,
                        bool is_row,
                        Trail& trail) {
    auto& queue = is_row ? pic.row_queue : pic.col_queue;
    auto& dirty = is_row ? pic.row_dirty : pic.col_dirty;
    // Draining rows changes only the current row's key and column keys
    // (and vice versa), so the next queued line's key is already what its
    // lookup will hash: it is hashed one line ahead, its slot prefetched
    // under this line's lookup and writes, and the hash handed on. On a
    // 90 x 69 count run the slot load was the top stall of the drain.
    const int kw = KW > 0 ? KW : pic.key_words;
    const std::uint64_t* keys = is_row ? pic.row_keys.data() : pic.col_keys.data();
    const std::uint16_t* tags = (is_row ? g_row_tags : g_col_tags).data();
    std::uint64_t h = 0;
    if (FAST && !queue.empty()) h = g_fast_cache.hash<KW>(keys + static_cast<std::size_t>(queue.front()) * kw, tags[queue.front()]);
    while (!queue.empty()) {
        const int index = queue.front();
        queue.pop_front();
        dirty[index] = 0;
        std::uint64_t next_h = 0;
        if (FAST && !queue.empty()) {
            const int nx = queue.front();
            next_h = g_fast_cache.hash<KW>(keys + static_cast<std::size_t>(nx) * kw, tags[nx]);
            g_fast_cache.prefetch(next_h);
        }
        BatchResult r = solve_one_batch<FAST, KW>(mapped, index, !is_row, pic, h);
        h = next_h;
        if (!r.success) {
            if constexpr (LEARN) {
                g_conflict_line = index;
                g_conflict_line_is_col = !is_row;
                g_conflict_clause = -1;
            }
            return false;
        }
        if (r.n8 > 0 || (r.n8 == BatchResult::kLegacy && r.ded != nullptr)) {
            write_intersection<LEARN>(r, index, pic, is_row, trail);
        }
    }
    queue.reset();
    return true;
}

// The literals of the trail entries clause_pass has not seen yet (reused;
// passes never nest: a probe's pass runs while no solve-trail pass does).
std::vector<int> g_clause_lits;

// Clause propagation (learning only) over the trail entries since the last
// pass: each becomes the literal it made true, and the store visits the
// clauses those make false, and whatever they force, to the clause fixpoint
// in one call. A forced cell is set, its row and column are marked dirty
// (the line drain picks them up) and it goes on the trail with an R_CLAUSE
// reason. On a probe trail this sees the probe's own cells only: the solve
// trail is at its clause fixpoint whenever a probe starts, and the watch
// lists are global, so the clauses the probe's cells make unit are found
// like any other. The one fresh clause a probe can meet is the one its
// cell's other probe just learnt, which the probe pixel satisfies (see
// probe_cell_t). Returns false on a conflict (g_conflict_clause set); the
// caller then reverts the round, as after a line contradiction. Never
// called inside a region search (see g_region_depth).
bool clause_pass(Picture& pic, Trail& trail) {
    if (g_clauses.size() == 0) {  // no units, no watches, nothing fresh
        trail.clause_next = trail.changed_cell_indices.size();
        return true;
    }
    const int W = pic.width();
    std::int8_t* px = pic.pixels.data();
    const std::size_t n = trail.changed_cell_indices.size();
    g_clause_lits.clear();
    for (std::size_t i = trail.clause_next; i < n; ++i) {
        const int e = trail.changed_cell_indices[i];
        const int cell = trail_row(e) * W + trail_col(e);
        g_clause_lits.push_back(cell * 2 + (px[cell] == FULL ? 1 : 0));
    }
    trail.clause_next = n;
    auto assigned = [px](int l) -> int {
        const std::int8_t v = px[l >> 1];
        if (v == UNKNOWN) return 0;
        return (v == FULL) == ((l & 1) != 0) ? 1 : -1;
    };
    auto force = [&pic, &trail, px, W](int l, int id) -> bool {
        const int cell = l >> 1;
        const std::int8_t v = (l & 1) ? FULL : EMPTY;
        if (px[cell] != UNKNOWN) return px[cell] == v;
        const int r = cell / W, c = cell % W;
        pic.set_known(r, c, v);
        pic.mark_row_dirty(r);
        pic.mark_col_dirty(c);
        // A unit clause is implied by the puzzle, so its literal is true at
        // level 0 wherever it is forced; conflict analysis then drops it.
        const std::uint16_t level = g_clauses.is_unit(id) ? 0 : trail.level;
        trail.push(trail_pack(r, c), Reason{R_CLAUSE, 0, level, static_cast<std::uint32_t>(id)});
        if (g_debug_stats) ++g_stat_clause_forced;
        if (g_memo_enabled && trail.stamps) stamp_cell(r, c);
        return true;
    };
    const int conflict = g_clauses.propagate(g_clause_lits.data(), static_cast<int>(g_clause_lits.size()), assigned, force);
    if (conflict >= 0) {
        g_conflict_clause = conflict;
        g_conflict_line = -1;
        if (g_debug_stats) ++g_stat_clause_conflicts;
        return false;
    }
    // The store visited every literal it forced within the call, so the
    // entries it pushed need no second visit.
    trail.clause_next = trail.changed_cell_indices.size();
    return true;
}

// Propagation to a fixpoint: drain rows and columns until neither queue
// holds a line. Returns false on a contradiction (a line with no valid
// filling), leaving the queues as they are for the caller's revert.
// Learning adds a clause pass whenever the drain left trail entries it has
// not seen, and drains again when the pass set cells, so the fixpoint is
// closed under both. The pass runs for new entries, and for clauses added
// since the store last examined its fresh list (a clause learnt at a
// contradiction takes effect at the next propagation). No pass runs inside
// a region search, on either trail (see g_region_depth): a clause learnt
// there waits for the search to return. So the propagation that first
// examines a clause is on the solve trail, or is a probe's that the clause
// cannot act in. A clause learnt at a committed contradiction: the node
// that learnt returns dead (a region loop breaks on it) and, once no region
// search is active, the next propagation is a parent's next branch value,
// before any node probes. A clause a probe learnt (probe_cell_t) asserts
// the probed cell's other value: the cell's other probe, if it runs, sets
// that value, and the forced commit's propagation holds it too; when both
// probes contradict, the node returns dead as above.
template <bool FAST, int KW, bool LEARN>
bool propagate_t(const std::vector<const LineSpec*>& mapped_rows,
                 const std::vector<const LineSpec*>& mapped_cols,
                 Picture& pic,
                 Trail& trail) {
    for (;;) {
        while (pic.has_dirty()) {
            if (!solve_lines<FAST, KW, LEARN>(mapped_rows, pic, true, trail)) return false;
            if (!solve_lines<FAST, KW, LEARN>(mapped_cols, pic, false, trail)) return false;
        }
        if constexpr (!LEARN) {
            return true;
        } else {
            if (g_region_depth > 0) return true;  // region searches run no clause pass
            if (trail.clause_next >= trail.changed_cell_indices.size() && !g_clauses.has_fresh()) return true;
            if (!clause_pass(pic, trail)) return false;  // sets g_conflict_clause
        }
    }
}

// The fast/legacy choice and the key word count are per puzzle; decided
// once per propagation, not per line, so the lookup's hash and key compare
// inside are straight-line code for 1 to 4 words (the fast cache's whole
// range) instead of loops with a runtime trip count (a 90 x 69 count run
// -10.7% instructions), and the drain loop itself stays inlined in the
// per-width instantiation instead of being called per queue.
// Learning mode (see g_learn_mode) is decided here too, once per
// propagation (probe_cell_t calls propagate_l directly), and runs the
// runtime-width drain only: instantiating every width on it as well left
// Picture::set_known out of line in the LTO build (+11.7% instructions on
// easy_large/7382 with learning off), and deciding it per deduced line cost
// +1.4%. write_intersection is flattened for the same reason: without its
// __attribute__((flatten)) (GCC/Clang only) LTO stops inlining set_known
// there too, about 11% more instructions on the off path.
template <bool LEARN>
inline bool propagate_l(const std::vector<const LineSpec*>& mapped_rows,
                        const std::vector<const LineSpec*>& mapped_cols,
                        Picture& pic,
                        Trail& trail) {
    if constexpr (LEARN) {
        return g_fast_mode ? propagate_t<true, 0, true>(mapped_rows, mapped_cols, pic, trail)
                           : propagate_t<false, 0, true>(mapped_rows, mapped_cols, pic, trail);
    }
    if (!g_fast_mode) return propagate_t<false, 0, false>(mapped_rows, mapped_cols, pic, trail);
    switch (pic.key_words) {
        case 1: return propagate_t<true, 1, false>(mapped_rows, mapped_cols, pic, trail);
        case 2: return propagate_t<true, 2, false>(mapped_rows, mapped_cols, pic, trail);
        case 3: return propagate_t<true, 3, false>(mapped_rows, mapped_cols, pic, trail);
        case 4: return propagate_t<true, 4, false>(mapped_rows, mapped_cols, pic, trail);
        default: return propagate_t<true, 0, false>(mapped_rows, mapped_cols, pic, trail);
    }
}

inline bool propagate(const std::vector<const LineSpec*>& mapped_rows,
                      const std::vector<const LineSpec*>& mapped_cols,
                      Picture& pic,
                      Trail& trail) {
    return g_learn_mode ? propagate_l<true>(mapped_rows, mapped_cols, pic, trail)
                        : propagate_l<false>(mapped_rows, mapped_cols, pic, trail);
}

// ---------------------------------------------------------------------------
// Conflict analysis (--learn)
//
// Every clause below is implied by the puzzle: a line explanation (the
// subset of a line's known cells that alone forces a cell or makes the line
// unsatisfiable) says "these cells as they are imply that", which every
// solution satisfies, and a learnt clause is a resolvent of such clauses and
// earlier learnt ones. Resolution over implied clauses yields implied
// clauses, so learning never removes a solution.
// ---------------------------------------------------------------------------

// The true literal of a known cell.
inline int cell_lit(const Picture& pic, int cell) { return cell * 2 + (pic.pixels[static_cast<std::size_t>(cell)] == FULL ? 1 : 0); }

// Scratch for explain_entry / analyze_conflict (never nested).
std::vector<std::int8_t> g_explain_line;
std::vector<int> g_explain_pos;

// The antecedents of trail entry `idx` of `trail`, as literals that are TRUE
// on the trail: with the entry's own literal L they form its reason clause
// L or not(a1) or ... For R_LINE, explain_deduction on the line as it was
// when the entry was set: a cell of the line counts as known iff it is on
// this trail below idx, or (when `trail` is a probe trail) on the solve
// trail at all. For R_CLAUSE, and R_PROBE with a clause id, the clause's
// other literals, negated (they are false on the trail). R_DECISION, and
// R_PROBE with id UINT32_MAX (a forced commit with no clause yet): none.
void explain_entry(const Trail& trail, const Trail* solve_trail, std::size_t idx, const Picture& pic,
                   const std::vector<const LineSpec*>& rows, const std::vector<const LineSpec*>& cols,
                   std::vector<int>& out_true_lits) {
    out_true_lits.clear();
    const int W = pic.width();
    const int e = trail.changed_cell_indices[idx];
    const int own = trail_row(e) * W + trail_col(e);
    const Reason r = trail.reasons[idx];
    if (r.kind == R_LINE) {
        const bool is_col = r.is_col != 0;
        const int line = static_cast<int>(r.id);
        const int n = is_col ? pic.height() : W;
        const int pos = is_col ? trail_row(e) : trail_col(e);
        g_explain_line.resize(static_cast<std::size_t>(n));
        g_explain_pos.resize(static_cast<std::size_t>(n));
        for (int p = 0; p < n; ++p) {
            const int cell = is_col ? p * W + line : line * W + p;
            const bool known = trail.cell_pos[static_cast<std::size_t>(cell)] < idx ||
                               (solve_trail != nullptr && solve_trail->cell_pos[static_cast<std::size_t>(cell)] != UINT32_MAX);
            g_explain_line[static_cast<std::size_t>(p)] = known ? pic.pixels[static_cast<std::size_t>(cell)] : UNKNOWN;
        }
        const LineSpec& spec = *(is_col ? cols : rows)[static_cast<std::size_t>(line)];
        const int k = explain_deduction(g_explain_line.data(), static_cast<std::size_t>(n), spec, pos,
                                        pic.pixels[static_cast<std::size_t>(own)], g_explain_pos.data());
        for (int i = 0; i < k; ++i) {
            const int p = g_explain_pos[static_cast<std::size_t>(i)];
            out_true_lits.push_back(cell_lit(pic, is_col ? p * W + line : line * W + p));
        }
    } else if (r.kind == R_CLAUSE || (r.kind == R_PROBE && r.id != UINT32_MAX)) {
        int n = 0;
        const int* l = g_clauses.lits(static_cast<int>(r.id), &n);
        [[maybe_unused]] bool has_own = false;
        for (int i = 0; i < n; ++i) {
            if ((l[i] >> 1) == own) {
                STATS_CHECK(l[i] == cell_lit(pic, own));  // the clause forced the value the cell has
                has_own = true;
                continue;
            }
            out_true_lits.push_back(l[i] ^ 1);
        }
        STATS_CHECK(has_own);
        g_clauses.bump(static_cast<int>(r.id));
    }
}

// Per-cell and per-level marks of one analysis, generation-stamped.
std::vector<std::uint32_t> g_seen, g_level_seen;
std::uint32_t g_seen_gen = 0;

// 1UIP analysis of the contradiction the last propagation on `trail`
// reported (g_conflict_clause, or else g_conflict_line), at trail.level.
// `solve_trail` is null when `trail` is the solve trail; for a probe trail
// it is the solve trail, and a cell not on the probe trail is looked up
// there. Adds the learnt clause to g_clauses and returns its id; -1 when
// there is nothing to learn (a contradiction at level 0, or one whose
// literals are all below the current level).
//
// The clause is built by resolution: start from the conflict's literals
// (all true), keep the negation of each one below the current level, and
// resolve away the current-level ones newest first, replacing each by its
// antecedents, until one current-level literal is left: the first unique
// implication point (UIP). A current-level entry with no antecedents (an
// R_PROBE commit with no clause) cannot be resolved away; its negation stays
// in the clause as a second current-level literal (still implied, not
// asserting). Level-0 literals are true in every solution and dropped.
int analyze_conflict(const Trail& trail, const Trail* solve_trail, const Picture& pic,
                     const std::vector<const LineSpec*>& rows, const std::vector<const LineSpec*>& cols) {
    if (g_debug_stats) ++g_stat_conflicts;
    const std::uint16_t cur = trail.level;
    if (cur == 0) return -1;
    const int W = pic.width(), H = pic.height();
    const std::size_t n_cells = static_cast<std::size_t>(H) * W;
    if (g_seen.size() != n_cells) g_seen.assign(n_cells, 0);
    if (g_level_seen.size() != 65536) g_level_seen.assign(65536, 0);
    if (++g_seen_gen == 0) {
        std::fill(g_seen.begin(), g_seen.end(), 0u);
        std::fill(g_level_seen.begin(), g_level_seen.end(), 0u);
        g_seen_gen = 1;
    }
    const std::uint32_t gen = g_seen_gen;
    static std::vector<int> conflict, antecedents, learnt;
    learnt.assign(1, -1);  // [0]: the UIP's negation, filled at the end
    int counter = 0;       // marked current-level entries not yet resolved
    // The reason of a known cell's entry: on `trail`, else on the solve trail.
    auto reason_of = [&](int cell) -> const Reason& {
        std::uint32_t p = trail.cell_pos[static_cast<std::size_t>(cell)];
        if (p != UINT32_MAX) return trail.reasons[p];
        STATS_CHECK(solve_trail != nullptr);
        p = solve_trail->cell_pos[static_cast<std::size_t>(cell)];
        STATS_CHECK(p != UINT32_MAX);
        return solve_trail->reasons[p];
    };
    auto mark = [&](int lit) {
        const int cell = lit >> 1;
        STATS_CHECK(lit == cell_lit(pic, cell) && pic.pixels[static_cast<std::size_t>(cell)] != UNKNOWN);
        if (g_seen[static_cast<std::size_t>(cell)] == gen) return;
        const std::uint16_t lv = reason_of(cell).level;
        STATS_CHECK(lv <= cur);
        if (lv == 0) return;
        g_seen[static_cast<std::size_t>(cell)] = gen;
        if (lv == cur) ++counter;
        else learnt.push_back(lit ^ 1);
    };

    conflict.clear();
    if (g_conflict_clause >= 0) {
        int n = 0;
        const int* l = g_clauses.lits(g_conflict_clause, &n);
        for (int i = 0; i < n; ++i) conflict.push_back(l[i] ^ 1);  // every literal is false
        g_clauses.bump(g_conflict_clause);
    } else {
        STATS_CHECK(g_conflict_line >= 0);
        const bool is_col = g_conflict_line_is_col;
        const int line = g_conflict_line;
        const int n = is_col ? H : W;
        g_explain_line.resize(static_cast<std::size_t>(n));
        g_explain_pos.resize(static_cast<std::size_t>(n));
        for (int p = 0; p < n; ++p) g_explain_line[static_cast<std::size_t>(p)] = pic.pixels[static_cast<std::size_t>(is_col ? p * W + line : line * W + p)];
        const LineSpec& spec = *(is_col ? cols : rows)[static_cast<std::size_t>(line)];
        const int k = explain_conflict(g_explain_line.data(), static_cast<std::size_t>(n), spec, g_explain_pos.data());
        for (int i = 0; i < k; ++i) {
            const int p = g_explain_pos[static_cast<std::size_t>(i)];
            conflict.push_back(cell_lit(pic, is_col ? p * W + line : line * W + p));
        }
    }
    for (const int l : conflict) mark(l);
    // No current-level literal: the contradiction already held below this
    // level (a unit clause forced here at level 0 completed it). Phase 1
    // learns nothing from it.
    if (counter == 0) return -1;

    int uip = -1;
    for (int t = 0; t < 2 && uip < 0; ++t) {
        const Trail* tr = t == 0 ? &trail : solve_trail;
        if (tr == nullptr) break;
        for (std::size_t i = tr->changed_cell_indices.size(); i-- > 0;) {
            const int e = tr->changed_cell_indices[i];
            const int cell = trail_row(e) * W + trail_col(e);
            if (g_seen[static_cast<std::size_t>(cell)] != gen || tr->reasons[i].level != cur) continue;
            const int lit = cell_lit(pic, cell);
            if (--counter == 0) {
                uip = lit;
                break;
            }
            const Reason r = tr->reasons[i];
            if (r.kind == R_DECISION || (r.kind == R_PROBE && r.id == UINT32_MAX)) {
                learnt.push_back(lit ^ 1);  // no antecedents to resolve with
                continue;
            }
            explain_entry(*tr, t == 0 ? solve_trail : nullptr, i, pic, rows, cols, antecedents);
            for (const int a : antecedents) mark(a);
        }
    }
    STATS_CHECK(uip >= 0);
    if (uip < 0) return -1;
    learnt[0] = uip ^ 1;
    // learnt[1]: the highest-level literal of the rest (the watch a
    // chronological revert unassigns first); lbd: the distinct levels among
    // learnt[1..], plus one.
    int lbd = 0;
    std::size_t best = 1;
    std::uint16_t best_level = 0;
    for (std::size_t i = 1; i < learnt.size(); ++i) {
        const std::uint16_t lv = reason_of(learnt[i] >> 1).level;
        if (lv > best_level) { best_level = lv; best = i; }
        if (g_level_seen[lv] != gen) { g_level_seen[lv] = gen; ++lbd; }
    }
    if (learnt.size() > 1) std::swap(learnt[1], learnt[best]);
    const int id = g_clauses.add(learnt.data(), static_cast<int>(learnt.size()), lbd + 1);
    if (g_debug_stats) { ++g_stat_learnt; g_stat_learnt_lits += learnt.size(); }
    if (g_learn_check) g_check_clauses.push_back(learnt);
    if (g_memo_enabled) memo_invalidate_all();
    return id;
}

// Between two subtrees of a branch node (learning only): when the store is
// over its limit, lock exactly the clauses that are reasons on the solve
// trail (the store is never told about reverts, and no probe trail is live
// here) and delete the worst of the rest.
void reduce_learnt(const Trail& trail) {
    if (static_cast<std::size_t>(g_clauses.size()) <= g_clauses.max_clauses()) return;
    g_clauses.unlock_all();
    for (const Reason& r : trail.reasons)
        if (r.kind == R_CLAUSE || (r.kind == R_PROBE && r.id != UINT32_MAX)) g_clauses.lock(static_cast<int>(r.id), true);
    const int before = g_clauses.size();
    g_clauses.reduce();
    if (g_debug_stats) g_stat_learnt_deleted += static_cast<std::uint64_t>(before - g_clauses.size());
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
    // Learning only: when the probe contradicted, the clause its conflict
    // analysis learnt, if that clause asserts the probed cell's other value
    // (the forced commit's reason, see the probing pass); -1 otherwise.
    int learnt_clause = -1;
};

// What the probes of one pass at one node imply about each other. When
// probing A=v settled B=w without a contradiction, the propagation from
// B=w alone reaches a subset of that fixpoint (line propagation is
// monotone and the fixpoint is closed under it), so B=w is consistent and
// settles at most as many cells: the fill of the A=v probe bounds the
// fill of the B=w probe from above. The pass records, per (cell, value),
// the smallest such bound seen; a later probe whose bound is already below
// the best branch score so far cannot become the branch cell (its own
// fill is at most the bound) and cannot be forced (it is consistent), so
// it is skipped and the bound stands in for its fill in the comparisons
// below. The branch cell and the forced cells of the pass are unchanged;
// the node's tree is bit-for-bit identical. Only recorded for probes whose
// fill is below the best score (note_below), since a higher bound could
// only matter once the best grows past it and those probes carry the long
// trails that make recording cost what the skips save. Bounds hold for one
// board: a forced commit advances the generation (invalidate()).
struct ProbeBounds {
    // One word per (cell, value): the generation's complement in the high
    // half, the fill in the low half, so within the current generation the
    // smallest fill is the minimum word and note() is a branch-free min. An
    // empty word (all ones) never matches the generation (never 0).
    std::vector<std::uint64_t> v;
    std::uint32_t cur = 0;
    int W = 0;
    int note_below = 0;
    void reset(int height, int width) {
        const std::size_t n = static_cast<std::size_t>(height) * static_cast<std::size_t>(width) * 2;
        if (v.size() != n || cur == 0xFFFFFFFFu) { v.assign(n, ~0ULL); cur = 0; }
        W = width;
        ++cur;
    }
    void invalidate() {
        if (cur == 0xFFFFFFFFu) { std::fill(v.begin(), v.end(), ~0ULL); cur = 0; }
        ++cur;
    }
    std::uint64_t stamp() const { return static_cast<std::uint64_t>(~cur) << 32; }
    std::size_t at(int row, int col, std::int8_t val) const {
        return (static_cast<std::size_t>(row) * static_cast<std::size_t>(W) + static_cast<std::size_t>(col)) * 2 + static_cast<std::size_t>(val);
    }
    void note(int row, int col, std::int8_t val, std::uint64_t stamped_fill) {
        std::uint64_t& w = v[at(row, col, val)];
        w = std::min(w, stamped_fill);
    }
    // The bound, or INT_MAX when no probe of this pass settled the cell to val.
    int bound(int row, int col, std::int8_t val) const {
        const std::uint64_t w = v[at(row, col, val)];
        return (w >> 32) == (stamp() >> 32) ? static_cast<int>(w & 0xFFFFFFFFu) : INT_MAX;
    }
};
std::uint64_t g_stat_probe_skips = 0;  // probes the bounds made unnecessary

// RAII guard that snapshots a small amount of Picture state at construction
// and restores pic to its entry state on destruction by walking the trail
// (and, when learning, clearing the cells' cell_pos).
template <bool LEARN>
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
        if constexpr (LEARN) {
            std::uint32_t* cell_pos = trail.cell_pos.data();
            const std::size_t tw = static_cast<std::size_t>(trail.W);
            for (auto it = trail.changed_cell_indices.rbegin();
                 it != trail.changed_cell_indices.rend(); ++it) {
                pic.unset(trail_row(*it), trail_col(*it));
                cell_pos[static_cast<std::size_t>(trail_row(*it)) * tw + trail_col(*it)] = UINT32_MAX;
            }
            trail.clause_next = 0;  // every entry is undone (the next probe clears the buffers)
        } else {
            for (auto it = trail.changed_cell_indices.rbegin();
                 it != trail.changed_cell_indices.rend(); ++it) {
                pic.unset(trail_row(*it), trail_col(*it));
            }
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
        pic.row_queue.reset();
        while (!pic.col_queue.empty()) {
            int j = pic.col_queue.front();
            pic.col_queue.pop_front();
            pic.col_dirty[j] = 0;
        }
        pic.col_queue.reset();
    }
};

template <bool LEARN>
ProbeResult probe_cell_t(int row,
                         int col,
                         std::int8_t val,
                         const std::vector<const LineSpec*>& mapped_rows,
                         const std::vector<const LineSpec*>& mapped_cols,
                         Picture& pic,
                         ProbeBounds* bounds,
                         const Trail* solve_trail) {
    // Reusable per-probe trail (one per LEARN instantiation). probe_cell
    // never nests (it calls only solve_lines, which never probes), and the
    // previous probe's ProbeGuard already reverted every cell it touched (and
    // cleared its cell_pos), so we just clear the index buffers and reuse
    // their capacity — no per-probe heap allocation.
    static thread_local Trail trail;
    trail.changed_cell_indices.clear();
    if constexpr (!LEARN) {
        (void)solve_trail;
    } else {
        trail.reasons.clear();
        STATS_CHECK(trail.clause_next == 0);  // the last probe's guard reset it
        if (trail.cell_pos.size() != static_cast<std::size_t>(pic.height()) * pic.width()) {
            trail.cell_pos.assign(static_cast<std::size_t>(pic.height()) * pic.width(), UINT32_MAX);
        }
        trail.W = pic.width();
        trail.level = static_cast<std::uint16_t>((solve_trail != nullptr ? solve_trail->level : 0) + 1);
    }
    // A valid memo answers the probe (see ProbeMemo).
    ProbeMemo* memo = nullptr;
    if (g_memo_enabled) {
        memo = &g_memo[static_cast<std::size_t>(row) * pic.width() + col].v[val == FULL ? 1 : 0];
        if (memo->gen != 0) {
            bool valid = true;
            for (int w = 0; w < g_memo_words && valid; ++w) {
                std::uint64_t m = memo->touched[w];
                while (m != 0) {
                    const int b = 64 * w + __builtin_ctzll(m);
                    m &= m - 1;
                    if (g_line_stamp[static_cast<std::size_t>(b)] > memo->gen) { valid = false; break; }
                }
            }
            if (g_debug_stats) { ++g_stat_memo_checked; if (valid) ++g_stat_memo_valid; }
            if (valid) {
                if (g_debug_stats && memo->ok) ++g_stat_probe_ok;
                // The caller compares absolute known counts within one pass.
                return memo->ok ? ProbeResult{true, count_solved_pixels(pic) + static_cast<int>(memo->fill)} : ProbeResult{false, 0};
            }
        }
    }
    if (g_debug_stats) { ++g_stat_probes; g_stat_probe_cur = 0; }
    struct ProbeStat { ~ProbeStat() { if (g_debug_stats) ++g_stat_probe_lookups_hist[std::min<std::uint64_t>(g_stat_probe_cur, 63)]; } } probe_stat;

    ProbeGuard<LEARN> guard(pic, trail);

    // Apply the probe pixel (record on trail).
    pic.set_known(row, col, val);
    if constexpr (LEARN) trail.push(trail_pack(row, col), Reason{R_DECISION, 0, trail.level, 0});
    else trail.changed_cell_indices.push_back(trail_pack(row, col));

    pic.mark_row_dirty(row);
    pic.mark_col_dirty(col);

    // No separate full-board validation pass: only the row+col just set can
    // be inconsistent (the board was consistent at probe entry), and both are
    // dirty, so solve_lines below re-solves them and reports any contradiction
    // via solve_line_batch's total==0. solve_check was a redundant O(H+W)
    // re-validation of every line on every probe.
    const bool consistent = propagate_l<LEARN>(mapped_rows, mapped_cols, pic, trail);
    if (memo != nullptr) {
        memo->gen = g_gen;
        memo->ok = consistent ? 1 : 0;
        memo->fill = static_cast<std::uint16_t>(trail.changed_cell_indices.size());
        std::uint64_t t[kMemoWords] = {0, 0, 0};
        const int H = g_memo_H;
        for (int e : trail.changed_cell_indices) {
            const int r = trail_row(e), b = H + trail_col(e);
            t[r >> 6] |= 1ULL << (r & 63);
            t[b >> 6] |= 1ULL << (b & 63);
        }
        for (int w = 0; w < kMemoWords; ++w) memo->touched[w] = t[w];
    }
    if (!consistent) {
        ProbeResult res{false, 0};
        if constexpr (LEARN) {
            // Learn from the contradiction here, while the probe trail still
            // holds its reasons and the conflict record is this probe's (the
            // guard reverts both when we return, and the pair's other probe
            // overwrites the record). The probe pixel is the only decision
            // at the probe level, so the UIP is the probe pixel unless a
            // cell the probe forced lies on every path from it to the
            // conflict. With the pixel as UIP the clause reads "the solve
            // trail's literals it names imply the other value": the forced
            // commit's reason. Otherwise it asserts that forced cell's
            // negation, still implied, but no reason for the commit.
            if (solve_trail != nullptr) {
                const int id = analyze_conflict(trail, solve_trail, pic, mapped_rows, mapped_cols);
                if (id >= 0) {
                    if (g_debug_stats) ++g_stat_probe_learnt;
                    int n = 0;
                    const int* l = g_clauses.lits(id, &n);
                    const int probe_lit = (row * pic.width() + col) * 2 + (val == FULL ? 1 : 0);
#ifdef NONOGRAM_STATS
                    // Every literal is false at the conflict: the asserting
                    // one on the probe trail (the UIP is a probe-level
                    // entry), the rest on the solve trail (the probe trail's
                    // other entries are at the probe level or at level 0).
                    for (int i = 0; i < n; ++i) {
                        const std::size_t cell = static_cast<std::size_t>(l[i] >> 1);
                        STATS_CHECK(pic.pixels[cell] != UNKNOWN && l[i] == (cell_lit(pic, static_cast<int>(cell)) ^ 1));
                        STATS_CHECK(i == 0 ? trail.cell_pos[cell] != UINT32_MAX
                                           : trail.cell_pos[cell] == UINT32_MAX && solve_trail->cell_pos[cell] != UINT32_MAX);
                    }
#endif
                    if (l[0] == (probe_lit ^ 1)) res.learnt_clause = id;
                    else if (g_debug_stats) ++g_stat_probe_uip_elsewhere;  // implied, but no reason for the commit
                }
            }
        }
        return res;
    }

    if (g_debug_stats) ++g_stat_probe_ok;
    const int filled = count_solved_pixels(pic);
    if (bounds != nullptr && filled < bounds->note_below) {
        // Every cell this probe settled is bounded by its fill (see ProbeBounds).
        const int W = pic.width();
        const std::int8_t* px = pic.pixels.data();
        const std::uint64_t stamped = bounds->stamp() | static_cast<std::uint32_t>(filled);
        for (int e : trail.changed_cell_indices) {
            const int r = trail_row(e), c = trail_col(e);
            bounds->note(r, c, px[r * W + c], stamped);
        }
    }
    if (g_debug_impl) {
        g_impl_off.push_back(static_cast<int>(g_impl_rec.size()));
        const int W = pic.width();
        for (int e : trail.changed_cell_indices) {
            const int v = pic.pixels[trail_row(e) * W + trail_col(e)] == FULL ? 1 : 0;
            g_impl_rec.push_back((e << 1) | v);
        }
    }
    return ProbeResult{true, filled};
}

// The learning choice once per probe (see g_learn_mode).
ProbeResult probe_cell(int row,
                       int col,
                       std::int8_t val,
                       const std::vector<const LineSpec*>& mapped_rows,
                       const std::vector<const LineSpec*>& mapped_cols,
                       Picture& pic,
                       ProbeBounds* bounds,
                       const Trail* solve_trail) {
    return g_learn_mode ? probe_cell_t<true>(row, col, val, mapped_rows, mapped_cols, pic, bounds, solve_trail)
                        : probe_cell_t<false>(row, col, val, mapped_rows, mapped_cols, pic, bounds, solve_trail);
}

// Implication-graph analysis of the current node's probe record (see
// g_debug_impl). Fills `forced` with (packed cell, value) pairs that every
// solution below this node must take; returns false when the node is dead.
bool impl_analyze(const Picture& pic, const std::vector<std::pair<int, int>>& unknown_coords,
                  std::vector<std::pair<int, std::int8_t>>& forced) {
    forced.clear();
    const int H = pic.height(), W = pic.width();
    const int U = static_cast<int>(unknown_coords.size());
    const int V = 2 * U;
    std::vector<int> cell_idx(static_cast<std::size_t>(H) * W, -1);
    for (int k = 0; k < U; ++k) cell_idx[unknown_coords[k].first * W + unknown_coords[k].second] = k;
    auto lit = [&](int packed) -> int {
        const int e = packed >> 1;
        const int k = cell_idx[trail_row(e) * W + trail_col(e)];
        return k < 0 ? -1 : 2 * k + (packed & 1);
    };
    // Adjacency as CSR.
    std::vector<int> deg(V + 1, 0);
    std::vector<std::pair<int, int>> edges;
    g_impl_off.push_back(static_cast<int>(g_impl_rec.size()));
    for (std::size_t i = 0; i + 1 < g_impl_off.size(); ++i) {
        const int a = lit(g_impl_rec[g_impl_off[i]]);
        if (a < 0) continue;
        for (int j = g_impl_off[i] + 1; j < g_impl_off[i + 1]; ++j) {
            const int b = lit(g_impl_rec[j]);
            if (b < 0) continue;
            edges.emplace_back(a, b);
            edges.emplace_back(b ^ 1, a ^ 1);
        }
    }
    g_impl_off.pop_back();
    g_stat_impl_edges += edges.size();
    for (auto& e : edges) ++deg[e.first + 1];
    for (int v = 0; v < V; ++v) deg[v + 1] += deg[v];
    std::vector<int> adj(edges.size());
    {
        std::vector<int> pos(deg.begin(), deg.end() - 1);
        for (auto& e : edges) adj[pos[e.first]++] = e.second;
    }
    // Tarjan SCC (iterative).
    std::vector<int> index(V, -1), low(V, 0), comp(V, -1), stk;
    std::vector<char> on(V, 0);
    int idx = 0, ncomp = 0;
    std::vector<int> comp_order;  // components in order of completion = reverse topological
    for (int s = 0; s < V; ++s) {
        if (index[s] >= 0) continue;
        std::vector<std::pair<int, int>> call;  // (vertex, next edge position)
        call.emplace_back(s, deg[s]);
        index[s] = low[s] = idx++; stk.push_back(s); on[s] = 1;
        while (!call.empty()) {
            int v = call.back().first;
            int& p = call.back().second;
            if (p < deg[v + 1]) {
                int w = adj[p++];
                if (index[w] < 0) {
                    index[w] = low[w] = idx++; stk.push_back(w); on[w] = 1;
                    call.emplace_back(w, deg[w]);
                } else if (on[w]) {
                    low[v] = std::min(low[v], index[w]);
                }
            } else {
                if (low[v] == index[v]) {
                    while (true) {
                        int w = stk.back(); stk.pop_back(); on[w] = 0; comp[w] = ncomp;
                        if (w == v) break;
                    }
                    ++ncomp;
                }
                call.pop_back();
                if (!call.empty()) {
                    int u = call.back().first;
                    low[u] = std::min(low[u], low[v]);
                }
            }
        }
    }
    // Tarjan numbers components in reverse topological order (a component is
    // completed after everything it reaches), so reach sets can be built in
    // component order 0..ncomp-1.
    const int words = (V + 63) / 64;
    std::vector<std::uint64_t> reach(static_cast<std::size_t>(ncomp) * words, 0);
    std::vector<std::vector<int>> members(ncomp);
    for (int v = 0; v < V; ++v) members[comp[v]].push_back(v);
    for (int c = 0; c < ncomp; ++c) {
        std::uint64_t* rc = &reach[static_cast<std::size_t>(c) * words];
        for (int v : members[c]) {
            rc[v >> 6] |= 1ULL << (v & 63);
            for (int p = deg[v]; p < deg[v + 1]; ++p) {
                const int w = adj[p];
                const int cw = comp[w];
                if (cw == c) continue;
                const std::uint64_t* rw = &reach[static_cast<std::size_t>(cw) * words];
                for (int i = 0; i < words; ++i) rc[i] |= rw[i];
            }
        }
    }
    for (int k = 0; k < U; ++k) {
        const int l0 = 2 * k, l1 = 2 * k + 1;
        const bool f0 = (reach[static_cast<std::size_t>(comp[l0]) * words + (l1 >> 6)] >> (l1 & 63)) & 1;  // l0 -> l1: l0 false
        const bool f1 = (reach[static_cast<std::size_t>(comp[l1]) * words + (l0 >> 6)] >> (l0 & 63)) & 1;  // l1 -> l0: l1 false
        if (f0 && f1) return false;
        if (f0) forced.emplace_back(trail_pack(unknown_coords[k].first, unknown_coords[k].second), FULL);
        else if (f1) forced.emplace_back(trail_pack(unknown_coords[k].first, unknown_coords[k].second), EMPTY);
    }
    return true;
}

// ---------------------------------------------------------------------------
// solve_backtrack
// ---------------------------------------------------------------------------

// Helper: revert pic to the state captured at branch entry. Walks the trail
// from its current size back to `mark`, restoring each cell to UNKNOWN.
// Restores unknown_count and the solved-row/col sets from snapshots. Drains
// any leftover queue entries (clearing their dirty bits) so the picture
// is in a clean queues-empty / dirty-zero state, matching branch entry.
template <bool LEARN>
void revert_branch_t(Picture& pic,
                     Trail& trail,
                     std::size_t mark,
                     int saved_unknown_count) {
    // Entries below `counted` were subtracted from the per-line counts at a
    // branch node; add them back. Entries above it never were. Two loops
    // over the two ranges, newest first, instead of a test per entry.
    const int* tr = trail.changed_cell_indices.data();
    const std::size_t n = trail.changed_cell_indices.size();
    const std::size_t split = std::max(mark, std::min(n, trail.counted));
    if (g_memo_enabled && n > mark) memo_next_gen();  // a revert changes its lines too
    if constexpr (LEARN) {
        std::uint32_t* cell_pos = trail.cell_pos.data();
        const std::size_t tw = static_cast<std::size_t>(trail.W);
        for (std::size_t i = n; i > mark; ) {
            const int e = tr[--i];
            cell_pos[static_cast<std::size_t>(trail_row(e)) * tw + trail_col(e)] = UINT32_MAX;
        }
    }
    for (std::size_t i = n; i > split; ) {
        const int e = tr[--i];
        pic.unset(trail_row(e), trail_col(e));
        stamp_cell(trail_row(e), trail_col(e));
    }
    for (std::size_t i = split; i > mark; ) {
        const int e = tr[--i];
        trail.unsettle(trail_row(e), trail_col(e));
        pic.unset(trail_row(e), trail_col(e));
        stamp_cell(trail_row(e), trail_col(e));
    }
    if (n > mark) {
        trail.changed_cell_indices.resize(mark);
        if constexpr (LEARN) trail.reasons.resize(mark);
    }
    if constexpr (LEARN) trail.clause_next = std::min(trail.clause_next, mark);
    if (trail.counted > mark) trail.counted = mark;
    pic.unknown_count = saved_unknown_count;
    while (!pic.row_queue.empty()) {
        int i = pic.row_queue.front();
        pic.row_queue.pop_front();
        pic.row_dirty[i] = 0;
    }
    pic.row_queue.reset();
    while (!pic.col_queue.empty()) {
        int j = pic.col_queue.front();
        pic.col_queue.pop_front();
        pic.col_dirty[j] = 0;
    }
    pic.col_queue.reset();
}

void revert_branch(Picture& pic, Trail& trail, std::size_t mark, int saved_unknown_count) {
    if (g_learn_mode) revert_branch_t<true>(pic, trail, mark, saved_unknown_count);
    else revert_branch_t<false>(pic, trail, mark, saved_unknown_count);
}

bool solve_backtrack(const std::vector<const LineSpec*>& mapped_rows,
                     const std::vector<const LineSpec*>& mapped_cols,
                     Picture& pic,
                     SolveState& state,
                     const OnSolution& on_solution,
                     Trail& trail) {
    // Benchmark abort: stop after node_limit backtrack nodes (return false
    // propagates as a stop signal, exactly like the --max callback). The
    // same path serves request_stop() (SIGTERM/SIGINT in main).
    if (state.node_limit && ++state.nodes > state.node_limit) {
        return false;
    }
    if (g_stop_requested) {
        g_stop_path = state.path_string();
        return false;
    }

    const int H = pic.height();
    const int W = pic.width();
    const std::int8_t* px = pic.pixels.data();

    // Per-row / per-col UNKNOWN counts, used as the "most constrained line"
    // signal in the composite sort key (lower => more constrained line).
    // Brought up to date from the trail entries pushed since the previous
    // branch node (see Trail); recounting them here by popcounting every
    // line key was 27% of the latched node on hard/6689.
    const int kw = pic.key_words;
    const std::uint64_t* rk = pic.row_keys.data();
    {
        const int* tr = trail.changed_cell_indices.data();
        const std::size_t n = trail.changed_cell_indices.size();
        for (std::size_t i = trail.counted; i < n; ++i) trail.settle(trail_row(tr[i]), trail_col(tr[i]));
        trail.counted = n;
    }
    const int* uir = trail.row_unknown.data();
    const int* uic = trail.col_unknown.data();

    state.result = 0;  // every early return below is a dead branch
    const char* region = state.region_row.empty() ? nullptr : state.region_row.data();
    int n_unknown = pic.unknown_count;
    if (region) {
        n_unknown = 0;
        for (int r = 0; r < H; ++r) if (region[r]) n_unknown += uir[r];
    }
    if (n_unknown == 0) {
        state.result = 1;
        return true;
    }

    // Looked up before the split and the probing: a hit here saves both.
    StateTable::Key state_key{0, 0};
    const bool use_state_cache = state.count_mode && !g_no_state_cache;
    bool cache_this_node = use_state_cache;  // cleared when the node's size bucket is gated off
    SmallTimer small_timer(n_unknown);
    int nb = 0;  // node-size bucket: log2 of the unknown cells
    while ((n_unknown >> (nb + 1)) != 0 && nb < SolveState::kScBuckets - 1) ++nb;
    if (use_state_cache && state.sc_off[nb]) {
        // Gated off: no key, no lookup, no store (the trail's dirty lists
        // keep accumulating and are applied at the next node that needs
        // the key). Sample again after a stretch, the tree changes.
        cache_this_node = false;
        if (++state.sc_skipped[nb] >= SolveState::kScResample) {
            state.sc_off[nb] = false;
            state.sc_skipped[nb] = 0;
        }
    }
    // Work is measured in cycles (rdtsc): the entry's work byte holds what
    // its subtree cost, which is what a hit saves, in the same currency as
    // the lookup's own cost. Both cost ~25 cycles to read.
    const std::uint64_t tsc_at_entry = cache_this_node ? __rdtsc() : 0;
    if (cache_this_node) {
        // The key combines one 128-bit hash per line with unknowns (XOR in
        // one word, sum in the other, so the order of lines does not
        // matter). Each line's contribution (its hash, or zero once it has
        // no unknowns) and their whole-grid combination are kept in
        // SolveState and updated here for the lines the trail flagged since
        // the last node, so a node pays for its changed lines only: the
        // multiply chain over every line cost 10810 +36% cycles per node,
        // this +4%. A region search combines the region's lines explicitly.
        StateTable::Key* rh = state.row_hash.data();
        StateTable::Key* ch = state.col_hash.data();
        const std::uint64_t* ck = pic.col_keys.data();
        StateTable::Key& gk = state.grid_key;
        for (int r : trail.dirty_rows) {
            gk.a ^= rh[r].a; gk.b -= rh[r].b;
            rh[r] = uir[r] > 0 ? residual_key(state.row_seed[static_cast<std::size_t>(r)], *mapped_rows[static_cast<std::size_t>(r)],
                                              px + static_cast<std::size_t>(r) * W, static_cast<std::size_t>(W),
                                              rk + static_cast<std::size_t>(r) * kw, kw, state.row_memo[static_cast<std::size_t>(r)])
                               : StateTable::Key{0, 0};
            gk.a ^= rh[r].a; gk.b += rh[r].b;
            trail.row_hash_dirty[static_cast<std::size_t>(r)] = 0;
        }
        trail.dirty_rows.clear();
        for (int c : trail.dirty_cols) {
            gk.a ^= ch[c].a; gk.b -= ch[c].b;
            if (uic[c] > 0) {
                std::size_t n;
                const std::int8_t* cells = line_cells(c, true, pic, n);
                ch[c] = residual_key(state.col_seed[static_cast<std::size_t>(c)], *mapped_cols[static_cast<std::size_t>(c)], cells, n,
                                     ck + static_cast<std::size_t>(c) * kw, kw, state.col_memo[static_cast<std::size_t>(c)]);
            } else {
                ch[c] = StateTable::Key{0, 0};
            }
            gk.a ^= ch[c].a; gk.b += ch[c].b;
            trail.col_hash_dirty[static_cast<std::size_t>(c)] = 0;
        }
        trail.dirty_cols.clear();
        if (!region) {
            state_key = StateTable::normalize(gk.a, gk.b);
        } else {
            // The region's rows, and the columns those rows have unknowns in.
            // The rows come from the region bitset less trail bucket 0 (rows
            // without unknowns), not a test of every row of the grid.
            static thread_local std::vector<std::uint64_t> col_mask;
            col_mask.assign(static_cast<std::size_t>(kw), 0);
            std::uint64_t ka = 0, kb = 0;
            const std::uint64_t* zero_rows = trail.row_bucket.data();
            for (int wd = 0; wd < trail.row_words; ++wd) {
                std::uint64_t m = state.region_bits[static_cast<std::size_t>(wd)] & ~zero_rows[wd];
                while (m != 0) {
                    const int r = 64 * wd + __builtin_ctzll(m);
                    m &= m - 1;
                    ka ^= rh[r].a; kb += rh[r].b;
                    for (int w = 0; w < kw; ++w) col_mask[static_cast<std::size_t>(w)] |= rk[r * kw + w] & kUnknownBits;
                }
            }
            for (int w = 0; w < kw; ++w) {
                std::uint64_t m = col_mask[static_cast<std::size_t>(w)];
                while (m != 0) {
                    const int c = 32 * w + (__builtin_ctzll(m) >> 1);
                    m &= m - 1;
                    ka ^= ch[c].a; kb += ch[c].b;
                }
            }
            state_key = StateTable::normalize(ka, kb);
        }
        // The lookup itself follows the split detection below, so the
        // table's cache misses overlap with that scan.
        state.state_cache.prefetch(state_key);
    }

    if (state.count_mode) {
        // Split the region's unknown cells into independent regions: rows
        // and columns joined by an unknown cell. More than one means the
        // count is the product of the regions' counts, each found by the
        // same search restricted to that region's rows.
        //
        // Connected components by closure over the packed row keys: a
        // component grows from its lowest row by OR-ing the unknown bits of
        // every row it reaches into a column mask and sweeping the remaining
        // rows for one that shares a column with the mask, until a sweep
        // adds nothing. Word operations per row per sweep, and the common
        // single-region node ends after the sweep that empties the row set.
        // A union-find over every unknown cell (two finds per cell) was 30%
        // of the branch node on easy_small/4774. The component's root is its
        // lowest row, and components are found in ascending order of that
        // row, which is the order the old row scan first met each root.
        const std::uint64_t* zero_rows = trail.row_bucket.data();  // bucket 0: rows without unknowns
        const int row_words = trail.row_words;
        std::uint64_t rows_left[kMaxFastCells / 64 + 1];  // H <= kMaxFastCells in fast mode; wider grids fall back below
        static thread_local std::vector<std::uint64_t> rows_left_v;
        std::uint64_t* left = rows_left;
        if (row_words > static_cast<int>(sizeof rows_left / sizeof rows_left[0])) {
            rows_left_v.resize(static_cast<std::size_t>(row_words));
            left = rows_left_v.data();
        }
        for (int wd = 0; wd < row_words; ++wd) {
            const int lo = 64 * wd;
            std::uint64_t valid = (H - lo >= 64) ? ~0ULL : ((1ULL << (H - lo)) - 1);
            left[wd] = valid & ~zero_rows[wd];
        }
        if (region) {
            for (int wd = 0; wd < row_words; ++wd) {
                std::uint64_t m = left[wd];
                while (m != 0) {
                    const int r = 64 * wd + __builtin_ctzll(m);
                    m &= m - 1;
                    if (!region[r]) left[wd] &= ~(1ULL << (r & 63));
                }
            }
        }
        // Component i's rows as a bitset at comp_rows[i * row_words ..]; a
        // per-row root array needed an H-wide fill at every node.
        int roots = 0;
        static thread_local std::vector<std::uint64_t> comp_rows;
        comp_rows.clear();
        std::uint64_t col_mask[kMaxFastCells / 32];
        static thread_local std::vector<std::uint64_t> col_mask_v;
        std::uint64_t* cm = col_mask;
        if (kw > static_cast<int>(sizeof col_mask / sizeof col_mask[0])) {
            col_mask_v.resize(static_cast<std::size_t>(kw));
            cm = col_mask_v.data();
        }
        for (;;) {
            int wd0 = 0;
            while (wd0 < row_words && left[wd0] == 0) ++wd0;
            if (wd0 == row_words) break;
            const int r0 = 64 * wd0 + __builtin_ctzll(left[wd0]);
            left[wd0] &= left[wd0] - 1;
            comp_rows.resize(comp_rows.size() + static_cast<std::size_t>(row_words), 0);
            std::uint64_t* comp = comp_rows.data() + static_cast<std::size_t>(roots) * row_words;
            comp[wd0] |= 1ULL << (r0 & 63);
            ++roots;
            for (int w = 0; w < kw; ++w) cm[w] = rk[r0 * kw + w] & kUnknownBits;
            bool grew = true;
            while (grew) {
                grew = false;
                for (int wd = wd0; wd < row_words; ++wd) {
                    std::uint64_t m = left[wd];
                    while (m != 0) {
                        const int r = 64 * wd + __builtin_ctzll(m);
                        m &= m - 1;
                        const std::uint64_t* k = rk + static_cast<std::size_t>(r) * kw;
                        std::uint64_t hit = 0;
                        for (int w = 0; w < kw; ++w) hit |= k[w] & cm[w];
                        if (hit == 0) continue;
                        for (int w = 0; w < kw; ++w) cm[w] |= k[w] & kUnknownBits;
                        left[wd] &= ~(1ULL << (r & 63));
                        comp[wd] |= 1ULL << (r & 63);
                        grew = true;
                    }
                }
            }
        }
        if (g_debug_stats && !state.used_backtrack && !region) {
            std::fprintf(stderr, "cache-stats: regions at first branch=%d, unknown cells per region:", roots);
            for (int i = 0; i < roots; ++i) {
                int cells = 0;
                const std::uint64_t* comp = comp_rows.data() + static_cast<std::size_t>(i) * row_words;
                for (int r = 0; r < H; ++r) if ((comp[r >> 6] >> (r & 63)) & 1) cells += uir[r];
                std::fprintf(stderr, " %d", cells);
            }
            std::fprintf(stderr, "\n");
        }
        if (cache_this_node) {
            ++state.state_lookups;
            int hit_work = 0;
            if (g_debug_stats) ++g_stat_sc_lookups[nb];
            const u128* hit = state.state_cache.find(state_key, &hit_work);
            state.sc_lookup_cycles[nb] += __rdtsc() - tsc_at_entry;
            if (hit) {
                ++state.state_hits;
                const std::uint64_t saved = StateTable::decode_work(hit_work);
                state.sc_saved[nb] += saved;
                if (g_debug_stats) { ++g_stat_sc_hits[nb]; g_stat_sc_saved[nb] += saved; }
            }
            if (++state.sc_lookups[nb] == SolveState::kScWindow) {
                if (static_cast<double>(state.sc_saved[nb]) < state.sc_yield * static_cast<double>(state.sc_lookup_cycles[nb])) {
                    state.sc_off[nb] = true;
                    if (g_debug_stats) ++g_stat_sc_gated[nb];
                }
                state.sc_lookups[nb] = 0;
                state.sc_saved[nb] = 0;
                state.sc_lookup_cycles[nb] = 0;
            }
            if (hit && !g_state_cache_probe_only) {
                state.result = *hit;
                return true;
            }
        }
        if (roots > 1) {
            ++state.regions_split;
            // Each region search below sums its leaves to this node's whole
            // share of the search space, so k regions would add it k times;
            // the share is restored once when the split completes.
            const double mass_at_split = g_explored_mass;
            std::vector<char> saved_region = state.region_row;
            std::vector<std::uint64_t> saved_region_bits = state.region_bits;
            const std::vector<std::uint64_t> comps(comp_rows.begin(), comp_rows.end());
            const double scale_at_split = g_mass_scale;
            g_mass_scale = scale_at_split / static_cast<double>(roots);
            u128 total = 1;
            for (int region_index = 0; region_index < roots; ++region_index) {
                const char region_char = static_cast<char>('a' + (region_index & 15));
                // A region seen before (the branch cell's lines split it
                // off, so it is the same under both branch values) is a
                // state-cache hit at the region search's root node: the
                // key there is the region's lines, exactly this region.
                const std::uint64_t* comp = comps.data() + static_cast<std::size_t>(region_index) * row_words;
                state.region_row.assign(static_cast<std::size_t>(H), 0);
                for (int r = 0; r < H; ++r) state.region_row[static_cast<std::size_t>(r)] = static_cast<char>((comp[r >> 6] >> (r & 63)) & 1);
                state.region_bits.assign(comp, comp + row_words);
                ++state.region_calls;
                const std::size_t mark = trail.changed_cell_indices.size();
                const int saved_unknown_count = pic.unknown_count;
                state.path_push(region_char);
                if (g_learn_mode) ++g_region_depth;  // no clause pass inside (see g_region_depth)
                if (!solve_backtrack(mapped_rows, mapped_cols, pic, state, on_solution, trail)) return false;
                if (g_learn_mode) {
                    --g_region_depth;
                    // The pass never saw the region's entries, which the
                    // revert below drops, so it resumes at this node's fixpoint.
                    STATS_CHECK(trail.clause_next <= mark);
                }
                state.path_pop();
                total *= state.result;
                revert_branch(pic, trail, mark, saved_unknown_count);
                if (total == 0) break;
            }
            state.region_row = saved_region;
            state.region_bits = saved_region_bits;
            state.result = total;
            g_mass_scale = scale_at_split;
            g_explored_mass = mass_at_split + g_half_pow[static_cast<std::size_t>(state.branch_depth)] * scale_at_split;
            if (cache_this_node) {
                if (state.state_cache.insert(state_key, total, __rdtsc() - tsc_at_entry)) ++state.state_evictions;
            }
            return true;
        }
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

    // Composite key: ascending line_constraint (uir[r] + uic[c]), then
    // descending neighbor, compared as the pair (line_constraint, -neighbor)
    // so std::less gives the desired order (a larger neighbor wins ties).

    if (state.skip_probing || (g_small_noprobe > 0 && n_unknown <= g_small_noprobe && state.branch_depth > 0)) {
        // Pick the unknown cell with the smallest composite key (most-
        // constrained line; tie-broken by highest neighbor score). Single
        // row-major scan, no coords/scores arrays materialized.
        // The winner is the row-major-first cell with the minimum key, and a
        // cell in a row with k unknowns has line_constraint >= k + min_uic
        // (min_uic = smallest unknown count of any column that still has
        // unknowns). So rows are visited by increasing k from the trail's
        // buckets, each row scanned once for its best cell, and the walk
        // stops at the first k whose bound exceeds the best key found. Rows
        // the old full scan would have picked from all have bound <= that
        // key, so they are all visited; ties on the key go to the smaller
        // row, then the smaller column, which is the row-major-first cell.
        std::pair<int, int> best_key;
        bool have = false;
        int min_uic = 1;
        while (trail.col_hist[static_cast<std::size_t>(min_uic)] == 0) ++min_uic;
        const int words = trail.row_words;
        for (int k = 1; k <= W; ++k) {
            if (have && k + min_uic > best_key.first) break;
            const std::uint64_t* bucket = trail.row_bucket.data() + static_cast<std::size_t>(k) * words;
            for (int wd = 0; wd < words; ++wd) {
                std::uint64_t rows = bucket[wd];
                while (rows != 0) {
                    const int r = 64 * wd + __builtin_ctzll(rows);
                    rows &= rows - 1;
                    if (region && !region[r]) continue;
                    const int uir_r = uir[r];
                    for (int w = 0; w < kw; ++w) {
                        std::uint64_t m = rk[r * kw + w] & kUnknownBits;
                        while (m != 0) {
                            const int c = 32 * w + (__builtin_ctzll(m) >> 1);
                            m &= m - 1;
                            // The key is (line constraint, -neighbours) in
                            // lexicographic order, so a cell whose line
                            // constraint alone loses needs no neighbour
                            // count (four pixel loads and branches).
                            const int lc = uir_r + uic[c];
                            if (have && lc > best_key.first) continue;
                            const std::pair<int, int> key(lc, -neighbor_score(r, c));
                            if (!have || key < best_key || (key == best_key && r < best_row)) {
                                best_key = key;
                                best_row = r;
                                best_col = c;
                                have = true;
                            }
                        }
                    }
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
        // Enumerate the unknown cells from the packed row keys (same
        // row-major, ascending-column order as a cell scan): each set bit of
        // key & kUnknownBits is an unknown cell at column 32*w + bit/2.
        unknown_coords.clear();
        enc.clear();
        int max_enc = 0;
        for (int r = 0; r < H; ++r) {
            if (region && !region[r]) continue;
            for (int w = 0; w < kw; ++w) {
                std::uint64_t m = rk[r * kw + w] & kUnknownBits;
                while (m != 0) {
                    const int c = 32 * w + (__builtin_ctzll(m) >> 1);
                    m &= m - 1;
                    unknown_coords.emplace_back(r, c);
                    int lc = uir[r] + uic[c];
                    int e = lc * 5 + (4 - neighbor_score(r, c));
                    enc.push_back(e);
                    if (e > max_enc) max_enc = e;
                }
            }
        }

        const std::size_t nu = static_cast<std::size_t>(n_unknown);
        static thread_local std::vector<int> order;
        static thread_local std::vector<int> bucket;
        order.resize(nu);
        bucket.assign(static_cast<std::size_t>(max_enc) + 2, 0);
        for (std::size_t i = 0; i < nu; ++i) bucket[enc[i] + 1]++;
        for (int k = 1; k <= max_enc + 1; ++k) bucket[k] += bucket[k - 1];
        for (std::size_t i = 0; i < nu; ++i) order[bucket[enc[i]]++] = static_cast<int>(i);

        int best_pixels = -1;
        int best_other = INT_MAX;          // the larger probe fill of the best cell
        int best_lo = -1;                  // the smaller one (ANYTIME_TB experiment)
        double best_score = -HUGE_VAL;     // BRANCH_K experiment only
        bool have_best = false;
        // Forced cells are committed in place and the pass continues; the
        // node restarts once at the end of a pass that committed anything
        // (see the restart below), not once per forced cell.
        bool committed = false;
        const std::int8_t* pxs = pic.pixels.data();
        if (g_debug_impl) { g_impl_rec.clear(); g_impl_off.clear(); }

        // A probe whose bound (ProbeBounds) is below the best score so far
        // is consistent and cannot become the branch cell, so it is skipped
        // and the bound stands in for its fill. The BRANCH_K score needs
        // both real fills and the ANYTIME_TB experiments read the loser's
        // fill on a tie, so the skip is off in those; it is on in the
        // shipped min-balanced and anytime-max orders.
        static thread_local ProbeBounds bounds;
        const bool skip_ok = !g_no_probe_skip && state.balance_k <= 0.0 && g_anytime_tb == 0;
        bounds.reset(H, W);

        for (std::size_t oi = 0; oi < order.size(); ++oi) {
            const int idx = order[oi];
            int row = unknown_coords[idx].first;
            int col = unknown_coords[idx].second;
            if (g_memo_enabled && oi + 4 < order.size()) {
                // The memo pair (one line) of a cell four probes ahead.
                const auto& nx = unknown_coords[static_cast<std::size_t>(order[oi + 4])];
                __builtin_prefetch(&g_memo[static_cast<std::size_t>(nx.first) * W + nx.second]);
            }
            if (pxs[row * W + col] != UNKNOWN) continue;  // settled by an earlier commit this pass
            // A probe's clause pass sees only the probe's cells, so the node
            // must be at its clause fixpoint (see clause_pass); inside a
            // region search neither trail runs a pass (see g_region_depth).
            if (g_learn_mode) STATS_CHECK(g_region_depth > 0 || trail.clause_next == trail.changed_cell_indices.size());

            // Each value is skipped only when a bound proves it consistent
            // (bound set) AND below the best score. A skipped value is never
            // a contradiction, so the "exactly one probe contradicts" test
            // that forces a cell, and the "both contradict" test that kills
            // the node, both see the same outcomes a full probe would: the
            // strategy label and the tree are unchanged. In particular, a
            // contradicting FULL never lets EMPTY be assumed consistent --
            // that would mark a contradiction on a node that is merely dead.
            ProbeResult full_res, empty_res;
            bounds.note_below = skip_ok ? best_pixels : INT_MIN;
            const int bound_full = skip_ok ? bounds.bound(row, col, FULL) : INT_MAX;
            if (bound_full < best_pixels) {
                if (g_debug_stats) ++g_stat_probe_skips;
                full_res = ProbeResult{true, bound_full};
            } else {
                full_res = probe_cell(row, col, FULL, mapped_rows, mapped_cols, pic, &bounds, &trail);
            }
            const int bound_empty = skip_ok ? bounds.bound(row, col, EMPTY) : INT_MAX;
            if (bound_empty < best_pixels) {
                if (g_debug_stats) ++g_stat_probe_skips;
                empty_res = ProbeResult{true, bound_empty};
            } else {
                empty_res = probe_cell(row, col, EMPTY, mapped_rows, mapped_cols, pic, &bounds, &trail);
            }

            state.record_probe((!full_res.ok) || (!empty_res.ok));

            if (!full_res.ok && !empty_res.ok) {
                return true; // dead branch (learning: each probe learnt its clause)
            }

            if (full_res.ok != empty_res.ok) {
                state.mark_contradiction();
                const std::int8_t forced = full_res.ok ? FULL : EMPTY;
                // Forced commit: record the pixel on the trail so a parent
                // backtrack frame can revert it if its own branch fails, then
                // propagate in place and keep probing the remaining cells.
                if (g_memo_enabled) memo_next_gen();
                pic.set_known(row, col, forced);
                // Learning: the contradicting probe's clause asserts this
                // value from solve-trail literals alone, so it is the
                // commit's reason (a unit clause: true in every solution,
                // level 0, as clause_pass records it). Without one (no
                // clause learnt, or its UIP was not the probe pixel):
                // UINT32_MAX, no antecedents, like a decision.
                Reason reason{R_PROBE, 0, trail.level, UINT32_MAX};
                if (g_learn_mode) {
                    const int learnt = full_res.ok ? empty_res.learnt_clause : full_res.learnt_clause;
                    if (learnt >= 0) {
                        reason.id = static_cast<std::uint32_t>(learnt);
                        if (g_clauses.is_unit(learnt)) reason.level = 0;
                    }
                }
                trail.record(trail_pack(row, col), reason);
                stamp_cell(row, col);
                pic.mark_row_dirty(row);
                pic.mark_col_dirty(col);
                if (!propagate(mapped_rows, mapped_cols, pic, trail)) {  // dead branch
                    if (g_learn_mode) analyze_conflict(trail, nullptr, pic, mapped_rows, mapped_cols);
                    return true;
                }
                if (pic.is_solved()) {
                    return solve_real(mapped_rows, mapped_cols, pic, state, on_solution, trail);
                }
                bounds.invalidate();  // the board grew: earlier bounds no longer hold
                committed = true;
                continue;
            }

            // Branch-cell score. The search is exhaustive, so what matters is
            // the size of BOTH subtrees: scoring a cell by the smaller of its
            // two propagations (a balanced split) shrinks the tree far more
            // than scoring by the larger one -- 9-Dom (webpbn 8098) 116k ->
            // 5.3k nodes, extreme/6574 102k -> 2.5k, the corpus -63%. Sum,
            // product and Wu et al.'s Min-logd (min plus a log asymmetry
            // bonus) were all measured worse.
            // Ties on the min go to the cell whose OTHER probe fills the
            // least, i.e. the more balanced pair: 9-Dom 5,340 -> 3,232 nodes,
            // 6574 2,508 -> 154, easy_large/5281 6,776 -> 1,375, no puzzle
            // in the corpus worse by more than 7 nodes, counts and strategy
            // labels unchanged. Preferring the imbalanced pair instead gives
            // 8,114 / 10,305.
            // Anytime mode is the exception: on an enumeration that never
            // finishes the objective is solutions per second, and the
            // greedy max (deepest dive first) delivers them 11% faster on
            // pikachu, so it keeps max with first-found ties.
            const int f = full_res.pixels_filled, e = empty_res.pixels_filled;
            const int lo = std::min(f, e), hi = std::max(f, e);
            bool better;
            const bool max_order = state.keep_probing || g_branch_max;
            if (max_order) {
                better = hi > best_pixels;
                if (g_anytime_tb == 1) better = better || (hi == best_pixels && lo > best_lo);
                if (g_anytime_tb == 2) better = better || (hi == best_pixels && lo < best_lo);
            } else if (state.balance_k > 0.0) {
                const double sc = state.balance_k * lo - hi;
                better = sc > best_score;
                if (better) best_score = sc;
            } else {
                better = lo > best_pixels || (lo == best_pixels && hi < best_other);
            }
            if (better) {
                best_pixels = max_order ? hi : lo;
                best_other = hi;
                best_lo = lo;
                best_row = row;
                best_col = col;
                best_first_val = ((f >= e) != g_first_val_low) ? FULL : EMPTY;
                have_best = true;
                if (g_early_solve && state.keep_probing && hi == H * W) break;
            }
        }

        if (committed) {
            // Deferred restart. Forcedness is monotone in the board, so the
            // set of cells committed by "commit every forced cell found in a
            // pass, then re-pass" is the same closure the old per-cell
            // restart reached, and the final pass commits nothing, so every
            // score the branch choice sees is fresh: the tree is identical,
            // with one restart per pass instead of one per forced cell
            // (9-Dom: 6.0M -> 4.0M probes for the same 5,340 nodes).
            return solve_real(mapped_rows, mapped_cols, pic, state, on_solution, trail);
        }
        if (!have_best) {
            return true;
        }
        if (g_debug_impl) {
            static thread_local std::vector<std::pair<int, std::int8_t>> forced;
            ++g_stat_impl_nodes;
            const bool alive = impl_analyze(pic, unknown_coords, forced);
            if (!alive) ++g_stat_impl_dead;
            if (!forced.empty()) { ++g_stat_impl_nodes_forced; g_stat_impl_forced += forced.size(); }
            if (g_debug_impl >= 2) {
                if (!alive) return true;
                if (!forced.empty()) {
                    state.mark_contradiction();
                    for (auto& fc : forced) {
                        const int r = trail_row(fc.first), c = trail_col(fc.first);
                        if (pxs[r * W + c] != UNKNOWN) {
                            if (pxs[r * W + c] != fc.second) return true;  // dead
                            continue;
                        }
                        if (g_memo_enabled) memo_next_gen();
                        pic.set_known(r, c, fc.second);
                        trail.record(fc.first, Reason{R_DECISION, 0, trail.level, 0});
                        stamp_cell(r, c);
                        pic.mark_row_dirty(r);
                        pic.mark_col_dirty(c);
                        if (!propagate(mapped_rows, mapped_cols, pic, trail)) return true;
                    }
                    return solve_real(mapped_rows, mapped_cols, pic, state, on_solution, trail);
                }
            }
        }
    }

    if (g_debug_stats && !state.used_backtrack) {
        // Search area at the first branch, after root propagation and
        // probing settled all they can: unknown cells, and the number of
        // clue placements still consistent with each line, as log2 of the
        // product over rows and over columns (min of the two bounds the
        // number of grids the search can visit).
        double lr = 0.0, lc = 0.0;
        for (int r = 0; r < H; ++r) lr += std::log2(count_line_completions(pic.pixels.data() + static_cast<std::size_t>(r) * W, 1, W, (*g_row_clues)[r]));
        std::vector<std::int8_t> col(static_cast<std::size_t>(H));
        for (int c = 0; c < W; ++c) {
            for (int r = 0; r < H; ++r) col[r] = pic.pixels[static_cast<std::size_t>(r) * W + c];
            lc += std::log2(count_line_completions(col.data(), 1, H, (*g_col_clues)[c]));
        }
        std::fprintf(stderr, "cache-stats: search-area=%d of %d cells; log2 placements rows=%.1f cols=%.1f min=%.1f\n",
                     pic.unknown_count, H * W, lr, lc, std::min(lr, lc));
    }
    // STATE_CACHE: the region's state is its rows with unknowns (with their
    // packed keys) plus the columns those rows have unknowns in; a region
    // row's known cells are in its key, so the key determines the count.
    state.mark_backtrack();
    if (g_debug_stats) ++g_stat_nodes;
    const std::uint64_t nodes_before = g_stat_all_nodes;
    if ((++state.branch_nodes & 0xFFFF) == 0 && g_progress_interval > 0.0) maybe_print_progress();
    const int sols_before = state.solutions_found;
    const bool latched_here = state.skip_probing;
    if (g_debug_stats) ++g_stat_all_nodes;

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
    u128 subtree_total = 0;
    for (int branch = 0; branch < 2; ++branch) {
        const std::int8_t val = (branch == 0) ? first_val : second_val;

        const std::size_t mark = trail.changed_cell_indices.size();
        const int saved_unknown_count = pic.unknown_count;

        // Apply the branch pixel (record on trail).
        if (g_memo_enabled) memo_next_gen();
        if (g_learn_mode) {  // the branch pixel opens a decision level
            STATS_CHECK(trail.level < UINT16_MAX);
            ++trail.level;
        }
        pic.set_known(row, col, val);
        trail.record(trail_pack(row, col), Reason{R_DECISION, 0, trail.level, 0});
        stamp_cell(row, col);
        pic.mark_row_dirty(row);
        pic.mark_col_dirty(col);

        const double mass_before = g_explored_mass;
        ++state.branch_depth;
        state.path_push(branch == 0 ? '0' : '1');
        const bool go_on = solve_real(mapped_rows, mapped_cols, pic, state, on_solution, trail);
        state.path_pop();
        --state.branch_depth;
        if (!go_on) {
            return false;
        }
        subtree_total += state.result;
        g_counted_so_far += state.result;  // a finished child; taken back when this node returns its own total
        // A child that branched has already accounted for its own subtree
        // through its children; only a leaf child is added here.
        if (g_explored_mass == mass_before) g_explored_mass += g_half_pow[static_cast<std::size_t>(state.branch_depth) + 1] * g_mass_scale;
        // The insert below touches the key's two windows, evicted from the
        // caches by the subtree; start fetching them under the last revert.
        if (branch == 1 && cache_this_node) state.state_cache.prefetch(state_key);
        revert_branch(pic, trail, mark, saved_unknown_count);
        if (g_learn_mode) {
            --trail.level;
            reduce_learnt(trail);  // between subtrees: no propagation is in flight
        }
    }
    state.result = subtree_total;
    g_counted_so_far -= subtree_total;  // the parent adds it back as one finished child
    if (cache_this_node) {
        if (state.state_cache.insert(state_key, subtree_total, __rdtsc() - tsc_at_entry)) ++state.state_evictions;
    }

    if (latched_here) {
        // Count mode has the exact subtree count; the solution counter
        // misses region leaves and state-cache hits, and a hit subtree
        // read as dead made the watchdog pin probing on (10810: 32 probes
        // per node instead of 12, 15% behind the no-cache run at 60 s).
        const bool dead = state.count_mode ? subtree_total == 0 : state.solutions_found == sols_before;
        state.note_latched_subtree(dead);
        if (g_debug_stats) {
            const std::uint64_t n = g_stat_all_nodes - nodes_before;
            int b = 0; while ((n >> (b + 1)) != 0 && b < 39) ++b;
            if (dead) ++g_stat_dead_hist[b]; else ++g_stat_live_hist[b];
        }
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
    if (!propagate(mapped_rows, mapped_cols, pic, trail)) {
        if (g_learn_mode) analyze_conflict(trail, nullptr, pic, mapped_rows, mapped_cols);
        state.result = 0;
        return true;
    }

    if (pic.is_solved()) {
        state.solution_found();
        state.result = 1;
        if (state.count_mode) {
            if (!g_first_solution_printed && g_progress_interval > 0.0) {
                g_first_solution_printed = true;
                std::printf("first solution after %.1fs\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - g_progress_start).count());
                std::fflush(stdout);
            }
            return true;
        }
        return on_solution(pic);
    }

    return solve_backtrack(mapped_rows, mapped_cols, pic, state, on_solution, trail);
}

// One Knuth dive: descend a single random root-to-leaf path, accumulating the
// product of viable-branch counts. Returns that weight if the path reaches a
// full solution, 0.0 if it dead-ends. Mutates pic in place (no backtracking;
// the caller discards pic after each dive).
double estimate_dive(const std::vector<const LineSpec*>& mapped_rows,
                     const std::vector<const LineSpec*>& mapped_cols,
                     Picture& pic, Trail& trail, std::mt19937_64& rng) {
    const int H = pic.height();
    const int W = pic.width();
    double weight = 1.0;
    std::vector<int> ur(static_cast<std::size_t>(H)), uc(static_cast<std::size_t>(W));
    while (true) {
        if (!propagate(mapped_rows, mapped_cols, pic, trail)) return 0.0;
        if (pic.is_solved()) return weight;

        // Pick the most-constrained unknown cell (fewest unknowns in row+col).
        std::fill(ur.begin(), ur.end(), 0);
        std::fill(uc.begin(), uc.end(), 0);
        const std::int8_t* px = pic.pixels.data();
        for (int r = 0; r < H; ++r) {
            const std::int8_t* row = px + static_cast<std::size_t>(r) * W;
            for (int c = 0; c < W; ++c) if (row[c] == UNKNOWN) { ++ur[r]; ++uc[c]; }
        }
        int br = -1, bc = -1, bk = INT_MAX;
        for (int r = 0; r < H; ++r) {
            const std::int8_t* row = px + static_cast<std::size_t>(r) * W;
            for (int c = 0; c < W; ++c) {
                if (row[c] != UNKNOWN) continue;
                int k = ur[r] + uc[c];
                if (k < bk) { bk = k; br = r; bc = c; }
            }
        }
        if (br < 0) return weight;  // no unknowns (already solved-equivalent)

        // Count viable values via lookahead probes (each reverts pic).
        ProbeResult pf = probe_cell(br, bc, FULL, mapped_rows, mapped_cols, pic, nullptr, nullptr);
        ProbeResult pe = probe_cell(br, bc, EMPTY, mapped_rows, mapped_cols, pic, nullptr, nullptr);
        const int nv = (pf.ok ? 1 : 0) + (pe.ok ? 1 : 0);
        if (nv == 0) return 0.0;
        std::int8_t val;
        if (nv == 2) { val = (rng() & 1ULL) ? FULL : EMPTY; weight *= 2.0; }
        else         { val = pf.ok ? FULL : EMPTY; }
        pic.set_pixel(br, bc, val);
        pic.mark_row_dirty(br);
        pic.mark_col_dirty(bc);
    }
}

} // anonymous namespace

namespace {
// LEARN_CHECK=1 (stats build): every clause the solve learnt must hold in
// every solution. The solutions come from a separate enumeration of the same
// puzzle without learning (count mode's region split and state cache never
// visit whole grids), stopped past 10,000 solutions, when the check is
// skipped. Prints each violated clause as "learn-check: clause violated".
void learn_check(const std::vector<std::vector<int>>& rows, const std::vector<std::vector<int>>& cols) {
    constexpr std::size_t kMaxSolutions = 10000;
    std::vector<std::vector<std::int8_t>> sols;
    bool too_many = false;
    const std::vector<std::vector<int>> clauses = g_check_clauses;
    g_learn_check_inner = true;
    solve(rows, cols, [&](const Picture& p) {
        if (sols.size() >= kMaxSolutions) { too_many = true; return false; }
        sols.push_back(p.pixels);
        return true;
    }, nullptr, false, 0.0, false, nullptr, false);
    g_learn_check_inner = false;
    if (too_many) {
        std::fprintf(stderr, "learn-check: skipped, more than %zu solutions (%zu clauses learnt)\n", kMaxSolutions, clauses.size());
        return;
    }
    std::size_t violated = 0;
    for (std::size_t k = 0; k < clauses.size(); ++k) {
        for (std::size_t s = 0; s < sols.size(); ++s) {
            bool sat = false;
            for (const int l : clauses[k]) sat = sat || sols[s][static_cast<std::size_t>(l >> 1)] == ((l & 1) ? FULL : EMPTY);
            if (sat) continue;
            ++violated;
            std::fprintf(stderr, "learn-check: clause violated: learnt #%zu (%zu literals) by solution %zu:", k, clauses[k].size(), s);
            for (const int l : clauses[k]) std::fprintf(stderr, " %c%d", (l & 1) ? '+' : '-', l >> 1);
            std::fprintf(stderr, "\n");
            break;
        }
    }
    std::fprintf(stderr, "learn-check: %zu clauses checked against %zu solutions, %zu violated\n", clauses.size(), sols.size(), violated);
}
}  // namespace

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

void solve(const std::vector<std::vector<int>>& rows,
           const std::vector<std::vector<int>>& cols,
           std::function<bool(const Picture&)> on_solution,
           Strategy* out_strategy,
           bool keep_probing,
           double balance_k,
           bool count_mode,
           std::string* out_count,
           bool learn) {
    // --learn turns clause-learning mode on; the stats build also takes
    // LEARN=1, so bench/nodes.py --env LEARN=1 gates the mode.
    g_learn_mode = learn;
#ifdef NONOGRAM_STATS
    if (const char* learn_env = std::getenv("LEARN"); learn_env != nullptr && !g_learn_check_inner)
        g_learn_mode = g_learn_mode || std::strcmp(learn_env, "1") == 0;
#endif
    if (g_learn_mode) {
        std::size_t max_clauses = 50000;
        if (const char* env = std::getenv("LEARN_MAX_CLAUSES")) {
            try { max_clauses = static_cast<std::size_t>(std::stoull(env)); } catch (...) {}
        }
        g_clauses.init(static_cast<int>(rows.size() * cols.size()), max_clauses);
        g_conflict_clause = -1;
        g_conflict_line = -1;
        g_conflict_line_is_col = false;
        g_check_clauses.clear();
        g_region_depth = 0;
    }
    std::size_t budget = 1024ULL * 1024ULL * 1024ULL;  // 1 GB default
    const char* env = std::getenv("LINE_CACHE_BUDGET_MB");
    if (env != nullptr) {
        try {
            budget = static_cast<std::size_t>(std::stoull(env)) * 1024ULL * 1024ULL;
        } catch (...) {}
    }
    const bool anytime = keep_probing || (std::getenv("ANYTIME") != nullptr);
    reset_line_cache(static_cast<int>(rows.size()), static_cast<int>(cols.size()), budget, anytime);

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
        spec_pool.back().id = static_cast<int>(spec_pool.size()) - 1;
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
    g_row_tags.resize(mapped_rows.size());
    for (std::size_t i = 0; i < mapped_rows.size(); ++i)
        g_row_tags[i] = static_cast<std::uint16_t>(mapped_rows[i]->id * 2);
    g_col_tags.resize(mapped_cols.size());
    for (std::size_t i = 0; i < mapped_cols.size(); ++i)
        g_col_tags[i] = static_cast<std::uint16_t>(mapped_cols[i]->id * 2 + g_tag_col_bit);

    SolveState state;
    state.keep_probing = anytime;
    g_explored_mass = 0.0;
    // Branch depth cannot exceed the cell count; 2^-d underflows to 0 past
    // ~1074, which only drops mass that could never register anyway.
    g_half_pow.resize(static_cast<std::size_t>(H) * W + 2);
    for (std::size_t d = 0; d < g_half_pow.size(); ++d) g_half_pow[d] = std::ldexp(1.0, -static_cast<int>(d));
    g_row_clues = &rows;
    g_col_clues = &cols;
    // The stats build's BRANCH_K knob is the same switch, for bench/nodes.py.
    state.balance_k = g_branch_k_set ? g_branch_k : balance_k;
    Trail trail;
    // Reserve enough headroom that the trail rarely reallocates.
    trail.changed_cell_indices.reserve(static_cast<std::size_t>(H) * static_cast<std::size_t>(W));
    trail.W = W;
    trail.cell_pos.assign(static_cast<std::size_t>(H) * static_cast<std::size_t>(W), UINT32_MAX);
    trail.reasons.reserve(static_cast<std::size_t>(H) * static_cast<std::size_t>(W));
    trail.level = 0;
    trail.row_unknown.assign(static_cast<std::size_t>(H), W);
    trail.col_unknown.assign(static_cast<std::size_t>(W), H);
    trail.counted = 0;
    trail.row_words = (H + 63) / 64;
    trail.row_bucket.assign(static_cast<std::size_t>(W + 1) * trail.row_words, 0);
    for (int r = 0; r < H; ++r)
        trail.row_bucket[static_cast<std::size_t>(W) * trail.row_words + (r >> 6)] |= 1ULL << (r & 63);
    trail.col_hist.assign(static_cast<std::size_t>(H + 1), 0);
    trail.col_hist[static_cast<std::size_t>(H)] = W;
    state.count_mode = count_mode;
    // Every branch settles a cell and every region split nests inside a
    // branch with fewer cells, so the path holds at most one character per
    // cell of each kind.
    state.path_buf.assign(2 * static_cast<std::size_t>(H) * static_cast<std::size_t>(W) + 4, 0);
    state.path_len = 0;
    if (count_mode && !g_no_state_cache) {
        std::size_t state_budget = 512ULL * 1024ULL * 1024ULL;
        if (const char* env = std::getenv("STATE_CACHE_MB")) {
            try { state_budget = static_cast<std::size_t>(std::stoull(env)) * 1024ULL * 1024ULL; } catch (...) {}
        }
        state.state_cache.init(state_budget);
        state.row_hash.assign(static_cast<std::size_t>(H), StateTable::Key{0, 0});
        state.col_hash.assign(static_cast<std::size_t>(W), StateTable::Key{0, 0});
        state.row_seed.resize(static_cast<std::size_t>(H));
        state.col_seed.resize(static_cast<std::size_t>(W));
        state.row_memo.assign(static_cast<std::size_t>(H), ResidualMemo{});
        state.col_memo.assign(static_cast<std::size_t>(W), ResidualMemo{});
        for (int r = 0; r < H; ++r) state.row_seed[static_cast<std::size_t>(r)] = line_seed(static_cast<std::uint64_t>(r) | (1ULL << 40));
        for (int c = 0; c < W; ++c) state.col_seed[static_cast<std::size_t>(c)] = line_seed(static_cast<std::uint64_t>(c) | (1ULL << 41));
        state.grid_key = StateTable::Key{0, 0};
        state.sc_yield = g_state_cache_yield;
        // Every line starts flagged, so the first node computes them all.
        trail.track_hash = true;
        trail.row_hash_dirty.assign(static_cast<std::size_t>(H), 1);
        trail.col_hash_dirty.assign(static_cast<std::size_t>(W), 1);
        trail.dirty_rows.resize(static_cast<std::size_t>(H));
        trail.dirty_cols.resize(static_cast<std::size_t>(W));
        for (int r = 0; r < H; ++r) trail.dirty_rows[static_cast<std::size_t>(r)] = r;
        for (int c = 0; c < W; ++c) trail.dirty_cols[static_cast<std::size_t>(c)] = c;
    }
#ifdef NONOGRAM_STATS
    g_memo_enabled = g_probe_memo && g_fast_mode && H + W <= 64 * kMemoWords;
#endif
    if (g_memo_enabled) {
        g_memo_H = H;
        g_memo_words = (H + W + 63) / 64;
        g_memo.assign(static_cast<std::size_t>(H) * W, ProbeMemoPair{});
        g_line_stamp.assign(static_cast<std::size_t>(H + W), 0u);
        g_gen = 1;
        trail.stamps = true;
    }
    const std::uint64_t solve_tsc0 = g_debug_stats ? __rdtsc() : 0;
    (void)solve_real(mapped_rows, mapped_cols, pic, state, on_solution, trail);
    if (g_debug_stats && count_mode) {
        const double total = static_cast<double>(__rdtsc() - solve_tsc0);
        std::fprintf(stderr, "cache-stats: small-region subtrees: <=31 cells %.1f%% of cycles (%llu roots), <=15 cells %.1f%% (%llu roots)\n",
                     100.0 * static_cast<double>(g_stat_small31_cycles) / total, static_cast<unsigned long long>(g_stat_small31_roots),
                     100.0 * static_cast<double>(g_stat_small15_cycles) / total, static_cast<unsigned long long>(g_stat_small15_roots));
    }
    if (out_count != nullptr) {
        u128 v = state.result;
        std::string s;
        do { s.insert(s.begin(), static_cast<char>('0' + static_cast<int>(v % 10))); v /= 10; } while (v != 0);
        *out_count = s;
    }
    if (g_debug_stats && count_mode)
        std::fprintf(stderr, "cache-stats: region splits=%llu region searches=%llu\n",
                     static_cast<unsigned long long>(state.regions_split), static_cast<unsigned long long>(state.region_calls));
    if (g_debug_stats && count_mode && !g_no_state_cache)
    {
        std::fprintf(stderr, "cache-stats: state lookups=%llu hits=%llu evictions=%llu entries=%zu of %zu slots\n",
                     static_cast<unsigned long long>(state.state_lookups), static_cast<unsigned long long>(state.state_hits),
                     static_cast<unsigned long long>(state.state_evictions), state.state_cache.size(), state.state_cache.slots());
        std::fprintf(stderr, "cache-stats: state cache by unknown cells (2^k..): lookups/hits/Mcycles-saved/times-gated");
        for (int k = 0; k < 24; ++k) {
            if (g_stat_sc_lookups[k] == 0) continue;
            std::fprintf(stderr, " %d:%llu/%llu/%llu/%llu", k, static_cast<unsigned long long>(g_stat_sc_lookups[k]),
                         static_cast<unsigned long long>(g_stat_sc_hits[k]), static_cast<unsigned long long>(g_stat_sc_saved[k] >> 20),
                         static_cast<unsigned long long>(g_stat_sc_gated[k]));
        }
        std::fprintf(stderr, "\n");
    }

    // Once per run: not again for learn_check's own enumeration.
    if (g_debug_stats && !g_learn_check_inner) {
        std::fprintf(stderr, "cache-stats: lookups=%llu misses=%llu probes=%llu\ncache-stats: deductions-per-entry histogram:",
                     static_cast<unsigned long long>(g_stat_lookups),
                     static_cast<unsigned long long>(g_stat_misses),
                     static_cast<unsigned long long>(g_stat_probes));
        for (int i = 0; i <= kMaxFastCells; ++i) {
            if (g_stat_ded_hist[i]) std::fprintf(stderr, " %d:%llu", i, static_cast<unsigned long long>(g_stat_ded_hist[i]));
        }
        std::fprintf(stderr, "\ncache-stats: lookup outcomes unsat=%llu no-deductions=%llu deductions=%llu; probes ok=%llu; branch-nodes=%llu\ncache-stats: lookups-per-probe histogram:",
                     static_cast<unsigned long long>(g_stat_unsat), static_cast<unsigned long long>(g_stat_noded),
                     static_cast<unsigned long long>(g_stat_ded), static_cast<unsigned long long>(g_stat_probe_ok),
                     static_cast<unsigned long long>(g_stat_nodes));
        for (int i = 0; i < 64; ++i) {
            if (g_stat_probe_lookups_hist[i]) std::fprintf(stderr, " %d:%llu", i, static_cast<unsigned long long>(g_stat_probe_lookups_hist[i]));
        }
        std::fprintf(stderr, "\ncache-stats: probe-pairs=%llu with-contradiction=%llu probes-skipped=%llu\n",
                     static_cast<unsigned long long>(g_stat_probe_pairs), static_cast<unsigned long long>(g_stat_probe_hits),
                     static_cast<unsigned long long>(g_stat_probe_skips));
        std::fprintf(stderr, "cache-stats: solutions_found=%d skip_probing=%d\n", state.solutions_found, static_cast<int>(state.skip_probing));
        if (g_learn_mode)
            std::fprintf(stderr, "cache-stats: learn conflicts=%llu clauses=%llu avg-len=%.1f forced-by-clauses=%llu clause-conflicts=%llu deleted=%llu probe-conflicts-learned=%llu probe-uip-elsewhere=%llu\n",
                         static_cast<unsigned long long>(g_stat_conflicts), static_cast<unsigned long long>(g_stat_learnt),
                         g_stat_learnt ? static_cast<double>(g_stat_learnt_lits) / static_cast<double>(g_stat_learnt) : 0.0,
                         static_cast<unsigned long long>(g_stat_clause_forced), static_cast<unsigned long long>(g_stat_clause_conflicts),
                         static_cast<unsigned long long>(g_stat_learnt_deleted), static_cast<unsigned long long>(g_stat_probe_learnt),
                         static_cast<unsigned long long>(g_stat_probe_uip_elsewhere));
        std::fprintf(stderr, "cache-stats: probe memo checked=%llu answered=%llu (PROBE_MEMO=1)\n",
                     static_cast<unsigned long long>(g_stat_memo_checked), static_cast<unsigned long long>(g_stat_memo_valid));
        std::fprintf(stderr, "cache-stats: latched subtrees dead[log2 size:count]:");
        for (int i = 0; i < 40; ++i) if (g_stat_dead_hist[i]) std::fprintf(stderr, " %d:%llu", i, static_cast<unsigned long long>(g_stat_dead_hist[i]));
        std::fprintf(stderr, "\ncache-stats: latched subtrees live[log2 size:count]:");
        for (int i = 0; i < 40; ++i) if (g_stat_live_hist[i]) std::fprintf(stderr, " %d:%llu", i, static_cast<unsigned long long>(g_stat_live_hist[i]));
        std::fprintf(stderr, "\n");
        if (g_debug_impl) std::fprintf(stderr, "impl-stats: nodes=%llu nodes-with-forced=%llu forced-cells=%llu dead=%llu edges=%llu\n",
            static_cast<unsigned long long>(g_stat_impl_nodes), static_cast<unsigned long long>(g_stat_impl_nodes_forced),
            static_cast<unsigned long long>(g_stat_impl_forced), static_cast<unsigned long long>(g_stat_impl_dead),
            static_cast<unsigned long long>(g_stat_impl_edges));
    }

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
    if (g_learn_check && g_learn_mode && !g_learn_check_inner) learn_check(rows, cols);
}

double explored_fraction() { return g_explored_mass; }
void set_progress_interval(double seconds) {
    g_progress_interval = seconds;
    g_progress_start = std::chrono::steady_clock::now();
    g_progress_next = g_progress_start + std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(seconds));
}
void request_stop() { g_stop_requested = 1; }
bool stop_requested() { return g_stop_requested != 0; }
const std::string& stop_position() { return g_stop_path; }
void print_count_progress(double elapsed) { print_progress_line(elapsed); }

double estimate_solutions(const std::vector<std::vector<int>>& rows,
                          const std::vector<std::vector<int>>& cols,
                          long n_dives,
                          unsigned long seed) {
    std::size_t budget = 1024ULL * 1024ULL * 1024ULL;
    const char* env = std::getenv("LINE_CACHE_BUDGET_MB");
    if (env != nullptr) {
        try { budget = static_cast<std::size_t>(std::stoull(env)) * 1024ULL * 1024ULL; }
        catch (...) {}
    }
    reset_line_cache(static_cast<int>(rows.size()), static_cast<int>(cols.size()), budget, false);

    const int H = static_cast<int>(rows.size());
    const int W = static_cast<int>(cols.size());

    // Dedupe specs by clue (same as solve()).
    std::vector<LineSpec> spec_pool;
    spec_pool.reserve(rows.size() + cols.size());
    std::map<std::vector<int>, const LineSpec*> clue_to_spec;
    auto get_spec = [&](const std::vector<int>& clue) -> const LineSpec* {
        auto it = clue_to_spec.find(clue);
        if (it != clue_to_spec.end()) return it->second;
        spec_pool.push_back(make_line_spec(clue));
        spec_pool.back().id = static_cast<int>(spec_pool.size()) - 1;
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
    g_row_tags.resize(mapped_rows.size());
    for (std::size_t i = 0; i < mapped_rows.size(); ++i)
        g_row_tags[i] = static_cast<std::uint16_t>(mapped_rows[i]->id * 2);
    g_col_tags.resize(mapped_cols.size());
    for (std::size_t i = 0; i < mapped_cols.size(); ++i)
        g_col_tags[i] = static_cast<std::uint16_t>(mapped_cols[i]->id * 2 + g_tag_col_bit);

    std::mt19937_64 rng(seed);
    double sum = 0.0;
    long hits = 0;
    double maxw = 0.0;
    for (long d = 0; d < n_dives; ++d) {
        Picture pic(H, W);
        Trail trail;
        trail.W = W;  // learning's propagation pushes onto it (see Trail::push)
        if (g_learn_mode) trail.cell_pos.assign(static_cast<std::size_t>(H) * static_cast<std::size_t>(W), UINT32_MAX);
        double est = estimate_dive(mapped_rows, mapped_cols, pic, trail, rng);
        sum += est;
        if (est > 0.0) { ++hits; if (est > maxw) maxw = est; }
    }
    double mean = (n_dives > 0) ? sum / static_cast<double>(n_dives) : 0.0;
    std::fprintf(stderr,
                 "estimate: dives=%ld solution-hits=%ld (%.3f%%) max-dive-weight=%.3e\n",
                 n_dives, hits,
                 n_dives ? 100.0 * static_cast<double>(hits) / static_cast<double>(n_dives) : 0.0,
                 maxw);
    return mean;
}
