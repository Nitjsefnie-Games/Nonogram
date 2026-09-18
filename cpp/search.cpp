#include "search.hpp"

#include "lines.hpp"
#include "picture.hpp"
#include "types.hpp"

// Vendored third-party hash map: ankerl::unordered_dense (MIT, v4.8.1).
// See cpp/external/ankerl/unordered_dense.h for license/copyright header.
// Used for the line-batch memoization cache in this file.
#include "external/ankerl/unordered_dense.h"

#include <sys/mman.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <climits>
#include <cmath>
#include <cstring>
#include <deque>
#include <functional>
#include <list>
#include <map>
#include <random>
#include <string>
#include <string_view>
#include <type_traits>
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
//                   hash fingerprint (kEmptyTag marks a free slot). Many
//                   entries share a spec, so the id alone let most same-spec
//                   probe neighbours through to the key compare; the
//                   fingerprint rejects 63/64 of them on the tag word.
//   [8kw+2]         deduction count
//   [8kw+3]         flags: bit0 = line satisfiable, bit1 = deductions spilled
//   [8kw+4, S)      deductions inline, or (spilled) uint32 offset into arena_
// ---------------------------------------------------------------------------

constexpr int kMaxFastCells = 128;   // 7-bit positions in the deduction bytes
constexpr std::uint16_t kEmptyTag = 0xFFFF;
constexpr std::uint8_t kFlagSat = 1;
constexpr std::uint8_t kFlagSpilled = 2;
constexpr std::size_t kHugePage = 2u << 20;

// Page-aligned, hugepage-advised byte buffer for the slot table.
struct SlotBuf {
    unsigned char* p = nullptr;
    std::size_t bytes = 0;
    SlotBuf() = default;
    SlotBuf(const SlotBuf&) = delete;
    SlotBuf& operator=(const SlotBuf&) = delete;
    ~SlotBuf() { release(); }
    void release() {
        if (p != nullptr) std::free(p);
        p = nullptr;
        bytes = 0;
    }
    void allocate(std::size_t n) {
        release();
        void* mem = nullptr;
        // Hugepage alignment only once the table is hugepage-sized: a 2 MB
        // alignment on a small table faults in a whole zeroed hugepage, which
        // showed up as ~0.3 ms of startup on puzzles that solve in 0.1 ms.
        const std::size_t align = (n >= kHugePage) ? kHugePage : 64;
        if (posix_memalign(&mem, align, n) != 0) throw std::bad_alloc();
        p = static_cast<unsigned char*>(mem);
        bytes = n;
        if (n >= kHugePage) madvise(p, n, MADV_HUGEPAGE);  // advisory; failure is harmless
    }
    void swap(SlotBuf& o) { std::swap(p, o.p); std::swap(bytes, o.bytes); }
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
// FIRST_VAL=1 explores the branch value whose probe settled FEWER cells
// first (the shipped order explores the one that settled more).
const bool g_first_val_low = std::getenv("FIRST_VAL") != nullptr;
// NO_SKIP=1 never latches the adaptive probing shut-off in default mode.
const bool g_no_skip = std::getenv("NO_SKIP") != nullptr;
// PROBE_WINDOW / PROBE_THRESH override the shut-off window and yield threshold.
const std::size_t g_probe_window = std::getenv("PROBE_WINDOW") ? std::strtoull(std::getenv("PROBE_WINDOW"), nullptr, 10) : 100;
const double g_probe_thresh = std::getenv("PROBE_THRESH") ? std::atof(std::getenv("PROBE_THRESH")) : 0.01;
// DEAD_WINDOW / DEAD_FRAC override the dead-work watchdog's window and fraction.
const std::uint64_t g_dead_window = std::getenv("DEAD_WINDOW") ? std::strtoull(std::getenv("DEAD_WINDOW"), nullptr, 10) : 4096;
const double g_dead_frac = std::getenv("DEAD_FRAC") ? std::atof(std::getenv("DEAD_FRAC")) : 0.9;
#else
constexpr std::uint64_t g_dead_window = 4096;
constexpr double g_dead_frac = 0.9;
constexpr int g_debug_impl = 0;
constexpr int g_anytime_tb = 0;
constexpr bool g_first_val_low = false;
constexpr bool g_no_skip = false;
constexpr std::size_t g_probe_window = 100;
constexpr double g_probe_thresh = 0.01;
#endif
std::uint64_t g_stat_probe_pairs = 0, g_stat_probe_hits = 0;  // probe pairs, and those with a contradiction
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

    // Returns the slot holding (key, tag), or nullptr.
    static std::uint16_t full_tag(std::uint16_t tag, std::uint64_t h) {
        return static_cast<std::uint16_t>(tag | ((h >> 58) << 10));
    }

    // On a miss, remembers the empty slot it stopped at so the insert that
    // follows need not hash and walk the probe sequence a second time.
    unsigned char* find(const std::uint64_t* key, std::uint16_t tag) {
        const std::uint64_t h = hash(key, tag);
        std::size_t idx = h & mask_;
        tag = full_tag(tag, h);
        for (;;) {
            unsigned char* s = slot(idx);
            const std::uint16_t t = load_u16(s + hdr_);
            if (t == kEmptyTag) { miss_slot_ = s; miss_tag_ = tag; return nullptr; }
            if (t == tag && keys_equal(s, key)) return s;
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

private:
    static constexpr std::size_t kMinSlots = 1u << 12;  // 128 KB at 32-byte slots

    std::uint64_t hash(const std::uint64_t* key, std::uint16_t tag) const {
        namespace wy = ankerl::unordered_dense::detail::wyhash;
        // One 64x64->128 multiply-fold covers the common <= 2-word key; the
        // tag is folded in by a cheap multiply so it perturbs every bit.
        const std::uint64_t t = static_cast<std::uint64_t>(tag) * 0x9E3779B97F4A7C15ULL;
        if (kw_ <= 2) {
            const std::uint64_t k1 = (kw_ == 2) ? key[1] : 0;
            return wy::mix(key[0] ^ t, k1 ^ 0xE7037ED1A0B428DBULL);
        }
        std::uint64_t h = wy::mix(t, 0xA0761D6478BD642FULL);
        for (int w = 0; w < kw_; ++w) h = wy::mix(h ^ key[w], 0xE7037ED1A0B428DBULL);
        return h;
    }

    bool keys_equal(const unsigned char* s, const std::uint64_t* key) const {
        if (kw_ == 2) {  // the common case: no loop with a runtime trip count
            std::uint64_t a, b;
            std::memcpy(&a, s, 8);
            std::memcpy(&b, s + 8, 8);
            return ((a ^ key[0]) | (b ^ key[1])) == 0;
        }
        std::uint64_t diff = 0;
        for (int w = 0; w < kw_; ++w) {
            std::uint64_t v;
            std::memcpy(&v, s + 8 * static_cast<std::size_t>(w), 8);
            diff |= v ^ key[w];
        }
        return diff == 0;
    }

    unsigned char* slot(std::size_t idx) const { return buf_.p + (idx << slot_shift_); }

    unsigned char* free_slot(std::uint64_t h) {
        std::size_t idx = h & mask_;
        while (load_u16(slot(idx) + hdr_) != kEmptyTag) idx = (idx + 1) & mask_;
        return slot(idx);
    }

    void alloc(std::size_t n) {
        buf_.allocate(n << slot_shift_);
        std::memset(buf_.p, 0xFF, buf_.bytes);
        nslots_ = n;
        mask_ = n - 1;
        count_ = 0;
    }

    void clear_slots() {
        std::memset(buf_.p, 0xFF, buf_.bytes);
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
            if (t == kEmptyTag) continue;
            std::uint64_t key[kMaxFastCells / 32];
            std::memcpy(key, s, hdr_);
            std::memcpy(free_slot(hash(key, static_cast<std::uint16_t>(t & 0x3FF))), s, slot_bytes);
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
// probe_cell uses a private local Trail because its mutations are reverted
// before returning to its caller (solve_backtrack), so they don't need to
// commingle with the outer trail.
// ---------------------------------------------------------------------------

struct Trail {
    std::vector<int> changed_cell_indices;  // packed (row, col); see trail_pack
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
};

// Trail entries carry (row, col) rather than a linear index so a revert can
// update the packed line keys without a division to recover the row.
inline int trail_pack(int row, int col) { return (row << 16) | col; }
inline int trail_row(int e) { return e >> 16; }
inline int trail_col(int e) { return e & 0xFFFF; }

// ---------------------------------------------------------------------------
// SolveState — mirrors picture.py SolveState (without print_state).
// ---------------------------------------------------------------------------

struct SolveState {

    int probing_min_solutions;

    int solutions_found = 0;
    bool used_contradiction = false;
    bool used_backtrack = false;
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
        }
        latched_dead = latched_live = 0;
    }
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
        if (g_debug_stats) { ++g_stat_probe_pairs; g_stat_probe_hits += found_contradiction; }
        // In anytime mode skip_probing can never latch, so the whole yield
        // window is dead work on every probe -- skip it entirely.
        if (keep_probing || skip_probing || probing_pinned || g_no_skip) return;
        // Mirror picture.py: don't let yield-window disable probing until
        // we've already found multiple solutions.
        if (solutions_found < probing_min_solutions) return;
        probe_outcomes.push_back(found_contradiction ? 1 : 0);
        if (probe_outcomes.size() > g_probe_window) {
            probe_outcomes.pop_front();
        }
        if (probe_outcomes.size() == g_probe_window) {
            int sum = 0;
            for (int v : probe_outcomes) sum += v;
            double yield_rate = static_cast<double>(sum) / static_cast<double>(g_probe_window);
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

template <bool FAST>
inline BatchResult solve_one_batch(const std::vector<const LineSpec*>& mapped,
                                   int index,
                                   bool is_col,
                                   Picture& pic) {
    if (!FAST) {
        return solve_one_batch_legacy(*mapped[index], index, is_col, pic);
    }

    const int kw = pic.key_words;
    const std::uint64_t* key =
        (is_col ? pic.col_keys.data() : pic.row_keys.data()) + static_cast<std::size_t>(index) * kw;
    const std::uint16_t tag = (is_col ? g_col_tags : g_row_tags)[static_cast<std::size_t>(index)];

    if (g_debug_stats) ++g_stat_lookups;
    unsigned char* s = g_fast_cache.find(key, tag);
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
// Every pixel transition UNKNOWN -> value is recorded into trail (linear cell
// index r*W + c). On revert, the caller truncates the trail and walks each
// recorded index back to UNKNOWN.
// ---------------------------------------------------------------------------

template <typename Iter, typename Pos, typename Val>
inline void write_intersection_impl(Iter first, Iter last, Pos pos_of, Val val_of,
                                    int line_index, Picture& pic, bool is_row, Trail& trail) {
    // No UNKNOWN re-check: the cache entry was computed for exactly the
    // line's current content (that is its key), and the line DP only emits
    // deductions for cells that are UNKNOWN in that content.
    if (is_row) {
        const int row = line_index;
        for (; first != last; ++first) {
            const int col = pos_of(*first);
            pic.set_known(row, col, val_of(*first));
            pic.mark_col_dirty(col);
            trail.changed_cell_indices.push_back(trail_pack(row, col));
        }
    } else {
        const int col = line_index;
        for (; first != last; ++first) {
            const int row = pos_of(*first);
            pic.set_known(row, col, val_of(*first));
            pic.mark_row_dirty(row);
            trail.changed_cell_indices.push_back(trail_pack(row, col));
        }
    }
}

void write_intersection(const BatchResult& r, int line_index, Picture& pic, bool is_row, Trail& trail) {
    if (r.n8 != BatchResult::kLegacy) {
        const std::uint8_t* d = static_cast<const std::uint8_t*>(r.ded);
        write_intersection_impl(d, d + r.n8,
                                [](std::uint8_t b) { return static_cast<int>(b & 0x7F); },
                                [](std::uint8_t b) { return static_cast<std::int8_t>(b >> 7); },
                                line_index, pic, is_row, trail);
    } else {
        const auto* v = static_cast<const std::vector<int>*>(r.ded);
        write_intersection_impl(v->begin(), v->end(),
                                [](int enc) { return deduce_pos(enc); },
                                [](int enc) { return deduce_val(enc); },
                                line_index, pic, is_row, trail);
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
    // The fast/legacy choice is per puzzle; decide it once per drain rather
    // than per line.
    auto drain = [&](auto fast) -> bool {
        while (!queue.empty()) {
            const int index = queue.front();
            queue.pop_front();
            dirty[index] = 0;
            BatchResult r = solve_one_batch<decltype(fast)::value>(mapped, index, !is_row, pic);
            if (!r.success) {
                return false;
            }
            if (r.n8 > 0 || (r.n8 == BatchResult::kLegacy && r.ded != nullptr)) {
                write_intersection(r, index, pic, is_row, trail);
            }
        }
        queue.reset();
        return true;
    };
    return g_fast_mode ? drain(std::true_type{}) : drain(std::false_type{});
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
        for (auto it = trail.changed_cell_indices.rbegin();
             it != trail.changed_cell_indices.rend(); ++it) {
            pic.unset(trail_row(*it), trail_col(*it));
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
    if (g_debug_stats) { ++g_stat_probes; g_stat_probe_cur = 0; }
    struct ProbeStat { ~ProbeStat() { if (g_debug_stats) ++g_stat_probe_lookups_hist[std::min<std::uint64_t>(g_stat_probe_cur, 63)]; } } probe_stat;

    ProbeGuard guard(pic, trail);

    // Apply the probe pixel (record on trail).
    pic.set_known(row, col, val);
    trail.changed_cell_indices.push_back(trail_pack(row, col));

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

    if (g_debug_stats) ++g_stat_probe_ok;
    if (g_debug_impl) {
        g_impl_off.push_back(static_cast<int>(g_impl_rec.size()));
        const int W = pic.width();
        for (int e : trail.changed_cell_indices) {
            const int v = pic.pixels[trail_row(e) * W + trail_col(e)] == FULL ? 1 : 0;
            g_impl_rec.push_back((e << 1) | v);
        }
    }
    return ProbeResult{true, count_solved_pixels(pic)};
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
void revert_branch(Picture& pic,
                   Trail& trail,
                   std::size_t mark,
                   int saved_unknown_count) {
    while (trail.changed_cell_indices.size() > mark) {
        const int e = trail.changed_cell_indices.back();
        // Entries below `counted` were subtracted from the per-line counts
        // at a branch node; add them back. Entries above it never were.
        if (trail.changed_cell_indices.size() <= trail.counted) {
            ++trail.row_unknown[trail_row(e)];
            ++trail.col_unknown[trail_col(e)];
        }
        trail.changed_cell_indices.pop_back();
        pic.unset(trail_row(e), trail_col(e));
    }
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

    // Per-row / per-col UNKNOWN counts, used as the "most constrained line"
    // signal in the composite sort key (lower => more constrained line).
    // Brought up to date from the trail entries pushed since the previous
    // branch node (see Trail); recounting them here by popcounting every
    // line key was 27% of the latched node on hard/6689.
    const int kw = pic.key_words;
    const std::uint64_t* rk = pic.row_keys.data();
    const int n_unknown = pic.unknown_count;
    {
        const int* tr = trail.changed_cell_indices.data();
        const std::size_t n = trail.changed_cell_indices.size();
        int* ru = trail.row_unknown.data();
        int* cu = trail.col_unknown.data();
        for (std::size_t i = trail.counted; i < n; ++i) {
            --ru[trail_row(tr[i])];
            --cu[trail_col(tr[i])];
        }
        trail.counted = n;
    }
    const int* uir = trail.row_unknown.data();
    const int* uic = trail.col_unknown.data();

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
        int line_constraint = uir[r] + uic[c];
        return std::pair<int, int>(line_constraint, -neighbor_score(r, c));
    };

    if (state.skip_probing) {
        // Pick the unknown cell with the smallest composite key (most-
        // constrained line; tie-broken by highest neighbor score). Single
        // row-major scan, no coords/scores arrays materialized.
        std::pair<int, int> best_key;
        bool have = false;
        // Lower bound on any cell's line_constraint in row r is
        // uir[r] + min_uic, so once a best key is known, rows whose bound
        // exceeds its line_constraint cannot beat OR tie it and are skipped
        // without touching their cells; rows that could tie are scanned in
        // the same row-major order as before, so the choice is identical.
        int min_uic = INT_MAX;
        for (int c = 0; c < W; ++c) if (uic[c] > 0 && uic[c] < min_uic) min_uic = uic[c];
        for (int r = 0; r < H; ++r) {
            if (uir[r] == 0 || (have && uir[r] + min_uic > best_key.first)) continue;
            for (int w = 0; w < kw; ++w) {
                std::uint64_t m = rk[r * kw + w] & kUnknownBits;
                while (m != 0) {
                    const int c = 32 * w + (__builtin_ctzll(m) >> 1);
                    m &= m - 1;
                    std::pair<int, int> k = composite_key(r, c);
                    if (!have || k < best_key) {
                        best_key = k;
                        best_row = r;
                        best_col = c;
                        have = true;
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

        for (int idx : order) {
            int row = unknown_coords[idx].first;
            int col = unknown_coords[idx].second;
            if (pxs[row * W + col] != UNKNOWN) continue;  // settled by an earlier commit this pass

            ProbeResult full_res = probe_cell(row, col, FULL, mapped_rows, mapped_cols, pic);
            ProbeResult empty_res = probe_cell(row, col, EMPTY, mapped_rows, mapped_cols, pic);

            state.record_probe((!full_res.ok) || (!empty_res.ok));

            if (!full_res.ok && !empty_res.ok) {
                return true; // dead branch
            }

            if (full_res.ok != empty_res.ok) {
                state.mark_contradiction();
                const std::int8_t forced = full_res.ok ? FULL : EMPTY;
                // Forced commit: record the pixel on the trail so a parent
                // backtrack frame can revert it if its own branch fails, then
                // propagate in place and keep probing the remaining cells.
                pic.set_known(row, col, forced);
                trail.changed_cell_indices.push_back(trail_pack(row, col));
                pic.mark_row_dirty(row);
                pic.mark_col_dirty(col);
                while (pic.has_dirty()) {
                    if (!solve_lines(mapped_rows, pic, true, trail)) return true;  // dead branch
                    if (!solve_lines(mapped_cols, pic, false, trail)) return true;
                }
                if (pic.is_solved()) {
                    return solve_real(mapped_rows, mapped_cols, pic, state, on_solution, trail);
                }
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
            if (state.keep_probing) {
                better = hi > best_pixels;
                if (g_anytime_tb == 1) better = better || (hi == best_pixels && lo > best_lo);
                if (g_anytime_tb == 2) better = better || (hi == best_pixels && lo < best_lo);
            } else if (g_branch_k_set) {
                const double sc = g_branch_k * lo - hi;
                better = sc > best_score;
                if (better) best_score = sc;
            } else {
                better = lo > best_pixels || (lo == best_pixels && hi < best_other);
            }
            if (better) {
                best_pixels = state.keep_probing ? hi : lo;
                best_other = hi;
                best_lo = lo;
                best_row = row;
                best_col = col;
                best_first_val = ((f >= e) != g_first_val_low) ? FULL : EMPTY;
                have_best = true;
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
                        pic.set_known(r, c, fc.second);
                        trail.changed_cell_indices.push_back(fc.first);
                        pic.mark_row_dirty(r);
                        pic.mark_col_dirty(c);
                        while (pic.has_dirty()) {
                            if (!solve_lines(mapped_rows, pic, true, trail)) return true;
                            if (!solve_lines(mapped_cols, pic, false, trail)) return true;
                        }
                    }
                    return solve_real(mapped_rows, mapped_cols, pic, state, on_solution, trail);
                }
            }
        }
    }

    state.mark_backtrack();
    if (g_debug_stats) ++g_stat_nodes;
    const std::uint64_t nodes_before = g_stat_all_nodes;
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
    for (int branch = 0; branch < 2; ++branch) {
        const std::int8_t val = (branch == 0) ? first_val : second_val;

        const std::size_t mark = trail.changed_cell_indices.size();
        const int saved_unknown_count = pic.unknown_count;

        // Apply the branch pixel (record on trail).
        pic.set_known(row, col, val);
        trail.changed_cell_indices.push_back(trail_pack(row, col));
        pic.mark_row_dirty(row);
        pic.mark_col_dirty(col);

        if (!solve_real(mapped_rows, mapped_cols, pic, state, on_solution, trail)) {
            return false;
        }
        revert_branch(pic, trail, mark, saved_unknown_count);
    }

    if (latched_here) {
        const bool dead = state.solutions_found == sols_before;
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
        while (pic.has_dirty()) {
            if (!solve_lines(mapped_rows, pic, true, trail)) return 0.0;
            if (!solve_lines(mapped_cols, pic, false, trail)) return 0.0;
        }
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
        ProbeResult pf = probe_cell(br, bc, FULL, mapped_rows, mapped_cols, pic);
        ProbeResult pe = probe_cell(br, bc, EMPTY, mapped_rows, mapped_cols, pic);
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
    Trail trail;
    // Reserve enough headroom that the trail rarely reallocates.
    trail.changed_cell_indices.reserve(static_cast<std::size_t>(H) * static_cast<std::size_t>(W));
    trail.row_unknown.assign(static_cast<std::size_t>(H), W);
    trail.col_unknown.assign(static_cast<std::size_t>(W), H);
    trail.counted = 0;
    (void)solve_real(mapped_rows, mapped_cols, pic, state, on_solution, trail);

    if (g_debug_stats) {
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
        std::fprintf(stderr, "\ncache-stats: probe-pairs=%llu with-contradiction=%llu\n",
                     static_cast<unsigned long long>(g_stat_probe_pairs), static_cast<unsigned long long>(g_stat_probe_hits));
        std::fprintf(stderr, "cache-stats: solutions_found=%d skip_probing=%d\n", state.solutions_found, static_cast<int>(state.skip_probing));
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
}

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
