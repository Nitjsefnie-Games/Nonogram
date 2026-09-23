#include "clauses.hpp"
#include <algorithm>
#include <cassert>
#include <functional>

void ClauseStore::init(int n_cells, std::size_t max_clauses) {
    arena_.clear();
    start_.clear();
    watches_.assign(static_cast<std::size_t>(n_cells) * 2, std::vector<int>());
    units_.clear();
    lbd_.clear();
    act_.clear();
    locked_.clear();
    dead_.clear();
    free_.clear();
    queue_.clear();
    max_clauses_ = max_clauses;
    live_ = 0;
}

int ClauseStore::add(const int* lits, int n, int lbd) {
    if (n < 1) return -1;
#ifndef NDEBUG
    for (int a = 0; a < n; ++a)
        for (int b = a + 1; b < n; ++b) assert((lits[a] >> 1) != (lits[b] >> 1) && "clause cells must be distinct");
#endif
    int id;
    if (!free_.empty()) {
        id = free_.back();
        free_.pop_back();
    } else {
        id = static_cast<int>(start_.size());
        start_.push_back(0);
        lbd_.push_back(0);
        act_.push_back(0.0f);
        locked_.push_back(0);
        dead_.push_back(0);
    }
    start_[id] = arena_.size();
    arena_.push_back(n);
    arena_.insert(arena_.end(), lits, lits + n);
    lbd_[id] = lbd;
    act_[id] = 0.0f;
    locked_[id] = 0;
    dead_[id] = 0;
    if (n == 1) {
        units_.push_back(id);
    } else {
        watches_[lits[0]].push_back(id);
        watches_[lits[1]].push_back(id);
    }
    ++live_;
    return id;
}

void ClauseStore::reduce() {
    if (static_cast<std::size_t>(live_) <= max_clauses_) return;
    const int n_ids = static_cast<int>(start_.size());
    std::vector<int> cand;
    for (int id = 0; id < n_ids; ++id)
        if (!dead_[id] && !locked_[id]) cand.push_back(id);
    // Half of the unlocked clauses, or as many as it takes to get back to
    // max_clauses when the locked ones are a large share of the store.
    const std::size_t excess = static_cast<std::size_t>(live_) - max_clauses_;
    const std::size_t n_del = std::min(cand.size(), std::max(cand.size() / 2, excess));
    if (n_del == 0) return;
    // Worst first: highest lbd, then lowest activity, then highest id (a total
    // order, so the deleted set does not depend on the partition algorithm).
    auto worse = [this](int a, int b) {
        if (lbd_[a] != lbd_[b]) return lbd_[a] > lbd_[b];
        if (act_[a] != act_[b]) return act_[a] < act_[b];
        return a > b;
    };
    std::nth_element(cand.begin(), cand.begin() + (n_del - 1), cand.end(), worse);
    for (std::size_t i = 0; i < n_del; ++i) {
        dead_[cand[i]] = 1;
        free_.push_back(cand[i]);
    }
    live_ -= static_cast<int>(n_del);
    // add pops from the back: reuse the lowest ids first.
    std::sort(free_.begin(), free_.end(), std::greater<int>());

    // Compact the arena (survivors keep their ids and literal order) and
    // rebuild the watches: each survivor still watches its lits[0] and lits[1].
    std::vector<int> arena;
    arena.reserve(arena_.size());
    for (auto& w : watches_) w.clear();
    units_.clear();
    for (int id = 0; id < n_ids; ++id) {
        if (dead_[id]) { start_[id] = 0; continue; }
        const int* c = &arena_[start_[id]];
        start_[id] = arena.size();
        arena.insert(arena.end(), c, c + 1 + c[0]);
        if (c[0] == 1) {
            units_.push_back(id);
        } else {
            watches_[c[1]].push_back(id);
            watches_[c[2]].push_back(id);
        }
    }
    arena_.swap(arena);
}
