#pragma once
#include <algorithm>
#include <cstddef>
#include <vector>
#ifdef NONOGRAM_STATS
#include <cstdio>
#include <cstdlib>
#endif

// Clauses over cell literals, with two-watched-literal unit propagation.
//
// Literal: cell * 2 + (value == FULL). neg(l) = l ^ 1. A literal is TRUE when
// the cell is known with that value, FALSE when known with the other, else
// unassigned. A clause is a disjunction of literals over distinct cells.
//
// The store keeps no assignment of its own: the caller's `assigned` callback
// reads it and `force` writes it. Undo is the caller's trail revert, and the
// store needs nothing on a revert. A watched clause watches lits[0] and
// lits[1] of its arena copy; after a propagate call returns -1, every watched
// clause has two watches that are not false, or a false watch and a true one
// set in the same propagation round or earlier (a round: the cells a node's
// fixpoint loop sets between two revert points, all of which reach propagate
// before the next revert point). Reverting to a round boundary unassigns the
// true watch only together with the false one, so the invariant survives with
// no undo. After a conflict the caller must revert the round: the queue is
// left half visited.
//
// Known limit (phase 1, chronological backtracking): a clause first examined
// while it forces or is satisfied gets a true watch set in that round and a
// false watch that may be older. A revert that unassigns the true watch but
// not the false one leaves the clause unit (or all but one false) without
// being re-examined until one of its watches is assigned again. That is a
// missed propagation only, never a wrong force or a wrong conflict: the store
// forces a literal only when it has seen every other literal false.
class ClauseStore {
public:
    void init(int n_cells, std::size_t max_clauses);

    // Returns the clause id, -1 if refused (n < 1). The store keeps its own
    // copy of the literals. add never propagates and never reads the
    // assignment: the clause takes effect at the next propagate call.
    //
    // A clause of n >= 2 goes on the fresh list, which the next propagate
    // call examines right after the units (see propagate), and gets its
    // watches there. So a learnt clause needs no asserting by the caller, in
    // whatever state the assignment is when it is added.
    //
    // A unit clause (n == 1) has no second literal to watch, so it is kept on
    // a list that every propagate call checks first, whatever its
    // `newly_true` list: a free unit literal is forced (and what that forces
    // propagates in the same call), a false one is a conflict. So a unit
    // clause takes effect on the next propagate call, and again after every
    // revert that unassigns it.
    int add(const int* lits, int n, int lbd);

    int size() const { return live_; }   // live clauses
    std::size_t max_clauses() const { return max_clauses_; }
    // Clauses added since the last propagate call examined them.
    bool has_fresh() const { return !fresh_.empty(); }

    // `assigned(l)` tells a literal's state (1 true, -1 false, 0 free);
    // `force(l, clause_id)` must make literal l true (set the cell, mark lines
    // dirty, push the trail entry with reason R_CLAUSE id) and return false if
    // the cell was already known with the other value. Neither may modify the
    // store. Visits, in order, the literals of `newly_true` (the trail's cells
    // since the last call, as literals), then every literal forced during the
    // call, in the order forced, so the clauses reach their fixpoint within
    // one call. Before those: the unit clauses, then the fresh clauses, each
    // once: one with a true literal is satisfied (watches: that literal and
    // another, not false if there is one); one with exactly one free literal
    // and the rest false forces it (watches: the forced literal at lits[0],
    // a false literal at lits[1], the first false one in add's order, which
    // for a learnt clause is its highest-level literal); one with two free
    // literals watches them; each of those leaves the fresh list. One with
    // every literal false is the conflict and stays fresh (it has no watches,
    // so the next call examines it again). Returns the conflicting clause id
    // (at once, leaving the rest unvisited: the caller reverts), or -1.
    template <class Assigned, class Force>
    int propagate(const int* newly_true, int n_new, Assigned assigned, Force force);

    // The clause's literals, *n of them. lits[0] and lits[1] are the watches.
    // The order is add's order only until propagate first visits the clause:
    // a visit swaps literals inside the clause (the false watch to index 1, a
    // replacement watch from index >= 2 into index 1). A literal the clause
    // forced is at lits[0] and stays there for as long as it is true, so while
    // the clause is a reason on a trail lits[0] is the literal it forced and
    // lits[1..] are its antecedents' negations: a visit with lits[0] true
    // keeps the clause as it is, and only a revert of lits[0] ends that.
    // reduce keeps the order.
    const int* lits(int id, int* n) const {
#ifdef NONOGRAM_STATS
        if (dead_[id]) {
            std::fprintf(stderr, "stats check failed: lits() of dead clause %d (%s:%d)\n", id, __FILE__, __LINE__);
            std::abort();
        }
#endif
        *n = arena_[start_[id]];
        return &arena_[start_[id] + 1];
    }
    int lbd(int id) const { return lbd_[id]; }
    void bump(int id) { act_[id] += 1.0f; }            // activity for deletion
    void lock(int id, bool locked) { locked_[id] = locked; }   // a reason on a trail is never deleted
    // The store is never told about reverts, so the caller recomputes the
    // locks from its trail before each reduce: unlock_all, then lock every
    // clause that is still a reason.
    void unlock_all() { std::fill(locked_.begin(), locked_.end(), 0); }

    // When size() > max_clauses: delete half of the unlocked clauses, or more
    // if that is what it takes to get back to max_clauses, choosing the
    // highest lbd first (ties: lowest activity, then highest id). Survivors
    // keep their ids and literal order; dead ids are reused by add. Watches
    // are rebuilt for the survivors, so call it between propagations only.
    // Dead ids leave the fresh list; a fresh survivor stays fresh.
    void reduce();

private:
    template <class Assigned, class Force>
    int visit_false(int f, Assigned& assigned, Force& force);

    std::vector<int> arena_;                 // clauses as [n, lit0, lit1, ...]
    std::vector<std::size_t> start_;         // offset of clause id in arena_
    std::vector<std::vector<int>> watches_;  // per literal: clauses watching it
    std::vector<int> units_;                 // ids of live unit clauses
    std::vector<int> fresh_;                 // ids added since propagate last examined them (unwatched)
    std::vector<int> lbd_;
    std::vector<float> act_;
    std::vector<char> locked_, dead_, is_fresh_;
    std::vector<int> free_;                  // dead ids, reused by add
    std::vector<int> queue_;                 // literals forced in propagate
    std::size_t max_clauses_ = 0;
    int live_ = 0;
};

// The clauses watching `f`, which has just become false. Each is kept with the
// false watch at index 1; it keeps watching f if lits[0] is true, moves the
// watch to a later literal that is not false if there is one, and otherwise
// forces lits[0] or, when that is false too, is the conflict.
template <class Assigned, class Force>
int ClauseStore::visit_false(int f, Assigned& assigned, Force& force) {
    std::vector<int>& ws = watches_[f];
    std::size_t i = 0, j = 0;
    const std::size_t n = ws.size();
    while (i < n) {
        const int id = ws[i++];
        int* c = &arena_[start_[id]];
        const int sz = c[0];
        int* l = c + 1;
        if (l[0] == f) { l[0] = l[1]; l[1] = f; }
        const int other = l[0];
        const int other_state = assigned(other);
        if (other_state > 0) { ws[j++] = id; continue; }
        int k = 2;
        while (k < sz && assigned(l[k]) < 0) ++k;
        if (k < sz) {
            // f is false and l[k] is not, so watches_[l[k]] is not ws.
            l[1] = l[k];
            l[k] = f;
            watches_[l[1]].push_back(id);
            continue;
        }
        ws[j++] = id;
        if (other_state < 0 || !force(other, id)) {
            while (i < n) ws[j++] = ws[i++];
            ws.resize(j);
            return id;
        }
        queue_.push_back(other);
    }
    ws.resize(j);
    return -1;
}

template <class Assigned, class Force>
int ClauseStore::propagate(const int* newly_true, int n_new, Assigned assigned, Force force) {
    queue_.clear();
    for (const int id : units_) {
        const int l = arena_[start_[id] + 1];
        const int state = assigned(l);
        if (state > 0) continue;
        if (state < 0 || !force(l, id)) return id;
        queue_.push_back(l);
    }
    for (std::size_t i = 0; i < fresh_.size(); ++i) {
        const int id = fresh_[i];
        int* c = &arena_[start_[id]];
        const int sz = c[0];
        int* l = c + 1;
        // w0, w1: the positions to watch; forced: whether l[w0] is forced.
        int t = -1, f1 = -1, f2 = -1, fl = -1;
        for (int k = 0; k < sz; ++k) {
            const int st = assigned(l[k]);
            if (st > 0) { if (t < 0) t = k; }
            else if (st == 0) { if (f1 < 0) f1 = k; else if (f2 < 0) f2 = k; }
            else if (fl < 0) fl = k;
        }
        int w0, w1;
        bool forced = false;
        if (t >= 0) {
            w0 = t;
            w1 = f1 >= 0 ? f1 : (t == 0 ? 1 : 0);
        } else if (f2 >= 0) {
            w0 = f1;
            w1 = f2;
        } else if (f1 >= 0) {
            w0 = f1;
            w1 = fl;
            forced = true;
        } else {
            // Every literal false: the conflict. It and the unexamined rest
            // stay fresh; the examined ones before it are watched now.
            fresh_.erase(fresh_.begin(), fresh_.begin() + static_cast<std::ptrdiff_t>(i));
            return id;
        }
        std::swap(l[0], l[w0]);
        if (w1 == 0) w1 = w0;
        std::swap(l[1], l[w1]);
        if (forced) {
            if (!force(l[0], id)) {
                fresh_.erase(fresh_.begin(), fresh_.begin() + static_cast<std::ptrdiff_t>(i));
                return id;
            }
            queue_.push_back(l[0]);
        }
        watches_[l[0]].push_back(id);
        watches_[l[1]].push_back(id);
        is_fresh_[id] = 0;
    }
    fresh_.clear();
    for (int i = 0; i < n_new; ++i) {
        const int conflict = visit_false(newly_true[i] ^ 1, assigned, force);
        if (conflict >= 0) return conflict;
    }
    for (std::size_t q = 0; q < queue_.size(); ++q) {
        const int conflict = visit_false(queue_[q] ^ 1, assigned, force);
        if (conflict >= 0) return conflict;
    }
    return -1;
}
