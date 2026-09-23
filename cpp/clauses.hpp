#pragma once
#include <cstddef>
#include <vector>

// Clauses over cell literals, with two-watched-literal unit propagation.
//
// Literal: cell * 2 + (value == FULL). neg(l) = l ^ 1. A literal is TRUE when
// the cell is known with that value, FALSE when known with the other, else
// unassigned. A clause is a disjunction of literals over distinct cells.
//
// The store keeps no assignment of its own: the caller's `assigned` callback
// reads it and `force` writes it. Undo is the caller's trail revert, and the
// store needs nothing on a revert. A clause watches lits[0] and lits[1] of its
// arena copy; after a propagate call returns -1, every clause has two watches
// that are not false, or a false watch and a true one set in the same
// propagation round or earlier (a round: the cells a node's fixpoint loop sets
// between two revert points, all of which reach propagate before the next
// revert point). Reverting to a round boundary unassigns the true watch only
// together with the false one, so the invariant survives with no undo. After a
// conflict the caller must revert the round: the queue is left half visited.
class ClauseStore {
public:
    void init(int n_cells, std::size_t max_clauses);

    // Returns the clause id, -1 if refused (n < 1). lits[0] and lits[1] become
    // the watches; the store keeps its own copy of the literals.
    //
    // add never propagates. A clause of n >= 2 is only examined when one of
    // its two watches becomes false, so it should be added with lits[0] and
    // lits[1] not false, or with lits[1] the false literal that a revert
    // unassigns first; a clause added with both watches false is still sound
    // (it never forces or conflicts wrongly) but misses propagation until a
    // watch is assigned again. Asserting a fresh clause is the caller's job.
    //
    // A unit clause (n == 1) is the exception: it has no second literal to
    // watch, so it is kept on a list that every propagate call checks first,
    // whatever its `newly_true` list: a free unit literal is forced (and what
    // that forces propagates in the same call), a false one is a conflict.
    // So a fresh unit clause takes effect on the next propagate call without
    // the caller asserting it, and again after every revert that unassigns it.
    int add(const int* lits, int n, int lbd);

    int size() const { return live_; }   // live clauses

    // `assigned(l)` tells a literal's state (1 true, -1 false, 0 free);
    // `force(l, clause_id)` must make literal l true (set the cell, mark lines
    // dirty, push the trail entry with reason R_CLAUSE id) and return false if
    // the cell was already known with the other value. Neither may modify the
    // store. Visits, in order, the literals of `newly_true` (the trail's cells
    // since the last call, as literals), then every literal forced during the
    // call, in the order forced, so the clauses reach their fixpoint within
    // one call. Returns the conflicting clause id (at once, leaving the rest
    // unvisited: the caller reverts), or -1.
    template <class Assigned, class Force>
    int propagate(const int* newly_true, int n_new, Assigned assigned, Force force);

    const int* lits(int id, int* n) const {
        *n = arena_[start_[id]];
        return &arena_[start_[id] + 1];
    }
    int lbd(int id) const { return lbd_[id]; }
    void bump(int id) { act_[id] += 1.0f; }            // activity for deletion
    void lock(int id, bool locked) { locked_[id] = locked; }   // a reason on a trail is never deleted

    // When size() > max_clauses: delete half of the unlocked clauses, or more
    // if that is what it takes to get back to max_clauses, choosing the
    // highest lbd first (ties: lowest activity, then highest id). Survivors
    // keep their ids and literal order; dead ids are reused by add. Watches
    // are rebuilt for the survivors, so call it between propagations only.
    void reduce();

private:
    template <class Assigned, class Force>
    int visit_false(int f, Assigned& assigned, Force& force);

    std::vector<int> arena_;                 // clauses as [n, lit0, lit1, ...]
    std::vector<std::size_t> start_;         // offset of clause id in arena_
    std::vector<std::vector<int>> watches_;  // per literal: clauses watching it
    std::vector<int> units_;                 // ids of live unit clauses
    std::vector<int> lbd_;
    std::vector<float> act_;
    std::vector<char> locked_, dead_;
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
