#include "../clauses.hpp"
#include <cstddef>
#include <cstdio>
#include <utility>
#include <vector>

// Variables are cells. P(x): cell x FULL, N(x): cell x EMPTY (the store's
// literal encoding: cell * 2 + (value == FULL)).
static int P(int x) { return x * 2 + 1; }
static int N(int x) { return x * 2; }

// A minimal assignment for the store's callbacks: val[cell] is -1 unknown,
// 0 EMPTY, 1 FULL; every forced literal is appended to the trail with the
// clause that forced it. `log` holds every assigned cell in order (set by the
// test or forced), so revert_to can undo a suffix like the solver's trail.
struct Env {
    std::vector<int> val;
    std::vector<std::pair<int, int>> trail;   // (literal, reason clause id)
    std::vector<int> log;                     // assigned cells, in order
    explicit Env(int n) : val(n, -1) {}
    int assigned(int l) const {
        const int v = val[l >> 1];
        if (v < 0) return 0;
        return v == (l & 1) ? 1 : -1;
    }
    bool force(int l, int id) {
        const int v = val[l >> 1];
        if (v >= 0) return v == (l & 1);
        val[l >> 1] = l & 1;
        trail.push_back({l, id});
        log.push_back(l >> 1);
        return true;
    }
    void set(int l) { val[l >> 1] = l & 1; log.push_back(l >> 1); }
    void revert() { for (int& v : val) v = -1; trail.clear(); log.clear(); }
    std::size_t mark() const { return log.size(); }
    // Undo every cell assigned since `m` (a LIFO revert of later rounds).
    void revert_to(std::size_t m) {
        while (log.size() > m) { val[log.back()] = -1; log.pop_back(); }
        std::vector<std::pair<int, int>> kept;
        for (const auto& t : trail) if (val[t.first >> 1] >= 0) kept.push_back(t);
        trail.swap(kept);
    }
    int propagate(ClauseStore& s, const std::vector<int>& newly_true) {
        return s.propagate(newly_true.data(), static_cast<int>(newly_true.size()),
                           [this](int l) { return assigned(l); },
                           [this](int l, int id) { return force(l, id); });
    }
};

static int fail(const char* msg) { std::printf("FAIL %s\n", msg); return 1; }

static bool same_lits(const ClauseStore& s, int id, const std::vector<int>& want) {
    int n = 0;
    const int* l = s.lits(id, &n);
    if (n != static_cast<int>(want.size())) return false;
    for (int i = 0; i < n; ++i) if (l[i] != want[i]) return false;
    return true;
}

// The chain {1,2}, {~2,3}, {~3,4}: asserting ~1 forces 2, 3, 4 in one call.
static bool chain_ok(ClauseStore& s, Env& e, const int ids[3]) {
    e.set(N(1));
    if (e.propagate(s, {N(1)}) != -1) return false;
    const std::vector<std::pair<int, int>> want = {{P(2), ids[0]}, {P(3), ids[1]}, {P(4), ids[2]}};
    return e.trail == want;
}

int main() {
    // Empty store: propagate with nothing new and reduce are no-ops.
    {
        ClauseStore s;
        s.init(6, 10);
        Env e(6);
        if (s.size() != 0) return fail("empty store has clauses");
        if (e.propagate(s, {}) != -1) return fail("empty store reported a conflict");
        s.reduce();
        if (s.size() != 0) return fail("reduce on an empty store");
        int none[1] = {0};
        if (s.add(none, 0, 1) != -1) return fail("add accepted an empty clause");
    }

    // (a) unit propagation chain, (c) revert and repeat, (b) a unit conflict.
    {
        ClauseStore s;
        s.init(6, 100);
        Env e(6);
        const int c0[] = {P(1), P(2)}, c1[] = {N(2), P(3)}, c2[] = {N(3), P(4)};
        const int ids[3] = {s.add(c0, 2, 2), s.add(c1, 2, 2), s.add(c2, 2, 2)};
        if (ids[0] < 0 || ids[1] < 0 || ids[2] < 0) return fail("add refused a binary clause");
        if (s.size() != 3) return fail("size after three adds");
        if (s.lbd(ids[1]) != 2) return fail("lbd not stored");

        if (!chain_ok(s, e, ids)) return fail("(a) chain did not force 2, 3, 4 with their reasons");

        // (c) revert the chain and propagate again: the watches moved by the
        // first pass still see every clause.
        for (int round = 0; round < 3; ++round) {
            e.revert();
            if (!chain_ok(s, e, ids)) return fail("(c) chain differs after a revert");
        }

        // (b) the unit clause {~4} contradicts the chain's 4.
        const int u[] = {N(4)};
        const int uid = s.add(u, 1, 1);
        if (uid < 0) return fail("(b) add refused a unit clause");
        if (e.propagate(s, {}) != uid) return fail("(b) unit {~4} not reported as the conflict");

        // After a revert the unit holds first (4 EMPTY), so the chain now ends
        // in a conflict on {~2,3} or {~3,4} (which one depends on the order
        // the forced literals are visited in).
        e.revert();
        e.set(N(1));
        const int r = e.propagate(s, {N(1)});
        if (r != ids[1] && r != ids[2]) return fail("(b) chain after the unit did not conflict");
        if (e.val[4] != 0) return fail("(b) unit did not force 4 EMPTY");
    }

    // (c) with watch moves: a ternary clause {0,1,2} moves its watch off ~0,
    // then forces 2 when 1 is set; after a revert it forces 0 from ~2, ~1.
    {
        ClauseStore s;
        s.init(6, 100);
        Env e(6);
        const int c[] = {P(0), P(1), P(2)};
        const int cid = s.add(c, 3, 2);
        e.set(N(0));
        if (e.propagate(s, {N(0)}) != -1 || !e.trail.empty()) return fail("(c) ternary clause forced with two free");
        e.set(N(1));
        if (e.propagate(s, {N(1)}) != -1) return fail("(c) ternary clause conflicted");
        if (e.trail != std::vector<std::pair<int, int>>{{P(2), cid}}) return fail("(c) moved watch did not force 2");
        for (int round = 0; round < 2; ++round) {
            e.revert();
            e.set(N(2));
            e.set(N(1));
            if (e.propagate(s, {N(2), N(1)}) != -1) return fail("(c) conflict after a revert");
            if (e.trail != std::vector<std::pair<int, int>>{{P(0), cid}}) return fail("(c) no force of 0 after a revert");
            // and a full falsification is a conflict
            e.revert();
            e.set(N(0));
            e.set(N(1));
            e.set(N(2));
            if (e.propagate(s, {N(0), N(1), N(2)}) != cid) return fail("(c) all-false clause not a conflict");
            e.revert();
        }
    }

    // Shared watch list, move path: A = {0,1,2} and B = {0,3} both watch 0,
    // A first. Setting ~0 moves A's watch to 2; B must still be visited and
    // force 3.
    {
        ClauseStore s;
        s.init(8, 100);
        Env e(8);
        const int a[] = {P(0), P(1), P(2)}, b[] = {P(0), P(3)};
        s.add(a, 3, 2);
        const int bid = s.add(b, 2, 2);
        e.set(N(0));
        if (e.propagate(s, {N(0)}) != -1) return fail("shared/move: conflict");
        if (e.trail != std::vector<std::pair<int, int>>{{P(3), bid}})
            return fail("shared/move: the entry after a moved watch was not visited");
    }

    // Shared watch list, keep path: C = {0,1} with 1 already true when 0 goes
    // false keeps its watch on 0 (D = {0,4}, behind it, forces 4). After
    // undoing both rounds, ~0 alone must make C force 1.
    {
        ClauseStore s;
        s.init(8, 100);
        Env e(8);
        const int c[] = {P(0), P(1)}, d[] = {P(0), P(4)};
        const int cid = s.add(c, 2, 2);
        const int did = s.add(d, 2, 2);
        e.set(P(1));
        if (e.propagate(s, {P(1)}) != -1 || !e.trail.empty()) return fail("shared/keep: round 1");
        const std::size_t m1 = e.mark();
        e.set(N(0));
        if (e.propagate(s, {N(0)}) != -1) return fail("shared/keep: conflict");
        if (e.trail != std::vector<std::pair<int, int>>{{P(4), did}}) return fail("shared/keep: D not forced");
        e.revert_to(m1);
        e.revert_to(0);
        e.set(N(0));
        if (e.propagate(s, {N(0)}) != -1) return fail("shared/keep: conflict after revert");
        if (e.trail != std::vector<std::pair<int, int>>{{P(1), cid}, {P(4), did}})
            return fail("shared/keep: a kept watch was lost (C did not force 1)");
    }

    // Shared watch list, conflict path: E = {0,5} and F = {0,6} watch 0, E
    // first. ~0 with ~5 already set: E is the conflict. After a revert, ~0
    // alone must still reach F, the entry behind the conflict.
    {
        ClauseStore s;
        s.init(8, 100);
        Env e(8);
        const int ec[] = {P(0), P(5)}, fc[] = {P(0), P(6)};
        const int eid = s.add(ec, 2, 2);
        const int fid = s.add(fc, 2, 2);
        e.set(N(0));
        e.set(N(5));
        if (e.propagate(s, {N(0), N(5)}) != eid) return fail("shared/conflict: E not the conflict");
        e.revert_to(0);
        e.set(N(0));
        if (e.propagate(s, {N(0)}) != -1) return fail("shared/conflict: conflict after revert");
        if (e.trail != std::vector<std::pair<int, int>>{{P(5), eid}, {P(6), fid}})
            return fail("shared/conflict: an entry behind the conflict was lost");
    }

    // Partial revert: G = {0,1,2}, H = {~2,3}. Round 1 sets ~0 (G moves its
    // watch to 2); round 2 sets ~1 (G forces 2, H forces 3). Undo round 2
    // only, keeping ~0, then propagate ~3: H forces ~2, G forces 1. Undo that
    // and replay round 2: the same trail as the first time.
    {
        ClauseStore s;
        s.init(8, 100);
        Env e(8);
        const int g[] = {P(0), P(1), P(2)}, h[] = {N(2), P(3)};
        const int gid = s.add(g, 3, 2);
        const int hid = s.add(h, 2, 2);
        e.set(N(0));
        if (e.propagate(s, {N(0)}) != -1 || !e.trail.empty()) return fail("partial: round 1");
        const std::size_t m1 = e.mark();
        const std::vector<std::pair<int, int>> round2 = {{P(2), gid}, {P(3), hid}};
        e.set(N(1));
        if (e.propagate(s, {N(1)}) != -1 || e.trail != round2) return fail("partial: round 2");
        e.revert_to(m1);
        if (e.val[0] != 0 || e.val[1] != -1 || e.val[2] != -1 || e.val[3] != -1 || !e.trail.empty())
            return fail("partial: revert_to did not undo exactly round 2");
        e.set(N(3));
        if (e.propagate(s, {N(3)}) != -1) return fail("partial: conflict after the partial revert");
        if (e.trail != std::vector<std::pair<int, int>>{{N(2), hid}, {P(1), gid}})
            return fail("partial: wrong propagation after the partial revert");
        e.revert_to(m1);
        e.set(N(1));
        if (e.propagate(s, {N(1)}) != -1 || e.trail != round2) return fail("partial: replayed round 2 differs");
        // and a full falsification of G after the partial revert is its conflict
        e.revert_to(m1);
        e.set(N(1));
        e.set(N(2));
        if (e.propagate(s, {N(1), N(2)}) != gid) return fail("partial: all-false G not the conflict");
    }

    // A unit clause added while its cell is free forces it on the next call,
    // even with nothing new, and what it forces propagates in the same call.
    {
        ClauseStore s;
        s.init(6, 100);
        Env e(6);
        const int b[] = {N(5), P(2)};
        const int bid = s.add(b, 2, 2);
        const int u[] = {P(5)};
        const int uid = s.add(u, 1, 1);
        if (e.propagate(s, {}) != -1) return fail("unit {5}: spurious conflict");
        const std::vector<std::pair<int, int>> want = {{P(5), uid}, {P(2), bid}};
        if (e.trail != want) return fail("unit {5} not forced, or its consequence not propagated");
        // a line setting 5 EMPTY afterwards contradicts the unit
        e.revert();
        e.set(N(5));
        if (e.propagate(s, {N(5)}) != uid) return fail("unit {5}: falsified unit not reported");
    }

    // (d) reduce: 20 clauses on disjoint cells, lbd 2..21, the two worst locked.
    {
        const int n_cells = 64;
        ClauseStore s;
        s.init(n_cells, 10);
        Env e(n_cells);
        std::vector<std::vector<int>> cl;
        std::vector<int> id;
        for (int i = 0; i < 20; ++i) {
            cl.push_back({P(3 * i), N(3 * i + 1), P(3 * i + 2)});
            id.push_back(s.add(cl[i].data(), 3, 2 + i));
            if (id[i] < 0) return fail("(d) add refused");
        }
        for (int i = 0; i < 20; i += 3) s.bump(id[i]);
        s.lock(id[19], true);
        s.lock(id[18], true);
        s.reduce();
        if (s.size() > 10) return fail("(d) reduce left more than max_clauses");
        std::vector<char> alive(20, 0);
        // the survivors are the locked two and the lowest lbds
        for (int i = 0; i < 20; ++i) alive[i] = (i < s.size() - 2) || i >= 18;
        for (int i = 0; i < 20; ++i) {
            if (!alive[i]) continue;
            if (!same_lits(s, id[i], cl[i])) return fail("(d) survivor's literals changed");
            if (s.lbd(id[i]) != 2 + i) return fail("(d) survivor's lbd changed");
        }
        // a surviving clause still propagates: falsify its last two literals
        if (e.propagate(s, {}) != -1) return fail("(d) spurious conflict");
        e.set(cl[0][1] ^ 1);
        e.set(cl[0][2] ^ 1);
        if (e.propagate(s, {cl[0][1] ^ 1, cl[0][2] ^ 1}) != -1) return fail("(d) survivor conflicted");
        if (e.trail != std::vector<std::pair<int, int>>{{cl[0][0], id[0]}}) return fail("(d) survivor did not force");
        // so does a locked one
        e.set(cl[19][0] ^ 1);
        e.set(cl[19][1] ^ 1);
        if (e.propagate(s, {cl[19][0] ^ 1, cl[19][1] ^ 1}) != -1) return fail("(d) locked clause conflicted");
        if (e.trail.back() != std::pair<int, int>{cl[19][2], id[19]}) return fail("(d) locked clause did not force");
        // a deleted clause does not
        const int gone = 17;
        if (alive[gone]) return fail("(d) test expects clause 17 deleted");
        e.trail.clear();
        e.set(cl[gone][0] ^ 1);
        e.set(cl[gone][1] ^ 1);
        if (e.propagate(s, {cl[gone][0] ^ 1, cl[gone][1] ^ 1}) != -1 || !e.trail.empty())
            return fail("(d) a deleted clause still propagates");
        // a new clause reuses a dead id and works
        e.revert();
        const int fresh[] = {N(62), N(63)};
        const int fid = s.add(fresh, 2, 3);
        bool reused = false;
        for (int i = 0; i < 20; ++i) if (!alive[i] && id[i] == fid) reused = true;
        if (!reused) return fail("(d) add did not reuse a deleted id");
        if (!same_lits(s, fid, {N(62), N(63)})) return fail("(d) reused id has wrong literals");
        e.set(P(62));
        if (e.propagate(s, {P(62)}) != -1) return fail("(d) reused clause conflicted");
        if (e.trail != std::vector<std::pair<int, int>>{{N(63), fid}}) return fail("(d) reused clause did not force");
    }

    std::printf("ok: clause store propagation, conflicts, reverts, units, reduce\n");
    return 0;
}
