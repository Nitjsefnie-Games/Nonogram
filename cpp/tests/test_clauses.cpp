#include "../clauses.hpp"
#include <cstdio>
#include <utility>
#include <vector>

// Variables are cells. P(x): cell x FULL, N(x): cell x EMPTY (the store's
// literal encoding: cell * 2 + (value == FULL)).
static int P(int x) { return x * 2 + 1; }
static int N(int x) { return x * 2; }

// A minimal assignment for the store's callbacks: val[cell] is -1 unknown,
// 0 EMPTY, 1 FULL; every forced literal is appended to the trail with the
// clause that forced it.
struct Env {
    std::vector<int> val;
    std::vector<std::pair<int, int>> trail;   // (literal, reason clause id)
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
        return true;
    }
    void set(int l) { val[l >> 1] = l & 1; }
    void revert() { for (int& v : val) v = -1; trail.clear(); }
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
