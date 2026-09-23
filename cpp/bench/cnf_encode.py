#!/usr/bin/env python3
"""Encode a nonogram as CNF whose models are exactly its solutions.

Variables: one per cell (true = FULL), and per line one per (block,
start position) with the block's feasible starts; the block variables
are one-hot per block, ordered with a gap between consecutive blocks,
and channelled to the cells: a block covers its cells and leaves the
cells just outside empty, and every FULL cell is covered by some block.
Given a solution grid the runs of FULL cells are exactly the blocks in
order, so the block variables are determined and the model count of the
formula is the solution count of the puzzle. The at-most-one of each
block is a ladder (sequential counter) whose auxiliaries are determined
too.

Usage: cnf_encode.py <puzzle file> [out.cnf]
Prints the variable count on stderr; the cell variable of (r, c) is
r * W + c + 1, so a model's grid can be read off the first H * W vars.
"""
import sys


def read(path):
    rows, cols, cur = [], [], None
    for line in open(path):
        line = line.strip()
        if line.startswith('#'):
            continue
        if line == '---':
            cur = cols
            continue
        if cur is None:
            cur = rows
        cur.append([int(x) for x in line.split()] if line and line != '0' else [])
    return rows, cols


class Cnf:
    def __init__(self, nvars):
        self.n = nvars
        self.clauses = []

    def new(self):
        self.n += 1
        return self.n

    def add(self, *lits):
        self.clauses.append(lits)

    def exactly_one(self, lits):
        self.add(*lits)
        # ladder at-most-one: s_i = OR(lits[0..i])
        if len(lits) <= 4:
            for i in range(len(lits)):
                for j in range(i + 1, len(lits)):
                    self.add(-lits[i], -lits[j])
            return
        prev = None
        for i, l in enumerate(lits):
            if i == len(lits) - 1:
                if prev is not None:
                    self.add(-l, -prev)
                break
            s = self.new()
            self.add(-l, s)
            if prev is not None:
                self.add(-prev, s)
                self.add(-l, -prev)
            prev = s


def encode_line(cnf, cells, clue):
    """cells: list of cell variables along the line; clue: block lengths."""
    n = len(cells)
    k = len(clue)
    if k == 0:
        for x in cells:
            cnf.add(-x)
        return
    # feasible start range of block j
    lo = [0] * k
    hi = [0] * k
    acc = 0
    for j in range(k):
        lo[j] = acc
        acc += clue[j] + 1
    acc = n
    for j in range(k - 1, -1, -1):
        acc -= clue[j]
        hi[j] = acc
        acc -= 1
    y = []  # y[j][t - lo[j]]
    for j in range(k):
        y.append([cnf.new() for _ in range(lo[j], hi[j] + 1)])
        cnf.exactly_one(y[j])
    # block j at t: covered cells full, boundaries empty
    for j in range(k):
        for t in range(lo[j], hi[j] + 1):
            v = y[j][t - lo[j]]
            for i in range(t, t + clue[j]):
                cnf.add(-v, cells[i])
            if t > 0:
                cnf.add(-v, -cells[t - 1])
            if t + clue[j] < n:
                cnf.add(-v, -cells[t + clue[j]])
    # ordering: block j+1 starts at least clue[j] + 1 after block j
    for j in range(k - 1):
        for t in range(lo[j], hi[j] + 1):
            v = y[j][t - lo[j]]
            for t2 in range(lo[j + 1], min(hi[j + 1], t + clue[j]) + 1):
                cnf.add(-v, -y[j + 1][t2 - lo[j + 1]])
    # every full cell is covered by some block
    for i in range(n):
        cover = []
        for j in range(k):
            for t in range(max(lo[j], i - clue[j] + 1), min(hi[j], i) + 1):
                cover.append(y[j][t - lo[j]])
        if cover:
            cnf.add(-cells[i], *cover)
        else:
            cnf.add(-cells[i])


def encode(rows, cols):
    H, W = len(rows), len(cols)
    cnf = Cnf(H * W)
    cell = lambda r, c: r * W + c + 1
    for r in range(H):
        encode_line(cnf, [cell(r, c) for c in range(W)], rows[r])
    for c in range(W):
        encode_line(cnf, [cell(r, c) for r in range(H)], cols[c])
    return cnf


def main():
    rows, cols = read(sys.argv[1])
    cnf = encode(rows, cols)
    out = open(sys.argv[2], 'w') if len(sys.argv) > 2 else sys.stdout
    out.write(f"c t mc\np cnf {cnf.n} {len(cnf.clauses)}\n")
    for cl in cnf.clauses:
        out.write(' '.join(map(str, cl)) + ' 0\n')
    print(f"{len(rows)}x{len(cols)}: {cnf.n} vars, {len(cnf.clauses)} clauses", file=sys.stderr)


if __name__ == '__main__':
    main()
