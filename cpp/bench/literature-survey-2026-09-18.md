# Nonogram exact-solving literature survey — techniques that change the search tree

Prepared 2026-09-18. Every claim is tagged **READ** (I read the source text this session),
**ABSTRACT-ONLY**, or **SECONDARY** (from a citing/reporting work).

Local copies of everything I downloaded are in this same scratchpad directory:
`wu2013.pdf` / `wu2013_cols.txt` / `pg*.png` (IEEE TCIAIG 2013 paper, rendered per column),
`bako2009.pdf` / `bako2009.txt` (Pattern Recognition 2009), `arslani_sat.pdf` / `.txt` (Basel BSc
thesis 2025), `survey_index.txt`, `survey_dom.html`, `survey_caching.html`, `survey_copris.html`,
`solving.txt` (Wolter's webpbn survey pages), `nonogrid.txt`, `acg10.txt`, `o.txt` (ICG-180067
abstract).

---

## 0. Puzzle-name cross-reference (READ — webpbn survey sample table)

Your hard cases are the survey's named hard puzzles. Column order of the survey's
"Part I: Solvers Capable of Uniqueness Checking" table is
`Wu(LalaFrogKK) | Syromolotov | Wolter(pbnsolve) | Olšák | Simpson | BGU | Tamura/Copris | Lagerkvist/Gecode | Kjellerstrand/Gecode | Kjellerstrand/Lazyfd`;
`+` = did not finish.

| puzzle | survey name | size | Wu | Syro | pbnsolve | Olšák | Simpson | BGU | Copris | Lagerkvist/Gecode | Kjell/Gecode | Lazyfd |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 8098 | 9-Dom | 19x19 | 20.3s | 11.7s | 11.0s | 4.0m | + | 2.8m | 4.9s | 12.6m | 3.8s | **1.8s** |
| 6574 | Forever | 25x25 | 0.34s | 13.9s | 3.7s | 2.0s | 18.9s | 44.3s | 6.4s | 4.7s | 2.3s | 1.7s |
| 12548 | Sierp | 47x40, multiple | + | 2.2m | + | + | + | + | **1.4m** | + | + | + |
| 22336 | Gettys | 99x59, multiple | (size) | + | + | + | + | + | **10.5m** | + | + | + |

Two facts worth internalising before reading the rest:

* **22336 and 12548 are not unsolved in the literature.** Tamura's Copris (block-position CSP
  compiled to SAT, solved by a CDCL solver) does 22336 in 10.5 min and 12548 in 1.4 min on
  2013-era hardware, while *every* probing/heuristic solver in the survey times out. Wolter:
  "It solved all 2,491 puzzles in our 'full' test set… only three puzzles took more than 40
  seconds… So this program was able to solve all 6,562 published black and white puzzles on
  webpbn." (READ)
* **n-Dom is the family where nogood-learning CP wins outright.** Lazyfd (G12 lazy clause
  generation over a `regular`/automaton model) does 9-Dom in 1.8s vs Wu's 20.3s and BGU's 2.8m,
  and Wolter notes it "really solves 14-Dom faster than 12-Dom" — i.e. the exponential-in-n
  behaviour that you see (8x per size step) is *not* intrinsic. (READ)

---

## (a) Ranked candidate techniques

Ranked by (expected tree reduction) × (confidence) ÷ (cost).

| # | Technique | Source | What it deduces beyond your FP1 | Measured effect in source | Impl. cost | Confidence it prunes *your* tree |
|---|---|---|---|---|---|---|
| 1 | **Min-logd branch score** = `argmax_p [ min(m_p0,m_p1) + \|log(1+m_p0) − log(1+m_p1)\| ]` | Wu et al. TCIAIG 2013 §V-A, Tables I–II (**READ**) | Nothing new deduced; it is a strictly better tie-break on the *same* probe data you already compute. Your `min(...)` is their "Min". | With FP2: 1000×25x25 set 2,064.1→1,199.0 backtracking calls (−42%), 1.84s→1.10s; 100-hard set 60,853→42,814 calls (−30%), 57.4s→37.6s. "Min-logd heuristic clearly performs the fastest in all cases." | ~10 lines | **High.** Same data, exact, no state. Cheapest real node win available. |
| 2 | **FP2 — contrapositive probing with per-grid `pclist`** | Wu et al. 2013 §III-B (**READ**) | From a probe deriving `p_a→p_b` it *stores* `¬p_b→¬p_a` into the grid `G_{p_b,0}`, so the later probe of `p_b=0` starts with `p_a=0` already set. Yields forced cells from 2-step implication chains that single-cell probing + propagation provably cannot reach (their Figs. 3, 5, 6). | Painted pixels before search: FP1 201.2 → FP2 210.6 (set 1), 39.2 → 43.2 (set 2). Backtracking calls with Min-logd: 2,821→1,199 (set 1, −57%); 60,618→42,814 (set 2, −29%). Time −19% avg despite maintaining `pclist` (0.55ms→0.60ms per FP call). | High: needs persistent `G_{p,0}/G_{p,1}` grid pair per unpainted pixel + `pclist` + `UpdateOnAllG`. Memory `2l²+1` grids. | **High** that it prunes; **medium** that it pays off in wall clock given your solver discards probe state. This is the single largest documented node reduction. |
| 3 | **Group-based fully probing** — condense the pixel-relation graph into equivalence groups ("consistent" `p≡q` / "inverse" `p≡¬q`) and branch on groups | Chen & Huang, ICGA J. 40(4):387–396, 2018 (**ABSTRACT-ONLY**, abstract read verbatim) | Your FP1 extracts only *forced* cells (one/both probes fail). It throws away the implication pairs. SCC condensation of that implication graph gives `p≡q` and `p≡¬q` classes: one branch decision then settles the whole class, and any class containing both `p` and `¬p` is an immediate contradiction. | Abstract: "the pixels can be aggregated into groups, thereby the space of search tree of the Nonogram backtracking algorithm can be reduced." No numbers accessible. | Medium — needs FP2's implication pairs first, then Tarjan SCC over `2l²` literals. | **Medium-high.** Mechanically sound and it is exactly the structure your FP1 currently discards. Unquantified. |
| 4 | **Nogood learning over the line propagator** (explaining DFA/MDD propagator + 1UIP, i.e. lazy clause generation), or hand off to a CDCL solver when probing stalls | Gange, Stuckey & Szymanek, *Constraints* 16(4):407–429, 2011 (**ABSTRACT-ONLY**; nonogram/n-Dom results **SECONDARY** via Wolter); Lazyfd & Copris rows above (**READ**); nonogrid README (**READ**) | Records *why* a subtree failed as a reusable clause over cell literals, so the same conflict is not re-derived in a sibling subtree. This is the only mechanism in the literature that demonstrably kills the n-Dom explosion. | Lazyfd 9-Dom 1.8s (vs your 12-Dom >15min); 14-Dom faster than 12-Dom. Copris solves 22336 in 10.5m, 12548 in 1.4m, and all 6,562 published webpbn B&W puzzles. nonogrid's `sat` feature: "only two puzzles found that solved longer than an hour: 25820 and 26520". Wolter cites the MDD-with-explanation paper as having "impressive test results comparing performance of an incremental propagation algorithm with learning to pbnsolve". | Very high (a second solver core). | **High** for the hard cases, but with a serious caveat: you count *all* solutions, and CDCL gives you satisfiability, not a model count. You would need all-SAT/blocking-clause enumeration, and clause learning interacts badly with naive solution enumeration. |
| 5 | **Reuse the probe grid as the child node's initial state** | Wu et al. 2013 §IV `BACKTRACKING` lines 8–11 (**READ**) | No new deduction on its own, but the child inherits the probe's propagated grid *and its `pclist`*, so with FP2 the child starts strictly more constrained than a fresh node. | Not isolated in the paper's ablation. | Low-medium (memory for the retained grid). | **Low** on nodes alone; **medium** in combination with #2. Cheap if you already keep the probe result. |
| 6 | **Block-position variables + cross-line channeling (CSP/order encoding)** | Tamura Copris model (**READ** via Wolter's derivation page); Aramian & Yeghiazaryan, CG 2024 (**ABSTRACT-ONLY**); Arslani BSc thesis 2025 §5 (**READ**) | Carries state your cell-only grid cannot express: "block *b* of row *i* cannot start at column *s*". Pruned block domains survive across passes and are what the learned clauses in Copris/Lazyfd are stated over. | Indirect: the two solvers using it (Copris, Lazyfd) are the only ones that solve 22336/12548. Arslani: Block-Based encoding yields smaller formulas and faster solving than Sequence-Enumeration over 2,242 webpbn puzzles. | High (new state representation, new propagator). | **Medium.** Strong empirical association, but always bundled with clause learning, so the block model alone is unproven. |
| 7 | **FP3 — full transitive closure of PROPAGATE implications *and* contrapositives (2-SAT graph traversal)** | Wu et al. 2013 §III-C (**READ**) | Closes the remaining gap: derives `p_i→p_j` and `¬p_j→¬p_i` for all `i<j` along an implication chain (their Fig. 8), which FP2 cannot do in one direction. | FP3 vs FP2: calls only **1% lower** (42,646.6 vs 42,814.3); painted pixels +0.7 (211.3 vs 210.6). Time complexity rises `O(kl⁵)`→`O(l⁶)`. Paper's own verdict: "propagating all the PROPAGATE implications and contrapositives in FP3 does not yield much improvement." | High | **Low.** Do FP2, skip FP3. |
| 8 | **Probing sequence / re-probing policy tuning** | Huang, Yeh, Huang, Guo & Chen, ICGA J. 40(4):397–405, 2018 (**SECONDARY**: search-engine abstract); Guo et al., ACG 2019, LNCS 12516:119–130 (**ABSTRACT-ONLY**, read verbatim) | Not a new deduction: the *order* in which pixels are probed within a pass, and when the pass restarts, changes how many pixels a single FP round paints (because contrapositives accumulate). | ACG abstract: "we found several critical factors influencing fully probing efficiency greatly, i.e. re-probing policy, probing sequence, and computational overhead… our new fully probing methods have the potential to improve the speed of solving nonogram puzzles significantly." ICGA 2018: "different probing sequences can lead to significant differences in the numbers of painted pixels after fully probing." No numbers accessible. | Low | **Medium**, but note: order-sensitivity is largely a *consequence* of FP2-style accumulated state. With your current pass semantics (commit all forced cells, restart once) order matters much less. Rank this *after* #2. |
| 9 | **Regional / prefix "summing" propagator** | Wolter, webpbn "advanced solving techniques" (**READ**) — human technique, no surveyed solver implements it | For a band of rows `1..r`: Σ(row clue totals over the band) must equal Σ over columns of (# cells that column places in rows `1..r`), and the latter is an *interval* derived from each column's block-position domain. Not your whole-grid flow check — it is a per-prefix count with per-column feasible-count bounds. | None. Wolter: "I've only ever used this trick on a handful of actual puzzles." | Medium | **Low-medium, speculative.** Flagged because it is the shape of reasoning that cracks n-Dom-like puzzles for humans, and because it is *not* the global flow bound you already measured to be useless. No published measurement exists. |
| 10 | **Pairwise probing** (probe *pairs* of cells jointly, 4 assignments) | Tsai & Huang, "A Pairwisely Probing Approach to Solving Nonogram Puzzles", TAAI 2021 (**TITLE-ONLY** — could not access) | The natural strengthening of FP: `O(l⁴)` probes instead of `O(l²)`. Subsumes B&K's same-line pair tests and extends them cross-line. | Unknown | Very high | **Unknown.** Listed for completeness; cost/benefit undetermined. |
| — | **B&K same-line pairwise 2-SAT clause collection** | Batenburg & Kosters, *Pattern Recognition* 42(8):1672–1683, 2009 (**READ**) | For every pair of undecided pixels on a line, test all 4 assignments with the line-fixability DP; each infeasible assignment is a 2-SAT clause; then find every literal with a path to its negation. | **Dominated.** Wu Table III: pixels painted before search = PROPAGATE 32.3, **2-SAT 94.9**, FP1 201.2, FP2 210.6, FP3 211.3 (set 1); 8.9 / 19.5 / 39.2 / 43.2 / 43.2 (set 2). Wu §III-C proves every pixel B&K's 2-SAT paints is also painted by FP3. Cost `O(kl⁶)` to build + `O(l⁷)` to close, vs `O(kl⁵)` for FP1/FP2. | High | **Very low — do not implement.** Strictly weaker than FP2/FP3 at higher cost. |
| — | **Discrete-tomography / network-flow (linesum) relaxation** | B&K 2009 §3.2 (**READ**); re-measured by Wu §V-A (**READ**) | Transportation-problem feasibility over row/column line sums with fixed pixels as capacity-0 arcs. | Wu: "for all the puzzles in the above two sets (1100 puzzles), the 2-SAT method with DT can paint in total only **five more pixels** than the 2-SAT method… DT is not critical." | — | **Dead end — confirmed.** This is exactly your bipartite-flow "summing" check; the literature independently measures it at ~5 pixels / 1100 puzzles. Your 0-pruned-nodes result is the expected result, not a bug. |
| — | **Reversed-clue canonicalisation of the line cache** | Wolter, "Effect of Line Solution Caching on Pbnsolve Run-times" (**READ**) | Treat clue `3 1 1` and `1 1 3` as the same cache key by reversing the state string. | "It does, but only very slightly. Even on 9-Dom, which has lots of reversed clues, the hit rate was improved only 0.3%." | Low | **Dead end.** |
| — | **Probe-count throttling by priority** | nonogrid README (**READ**) | Probe only cells with `P = N + R + C ≥ threshold`, where `N∈[0,4]` = solved/edge neighbours, `R`,`C` = row/column solved-fraction. | "`LOW_PRIORITY=1 nonogrid puzzles/6574.xml` can be solved 3 times faster than standard way". | Low | **Low** — and you already tested and rejected "probe only the K most constrained cells". The 3x is wall-clock on one puzzle, not a tree reduction. |

---

## (b) Precise algorithms for the top 3

### b.1 FP1 and FP2 (Wu, Sun, Chen, Chen, Kuo, Kang, Lin, IEEE TCIAIG 5(3):251–264, 2013) — READ

Exact author list from the paper: **I.-Chen Wu, Der-Johng Sun, Lung-Ping Chen, Kan-Yueh Chen,
Ching-Hua Kuo, Hao-Hua Kang, Hung-Hsuan Lin.** (The "Yen" in your brief is not an author of this
paper; S.-J. Yen is a co-author of the tournament reports.)

Definitions quoted from the paper:

> "The idea of FP is to make guesses for all unpainted pixels `p` (with a value of `?`) in advance,
> and then perform propagation for each guess. The first proposed FP method, named FP1, maintains
> pairs of grids `G_{p,0}, G_{p,1}` for all pixels `p`, where both `G_{p,0}` and `G_{p,1}` represent
> the painted grids after guessing `p` to be 0 and 1, respectively."

> "The FP method is different from backtracking in that the FP method is not recursive. That is,
> each pixel is only tested once, just enough to provide backtracking with more accurate guidance
> when choosing pixels to guess."

**FP1, verbatim pseudocode (paper p. 256):**

```
procedure FP1(G)
  1. Initialize G_{p,0}, G_{p,1} for all unpainted pixel p
  2. repeat
  3.    PROPAGATE(G)
  4.    if (status(G) is CONFLICT or SOLVED) then return
  5.    UPDATEONALLG(G)
  6.    for (each unpainted pixel p in G) do
  7.       PROBE(p)
  8.       if (status(G) is CONFLICT or SOLVED) then return
  9.       if (status(G) is PAINTED) then break
 10.    end for
 11. until Π(G) = ∅
end procedure

Procedure PROBE(p)
  1. PROBEG(p,0);   // guess p = 0 and probe G_{p,0}
  2. PROBEG(p,1);   // guess p = 1 and probe G_{p,1}
  3. if (both status(G_{p,0}) and status(G_{p,1}) are CONFLICT)
         then status(G) ← CONFLICT; return
  4. if (status(G_{p,0}) is CONFLICT) then
  5.    Let Π be the set of newly painted pixels in G_{p,1} with respect to G
  6. else if (status(G_{p,1}) is CONFLICT) then
  7.    Let Π be the set of newly painted pixels in G_{p,0} with respect to G
  8. else
  9.    Let Π be the set of pixels with the same value 0 or 1 in both
          G_{p,0} and G_{p,1} with respect to G
 10. end if
 11. if (Π ≠ ∅) then UPDATEONALLG(Π); status(G) ← PAINTED
 12. else status(G) ← INCOMPLETE
end procedure
```

Notes that matter for your implementation:

* **Line 9 of FP1 is the paper's re-probing policy:** `break` out of the probe sweep as soon as
  *any* probe paints something, and restart from `PROPAGATE`. Your solver instead finishes the
  sweep, commits all forced cells of the pass, and restarts the node once. Those are two of the
  three policies the ACG 2019 paper says matters; the third is FP2's worklist (below).
* **Case 4 of `PROBE` (line 9) is probe merging** — commit cells that both probes assign
  identically. You measured this as "no node change on tested puzzles". In the paper it is a core
  deduction of FP1, not an optional extra, so your negative result is a genuine divergence worth
  keeping in mind when reading their node counts.
* `UPDATEONALLG` pushes `G`'s newly painted pixels into *every* `G_{p,0}/G_{p,1}` — i.e.
  pass-to-pass probe-state reuse, which you also tested and rejected. FP2's gain depends on this
  machinery existing, so "FP2 without persistent probe grids" is not the paper's FP2.
* Complexity: `O(kl⁵)` for FP1 and FP2; `PROBE` invoked `O(l⁴)` times; `2l²+1` grids each
  maintained by *incremental* propagation.

**FP2 — the contrapositive extension.** The stated defect of FP1:

> "The major problem of FP1 is that it does not make use of the contrapositive, that is, an
> implication `p_a → p_b` implies the contrapositive `¬p_b → ¬p_a`. In FP1, if we obtain `p_b = 1`
> by performing PROBE on `G_{p_a,1}` (assuming `p_a = 1`), then this implies `p_a → p_b`, which
> implies its contrapositive `¬p_b → ¬p_a`. Then, the contrapositive implies `p_a = 0` on
> `G_{p_b,0}` (assuming `p_b = 0`). Unfortunately, the FP1 method does not request to set (or paint)
> `p_a = 0` on `G_{p_b,0}` in this case. These implications and contrapositives… are also called
> pixel relations in this paper."

Mechanism, quoted:

> "In the new method, all grids, say `G_{p,0}`, are associated with a list of pairs of pixels and
> their values, denoted by `pclist(G)`, used to record the derived values for given pixels, mainly
> for the contrapositive. … When probing a grid, update the cells of the grid from the list before
> using PROPAGATE. … The next issue is when to probe a grid. For this issue, we maintain a list of
> pixels `p`, denoted by `P`, to indicate both grids `(G_{p,0}, G_{p,1})` to be probed. Whenever a
> new contrapositive is added into `pclist(G_{p,0})` or `pclist(G_{p,1})`, pixel `p` is put into
> `P`, unless `p` is already in `P`."

**FP2, verbatim pseudocode (paper p. 257):**

```
procedure FP2(G)
  1. Initialize G_{p,0}, G_{p,1} for all unpainted pixel p
  2. Initialize the list P to contain all unpainted pixels p in G
  3. PROPAGATE(G)
  4. if (status(G) is CONFLICT or SOLVED) then return
  5. while (P ≠ ∅) do
  6.    Retrieve one pixel p from P
  7.    PROBE(p)
  8.    if (status(G) is CONFLICT or SOLVED) then return
  9. end while
end procedure

procedure PROBEG(p, c)
  1. Update pixels of G_{p,c} from the pixel list pclist(G_{p,c})
  2. PROPAGATE(G_{p,c})
  3. if status(G_{p,c}) is CONFLICT then return
  4. Π ← Π(G_{p,c})
  5. for each pixel p' with value c' in Π do
  6.    Put the contrapositive, (p, c̄), into pclist(G_{p',c'}),
          where c̄ is complement of c
  7.    Add p' into P, if p' is not in P yet
  8. end for
end procedure

procedure UPDATEONALLG(Π)
  1. for each p ∈ Π with value c in G do
  2.    Put the pair (p, c) into all the lists of G, all G_{p,0} and G_{p,1}
  3.    Add p' into P, if p' is not in P yet
  4. end for
end procedure
```

(`PROBE` is unchanged from FP1. In FP1, `PROBEG(p,c)` is simply `PROPAGATE(G_{p,c})`.)

Read line 6 of `PROBEG` carefully — it is the whole idea: probing `p=c` and deriving `p'=c'`
records the *contrapositive* `(p, c̄)` into the grid for the *opposite* guess on `p'`. So the
pclist of `G_{p',c'}`… note the paper's own worked example (Fig. 3) uses `pclist(G_{p_b,0})`
receiving `(p_a, 0)` from `p_a → p_b`; the printed line 6 says `pclist(G_{p',c'})`, and the
narrative is the authority: the pair stored is "for the contrapositive", i.e. it lands in the grid
that assumes `¬p'`. Implement from the narrative and the Fig. 3 walkthrough, not from the index
in line 6 alone.

**Backtracking (verbatim, paper p. 259):**

```
procedure BACKTRACKING(G)
  1. INITIALIZE all G_{p,0} and G_{p,1} for all p
  2. FP3(G); // or FP1(G), FP2(G)
  3. if (status(G) is CONFLICT) then return
  4. if (status(G) is SOLVED) then
  5.    Continue to solve more or stop depending on the requirement.
  6. endif
  7. p = CHOOSEPIXEL()
  8. G = G_{p,0}
  9. BACKTRACKING(G_{p,0})
 10. G = G_{p,1}
 11. BACKTRACKING(G_{p,1})
end procedure
```

Lines 8–11: **the probe grid becomes the child node's state.** The probe work is not discarded.

**Measured numbers (Tables I–III, READ from the rendered page 11).**
Set 1 = 1000 random 25x25 puzzles from the TAAI 2011 tournament (50%→35% black density, not
guaranteed unique). Set 2 = 100 harder unique 25x25 puzzles, solver forced to find a *second*
solution, i.e. full enumeration — the closest analogue to what your solver does.
Hardware: Intel i5-2400 3.10 GHz. "#Calls" = calls to `BACKTRACKING`.

Table I (set 1):

| heuristic | FP1 #Calls | FP1 s | FP2 #Calls | FP2 s | FP3 #Calls | FP3 s |
|---|---|---|---|---|---|---|
| Min-logd | 2,821.1 | 1.30 | **1,199.0** | 1.10 | 1,195.2 | 1.08 |
| Mul | 3,261.4 | 1.47 | 2,488.6 | 1.35 | 2,486.3 | 1.37 |
| Min-logm | 2,545.7 | 1.68 | 2,008.5 | 1.56 | 2,001.3 | 1.59 |
| Sqrt | 2,644.9 | 2.01 | 1,888.1 | 1.75 | 1,875.6 | 1.75 |
| Min *(= your current score)* | 2,685.1 | 1.70 | 2,064.1 | 1.84 | 2,048.2 | 1.81 |
| Sum | 6,175.5 | 3.09 | 4,177.3 | 2.43 | 4,139.0 | 2.43 |
| Max | 9,646.6 | 4.17 | 6,758.4 | 3.75 | 6,687.4 | 3.54 |

Table II (set 2, full enumeration):

| heuristic | FP1 #Calls | FP1 s | FP2 #Calls | FP2 s | FP3 #Calls | FP3 s |
|---|---|---|---|---|---|---|
| Min-logd | 60,618.3 | 44.6 | **42,814.3** | 37.6 | 42,646.6 | 38.0 |
| Mul | 68,678.3 | 45.4 | 50,407.0 | 38.8 | 50,263.8 | 39.0 |
| Min-logm | 85,184.3 | 65.0 | 57,975.3 | 55.4 | 57,691.6 | 50.4 |
| Sqrt | 89,538.6 | 67.2 | 63,333.6 | 52.3 | 63,085.8 | 53.1 |
| Min | 89,807.7 | 73.0 | 60,853.1 | 57.4 | 60,634.8 | 58.5 |
| Sum | 190,544.5 | 101.4 | 155,508.6 | 92.3 | 155,168.1 | 93.6 |
| Max | 247,925.2 | 131.0 | 197,864.7 | 115.5 | 197,112.5 | 116.7 |

Table III — average pixels painted before backtracking (625-cell grids):

| | PROPAGATE | 2-SAT (B&K) | FP1 | FP2 | FP3 |
|---|---|---|---|---|---|
| set 1 | 32.3 | 94.9 | 201.2 | 210.6 | 211.3 |
| set 2 | 8.9 | 19.5 | 39.2 | 43.2 | 43.2 |

Also READ: line caching is used throughout ("we also supported caching for all lines, as suggested
in [Wolter]… we simply use a sufficiently large cache"), so these numbers are on top of a line
cache like yours.

### b.2 The seven choose-pixel heuristics, exact formulas (Wu 2013 §V, READ)

`m_{p,c}` = "the number of extra pixels painted in `G_{p,c}`, not in `G`" — i.e. exactly your
"cells settled by the FULL/EMPTY probe". `V_min(p) = min(m_{p,0}, m_{p,1})`,
`V_max(p) = max(m_{p,0}, m_{p,1})`, `V_log(p,c) = log(1 + m_{p,c}) + 1`.

```
1) Sum      : p* = argmax_p ( m_{p,0} + m_{p,1} )
2) Min      : p* = argmax_p V_min(p)                      <-- what you do now
3) Max      : p* = argmax_p V_max(p)                      <-- your "anytime" mode
4) Mul      : p* = argmax_p ( (m_{p,0}+1) * (m_{p,1}+1) )
5) Sqrt     : p* = argmax_p ( V_min(p) + sqrt( V_max(p) / (V_min(p)+1) ) )
6) Min-logm : p* = argmax_p ( V_min(p) + V_log(p,0) * V_log(p,1) )
7) Min-logd : p* = argmax_p ( V_min(p) + | V_log(p,0) - V_log(p,1) | )
```

`Min-logd` is `Min` plus a bounded asymmetry bonus: the added term is at most
`log(1+l²)`≈6.4 for a 25x25 grid while `V_min` ranges over hundreds, so in practice it is a
**tie-break among equal-`V_min` pixels that prefers the maximally *imbalanced* probe pair** —
the opposite of a balanced split, applied only within ties. That is the whole delta, and it buys
30–42% of the nodes in both of the paper's sets. It is the first thing I would try.

Paper's verdict, verbatim: "the results in Tables I and II indicate that the Min-logd heuristic
clearly performs the fastest in all cases. For this reason, we use Min-logd for the rest of the
experiments."

### b.3 Pixel-relation graph → forced cells and equivalence groups

Two sources, combined. The 2-SAT machinery is quoted from B&K (READ); the grouping application is
Chen & Huang's (ABSTRACT-ONLY), so the algorithm below is my reconstruction of the abstract's
claim on top of B&K's verbatim definitions. Treat the reconstruction as mine, not theirs.

B&K's definitions, verbatim (*Pattern Recognition* 42(8), §4):

> "Note that any such implication relation between two variables can be written in one of the forms
> `x ∨ y`, `x ∨ ¬y`, `¬x ∨ y` or `¬x ∨ ¬y`. This is the standard form of a 2-SAT clause… When
> solving a Nonogram, the goal is not to find an assignment of all variables that satisfies the
> 2-SAT constraints. Rather, we search for variables that must have the same truth value in all
> satisfying assignments. Assume that at least one such assignment exists. Then a variable `x` is
> false in all satisfying assignments if and only if there is a path from `x` to `¬x` in the
> dependency graph. Alternatively, `x` must be true in all satisfying assignments if and only if
> there is such a path from `¬x` to `x`."

> "Our procedure for combining the information from the subproblems (one for each line, and a
> complete DT problem) is as follows: for each pair of undecided pixels `(x, y)` involved in the
> subproblem, all four assignments are tested. If `x` and `y` are on one line, for each assignment
> the fixability computation from Section 3.1 is performed. Each such test that returns false
> provides an additional 2-SAT clause (e.g., `x ∨ ¬y`)."

And their explicit statement of the method's limit (why 3-SAT would be needed):

> "Although the 2-SAT approach is a powerful way to combine the knowledge from different partial
> problems, it generally does not capture all information that is present. For example, the three
> character string `s` over `{0,1,x}` with description `d = 0* 1^1 0*` yields rules that do not
> forbid the fix `000`, which is not a good fix."

Their driver loops, verbatim (Fig. 4):

```
Solver0(X, N)
  repeat
    X <- FullSettle(X, N);
    Collect a set S of 2-SAT clauses from relaxations;
    X <- 2SATSolve(X, S, N);
  until (no new fixed pixel(s) found) or (contradiction found);

Solver1(X, N)
  repeat
    X <- Solver0(X, N);
    repeat
      Select an undecided pixel X_ij;
      for each pixel value sigma in {0,1}
        Y <- X; Y_ij <- sigma;
        Y <- Solver0(Y, N);
        if (contradiction found) X_ij <- 1 - sigma;
    ...
```

`Solver1` is your FP1 with `Solver0` (rather than plain propagation) inside the probe. B&K note:
"the algorithm only makes a single guess at a time (i.e., no recursive guesses), thereby avoiding
the creation of a search tree." Their line-fixability DP `Fix(ℓ,k)` is `O(k·ℓ²)`; Wu's is
`O(kℓ)`/`O(ℓ)` with bit-vector merging ("we use two bits to indicate a pixel, bits 01 for 0, 10
for 1, and 11 for `?`, and use a 64-bit word to represent a line with no more than 32 pixels. Thus
the merging operation simply performs one bitwise-or operation on two 64-bit words").

**My reconstruction of group-based FP, cheap version, no pair probing needed:**

You already run both probes on every unknown cell at every branch node. Harvest the implication
graph for free:

```
# Nodes: 2*U literals for U unknown cells: (p,0) and (p,1).
# Build during the existing FP1 sweep, cost ~= what you already pay.
for each unknown p:
    for c in {0,1}:
        if probe(p,c) is CONFLICT:            # already handled: p forced to 1-c
            add_edge((p,c) -> (p,1-c))        # literal implies its own negation
        else:
            for each cell q newly settled to value v by probe(p,c):
                add_edge((p,c)   -> (q,v))    # PROPAGATE implication
                add_edge((q,1-v) -> (p,1-c))  # contrapositive  <-- FP2's insight, graph form

# 1. Forced cells: any literal with a path to its own negation is false.
#    Tarjan SCC over the 2U-literal graph, O(V+E).
for each literal L:
    if scc(L) == scc(neg(L)):  # both in one SCC => contradiction at this node
        node is dead
    elif path(L -> neg(L)):    # cheap: reachability in the condensation DAG
        force L false

# 2. Groups: every SCC is an equivalence class.
#    L and L' in one SCC  =>  they take the same truth value in every solution.
#    (p,1) and (q,1) in one SCC  =>  cells p,q are equal;
#    (p,1) and (q,0) in one SCC  =>  cells p,q are opposite.
#    Branch on a representative of the LARGEST class: one decision now settles
#    |class| cells instead of 1, and the branch score becomes
#    m'_{p,c} = m_{p,c} + (cells settled by the class it fixes).
```

The only piece your solver does not already compute is the contrapositive edge, and at graph level
that edge is free — you do not need FP2's persistent `G_{p,c}` grids or `pclist` to *record* it, only
to *exploit* it inside the same pass. So the staging I would suggest is:

1. Min-logd score (hours of work, 30–42% nodes in the paper's two sets).
2. Implication-graph harvest + SCC → extra forced cells and group-aware branch scoring
   (no new probe cost; the cheap surrogate for FP2 + group-based FP).
3. Only if (2) pays, full FP2 with persistent probe grids and `pclist`.
4. FP3, B&K 2-SAT pair tests, and the DT/flow bound: skip — measured worthless or dominated.

---

## (c) Item 6 — 2020–2026 work on exact nonogram solving

* **Guo, Huang, Yeh, Chang, Chen & Huang (2020).** "On Efficiency of Fully Probing Mechanisms in
  Nonogram Solving Algorithm", *Advances in Computer Games (ACG 2019)*, LNCS 12516, 119–130. New FP
  mechanisms built around three factors — re-probing policy, probing sequence, computational
  overhead; evaluated on past nonogram-tournament puzzle sets. **ABSTRACT-ONLY.**
* **Chen & Lin (2020).** "Requiem wins Nonogram tournament", *ICGA Journal* 42(1). Tournament
  report for the Requiem solver (bitboard + BMI line solver, `O(kl)` DP, "freedom" parameter).
  **TITLE-ONLY.**
* **Tsai & Huang (2021).** "A Pairwisely Probing Approach to Solving Nonogram Puzzles", TAAI 2021,
  Chaoyang Univ. of Technology. Probing over *pairs* of cells. **TITLE-ONLY — could not access.**
* **Guo, Y.-R.** MSc thesis, National Taichung Univ. of Education: "A Study of Efficient Fully
  Probing Mechanisms for Solving Nonogram Puzzles"; **Tsai, T.-S.** MSc thesis, same institution:
  "A Study on Fully Probing and Backtracking Algorithms in Solving Nonogram Puzzles". Listed on
  K.-C. Huang's homepage. Likely the full-detail versions of the ICGA/ACG papers.
  **TITLE-ONLY.**
* **Huang & co. (2020–2022), TCGA/NCS workshop papers (Chinese).** "以視覺化工具輔助 Nonograms
  解題搜尋階段相關問題研究" (visualisation tooling for the search phase, TCGA 2020);
  "Nonograms 中單行求解方法探討" (single-line solving methods, TCGA 2020); "以區塊挪移方式進行
  Nonogram 求解之研究" (block-shifting solving, NCS 2021); "Nonogram 解題中以邏輯規則進行單行求解
  效益之探討" (logical-rule line solving, TCGA 2022). **TITLE-ONLY.**
* **Aramian & Yeghiazaryan (2025, CG 2024).** "Solving Nonograms: A Constraint Satisfaction
  Approach", LNCS, doi 10.1007/978-3-031-86585-5_11. Per-line CSPs with *clue start positions* as
  variables (non-overlap + spacing constraints), then a board-level CSP with rows/columns as
  variables and intersection colour-consistency constraints; inference + backtracking with
  nonogram-specific logical analysis. **ABSTRACT-ONLY** (abstract read verbatim).
* **Arslani (2025).** "SAT Modeling and SAT Solver Implementation for Nonograms", BSc thesis,
  Univ. of Basel (Helmert group), 20 Aug 2025. Two CNF encodings derived from the Aramian &
  Yeghiazaryan CSP — Sequence Enumeration (one var per valid line colouring) and Block-Based (one
  var per block start position, with order-encoded at-most-one, block-consistency and cell-
  consistency clauses); own CDCL solver with nonogram-specific decision heuristics (Line-Aware,
  Block-Length-Aware, Sequential Order) vs Random/VSIDS; 2,242 webpbn puzzles. Finding: Block-Based
  is smaller and faster; structure-exploiting heuristics can beat general-purpose ones; overall
  performance still dominated by implementation efficiency vs MiniSat. **READ.**
* **He, Ju, Calver & Gao (Aug 2026).** "Evaluating SAT Solver Metrics as Predictors of
  Human-Perceived Nonogram Difficulty", arXiv:2608.23300. Human-difficulty study; finds solver
  metrics do not correlate with perceived difficulty, and that humans "prefer complex propagation".
  No solver-performance contribution. **ABSTRACT-ONLY.**
* **"Solving nonograms using Neural Networks" (2025).** arXiv:2501.05882 / *Array* (ScienceDirect).
  NN-assisted DFS/GA. Not exact-performance relevant. **ABSTRACT-ONLY.**
* **"Harmony Search Algorithm with Two Problem-Specific Operators for Solving Nonogram Puzzle"
  (2025).** *Mathematics* 13(9):1470 (MDPI). Metaheuristic, not exact. **TITLE-ONLY.**
* **"On Solving Simple Curved Nonograms" (2025).** arXiv:2505.01554. A different puzzle variant
  (curved nonograms), not grid nonograms. **TITLE-ONLY.**
* **"Algorithms Effectiveness comparison in solving Nonogram boards" (2021).** ResearchGate
  preprint; student-level comparison. **TITLE-ONLY.**

Nothing in 2020–2026 that I could find reports a *measured* exact-solving speedup on the webpbn
survey set. The measured state of the art on the hard puzzles remains Copris (2013, per Wolter's
survey) and lazy-clause-generation CP; the measured state of the art on probing remains
Wu 2013 + the two 2018 ICGA papers + ACG 2019, whose numbers are behind paywalls.

---

## (d) Could not access — what I tried

* **`ir.lib.nycu.edu.tw`** — this box's resolver still fails on it, but the host *is* alive
  (`140.113.37.243` via 8.8.8.8) and the repository has migrated to DSpace 7, so the legacy
  `/bitstream/...` URL now returns the SPA shell. Recipe that worked, for reuse:
  `curl --resolve ir.lib.nycu.edu.tw:443:140.113.37.243 'https://ir.lib.nycu.edu.tw/server/api/pid/find?id=hdl:11536/22772'`
  → item UUID → `/bundles` → ORIGINAL bundle → `/bitstreams` → `_links.content.href`.
  **I got the full Wu 2013 PDF this way.** Note: its math is set in subsetted TrueType with no
  ToUnicode map, so `pdftotext` silently drops every formula — I rendered the pages with
  `pdftoppm` and read them as images instead.
* **IOS Press `content.iospress.com`** — HTTP 403 to WebFetch *and* to the r.jina.ai reader for
  ICG-180067, ICG-180069, ICG-190097.
* **`journals.sagepub.com`** — ICG-180067 abstract obtained via r.jina.ai (quoted above);
  ICG-180069 ("Exploring effects of fully probing sequence…") and ICG-190097 ("A fast nonogram
  solver that won the TAAI 2017 and ICGA 2018 tournaments") returned a Cloudflare CAPTCHA
  interstitial. **Both remain ABSTRACT-ONLY/SECONDARY.** ICG-180069's abstract content in this
  report comes from a search-engine summary, not the page.
* **Springer `link.springer.com`** — direct `curl` gets a JS "Client Challenge"; WebFetch gets a
  303 to the IdP. The r.jina.ai reader does return the abstract + reference list (that is how I got
  the ACG 2019 and CG 2024 abstracts), but never the body. ACG 2019's experimental numbers are
  therefore unobtained.
* **Gange, Stuckey & Szymanek, "MDD propagators with explanation", *Constraints* 16(4):407–429
  (2011)** — abstract obtained from research.monash.edu; full text paywalled. Tried
  `people.eng.unimelb.edu.au/pstuckey/papers/{mdd,cp2011mdd,mddexpl}.pdf` (all HTML 404s) and
  ResearchGate (403). **Its nonogram/n-Dom comparison against pbnsolve — the single most
  interesting unquantified claim in this report — is SECONDARY only, from Wolter's survey.** Its
  reference list does cite `webpbn.com/survey/dom.html`, confirming n-Dom is in its benchmark set.
* **ResearchGate** — 403 on every attempt.
* **Semantic Scholar API** — HTTP 429, then empty result sets; no abstracts retrieved.
* **Tsai & Huang, TAAI 2021 "A Pairwisely Probing Approach…"** — only the author's homepage listing
  and the TAAI 2021 programme page; no IEEE/preprint copy located.
* **NTCU MSc theses (Guo, Tsai)** — not attempted beyond the homepage listing; they would need the
  Taiwanese NDLTD/Airiti repositories.
* **BGU solver (`cs.bgu.ac.il/~benr/nonograms/`)** — not fetched; the survey's description (READ)
  is the basis for what I say about it: probing like pbnsolve but more probes per decision point,
  plus the line-solution hash cache it originated.
