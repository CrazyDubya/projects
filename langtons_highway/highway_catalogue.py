#!/usr/bin/env python3
"""
Catalogue every "highway" of Langton's Ant up to a given period.

A highway is a periodic-with-drift orbit on an otherwise white plane: after
P steps the whole picture (cells + ant) has moved by a vector d != 0.
Equivalently: a configuration C and a P-step walk such that running the ant
for P steps on C yields C translated by d.  For fixed (P, d) that
fixed-point condition is a finite SAT problem; this module builds it,
enumerates every solution with CaDiCaL, and re-verifies each one by plain
simulation (``verify``), independently of the encoding.

Scope of an exhaustive run: all even P <= PMAX, every drift d with
|dx|+|dy| <= DMAX (one representative per 90-degree rotation class; the
initial heading is left free so rotations are covered).  Within that scope
the enumeration is complete: the walk may wander anywhere it can while still
returning to d after P steps (the "lens" |c|_1 + |c-d|_1 <= P).

Usage:
    python highway_catalogue.py --pmax 120 --dmax 10 --workers 4
    python highway_catalogue.py --single 104 2 2      # one (P, d) instance

Author: Claude (with Stephen)
Created: 2026-09-20
"""

from __future__ import annotations

import argparse
import bisect
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ant import Ant  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "data" / "catalogue"

DX = (0, 1, 0, -1)  # N E S W
DY = (1, 0, -1, 0)

Cell = Tuple[int, int]


def m1(c: Cell) -> int:
    return abs(c[0]) + abs(c[1])


# ----------------------------------------------------------------------
# CNF construction
# ----------------------------------------------------------------------
class Instance:
    """SAT encoding of 'period-P highway with drift d'."""

    def __init__(self, P: int, d: Cell, radius: int = 0):
        assert P % 2 == 0 and (d[0] + d[1]) % 2 == 0 and d != (0, 0)
        assert m1(d) <= P, "drift longer than the period is impossible"
        self.P, self.d, self.radius = P, d, radius
        self.nv = 0
        self.clauses: List[List[int]] = []
        self._build()

    def new(self) -> int:
        self.nv += 1
        return self.nv

    def add(self, *lits: int) -> None:
        self.clauses.append(list(lits))

    def _build(self) -> None:
        P, d = self.P, self.d
        dx, dy = d
        # lens: every cell a P-step walk from 0 to d can touch
        # Window.  Unrestricted: the lens of all P-step walks from 0 to d.
        # With a radius R: the period is anchored at the footprint's
        # leftmost-bottom cell (x >= 0, and y >= 0 when x == 0), so any
        # highway whose footprint bounding box is at most (R+1) x (R+1)
        # appears with x in [0, R] and y in [-R, R].  The window used for
        # colour/periodicity constraints stays convex (x >= 0 only); the
        # anchor rule is applied to the position variables.
        R = self.radius
        if R:
            assert 1 <= dx <= R and 0 <= dy <= R
            lens = [(x, y) for x in range(0, R + 1) for y in range(-R, R + 1)
                    if m1((x, y)) + m1((x - dx, y - dy)) <= P]
        else:
            lens = [(x, y) for x in range(-P, P + 1) for y in range(-P, P + 1)
                    if m1((x, y)) + m1((x - dx, y - dy)) <= P]
        self.lens = lens
        lens_set = set(lens)

        def allowed(c: Cell) -> bool:
            return not (R and c[0] == 0 and c[1] < 0)

        # reach(i): cells the ant can occupy before step i
        reach: List[List[Cell]] = []
        for i in range(P + 1):
            reach.append([c for c in lens
                          if m1(c) <= i and m1((c[0] - dx, c[1] - dy)) <= P - i
                          and (c[0] + c[1]) % 2 == i % 2 and allowed(c)])
        self.reach = reach
        pos: Dict[Tuple[int, Cell], int] = {}
        for i in range(P + 1):
            for c in reach[i]:
                pos[(i, c)] = self.new()
        self.pos = pos
        hd = [[self.new() for _ in range(4)] for _ in range(P + 1)]
        self.hd = hd
        r = [self.new() for _ in range(P)]
        self.r = r
        # colour epochs per cell
        vis: Dict[Cell, List[int]] = {c: [] for c in lens}
        for i in range(P):
            for c in reach[i]:
                vis[c].append(i)
        colv: Dict[Cell, List[int]] = {}
        for c in lens:
            colv[c] = [self.new() for _ in range(len(vis[c]) + 1)]
        self.vis, self.colv = vis, colv

        def col_at(i: int, c: Cell) -> int:
            return colv[c][bisect.bisect_left(vis[c], i)]

        self.col_at = col_at

        # start: ant at origin (heading free), end: at d with the same heading
        self.add(pos[(0, (0, 0))])
        self.add(*hd[0])
        for k in range(4):
            for k2 in range(k + 1, 4):
                self.add(-hd[0][k], -hd[0][k2])
        assert (P, d) in pos
        self.add(pos[(P, d)])
        for k in range(4):
            self.add(-hd[P][k], hd[0][k])
            self.add(hd[P][k], -hd[0][k])

        for i in range(P):
            # heading update: white -> right (k+1), black -> left (k-1)
            for k in range(4):
                self.add(-hd[i][k], r[i], hd[i + 1][(k + 1) % 4])
                self.add(-hd[i][k], -r[i], hd[i + 1][(k + 3) % 4])
            for k in range(4):
                for k2 in range(k + 1, 4):
                    self.add(-hd[i + 1][k], -hd[i + 1][k2])
            reach_next = set(reach[i + 1])
            for c in reach[i]:
                pv = pos[(i, c)]
                cv = col_at(i, c)
                # read
                self.add(-pv, -cv, r[i])
                self.add(-pv, cv, -r[i])
                # flip
                nv = colv[c][bisect.bisect_left(vis[c], i) + 1]
                self.add(-pv, -cv, -nv)
                self.add(-pv, cv, nv)
                self.add(pv, -cv, nv)
                self.add(pv, cv, -nv)
                # move forward
                for k in range(4):
                    nc = (c[0] + DX[k], c[1] + DY[k])
                    if nc in reach_next:
                        self.add(-pv, -hd[i + 1][k], pos[(i + 1, nc)])
                    else:
                        self.add(-pv, -hd[i + 1][k])
            # move backward (keeps position unique)
            for nc in reach[i + 1]:
                nv_ = pos[(i + 1, nc)]
                preds = []
                for k in range(4):
                    pc = (nc[0] - DX[k], nc[1] - DY[k])
                    if (i, pc) in pos:
                        preds.append(pos[(i, pc)])
                        self.add(-nv_, -hd[i + 1][k], pos[(i, pc)])
                    else:
                        self.add(-nv_, -hd[i + 1][k])
                self.add(-nv_, *preds)
        # periodicity: C(q) == colour of q+d after the period, white beyond
        for q in lens:
            qd = (q[0] + dx, q[1] + dy)
            c0 = colv[q][0]
            if qd in lens_set:
                cP = colv[qd][-1]
                self.add(-c0, cP)
                self.add(c0, -cP)
            else:
                self.add(-c0)

    # ------------------------------------------------------------------
    def decode(self, model: List[int]) -> dict:
        val = set(v for v in model if v > 0)
        reads = [1 if self.r[i] in val else 0 for i in range(self.P)]
        h0 = [k for k in range(4) if self.hd[0][k] in val][0]
        C = sorted(q for q in self.lens if self.colv[q][0] in val)
        return {"P": self.P, "d": list(self.d), "h0": h0, "reads": reads, "C": C}

    def block(self, solver, reads: List[int], h0: int) -> None:
        """Forbid this exact (reads, h0) assignment."""
        clause = [-self.hd[0][h0]]
        for i, b in enumerate(reads):
            clause.append(-self.r[i] if b else self.r[i])
        solver.add_clause(clause)


# ----------------------------------------------------------------------
# independent verification and canonical form
# ----------------------------------------------------------------------
def walk_from_reads(reads: List[int], h0: int) -> Tuple[List[Cell], List[int]]:
    """Positions p_0..p_P and headings h_0..h_P implied by the reads."""
    x = y = 0
    h = h0
    ps, hs = [(0, 0)], [h0]
    for b in reads:
        h = (h + 3) % 4 if b else (h + 1) % 4
        x += DX[h]
        y += DY[h]
        ps.append((x, y))
        hs.append(h)
    return ps, hs


def verify(sol: dict) -> bool:
    """Independent check that (reads, h0, d) really is a highway.

    Uses nothing from the SAT model except the read sequence and the initial
    heading.  The walk determines the visit counts N; the only configuration
    consistent with a periodic drift is C(q) = parity of N(q+d) + N(q+2d) + ...
    (cells written by earlier periods).  Simulating the plain ant on C must
    reproduce the reads and end at d with the same heading.  Given that, the
    picture after the period equals C shifted by d on every cell the next
    period can read (the window is convex), so the cycle repeats forever.
    """
    from collections import Counter

    P, (dx, dy), h0, reads = sol["P"], sol["d"], sol["h0"], sol["reads"]
    ps, hs = walk_from_reads(reads, h0)
    if ps[-1] != (dx, dy) or hs[-1] != h0:
        return False
    N = Counter(ps[:-1])
    lens = [(x, y) for x in range(-P, P + 1) for y in range(-P, P + 1)
            if m1((x, y)) + m1((x - dx, y - dy)) <= P]
    C = set()
    for q in lens:
        par, m = 0, 1
        while m * m1((dx, dy)) <= 2 * P:
            par ^= N[(q[0] + m * dx, q[1] + m * dy)] & 1
            m += 1
        if par:
            C.add(q)
    ant = Ant(x=0, y=0, heading=h0, black=C)
    for i in range(P):
        if ant.colour((ant.x, ant.y)) != reads[i]:
            return False
        ant.step()
    return (ant.x, ant.y, ant.heading) == (dx, dy, h0)


def canonical(reads: List[int]) -> str:
    s = "".join(map(str, reads))
    return min(s[i:] + s[:i] for i in range(len(s)))


def primitive_period(reads: List[int]) -> int:
    n = len(reads)
    for k in range(1, n + 1):
        if n % k == 0 and reads == reads[k:] + reads[:k]:
            return k
    return n


# ----------------------------------------------------------------------
# enumeration for one (P, d)
# ----------------------------------------------------------------------
def solve_instance(P: int, d: Cell, radius: int = 0, verbose: bool = False) -> dict:
    from pysat.solvers import Solver

    t0 = time.time()
    inst = Instance(P, d, radius)
    t_build = time.time() - t0
    sols: List[dict] = []
    seen = set()
    with Solver(name="cadical153", bootstrap_with=inst.clauses) as s:
        while s.solve():
            sol = inst.decode(s.get_model())
            reads, h0 = sol["reads"], sol["h0"]
            sol["verified"] = verify(sol)
            sol["canonical"] = canonical(reads)
            sol["primitive_period"] = primitive_period(reads)
            ps, hs = walk_from_reads(reads, h0)
            sol["cells_visited"] = len(set(ps[:-1]))
            if sol["canonical"] not in seen:
                seen.add(sol["canonical"])
                sols.append(sol)
            # block every cyclic shift of this solution (same highway, other phase)
            for sft in range(P):
                shifted = reads[sft:] + reads[:sft]
                inst.block(s, shifted, hs[sft])
    return {
        "P": P, "d": list(d), "radius": radius, "vars": inst.nv, "clauses": len(inst.clauses),
        "lens_cells": len(inst.lens), "build_s": round(t_build, 1),
        "total_s": round(time.time() - t0, 1), "solutions": sols,
    }


def fundamental_drifts(P: int, dmax: int, radius: int = 0) -> List[Cell]:
    """One representative per rotation class: dx >= 1, dy >= 0.

    Unrestricted window: |d|_1 <= dmax.  Box window: every d that fits the
    window (1 <= dx <= radius, 0 <= dy <= radius), which is all of them.
    """
    out = []
    if radius:
        for dx in range(1, radius + 1):
            for dy in range(0, radius + 1):
                if (dx + dy) % 2 == P % 2 and dx + dy <= P:
                    out.append((dx, dy))
        return out
    for dx in range(1, dmax + 1):
        for dy in range(0, dmax + 1 - dx):
            if (dx + dy) % 2 == P % 2 and dx + dy <= P:
                out.append((dx, dy))
    return out


def _job(args):
    P, d, radius = args
    return solve_instance(P, d, radius)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pmax", type=int, default=104)
    ap.add_argument("--pmin", type=int, default=2)
    ap.add_argument("--dmax", type=int, default=8)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--radius", type=int, default=0,
                    help="restrict the footprint to a box of this radius (0 = unrestricted)")
    ap.add_argument("--single", type=int, nargs=3, metavar=("P", "DX", "DY"))
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    if a.single:
        P, dx, dy = a.single
        res = solve_instance(P, (dx, dy), a.radius, verbose=True)
        print(json.dumps({k: v for k, v in res.items() if k != "solutions"}))
        for s in res["solutions"]:
            print("  solution: verified=%s primitive=%d cells=%d h0=%d reads=%s" % (
                s["verified"], s["primitive_period"], s["cells_visited"], s["h0"],
                "".join(map(str, s["reads"]))))
        return

    jobs = [(P, d, a.radius) for P in range(a.pmin, a.pmax + 1, 2)
            for d in fundamental_drifts(P, a.dmax, a.radius)]
    scope = f"footprint box {a.radius + 1}x{a.radius + 1}" if a.radius else f"|d|_1 <= {a.dmax}, unrestricted"
    print(f"{len(jobs)} instances, P in [{a.pmin},{a.pmax}], {scope}", flush=True)
    tag = f"r{a.radius}" if a.radius else "unrestricted"
    log = open(a.out / f"instances_{tag}.jsonl", "a")
    found = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futs = {pool.submit(_job, j): j for j in jobs}
        for n, f in enumerate(as_completed(futs), 1):
            res = f.result()
            log.write(json.dumps(res) + "\n")
            log.flush()
            prim = [s for s in res["solutions"] if s["primitive_period"] == res["P"]]
            found += len(prim)
            flag = f"  <-- {len(prim)} primitive highway(s)!" if prim else ""
            print(f"[{n}/{len(jobs)}] P={res['P']:4d} d={tuple(res['d'])} "
                  f"{res['total_s']:7.1f}s  sols={len(res['solutions'])}{flag}", flush=True)
    print(f"done in {time.time() - t0:.0f}s; primitive highways found: {found}")


if __name__ == "__main__":
    main()
