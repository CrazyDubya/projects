#!/usr/bin/env python3
"""
The time-reversed Langton's Ant.

The forward map on (grid, position, heading) is a bijection, so the ant has a
unique past as well as a unique future.  This module implements the inverse
step and asks two questions:

  1. Started from an empty grid, does the *backward* dynamics also build a
     highway?
  2. Is the backward picture related to the forward picture by a symmetry?

Run:  python backward.py

Author: Claude (with Stephen)
Created: 2026-09-20
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from ant import Ant, DX, DY

Cell = Tuple[int, int]


def back_step(x: int, y: int, h: int, black: Set[Cell]) -> Tuple[int, int, int]:
    """Invert one forward step; returns the state before it."""
    p = (x - DX[h], y - DY[h])          # the ant stood here and stepped to (x, y)
    c_after = 1 if p in black else 0
    c_before = 1 - c_after               # that step flipped the cell
    if c_before:
        black.add(p)
    else:
        black.discard(p)
    # white -> turned right, black -> turned left
    h_prev = (h - 1) & 3 if c_before == 0 else (h + 1) & 3
    return p[0], p[1], h_prev


def run_backward(steps: int) -> Tuple[List[Tuple[int, int, int]], Set[Cell]]:
    """Iterate the inverse step from an empty grid; returns states and grid."""
    x = y = h = 0
    black: Set[Cell] = set()
    states = [(x, y, h)]
    for _ in range(steps):
        x, y, h = back_step(x, y, h, black)
        states.append((x, y, h))
    return states, black


def drift_period(states, max_period: int = 512, repeats: int = 4) -> Optional[dict]:
    """Smallest period whose displacement is constant and non-zero."""
    t = len(states) - 1
    for p in range(1, max_period + 1):
        if t < (repeats + 1) * p:
            break
        x0, y0, h0 = states[t]
        x1, y1, h1 = states[t - p]
        d = (x0 - x1, y0 - y1)
        if d == (0, 0) or h0 != h1:
            continue
        if all((states[t - k * p][0] - states[t - (k + 1) * p][0],
                states[t - k * p][1] - states[t - (k + 1) * p][1]) == d
               for k in range(1, repeats + 1)):
            return {"period": p, "drift": d}
    return None


TRANSFORMS: Dict[str, callable] = {
    "identity": lambda c: (c[0], c[1]),
    "rot90": lambda c: (-c[1], c[0]),
    "rot180": lambda c: (-c[0], -c[1]),
    "rot270": lambda c: (c[1], -c[0]),
    "flip_x": lambda c: (-c[0], c[1]),
    "flip_y": lambda c: (c[0], -c[1]),
    "transpose": lambda c: (c[1], c[0]),
    "anti_transpose": lambda c: (-c[1], -c[0]),
}


def match_symmetry(back: Set[Cell], fwd: Set[Cell], span: int = 3) -> Optional[str]:
    """Find a dihedral transform (plus small shift) carrying back onto fwd."""
    if len(back) != len(fwd):
        return None
    for name, f in TRANSFORMS.items():
        t = {f(c) for c in back}
        for ox in range(-span, span + 1):
            for oy in range(-span, span + 1):
                if {(c[0] + ox, c[1] + oy) for c in t} == fwd:
                    return f"{name} shifted by ({ox},{oy})"
    return None


def backward_onset(states, d: dict) -> int:
    """Earliest step from which the trajectory is periodic with the given drift.

    The scan has to start a full period before the end, otherwise the first
    partner index states[t - 1 + period] is already past the end of the list.
    """
    p, (dx, dy) = d["period"], d["drift"]
    t = len(states) - 1 - p
    while t > 0:
        a, b = states[t - 1], states[t - 1 + p]
        if (b[0] - a[0], b[1] - a[1]) != (dx, dy) or a[2] != b[2]:
            break
        t -= 1
    return t


def main() -> None:
    N = 20000
    states, back_black = run_backward(N)
    fwd = Ant()
    fwd.run(N)

    print(f"backward from an empty grid, {N:,} steps")
    print(f"  cells written   : {len(back_black):,}   (forward: {len(fwd.black):,})")
    xs = [s[0] for s in states]
    ys = [s[1] for s in states]
    fx = [h[0] for h in fwd.history] + [fwd.x]
    fy = [h[1] for h in fwd.history] + [fwd.y]
    print(f"  bounding box    : {max(xs)-min(xs)+1} x {max(ys)-min(ys)+1}"
          f"   (forward: {max(fx)-min(fx)+1} x {max(fy)-min(fy)+1})")

    d = drift_period(states)
    print(f"  highway?        : {d if d else 'no constant drift found'}")
    fd = drift_period([(h[0], h[1], h[3]) for h in fwd.history] + [(fwd.x, fwd.y, fwd.heading)])
    print(f"  forward highway : {fd}")

    sym = match_symmetry(back_black, fwd.black)
    print(f"  symmetry to fwd : {sym if sym else 'none among the 8 dihedral maps'}")

    # where does the backward highway first lock in?
    if d:
        print(f"  backward onset  : {backward_onset(states, d):,}")


if __name__ == "__main__":
    main()
