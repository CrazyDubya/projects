#!/usr/bin/env python3
"""
Pure-Python Langton's Ant with a machine-checked highway certificate.

This is the slow, readable reference implementation.  ``ant_core.c`` is the
fast one used for the exhaustive survey; ``tests/`` cross-check the two.

Rule (the whole rule):
    on a WHITE cell: turn right, flip the cell to black, step forward
    on a BLACK cell: turn left,  flip the cell to white, step forward

The "highway" is a 104-step cycle after which the whole local picture has
moved by (+-2, +-2).  ``certify`` checks a sufficient condition for that
cycle to continue forever; the argument is written out in README.md.

Author: Claude (with Stephen)
Created: 2026-09-20
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

DX = (0, 1, 0, -1)  # N, E, S, W
DY = (1, 0, -1, 0)

Cell = Tuple[int, int]


def pattern_cells(n: int, index: int) -> List[Cell]:
    """Black cells of the n x n starting pattern with the given index.

    Bit ``b`` of ``index`` (0 = least significant) sets cell
    ``(x = b % n, y = b // n)`` black.
    """
    return [(b % n, b // n) for b in range(n * n) if (index >> b) & 1]


@dataclass
class Ant:
    """A Langton's Ant on an unbounded sparse grid."""

    x: int = 0
    y: int = 0
    heading: int = 0  # 0=N 1=E 2=S 3=W
    black: set = field(default_factory=set)
    step_no: int = 0
    # per-cell bookkeeping used by the certificate
    first_visit: Dict[Cell, int] = field(default_factory=dict)
    last_write: Dict[Cell, int] = field(default_factory=dict)
    # per-step history: (x, y, colour_read, heading, prev_last_write, fresh)
    history: List[Tuple[int, int, int, int, int, bool]] = field(default_factory=list)
    bbox: List[int] = field(default_factory=lambda: [0, 0, 0, 0])  # minx maxx miny maxy

    @classmethod
    def from_pattern(cls, n: int, index: int, ax: Optional[int] = None,
                     ay: Optional[int] = None) -> "Ant":
        """Ant on an n x n pattern; default start is cell (n//2, n//2)."""
        ax = n // 2 if ax is None else ax
        ay = n // 2 if ay is None else ay
        ant = cls(x=ax, y=ay, black=set(pattern_cells(n, index)))
        ant.bbox = [0, n - 1, 0, n - 1]
        return ant

    def colour(self, c: Cell) -> int:
        return 1 if c in self.black else 0

    def step(self) -> None:
        """Advance one step, recording the history the certificate needs."""
        c = (self.x, self.y)
        col = self.colour(c)
        fresh = c not in self.first_visit
        self.history.append(
            (self.x, self.y, col, self.heading, self.last_write.get(c, -1), fresh)
        )
        if fresh:
            self.first_visit[c] = self.step_no
            b = self.bbox
            b[0] = min(b[0], self.x)
            b[1] = max(b[1], self.x)
            b[2] = min(b[2], self.y)
            b[3] = max(b[3], self.y)
        self.last_write[c] = self.step_no
        if col:
            self.black.discard(c)
            self.heading = (self.heading + 3) & 3
        else:
            self.black.add(c)
            self.heading = (self.heading + 1) & 3
        self.x += DX[self.heading]
        self.y += DY[self.heading]
        self.step_no += 1

    def run(self, steps: int) -> None:
        for _ in range(steps):
            self.step()

    def frames(self, steps: int) -> Iterator["Ant"]:
        """Yield self after every step (for renderers)."""
        for _ in range(steps):
            self.step()
            yield self

    # ------------------------------------------------------------------
    # highway certificate
    # ------------------------------------------------------------------
    def _pos(self, t: int) -> Tuple[int, int, int]:
        """(x, y, heading) at the start of step t (t may equal step_no)."""
        if t == self.step_no:
            return self.x, self.y, self.heading
        h = self.history[t]
        return h[0], h[1], h[3]

    def drift_hint(self, period: int = 104, repeats: int = 2) -> Optional[Cell]:
        """Cheap heuristic: constant non-zero displacement per period."""
        t = self.step_no
        if t < (repeats + 1) * period:
            return None
        x0, y0, _ = self._pos(t)
        x1, y1, _ = self._pos(t - period)
        d = (x0 - x1, y0 - y1)
        if d == (0, 0):
            return None
        for k in range(1, repeats + 1):
            xa, ya, _ = self._pos(t - k * period)
            xb, yb, _ = self._pos(t - (k + 1) * period)
            if (xa - xb, ya - yb) != d:
                return None
        return d

    def certify(self, period: int) -> Optional[dict]:
        """Check conditions (A) and (B) for the last ``period`` steps.

        Returns a dict with drift/onset on success, or None.
        """
        s = self.step_no
        P = period
        a, b = s - P, s - 2 * P
        if b < 0:
            return None
        H = self.history
        xa, ya, ha = self._pos(a)
        xb, yb, hb = self._pos(b)
        dx, dy = xa - xb, ya - yb
        if (dx, dy) == (0, 0) or ha != hb:
            return None
        # (A) exact translate
        for i in range(P):
            u, v = H[a + i], H[b + i]
            if u[2] != v[2] or u[0] - v[0] != dx or u[1] - v[1] != dy:
                return None
        # (B)
        minx, maxx, miny, maxy = self.bbox
        for u in range(a, s):
            x, y, col, _, prevw, fresh = H[u]
            if fresh:
                if col != 0:
                    return None
                qx, qy = x, y
                while True:
                    qx += dx
                    qy += dy
                    if qx < minx or qx > maxx or qy < miny or qy > maxy:
                        break
                    if (qx, qy) in self.first_visit or (qx, qy) in self.black:
                        return None
            elif prevw < b:
                return None
        # onset
        t = a
        while t > 0:
            u, v = H[t - 1], H[t - 1 + P]
            if u[2] != v[2] or u[3] != v[3] or v[0] - u[0] != dx or v[1] - u[1] != dy:
                break
            t -= 1
        return {"onset": t, "cert_step": s, "period": P, "dx": dx, "dy": dy}

    def find_highway(self, max_steps: int, period: int = 104) -> Optional[dict]:
        """Run until the highway certificate holds or max_steps is reached."""
        while self.step_no < max_steps:
            self.step()
            if self.drift_hint(period) is not None:
                r = self.certify(period)
                if r is not None:
                    return r
        return None


def steps_to_highway(n: int, index: int, max_steps: int = 2_000_000) -> Optional[dict]:
    """Convenience wrapper used by tests and small experiments."""
    return Ant.from_pattern(n, index).find_highway(max_steps)


if __name__ == "__main__":
    import sys

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    index = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    print(steps_to_highway(n, index))
