#!/usr/bin/env python3
"""
How hard is it to trap a highway?

Let the ant build its highway from an empty grid, then drop an obstacle in the
road ahead and see what happens: does it fall back into chaos, does it come
out the other side, how long does that take, and which way does it leave?

A counterexample to the highway conjecture would have to be a configuration
that keeps reabsorbing its own highway forever, so the empirical question
"how hard is it to trap one" is the closest thing to a direct probe.

Soundness note: the certificate treats everything outside the visited
bounding box as background.  An obstacle placed beyond that box would break
the assumption, so the box is widened to cover every obstacle cell before the
run continues.

    python trap.py --out data/trapping.json

Author: Claude (with Stephen)
Created: 2026-09-20
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Optional, Set, Tuple

from ant import Ant

HERE = Path(__file__).resolve().parent
Cell = Tuple[int, int]


def established_highway(lead_periods: int = 6) -> Tuple[Ant, dict, int]:
    """Run from empty until the highway is certified, plus a few more periods."""
    ant = Ant()
    res = ant.find_highway(200_000)
    assert res is not None
    ant.run(res["period"] * lead_periods)
    return ant, res, ant.step_no


def place(ant: Ant, cells: Set[Cell]) -> None:
    """Add obstacle cells and widen the bounding box to keep the proof sound."""
    for c in cells:
        ant.black.add(c)
    b = ant.bbox
    for (x, y) in cells:
        b[0] = min(b[0], x)
        b[1] = max(b[1], x)
        b[2] = min(b[2], y)
        b[3] = max(b[3], y)


def obstacle(shape: str, centre: Cell, along: Cell, perp: Cell, size: int) -> Set[Cell]:
    cx, cy = centre
    if shape == "cell":
        return {(cx, cy)}
    if shape == "block":
        return {(cx + i, cy + j) for i in range(size) for j in range(size)}
    if shape == "wall":                      # across the road
        half = size // 2
        return {(cx + perp[0] * k, cy + perp[1] * k) for k in range(-half, half + 1)}
    if shape == "spike":                     # along the road
        return {(cx + along[0] * k, cy + along[1] * k) for k in range(size)}
    raise ValueError(shape)


def run_to_contact(ant: Ant, cells: Set[Cell], budget: int) -> Optional[int]:
    """Step until the ant stands on an obstacle cell; None if it never does."""
    target = set(cells)
    limit = ant.step_no + budget
    while ant.step_no < limit:
        if (ant.x, ant.y) in target:
            return ant.step_no
        ant.step()
    return None


def trial(m: int, lateral: int, shape: str, size: int, budget: int) -> dict:
    ant, res, t0 = established_highway()
    d = (res["dx"], res["dy"])                       # drift per period
    ux, uy = (1 if d[0] > 0 else -1), (1 if d[1] > 0 else -1)
    perp = (ux, -uy)                                  # across the direction of travel
    centre = (ant.x + d[0] * m + perp[0] * lateral,
              ant.y + d[1] * m + perp[1] * lateral)
    cells = obstacle(shape, centre, (ux, uy), perp, size)
    already_black = {c for c in cells if c in ant.black}
    place(ant, cells)

    contact = run_to_contact(ant, cells, budget // 4)
    out = {
        "periods_ahead": m, "lateral": lateral, "shape": shape, "size": size,
        "obstacle_cells": len(cells), "already_black": len(already_black),
        "placed_at_step": t0, "contact_step": contact,
    }
    if contact is None:                               # the road missed it entirely
        out.update({"outcome": "missed", "recovery_steps": None,
                    "new_drift": None, "same_direction": None, "period": None})
        return out

    after = ant.find_highway(contact + budget)
    if after is None:
        out.update({"outcome": "never_recovered", "recovery_steps": None,
                    "new_drift": None, "same_direction": None, "period": None})
        return out
    rec = after["onset"] - contact
    out.update({
        "outcome": "recovered" if rec > 0 else "undisturbed",
        "recovery_steps": rec,
        "new_drift": [after["dx"], after["dy"]],
        "same_direction": (after["dx"], after["dy"]) == d,
        "period": after["period"],
    })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--budget", type=int, default=2_000_000)
    ap.add_argument("--random-trials", type=int, default=120)
    ap.add_argument("--seed", type=int, default=20260920)
    ap.add_argument("--out", type=Path, default=HERE / "data" / "trapping.json")
    a = ap.parse_args()

    trials: List[dict] = []
    # structured sweep
    for shape, size in (("cell", 1), ("block", 2), ("wall", 9), ("wall", 21), ("spike", 8)):
        for m in (2, 5, 10, 20, 40):
            for lateral in (-6, -3, 0, 3, 6):
                t = trial(m, lateral, shape, size, a.budget)
                trials.append(t)
                flag = "   <-- NEVER RECOVERED" if t["outcome"] == "never_recovered" else ""
                print(f"{shape:>5}/{size:<2} m={m:<3} lat={lateral:<3} "
                      f"{t['outcome']:<15} "
                      f"steps={t['recovery_steps'] if t['recovery_steps'] is not None else '-':>8} "
                      f"drift={t['new_drift']}{flag}", flush=True)

    # random scatter of small obstacles in the road
    rng = random.Random(a.seed)
    for _ in range(a.random_trials):
        ant, res, t0 = established_highway()
        d = (res["dx"], res["dy"])
        ux, uy = (1 if d[0] > 0 else -1), (1 if d[1] > 0 else -1)
        perp = (ux, -uy)
        cells = set()
        for _ in range(rng.randint(1, 12)):
            m = rng.randint(2, 40)
            lat = rng.randint(-8, 8)
            cells.add((ant.x + d[0] * m + perp[0] * lat, ant.y + d[1] * m + perp[1] * lat))
        place(ant, cells)
        contact = run_to_contact(ant, cells, a.budget // 4)
        row = {"periods_ahead": None, "lateral": None, "shape": "random",
               "size": len(cells), "obstacle_cells": len(cells), "already_black": None,
               "placed_at_step": t0, "contact_step": contact}
        if contact is None:
            row.update({"outcome": "missed", "recovery_steps": None,
                        "new_drift": None, "same_direction": None, "period": None})
        else:
            after = ant.find_highway(contact + a.budget)
            if after is None:
                row.update({"outcome": "never_recovered", "recovery_steps": None,
                            "new_drift": None, "same_direction": None, "period": None})
            else:
                r = after["onset"] - contact
                row.update({"outcome": "recovered" if r > 0 else "undisturbed",
                            "recovery_steps": r, "new_drift": [after["dx"], after["dy"]],
                            "same_direction": (after["dx"], after["dy"]) == d,
                            "period": after["period"]})
        trials.append(row)

    from collections import Counter
    rec = [t for t in trials if t["outcome"] == "recovered"]
    steps = sorted(t["recovery_steps"] for t in rec)
    summary = {
        "trials": len(trials),
        "outcomes": dict(Counter(t["outcome"] for t in trials)),
        "recovered": len(rec),
        "never_recovered": sum(1 for t in trials if t["outcome"] == "never_recovered"),
        "budget": a.budget,
        "recovery_steps_min": steps[0] if steps else None,
        "recovery_steps_median": steps[len(steps) // 2] if steps else None,
        "recovery_steps_max": steps[-1] if steps else None,
        "periods_seen": sorted({t["period"] for t in rec}),
        "kept_same_direction": sum(1 for t in rec if t["same_direction"]),
        "turned_elsewhere": sum(1 for t in rec if not t["same_direction"]),
        "trials_detail": trials,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "trials_detail"}, indent=2))


if __name__ == "__main__":
    main()
