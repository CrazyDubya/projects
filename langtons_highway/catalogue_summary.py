#!/usr/bin/env python3
"""
Summarise highway_catalogue.py runs into data/catalogue/summary.json.

Reads every data/catalogue/instances_*.jsonl, checks that every solution
passed independent verification, groups solutions by canonical read
sequence, and records the exact scope (periods, drifts, window) that was
searched exhaustively.

Author: Claude (with Stephen)
Created: 2026-09-20
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAT = HERE / "data" / "catalogue"


def main() -> None:
    summary = {"runs": {}, "highways": {}}
    for path in sorted(CAT.glob("instances_*.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if not rows:
            continue
        tag = path.stem.replace("instances_", "")
        by_p = defaultdict(list)
        for r in rows:
            by_p[r["P"]].append(r)
        run = {
            "instances": len(rows),
            "periods": sorted(by_p),
            "drift_l1_max": max(abs(r["d"][0]) + abs(r["d"][1]) for r in rows),
            "radius": rows[0].get("radius", 0),
            "solve_seconds": round(sum(r["total_s"] for r in rows)),
            "slowest_instance_seconds": max(r["total_s"] for r in rows),
            "solutions": sum(len(r["solutions"]) for r in rows),
            "unverified_solutions": sum(1 for r in rows for s in r["solutions"] if not s["verified"]),
        }
        # completeness: every (P, d) expected for the scope must be present
        missing = []
        for P in run["periods"]:
            have = {tuple(r["d"]) for r in by_p[P]}
            if run["radius"]:
                R = run["radius"]
                want = {(dx, dy) for dx in range(1, R + 1) for dy in range(0, R + 1)
                        if (dx + dy) % 2 == P % 2 and dx + dy <= P}
            else:
                D = run["drift_l1_max"]
                want = {(dx, dy) for dx in range(1, D + 1) for dy in range(0, D + 1 - dx)
                        if (dx + dy) % 2 == P % 2 and dx + dy <= P}
            missing += [(P, d) for d in sorted(want - have)]
        run["missing_instances"] = missing
        summary["runs"][tag] = run
        for r in rows:
            for s in r["solutions"]:
                if s["primitive_period"] != r["P"]:
                    continue
                key = s["canonical"]
                h = summary["highways"].setdefault(key, {
                    "period": r["P"], "cells_per_period": s["cells_visited"],
                    "black_reads": sum(s["reads"]), "found_in": [], "verified": True})
                h["found_in"].append({"run": tag, "d": r["d"], "h0": s["h0"]})
                h["verified"] = h["verified"] and s["verified"]
    (CAT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
