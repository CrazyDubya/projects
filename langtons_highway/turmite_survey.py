#!/usr/bin/env python3
"""
Sweep every turmite rule up to a given length and classify its behaviour.

A turmite rule is a string of L/R, one character per colour: on a cell of
colour k the ant turns rule[k], sets the cell to (k+1) mod n, and steps
forward.  "RL" is Langton's Ant.

Swapping every L and R mirrors the trajectory without changing its class, so
only rules beginning with 'R' are run; the mirror is implied.

Writes data/turmites/rules.csv (one row per rule) and data/turmites/summary.json.

    python turmite_survey.py --max-len 12 --max-steps 20000000

Author: Claude (with Stephen)
Created: 2026-09-20
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "data" / "turmites"
FIELDS = ["rule", "status", "onset", "cert_step", "period", "dx", "dy",
          "visited", "bbox_w", "bbox_h", "onset_exact"]


def build(build_dir: Path) -> Path:
    build_dir.mkdir(parents=True, exist_ok=True)
    exe = build_dir / "turmite_core"
    src = HERE / "turmite_core.c"
    if not exe.exists() or exe.stat().st_mtime < src.stat().st_mtime:
        cc = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
        if cc is None:
            sys.exit("no C compiler found")
        subprocess.run([cc, "-O2", "-Wall", "-o", str(exe), str(src)], check=True)
    return exe


def rules_up_to(max_len: int):
    """Every rule of length 1..max_len starting with 'R' (mirrors implied)."""
    for n in range(1, max_len + 1):
        for tail in product("RL", repeat=n - 1):
            yield "R" + "".join(tail)


def run_rule(exe: Path, rule: str, max_steps: int) -> list[str]:
    out = subprocess.run([str(exe), rule, str(max_steps)],
                         capture_output=True, text=True, check=True).stdout
    return out.strip().split(",")


def summarize(rows: list[list[str]], max_len: int, max_steps: int) -> dict:
    """Aggregate result rows, keeping every caveat the core reported."""
    status = Counter(r[1] for r in rows)
    by_len = defaultdict(Counter)
    for r in rows:
        by_len[len(r[0])][r[1]] += 1
    hw = [r for r in rows if r[1] == "HIGHWAY"]
    periods = Counter(int(r[4]) for r in hw)
    drifts = Counter(f"{r[5]},{r[6]}" for r in hw)
    # An onset is exact only if the backward scan stopped on a genuine
    # mismatch.  When it ran out of ring buffer it stopped early, and the
    # reported figure is an upper bound on the true onset, not a measurement.
    inexact = [r[0] for r in hw if r[10] == "0"]
    exact_onsets = [int(r[2]) for r in hw if r[10] == "1"]
    return {
        "rules_tested": len(rows),
        "max_len": max_len,
        "max_steps": max_steps,
        "status": dict(status),
        "highway_fraction": len(hw) / len(rows) if rows else None,
        "periods": {str(k): v for k, v in sorted(periods.items())},
        "drifts": dict(drifts),
        "by_length": {str(k): dict(v) for k, v in sorted(by_len.items())},
        "onset_inexact_rules": sorted(inexact),
        "largest_exact_onset": max(exact_onsets) if exact_onsets else None,
        "highways": sorted(
            ({"rule": r[0], "period": int(r[4]), "drift": [int(r[5]), int(r[6])],
              "onset": int(r[2]), "onset_exact": r[10] == "1",
              # column 7 is the cumulative count of distinct cells visited by
              # certification time, not the footprint of a single period
              "visited_at_certification": int(r[7])} for r in hw),
            key=lambda d: (d["period"], d["rule"])),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-len", type=int, default=12)
    ap.add_argument("--max-steps", type=int, default=20_000_000)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--summarize-only", action="store_true",
                    help="rebuild summary.json from an existing rules.csv")
    a = ap.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)
    if a.summarize_only:
        with open(a.out / "rules.csv") as fh:
            rows = [line.rstrip("\n").split(",") for line in fh.readlines()[1:] if line.strip()]
        summary = summarize(rows, a.max_len, a.max_steps)
        (a.out / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps({k: v for k, v in summary.items() if k != "highways"}, indent=2))
        return
    exe = build(HERE / "build")
    rules = list(rules_up_to(a.max_len))
    print(f"{len(rules)} rules (lengths 1..{a.max_len}, mirrors implied), "
          f"budget {a.max_steps:,} steps", flush=True)

    rows = []
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for i, row in enumerate(pool.map(lambda r: run_rule(exe, r, a.max_steps), rules), 1):
            rows.append(row)
            if row[1] == "HIGHWAY":
                print(f"[{i}/{len(rules)}] {row[0]:>13}  HIGHWAY  period {row[4]:>5} "
                      f"drift ({row[5]},{row[6]})  onset {int(row[2]):,}", flush=True)
            elif i % 250 == 0:
                print(f"[{i}/{len(rules)}] ...", flush=True)

    with open(a.out / "rules.csv", "w") as fh:
        fh.write(",".join(FIELDS) + "\n")
        for r in rows:
            fh.write(",".join(r) + "\n")

    summary = summarize(rows, a.max_len, a.max_steps)
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "highways"}, indent=2))
    print(f"{len(summary['highways'])} rules certify a highway")


if __name__ == "__main__":
    main()
