#!/usr/bin/env python3
"""
Exhaustive (or random) survey of Langton's Ant starting patterns.

Compiles ``ant_core.c``, fans the pattern space out over CPU cores, and
writes per-size summaries into ``data/``:

    data/n{N}_summary.json      counts, extremes, timing, parameters
    data/n{N}_onset_hist.csv    onset_step,count   (exact histogram)
    data/n{N}_full.csv          one row per pattern (only when --full)

Examples:
    python survey.py --n 4 --full            # all 65,536 4x4 patterns
    python survey.py --n 5                   # all 33,554,432 5x5 patterns
    python survey.py --n 8 --random 200000   # 200k random 8x8 patterns

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
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FIELDS = ["index", "status", "onset", "cert_step", "period", "dx", "dy",
          "visited", "bbox_w", "bbox_h", "onset_exact"]


def build(build_dir: Path) -> Path:
    """Compile ant_core.c (if needed) and return the binary path."""
    build_dir.mkdir(parents=True, exist_ok=True)
    exe = build_dir / "ant_core"
    src = HERE / "ant_core.c"
    if not exe.exists() or exe.stat().st_mtime < src.stat().st_mtime:
        cc = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
        if cc is None:
            sys.exit("no C compiler found (need cc/gcc/clang)")
        subprocess.run([cc, "-O2", "-Wall", "-o", str(exe), str(src)], check=True)
    return exe


def run_chunk(exe: Path, n: int, ax: int, ay: int, start: int, end: int,
              max_steps: int, out: Path, seed: int | None) -> Path:
    cmd = [str(exe), str(n), str(ax), str(ay), str(start), str(end), str(max_steps)]
    if seed is not None:
        cmd.append(str(seed))
    with open(out, "w") as fh:
        subprocess.run(cmd, stdout=fh, check=True)
    return out


def aggregate(files: list[Path], full_out: Path | None) -> dict:
    """Stream every result line once and build the summary."""
    status = Counter()
    onset_hist = Counter()
    period = Counter()
    drift = Counter()
    inexact = 0
    n_rows = 0
    fastest = None   # (onset, index)
    slowest = None
    timeouts: list[int] = []
    oob: list[int] = []
    onset_sum = 0
    visited_sum = 0
    full = open(full_out, "w") if full_out else None
    if full:
        full.write(",".join(FIELDS) + "\n")
    for f in files:
        with open(f) as fh:
            for line in fh:
                if full:
                    full.write(line)
                parts = line.rstrip("\n").split(",")
                idx = int(parts[0])
                st = parts[1]
                status[st] += 1
                n_rows += 1
                visited_sum += int(parts[7])
                if st == "HIGHWAY":
                    onset = int(parts[2])
                    onset_hist[onset] += 1
                    onset_sum += onset
                    period[int(parts[4])] += 1
                    drift[f"{parts[5]},{parts[6]}"] += 1
                    if parts[10] == "0":
                        inexact += 1
                    if fastest is None or onset < fastest[0]:
                        fastest = (onset, idx)
                    if slowest is None or onset > slowest[0]:
                        slowest = (onset, idx)
                elif st == "TIMEOUT":
                    if len(timeouts) < 1000:
                        timeouts.append(idx)
                else:
                    if len(oob) < 1000:
                        oob.append(idx)
    if full:
        full.close()
    # median from the exact histogram
    median = None
    if onset_hist:
        total = sum(onset_hist.values())
        acc = 0
        for k in sorted(onset_hist):
            acc += onset_hist[k]
            if acc * 2 >= total:
                median = k
                break
    n_hw = status["HIGHWAY"]
    return {
        "patterns": n_rows,
        "status": dict(status),
        "highway_fraction": n_hw / n_rows if n_rows else None,
        "onset_min": fastest[0] if fastest else None,
        "onset_min_pattern": fastest[1] if fastest else None,
        "onset_max": slowest[0] if slowest else None,
        "onset_max_pattern": slowest[1] if slowest else None,
        "onset_mean": onset_sum / n_hw if n_hw else None,
        "onset_median": median,
        "visited_mean": visited_sum / n_rows if n_rows else None,
        "periods": {str(k): v for k, v in sorted(period.items())},
        "drifts": dict(drift),
        "onset_not_exact": inexact,
        "timeouts_first_1000": timeouts,
        "oob_first_1000": oob,
        "_onset_hist": onset_hist,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, required=True, help="pattern box side (1..8)")
    ap.add_argument("--ant", type=int, nargs=2, metavar=("AX", "AY"),
                    help="ant start cell (default n//2 n//2)")
    ap.add_argument("--max-steps", type=int, default=2_000_000)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--random", type=int, metavar="COUNT",
                    help="sample COUNT random patterns instead of all 2^(n*n)")
    ap.add_argument("--seed", type=int, default=20260920)
    ap.add_argument("--full", action="store_true", help="also write per-pattern CSV")
    ap.add_argument("--out", type=Path, default=DATA)
    ap.add_argument("--build-dir", type=Path, default=HERE / "build")
    a = ap.parse_args()

    n = a.n
    ax, ay = a.ant if a.ant else (n // 2, n // 2)
    total = a.random if a.random else 2 ** (n * n)
    exe = build(a.build_dir)
    a.out.mkdir(parents=True, exist_ok=True)
    tag = f"n{n}" + (f"_random{a.random}" if a.random else "")

    chunks = max(1, min(a.workers * 16, total))
    size = -(-total // chunks)
    tmp = Path(tempfile.mkdtemp(prefix="antsurvey_"))
    t0 = time.time()
    jobs = []
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for c in range(chunks):
            s, e = c * size, min(total, (c + 1) * size)
            if s >= e:
                break
            seed = (a.seed * 1000003 + c) if a.random else None
            jobs.append(pool.submit(run_chunk, exe, n, ax, ay, s, e, a.max_steps,
                                    tmp / f"chunk_{c:04d}.csv", seed))
        files = [j.result() for j in jobs]
    elapsed = time.time() - t0

    full_out = a.out / f"{tag}_full.csv" if a.full else None
    summary = aggregate(files, full_out)
    hist = summary.pop("_onset_hist")
    with open(a.out / f"{tag}_onset_hist.csv", "w") as fh:
        fh.write("onset_step,count\n")
        for k in sorted(hist):
            fh.write(f"{k},{hist[k]}\n")
    summary.update({
        "n": n, "ant_start": [ax, ay], "max_steps": a.max_steps,
        "mode": "random" if a.random else "exhaustive", "seed": a.seed if a.random else None,
        "wall_seconds": round(elapsed, 1), "workers": a.workers,
    })
    with open(a.out / f"{tag}_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    shutil.rmtree(tmp, ignore_errors=True)
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("timeouts_first_1000", "oob_first_1000")}, indent=2))
    print("timeouts:", len(summary["timeouts_first_1000"]), "oob:", len(summary["oob_first_1000"]))


if __name__ == "__main__":
    main()
