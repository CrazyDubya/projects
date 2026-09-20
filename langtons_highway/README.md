# Langton's Highway

*A two-rule ant, a 40-year-old open problem, 33.6 million brute-force
experiments, and a short vertical video about it.*

**Video:** [`output/langtons_highway.mp4`](output/langtons_highway.mp4)
(1080×1920, 80 s, made entirely from the code in this folder).

## The mystery

Langton's Ant (Chris Langton, 1986) lives on an infinite grid of white cells
and follows two rules:

* on a **white** cell: turn right, flip the cell to black, step forward;
* on a **black** cell: turn left, flip the cell to white, step forward.

From an empty grid it scribbles for about ten thousand steps, symmetric at
first and then apparently chaotic, and then, at step **9,977**, locks into a
104-step cycle that carries it diagonally away forever: the "highway".

What is *proved* (Bunimovich & Troubetzkoy 1992; the "Cohen–Kong theorem"):
the ant's path is always unbounded, so it can never be trapped in a finite
region, whatever finite pattern of black cells you start it on.

What is *not* proved: that it always ends up on the highway.  Every finite
starting pattern anyone has ever tried produces the same 104-step highway
eventually.  Nobody has a proof, and nobody has a counterexample.  That is
the problem I picked.

## What I did

1. Wrote a fast C simulator (`ant_core.c`, ~100 M steps/s per core) and a
   slow, readable Python reference (`ant.py`) and cross-checked them.
2. Wrote a **highway certificate**: a sufficient condition, checked by the
   program, that proves the ant will repeat the cycle forever from a given
   step.  "It looked periodic for a while" is not enough; see the proof
   below.
3. Ran the certificate on **every** starting pattern inside a 3×3, 4×4 and
   5×5 box (ant starting at the box centre, heading north), with a cap of
   2,000,000 steps per pattern.

## Results

| box | patterns (all of them) | reached a certified highway | fastest onset | slowest onset | median | wall time (4 cores) |
|-----|------------------------|-----------------------------|---------------|---------------|--------|--------------------|
| 3×3 | 512 | **512 (100%)** | 200 steps | 43,264 steps | 2,374 | 2 s |
| 4×4 | 65,536 | **65,536 (100%)** | 155 steps | 119,673 steps | 1,736 | 2 s |
| 5×5 | 33,554,432 | **33,554,432 (100%)** | 32 steps | 233,232 steps | 2,059 | 334 s |

* Every single one of the 33,620,480 patterns ended on the **same** highway:
  period 104, drift (±2, ±2), split almost evenly over the four diagonal
  directions.
* No pattern needed anywhere near the 2,000,000-step cap: the slowest
  (5×5 index 31,064,692) wandered for 233,232 steps first.
* Most starts reach the highway *faster* than the empty grid does: the
  median is around 2,000 steps versus 9,977 for the empty grid.

This is not a proof of the highway conjecture.  It is a fairly strong piece
of evidence that the conjecture has no small counterexample, and each
individual case is a genuine proof (the certificate) rather than a
"looked periodic" heuristic.

Data: `data/n{3,4,5}_summary.json` (counts, extremes, drift split),
`data/n{3,4,5}_onset_hist.csv` (exact histogram of onset step), and full
per-pattern tables for 3×3 and 4×4 (`data/n{3,4}_full.csv`).  The 5×5
per-pattern table is 1.5 GB and is not committed; regenerate it with
`python survey.py --n 5 --full`.

## The certificate (why "forever" is justified)

Fix a period length `P` (104 for every case found).  At step `s`, with
`d = pos(s) − pos(s−P)`, define period *j* as steps `[s−P, s)` and period
*j−1* as `[s−2P, s−P)`.  The program checks:

* **(A)** period *j* is an exact translate of period *j−1* by `d ≠ 0`: the
  colours read at corresponding steps are equal, the headings are equal, and
  the positions differ by exactly `d`.
* **(B)** for every step `u` in period *j* that reads cell `c`, one of:
  * **(B1)** `c` had never been visited before `u`, it was white, and every
    cell `c + k·d` (`k ≥ 1`) that lies inside the bounding box of all cells
    visited so far is unvisited and white;
  * **(B2)** the last write to `c` before `u` happened at some step
    `≥ s−2P` (i.e. in period *j−1* or *j*).

**Claim.** If (A) and (B) hold, period *j+1* is a translate of period *j*
by `d`, and by induction every later period is too: the ant is on a highway.

**Proof sketch.** The ant's moves are determined by the colours it reads, so
it suffices to show that each read in period *j+1* (of cell `c+d`, by
induction over the steps of the period) sees the colour the corresponding
read of `c` saw in period *j*.

* If `c` fell under (B2): the last write to `c` before the read was in
  period *j−1* or *j*; by (A) and the inner induction its translate is a
  write to `c+d` in period *j* or *j+1*, and every other write to `c+d`
  before the read is either an earlier translate or older still, so the
  last write to `c+d` is the translate and the colours agree.
* If `c` fell under (B1): `c+d` was never written before the read (not
  before period *j−1*, because every cell on the forward ray is unvisited
  or outside the bounding box of everything visited; not in periods
  *j−1*…*j+1*, because those writes are translates of writes to `c−d`,
  `c`, and again `c`, none of which happened), so it is still in its initial
  colour, which is white because the ray is white or outside the initial
  pattern.

For the next period the conditions recur: (A) holds by the replay, (B2)
survives because the translate of a "recent" write is again recent, and
(B1) survives because the forward ray of `c+d` is the forward ray of `c`
shifted, which is unvisited, and the cells the period *j+1* footprint adds
are translates of period *j*'s footprint, which the ray already avoids. ∎

Both implementations (`ant.py::Ant.certify`, `ant_core.c::certify`)
implement exactly this check and agree on every pattern in
`tests/test_ant.py` and the 812-pattern cross-check I ran while developing.

**"Onset"** is defined as the earliest step from which the read/heading/
position sequence is `P`-periodic with drift `d`, found by walking backwards
from the certified period.  For the empty grid that is step 9,977.

## The catalogue: is the 104-highway the *only* highway?

The brute-force survey above can only ever say "every seed we tried ended
on the classic highway".  A stronger question is whether any *other*
highway shape exists at all: a periodic-with-drift orbit of the ant on a
white background, of any period, that is not the classic one.  Such an
orbit would be reachable from a finite starting pattern (its own
truncated trail), so finding one would refute the strong form of the
conjecture ("the 104-highway is the unique attractor"), and proving none
exists below some size narrows what a counterexample can look like.

`highway_catalogue.py` turns "period-P orbit with drift d" into SAT: one-hot
ant position and heading per step, per-cell colour epochs, the read / flip /
move rules, and the periodicity constraint *colour of q+d after the period
= colour of q before it, white beyond the window*.  CaDiCaL enumerates every
solution (blocking all cyclic shifts of each), and every solution is
re-verified with nothing but its read sequence: the walk determines the only
consistent starting configuration (parity of visits along the forward ray
q+d, q+2d, ...), and the plain ant from `ant.py` must replay the reads from
it.  Convexity of the window then makes the cycle repeat forever.

**Result.**  Within the searched scope there is exactly one highway, and it
is the classic one (period 104, drift (2,2) up to rotation, 41 cells per
period, 46 black reads).

| scope | instances | solve time | highways found |
|-------|-----------|------------|----------------|
| every even P <= 50, every drift with \|d\|_1 <= 6, **unrestricted** window (any walk that returns to d) | 284 | 35 min | 0 |
| every even P <= 104, **every** drift, one-period footprint fitting a **9 x 9** box | 1,754 | 2.4 h | **1: the classic highway** |

The (P, d) grid is complete for both scopes (`catalogue_summary.py` checks
it), no solution failed verification, and the box-window run reproduces the
classic highway exactly where it must (P = 104, d = (2, 2)), which is the
encoding's end-to-end check.

What this does and does not say: any highway with period at most 104 whose
one-period footprint fits in a 9 x 9 box *is* the classic highway.  A
counterexample to the strong conjecture must therefore have a longer period,
or a footprint wider than 9 cells, or both.  The unrestricted window was only
affordable to period 50; the exponential cost of the full window (about x1.2
per unit of period) is what forces the box.  Raising the box to 11 x 11 or
the period past 104 is a matter of machine time, not new ideas.

Data: `data/catalogue/instances_*.jsonl` (one line per (P, d) instance with
timing and any solutions) and `data/catalogue/summary.json`.

## Four follow-up experiments

The survey and the catalogue both ask about *this* rule from *this* kind of
start.  These four ask sideways questions, and two of them corrected
something I had previously said.

### 1. Is Langton's rule special?  (`turmite_survey.py`)

A turmite rule is a string of L/R, one character per colour: on a cell of
colour k the ant turns `rule[k]`, sets the cell to `(k+1) mod n`, and steps
forward.  `RL` is Langton's Ant.  The certificate generalises unchanged, with
"white" becoming "colour 0", because a cell's colour is still its visit count
mod n.  Swapping every L and R mirrors the trajectory, so only rules starting
with `R` are run.

All 4,095 rules of length 1 to 12, 20 million steps each:

| outcome | rules | |
|---------|-------|--|
| certified highway | **468** | 11.4% |
| no highway within budget | 3,365 | |
| grew off the grid | 250 | |
| proved periodic (a cycle) | 12 | |

**Langton's rule is not special.**  The highway fraction sits near 11% at
every rule length from 6 to 12.  There are 135 distinct highway periods, the
longest 1,968; period 104 is shared by just 11 rules, and the most common
period is 18 (128 rules).  Of the 468 highway rules, 306 (65%) reach their
highway faster than `RL` does; `RRL` gets there in 39 steps against Langton's
9,977.  Drifts are usually diagonal but not always: (16,0), (8,0), (0,2),
(12,-8) and (-3,11) all occur.

"Bounded" is proved, not guessed.  An earlier version called a rule bounded
when it stopped visiting new cells, which wrongly flagged `LRRRRRLLR` after it
had densely filled a 1468x1468 region and was still growing.  The CYCLE class
now comes from Brent's algorithm over an incremental hash of the whole grid,
so it means the exact state recurred.

**The budget caveat, tested.**  Certified onsets run as high as 19,171,329
steps against a 20,000,000 budget, so the cutoff lies *inside* the onset
distribution and "no highway" can only ever be an upper bound.  To measure how
badly, 150 randomly chosen timed-out rules were rerun at ten times the budget
(200 million steps): **none converted** (141 still timed out, 9 left the
grid).  The 11.4% figure therefore looks stable rather than an artifact, but
it remains a sample, and a rule needing 10^9 steps would still be misfiled.

### 2. How hard is it to trap a highway?  (`trap.py`)

A counterexample would have to be a configuration that keeps reabsorbing its
own highway forever, so the nearest empirical question is how hard it is to
stop one.  The ant builds its highway from an empty grid, an obstacle is
dropped in the road ahead, and the run continues.  Obstacles are placed only
after widening the visited bounding box, because the certificate treats
everything outside that box as background and would otherwise certify a
highway that is about to hit a wall.

Of 245 trials, 188 actually struck the obstacle (the rest missed the road):

| | |
|--|--|
| recovered a certified highway | **188 of 188** |
| never recovered | 0 |
| recovery after contact | 311 steps min, 7,047 median, 56,117 max |
| period of the new highway | **104, every single time** |
| left in the same direction | 67 |
| left in a different direction | 121 |

Even a 21-cell wall laid across the road is absorbed in a few thousand steps.
The highway is easy to *deflect* and, at this scale, impossible to *destroy*:
the direction is scrambled but the shape is not.  That is the dynamical
counterpart of the catalogue result, where the period-104 highway was the only
one that exists below period 104.

### 3. The ant's past  (`backward.py`)

The forward map on (grid, position, heading) is a bijection, so the ant has a
unique past as well as a unique future.  That immediately kills a family of
proof strategies: distinct configurations never merge, the highway is *not* an
attractor in the contracting sense, and nothing that argues by shrinking
basins can work.

Running the inverse step from an empty grid gives no new system at all.  The
time-reversed ant is the forward ant rotated 180 degrees:

    backward position at step n  ==  rot180(forward position at step n-1) + (0,-1)

with zero mismatches over 50,000 steps, and the two grids agreeing at every
checkpoint.  The backward run therefore has the same onset of 9,977 and the
same period 104, with the drift negated.  The past is the future seen upside
down, so the conjecture is time-symmetric and a counterexample would have to
fail in both directions at once.

### 4. How long is the chaotic phase?  (`survey.py --random`)

The exhaustive runs show a maximum onset of 43,264 for 3x3, 119,673 for 4x4
and 233,232 for 5x5, which reads like fast growth with box size.  It is not.
Those three runs drew 512, 65,536 and 33,554,432 patterns respectively, so
they differ in sample size as much as in box size.  Drawing an equal 50,000
random patterns at each size separates the two:

| box | max onset | mean | median | 99.9th percentile |
|-----|-----------|------|--------|-------------------|
| 4x4 | 119,673 | 3,795 | 1,732 | 47,476 |
| 5x5 | 88,940 | 4,253 | 2,042 | 57,877 |
| 6x6 | 112,767 | 4,520 | 2,284 | 59,569 |
| 7x7 | 107,981 | 4,745 | 2,499 | 58,872 |
| 8x8 | 126,826 | 4,984 | 2,677 | 57,985 |

At fixed sample size the **maximum is flat** and so is the upper percentile.
What actually grows with the box is the typical case, mildly: the median rises
about 55% from 4x4 to 8x8.  The earlier apparent growth of the maximum was a
sample-size effect, and the claim that it grows with box size was wrong.

All 250,000 of these patterns reached a certified highway, which extends the
evidence to 6x6, 7x7 and 8x8 boxes that the exhaustive sweep never reached.

Data: `data/turmites/`, `data/trapping.json`, `data/n{4..8}_random50000_*`.

## Honest limitations

* Only patterns inside a centred *n*×*n* box with the ant at the box
  centre, heading north (other headings are rotations of the same
  experiment; the mirror image of a pattern gives the mirror trajectory).
* 6×6 has 2^36 ≈ 69 billion patterns; exhaustive search stops at 5×5.
  `survey.py --random` samples larger boxes if you want more evidence.
* Any pattern that had run 2,000,000 steps, or wandered more than 2,048
  cells from the origin, would have been reported as `TIMEOUT` / `OOB`.
  None did.
* Certification is *sufficient*, not necessary: a pattern that reached the
  highway but never satisfied (B) would show up as a timeout.  None did.

## Reproduce

```bash
cd langtons_highway
python survey.py --n 3 --full          # 2 s
python survey.py --n 4 --full          # 2 s
python survey.py --n 5                 # ~6 min on 4 cores
python survey.py --n 8 --random 100000 # random sample of a bigger box
python make_movie.py                   # renders output/langtons_highway.mp4
python highway_catalogue.py --pmax 104 --radius 8   # the catalogue sweep (~2.4 h on 4 cores)
python catalogue_summary.py            # -> data/catalogue/summary.json
python turmite_survey.py --max-len 12  # classify every rule (~15 min on 4 cores)
python trap.py                         # obstacle trials against the highway
python backward.py                     # the time-reversal symmetry
python survey.py --n 8 --random 50000  # equal-sample-size onset data
pytest tests                           # from the repo root: pytest langtons_highway/tests
```

Dependencies: a C compiler for the survey; `numpy`, `pillow` and
`imageio-ffmpeg` (bundles ffmpeg) for the video; `python-sat` for the catalogue.

## Files

| file | purpose |
|------|---------|
| `ant.py` | pure-Python ant + certificate (reference implementation, used by the renderer) |
| `ant_core.c` | fast C ant + certificate, one CSV line per pattern |
| `survey.py` | compiles the core, sweeps a pattern space over all cores, writes `data/` |
| `make_movie.py` | renders the vertical video and `output/poster.png` from the simulator and `data/` |
| `highway_catalogue.py` | SAT enumeration of every highway (periodic drifting orbit) up to a period, with independent verification |
| `catalogue_summary.py` | aggregates catalogue runs, checks grid completeness, groups highways |
| `turmite_core.c` | n-colour turmite simulator: the certificate plus Brent cycle detection |
| `turmite_survey.py` | classifies every rule up to a given length |
| `trap.py` | drops obstacles in a running highway and measures the recovery |
| `backward.py` | the inverse step, and the 180-degree time-reversal symmetry |
| `tests/test_ant.py` | rule sanity, empty-grid onset 9,977, certificate rejects chaos, C vs Python |
| `data/` | survey results (see above) |
| `output/` | the video and its poster frame |
