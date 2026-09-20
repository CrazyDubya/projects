"""
Tests for the Langton's Ant simulator and highway certificate.

Run with:  pytest langtons_highway/tests
"""

import csv
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from ant import Ant, pattern_cells, steps_to_highway  # noqa: E402


def test_first_steps_follow_the_rule():
    ant = Ant()
    ant.step()  # white cell: turn right (N -> E), flip, move
    assert (0, 0) in ant.black
    assert ant.heading == 1 and (ant.x, ant.y) == (1, 0)
    ant.step()  # another white cell: E -> S
    assert ant.heading == 2 and (ant.x, ant.y) == (1, -1)
    # walk back onto (0,0) which is now black: should turn left
    ant2 = Ant(black={(0, 0)})
    ant2.step()
    assert (0, 0) not in ant2.black
    assert ant2.heading == 3 and (ant2.x, ant2.y) == (-1, 0)


def test_pattern_bits_map_row_major():
    assert pattern_cells(3, 0) == []
    assert pattern_cells(3, 1) == [(0, 0)]
    assert pattern_cells(3, 1 << 4) == [(1, 1)]
    assert pattern_cells(3, 1 << 8) == [(2, 2)]
    assert Ant.from_pattern(4, 0).x == 2 and Ant.from_pattern(4, 0).y == 2


def test_empty_grid_highway_onset_is_9977():
    r = Ant().find_highway(200_000)
    assert r is not None
    assert r["onset"] == 9977
    assert r["period"] == 104
    assert (abs(r["dx"]), abs(r["dy"])) == (2, 2)


def test_certificate_rejects_chaos():
    ant = Ant()
    ant.run(2000)
    assert ant.drift_hint(104) is None
    assert ant.certify(104) is None


def test_highway_keeps_translating_after_certificate():
    """Sanity check of the certificate's promise on the real trajectory."""
    ant = Ant()
    r = ant.find_highway(200_000)
    s, dx, dy = r["cert_step"], r["dx"], r["dy"]
    ant.run(104 * 20)
    for k in range(20):
        a, b = ant.history[s - 104 + k * 104], ant.history[s + k * 104]
        assert (b[0] - a[0], b[1] - a[1]) == (dx, dy)
        assert a[2] == b[2] and a[3] == b[3]


@pytest.mark.skipif(shutil.which("cc") is None and shutil.which("gcc") is None,
                    reason="no C compiler")
def test_c_core_agrees_with_python_reference():
    cc = shutil.which("cc") or shutil.which("gcc")
    tmp = Path(tempfile.mkdtemp())
    exe = tmp / "ant_core"
    subprocess.run([cc, "-O2", "-o", str(exe), str(HERE / "ant_core.c")], check=True)
    out = subprocess.run([str(exe), "3", "1", "1", "0", "40", "2000000"],
                         capture_output=True, text=True, check=True).stdout
    rows = list(csv.reader(out.strip().splitlines()))
    assert len(rows) == 40
    for row in rows:
        idx = int(row[0])
        r = steps_to_highway(3, idx)
        assert row[1] == "HIGHWAY"
        assert (int(row[2]), int(row[3]), int(row[4]), int(row[5]), int(row[6])) == (
            r["onset"], r["cert_step"], r["period"], r["dx"], r["dy"])


def test_catalogue_verifier_accepts_classic_and_rejects_corruption():
    from highway_catalogue import verify, canonical

    ant = Ant()
    r = ant.find_highway(200_000)
    s, P = r["cert_step"], 104
    reads = [ant.history[s - P + i][2] for i in range(P)]
    h0 = ant.history[s - P][3]
    sol = {"P": P, "d": [r["dx"], r["dy"]], "h0": h0, "reads": reads}
    # the SAT catalogue normalises drift to the quadrant dx>=1, dy>=0 by
    # rotation; the verifier itself works for any drift
    assert verify(sol)
    bad = dict(sol, reads=reads[:10] + [1 - reads[10]] + reads[11:])
    assert not verify(bad)
    assert len(canonical(reads)) == P


def test_catalogue_small_period_has_no_highway():
    pytest.importorskip("pysat")
    from highway_catalogue import solve_instance

    res = solve_instance(12, (2, 0))
    assert res["solutions"] == []


def test_backward_step_inverts_forward_step():
    from backward import back_step

    ant = Ant()
    ant.run(500)
    x, y, h, black = ant.x, ant.y, ant.heading, set(ant.black)
    for i in range(499, -1, -1):
        x, y, h = back_step(x, y, h, black)
        hx, hy, _, hh, _, _ = ant.history[i]
        assert (x, y, h) == (hx, hy, hh)
    assert (x, y, h) == (0, 0, 0)
    assert black == set()


def test_time_reversal_is_a_180_degree_rotation():
    """Backward from an empty grid is the forward run rotated by 180 degrees."""
    from backward import run_backward

    n = 2000
    states, back_black = run_backward(n)
    fwd = Ant()
    fwd.run(n)
    # the cell written at backward step k is the rot180 image of forward step k-1
    assert all((states[k][0], states[k][1]) == (-fwd.history[k - 1][0], -fwd.history[k - 1][1] - 1)
               for k in range(1, n + 1))
    assert {(-c[0], -c[1] - 1) for c in fwd.black} == back_black


def test_trap_placement_widens_the_bounding_box():
    """An obstacle outside the visited box must extend it, or the proof breaks."""
    from trap import place

    ant = Ant()
    ant.run(200)
    far = {(500, -400)}
    place(ant, far)
    minx, maxx, miny, maxy = ant.bbox
    assert minx <= 500 <= maxx and miny <= -400 <= maxy
    assert (500, -400) in ant.black


@pytest.mark.skipif(shutil.which("cc") is None and shutil.which("gcc") is None,
                    reason="no C compiler")
def test_turmite_core_reproduces_langtons_ant():
    """Rule 'RL' in the generalised n-colour core must match the specialised ant."""
    cc = shutil.which("cc") or shutil.which("gcc")
    tmp = Path(tempfile.mkdtemp())
    exe = tmp / "turmite_core"
    subprocess.run([cc, "-O2", "-o", str(exe), str(HERE / "turmite_core.c")], check=True)
    row = subprocess.run([str(exe), "RL", "200000"], capture_output=True, text=True,
                         check=True).stdout.strip().split(",")
    assert row[0] == "RL" and row[1] == "HIGHWAY"
    assert int(row[2]) == 9977          # same onset as ant.py
    assert int(row[4]) == 104           # same period
    assert (abs(int(row[5])), abs(int(row[6]))) == (2, 2)
    # a rule that only spins must be proved periodic, not merely timed out
    row = subprocess.run([str(exe), "RR", "200000"], capture_output=True, text=True,
                         check=True).stdout.strip().split(",")
    assert row[1] == "CYCLE"
