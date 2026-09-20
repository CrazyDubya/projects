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
