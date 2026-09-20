#!/usr/bin/env python3
"""
Render the "Langton's Highway" social-media video.

Output: output/langtons_highway.mp4 (1080x1920, 30 fps, H.264 + AAC) and
output/poster.png.  Everything on screen is generated from the simulator
in ``ant.py`` and the survey results in ``data/``; nothing is hand-drawn.

    python make_movie.py            # full render (a few minutes)
    python make_movie.py --preview  # every 5th frame, quick check

Author: Claude (with Stephen)
Created: 2026-09-20
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import wave
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ant import Ant, pattern_cells

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE / "output"

W, H, FPS = 1080, 1920, 30
BG = (11, 13, 20)
INK = (236, 233, 224)
DIM = (150, 156, 170)
AMBER = (255, 184, 77)
RED = (255, 59, 92)
TEAL = (94, 210, 200)
PANEL = (26, 31, 44)
GRIDL = (44, 50, 66)
PAL = np.array([BG, PANEL, AMBER], np.uint8)

FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")


def font(size: int, bold: bool = True, mono: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSansMono" if mono else "DejaVuSans"
    name += "-Bold.ttf" if bold else ".ttf"
    return ImageFont.truetype(str(FONT_DIR / name), size)


F_HERO, F_H1, F_H2, F_BODY, F_CAP = font(104), font(72), font(54), font(42, False), font(34, False)
F_NUM, F_NUM_S = font(84, True, True), font(40, True, True)


# ----------------------------------------------------------------------
# drawing helpers
# ----------------------------------------------------------------------
def wrap(draw: ImageDraw.ImageDraw, text: str, fnt, max_w: int) -> List[str]:
    lines: List[str] = []
    for para in text.split("\n"):
        words, cur = para.split(" "), ""
        for w in words:
            trial = (cur + " " + w).strip()
            if draw.textlength(trial, font=fnt) <= max_w:
                cur = trial
            else:
                lines.append(cur)
                cur = w
        lines.append(cur)
    return lines


def text_block(draw, text, y, fnt, fill=INK, max_w=W - 120, align="center",
               spacing=1.25, x=None) -> int:
    """Draw wrapped text; returns the y just below the block."""
    lh = int(fnt.size * spacing)
    for line in wrap(draw, text, fnt, max_w):
        tw = draw.textlength(line, font=fnt)
        if align == "center":
            xx = (W - tw) / 2
        elif align == "left":
            xx = x if x is not None else 60
        else:
            xx = W - 60 - tw
        draw.text((xx, y), line, font=fnt, fill=fill)
        y += lh
    return y


def blend(c1, c2, t):
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))


def ease(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


class Canvas:
    def __init__(self):
        self.arr = np.empty((H, W, 3), np.uint8)
        self.arr[:] = BG

    def paste(self, img: np.ndarray, x: int, y: int) -> None:
        h, w = img.shape[:2]
        self.arr[y:y + h, x:x + w] = img

    def pil(self) -> Tuple[Image.Image, ImageDraw.ImageDraw]:
        im = Image.fromarray(self.arr)
        return im, ImageDraw.Draw(im)


# ----------------------------------------------------------------------
# board: a fixed window on the ant's grid
# ----------------------------------------------------------------------
class Board:
    def __init__(self, ant: Ant, cx: int, cy: int, cells_w: int, cells_h: int,
                 px: int, grid_lines: bool = False):
        self.ant = ant
        self.cw, self.ch, self.px, self.lines = cells_w, cells_h, px, grid_lines
        self.x0 = cx - cells_w // 2
        self.ytop = cy + cells_h // 2          # grid y shown on the top row
        self.state = np.zeros((cells_h, cells_w), np.uint8)
        self.seen = 0
        for (x, y) in ant.black:
            self._set(x, y, 2)

    def _set(self, x: int, y: int, v: int) -> None:
        c, r = x - self.x0, self.ytop - y
        if 0 <= c < self.cw and 0 <= r < self.ch:
            self.state[r, c] = v

    def sync(self) -> None:
        """Apply every step taken since the last sync."""
        hist = self.ant.history
        for i in range(self.seen, self.ant.step_no):
            x, y, col = hist[i][0], hist[i][1], hist[i][2]
            self._set(x, y, 1 if col else 2)
        self.seen = self.ant.step_no

    def render(self, show_ant: bool = True, dim: float = 1.0) -> np.ndarray:
        self.sync()
        img = PAL[self.state]
        px = self.px
        img = np.repeat(np.repeat(img, px, 0), px, 1)
        if self.lines:
            img[::px, :] = GRIDL
            img[:, ::px] = GRIDL
        if show_ant:
            c, r = self.ant.x - self.x0, self.ytop - self.ant.y
            if 0 <= c < self.cw and 0 <= r < self.ch:
                y0, x0 = r * px, c * px
                if px >= 20:
                    im = Image.fromarray(img)
                    d = ImageDraw.Draw(im)
                    cx, cy = x0 + px / 2, y0 + px / 2
                    s = px * 0.36
                    hx, hy = (0, -1), (1, 0)
                    dirs = [(0, -1), (1, 0), (0, 1), (-1, 0)][self.ant.heading]
                    hx, hy = dirs
                    tip = (cx + hx * s, cy + hy * s)
                    left = (cx - hx * s * 0.6 + hy * s * 0.6, cy - hy * s * 0.6 - hx * s * 0.6)
                    right = (cx - hx * s * 0.6 - hy * s * 0.6, cy - hy * s * 0.6 + hx * s * 0.6)
                    d.polygon([tip, left, right], fill=RED)
                    img = np.asarray(im)
                else:
                    pad = 1 if px >= 6 else 0
                    img[max(0, y0 - pad):y0 + px + pad, max(0, x0 - pad):x0 + px + pad] = RED
        if dim < 1.0:
            img = (img.astype(np.float32) * dim).astype(np.uint8)
        return img


def thumb(n: int, index: int, cell: int, gap: int = 2, ant_xy=None) -> np.ndarray:
    """Small picture of an n x n starting pattern."""
    size = n * cell + (n + 1) * gap
    img = np.empty((size, size, 3), np.uint8)
    img[:] = GRIDL
    black = set(pattern_cells(n, index))
    ax, ay = ant_xy if ant_xy else (n // 2, n // 2)
    for y in range(n):
        for x in range(n):
            r, c = (n - 1 - y), x
            y0, x0 = gap + r * (cell + gap), gap + c * (cell + gap)
            img[y0:y0 + cell, x0:x0 + cell] = AMBER if (x, y) in black else PANEL
            if (x, y) == (ax, ay):
                m = max(1, cell // 4)
                img[y0 + m:y0 + cell - m, x0 + m:x0 + cell - m] = RED
    return img


# ----------------------------------------------------------------------
# audio
# ----------------------------------------------------------------------
class Audio:
    SR = 44100

    def __init__(self, seconds: float):
        self.n = int(seconds * self.SR)
        self.buf = np.zeros(self.n, np.float32)

    def _add(self, t: float, sig: np.ndarray) -> None:
        i = int(t * self.SR)
        if i >= self.n:
            return
        sig = sig[: self.n - i]
        self.buf[i:i + len(sig)] += sig

    def tick(self, t: float, freq: float = 1200.0, gain: float = 0.22, dur: float = 0.035):
        k = np.arange(int(dur * self.SR))
        env = np.exp(-k / (dur * self.SR / 5))
        self._add(t, (gain * env * np.sin(2 * np.pi * freq * k / self.SR)).astype(np.float32))

    def chord(self, t: float, freqs, dur: float, gain: float = 0.12, attack: float = 0.8,
              release: float = 2.0):
        k = np.arange(int(dur * self.SR)) / self.SR
        env = np.minimum(k / attack, 1.0) * np.minimum((dur - k) / release, 1.0)
        env = np.clip(env, 0, 1)
        sig = sum(np.sin(2 * np.pi * f * k) * (0.7 ** i) for i, f in enumerate(freqs))
        self._add(t, (gain * env * sig / len(freqs) * 2).astype(np.float32))

    def drone(self, t0: float, t1: float, gain: float = 0.05):
        k = np.arange(int((t1 - t0) * self.SR)) / self.SR
        env = np.minimum(k / 2.0, 1.0) * np.minimum((t1 - t0 - k) / 2.0, 1.0)
        lfo = 0.75 + 0.25 * np.sin(2 * np.pi * 0.1 * k)
        sig = (np.sin(2 * np.pi * 55 * k) + 0.5 * np.sin(2 * np.pi * 110 * k)
               + 0.25 * np.sin(2 * np.pi * 164.81 * k))
        self._add(t0, (gain * env * lfo * sig).astype(np.float32))

    def write(self, path: Path) -> None:
        peak = float(np.max(np.abs(self.buf))) or 1.0
        pcm = (self.buf / peak * 0.85 * 32767).astype(np.int16)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.SR)
            wf.writeframes(pcm.tobytes())


# ----------------------------------------------------------------------
# scenes
# ----------------------------------------------------------------------
Frame = np.ndarray


def fade(frame: Frame, a: float) -> Frame:
    if a >= 1.0:
        return frame
    return (frame.astype(np.float32) * a + np.array(BG, np.float32) * (1 - a)).astype(np.uint8)


def load_summaries() -> dict:
    out = {}
    for n in (3, 4, 5):
        p = DATA / f"n{n}_summary.json"
        if p.exists():
            out[n] = json.loads(p.read_text())
    return out


def load_hist(n: int):
    p = DATA / f"n{n}_onset_hist.csv"
    rows = [line.split(",") for line in p.read_text().splitlines()[1:]]
    return np.array([int(r[0]) for r in rows]), np.array([int(r[1]) for r in rows])


class Movie:
    def __init__(self, preview: bool = False):
        self.preview = preview
        self.t = 0                      # frame counter
        self.audio_events: List[Tuple[str, float, dict]] = []
        self.summ = load_summaries()
        self.poster: Optional[Frame] = None
        self.writer = None

    # -- bookkeeping ----------------------------------------------------
    def emit(self, frame: Frame) -> None:
        """Stream one frame to the encoder (every 5th in preview mode)."""
        if self.writer is not None and (not self.preview or self.t % 5 == 0):
            self.writer.send(np.ascontiguousarray(frame).tobytes())
        self.t += 1

    def sec(self) -> float:
        return self.t / FPS

    def ev(self, kind: str, **kw) -> None:
        self.audio_events.append((kind, self.sec(), kw))

    # -- scene 1+2: hook and rules --------------------------------------
    def scene_rules(self) -> None:
        n_cells, px = 11, 88
        ant = Ant()
        board = Board(ant, 0, 0, n_cells, n_cells, px, grid_lines=True)
        bw = n_cells * px
        bx, by = (W - bw) // 2, 560
        total = 12 * FPS
        step_times = [45 + 18 * i for i in range(10)] + [225 + 9 * i for i in range(15)]
        steps_done = 0
        last_read = None
        for f in range(total):
            while steps_done < len(step_times) and f >= step_times[steps_done]:
                last_read = ant.colour((ant.x, ant.y))
                ant.step()
                steps_done += 1
                self.ev("tick", freq=880 if last_read else 1320)
            cv = Canvas()
            cv.paste(board.render(), bx, by)
            im, d = cv.pil()
            a = ease(f / 20)
            text_block(d, "TWO RULES.", 120, F_HERO, blend(BG, AMBER, a))
            if f >= 30:
                a2 = ease((f - 30) / 20)
                text_block(d, "One unsolved mystery.", 260, F_H2, blend(BG, INK, a2))
            if f >= 45:
                white_on = last_read == 0
                c_w = INK if white_on else DIM
                c_b = INK if not white_on else DIM
                if steps_done == 0:
                    c_w = c_b = DIM
                y = 380
                d.text((60, y), "WHITE cell:", font=F_BODY, fill=c_w)
                d.text((330, y), "turn RIGHT, flip it, step.", font=F_BODY, fill=c_w)
                d.text((60, y + 60), "BLACK cell:", font=F_BODY, fill=c_b)
                d.text((330, y + 60), "turn LEFT, flip it, step.", font=F_BODY, fill=c_b)
            if f >= 225:
                a3 = ease((f - 225) / 20)
                text_block(d, "That is the entire program.", 1560, F_H2, blend(BG, INK, a3))
                text_block(d, "Langton's Ant, 1986", 1650, F_CAP, blend(BG, DIM, a3))
            d.text((60, 1800), f"step {ant.step_no:>6,}", font=F_NUM_S, fill=DIM)
            self.emit(fade(np.asarray(im), ease(f / 12)))
        self.ant_after_rules = ant

    # -- scene 3+4: chaos then highway -----------------------------------
    def scene_chaos_highway(self) -> None:
        # pre-run to find where the mess lives and which way the road goes
        probe = Ant()
        res = probe.find_highway(200_000)
        assert res is not None
        onset = res["onset"]
        probe2 = Ant()
        probe2.run(onset)
        minx, maxx, miny, maxy = probe2.bbox
        sx, sy = (1 if res["dx"] > 0 else -1), (1 if res["dy"] > 0 else -1)
        cx = (minx + maxx) // 2 + sx * 30
        cy = (miny + maxy) // 2 + sy * 30
        n_cells, px = 135, 8
        ant = self.ant_after_rules
        board = Board(ant, cx, cy, n_cells, n_cells, px)
        bx, by = (W - n_cells * px) // 2, 520

        # chaos: exponential ramp from the current step to the onset
        total = 16 * FPS
        start = ant.step_no
        b = math.log(90) / total
        denom = math.exp(b * total) - 1
        for f in range(total):
            target = start + int(round((onset - start) * (math.exp(b * (f + 1)) - 1) / denom))
            speed = target - ant.step_no
            while ant.step_no < target:
                ant.step()
            self.ev("tick", freq=300 + 8 * speed, gain=0.12)
            cv = Canvas()
            cv.paste(board.render(), bx, by)
            im, d = cv.pil()
            text_block(d, "Now let it run.", 120, F_H1, AMBER)
            if f >= 3 * FPS:
                text_block(d, "Thousands of steps of pure mess.", 240, F_BODY, INK)
            if f >= 9 * FPS:
                text_block(d, "No pattern. No repetition. Chaos.", 310, F_BODY, INK)
            d.text((60, 1800), f"step {ant.step_no:>7,}", font=F_NUM_S, fill=DIM)
            self.emit(np.asarray(im))

        # highway
        self.ev("chord", freqs=(220, 277.18, 329.63, 440), dur=8.0, gain=0.14)
        total = 9 * FPS
        per = 14
        for f in range(total):
            while ant.step_no < onset + per * (f + 1):
                ant.step()
            if f % 7 == 0:
                self.ev("tick", freq=660, gain=0.10, dur=0.05)
            cv = Canvas()
            cv.paste(board.render(), bx, by)
            im, d = cv.pil()
            text_block(d, f"Step {onset:,}.", 120, F_H1, AMBER)
            a = ease((f - 15) / 20)
            text_block(d, "It locks into a 104-step loop and builds a highway.", 240,
                       F_BODY, blend(BG, INK, a))
            if f >= 4 * FPS:
                a = ease((f - 4 * FPS) / 15)
                text_block(d, "Forever.", 340, F_H2, blend(BG, TEAL, a))
            d.text((60, 1800), f"step {ant.step_no:>7,}", font=F_NUM_S, fill=DIM)
            fr = np.asarray(im)
            if f == total - 1:
                self.poster = fr
            self.emit(fr)
        self.highway_board = board
        self.onset_classic = onset

    # -- scene 5: the mystery --------------------------------------------
    def scene_mystery(self) -> None:
        board = self.highway_board
        bx, by = (W - board.cw * board.px) // 2, 520
        base = board.render(show_ant=True, dim=0.28)
        self.ev("chord", freqs=(220, 261.63, 329.63), dur=9.0, gain=0.08, attack=2.0)
        total = 9 * FPS
        lines = [
            (0, "PROVEN (1992):", AMBER, F_H2),
            (0, "the ant can never stay trapped inside a bounded region.", INK, F_BODY),
            (3 * FPS, "UNPROVEN, 40 years on:", AMBER, F_H2),
            (3 * FPS, "that it ALWAYS ends up building this highway, no matter what "
                      "finite mess you start it in.", INK, F_BODY),
            (6 * FPS, "Every experiment ever run says yes.", TEAL, F_H2),
            (6 * FPS, "Nobody can prove it.", TEAL, F_H2),
        ]
        for f in range(total):
            cv = Canvas()
            cv.paste(base, bx, by)
            im, d = cv.pil()
            y = 150
            for t0, txt, col, fnt in lines:
                if f >= t0:
                    a = ease((f - t0) / 15)
                    y = text_block(d, txt, y, fnt, blend(BG, col, a)) + 20
                else:
                    break
            self.emit(np.asarray(im))

    # -- scene 6: the experiment ------------------------------------------
    def scene_experiment(self) -> None:
        rng = np.random.default_rng(7)
        summ = self.summ
        sizes = sorted(summ)
        total_patterns = sum(summ[n]["patterns"] for n in sizes)
        total_highway = sum(summ[n]["status"].get("HIGHWAY", 0) for n in sizes)
        big_n = max(sizes)
        onsets, counts = load_hist(big_n)
        s = summ[big_n]
        size_words = " + ".join(f"every {n}×{n}" for n in sizes)

        # part 1: wall of patterns + counter (8 s)
        cols, rows, cell, gap = 12, 9, 8, 2
        tsize = 5 * cell + 6 * gap
        pad = (W - cols * tsize) // (cols + 1)
        total = 8 * FPS
        wall = [(int(rng.integers(3, 6)), int(rng.integers(0, 2 ** 25))) for _ in range(cols * rows)]
        for f in range(total):
            if f % 3 == 0:
                for _ in range(24):
                    i = int(rng.integers(0, len(wall)))
                    n = int(rng.integers(3, 6))
                    wall[i] = (n, int(rng.integers(0, 2 ** (n * n))))
                self.ev("tick", freq=1600, gain=0.05, dur=0.02)
            cv = Canvas()
            for i, (n, idx) in enumerate(wall):
                r, c = divmod(i, cols)
                th = thumb(n, idx, (tsize - 6 * gap) // 5 if n == 5 else (tsize - (n + 1) * gap) // n, gap)
                x = pad + c * (tsize + pad)
                y = 640 + r * (tsize + 14)
                cv.paste(th, x, y)
            im, d = cv.pil()
            text_block(d, "So I checked.", 120, F_H1, AMBER)
            text_block(d, f"{size_words} starting pattern, each run by a C simulator "
                          "until it either built a highway or gave up.", 230, F_BODY, INK)
            k = ease(f / (total - 30))
            n_shown = int(round(total_patterns * k))
            d.text((60, 1440), f"{n_shown:>12,}", font=F_NUM, fill=INK)
            d.text((60, 1540), "patterns tested", font=F_CAP, fill=DIM)
            if f >= total - 45:
                a = ease((f - (total - 45)) / 15)
                col = blend(BG, TEAL, a)
                d.text((60, 1640), f"{total_highway:>12,}", font=F_NUM, fill=col)
                d.text((60, 1740), "built the highway", font=F_CAP, fill=col)
                pct = 100.0 * total_highway / total_patterns
                d.text((640, 1740), f"{pct:.4g}%", font=F_NUM_S, fill=col)
            self.emit(np.asarray(im))

        # part 2: histogram of onset steps (9 s)
        self.ev("chord", freqs=(261.63, 329.63, 392.0, 523.25), dur=9.0, gain=0.10)
        lo, hi = math.log10(max(onsets.min(), 50)), math.log10(onsets.max() * 1.05)
        nb = 60
        bins = np.zeros(nb)
        idxs = np.clip(((np.log10(onsets) - lo) / (hi - lo) * nb).astype(int), 0, nb - 1)
        np.add.at(bins, idxs, counts)
        bins = bins / bins.max()
        gx0, gx1, gy0, gy1 = 90, W - 90, 1240, 640
        fast, slow = s["onset_min"], s["onset_max"]
        fast_p, slow_p = s["onset_min_pattern"], s["onset_max_pattern"]
        total = 9 * FPS
        for f in range(total):
            cv = Canvas()
            im, d = cv.pil()
            y = text_block(d, "How long until the highway?", 100, F_H1, AMBER)
            text_block(d, f"Steps of chaos before the loop appears, for all "
                          f"{s['patterns']:,} {big_n}×{big_n} patterns.", y + 10, F_BODY, INK)
            grow = ease(f / 40)
            bw_ = (gx1 - gx0) / nb
            for i in range(nb):
                hgt = bins[i] * (gy0 - gy1) * grow
                x0 = gx0 + i * bw_
                d.rectangle([x0 + 1, gy0 - hgt, x0 + bw_ - 1, gy0], fill=AMBER)
            d.line([gx0, gy0, gx1, gy0], fill=DIM, width=2)
            for e in (100, 1000, 10000, 100000):
                if lo <= math.log10(e) <= hi:
                    x = gx0 + (math.log10(e) - lo) / (hi - lo) * (gx1 - gx0)
                    d.line([x, gy0, x, gy0 + 12], fill=DIM, width=2)
                    lab = f"{e:,}"
                    d.text((x - d.textlength(lab, font=F_CAP) / 2, gy0 + 18), lab, font=F_CAP, fill=DIM)
            d.text((gx0, gy0 + 64), "steps before the highway (log scale)", font=F_CAP, fill=DIM)

            def marker(val, label, y, when, colour=INK):
                if f < when:
                    return
                a = ease((f - when) / 15)
                x = gx0 + (math.log10(val) - lo) / (hi - lo) * (gx1 - gx0)
                d.line([x, gy1 - 10, x, gy0], fill=blend(BG, colour, a), width=3)
                d.text((min(x + 10, W - 400), y), label, font=F_CAP, fill=blend(BG, colour, a))

            marker(self.onset_classic, f"empty grid: {self.onset_classic:,}", gy1 - 10, 2 * FPS, TEAL)
            marker(fast, f"fastest: {fast:,}", gy1 + 40, 3 * FPS, RED)
            marker(slow, f"slowest: {slow:,}", gy1 + 90, 4 * FPS, RED)
            if f >= 5 * FPS:
                a = ease((f - 5 * FPS) / 15)
                ims = Image.fromarray(fade(thumb(big_n, fast_p, 40, 4), a))
                im.paste(ims, (120, 1400))
                d.text((120, 1400 + ims.height + 12), f"fastest start\n{fast:,} steps",
                       font=F_CAP, fill=blend(BG, INK, a))
                ims = Image.fromarray(fade(thumb(big_n, slow_p, 40, 4), a))
                im.paste(ims, (W - 120 - ims.width, 1400))
                d.text((W - 120 - ims.width, 1400 + ims.height + 12),
                       f"slowest start\n{slow:,} steps", font=F_CAP, fill=blend(BG, INK, a))
                d.text((W / 2 - 130, 1440), "same highway,\nevery time", font=F_CAP, fill=DIM)
            self.emit(np.asarray(im))
        self.slow_n, self.slow_p, self.slow_onset = big_n, slow_p, slow

    # -- scene 7: the slowest run -----------------------------------------
    def scene_slowest(self) -> None:
        n, p = self.slow_n, self.slow_p
        probe = Ant.from_pattern(n, p)
        res = probe.find_highway(5_000_000)
        assert res is not None
        onset = res["onset"]
        probe2 = Ant.from_pattern(n, p)
        probe2.run(onset)
        minx, maxx, miny, maxy = probe2.bbox
        span = max(maxx - minx, maxy - miny) + 44
        px = max(3, min(8, W // span))
        n_cells = W // px
        sx, sy = (1 if res["dx"] > 0 else -1), (1 if res["dy"] > 0 else -1)
        cx = (minx + maxx) // 2 + sx * (n_cells // 2 - (maxx - minx) // 2 - 6)
        cy = (miny + maxy) // 2 + sy * (n_cells // 2 - (maxy - miny) // 2 - 6)
        ant = Ant.from_pattern(n, p)
        board = Board(ant, cx, cy, n_cells, n_cells, px)
        bx, by = (W - n_cells * px) // 2, 520
        big = thumb(n, p, 60, 6)

        total = 10 * FPS
        intro = int(1.5 * FPS)
        run_frames = total - intro
        tail = 2600
        end_step = onset + tail
        for f in range(total):
            cv = Canvas()
            if f < intro:
                cv.paste(big, (W - big.shape[1]) // 2, 900)
            else:
                g = (f - intro + 1) / run_frames
                target = int(end_step * ease(min(1.0, g * 1.15)))
                if f == total - 1:
                    target = end_step
                while ant.step_no < target:
                    ant.step()
                if ant.step_no >= onset and f % 7 == 0:
                    self.ev("tick", freq=660, gain=0.10, dur=0.05)
                cv.paste(board.render(), bx, by)
            im, d = cv.pil()
            y = text_block(d, "The slowest start I found.", 100, F_H1, AMBER)
            text_block(d, f"A {n}×{n} seed that wanders for {onset:,} steps "
                          "before the very same 104-step loop appears.", y + 10, F_BODY, INK)
            if f >= intro:
                d.text((60, 1800), f"step {ant.step_no:>8,}", font=F_NUM_S, fill=DIM)
                if ant.step_no >= onset:
                    d.text((W - 460, 1800), "highway", font=F_NUM_S, fill=TEAL)
            self.emit(np.asarray(im))

    # -- scene 8: close -----------------------------------------------------
    def scene_close(self) -> None:
        summ = self.summ
        total_patterns = sum(summ[n]["patterns"] for n in summ)
        self.ev("chord", freqs=(220, 277.18, 329.63, 440, 554.37), dur=8.0, gain=0.14)
        total = 8 * FPS
        lines = [
            (0, "Every highway was machine-checked.", AMBER, F_H2),
            (0, "Not \"it looks periodic\": a certificate that proves the loop repeats "
                "forever from that step on.", INK, F_BODY),
            (2 * FPS, "It is still not a theorem.", TEAL, F_H2),
            (2 * FPS, f"It is {total_patterns:,} more reasons to believe it.", INK, F_BODY),
            (4 * FPS, "Code, data and the proof-checker:", DIM, F_CAP),
            (4 * FPS, "github.com/CrazyDubya/projects\nlangtons_highway/", INK, F_H2),
        ]
        for f in range(total):
            cv = Canvas()
            im, d = cv.pil()
            y = 300
            for t0, txt, col, fnt in lines:
                if f >= t0:
                    a = ease((f - t0) / 15)
                    y = text_block(d, txt, y, fnt, blend(BG, col, a)) + 24
            a_out = ease((total - f) / 25)
            self.emit(fade(np.asarray(im), a_out))

    # -- assembly -------------------------------------------------------------
    def render(self, path: Path) -> None:
        """Render every scene straight into the encoder, then add audio."""
        import imageio_ffmpeg

        path.parent.mkdir(parents=True, exist_ok=True)
        video_only = path.with_suffix(".video.mp4")
        self.writer = imageio_ffmpeg.write_frames(
            str(video_only), (W, H), fps=6 if self.preview else FPS, codec="libx264",
            pix_fmt_out="yuv420p", quality=None,
            output_params=["-crf", "24", "-preset", "medium", "-movflags", "+faststart"],
            macro_block_size=1,
        )
        self.writer.send(None)
        self.scene_rules()
        self.scene_chaos_highway()
        self.scene_mystery()
        self.scene_experiment()
        self.scene_slowest()
        self.scene_close()
        self.writer.close()
        print(f"rendered {self.t} frames ({self.t / FPS:.1f} s), muxing audio...")

        secs = self.t / FPS
        au = Audio(secs + 0.5)
        au.drone(0, secs)
        for kind, t, kw in self.audio_events:
            getattr(au, kind)(t, **kw)
        wav = path.with_suffix(".wav")
        au.write(wav)
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(video_only), "-i", str(wav),
                        "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", "-shortest", str(path)],
                       check=True)
        video_only.unlink()
        wav.unlink()
        if self.poster is not None:
            Image.fromarray(self.poster).save(OUT / "poster.png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preview", action="store_true", help="render every 5th frame at 6 fps")
    ap.add_argument("--out", type=Path, default=OUT / "langtons_highway.mp4")
    a = ap.parse_args()
    m = Movie(preview=a.preview)
    m.render(a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
