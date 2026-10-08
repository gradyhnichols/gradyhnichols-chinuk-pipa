"""Render Duployan text without Pillow's Raqm layout engine (intended for platforms where Raqm is unavailable).

Shaping is done with HarfBuzz (``uharfbuzz``), and glyph outlines are drawn from the font with fontTools,
flattened to polygons and filled with Pillow (even-odd within each glyph, union across glyphs). It is meant as a
stand-in for Pillow+Raqm when generating synthetic training data; the two outputs are similar but not identical.

    from chinukpipa.render import Renderer
    im = Renderer("fonts/NotoSansDuployan-Regular.ttf").render("\U0001BC05\U0001BC41", size=64)
"""
from __future__ import annotations

from PIL import Image, ImageChops, ImageDraw
from fontTools.pens.basePen import BasePen
from fontTools.ttLib import TTFont


class _FlattenPen(BasePen):
    def __init__(self, glyphset, steps: int = 8):
        super().__init__(glyphset)
        self.contours: list[list[tuple[float, float]]] = []
        self.cur: list[tuple[float, float]] = []
        self.steps = steps

    def _moveTo(self, p):
        if self.cur:
            self.contours.append(self.cur)
        self.cur = [p]

    def _lineTo(self, p):
        self.cur.append(p)

    def _curveToOne(self, p1, p2, p3):
        p0 = self.cur[-1]
        for i in range(1, self.steps + 1):
            t = i / self.steps
            mt = 1 - t
            self.cur.append((mt**3 * p0[0] + 3 * mt * mt * t * p1[0] + 3 * mt * t * t * p2[0] + t**3 * p3[0],
                             mt**3 * p0[1] + 3 * mt * mt * t * p1[1] + 3 * mt * t * t * p2[1] + t**3 * p3[1]))

    def _qCurveToOne(self, p1, p2):
        p0 = self.cur[-1]
        for i in range(1, self.steps + 1):
            t = i / self.steps
            mt = 1 - t
            self.cur.append((mt * mt * p0[0] + 2 * mt * t * p1[0] + t * t * p2[0],
                             mt * mt * p0[1] + 2 * mt * t * p1[1] + t * t * p2[1]))

    def _closePath(self):
        if self.cur:
            self.contours.append(self.cur)
        self.cur = []

    _endPath = _closePath


class Renderer:
    def __init__(self, font_path: str):
        import uharfbuzz as hb
        self.hb = hb
        self.font_path = font_path
        self.tt = TTFont(font_path)
        self.glyphset = self.tt.getGlyphSet()
        self.names = self.tt.getGlyphOrder()
        self.upem = self.tt["head"].unitsPerEm
        self.hbfont = hb.Font(hb.Face(hb.Blob.from_file_path(font_path)))
        self._cache: dict[str, list] = {}

    def shape(self, text: str):
        buf = self.hb.Buffer()
        buf.add_str(text)
        buf.guess_segment_properties()
        self.hb.shape(self.hbfont, buf, {})
        return buf.glyph_infos, buf.glyph_positions

    def has_dotted_circle(self, text: str) -> bool:
        infos, _ = self.shape(text)
        return any(self.names[g.codepoint].startswith("uni25CC") for g in infos)

    def _outline(self, name: str):
        if name not in self._cache:
            pen = _FlattenPen(self.glyphset)
            self.glyphset[name].draw(pen)
            if pen.cur:
                pen.contours.append(pen.cur)
            self._cache[name] = pen.contours
        return self._cache[name]

    def render(self, text: str, size: int = 64, pad: int = 4, jitter: float = 0.0, rng=None) -> Image.Image:
        """Black text on white, cropped to the ink with `pad` pixels of margin.

        jitter > 0 imitates handwriting in outline space: each glyph gets its own small rotation, scale and
        shear about its attachment point, and every outline point a smooth low-frequency wobble.
        """
        import math
        import random as _random
        rng = rng or _random.Random()
        infos, poss = self.shape(text)
        s = size / self.upem
        x = y = 0.0
        glyphs = []   # one list of contours per glyph
        for info, pos in zip(infos, poss):
            ox, oy = x + pos.x_offset, y + pos.y_offset
            contours = self._outline(self.names[info.codepoint])
            if jitter > 0:
                th = rng.gauss(0, 0.08 * jitter)
                sc = 1 + rng.gauss(0, 0.10 * jitter)
                sh = rng.gauss(0, 0.10 * jitter)
                ca, sa = math.cos(th) * sc, math.sin(th) * sc
                ph1, ph2, fr = rng.uniform(0, 6.3), rng.uniform(0, 6.3), rng.uniform(0.002, 0.006)
                amp = 18 * jitter
                def tf(px, py):
                    px, py = px + sh * py, py
                    qx, qy = ca * px - sa * py, sa * px + ca * py
                    return (qx + amp * math.sin(fr * (qy + ox) + ph1), qy + amp * math.sin(fr * (qx + oy) + ph2))
                contours = [[tf(px, py) for px, py in c] for c in contours]
            glyphs.append([[((ox + px) * s, -(oy + py) * s) for px, py in c] for c in contours])
            x += pos.x_advance
            y += pos.y_advance
        pts = [p for g in glyphs for c in g for p in c]
        if not pts:
            return Image.new("L", (2 * pad + 1, 2 * pad + 1), 255)
        minx, miny = min(p[0] for p in pts), min(p[1] for p in pts)
        maxx, maxy = max(p[0] for p in pts), max(p[1] for p in pts)
        w, h = int(maxx - minx) + 2 * pad + 2, int(maxy - miny) + 2 * pad + 2

        ink = Image.new("1", (w, h), 0)
        for contours in glyphs:
            # even-odd within a glyph (rings keep their counters), union across glyphs (joins stay solid)
            g = Image.new("1", (w, h), 0)
            for c in contours:
                if len(c) < 3:
                    continue
                layer = Image.new("1", (w, h), 0)
                ImageDraw.Draw(layer).polygon([(px - minx + pad, py - miny + pad) for px, py in c], fill=1)
                g = ImageChops.logical_xor(g, layer)
            ink = ImageChops.logical_or(ink, g)
        return Image.eval(ink.convert("L"), lambda v: 255 - v)
