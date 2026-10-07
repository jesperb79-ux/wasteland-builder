"""Deterministic chunk grid for OutbreakGeo (metres; x east, y north; origin = the place centre).

Chunk (i, j) is the half-open square [i·S, (i+1)·S) × [j·S, (j+1)·S), S = the chunk size, with the id
"c{i}_{j}": the same grid and naming as Wasteland's 60 m `cell` / `wb_tile`. It is anchored at the
place centre, so growing the play area round the same centre never renumbers a chunk. The world
bounds are half-open too, [min, max); only chunks that overlap them exist.

Features that cross a chunk border:
- buildings are never split. Each has one owner chunk, where a builder places it once: the chunk
  holding its footprint's centroid (as Wasteland's `cell`), or the chunk it covers most when the
  footprint doesn't reach the centroid's chunk. `chunks` lists every chunk the footprint covers with
  positive area;
- roads are cut by arc length into spans (s0, s1) per chunk, inside the bounds only. Neighbouring
  spans share their end point exactly, so one piece per span leaves no gap. Every piece of positive
  length stays in the chunk it lies in, however short (a line grazing a chunk corner); only cuts
  that coincide numerically (a line through a grid corner) are merged;
- areas and collision outlines list every chunk they cover with positive area; a builder cuts them
  with `clip_polygon` to the chunk's part of the bounds (`cell_in_world`, OutbreakGeo's `bounds_m`).
"""
from __future__ import annotations

import math
import re

import numpy as np
import shapely
from shapely.geometry import MultiLineString, Polygon, box
from shapely.geometry.polygon import orient

DEFAULT_CHUNK_M = 60.0
AREA_EPS = 1e-6        # m²: a smaller overlap with a chunk doesn't make a feature a member
T_EPS = 1e-12          # cut parameters (0…1 along a segment) closer than this are one cut: float noise, not a piece
_ID = re.compile(r'^c(-?\d+)_(-?\d+)$')


def chunk_id(i: int, j: int) -> str:
    return f'c{i}_{j}'


def parse_chunk_id(cid: str) -> tuple[int, int]:
    m = _ID.match(cid)
    if not m:
        raise ValueError(f'not a chunk id: {cid!r}')
    return int(m.group(1)), int(m.group(2))


def to_shape(polygon: dict) -> Polygon:
    """An OutbreakGeo polygon ({"outer": ring, "holes": [ring, …]}) as a valid Shapely geometry."""
    g = Polygon(polygon['outer'], polygon.get('holes') or [])
    return g if g.is_valid else g.buffer(0)


class Grid:
    def __init__(self, bounds, size: float = DEFAULT_CHUNK_M):
        if isinstance(size, bool) or not isinstance(size, (int, float)) or not math.isfinite(size) or size <= 0:
            raise ValueError(f'chunk size must be a positive number of metres, got {size!r}')
        minx, miny, maxx, maxy = (float(v) for v in bounds)
        if not (maxx > minx and maxy > miny):
            raise ValueError(f'empty world bounds {bounds!r}')
        self.size = float(size)
        self.bounds = (minx, miny, maxx, maxy)
        self.world = box(minx, miny, maxx, maxy)
        self.i_range = self._cells(minx, maxx)
        self.j_range = self._cells(miny, maxy)

    def _cell(self, v: float) -> int:
        """The k with k·S <= v < (k+1)·S, judged by the published borders k·S themselves: floor(v / S) can be
        one off at a border when S isn't a power of two (30.1 · 3 / 30.1 isn't 3)."""
        k = math.floor(v / self.size)
        if v < k * self.size:
            k -= 1
        elif v >= (k + 1) * self.size:
            k += 1
        return k

    def _cells(self, lo: float, hi: float) -> range:
        """The cells that overlap [lo, hi) with positive width."""
        last = self._cell(hi)
        if hi == last * self.size:          # hi on a border: that cell starts where the bounds end
            last -= 1
        return range(self._cell(lo), last + 1)

    def index(self, x: float, y: float) -> tuple[int, int]:
        return self._cell(x), self._cell(y)

    def chunks(self) -> list[tuple[int, int]]:
        return [(i, j) for i in self.i_range for j in self.j_range]

    def chunk_bounds(self, i: int, j: int) -> list[float]:
        """The whole grid cell, independent of the bounds."""
        S = self.size
        return [i * S, j * S, (i + 1) * S, (j + 1) * S]

    def cell_in_world(self, i: int, j: int) -> list[float]:
        """The part of the cell inside the bounds: what a builder fills for this chunk (edge chunks are narrower)."""
        x0, y0, x1, y1 = self.chunk_bounds(i, j)
        minx, miny, maxx, maxy = self.bounds
        return [max(x0, minx), max(y0, miny), min(x1, maxx), min(y1, maxy)]

    def contains(self, x: float, y: float) -> bool:
        """Inside the half-open bounds."""
        minx, miny, maxx, maxy = self.bounds
        return minx <= x < maxx and miny <= y < maxy

    @property
    def max_edges(self) -> MultiLineString:
        """The two edges of the bounds that lie outside them (half-open)."""
        minx, miny, maxx, maxy = self.bounds
        return MultiLineString([[(maxx, miny), (maxx, maxy)], [(minx, maxy), (maxx, maxy)]])

    def polygon_chunks(self, g) -> list[tuple[int, int]]:
        """Every chunk the geometry covers with positive area (inside the bounds), sorted."""
        g = g.intersection(self.world)
        if g.is_empty:
            return []
        x0, y0, x1, y1 = g.bounds
        ii = range(max(self._cell(x0), self.i_range.start), min(self._cell(x1), self.i_range.stop - 1) + 1)
        jj = range(max(self._cell(y0), self.j_range.start), min(self._cell(y1), self.j_range.stop - 1) + 1)
        cand = [(i, j) for i in ii for j in jj]
        if not cand:
            return []
        S = self.size
        ci = np.array([c[0] for c in cand], float)
        cj = np.array([c[1] for c in cand], float)
        areas = shapely.area(shapely.intersection(g, shapely.box(ci * S, cj * S, (ci + 1) * S, (cj + 1) * S)))
        return [c for c, a in zip(cand, areas) if a > AREA_EPS]

    def owner(self, g, members: list[tuple[int, int]]) -> tuple[int, int]:
        """The chunk a building is placed in: the one holding its centroid (Wasteland's `cell`), or,
        when the footprint doesn't cover that chunk (a ring round it), the member it covers most."""
        c = g.centroid
        ij = self.index(c.x, c.y)
        if ij in members or not members:
            return ij
        overlap = {m: g.intersection(box(*self.chunk_bounds(*m))).area for m in members}
        return max(sorted(members), key=lambda m: overlap[m])

    def line_spans(self, pts) -> list[tuple[int, int, float, float]]:
        """Cut a polyline by arc length into (i, j, s0, s1) spans, one per stretch inside a chunk and
        inside the bounds, in order along the line. Consecutive pieces in the same chunk merge."""
        S = self.size
        minx, miny, maxx, maxy = self.bounds
        spans = []
        s = 0.0
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            dx, dy = x1 - x0, y1 - y0
            L = math.hypot(dx, dy)
            if L == 0:
                continue
            ts = {0.0, 1.0}
            for a0, d, lo, hi in ((x0, dx, minx, maxx), (y0, dy, miny, maxy)):
                if d == 0:
                    continue
                a_lo, a_hi = min(a0, a0 + d), max(a0, a0 + d)
                # every published border k·S within reach (one cell of margin: the division can be one off)
                cuts = [k * S for k in range(self._cell(a_lo), self._cell(a_hi) + 2)] + [lo, hi]
                for b in cuts:
                    t = (b - a0) / d
                    if 0 < t < 1:
                        ts.add(t)
            merged = []
            for t in sorted(ts):
                if merged and t - merged[-1] < T_EPS:
                    merged[-1] = 1.0 if t == 1.0 else merged[-1]
                    continue
                merged.append(t)
            for ta, tb in zip(merged, merged[1:]):
                mx, my = x0 + dx * (ta + tb) / 2, y0 + dy * (ta + tb) / 2
                if not (minx <= mx < maxx and miny <= my < maxy):
                    continue
                i, j = self.index(mx, my)
                sa, sb = s + ta * L, s + tb * L
                last = spans[-1] if spans else None
                if last is not None and last[3] == sa and (last[0], last[1]) == (i, j):
                    last[3] = sb                       # the same chunk continues (a vertex inside it)
                    continue
                spans.append([i, j, sa, sb])
            s += L
        return [tuple(sp) for sp in spans]


def _lerp(p, q, t):
    if t <= 0:
        return (p[0], p[1])
    if t >= 1:
        return (q[0], q[1])
    return (p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t)


def cut_line(pts, s0: float, s1: float) -> list[tuple[float, float]]:
    """The part of a polyline between arc lengths s0 and s1 (metres from its first point)."""
    out = []
    s = 0.0
    for p, q in zip(pts, pts[1:]):
        L = math.hypot(q[0] - p[0], q[1] - p[1])
        if L == 0:
            continue
        a, b = max(s0, s), min(s1, s + L)
        if a <= b:
            for t in ((a - s) / L, (b - s) / L):
                pt = _lerp(p, q, t)
                if not out or out[-1] != pt:
                    out.append(pt)
        s += L
    return out


def clip_polygon(polygon: dict, cell_bounds) -> list[dict]:
    """An OutbreakGeo polygon cut to a box, normally a chunk's `bounds_m` (its part of the play area, so
    nothing outside the bounds comes back): a list of {"outer", "holes"} pieces."""
    g = to_shape(polygon).intersection(box(*cell_bounds))
    parts = [g] if g.geom_type == 'Polygon' else [p for p in getattr(g, 'geoms', []) if p.geom_type == 'Polygon']
    # The contract's orientation (outer counter-clockwise, holes clockwise), whatever the intersection returns.
    parts = [orient(p, sign=1.0) for p in parts if p.area > AREA_EPS]
    return [{'outer': [list(c) for c in p.exterior.coords[:-1]], 'holes': [[list(c) for c in r.coords[:-1]] for r in p.interiors]}
            for p in parts]
