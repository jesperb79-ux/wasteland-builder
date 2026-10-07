"""city.json (+ osm.json, terrain.json) → OutbreakGeo v1, the place's geography for Outbreak.

    .venv/bin/python pipeline/outbreak <slug | folder with city.json> [--out FILE] [--chunk-size 60]

Wasteland's geographic preprocessing (prepare_city.py) is the cut point. This adapter keeps its semantic
output (footprints, heights, centrelines, ground classes, the terrain grid) and leaves behind what only
the Three.js game needs: rendered meshes (`surfaces`, building `parts`, wall, water and outskirts meshes,
curbs), Wasteland's render look (wall styles, palette colours) and the Mad Max decoration (edge
containers, generated hedges and fences, scattered trees, props). osm.json supplies the mapped values
city.json has turned into render categories (materials, colours) and road traversal tags, and tells
where each value came from. Collision comes from the buildings, not from Wasteland's game outlines.
Format and rules: docs/outbreak-real-city.md.

Provenance is established, not assumed. Wasteland's refinement rounds (overrides.json, custom/*.py,
refinements.md) can hold Google Street View observations and record no source per entry, so a folder
with them is refused outright. And the folder's city.json must equal what prepare_city makes from the
same place.json, osm.json and terrain.json with the same code, without refinements: a city.json built
any other way is refused too.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import LineString, Polygon, box
from shapely.ops import unary_union
from shapely.strtree import STRtree
from shapely.validation import explain_validity

import prepare_city
from common import CITIES, ROOT, Projection
from prepare_city import STYLE, num, r2

from . import schema
from .chunks import AREA_EPS, DEFAULT_CHUNK_M, Grid, chunk_id, cut_line, parse_chunk_id, to_shape

SCHEMA = 'outbreak-geo'
SCHEMA_VERSION = 1
MANIFEST_SCHEMA = 'outbreak-source-manifest'
ADAPTER_VERSION = '0.3.0'
CITY_VERSIONS = {1}
# The local plane OutbreakGeo is in: place.json "projection": "wgs84" (common.Projection). Wasteland's default
# spherical plane is off in scale by up to ~0.7 % (0.24 % east-west in Sweden), so it is refused.
PROJECTION = 'wgs84'
PROJECTION_NAME = 'equirectangular-wgs84'
OSM_ATTRIBUTION = 'Map data © OpenStreetMap contributors (ODbL)'
LEGACY_ENCODING = 'cp1252'      # Wasteland on Windows writes JSON in the ANSI code page (Western: cp1252)
FLOATING_M = 2.5                # a building part starting higher than this doesn't block the ground (prepare_city's rule)
WORLD_TOLERANCE_M = 1e-6
# Highway classes a car can physically drive on. Pedestrian streets are excluded; legal access (access,
# motor_vehicle, oneway …) is carried raw in each road's tags for the game to judge.
DRIVABLE = {'motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'unclassified', 'residential', 'living_street', 'service',
            'track', 'busway', 'road', 'raceway', 'motorway_link', 'trunk_link', 'primary_link', 'secondary_link', 'tertiary_link'}
# OSM tags that decide how a road is traversed and how it meets others, copied verbatim when present (the
# highway and surface values are the road's class and surface), with the mapped lanes and width.
TRAVERSAL_TAGS = ('layer', 'level', 'bridge', 'tunnel', 'covered', 'oneway', 'junction', 'access', 'vehicle', 'motor_vehicle',
                  'motorcar', 'foot', 'bicycle', 'service', 'lanes', 'width')
# The tags that place a road vertically; two roads crossing in the plane where these differ are told apart.
LEVEL_TAGS = ('layer', 'bridge', 'tunnel', 'covered', 'level')
# Files whose content decides the output: Wasteland's preprocessing and this adapter. Pinned by SHA-256.
CODE_FILES = ('pipeline/common.py', 'pipeline/prepare_city.py', 'pipeline/style.json')
# building=* values prepare_city doesn't build (its outline filter, mirrored to read the mapped outlines).
NOT_BUILT = ('no', 'roof', 'construction', 'proposed', 'demolished', 'collapsed')
LEFTOVER_M2 = 1.0               # an outline's area without a building volume above this is reported

ORIGINS = {
    'osm': ("The mapped OpenStreetMap value itself (geometry clipped to the play area and simplified by at most 8 cm).", 'high'),
    'derived': ('Computed from OpenStreetMap data by a fixed rule: buffered centrelines, unions, storeys from a height '
                'or a height from storeys, a tagged height minus the roof, clamped values, terrain sampling.', 'medium'),
    'default': ('A fixed default where OpenStreetMap is silent and the default is nearly always right (no minimum height).', 'medium'),
    'inferred': ("Wasteland's guess where OpenStreetMap is silent: storeys from building kind and size, a flat or "
                 'pitched roof, sidewalks along urban streets.', 'low'),
    'mixed': ('OpenStreetMap data merged with generated content (grass includes the lawns Wasteland draws round small houses).', 'low'),
    'unknown': ('Cannot be told: the source file (osm.json) was not available to the adapter.', 'unknown'),
    'none': ('No value: OpenStreetMap has none.', 'none'),
}
# Wasteland ground layer → OutbreakGeo area class, surface, origin. The layers are disjoint (each piece
# of ground has one class), so they map straight onto terrain materials.
AREA_LAYERS = {
    'asphalt': ('carriageway', 'asphalt', 'derived'),
    'cobble': ('carriageway', 'cobblestone', 'derived'),
    'sidewalk': ('sidewalk', 'paving', 'inferred'),
    'paving': ('pedestrian', 'paving', 'derived'),
    'path': ('path', 'unpaved', 'derived'),
    'parking': ('parking', 'asphalt', 'osm'),
    'park': ('park', None, 'osm'),
    'grass': ('grass', None, 'mixed'),
    'forest': ('forest', None, 'osm'),
    'cemetery': ('cemetery', None, 'osm'),
    'pitch': ('pitch', None, 'osm'),
}
# prepare_city's ground-layer priority (its `layers` list, highest first, the layers city.json keeps): where
# layers overlap after its per-layer simplification, the higher one keeps the ground.
LAYER_PRIORITY = ('asphalt', 'cobble', 'sidewalk', 'paving', 'path', 'parking', 'pitch', 'park', 'grass', 'cemetery', 'forest')
ROAD_CLASSES = {'carriageway', 'pedestrian', 'path'}    # ground classes that may lie over water: bridge decks
SLIVER_M2 = 0.01                # a piece smaller than this, left by resolving an overlap, is dropped and reported
AREA_CLASSES = {
    'water': ('derived', [], 'Sea, lakes and rivers: the play area minus Wasteland\'s land (coastlines, water areas, rivers buffered by width).'),
    'carriageway': ('derived', ['asphalt', 'cobble'], 'Roadway surface: centrelines buffered by their tagged or default width, plus mapped road areas.'),
    'sidewalk': ('inferred', ['sidewalk'], 'Sidewalks along urban streets unless OpenStreetMap says none or maps them separately.'),
    'pedestrian': ('derived', ['paving'], 'Pedestrian streets, squares, paved footways and platforms.'),
    'path': ('derived', ['path'], 'Unpaved paths and tracks.'),
    'parking': ('osm', ['parking'], 'Surface car parks.'),
    'park': ('osm', ['park'], 'Parks and gardens.'),
    'grass': ('mixed', ['grass'], 'Grass and meadow, plus generated lawns round small houses.'),
    'forest': ('osm', ['forest'], 'Woods.'),
    'cemetery': ('osm', ['cemetery'], 'Cemeteries.'),
    'pitch': ('osm', ['pitch'], 'Sports pitches and tracks.'),
}
# Terrain licences, recognised from fetch_terrain.py's attribution text. Anything else is refused.
TERRAIN_LICENCES = (
    ('Markhöjdmodell © Lantmäteriet', {'data': 'Lantmäteriet Markhöjdmodell', 'license': 'CC-BY-4.0',
                                       'url': 'https://creativecommons.org/licenses/by/4.0/'}),
    ('Copernicus DEM GLO-30', {'data': 'Copernicus DEM GLO-30', 'license': 'Copernicus DEM licence (free of charge, attribution required)',
                               'url': 'https://spacedata.copernicus.eu/collections/copernicus-digital-elevation-model'}),
)
_VOLUME_ID = re.compile(r'^([nwr])(\d+)(r?)$')
_OSM_TYPES = {'n': 'node', 'w': 'way', 'r': 'relation'}


class AdapterError(ValueError):
    """The input can't be converted (missing or malformed structure)."""


class PolicyError(AdapterError):
    """The input breaks a source policy: unknown provenance, refinements, an unknown licence."""


# ------------------------------------------------------------------------------------------ small helpers
def _finite(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def rnd(v: float, nd: int) -> float:
    return round(float(v), nd) + 0.0          # + 0.0: no "-0.0" in the output


def _canonical(v) -> str:
    return json.dumps(v, sort_keys=True, separators=(',', ':'))


def _content_id(prefix: str, *parts) -> str:
    return f'{prefix}-' + hashlib.sha1(_canonical(parts).encode()).hexdigest()[:12]


def _point(p):
    if isinstance(p, (list, tuple)) and len(p) == 2 and _finite(p[0]) and _finite(p[1]):
        return [float(p[0]) + 0.0, float(p[1]) + 0.0]
    return None


def _ring(raw, what: str):
    """A ring as [[x, y], …] without the closing point. Raises on anything else."""
    pts = [_point(p) for p in raw] if isinstance(raw, list) else None
    if not pts or any(p is None for p in pts):
        raise AdapterError(f'{what}: a ring must be a list of [x, y] numbers')
    if pts[0] == pts[-1]:
        pts = pts[:-1]
    if len(pts) < 3:
        raise AdapterError(f'{what}: a ring needs three distinct points')
    return pts


def _signed_area(ring) -> float:
    return sum(ax * by - bx * ay for (ax, ay), (bx, by) in zip(ring, ring[1:] + ring[:1])) / 2


def _oriented(ring, ccw: bool):
    """The ring counter-clockwise (outer) or clockwise (hole), keeping its first point."""
    return ring if (_signed_area(ring) > 0) == ccw else ring[:1] + ring[:0:-1]


def _polygon(raw, what: str):
    """(OutbreakGeo polygon, Shapely polygon) from {"outer": ring, "holes": [ring, …]}, oriented. Raises
    unless it is a valid polygon with area: an invalid ring is never repaired behind the reader's back."""
    if not isinstance(raw, dict) or not isinstance(raw.get('holes', []), list):
        raise AdapterError(f'{what}: a polygon must be {{"outer": ring, "holes": [ring, ...]}}')
    outer = _ring(raw.get('outer'), what)
    holes = [_ring(h, f'{what} (hole)') for h in raw.get('holes') or []]
    poly = {'outer': _oriented(outer, True), 'holes': [_oriented(h, False) for h in holes]}
    shape = Polygon(poly['outer'], poly['holes'])
    if not shape.is_valid:
        raise AdapterError(f'{what}: not a valid polygon ({explain_validity(shape)})')
    if shape.area <= AREA_EPS:
        raise AdapterError(f'{what}: the polygon has no area')
    return poly, shape


def _pieces(g) -> list[Polygon]:
    """The polygons with area in any Shapely geometry."""
    parts = [g] if g.geom_type == 'Polygon' else list(getattr(g, 'geoms', []))
    return [p for p in parts if p.geom_type == 'Polygon' and p.area > AREA_EPS]


def _from_shape(p: Polygon, exact: bool = False) -> dict:
    """A computed Shapely polygon as an OutbreakGeo polygon: millimetres when that keeps it valid, full
    precision when `exact` (collision must cover the footprints it is checked against)."""
    rings = [list(p.exterior.coords)[:-1]] + [list(r.coords)[:-1] for r in p.interiors]
    full = lambda v: float(v) + 0.0                                              # noqa: E731
    for fix in ((full,) if exact else (lambda v: rnd(v, 3), full)):
        out = [[[fix(x), fix(y)] for x, y in ring] for ring in rings]
        poly = {'outer': _oriented(out[0], True), 'holes': [_oriented(h, False) for h in out[1:]]}
        shape = Polygon(poly['outer'], poly['holes'])
        if shape.is_valid and shape.area > AREA_EPS:
            return poly
    return poly


def _tag(tags: dict, *keys):
    """The first non-empty value among the keys, verbatim."""
    for k in keys:
        v = tags.get(k)
        if v not in (None, ''):
            return str(v)
    return None


def _tag_min_height(tags: dict) -> float:
    """Where a building's walls start over the ground, by prepare_city's rule for volumes: min_height, else
    building:min_level storeys. The same rule decides whether an outline stands on the ground."""
    return num(tags.get('min_height')) or (num(tags.get('building:min_level')) or 0) * STYLE['storey_m']


def _at_ground(tags: dict):
    """Whether a covered way runs at ground level: layer 0 (or untagged) and level 0 among its levels (or
    untagged). False above or below the ground; None when layer or level can't be read."""
    try:
        layer = int(str(tags.get('layer', '0')).strip())
        levels = [float(v) for v in re.split(r'[;,]', str(tags.get('level', '0'))) if v.strip()]
    except ValueError:
        return None
    return layer == 0 and 0.0 in levels


def _osm_ref(fid: str):
    """'w123' → ({'type': 'way', 'id': 123}, False); 'w123r' (an outline minus its parts) → (…, True)."""
    m = _VOLUME_ID.match(fid)
    if not m:
        return None, False
    return {'type': _OSM_TYPES[m.group(1)], 'id': int(m.group(2))}, bool(m.group(3))


def _bbox_dict(b):
    if isinstance(b, list) and len(b) == 4 and all(_finite(v) for v in b) and b[0] < b[2] and b[1] < b[3]:
        return {'south': b[0], 'west': b[1], 'north': b[2], 'east': b[3]}
    return None


# ------------------------------------------------------------------------------------------ Wasteland refinements
# overrides.json "defaults" that only choose how the place is built, never what it looks like.
BUILD_SETTINGS = {'terrain': lambda v: isinstance(v, bool), 'terrain_scale': lambda v: _finite(v) and v > 0}
REFUSAL = ('Wasteland refinement data in this city: {}. Refinement rounds can take such facts from Google Street View '
           "and Wasteland doesn't record where each entry came from, so OutbreakGeo can't use a city.json built with "
           "them, and no option changes that. Build the Outbreak source in a folder refine-city hasn't touched "
           '(python3 wasteland.py new ... then build <slug> --only fetch,terrain,prepare). Corrections belong in '
           'OutbreakOverrides, which records a basis and a source per entry.')


def wasteland_refinements(overrides) -> list[str]:
    """What in a parsed overrides.json changes the place's data rather than a build setting: building, road
    and area entries, other defaults (max_levels caps storeys) and any key this adapter doesn't know."""
    if overrides is None:
        return []
    if not isinstance(overrides, dict):
        return ['overrides.json (not an object)']
    found = []
    for key in sorted(overrides):
        value = overrides[key]
        if key == 'defaults' and isinstance(value, dict):
            found += [f'overrides.json defaults.{k}' for k in sorted(value) if k not in BUILD_SETTINGS or not BUILD_SETTINGS[k](value[k])]
        elif value or key == 'defaults':              # an empty "buildings": {} or "areas": [] changes nothing
            found.append(f'overrides.json {key}' + (f' ({len(value)})' if isinstance(value, (list, dict)) else ''))
    return found


def refinement_traces(folder: Path) -> list[str]:
    """refine-city's other files. They don't feed city.json, but they show the folder has been refined."""
    found = [f'custom/{p.name}' for p in sorted((folder / 'custom').glob('*.py'))] if (folder / 'custom').is_dir() else []
    return found + (['refinements.md'] if (folder / 'refinements.md').exists() else [])


# prepare_city in a child process in Python's UTF-8 mode, with Wasteland's city folder pointed at a
# temporary one: its files are read and written as UTF-8 whatever the locale, and nothing leaks into
# this process.
_REBUILD = ('import sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); import common; '
            'common.CITIES = Path(sys.argv[2]); import prepare_city; prepare_city.main(["city"])')


def clean_rebuild(folder: Path, settings: dict) -> dict:
    """city.json as prepare_city makes it from the folder's place.json, osm.json and terrain.json and the
    build settings, in a temporary folder without any refinement. Nothing in `folder` changes. The inputs
    are decoded by this adapter's rule and written as ASCII JSON, and prepare_city runs in UTF-8 mode, so
    the result doesn't depend on the locale."""
    missing = [n for n in ('place.json', 'osm.json') if not (folder / n).exists()]
    if missing:
        raise PolicyError(f"can't verify city.json without {' and '.join(missing)}: its provenance can't be established")
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / 'city'
        work.mkdir()
        for name in ('place.json', 'osm.json', 'terrain.json'):
            if (folder / name).exists():
                (work / name).write_text(json.dumps(read_json(folder / name, [])[0]), encoding='ascii')
        if settings:
            (work / 'overrides.json').write_text(json.dumps({'defaults': settings}), encoding='ascii')
        proc = subprocess.run([sys.executable, '-X', 'utf8', '-c', _REBUILD, str(ROOT / 'pipeline'), tmp],
                              capture_output=True, text=True, encoding='utf-8', errors='replace')
        if proc.returncode or not (work / 'city.json').exists():
            raise AdapterError(f'the clean rebuild of city.json failed: {(proc.stderr or proc.stdout).strip()[-600:]}')
        return read_json(work / 'city.json', [])[0]


# ------------------------------------------------------------------------------------------ the adapter
class _Adapter:
    def __init__(self, city, osm, settings: dict, grid: Grid, proj: Projection, warnings: list):
        self.city, self.settings, self.grid, self.proj, self.warnings = city, settings, grid, proj, warnings
        self.world = grid.world.buffer(WORLD_TOLERANCE_M, join_style=2)
        self.tag_index, self.way_nodes, self.osm_reader = None, {}, None
        self.unmatched_nodes = []   # roads whose osm.json node ids don't line up with their centreline
        if isinstance(osm, dict):
            if not isinstance(osm.get('elements'), list):
                raise AdapterError('osm.json has no "elements" list')
            self.tag_index, seen = {}, {}
            for el in osm['elements']:
                if isinstance(el, dict) and el.get('type') in ('node', 'way', 'relation'):
                    key = (el['type'], el.get('id'))
                    # One OSM identity, one content: picking one of two would depend on the order of osm.json.
                    if key in seen and seen[key] != _canonical(el):
                        raise AdapterError(f'osm.json has two different entries for {el["type"]} {el.get("id")} '
                                           '(tags, nodes or geometry differ); refusing to pick one')
                    seen[key] = _canonical(el)
                    self.tag_index[key] = el.get('tags') or {}
                    nodes, geom = el.get('nodes'), el.get('geometry')
                    # A way's node ids beside its geometry, one per point (fetch_osm.py and Overpass keep both).
                    if el['type'] == 'way' and isinstance(nodes, list) and isinstance(geom, list) and len(nodes) == len(geom) \
                            and all(isinstance(g, dict) and _finite(g.get('lat')) and _finite(g.get('lon')) for g in geom) \
                            and all(_int(n) and n > 0 for n in nodes):
                        self.way_nodes[el.get('id')] = (nodes, geom)
            self.osm_reader = prepare_city.OSM(osm, proj)
        self.missing_tags = 0
        self.passages = []          # building passages and covered ways, cut out of building collision

    def warn(self, msg):
        self.warnings.append(msg)

    def tags(self, ref):
        """Raw OSM tags of an element, or None when they can't be known."""
        if self.tag_index is None or ref is None:
            return None
        tags = self.tag_index.get((ref['type'], ref['id']))
        if tags is None:
            self.missing_tags += 1
        return tags

    def inside(self, shape, what):
        if not self.world.contains(shape):
            raise AdapterError(f'{what} extends outside the play area')

    def members(self, shape, what):
        members = self.grid.polygon_chunks(shape)
        if not members:
            raise AdapterError(f'{what} covers no chunk')
        return members

    def unique(self, items, kind, geom_key):
        """(feature id, item) pairs: city.json ids, with #1, #2 … where one OSM element gave several pieces (a
        multipolygon, an outline cut by the play-area edge), numbered by geometry, then by content. Exact
        duplicates collapse to one; two entries with one id and one geometry but different values are refused."""
        if not isinstance(items, list):
            raise AdapterError(f'city.json "{kind}s" must be a list')
        groups = defaultdict(dict)
        for it in items:
            if not isinstance(it, dict) or not isinstance(it.get('id'), str) or not it['id'] or it['id'] != ''.join(it['id'].split()):
                raise AdapterError(f'{kind}: an entry without a valid id')
            groups[it['id']].setdefault(_canonical(it), it)
        out = []
        for fid in sorted(groups):
            n = sum(1 for it in items if isinstance(it, dict) and it.get('id') == fid)
            group = sorted(groups[fid].values(), key=lambda it: (_canonical(geom_key(it)), _canonical(it)))
            if n > len(group):
                self.warn(f'{kind} {fid}: {n - len(group)} exact duplicate(s) in city.json dropped')
            geoms = [_canonical(geom_key(it)) for it in group]
            if len(set(geoms)) < len(geoms):
                raise AdapterError(f'{kind} {fid}: entries with the same geometry but different values')
            if len(group) == 1:
                out.append((fid, group[0]))
                continue
            out += [(f'{fid}#{k}', it) for k, it in enumerate(group, 1)]
            self.warn(f'{kind} {fid}: {len(group)} pieces in city.json, numbered {fid}#1…#{len(group)}')
        return out

    # ---------------------------------------------------------------- buildings
    def building(self, fid, b):
        base_id = fid.split('#')[0]
        what = f'building {fid}'
        footprint, shape = _polygon(b.get('footprint'), what)
        self.inside(shape, what)
        wall_h, levels = b.get('h'), b.get('levels')
        if not (_finite(wall_h) and wall_h > 0) or not (_int(levels) and levels >= 1):
            raise AdapterError(f'{what}: height or storeys missing or malformed')
        members = self.members(shape, what)
        owner = self.grid.owner(shape, members)

        ref, remainder = _osm_ref(base_id)
        raw = self.tags(ref)
        known = raw is not None
        tags = raw or {}

        def tag_num(key):
            return num(tags.get(key)) if known else None

        def exact(key, value):
            """The tag holds this very value (not clamped, rounded or reduced by a roof height)."""
            v = tag_num(key)
            return v is not None and abs(v - value) < 0.01

        fallback = 'inferred' if known else 'unknown'
        has_h, has_l = tag_num('height') is not None, tag_num('building:levels') is not None
        height_o = 'osm' if has_h and exact('height', wall_h) else 'derived' if has_h or has_l else fallback
        levels_o = 'osm' if has_l and exact('building:levels', levels) else 'derived' if has_h or has_l else fallback
        if tag_num('min_height') is not None or tag_num('building:min_level') is not None:
            min_h = _tag_min_height(tags)
            min_o = 'osm' if tag_num('min_height') or tag_num('building:min_level') is None else 'derived'
        else:
            min_h, min_o = (0.0, 'default') if known else (None, 'unknown')
        if min_h is not None and min_h >= wall_h:
            raise AdapterError(f'{what}: walls start at {min_h:g} m, not below their top at {wall_h:g} m')

        kind = str(b.get('kind') or 'yes')
        # prepare_city turns an untyped building with amenity=place_of_worship into a church
        kind_o = 'unknown' if not known else (
            'derived' if kind == 'church' and (tags.get('building') or tags.get('building:part') or 'yes') == 'yes' else 'osm')
        part = 'remainder' if remainder else ('building' if 'building' in tags else 'part' if 'building:part' in tags else 'unknown')
        name = str(b.get('name') or '')
        name_o = 'none' if not name else 'osm' if known else 'unknown'

        def mapped(*keys):
            value = _tag(tags, *keys)
            return value, 'osm' if value is not None else 'none' if known else 'unknown'

        # The mapped values, verbatim. Wasteland's wall styles and palette colours are render choices: not here.
        facade_material, facade_material_o = mapped('building:material', 'building:facade:material')
        facade_colour, facade_colour_o = mapped('building:colour', 'colour')
        roof_material, roof_material_o = mapped('roof:material')
        roof_colour, roof_colour_o = mapped('roof:colour')
        roof_shape, shape_o = mapped('roof:shape')
        if roof_shape is None:
            # Wasteland picks a flat or pitched roof where OSM is silent; its roof_style says which.
            roof_style = b.get('roof_style')
            roof_shape = 'flat' if roof_style == 'flat' else 'pitched' if roof_style in ('tiles', 'metal', 'slate') else None
            shape_o = 'unknown' if roof_shape is None else fallback

        base = b.get('base') if _finite(b.get('base')) else None
        base_min = b.get('base_min') if _finite(b.get('base_min')) else None
        landmark = bool(b.get('landmark'))
        return {
            'id': fid, 'osm': ref, 'part': part, 'name': name, 'kind': kind,
            'footprint': footprint, 'area_m2': rnd(shape.area, 2),
            'wall_height_m': float(wall_h), 'min_height_m': None if min_h is None else rnd(min_h, 2), 'levels': levels,
            'roof': {'shape': roof_shape, 'material': roof_material, 'colour': roof_colour},
            'facade': {'material': facade_material, 'colour': facade_colour},
            'base_m': base, 'base_min_m': base_min, 'landmark': landmark,
            'lod': {'class': 'A', 'basis': 'landmark'} if landmark else {'class': None, 'basis': 'unassigned'},
            'chunk': chunk_id(*owner), 'chunks': [chunk_id(*m) for m in members],
            'provenance': {
                'footprint': 'derived' if remainder else 'osm', 'kind': kind_o, 'name': name_o,
                'wall_height': height_o, 'levels': levels_o, 'min_height': min_o,
                'roof_shape': shape_o, 'roof_material': roof_material_o, 'roof_colour': roof_colour_o,
                'facade_material': facade_material_o, 'facade_colour': facade_colour_o,
                'base': 'derived' if base is not None else 'none', 'landmark': 'derived',
            },
        }

    # ---------------------------------------------------------------- roads
    def road(self, fid, r):
        what = f'road {fid}'
        pts = [_point(p) for p in r.get('p')] if isinstance(r.get('p'), list) else None
        if not pts or len(pts) < 2 or any(p is None for p in pts):
            raise AdapterError(f'{what}: the centreline must be a list of at least two [x, y] numbers')
        width = r.get('w')
        if not (_finite(width) and width > 0):
            raise AdapterError(f'{what}: width missing or malformed')
        spans = self.grid.line_spans(pts)
        if not spans:
            self.warn(f'{what}: no part inside the half-open play area; skipped')
            return None
        ref, _ = _osm_ref(fid.split('#')[0])
        raw = self.tags(ref)
        known, tags = raw is not None, raw or {}
        name = str(r.get('name') or '')
        kind = str(r.get('kind') or '')
        surface = str(r.get('surface') or '') or None

        def origin(from_osm):
            return from_osm if known else 'unknown'

        # prepare_city rounds widths to 0.1 m and clamps them; a clamped tag is no longer the mapped value
        tagged_w = num(tags.get('width'))
        width_tag = ('osm' if tagged_w is not None and abs(tagged_w - width) <= 0.05 + 1e-9
                     else 'derived' if tagged_w is not None or num(tags.get('lanes')) is not None else 'inferred')
        layer = None
        if known:
            try:
                layer = int(str(tags.get('layer', '0')).strip())
            except ValueError:
                self.warn(f'{what}: layer "{tags.get("layer")}" is not a number; layer unknown')
        if known and (tags.get('tunnel') == 'building_passage' or tags.get('covered') == 'yes'):
            ground = _at_ground(tags)
            if ground:                           # cut out of building collision as prepare_city cuts its outlines
                self.passages.append(LineString(pts).buffer(width / 2 + 0.3, cap_style=2))
            elif ground is None:
                self.warn(f'{what}: a covered way whose layer or level can\'t be read; it is not cut out of building collision')
        return {
            'id': fid, 'osm': ref, 'name': name, 'class': kind, 'drivable': kind in DRIVABLE,
            'width_m': float(width), 'surface': surface, 'centerline': pts, 'length_m': sum(math.dist(a, b) for a, b in zip(pts, pts[1:])),
            'nodes': self.road_nodes(fid, ref, pts),
            'layer': layer, 'tags': {k: str(tags[k]) for k in TRAVERSAL_TAGS if k in tags},
            'chunks': sorted({chunk_id(i, j) for i, j, _, _ in spans}, key=parse_chunk_id),
            'spans': [{'chunk': chunk_id(i, j), 's0_m': s0, 's1_m': s1} for i, j, s0, s1 in spans],
            'provenance': {
                'geometry': 'osm', 'class': origin('osm'), 'width': origin(width_tag),
                'surface': origin('osm' if tags.get('surface') else 'none'),
                'name': 'none' if not name else origin('osm'), 'traversal': origin('osm'),
            },
        }

    def road_nodes(self, fid, ref, pts):
        """The OSM node id of each centreline vertex, or None. Ids are attached only where the way in osm.json
        lines up with the centreline point for point: its geometry, projected and rounded as prepare_city does,
        must be exactly the centreline, so an id can't land on the wrong vertex or on a road it doesn't belong to."""
        if not ref or ref['type'] != 'way' or ref['id'] not in self.way_nodes:
            return None
        nodes, geom = self.way_nodes[ref['id']]
        expected = [[r2(x) + 0.0, r2(y) + 0.0] for x, y in (self.proj.xy(g['lat'], g['lon']) for g in geom)]
        if len(nodes) != len(pts) or expected != pts:
            self.unmatched_nodes.append(fid)
            return None
        return list(nodes)

    # ---------------------------------------------------------------- areas, water, collision, junctions
    def polygon_feature(self, prefix, raw_poly, extra, what):
        poly, shape = _polygon(raw_poly, what)
        self.inside(shape, what)
        return {'id': _content_id(prefix, extra, poly), **extra, 'polygon': poly, 'area_m2': rnd(shape.area, 2),
                'chunks': [chunk_id(*m) for m in self.members(shape, what)]}

    def areas(self):
        """Ground classes, disjoint. prepare_city makes its layers disjoint and then simplifies each one on its
        own (0.8 m), so neighbouring layers can overlap a little in city.json; where they do, the layer
        prepare_city ranks higher keeps the ground (LAYER_PRIORITY). Land classes other than roads are cut
        to the land; road classes may lie over water (bridge decks). Water is the play area minus the land."""
        layers = self.city.get('areas', {})
        if not isinstance(layers, dict):
            raise AdapterError('city.json "areas" must be an object of ground layers')
        parsed = {}
        for layer in sorted(layers):
            if layer not in AREA_LAYERS:
                raise AdapterError(f'city.json areas: unknown ground layer "{layer}" (city.json version 1 has {sorted(AREA_LAYERS)})')
            if not isinstance(layers[layer], list):
                raise AdapterError(f'city.json areas.{layer} must be a list of polygons')
            parsed[layer] = {}
            for k, raw in enumerate(layers[layer]):
                poly, shape = _polygon(raw, f'areas.{layer}[{k}]')
                self.inside(shape, f'areas.{layer}[{k}]')
                if _canonical(poly) in parsed[layer]:
                    self.warn(f'areas.{layer}: polygon {_content_id(layer, poly)} appears more than once in city.json; kept once')
                parsed[layer][_canonical(poly)] = (poly, shape)
        land = self.city.get('land')
        water = None
        if land is None:
            self.warn('city.json has no "land": no water areas')
        elif not isinstance(land, list):
            raise AdapterError('city.json "land" must be a list of polygons')
        else:
            shapes = [_polygon(p, f'land[{k}]')[1] for k, p in enumerate(land)]
            water = self.grid.world.difference(unary_union(shapes)) if shapes else self.grid.world

        out, slivers = {}, []

        def emit(cls, extra, poly, what):
            f = self.polygon_feature(cls, poly, extra, what)
            if f['id'] in out:
                self.warn(f'areas: polygon {f["id"]} appears more than once after resolving overlaps; kept once')
            out[f['id']] = f

        taken = Polygon()
        for layer in LAYER_PRIORITY:
            if layer not in parsed:
                continue
            cls, surface, origin = AREA_LAYERS[layer]
            extra = {'class': cls, 'surface': surface, 'origin': origin}
            for key in sorted(parsed[layer]):
                poly, shape = parsed[layer][key]
                cut = [g for g in (taken, None if cls in ROAD_CLASSES else water) if g is not None and shape.intersection(g).area > 0]
                if not cut:
                    emit(cls, extra, poly, f'areas.{layer}')
                    continue
                for p in _pieces(shape.difference(unary_union(cut))):
                    if p.area < SLIVER_M2:
                        slivers.append(p.area)
                    else:
                        emit(cls, extra, _from_shape(p, exact=True), f'areas.{layer}')
            taken = unary_union([taken] + [shape for _, shape in parsed[layer].values()])
        if slivers:
            self.warn(f'areas: {len(slivers)} slivers under {SLIVER_M2:g} m² left by resolving overlaps between ground classes '
                      f'({sum(slivers):.4f} m²)')
        if water is not None:
            pieces = _pieces(water)
            small = [p for p in pieces if p.area < 1.0]
            if small:
                self.warn(f'water: {len(small)} slivers under 1 m² along the shore left out ({sum(p.area for p in small):.2f} m²)')
            for p in pieces:
                if p.area >= 1.0:
                    emit('water', {'class': 'water', 'surface': None, 'origin': 'derived'}, _from_shape(p, exact=True), 'water')
        return sorted(out.values(), key=lambda f: (f['class'], f['id']))

    def outlines(self):
        """The mapped building outlines (building=*), read as prepare_city reads them (clipped 0.5 m inside
        the bounds, pieces under 4 m² dropped), holes kept: {"w12": (osm ref, tags, geometry)}, or None
        without osm.json."""
        if self.osm_reader is None:
            return None
        inner = box(*self.grid.bounds).buffer(-0.5)
        found = defaultdict(list)
        try:
            for el, tags, g in self.osm_reader.areas(lambda t: 'building' in t and t['building'] not in NOT_BUILT):
                ref = {'type': el['type'], 'id': el['id']}
                pieces = prepare_city.clean(g.intersection(inner), 4.0)
                if pieces:                       # an outline outside the play area (the download has a margin) has none
                    found[f'{el["type"][0]}{el["id"]}'] += [(ref, tags, p) for p in pieces]
        except (KeyError, TypeError, ValueError) as exc:
            raise AdapterError(f'osm.json is malformed: {exc!r}') from None
        return {fid: (pieces[0][0], pieces[0][1], unary_union([p for _, _, p in pieces])) for fid, pieces in found.items()}

    def collision(self, buildings):
        """What blocks movement at ground level, each piece tied to its OSM outline and building volumes.

        A mapped outline standing on the ground blocks with its whole extent together with the ground-
        standing volumes inside it (holes kept, building passages and covered ways at ground level cut
        out): what city.json leaves out of its buildings, such as an outline's leftover beside its parts
        (prepare_city drops one under 20 % of the outline), still blocks and is reported. "Standing on the
        ground" is one rule for outlines and volumes: walls starting at most 2.5 m up (min_height, else
        building:min_level storeys). A ground-standing volume no outline holds (a building:part mapped
        alone; every volume without osm.json) blocks on its own. City walls block."""
        passage = unary_union(self.passages) if self.passages else None
        ground = [b for b in buildings if b['min_height_m'] is None or b['min_height_m'] <= FLOATING_M]
        shapes = {b['id']: Polygon(b['footprint']['outer'], b['footprint']['holes']) for b in buildings}
        out, held = {}, set()

        def add(cid, kind, ref, members, shape):
            if passage is not None:
                shape = shape.difference(passage)
            polys = sorted((_from_shape(p, exact=True) for p in _pieces(shape)), key=_canonical)
            for k, poly in enumerate(polys, 1):
                pid = cid + (f':{k}' if len(polys) > 1 else '')
                if pid in out:
                    raise AdapterError(f'collision {pid}: two pieces with one id')
                out[pid] = {'id': pid, 'kind': kind, 'osm': ref, 'buildings': members, 'polygon': poly,
                            'chunks': [chunk_id(*m) for m in self.members(to_shape(poly), pid)]}

        outlines = self.outlines()
        if outlines is None:
            if buildings:
                self.warn('osm.json not available: collision follows the building volumes; mapped outlines and building passages are unknown')
        else:
            for fid in sorted(outlines):
                ref, tags, outline = outlines[fid]
                if _tag_min_height(tags) > FLOATING_M:
                    continue                     # a raised outline (a covered bridge); its ground parts block on their own
                # Only the ground-standing volumes inside count: an overhang never blocks the ground below it.
                members = sorted(b['id'] for b in ground if outline.contains(shapes[b['id']].representative_point()))
                volumes = unary_union([shapes[m] for m in members]) if members else Polygon()
                leftover = outline.difference(volumes.buffer(0.25, join_style=2)).area
                if leftover > LEFTOVER_M2:
                    self.warn(f'building outline {fid}: {leftover:.1f} m² of it has no building volume in city.json '
                              '(prepare_city drops an outline\'s leftover beside its parts when under 20 %); collision covers it')
                held |= set(members)
                add(f'col:{fid}', 'building', ref, members, outline.union(volumes))
        for b in ground:
            if b['id'] not in held:
                add(f'col:{b["id"]}', 'building', b['osm'], [b['id']], shapes[b['id']])
        walls = self.city.get('walls') or []
        if not isinstance(walls, list):
            raise AdapterError('city.json "walls" must be a list')
        for k, w in enumerate(walls):
            if not isinstance(w, dict):
                raise AdapterError(f'city.json walls[{k}] must be an object')
            if w.get('kind') != 'city_wall':
                continue                         # hedges and fences Wasteland generates round gardens
            _, shape = _polygon({'outer': w.get('poly'), 'holes': []}, f'city wall walls[{k}]')
            for p in _pieces(shape.intersection(self.grid.world)):
                poly = _from_shape(p)
                cid = 'col:' + _content_id('wall', poly)
                if cid in out:
                    self.warn(f'city wall {cid} appears more than once in city.json; kept once')
                out[cid] = {'id': cid, 'kind': 'city_wall', 'osm': None, 'buildings': [], 'polygon': poly,
                            'chunks': [chunk_id(*m) for m in self.members(to_shape(poly), cid)]}
        return sorted(out.values(), key=lambda f: f['id'])

    def topology(self, roads):
        """How far road connectivity is known: OSM node ids for every road, for some, or for none."""
        with_nodes = sum(1 for r in roads if r['nodes'] is not None)
        if self.unmatched_nodes:
            self.warn(f'{len(self.unmatched_nodes)} roads have node ids in osm.json that don\'t line up with their centreline '
                      f'point for point; their ids are not used: {", ".join(sorted(self.unmatched_nodes)[:20])}')
        if with_nodes < len(roads):
            self.warn(f'{len(roads) - with_nodes} roads have no usable OSM node ids: where they touch other roads at the same '
                      'level the junction is marked shared_position, not verified')
        return 'none' if not roads else 'osm_nodes' if with_nodes == len(roads) else 'positions' if not with_nodes else 'mixed'

    # ---------------------------------------------------------------- terrain
    def terrain(self, terrain_src):
        t = self.city.get('terrain')
        if t is None:
            self.warn('no terrain in city.json: the city is flat (heights are 0)')
            return None
        if not isinstance(t, dict):
            raise AdapterError('city.json "terrain" must be an object or null')
        x0, y0, step, n, d = (t.get(k) for k in ('x0', 'y0', 'step', 'n', 'd'))
        if not (_finite(x0) and _finite(y0) and _finite(step) and step > 0 and _int(n) and n >= 2
                and isinstance(d, list) and len(d) == n * n and all(_finite(v) for v in d)):
            raise AdapterError('city.json "terrain" is malformed (needs x0, y0, step > 0, n ≥ 2 and n×n finite heights in d)')
        if not raster_covers(self.grid.bounds, x0, y0, step, n):
            raise AdapterError(f'city.json "terrain" covers x {x0:g}…{x0 + (n - 1) * step:g}, y {y0:g}…{y0 + (n - 1) * step:g} m, '
                               f'not the whole play area {self.grid.bounds}')
        scale = float(self.settings.get('terrain_scale', 1.0))
        lake = t.get('lake')
        water_z = self.city.get('water_z', STYLE['water_z'])
        if not _finite(water_z):
            raise AdapterError('city.json "water_z" must be a number')
        return {
            'source': t.get('source'), 'kind': terrain_src.get('kind') if isinstance(terrain_src, dict) else None,
            'attribution': t.get('attribution'),
            'processing': ("Wasteland prepare_city: resampled to this grid; a surface model (dsm) is cleaned of mapped buildings "
                           "and forests, opened and blurred, a ground model (dtm) only lightly blurred; water is flattened and "
                           "shores eased down to it."),
            'grid': {'origin_m': [float(x0), float(y0)], 'step_m': float(step), 'n': n,
                     'layout': 'row-major: heights_m[j*n + i] is the height at x = origin_x + i*step_m, y = origin_y + j*step_m'},
            'heights_m': [float(v) for v in d],
            'vertical_datum': {
                'zero': "the flat city's ground level, 1.2 m above the main water surface",
                'water_surface_m': float(water_z), 'scale': scale,
                'source_height_of_zero_m': rnd(lake - water_z, 3) if _finite(lake) else None,
                'to_source_height': 'approximately heights_m / scale + source_height_of_zero_m (before clamping and shore easing)',
            },
            'range_in_bounds_m': terrain_range(self.grid.bounds, x0, y0, step, n, d),
        }


def vertical_level(road: dict):
    """Where a road lies vertically, from its OSM tags: (layer, level). The layer is the tagged one, else 1 on a
    bridge, −1 in a tunnel (a building passage runs at the ground), else 0; None when it can't be read (a
    malformed layer) or the tags are unknown (no osm.json). The level is the indoor level when tagged other
    than 0. A covered way stays at its layer: a roof over a street doesn't lift it."""
    t = road['tags']
    if 'layer' in t:
        z = road['layer']
    elif road['layer'] is None:
        z = None
    elif t.get('bridge', 'no') != 'no':
        z = 1
    elif t.get('tunnel', 'no') not in ('no', 'building_passage'):
        z = -1
    else:
        z = 0
    level = t.get('level')
    return (z, None if level in (None, '', '0') else level)


def junctions(roads, grid: Grid) -> list[dict]:
    """Where roads meet, inside the bounds.

    basis osm_node (verified): an OSM node two or more roads share. That is topology, nothing else is:
    roads that only cross in the plane (a bridge over a street) don't meet, even at one position.
    basis shared_position (not verified): where some road at a position has no usable node ids, the roads at
    that position on the same vertical level (vertical_level) are possibly connected; node null. Roads on
    different levels at one position never form one (they are a crossing, see crossings)."""
    at = defaultdict(list)
    for r in roads:
        for (x, y), n in zip(r['centerline'], r['nodes'] or [None] * len(r['centerline'])):
            if grid.contains(x, y):
                at[(x, y)].append((r, n))
    out = []
    for (x, y), hits in at.items():
        by_node = defaultdict(set)
        for r, n in hits:
            if n is not None:
                by_node[n].add(r['id'])
        out += [{'id': f'j:n{n}', 'point': [x, y], 'roads': sorted(ids), 'node': n, 'basis': 'osm_node'}
                for n, ids in by_node.items() if len(ids) > 1]
        if not any(n is None for _, n in hits):
            continue
        groups, unknown = defaultdict(set), defaultdict(bool)
        for r, n in hits:
            groups[vertical_level(r)].add(r['id'])
            unknown[vertical_level(r)] |= n is None
        # Only where a road on that level has no node id here: roads that all have ids meet by them or not at all.
        levels = sorted((lv for lv, ids in groups.items() if len(ids) > 1 and unknown[lv]), key=_canonical)
        for k, lv in enumerate(levels, 1):
            suffix = f'#{k}' if len(levels) > 1 else ''
            out.append({'id': f'j:{x:.2f},{y:.2f}{suffix}', 'point': [x, y], 'roads': sorted(groups[lv]), 'node': None,
                        'basis': 'shared_position'})
    return sorted(out, key=lambda j: j['id'])


def _meeting_points(g) -> list[tuple]:
    """The points where two centrelines meet: each point, and both ends of a stretch they share."""
    out = []
    for part in shapely.get_parts(g):
        if part.is_empty:
            continue
        if part.geom_type == 'Point':
            out.append((part.x, part.y))
        else:
            cs = list(part.coords)
            out += [cs[0], cs[-1]]
    return out


def crossings(roads, grid: Grid, joined=None) -> list[dict]:
    """Where two roads' centrelines meet in the plane inside the bounds without being joined there: not at a
    node both share (osm_node junction) and not in one shared_position junction. Each is a fact a road
    builder needs, never a junction: status separated when the two lie on different vertical levels (layer,
    bridge, tunnel, indoor level; `differ` names the tags whose values differ), unresolved when the source
    puts them on one level, or can't tell, yet they share no node (an unmapped junction, or roads that
    really pass each other)."""
    if joined is None:
        joined = junctions(roads, grid)
    together = defaultdict(list)                     # (road, road) → the points where a junction joins them
    for j in joined:
        for p in j['roads']:
            for q in j['roads']:
                if p < q:
                    together[(p, q)].append(j['point'])
    ordered = sorted(roads, key=lambda r: r['id'])
    lines = [LineString(r['centerline']) for r in ordered]
    tree = STRtree(lines)
    out = {}
    for a, ga in enumerate(lines):
        for b in sorted(int(k) for k in tree.query(ga)):
            if b <= a:
                continue
            ra, rb = ordered[a], ordered[b]
            for x, y in _meeting_points(ga.intersection(lines[b])):
                x, y = float(x) + 0.0, float(y) + 0.0
                # A junction point is a shared vertex; GEOS returns it exactly, the tolerance is only a guard.
                if not grid.contains(x, y) or any(math.dist((x, y), p) <= 1e-6 for p in together.get((ra['id'], rb['id']), ())):
                    continue
                la, lb = vertical_level(ra), vertical_level(rb)
                separated = None not in (la[0], lb[0]) and la != lb
                differ = sorted(k for k in LEVEL_TAGS if ra['tags'].get(k) != rb['tags'].get(k))
                cid = f'x:{ra["id"]}:{rb["id"]}:{x:.3f},{y:.3f}'
                out.setdefault(cid, {'id': cid, 'point': [x, y], 'roads': [ra['id'], rb['id']],
                                     'status': 'separated' if separated else 'unresolved', 'differ': differ})
    return [out[k] for k in sorted(out)]


def raster_covers(bounds, x0, y0, step, n) -> bool:
    """Whether a height grid spans the whole play area, so every point of it has heights round it."""
    minx, miny, maxx, maxy = bounds
    return x0 <= minx and y0 <= miny and x0 + (n - 1) * step >= maxx and y0 + (n - 1) * step >= maxy


def terrain_range(bounds, x0, y0, step, n, d) -> list[float]:
    """Lowest and highest height of the grid cells that cover the bounds (the grid must cover them)."""
    minx, miny, maxx, maxy = bounds
    i0, i1 = math.floor((minx - x0) / step), min(n - 1, math.ceil((maxx - x0) / step))
    j0, j1 = math.floor((miny - y0) / step), min(n - 1, math.ceil((maxy - y0) / step))
    cells = [d[j * n + i] for j in range(j0, j1 + 1) for i in range(i0, i1 + 1)]
    return [rnd(min(cells), 3), rnd(max(cells), 3)]


def _licences(terrain_attribution: list[str], synthetic: bool, warnings: list) -> list[dict]:
    out = [{'data': 'OpenStreetMap', 'license': 'ODbL-1.0', 'attribution': OSM_ATTRIBUTION, 'url': 'https://www.openstreetmap.org/copyright'}]
    for a in terrain_attribution:
        known = next((lic for key, lic in TERRAIN_LICENCES if key in a), None)
        if known:
            out.append({**known, 'attribution': a})
        elif synthetic:
            out.append({'data': 'synthetic terrain (test data, not a real place)', 'license': None, 'attribution': a, 'url': None})
            warnings.append(f'synthetic input: terrain "{a}" has no licence; this OutbreakGeo is test data only')
        else:
            raise PolicyError(f'terrain licence not recognised from its attribution "{a}" (known: Lantmäteriet Markhöjdmodell, '
                              'Copernicus DEM GLO-30). Its terms must be known before the data can be used.')
    return out


def _record(inputs: dict, key: str, present: bool) -> dict:
    return {'file': None, 'sha256': None, 'bytes': None, 'encoding': None, **(inputs.get(key) or {}), 'present': present}


def _sha256_text(path: Path) -> str:
    """SHA-256 of a source file with line endings normalised, so a Windows checkout pins what Linux does."""
    return hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def code_pins() -> dict:
    """SHA-256 of every file whose content decides the output: Wasteland's preprocessing and this adapter."""
    here = Path(__file__).resolve().parent
    files = [ROOT / f for f in CODE_FILES] + sorted(here.glob('*.py')) + sorted((here / 'schemas').glob('*.json'))
    return {p.relative_to(ROOT).as_posix(): _sha256_text(p) for p in files}


def runtime() -> dict:
    """Library versions that can change computed geometry (GEOS above all)."""
    return {'python': platform.python_version(), 'shapely': shapely.__version__, 'geos': shapely.geos_version_string,
            'numpy': np.__version__}


def build_geo(city, *, slug=None, osm=None, overrides=None, terrain_src=None, inputs=None, generator=None,
              chunk_size=DEFAULT_CHUNK_M, warnings=None, verified=False, synthetic=False) -> dict:
    """OutbreakGeo v1 from a parsed city.json, plus (optional, for provenance) the parsed osm.json and
    terrain.json, and file records {name: {file, sha256, bytes, encoding}} of the inputs. A parsed
    overrides.json is checked against the refinement policy and only its build settings are kept.

    verified: city.json was proven equal to a clean rebuild (convert_city_dir does this); recorded as is.
    synthetic: test data, not a real place; allows an unlicensed terrain and says so in the output."""
    warnings = list(warnings or [])
    inputs = inputs or {}
    if not isinstance(city, dict):
        raise AdapterError('city.json must be a JSON object')
    if city.get('version') not in CITY_VERSIONS:
        raise AdapterError(f'unsupported city.json version {city.get("version")!r} (this adapter reads {sorted(CITY_VERSIONS)})')
    place = city.get('place')
    if not isinstance(place, dict) or not isinstance(place.get('center'), list) or len(place['center']) != 2 \
            or not all(_finite(v) for v in place['center']):
        raise AdapterError('city.json "place" needs a centre [lat, lon]')
    lat0, lon0 = (float(v) for v in place['center'])
    if not (-90 < lat0 < 90 and -180 <= lon0 <= 180):
        raise AdapterError(f'city.json centre {place["center"]} is not a latitude/longitude')
    half = city.get('half')
    if not (_finite(half) and half > 0):
        raise AdapterError('city.json "half" (half the play-area size in metres) is missing or not positive')
    refinements = wasteland_refinements(overrides)
    if refinements:
        raise PolicyError(REFUSAL.format(', '.join(refinements)))
    settings = dict((overrides or {}).get('defaults') or {})

    model = place.get('projection', 'sphere')
    if model not in Projection.MODELS:
        raise AdapterError(f'city.json place: unknown projection {model!r} (known: {", ".join(Projection.MODELS)})')
    if model != PROJECTION:
        sphere, true = Projection(lat0, lon0), Projection(lat0, lon0, PROJECTION)
        raise PolicyError(f'the city was built on Wasteland\'s spherical plane (place.json has no "projection": "{PROJECTION}"), whose '
                          f'lengths are off by {100 * (sphere.kx / true.kx - 1):+.2f} % east-west and {100 * (sphere.ky / true.ky - 1):+.2f} % '
                          f'north-south at this latitude, so its metres aren\'t metres. Set "projection": "{PROJECTION}" in place.json and '
                          'run fetch (for the terrain), terrain and prepare again.')
    proj = Projection(lat0, lon0, PROJECTION)
    grid = Grid((-half, -half, half, half), chunk_size)
    ad = _Adapter(city, osm, settings, grid, proj, warnings)
    if osm is None:
        ad.warn('osm.json not available: tag-based provenance is "unknown"')

    buildings = sorted((ad.building(fid, b) for fid, b in ad.unique(city.get('buildings') or [], 'building', lambda b: b.get('footprint'))),
                       key=lambda f: f['id'])
    roads = sorted((f for fid, r in ad.unique(city.get('roads') or [], 'road', lambda r: r.get('p')) for f in [ad.road(fid, r)] if f),
                   key=lambda f: f['id'])
    areas = ad.areas()
    collision = ad.collision(buildings)
    topology = ad.topology(roads)
    joined = junctions(roads, grid)
    crossed = crossings(roads, grid, joined)
    unresolved = sum(1 for c in crossed if c['status'] == 'unresolved')
    if unresolved:
        ad.warn(f'{unresolved} places where two roads cross in the plane on one level without sharing a node '
                '(navigation.crossings, status unresolved): not joined')
    terrain = ad.terrain(terrain_src)
    # A height grid sampled on another plane than the city's would put every height in the wrong place (2.6 m
    # off at the grid's edge in Borås). Only a terrain the city uses matters; a flat city ignores terrain.json.
    if terrain is not None and isinstance(terrain_src, dict) and terrain_src.get('projection', 'sphere') != PROJECTION:
        raise PolicyError(f'terrain.json was sampled on another plane ({terrain_src.get("projection", "sphere")!r}) than the city '
                          f'({PROJECTION!r}), so its heights would be misplaced: fetch it again (build --only terrain --refresh)')
    if ad.missing_tags:
        ad.warn(f'{ad.missing_tags} features have no element in osm.json: their provenance is "unknown"')

    index = {c: {'buildings': [], 'roads': [], 'areas': [], 'collision': []} for c in grid.chunks()}
    by_id = {chunk_id(*c): v for c, v in index.items()}
    for b in buildings:
        by_id[b['chunk']]['buildings'].append(b['id'])
    for kind, feats in (('roads', roads), ('areas', areas), ('collision', collision)):
        for f in feats:
            for c in f['chunks']:
                by_id[c][kind].append(f['id'])
    chunks_out = [{'id': chunk_id(i, j), 'i': i, 'j': j, 'bounds_m': grid.cell_in_world(i, j),
                   **{k: sorted(v) for k, v in index[(i, j)].items()}} for i, j in sorted(index)]

    osm_meta = osm if isinstance(osm, dict) else None
    tsrc = terrain_src if isinstance(terrain_src, dict) else None
    # Terrain is credited only when the city uses it (a flat city ignores terrain.json).
    terrain_attr = []
    if terrain:
        text = terrain['attribution'] or (tsrc or {}).get('attribution') or ''
        terrain_attr = [a.strip() for a in str(text).split(' · ') if a.strip()]
        if not terrain_attr:
            if not synthetic:
                raise PolicyError('the terrain has no attribution, so its source and licence are unknown')
            terrain_attr = ['synthetic terrain']
    licences = _licences(terrain_attr, synthetic, warnings)
    attribution = [OSM_ATTRIBUTION] + terrain_attr
    generator = {'name': 'wasteland-builder outbreak adapter', 'version': ADAPTER_VERSION, 'commit': None, 'dirty': None, **(generator or {})}
    s, w = proj.latlon(-half, -half)
    n, e = proj.latlon(half, half)
    manifest = {
        'schema': MANIFEST_SCHEMA, 'schema_version': 1,
        'place': {'center': {'lat': lat0, 'lon': lon0}, 'bbox_wgs84': _bbox_dict(place.get('bbox')), 'size_m': place.get('size_m'),
                  'query': place.get('query'), 'nominatim_osm': place.get('osm') if isinstance(place.get('osm'), dict) else None},
        'pipeline': {**generator, 'wasteland_city_version': city['version'], 'code_sha256': code_pins(), 'runtime': runtime()},
        'inputs': {
            'place': _record(inputs, 'place', 'place' in inputs),
            'city': {**_record(inputs, 'city', True), 'clean_rebuild': bool(verified)},
            'osm': {**_record(inputs, 'osm', osm_meta is not None),
                    'generator': osm_meta.get('generator') if osm_meta else None,
                    'timestamp_osm_base': (osm_meta.get('osm3s') or {}).get('timestamp_osm_base') if osm_meta else None,
                    'elements': len(osm_meta.get('elements') or []) if osm_meta else None},
            'terrain': {**_record(inputs, 'terrain', tsrc is not None),
                        'source': tsrc.get('source') if tsrc else None, 'kind': tsrc.get('kind') if tsrc else None,
                        'tiles': list(tsrc.get('tiles') or []) if tsrc else [], 'attribution': tsrc.get('attribution') if tsrc else None,
                        'projection': tsrc.get('projection') if tsrc else None,
                        'coverage': tsrc.get('coverage') if tsrc and isinstance(tsrc.get('coverage'), dict) else None},
            'wasteland_overrides': {**_record(inputs, 'overrides', isinstance(overrides, dict)), 'build_settings': settings},
        },
        'pinning': ('Inputs and code are pinned by SHA-256. Rebuilding the same OutbreakGeo needs these exact files: archive '
                    'osm.json and terrain.json with the build; downloading OpenStreetMap again later gives different data.'),
    }
    return {
        'schema': SCHEMA, 'schema_version': SCHEMA_VERSION,
        'metadata': {
            'generator': generator,
            'source_city': {'slug': slug, 'name': str(place.get('name') or ''), 'display_name': str(place.get('display_name') or ''),
                            'country_code': str(place.get('country_code') or ''), 'wasteland_city_version': city['version']},
            'synthetic': bool(synthetic),
            'center': {'lat': lat0, 'lon': lon0},
            'bounds': {'min_m': [-float(half), -float(half)], 'max_m': [float(half), float(half)], 'size_m': 2 * float(half),
                       'wgs84': {'south': rnd(s, 7), 'west': rnd(w, 7), 'north': rnd(n, 7), 'east': rnd(e, 7)}},
            'coordinates': {
                'unit': 'm', 'frame': 'local tangent plane', 'origin': 'the place centre', 'axes': {'x': 'east', 'y': 'north', 'z': 'up'},
                'projection': PROJECTION_NAME,
                'to_local': (f'x = (lon - lon0) * {proj.kx!r}; y = (lat - lat0) * {proj.ky!r} (metres per degree from the WGS84 '
                             'prime-vertical and meridional radii at lat0)'),
                'rings': 'outer rings counter-clockwise, holes clockwise, no repeated closing point',
                'heights': 'metres over the terrain datum (terrain.vertical_datum); building heights over the building base',
            },
            'chunking': {'scheme': 'uniform grid anchored at the place centre', 'size_m': grid.size, 'origin_m': [0.0, 0.0],
                         'cell': 'chunk (i, j) = [i*size_m, (i+1)*size_m) x [j*size_m, (j+1)*size_m), clipped to the bounds',
                         'id_format': 'c{i}_{j}', 'count': len(chunks_out)},
            'warnings': sorted(warnings),             # sorted: independent of the input order
        },
        'provenance': {
            'attribution': attribution, 'licenses': licences,
            'origins': {k: {'description': d, 'confidence': c} for k, (d, c) in ORIGINS.items()},
            'area_classes': {k: {'origin': o, 'wasteland_layers': layers, 'description': d} for k, (o, layers, d) in AREA_CLASSES.items()},
            'policy': {
                'street_view': 'Not a data source. Google Street View is never used to reconstruct buildings or facades in OutbreakGeo.',
                'wasteland_refinements': ("Not used: a city with Wasteland refinement data (overrides.json entries, custom/*.py, "
                                          "refinements.md) is refused; only the build settings terrain and terrain_scale are read."),
                'city_json': ('Verified: identical to a rebuild by prepare_city from the pinned place.json, osm.json and terrain.json '
                              'with the pinned code and no refinements.' if verified else
                              'Not verified: made from a city.json without its folder (library or test use). Not for publication.'),
            },
            'sources': manifest,
        },
        'terrain': terrain,
        'roads': roads,
        'buildings': buildings,
        'areas': areas,
        'navigation': {'collision': collision, 'drivable_roads': [r['id'] for r in roads if r['drivable']],
                       'junctions': joined, 'crossings': crossed, 'topology': topology},
        'chunks': chunks_out,
    }


# ------------------------------------------------------------------------------------------ the contract beyond the schema
def _inside_length(pts, grid: Grid) -> float:
    """Arc length of a polyline inside the half-open bounds, segment by segment (a road may run back along itself)."""
    total = 0.0
    for a, b in zip(pts, pts[1:]):
        if a != b:
            seg = LineString([a, b])
            total += seg.intersection(grid.world).length - seg.intersection(grid.max_edges).length
    return total


def check_geo(geo: dict) -> list[str]:
    """What the JSON Schema can't express, checked on a (serialised) OutbreakGeo from first principles, not by
    rerunning the adapter: ids, chunk references and the chunk index, the owner rule, spans (each inside its
    chunk, half-open, together covering the road inside the bounds), polygon validity, collision covering
    every ground-standing building, junctions and topology from the roads' nodes, the terrain raster.
    [] when the document keeps its contract."""
    errs = []
    meta = geo['metadata']
    (x0, y0), (x1, y1) = meta['bounds']['min_m'], meta['bounds']['max_m']
    grid = Grid((x0, y0, x1, y1), meta['chunking']['size_m'])
    world = grid.world.buffer(WORLD_TOLERANCE_M, join_style=2)
    chunk_list = geo['chunks']
    # The declared frame and grid: a consumer places everything by these.
    if meta['chunking']['origin_m'] != [0.0, 0.0] or meta['chunking']['count'] != len(chunk_list):
        errs.append('metadata.chunking: the grid is anchored at [0, 0] and count is the number of chunks')
    if not (x0 == -x1 and y0 == -y1 and x1 - x0 == y1 - y0 == meta['bounds']['size_m']):
        errs.append('metadata.bounds: not a square of size_m centred on the place')
    if meta['coordinates']['projection'] != PROJECTION_NAME:
        errs.append(f'metadata.coordinates.projection must be {PROJECTION_NAME} (lengths true to the WGS84 ellipsoid)')
    proj = Projection(meta['center']['lat'], meta['center']['lon'], PROJECTION)
    (s, w), (n, e) = proj.latlon(x0, y0), proj.latlon(x1, y1)
    wgs = meta['bounds']['wgs84']
    if max(abs(s - wgs['south']), abs(w - wgs['west']), abs(n - wgs['north']), abs(e - wgs['east'])) > 1e-6:
        errs.append('metadata.bounds.wgs84 disagrees with the centre and the bounds')
    by_id = {c['id']: c for c in chunk_list}
    if [c['id'] for c in chunk_list] != [chunk_id(i, j) for i, j in grid.chunks()]:
        errs.append('chunks: not exactly the grid cells that overlap the bounds, in order')
    for c in chunk_list:
        if (c['id'] != chunk_id(c['i'], c['j'])) or c['bounds_m'] != grid.cell_in_world(c['i'], c['j']):
            errs.append(f'chunk {c["id"]}: i, j or bounds_m disagree with the grid')
        if not (c['bounds_m'][2] > c['bounds_m'][0] and c['bounds_m'][3] > c['bounds_m'][1]):
            errs.append(f'chunk {c["id"]}: no area inside the bounds')

    def holder(x, y):
        """The chunk whose published bounds_m hold the point, half-open; judged from the document alone."""
        found = [c['id'] for c in chunk_list if c['bounds_m'][0] <= x < c['bounds_m'][2] and c['bounds_m'][1] <= y < c['bounds_m'][3]]
        return found[0] if len(found) == 1 else None

    def polygon_ok(what, poly, chunk_ids):
        shape = Polygon(poly['outer'], poly['holes'])
        if not shape.is_valid or shape.area <= AREA_EPS:
            errs.append(f'{what}: not a valid polygon with area')
            return None
        if _signed_area(poly['outer']) <= 0 or any(_signed_area(h) >= 0 for h in poly['holes']):
            errs.append(f'{what}: ring orientation')
        if not world.contains(shape):
            errs.append(f'{what}: outside the bounds')
        if chunk_ids != [chunk_id(*m) for m in grid.polygon_chunks(shape)]:
            errs.append(f'{what}: chunks are not exactly the chunks it covers')
        return shape

    index = defaultdict(lambda: defaultdict(list))
    for kind in ('buildings', 'roads', 'areas'):
        ids = [f['id'] for f in geo[kind]]
        if len(set(ids)) != len(ids) or ids != sorted(ids) and kind != 'areas':
            errs.append(f'{kind}: ids not unique and sorted')
    building_ids = {b['id'] for b in geo['buildings']}
    footprints = {}
    for b in geo['buildings']:
        shape = polygon_ok(f'building {b["id"]}', b['footprint'], b['chunks'])
        if shape is not None:
            footprints[b['id']] = shape
            # The owner rule from the published chunk bounds: the chunk holding the centroid if the footprint
            # covers it, else the covered chunk with the largest overlap (the first such in chunk order).
            centroid = holder(shape.centroid.x, shape.centroid.y)
            if centroid in b['chunks']:
                expected = centroid
            else:
                overlap = {c: shape.intersection(box(*by_id[c]['bounds_m'])).area for c in b['chunks'] if c in by_id}
                expected = max(sorted(overlap, key=parse_chunk_id), key=lambda c: overlap[c]) if overlap else None
            if b['chunk'] != expected:
                errs.append(f"building {b['id']}: owner chunk {b['chunk']} breaks the owner rule (centroid's chunk, else largest overlap)")
        if b['min_height_m'] is not None and b['min_height_m'] >= b['wall_height_m']:
            errs.append(f'building {b["id"]}: min_height_m is not below wall_height_m')
        index[b['chunk']]['buildings'].append(b['id'])
    for r in geo['roads']:
        pts, spans = r['centerline'], r['spans']
        length = sum(math.dist(a, b) for a, b in zip(pts, pts[1:]))
        if abs(r['length_m'] - length) > 1e-9:
            errs.append(f'road {r["id"]}: length_m is not the length of its centreline')
        if r['nodes'] is not None and len(r['nodes']) != len(pts):
            errs.append(f"road {r['id']}: nodes don't match the centreline vertices")
        end = -math.inf
        for s in spans:
            s0, s1 = s['s0_m'], s['s1_m']
            if not 0 <= s0 < s1 <= length + 1e-9 or s0 < end or s['chunk'] not in by_id:
                errs.append(f'road {r["id"]}: span {s} is empty, out of order, past the end or in no chunk')
                continue
            end = s1
            i, j = parse_chunk_id(s['chunk'])
            cx0, cy0, cx1, cy1 = grid.cell_in_world(i, j)
            tol = 1e-9
            piece = cut_line(pts, s0, s1)
            if any(not (cx0 - tol <= x <= cx1 + tol and cy0 - tol <= y <= cy1 + tol) for x, y in piece):
                errs.append(f'road {r["id"]}: span {s} leaves its chunk')
            # Half-open ownership for every straight stretch of the span: a stretch lying on a chunk border
            # belongs to the higher chunk, so its midpoint must fall in this one. Stretches under `tol` are
            # float noise of the interpolation (a span starting at a vertex on a border), not road.
            stretches = [((ax + bx) / 2, (ay + by) / 2) for (ax, ay), (bx, by) in zip(piece, piece[1:])
                         if math.dist((ax, ay), (bx, by)) > tol] or cut_line(pts, (s0 + s1) / 2, (s0 + s1) / 2)
            if any(holder(mx, my) != s['chunk'] for mx, my in stretches):
                errs.append(f'road {r["id"]}: part of span {s} belongs to another chunk (half-open cells)')
        if abs(sum(s['s1_m'] - s['s0_m'] for s in spans) - _inside_length(pts, grid)) > 1e-6:
            errs.append(f"road {r['id']}: spans don't cover its part inside the bounds")
        if r['chunks'] != sorted({s['chunk'] for s in spans}, key=parse_chunk_id):
            errs.append(f'road {r["id"]}: chunks differ from its spans')
        for c in r['chunks']:
            index[c]['roads'].append(r['id'])
    ground = []
    for a in geo['areas']:
        shape = polygon_ok(f'area {a["id"]}', a['polygon'], a['chunks'])
        if shape is not None:
            ground.append((a, shape))
        for c in a['chunks']:
            index[c]['areas'].append(a['id'])
    # One class per piece of land; water lies under nothing but road classes (bridge decks).
    tree = STRtree([s for _, s in ground])
    for k, (a, shape) in enumerate(ground):
        for m in tree.query(shape):
            b, other = ground[m]
            if m <= k:
                continue
            if (a['class'] == 'water') != (b['class'] == 'water') and ({a['class'], b['class']} - {'water'}) <= ROAD_CLASSES:
                continue
            overlap = shape.intersection(other).area
            if overlap > 1e-4:
                errs.append(f'areas {a["id"]} ({a["class"]}) and {b["id"]} ({b["class"]}) overlap by {overlap:.4f} m²')
    cover = defaultdict(list)
    col_ids = [c['id'] for c in geo['navigation']['collision']]
    if len(set(col_ids)) != len(col_ids) or col_ids != sorted(col_ids):
        errs.append('collision: ids not unique and sorted')
    for c in geo['navigation']['collision']:
        shape = polygon_ok(f'collision {c["id"]}', c['polygon'], c['chunks'])
        if c['kind'] == 'city_wall' and (c['buildings'] or c['osm'] is not None):
            errs.append(f'collision {c["id"]}: a city wall refers to buildings')
        for m in c['buildings']:
            if m not in building_ids:
                errs.append(f'collision {c["id"]}: unknown building {m}')
            elif shape is not None:
                cover[m].append(shape)
        for ch in c['chunks']:
            index[ch]['collision'].append(c['id'])
    passages = [LineString(r['centerline']).buffer(r['width_m'] / 2 + 0.301, cap_style=2) for r in geo['roads']
                if (r['tags'].get('tunnel') == 'building_passage' or r['tags'].get('covered') == 'yes') and _at_ground(r['tags'])]
    allowed = unary_union(passages) if passages else None
    for b in geo['buildings']:
        if b['id'] not in footprints or (b['min_height_m'] is not None and b['min_height_m'] > FLOATING_M):
            continue
        rest = footprints[b['id']].difference(unary_union(cover[b['id']])) if cover[b['id']] else footprints[b['id']]
        if allowed is not None:
            rest = rest.difference(allowed)
        if rest.area > 1e-6:
            errs.append(f'building {b["id"]}: {rest.area:.4f} m² of its footprint has no collision that lists it')
    for c in chunk_list:
        for kind in ('buildings', 'roads', 'areas', 'collision'):
            if c[kind] != sorted(index[c['id']][kind]):
                errs.append(f'chunk {c["id"]}: {kind} index disagrees with the features')
    stray = set(index) - set(by_id)
    if stray:
        errs.append(f"features refer to chunks that don't exist: {sorted(stray)}")
    nav = geo['navigation']
    if nav['drivable_roads'] != [r['id'] for r in geo['roads'] if r['drivable']]:
        errs.append('navigation.drivable_roads disagrees with the roads')
    # One OSM node, one place: an id at two positions was attached to the wrong vertex somewhere.
    node_at = {}
    for r in geo['roads']:
        for p, nd in zip(r['centerline'], r['nodes'] or []):
            if node_at.setdefault(nd, p) != p:
                errs.append(f'road {r["id"]}: node {nd} at {p}, elsewhere at {node_at[nd]}')
    joined = junctions(geo['roads'], grid)
    if nav['junctions'] != joined:
        errs.append('navigation.junctions are not exactly where the roads share a node (or, without node ids, a position on one level)')
    if nav['crossings'] != crossings(geo['roads'], grid, joined):
        errs.append('navigation.crossings are not exactly where roads meet in the plane without being joined')
    with_nodes = sum(1 for r in geo['roads'] if r['nodes'] is not None)
    expected = 'none' if not geo['roads'] else 'osm_nodes' if with_nodes == len(geo['roads']) else 'positions' if not with_nodes else 'mixed'
    if nav['topology'] != expected:
        errs.append(f'navigation.topology is {nav["topology"]!r}, the roads say {expected!r}')
    t = geo['terrain']
    if t is not None:
        g = t['grid']
        n = int(g['n'])                          # JSON Schema's integer allows 3.0
        (tx, ty), step = g['origin_m'], g['step_m']
        if len(t['heights_m']) != n * n:
            errs.append(f'terrain: {len(t["heights_m"])} heights for an {n}x{n} grid')
        elif not raster_covers(grid.bounds, tx, ty, step, n):
            errs.append('terrain: the height grid doesn\'t cover the whole play area')
        elif t['range_in_bounds_m'] != terrain_range(grid.bounds, tx, ty, step, n, t['heights_m']):
            errs.append('terrain: range_in_bounds_m disagrees with the heights')
    return errs


# ------------------------------------------------------------------------------------------ files
INPUT_FILES = {'place': 'place.json', 'city': 'city.json', 'osm': 'osm.json', 'terrain': 'terrain.json', 'overrides': 'overrides.json'}


def read_json(path: Path, warnings: list):
    """(data, file record). UTF-8, or Windows-1252, which Wasteland writes on Western Windows: an explicit
    rule, the same under any locale or PYTHONUTF8 setting, and recorded."""
    raw = path.read_bytes()
    for encoding in ('utf-8', LEGACY_ENCODING):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise AdapterError(f'{path.name} is neither UTF-8 nor {LEGACY_ENCODING}')
    if encoding != 'utf-8':
        warnings.append(f'{path.name} is not UTF-8; read as {encoding} (Wasteland on Windows writes the ANSI code page)')
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise AdapterError(f'{path.name} is not valid JSON: {exc}') from None
    return data, {'file': path.name, 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw), 'encoding': encoding}


def git_generator(root: Path = ROOT) -> dict:
    """The commit this adapter runs from and whether the working tree differs from it (None: not a git checkout)."""
    try:
        head = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True, text=True, timeout=30)
        status = subprocess.run(['git', '-C', str(root), 'status', '--porcelain'], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return {'commit': None, 'dirty': None}
    if head.returncode or status.returncode:
        return {'commit': None, 'dirty': None}
    return {'commit': head.stdout.strip(), 'dirty': bool(status.stdout.strip())}


def convert_city_dir(folder: Path, *, chunk_size=DEFAULT_CHUNK_M, generator=None, synthetic=False, timings=None) -> dict:
    """OutbreakGeo from a Wasteland city folder, after the refinement policy and the clean-rebuild check.
    `timings` (a dict), when given, gets the seconds the clean rebuild and the conversion took."""
    folder = Path(folder)
    timings = timings if timings is not None else {}
    if not (folder / 'city.json').exists():
        raise AdapterError(f'{folder / "city.json"} is missing. Run: python3 wasteland.py build {folder.name} --only fetch,terrain,prepare')
    traces = refinement_traces(folder)
    if traces:
        raise PolicyError(REFUSAL.format(', '.join(traces)))
    warnings, data, records = [], {}, {}
    for key, name in INPUT_FILES.items():
        if (folder / name).exists():
            data[key], records[key] = read_json(folder / name, warnings)
    refinements = wasteland_refinements(data.get('overrides'))
    if refinements:
        raise PolicyError(REFUSAL.format(', '.join(refinements)))
    settings = dict((data.get('overrides') or {}).get('defaults') or {})
    started = time.perf_counter()
    rebuilt = clean_rebuild(folder, settings)
    timings['clean_rebuild_s'] = time.perf_counter() - started
    if rebuilt != data['city']:
        differ = sorted(k for k in set(rebuilt) | set(data['city']) if rebuilt.get(k) != data['city'].get(k)) if isinstance(data['city'], dict) else ['all']
        raise PolicyError('city.json is not what prepare_city makes from this folder\'s place.json, osm.json and terrain.json with '
                          f'the current code and no refinements (it differs in: {", ".join(differ)}). It was built from other '
                          'inputs, other code or with refinements, so its provenance can\'t be established. Rebuild it: '
                          f'python3 wasteland.py build {folder.name} --only prepare')
    started = time.perf_counter()
    geo = build_geo(data['city'], slug=folder.name, osm=data.get('osm'), overrides=data.get('overrides'),
                    terrain_src=data.get('terrain'), inputs=records, generator=generator, chunk_size=chunk_size,
                    warnings=warnings, verified=True, synthetic=synthetic)
    timings['build_geo_s'] = time.perf_counter() - started
    return geo


def dumps(geo: dict) -> str:
    """The canonical text of an OutbreakGeo: UTF-8, sorted keys, compact; byte-identical for identical input."""
    return json.dumps(geo, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n'


def validate(geo: dict) -> list[str]:
    """Schema and contract errors of an OutbreakGeo, checked on its serialised form."""
    doc = json.loads(dumps(geo))
    return schema.validate(doc, 'outbreak_geo.v1') or check_geo(doc)


def write_geo(path: Path, geo: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_bytes(dumps(geo).encode('utf-8'))
    tmp.replace(path)


def _say(msg: str):
    """Print what the console can show (a Windows pipe is cp1252; place names may not fit)."""
    enc = getattr(sys.stdout, 'encoding', None) or 'utf-8'
    print(msg.encode(enc, 'replace').decode(enc), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python pipeline/outbreak', description='city.json -> OutbreakGeo v1 (docs/outbreak-real-city.md)')
    ap.add_argument('city', help='city slug (cities/<slug>) or a folder with city.json')
    ap.add_argument('--out', help='output file (default: cities/<slug>/outbreak/OutbreakGeo.json)')
    ap.add_argument('--chunk-size', type=float, default=DEFAULT_CHUNK_M, help=f'chunk size in metres (default {DEFAULT_CHUNK_M:g})')
    args = ap.parse_args(argv)
    folder = Path(args.city) if (Path(args.city) / 'city.json').exists() else CITIES / args.city
    try:
        geo = convert_city_dir(folder, chunk_size=args.chunk_size, generator=git_generator())
        problems = validate(geo)
    except (AdapterError, ValueError) as exc:
        raise SystemExit(f'error: {exc}') from None
    if problems:
        raise SystemExit('error: the OutbreakGeo breaks its contract: ' + '; '.join(problems[:10]))
    out = Path(args.out) if args.out else folder / 'outbreak' / 'OutbreakGeo.json'
    write_geo(out, geo)
    _say(f'OutbreakGeo v{SCHEMA_VERSION}: {len(geo["buildings"])} buildings, {len(geo["roads"])} roads, {len(geo["areas"])} areas, '
         f'{geo["metadata"]["chunking"]["count"]} chunks of {geo["metadata"]["chunking"]["size_m"]:g} m -> {out}')
    for w in geo['metadata']['warnings']:
        _say(f'  ! {w}')
    return out
