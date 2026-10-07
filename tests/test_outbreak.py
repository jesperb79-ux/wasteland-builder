"""Offline tests of the Outbreak real-city adapter (P00): no network, Blender, Node or Roblox Studio.

    .venv/bin/python -m unittest discover tests        (runs with the Wasteland pipeline tests)

The synthetic fixture (outbreak_fixture.py) goes through Wasteland's own prepare_city.py, and its
city.json through pipeline/outbreak into OutbreakGeo v1, which is checked for metre coordinates, OSM
ids, chunks, provenance, attribution, determinism and the absence of Wasteland's render data. Every
output is checked against the schema and the contract beyond it (adapter.check_geo) in its serialised
form. Hand-made inputs cover the edge cases and the counter-examples from review; the unit and
chunk-grid rules are tested on their own. Everything runs in temporary folders.
"""
import contextlib
import copy
import hashlib
import io
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'pipeline'))
sys.path.insert(0, str(ROOT / 'tests'))

import common  # noqa: E402
import outbreak_fixture as fx  # noqa: E402
import prepare_city  # noqa: E402
from outbreak import adapter, chunks, schema, units  # noqa: E402
from shapely.geometry import Polygon  # noqa: E402
from shapely.ops import unary_union  # noqa: E402

GEN = {'commit': '0' * 40, 'dirty': False}       # a fixed generator, so outputs compare byte for byte
EXAMPLES = ROOT / 'pipeline/outbreak/examples'
MESH_KEYS = {'v', 'uv', 't', 'mesh', 'parts', 'surfaces', 'outskirts', 'water_bodies', 'curbs', 'shore', 'bridge_edges',
             'barriers', 'props', 'trees', 'rails', 'trams', 'walls', 'footprints', 'land', 'landmarks', 'places', 'streets', 'stats',
             'style', 'roof_style'}


def keys_in(doc):
    """Every object key anywhere in a JSON document."""
    if isinstance(doc, dict):
        for k, v in doc.items():
            yield k
            yield from keys_in(v)
    elif isinstance(doc, list):
        for v in doc:
            yield from keys_in(v)


def contract(geo):
    """Schema and contract errors of an OutbreakGeo, on its serialised form."""
    return adapter.validate(geo)


def minimal_city(**extra):
    return {'version': 1, 'place': {'name': 'Tiny', 'center': [60.0, 15.0], 'projection': 'wgs84'}, 'half': 100, **extra}


def building(fid, x0, y0, x1, y1, **extra):
    return {'id': fid, 'kind': 'yes', 'footprint': {'outer': [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], 'holes': []},
            'h': 6.0, 'levels': 2, 'style': 'plaster', 'colour': 'white', 'roof_style': 'flat', 'roof_colour': 'tar', **extra}


def road(fid, pts, kind='residential', w=6.0):
    return {'id': fid, 'kind': kind, 'w': w, 'p': [list(p) for p in pts]}


def osm_of(**tags):
    """osm.json with these elements' tags: osm_of(w1={...}, r2={...})."""
    return {'elements': [{'type': {'n': 'node', 'w': 'way', 'r': 'relation'}[k[0]], 'id': int(k[1:]), 'tags': t} for k, t in tags.items()]}


def prepared(folder: Path, place=None, osm=None, terrain=None, overrides=None) -> Path:
    """A Wasteland city folder made by prepare_city.py from the fixture's (or the given) inputs."""
    fx.write(folder)
    for name, data in (('osm.json', osm), ('terrain.json', terrain), ('overrides.json', overrides), ('place.json', place)):
        if data is False:
            (folder / name).unlink()
        elif data is not None:
            (folder / name).write_text(json.dumps(data), encoding='ascii')
    old = common.CITIES                 # Wasteland finds a city by slug under common.CITIES
    common.CITIES = folder.parent
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            prepare_city.main([folder.name])
    finally:
        common.CITIES = old
    return folder


# ------------------------------------------------------------------------------------------ units
class UnitsTest(unittest.TestCase):
    def test_metres_to_studs(self):
        self.assertEqual(units.METRES_PER_STUD, 0.28)
        self.assertAlmostEqual(units.STUDS_PER_METRE, 3.571428571, places=9)
        self.assertEqual(units.metres_to_studs(1.0), 1 / 0.28)
        self.assertEqual(units.metres_to_studs(0.28), 1.0)
        self.assertAlmostEqual(units.metres_to_studs(7.0), 25.0, places=12)
        self.assertAlmostEqual(units.metres_to_studs(60.0), 214.285714286, places=8)     # one chunk
        self.assertAlmostEqual(units.metres_to_studs(300.0), 1071.428571429, places=8)   # the PoC square
        self.assertAlmostEqual(units.studs_to_metres(25.0), 7.0, places=12)
        for m in (0.0, 0.01, 3.2, 150.0, -42.5, 1234.567):
            self.assertAlmostEqual(units.studs_to_metres(units.metres_to_studs(m)), m, places=9)

    def test_roblox_axes(self):
        X, Y, Z = units.local_to_roblox(10.0, 20.0, 3.0)
        self.assertEqual((X, Y, Z), (10 / 0.28, 3 / 0.28, -20 / 0.28))       # east → +X, up → +Y, north → −Z
        x, y, z = units.roblox_to_local(X, Y, Z)
        self.assertAlmostEqual(x, 10.0, places=9)
        self.assertAlmostEqual(y, 20.0, places=9)
        self.assertAlmostEqual(z, 3.0, places=9)
        self.assertEqual(units.local_to_roblox(0.0, 0.0), (0.0, 0.0, 0.0))
        self.assertEqual(str(units.local_to_roblox(0.0, 0.0)[2]), '0.0')      # never −0.0

    def test_rejects_what_is_not_a_length(self):
        for bad in (float('nan'), float('inf'), '1', None, True):
            with self.assertRaises(ValueError):
                units.metres_to_studs(bad)
            with self.assertRaises(ValueError):
                units.studs_to_metres(bad)


# ------------------------------------------------------------------------------------------ chunk grid
class ChunkGridTest(unittest.TestCase):
    def setUp(self):
        self.grid = chunks.Grid((-150, -150, 150, 150), 60)

    def test_half_open_cells(self):
        g = self.grid
        self.assertEqual(g.index(0, 0), (0, 0))
        self.assertEqual(g.index(-0.001, 0), (-1, 0))
        self.assertEqual(g.index(60, 0), (1, 0))
        self.assertEqual(g.index(59.999, -60), (0, -1))
        self.assertEqual(list(g.i_range), [-3, -2, -1, 0, 1, 2])
        self.assertEqual(len(g.chunks()), 36)
        self.assertEqual(g.chunk_bounds(-3, 2), [-180.0, 120.0, -120.0, 180.0])
        self.assertEqual(g.cell_in_world(-3, 2), [-150.0, 120.0, -120.0, 150.0], 'edge chunks end at the bounds')
        self.assertEqual(g.cell_in_world(0, 0), [0.0, 0.0, 60.0, 60.0])
        self.assertTrue(g.contains(-150, -150))
        self.assertFalse(g.contains(150, 0), 'the max edge is outside')
        # Anchored at the centre: a bigger play area keeps every chunk id and border.
        self.assertEqual(chunks.Grid((-500, -500, 500, 500), 60).chunk_bounds(-3, 2), g.chunk_bounds(-3, 2))

    def test_decimal_chunk_sizes_follow_the_published_borders(self):
        """floor(x / 30.1) is one off at x = -3·30.1: cells are judged by their borders k·S, as published."""
        for S in (30.1, 30.4, 7.3, 0.1):
            g = chunks.Grid((-3 * S, -3 * S, 3 * S, 3 * S), S)
            self.assertEqual(len(g.chunks()), 36, S)
            self.assertTrue(all(b[2] > b[0] and b[3] > b[1] for b in (g.cell_in_world(i, j) for i, j in g.chunks())), 'no empty chunks')
            for k in range(-3, 3):
                x0, _, x1, _ = g.chunk_bounds(k, 0)
                self.assertEqual(g.index(x0, 0)[0], k, (S, k))                                  # a border belongs to the higher cell
                self.assertEqual(g.index(math.nextafter(x1, -math.inf), 0)[0], k, (S, k))
            self.assertEqual([(i, j) for i, j, _, _ in g.line_spans([(-3 * S, 0.1 * S), (-3 * S, 0.2 * S)])], [(-3, 0)], S)
            self.assertEqual([(i, j) for i, j, _, _ in g.line_spans([(S, -0.5 * S), (S, 0.5 * S)])], [(1, -1), (1, 0)], S)

    def test_ids(self):
        self.assertEqual(chunks.chunk_id(-3, 2), 'c-3_2')
        self.assertEqual(chunks.parse_chunk_id('c-3_2'), (-3, 2))
        for bad in ('c3', 'x1_2', 'c1_2_3', 'c1.5_2'):
            with self.assertRaises(ValueError):
                chunks.parse_chunk_id(bad)

    def test_bad_grid(self):
        for size in (0, -60, float('nan'), True):
            with self.assertRaises(ValueError):
                chunks.Grid((-150, -150, 150, 150), size)
        with self.assertRaises(ValueError):
            chunks.Grid((10, 0, 10, 5), 60)

    def test_polygon_across_a_border(self):
        g = self.grid
        sq = Polygon([(-10, 5), (10, 5), (10, 15), (-10, 15)])
        self.assertEqual(g.polygon_chunks(sq), [(-1, 0), (0, 0)])
        self.assertEqual(g.owner(sq, g.polygon_chunks(sq)), (0, 0))         # centroid (0, 10): x = 0 belongs to i = 0
        touching = Polygon([(0, 0), (60, 0), (60, 10), (0, 10)])            # an edge on x = 60 doesn't make it a member there
        self.assertEqual(g.polygon_chunks(touching), [(0, 0)])
        outside = Polygon([(200, 200), (210, 200), (210, 210)])
        self.assertEqual(g.polygon_chunks(outside), [])

    def test_owner_when_the_footprint_misses_its_centroid_chunk(self):
        g = self.grid
        # A ring round chunk (0, 0): its centroid (8.05, 8.05) lies in a chunk it doesn't cover.
        ring = Polygon([(-70, -70), (130, -70), (130, 130), (-70, 130)], [[(-5, -5), (-5, 125), (125, 125), (125, -5)]])
        members = g.polygon_chunks(ring)
        self.assertEqual(g.index(ring.centroid.x, ring.centroid.y), (0, 0))
        self.assertNotIn((0, 0), members)
        self.assertEqual(g.owner(ring, members), (-1, -1), 'the chunk it covers most')
        # A courtyard block whose centroid is in the courtyard keeps the centroid's chunk.
        court = Polygon([(-140, 60), (-95, 60), (-95, 100), (-140, 100)], [[(-128, 72), (-128, 88), (-107, 88), (-107, 72)]])
        self.assertFalse(court.contains(court.centroid))
        self.assertEqual(g.owner(court, g.polygon_chunks(court)), (-2, 1))

    def test_line_spans(self):
        g = self.grid
        spans = g.line_spans([(-170, 8), (32, 8), (170, 8)])
        self.assertEqual([(i, j) for i, j, _, _ in spans], [(i, 0) for i in range(-3, 3)])
        self.assertAlmostEqual(spans[0][2], 20.0)           # the bounds start 20 m along the line
        self.assertAlmostEqual(spans[-1][3], 320.0)
        for a, b in zip(spans, spans[1:]):
            self.assertEqual(a[3], b[2], 'neighbouring spans share their end point')
        self.assertAlmostEqual(sum(s1 - s0 for _, _, s0, s1 in spans), 300.0)
        # The cuts fall exactly on the chunk borders.
        line = [(-170, 8), (32, 8), (170, 8)]
        for _, _, s0, _ in spans[1:]:
            x, y = chunks.cut_line(line, s0, s0 + 1)[0]
            self.assertAlmostEqual(x / 60, round(x / 60), places=9)

    def test_line_on_a_border(self):
        g = self.grid
        self.assertEqual({(i, j) for i, j, _, _ in g.line_spans([(60, -10), (60, 10)])}, {(1, -1), (1, 0)})
        self.assertEqual(g.line_spans([(150, -10), (150, 10)]), [], 'the max edge of the bounds is outside')
        self.assertEqual(g.line_spans([(-150, 0), (-150, 10)])[0][:2], (-3, 0), 'the min edge is inside')
        self.assertEqual(g.line_spans([(5, 5), (5, 5)]), [])

    def test_cut_line_and_clip(self):
        line = [(-100, -100), (100, 100)]
        spans = self.grid.line_spans(line)
        pieces = [chunks.cut_line(line, s0, s1) for _, _, s0, s1 in spans]
        for a, b in zip(pieces, pieces[1:]):
            self.assertEqual(a[-1], b[0], 'pieces join without a gap')
        self.assertEqual(pieces[0][0], (-100, -100))
        self.assertEqual(pieces[-1][-1], (100, 100))
        poly = {'outer': [[-130, -20], [70, -20], [70, 90], [-130, 90]], 'holes': [[[-10, 10], [-10, 30], [10, 30], [10, 10]]]}
        total = sum(sum(Polygon(p['outer'], p['holes']).area for p in chunks.clip_polygon(poly, self.grid.cell_in_world(i, j)))
                    for i, j in self.grid.polygon_chunks(chunks.to_shape(poly)))
        self.assertAlmostEqual(total, 200 * 110 - 400, places=6)
        # Clipped to an edge chunk's part of the bounds, nothing comes back from outside them.
        wide = {'outer': [[100, 100], [200, 100], [200, 200], [100, 200]], 'holes': []}
        self.assertAlmostEqual(sum(Polygon(p['outer']).area for p in chunks.clip_polygon(wide, self.grid.cell_in_world(2, 2))), 30 * 30)
        # The pieces keep the contract's orientation: outer counter-clockwise, holes clockwise, a hole kept whole
        # or opened by the chunk border.
        holed = {'outer': [[0, 0], [20, 0], [20, 20], [0, 20]], 'holes': [[[5, 5], [5, 15], [15, 15], [15, 5]]]}
        for bounds, holes in (([0, 0, 20, 20], 1), ([0, 0, 10, 20], 0), ([-5, -5, 25, 25], 1)):
            pieces = chunks.clip_polygon(holed, bounds)
            self.assertEqual(sum(len(p['holes']) for p in pieces), holes, bounds)
            for p in pieces:
                self.assertGreater(adapter._signed_area(p['outer']), 0, bounds)
                self.assertTrue(all(adapter._signed_area(h) < 0 for h in p['holes']), bounds)


# ------------------------------------------------------------------------------------------ the fixture city
class FixtureCityTest(unittest.TestCase):
    """Synthby through prepare_city.py and the adapter (synthetic: its terrain has no licence)."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.folder = prepared(Path(cls.tmp) / fx.SLUG)
        cls.city = adapter.read_json(cls.folder / 'city.json', [])[0]
        cls.osm = adapter.read_json(cls.folder / 'osm.json', [])[0]
        cls.geo = adapter.convert_city_dir(cls.folder, generator=GEN, synthetic=True)
        cls.b = {b['id']: b for b in cls.geo['buildings']}
        cls.r = {r['id']: r for r in cls.geo['roads']}
        cls.city_b = {b['id']: b for b in cls.city['buildings']}

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_schema_and_contract(self):
        self.assertEqual(contract(self.geo), [])
        self.assertEqual(schema.validate(self.geo['provenance']['sources'], 'source_manifest.v1'), [])
        self.assertEqual((self.geo['schema'], self.geo['schema_version']), ('outbreak-geo', 1))

    def test_metadata_and_metres(self):
        m = self.geo['metadata']
        self.assertEqual(m['center'], {'lat': 60.0, 'lon': 15.0})
        self.assertEqual((m['bounds']['min_m'], m['bounds']['max_m'], m['bounds']['size_m']), ([-150.0, -150.0], [150.0, 150.0], 300.0))
        self.assertEqual(m['coordinates']['unit'], 'm')
        self.assertEqual(m['coordinates']['axes'], {'x': 'east', 'y': 'north', 'z': 'up'})
        self.assertEqual((m['chunking']['size_m'], m['chunking']['count']), (60.0, 36))
        self.assertEqual((m['source_city']['slug'], m['source_city']['name']), (fx.SLUG, 'Synthby'))
        self.assertTrue(m['synthetic'])
        wgs = m['bounds']['wgs84']
        self.assertLess(wgs['south'], 60.0)
        self.assertGreater(wgs['north'], 60.0)
        self.assertAlmostEqual((wgs['north'] - wgs['south']) * fx.P.ky, 300, delta=0.1)
        self.assertFalse(any('stud' in k for k in keys_in(self.geo)), 'no studs inside OutbreakGeo')

    def test_stable_osm_ids(self):
        self.assertEqual(set(self.b), {'w10', 'w11', 'w12', 'w13', 'r30'})
        self.assertEqual(self.b['w11']['osm'], {'type': 'way', 'id': 11})
        self.assertEqual(self.b['r30']['osm'], {'type': 'relation', 'id': 30})
        self.assertEqual({b['part'] for b in self.b.values()}, {'building'})
        self.assertEqual(set(self.r), {'w1', 'w2', 'w3'})
        self.assertEqual(self.r['w2']['osm'], {'type': 'way', 'id': 2})

    def test_footprints_preserved(self):
        for fid, b in self.b.items():
            self.assertEqual(b['footprint']['outer'], self.city_b[fid]['footprint']['outer'], fid)
            self.assertEqual(b['footprint']['holes'], self.city_b[fid]['footprint']['holes'], fid)
            self.assertGreater(adapter._signed_area(b['footprint']['outer']), 0, 'outer rings counter-clockwise')
        for fid, (x0, y0, x1, y1, _) in fx.BUILDINGS.items():
            xs = [p[0] for p in self.b[f'w{fid}']['footprint']['outer']]
            ys = [p[1] for p in self.b[f'w{fid}']['footprint']['outer']]
            self.assertEqual((min(xs), min(ys), max(xs), max(ys)), (x0, y0, x1, y1), f'w{fid} keeps its metres')
        court = self.b['r30']
        self.assertEqual(len(court['footprint']['holes']), 1)
        self.assertAlmostEqual(court['area_m2'], 45 * 40 - 21 * 16, delta=0.5)
        self.assertAlmostEqual(self.b['w11']['area_m2'], 45 * 18, delta=0.5)

    def test_heights_levels_and_provenance(self):
        b = self.b
        self.assertGreaterEqual(len({x['wall_height_m'] for x in b.values()}), 4, 'different heights')
        for fid, x in b.items():
            self.assertEqual((x['wall_height_m'], x['levels']), (self.city_b[fid]['h'], self.city_b[fid]['levels']), fid)
            self.assertEqual(x['base_m'], self.city_b[fid]['base'])
            self.assertEqual(x['provenance']['base'], 'derived')
            self.assertEqual((x['min_height_m'], x['provenance']['min_height']), (0.0, 'default'))
        p = {fid: x['provenance'] for fid, x in b.items()}
        self.assertEqual((b['w11']['levels'], p['w11']['levels'], p['w11']['wall_height']), (5, 'osm', 'derived'))
        self.assertEqual((b['w12']['wall_height_m'], p['w12']['wall_height'], p['w12']['levels']), (12.5, 'osm', 'derived'))
        self.assertEqual((b['w12']['roof']['shape'], p['w12']['roof_shape']), ('flat', 'osm'))
        self.assertEqual((b['w12']['facade']['material'], p['w12']['facade_material']), ('brick', 'osm'))
        self.assertEqual((p['w10']['levels'], p['w10']['wall_height'], p['w10']['roof_shape']), ('inferred', 'inferred', 'inferred'))
        self.assertEqual((b['w10']['facade'], p['w10']['facade_material'], p['w10']['facade_colour']),
                         ({'material': None, 'colour': None}, 'none', 'none'), "Wasteland's guessed look is not geography")
        self.assertEqual((b['r30']['levels'], p['r30']['levels']), (4, 'osm'))
        origins = self.geo['provenance']['origins']
        self.assertEqual(origins['inferred']['confidence'], 'low')
        self.assertEqual(origins['osm']['confidence'], 'high')

    def test_landmark_and_lod(self):
        church = self.b['w13']
        self.assertEqual((church['kind'], church['provenance']['kind']), ('church', 'derived'))
        self.assertEqual((church['name'], church['provenance']['name']), ('Synthby kyrka', 'osm'))
        self.assertTrue(church['landmark'])
        self.assertEqual(church['lod'], {'class': 'A', 'basis': 'landmark'})
        for fid in ('w10', 'w11', 'w12', 'r30'):
            self.assertFalse(self.b[fid]['landmark'])
            self.assertEqual(self.b[fid]['lod'], {'class': None, 'basis': 'unassigned'})

    def test_buildings_across_chunk_borders(self):
        self.assertEqual(self.b['w11']['chunks'], ['c-1_-1', 'c0_-1'])      # x −30…15 crosses x = 0
        self.assertEqual(self.b['r30']['chunks'], ['c-3_1', 'c-2_1'])       # x −140…−95 crosses x = −120
        bounds = {c['id']: c['bounds_m'] for c in self.geo['chunks']}
        owned = [fid for c in self.geo['chunks'] for fid in c['buildings']]
        self.assertEqual(sorted(owned), sorted(self.b), 'each building is placed in exactly one chunk')
        for fid, x in self.b.items():
            self.assertIn(x['chunk'], x['chunks'])
            self.assertEqual(chunks.parse_chunk_id(x['chunk']), tuple(self.city_b[fid]['cell']), "the owner is Wasteland's cell")
            pieces = sum(Polygon(p['outer'], p['holes']).area for c in x['chunks'] for p in chunks.clip_polygon(x['footprint'], bounds[c]))
            self.assertAlmostEqual(pieces, x['area_m2'], delta=0.01)

    def test_roads(self):
        r = self.r
        for fid, x in r.items():
            city_r = next(c for c in self.city['roads'] if c['id'] == fid)
            self.assertEqual(x['width_m'], city_r['w'])
            self.assertEqual(x['centerline'], city_r['p'])
            self.assertEqual((x['layer'], x['provenance']['traversal']), (0, 'osm'))
        self.assertEqual(r['w1']['centerline'], [[-170.0, 8.0], [32.0, 8.0], [170.0, 8.0]], 'metres from the OSM input')
        self.assertEqual([r[k]['width_m'] for k in ('w1', 'w2', 'w3')], [6.5, 8.0, 2.4])
        self.assertEqual([r[k]['drivable'] for k in ('w1', 'w2', 'w3')], [True, True, False])
        self.assertEqual((r['w2']['surface'], r['w2']['provenance']['surface']), ('sett', 'osm'))
        self.assertEqual((r['w1']['surface'], r['w1']['provenance']['surface']), (None, 'none'))
        self.assertEqual(r['w1']['provenance']['width'], 'inferred')
        self.assertEqual(r['w2']['name'], 'Kyrkvägen')
        self.assertEqual(self.geo['navigation']['drivable_roads'], ['w1', 'w2'])
        # Storgatan and Kyrkvägen share node 100 at (32, 8); the footway meets nothing.
        self.assertEqual(r['w1']['nodes'], fx.ROAD_NODES[1])
        self.assertEqual(self.geo['navigation']['topology'], 'osm_nodes')
        self.assertEqual(self.geo['navigation']['junctions'],
                         [{'id': 'j:n100', 'point': [32.0, 8.0], 'roads': ['w1', 'w2'], 'node': 100, 'basis': 'osm_node'}])

    def test_road_spans(self):
        w1 = self.r['w1']
        self.assertEqual([s['chunk'] for s in w1['spans']], [f'c{i}_0' for i in range(-3, 3)])
        self.assertAlmostEqual(w1['spans'][0]['s0_m'], 20.0, places=9)
        self.assertAlmostEqual(w1['spans'][-1]['s1_m'], 320.0, places=9)
        self.assertEqual(w1['length_m'], 340.0)
        for a, b in zip(w1['spans'], w1['spans'][1:]):
            self.assertEqual(a['s1_m'], b['s0_m'])
        w2 = self.r['w2']
        self.assertEqual([s['chunk'] for s in w2['spans']], [f'c0_{j}' for j in range(-3, 3)])
        by_chunk = {c['id']: c for c in self.geo['chunks']}
        for x in self.r.values():
            for c in x['chunks']:
                self.assertIn(x['id'], by_chunk[c]['roads'])

    def test_areas(self):
        areas = self.geo['areas']
        self.assertTrue({'water', 'carriageway', 'sidewalk', 'pedestrian', 'park', 'grass'} <= {a['class'] for a in areas})
        water = [a for a in areas if a['class'] == 'water']
        self.assertAlmostEqual(sum(a['area_m2'] for a in water), 55 * 50, delta=55 * 50 * 0.02)
        self.assertEqual({a['origin'] for a in areas if a['class'] == 'sidewalk'}, {'inferred'})
        self.assertEqual({a['origin'] for a in areas if a['class'] == 'grass'}, {'mixed'})
        self.assertEqual({a['surface'] for a in areas if a['class'] == 'carriageway'}, {'asphalt', 'cobblestone'})
        bounds = {c['id']: c['bounds_m'] for c in self.geo['chunks']}
        for a in areas:
            pieces = sum(Polygon(p['outer'], p['holes']).area for c in a['chunks'] for p in chunks.clip_polygon(a['polygon'], bounds[c]))
            self.assertAlmostEqual(pieces, a['area_m2'], delta=0.01 + a['area_m2'] * 1e-6)
        self.assertGreater(len({c for a in areas for c in a['chunks']}), 1)

    def test_collision_from_the_buildings(self):
        coll = {c['id']: c for c in self.geo['navigation']['collision']}
        self.assertGreater(len(self.city['barriers']), 0, "the battle game's edge containers are in city.json")
        self.assertEqual(set(coll), {f'col:{fid}' for fid in self.b}, 'one mapped outline per building, no containers')
        for fid, b in self.b.items():
            c = coll[f'col:{fid}']
            self.assertEqual((c['kind'], c['osm'], c['buildings'], c['chunks']), ('building', b['osm'], [fid], b['chunks']))
            self.assertAlmostEqual(Polygon(c['polygon']['outer'], c['polygon']['holes']).area, b['area_m2'], delta=0.05)
        self.assertEqual(len(coll['col:r30']['polygon']['holes']), 1, 'the courtyard stays open')
        self.assertFalse(any('outline' in w for w in self.geo['metadata']['warnings']))

    def test_terrain(self):
        t, ct = self.geo['terrain'], self.city['terrain']
        self.assertEqual(t['grid']['origin_m'], [ct['x0'], ct['y0']])
        self.assertEqual((t['grid']['step_m'], t['grid']['n']), (ct['step'], ct['n']))
        self.assertEqual(t['heights_m'], ct['d'])
        self.assertEqual(t['kind'], 'dtm')
        self.assertAlmostEqual(t['vertical_datum']['source_height_of_zero_m'], ct['lake'] - self.city['water_z'], places=3)
        self.assertEqual(t['vertical_datum']['scale'], 1.0)
        lo, hi = t['range_in_bounds_m']
        self.assertTrue(0 <= lo < hi < 10)

    def test_attribution_and_licences(self):
        prov = self.geo['provenance']
        self.assertEqual(prov['attribution'], [adapter.OSM_ATTRIBUTION, 'Terrain: synthetic test grid'])
        self.assertEqual(prov['licenses'][0]['license'], 'ODbL-1.0')
        self.assertIsNone(prov['licenses'][1]['license'])
        self.assertTrue(any('synthetic input' in w for w in self.geo['metadata']['warnings']))
        lic = adapter._licences(['Höjddata: Markhöjdmodell © Lantmäteriet (CC BY 4.0)', 'Terrain: Copernicus DEM GLO-30 © DLR e.V.'], False, [])
        self.assertEqual([x['license'] for x in lic][:2], ['ODbL-1.0', 'CC-BY-4.0'])
        self.assertEqual(lic[2]['data'], 'Copernicus DEM GLO-30')
        with self.assertRaises(adapter.PolicyError):
            adapter._licences(['Unknown DEM'], False, [])
        self.assertIn('Street View', prov['policy']['street_view'])
        self.assertIn('refused', prov['policy']['wasteland_refinements'])
        self.assertTrue(prov['policy']['city_json'].startswith('Verified'))
        self.assertNotIn('wasteland_override', prov['origins'])

    def test_source_manifest(self):
        s = self.geo['provenance']['sources']
        for key, name in (('city', 'city.json'), ('osm', 'osm.json'), ('terrain', 'terrain.json'), ('place', 'place.json')):
            rec = s['inputs'][key]
            raw = (self.folder / name).read_bytes()
            self.assertEqual((rec['file'], rec['present'], rec['sha256'], rec['bytes'], rec['encoding']),
                             (name, True, hashlib.sha256(raw).hexdigest(), len(raw), 'utf-8' if raw.isascii() else rec['encoding']))
        self.assertTrue(s['inputs']['city']['clean_rebuild'])
        self.assertEqual(s['inputs']['osm']['elements'], len(fx.elements()))
        self.assertEqual((s['inputs']['terrain']['kind'], s['inputs']['terrain']['tiles']), ('dtm', ['synthetic-1']))
        self.assertEqual((s['inputs']['wasteland_overrides']['present'], s['inputs']['wasteland_overrides']['build_settings']), (False, {}))
        self.assertEqual((s['pipeline']['commit'], s['pipeline']['dirty']), (GEN['commit'], False))
        code = s['pipeline']['code_sha256']
        for f in ('pipeline/prepare_city.py', 'pipeline/style.json', 'pipeline/common.py', 'pipeline/outbreak/adapter.py',
                  'pipeline/outbreak/schemas/outbreak_geo.v1.schema.json'):
            self.assertEqual(code[f], hashlib.sha256((ROOT / f).read_bytes().replace(b'\r\n', b'\n')).hexdigest(), f)
        self.assertEqual(set(s['pipeline']['runtime']), {'python', 'shapely', 'geos', 'numpy'})
        self.assertEqual(s['place']['center'], {'lat': 60.0, 'lon': 15.0})
        self.assertIsNotNone(s['place']['bbox_wgs84'])
        self.assertIn('SHA-256', s['pinning'])

    def test_no_render_data(self):
        self.assertEqual(set(self.geo), {'schema', 'schema_version', 'metadata', 'provenance', 'terrain', 'roads', 'buildings', 'areas',
                                         'navigation', 'chunks'})
        self.assertEqual(MESH_KEYS & set(keys_in(self.geo)), set())
        self.assertLess(len(adapter.dumps(self.geo)), len(json.dumps(self.city)) / 3)

    def test_deterministic(self):
        again = adapter.convert_city_dir(self.folder, generator=GEN, synthetic=True)
        self.assertEqual(adapter.dumps(again), adapter.dumps(self.geo))
        # The order of features in city.json and osm.json doesn't change a single byte.
        terrain = adapter.read_json(self.folder / 'terrain.json', [])[0]
        ordered = adapter.build_geo(copy.deepcopy(self.city), slug=fx.SLUG, osm=copy.deepcopy(self.osm), terrain_src=terrain,
                                    generator=GEN, synthetic=True)
        city = copy.deepcopy(self.city)
        for key in ('buildings', 'roads', 'land', 'walls'):
            city[key].reverse()
        for layer in city['areas'].values():
            layer.reverse()
        osm = copy.deepcopy(self.osm)
        osm['elements'].reverse()
        shuffled = adapter.build_geo(city, slug=fx.SLUG, osm=osm, terrain_src=terrain, generator=GEN, synthetic=True)
        self.assertEqual(adapter.dumps(shuffled), adapter.dumps(ordered))
        for key in ('roads', 'buildings', 'areas', 'navigation', 'chunks', 'terrain'):
            self.assertEqual(ordered[key], self.geo[key], key)

    def test_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            # The fixture's terrain is synthetic, with no licence: the command line refuses it.
            out = Path(tmp) / 'a.json'
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as cm:
                adapter.main([str(self.folder), '--out', str(out)])
            self.assertIn('terrain licence not recognised', str(cm.exception))
            self.assertFalse(out.exists())
            # A flat copy needs no terrain licence; two runs give the same bytes.
            flat = prepared(Path(tmp) / 'flat', terrain=False)
            outs = [Path(tmp) / 'a.json', Path(tmp) / 'b.json']
            for o in outs:
                with contextlib.redirect_stdout(io.StringIO()):
                    adapter.main([str(flat), '--out', str(o)])
            a, b = (p.read_bytes() for p in outs)
            self.assertEqual(a, b)
            geo = json.loads(a.decode('utf-8'))
            self.assertEqual(contract(geo), [])
            self.assertFalse(geo['metadata']['synthetic'])
            self.assertIn('Kyrkvägen'.encode('utf-8'), a)
            # A Windows pipe (cp1252) can't show every character of a place or a warning; the CLI must not fail on it.
            console = io.TextIOWrapper(io.BytesIO(), encoding='cp1252')
            with contextlib.redirect_stdout(console):
                adapter.main([str(flat), '--out', str(outs[0])])
            console.flush()
            self.assertIn(b'OutbreakGeo v1', console.buffer.getvalue())

    def test_chunk_size_is_a_setting(self):
        geo = adapter.convert_city_dir(self.folder, generator=GEN, chunk_size=100, synthetic=True)
        self.assertEqual(contract(geo), [])
        self.assertEqual((geo['metadata']['chunking']['size_m'], geo['metadata']['chunking']['count']), (100.0, 16))
        self.assertEqual({c['id'] for c in geo['chunks']}, {f'c{i}_{j}' for i in range(-2, 2) for j in range(-2, 2)})
        self.assertEqual({b['id']: b['chunks'] for b in geo['buildings']}['w11'], ['c-1_-1', 'c0_-1'])

    def test_provenance_unknown_without_osm(self):
        geo = adapter.build_geo(copy.deepcopy(self.city), generator=GEN, synthetic=True)
        p = {b['id']: b['provenance'] for b in geo['buildings']}
        self.assertEqual((p['w12']['wall_height'], p['w12']['levels'], p['w12']['facade_material']), ('unknown', 'unknown', 'unknown'))
        self.assertEqual({b['min_height_m'] for b in geo['buildings']}, {None})
        self.assertTrue(any('osm.json not available' in w for w in geo['metadata']['warnings']))
        self.assertTrue(geo['provenance']['policy']['city_json'].startswith('Not verified'))
        self.assertFalse(geo['provenance']['sources']['inputs']['city']['clean_rebuild'])
        self.assertEqual(contract(geo), [])


# ------------------------------------------------------------------------------------------ provenance of city.json
class ProvenanceTest(unittest.TestCase):
    """The policy on Wasteland refinements and the clean-rebuild check."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_overrides_emptied_after_the_build_are_caught(self):
        """The navigator's counter-example: built with an override, overrides.json removed afterwards."""
        folder = prepared(self.tmp / 'city', overrides={'buildings': {'w10': {'height': 50}}})
        self.assertEqual({b['id']: b['h'] for b in json.loads((folder / 'city.json').read_text())['buildings']}['w10'], 50)
        (folder / 'overrides.json').unlink()
        before = {p.name: p.read_bytes() for p in folder.iterdir()}
        with self.assertRaises(adapter.PolicyError) as cm:
            adapter.convert_city_dir(folder, generator=GEN, synthetic=True)
        self.assertIn('buildings', str(cm.exception))
        self.assertEqual({p.name: p.read_bytes() for p in folder.iterdir()}, before, 'the check changes nothing in the folder')

    def test_prepare_is_deterministic_across_processes(self):
        """The clean-rebuild check relies on it: prepare_city's output may not depend on the hash seed."""
        code = ('import contextlib, hashlib, io, sys; from pathlib import Path; sys.path[:0] = sys.argv[2:]; '
                'import common, outbreak_fixture as fx, prepare_city; f = Path(sys.argv[1]); fx.write(f); common.CITIES = f.parent; '
                'contextlib.redirect_stdout(io.StringIO()).__enter__(); prepare_city.main([f.name]); '
                'sys.__stdout__.write(hashlib.sha256((f / "city.json").read_bytes()).hexdigest())')
        digests = set()
        for seed in ('1', '2'):
            out = subprocess.run([sys.executable, '-c', code, str(self.tmp / f'seed{seed}' / 'city'), str(ROOT / 'pipeline'), str(ROOT / 'tests')],
                                 capture_output=True, text=True, check=True, env={**os.environ, 'PYTHONHASHSEED': seed, 'PYTHONUTF8': '1'})
            digests.add(out.stdout.strip())
        self.assertEqual(len(digests), 1, digests)

    def test_a_city_json_from_other_inputs_is_caught(self):
        folder = prepared(self.tmp / 'city')
        city = json.loads((folder / 'city.json').read_text())
        city['roads'][0]['w'] = 9.9
        (folder / 'city.json').write_text(json.dumps(city))
        with self.assertRaises(adapter.PolicyError):
            adapter.convert_city_dir(folder, generator=GEN, synthetic=True)
        (folder / 'osm.json').unlink()
        with self.assertRaises(adapter.PolicyError) as cm:
            adapter.convert_city_dir(folder, generator=GEN, synthetic=True)
        self.assertIn('osm.json', str(cm.exception))

    def test_build_settings_are_used_by_the_rebuild(self):
        folder = prepared(self.tmp / 'city', overrides={'defaults': {'terrain_scale': 2.0}})
        geo = adapter.convert_city_dir(folder, generator=GEN, synthetic=True)
        self.assertEqual(geo['terrain']['vertical_datum']['scale'], 2.0)
        self.assertEqual(geo['provenance']['sources']['inputs']['wasteland_overrides']['build_settings'], {'terrain_scale': 2.0})
        self.assertEqual(contract(geo), [])

    def test_wasteland_refinements_are_refused(self):
        folder = prepared(self.tmp / 'city')
        city = json.loads((folder / 'city.json').read_text())
        refused = [{'buildings': {'w10': {'colour': 'red', 'note': 'seen in Street View'}}}, {'roads': {'w1': {'hide': True}}},
                   {'areas': [{'center': [60.0, 15.0], 'radius': 40, 'buildings': {'colour_mix': {'falu': 1}}}]},
                   {'defaults': {'max_levels': 2}}, {'defaults': {'terrain_scale': 'big'}}, {'defaults': None}, {'notes': 'x'}, []]
        for ov in refused:
            with self.assertRaises(adapter.PolicyError, msg=ov) as cm:
                adapter.build_geo(city, overrides=ov, synthetic=True)
            self.assertIn('refinement', str(cm.exception))
        # No escape hatch, in the API or on the command line.
        with self.assertRaises(TypeError):
            adapter.build_geo(city, overrides=refused[0], allow_wasteland_overrides=True)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            adapter.main([str(folder), '--allow-wasteland-overrides'])
        # Empty sections change nothing about the place.
        geo = adapter.build_geo(city, overrides={'defaults': {'terrain': True}, 'buildings': {}, 'areas': []}, synthetic=True)
        self.assertEqual(geo['provenance']['sources']['inputs']['wasteland_overrides']['build_settings'], {'terrain': True})

    def test_refined_folder_is_refused(self):
        """Any refine-city file makes the folder unusable, even when its overrides.json looks clean."""
        for name, content in (('overrides.json', '{"buildings": {"w12": {"levels": 4}}}'), ('refinements.md', '# Round 1\n'),
                              ('custom/w13.py', 'ctx.box()\n')):
            with self.subTest(name):
                folder = prepared(self.tmp / name.replace('/', '_'), terrain=False)
                (folder / name).parent.mkdir(parents=True, exist_ok=True)
                (folder / name).write_text(content, encoding='utf-8')
                with self.assertRaises(adapter.PolicyError) as cm:
                    adapter.convert_city_dir(folder, generator=GEN)
                self.assertIn(name.split('/')[0], str(cm.exception))
                out = self.tmp / 'geo.json'
                with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
                    adapter.main([str(folder), '--out', str(out)])
                self.assertFalse(out.exists())

    def test_terrain_licence_must_be_known(self):
        terrain = {'x0': -100, 'y0': -100, 'step': 100, 'n': 3, 'd': [0.0] * 9, 'lake': 1.2}
        for attribution in ('Unknown DEM', None, ''):
            city = minimal_city(terrain={**terrain, 'attribution': attribution})
            with self.assertRaises(adapter.PolicyError, msg=attribution):
                adapter.build_geo(city)
            geo = adapter.build_geo(city, synthetic=True)
            self.assertTrue(geo['metadata']['synthetic'])
        geo = adapter.build_geo(minimal_city(terrain={**terrain, 'attribution': 'Höjddata: Markhöjdmodell © Lantmäteriet (CC BY 4.0)'}))
        self.assertEqual(geo['provenance']['licenses'][1]['license'], 'CC-BY-4.0')
        self.assertFalse(geo['metadata']['synthetic'])

    def test_conversion_is_the_same_under_any_locale(self):
        """cp1252 inputs, as Wasteland writes them on Western Windows: the whole conversion, clean rebuild
        included, gives the same bytes with PYTHONUTF8 off and on."""
        folder = prepared(self.tmp / 'city', terrain=False)
        for name in ('place.json', 'osm.json'):
            data = json.loads((folder / name).read_text(encoding='ascii'))
            (folder / name).write_bytes(json.dumps(data, ensure_ascii=False).encode('cp1252'))
        self.assertIn('Kyrkvägen'.encode('cp1252'), (folder / 'osm.json').read_bytes())
        code = ('import hashlib, sys; sys.path[:0] = [sys.argv[2]]; from pathlib import Path; from outbreak import adapter; '
                'geo = adapter.convert_city_dir(Path(sys.argv[1]), generator={"commit": None, "dirty": None}); '
                'print(hashlib.sha256(adapter.dumps(geo).encode()).hexdigest(), geo["provenance"]["sources"]["inputs"]["osm"]["encoding"])')
        outs = {subprocess.run([sys.executable, '-c', code, str(folder), str(ROOT / 'pipeline')], capture_output=True, text=True, check=True,
                               env={**os.environ, 'PYTHONUTF8': utf8}).stdout.strip() for utf8 in ('0', '1')}
        self.assertEqual(len(outs), 1, outs)
        self.assertTrue(outs.pop().endswith(' cp1252'))

    def test_names_outside_cp1252_convert_under_any_locale(self):
        """A name cp1252 can't hold (Ł): the clean rebuild runs prepare_city in UTF-8 mode, so the conversion
        works and gives the same bytes with PYTHONUTF8 off and on."""
        osm = {'version': 0.6, 'generator': 'outbreak test fixture', 'osm3s': {'timestamp_osm_base': 'synthetic'},
               'elements': fx.elements() + [fx.way(80, [(-140, -140), (-60, -140)], {'highway': 'residential', 'name': 'Ulica Łódzka'})]}
        folder = self.tmp / 'lodz'
        fx.write(folder)
        (folder / 'terrain.json').unlink()
        (folder / 'osm.json').write_text(json.dumps(osm), encoding='ascii')
        city = adapter.clean_rebuild(folder, {})
        (folder / 'city.json').write_bytes(json.dumps(city, ensure_ascii=False).encode('utf-8'))
        self.assertIn('Ulica Łódzka', {r['name'] for r in city['roads']})
        code = ('import hashlib, sys; sys.path[:0] = [sys.argv[2]]; from pathlib import Path; from outbreak import adapter; '
                'geo = adapter.convert_city_dir(Path(sys.argv[1]), generator={"commit": None, "dirty": None}); '
                'print(hashlib.sha256(adapter.dumps(geo).encode()).hexdigest(), ascii([r["name"] for r in geo["roads"] if r["id"] == "w80"]))')
        outs = {subprocess.run([sys.executable, '-c', code, str(folder), str(ROOT / 'pipeline')], capture_output=True, text=True, check=True,
                               env={**os.environ, 'PYTHONUTF8': utf8}).stdout.strip() for utf8 in ('0', '1')}
        self.assertEqual(len(outs), 1, outs)
        self.assertTrue(outs.pop().endswith("['Ulica \\u0141\\xf3dzka']"))

    def test_legacy_encoding_is_the_same_under_any_locale(self):
        path = self.tmp / 'place.json'
        path.write_bytes('{"name": "Kyrkvägen"}'.encode('cp1252'))
        code = ('import sys; sys.path[:0] = [sys.argv[2]]; from pathlib import Path; from outbreak import adapter; '
                'w = []; d, rec = adapter.read_json(Path(sys.argv[1]), w); print(ascii(d["name"]), rec["encoding"], len(w))')
        outs = {subprocess.run([sys.executable, '-c', code, str(path), str(ROOT / 'pipeline')], capture_output=True, text=True, check=True,
                               env={**os.environ, 'PYTHONUTF8': utf8}).stdout.strip() for utf8 in ('0', '1')}
        self.assertEqual(outs, {"'Kyrkv\\xe4gen' cp1252 1"})


# ------------------------------------------------------------------------------------------ through the preprocessor
class ThroughPrepareTest(unittest.TestCase):
    """Cases whose cause is in prepare_city.py itself, so they go through it, not hand-made city.json."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        extra = [
            # An outline mostly covered by one part: prepare_city drops the 4.8 m strip left beside it (under 20 %).
            fx.way(60, fx.rect(80, 60, 120, 100), {'building': 'yes'}),
            fx.way(61, fx.rect(85, 60, 120, 100), {'building:part': 'yes', 'building:levels': '3'}),
            # A building that starts two storeys up (raised by building:min_level, no min_height).
            fx.way(62, fx.rect(90, 110, 110, 130), {'building': 'yes', 'building:min_level': '2', 'building:levels': '4'}),
            # A ground outline with a part from 5 m that overhangs it by 2.5 m to the west.
            fx.way(63, fx.rect(0, -140, 20, -120), {'building': 'yes'}),
            fx.way(64, fx.rect(-2.5, -140, 20, -120), {'building:part': 'yes', 'min_height': '5', 'height': '9'}),
            # A street with nodes 1-2-3, a bridge drawn through node 2's position with nodes 4-5-6, and a lane
            # that really leaves the street at node 3.
            {**fx.way(70, [(-140, 120), (-100, 120), (-60, 120)], {'highway': 'residential'}), 'nodes': [1, 2, 3]},
            {**fx.way(71, [(-100, 140), (-100, 120), (-100, 105)], {'highway': 'primary', 'bridge': 'yes', 'layer': '1'}), 'nodes': [4, 5, 6]},
            {**fx.way(72, [(-60, 120), (-60, 140)], {'highway': 'service'}), 'nodes': [3, 7]},
        ]
        osm = {'version': 0.6, 'generator': 'outbreak test fixture', 'osm3s': {'timestamp_osm_base': 'synthetic'},
               'elements': fx.elements() + extra}
        cls.folder = prepared(cls.tmp / 'city', osm=osm)
        cls.city = adapter.read_json(cls.folder / 'city.json', [])[0]
        cls.geo = adapter.convert_city_dir(cls.folder, generator=GEN, synthetic=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_contract(self):
        self.assertEqual(contract(self.geo), [])

    def test_an_outline_leftover_still_blocks(self):
        ids = {b['id'] for b in self.city['buildings']}
        self.assertIn('w61', ids)
        self.assertFalse({'w60', 'w60r'} & ids, 'prepare_city dropped the leftover beside the part')
        coll = {c['id']: c for c in self.geo['navigation']['collision']}
        c = coll['col:w60']
        self.assertEqual((c['osm'], c['buildings']), ({'type': 'way', 'id': 60}, ['w61']))
        self.assertAlmostEqual(Polygon(c['polygon']['outer'], c['polygon']['holes']).area, 40 * 40, delta=0.5)
        self.assertNotIn('col:w61', coll, 'the part is held by its outline')
        self.assertTrue(any('building outline w60' in w and 'm²' in w for w in self.geo['metadata']['warnings']))

    def test_raised_geometry_never_blocks_the_ground(self):
        b = {x['id']: x for x in self.geo['buildings']}
        self.assertEqual(b['w62']['min_height_m'], 2 * prepare_city.STYLE['storey_m'])
        coll = {c['id']: c for c in self.geo['navigation']['collision']}
        self.assertFalse([c for c in coll.values() if 'w62' in c['buildings'] or c['osm'] == {'type': 'way', 'id': 62}],
                         'raised by building:min_level: no ground collision')
        c = coll['col:w63']
        self.assertEqual(c['buildings'], [], 'the part from 5 m is not a ground volume')
        self.assertAlmostEqual(Polygon(c['polygon']['outer'], c['polygon']['holes']).area, 20 * 20, delta=0.5,
                               msg='the overhang adds nothing outside the outline')
        self.assertEqual(b['w64']['min_height_m'], 5.0)

    def test_ground_classes_are_disjoint(self):
        """The navigator's case: a round residential street (64 points, radius 100 m) in a park. prepare_city
        simplifies each ground layer on its own, so carriageway and sidewalk overlap in city.json; in OutbreakGeo
        each piece of land has one class, the higher layer keeping it."""
        ring = [(100 * math.cos(2 * math.pi * k / 64), 100 * math.sin(2 * math.pi * k / 64)) for k in range(64)]
        osm = {'version': 0.6, 'generator': 'outbreak test fixture', 'osm3s': {'timestamp_osm_base': 'synthetic'},
               'elements': [fx.way(1, ring + ring[:1], {'highway': 'residential'}), fx.way(2, fx.rect(-145, -145, 145, 145), {'leisure': 'park'})]}
        folder = prepared(self.tmp / 'roundabout', osm=osm, terrain=False)
        city = adapter.read_json(folder / 'city.json', [])[0]
        layer = {k: unary_union([Polygon(p['outer'], p['holes']) for p in city['areas'].get(k, [])]) for k in ('asphalt', 'sidewalk', 'park')}
        self.assertGreater(layer['asphalt'].intersection(layer['sidewalk']).area, 1.0, 'the overlap is real in city.json')
        geo = adapter.convert_city_dir(folder, generator=GEN)
        self.assertEqual(contract(geo), [])
        by_class = {c: unary_union([Polygon(a['polygon']['outer'], a['polygon']['holes']) for a in geo['areas'] if a['class'] == c])
                    for c in ('carriageway', 'sidewalk', 'park')}
        self.assertLess(by_class['carriageway'].intersection(by_class['sidewalk']).area, 1e-4)
        self.assertAlmostEqual(by_class['carriageway'].area, layer['asphalt'].area, delta=0.01, msg='the higher layer keeps its ground')
        self.assertAlmostEqual(by_class['sidewalk'].area, layer['sidewalk'].difference(layer['asphalt']).area, delta=0.05)

    def test_the_same_place_is_not_the_same_node(self):
        r = {x['id']: x for x in self.geo['roads']}
        self.assertEqual((r['w70']['nodes'], r['w71']['nodes']), ([1, 2, 3], [4, 5, 6]))
        self.assertEqual(r['w71']['centerline'][1], r['w70']['centerline'][1], 'they cross at one position')
        junctions = {j['id']: j for j in self.geo['navigation']['junctions']}
        self.assertEqual(set(junctions), {'j:n100', 'j:n3'})
        self.assertEqual(junctions['j:n3']['roads'], ['w70', 'w72'])
        self.assertEqual(self.geo['navigation']['topology'], 'osm_nodes')


# ------------------------------------------------------------------------------------------ edge cases
class EdgeCaseTest(unittest.TestCase):
    def test_minimal_input(self):
        geo = adapter.build_geo(minimal_city())
        self.assertEqual(contract(geo), [])
        self.assertEqual((geo['buildings'], geo['roads'], geo['areas'], geo['terrain']), ([], [], [], None))
        self.assertEqual(geo['metadata']['chunking']['count'], 16)
        warnings = ' | '.join(geo['metadata']['warnings'])
        for text in ('flat', 'osm.json not available', 'no "land"'):
            self.assertIn(text, warnings)
        self.assertEqual(geo['provenance']['attribution'], [adapter.OSM_ATTRIBUTION])

    def test_flat_land_without_water(self):
        land = [{'outer': [[-100, -100], [100, -100], [100, 100], [-100, 100]], 'holes': []}]
        geo = adapter.build_geo(minimal_city(land=land))
        self.assertEqual(geo['areas'], [])

    def test_malformed_input_is_refused(self):
        bow_tie = {'outer': [[0, 0], [10, 10], [10, 0], [0, 10]], 'holes': []}
        square = {'outer': [[0, 0], [10, 0], [10, 10], [0, 10]], 'holes': []}
        bad = [
            [1, 2, 3],
            minimal_city(version=2),
            {'version': 1, 'half': 100},
            {'version': 1, 'place': {'center': ['60', '15']}, 'half': 100},
            {'version': 1, 'place': {'center': [95.0, 15.0]}, 'half': 100},
            minimal_city(half=0),
            minimal_city(buildings={'w1': {}}),
            minimal_city(buildings=[{'footprint': square['outer'], 'h': 5, 'levels': 1}]),                 # no id
            minimal_city(buildings=[building('w1', 0, 0, 10, 10, footprint=bow_tie)]),                     # self-intersecting
            minimal_city(buildings=[building('w1', 0, 0, 10, 10, footprint={**square, 'holes': [[[1, 1], [2, 2]]]})]),
            minimal_city(buildings=[building('w1', 0, 0, 10, 10, footprint={**square, 'holes': [[[20, 20], [30, 20], [30, 30]]]})]),
            minimal_city(buildings=[building('w1', 90, 0, 110, 10)]),                                        # outside the bounds
            minimal_city(buildings=[building('w1', 0, 0, 10, 10, h=float('nan'))]),
            minimal_city(buildings=[building('w1', 0, 0, 10, 10, levels=0)]),
            minimal_city(roads=[road('w7', [[0, 0]])]),
            minimal_city(roads=[{'id': 'w8', 'kind': 'residential', 'p': [[0, 0], [10, 0]]}]),
            minimal_city(areas=[]),
            minimal_city(areas={'park': 4}),
            minimal_city(areas={'lava': []}),
            minimal_city(areas={'park': [bow_tie]}),
            minimal_city(land=[bow_tie]),
            minimal_city(land={'outer': []}),
            minimal_city(walls=[{'kind': 'city_wall', 'poly': [[0, 0], [1, 1]]}]),
            minimal_city(terrain={'x0': 0, 'y0': 0, 'step': 1, 'n': 3, 'd': [0.0] * 8}),
            minimal_city(terrain={'x0': 0, 'y0': 0, 'step': 1, 'n': 2, 'd': [0.0, 1.0, float('nan'), 2.0]}),
        ]
        for city in bad:
            with self.assertRaises(adapter.AdapterError, msg=repr(city)[:120]):
                adapter.build_geo(city, synthetic=True)
        with self.assertRaises(ValueError):
            adapter.build_geo(minimal_city(), chunk_size=0)
        # min_height (from OSM) at or above the top of the walls
        with self.assertRaises(adapter.AdapterError):
            adapter.build_geo(minimal_city(buildings=[building('w1', 0, 0, 10, 10)]), osm=osm_of(w1={'building': 'yes', 'min_height': '6'}))

    def test_a_road_outside_the_half_open_bounds_is_skipped_with_a_warning(self):
        geo = adapter.build_geo(minimal_city(roads=[road('w9', [[-200, 0], [-150, 0]]), road('w8', [[100, -10], [100, 10]]),
                                                    road('w6', [[-50, 5], [50, 5]])]))
        self.assertEqual([r['id'] for r in geo['roads']], ['w6'])
        warnings = ' | '.join(geo['metadata']['warnings'])
        self.assertIn('road w9', warnings)
        self.assertIn('road w8', warnings)

    def test_one_element_in_pieces(self):
        pieces = [building('w5', 0, 0, 10, 10), building('w5', -40, 0, -30, 10)]
        a = adapter.build_geo(minimal_city(buildings=pieces))
        b = adapter.build_geo(minimal_city(buildings=pieces[::-1]))
        self.assertEqual([x['id'] for x in a['buildings']], ['w5#1', 'w5#2'])
        self.assertEqual(adapter.dumps(a), adapter.dumps(b))
        self.assertEqual({x['osm']['id'] for x in a['buildings']}, {5})

    def test_duplicates(self):
        # One id, one footprint, two heights: which one is w1#1 would depend on the input order. Refused.
        with self.assertRaises(adapter.AdapterError):
            adapter.build_geo(minimal_city(buildings=[building('w1', 0, 0, 10, 10, h=6.0), building('w1', 0, 0, 10, 10, h=9.0)]))
        # Exact duplicates collapse to one, with a warning, whatever the order.
        twice = minimal_city(buildings=[building('w1', 0, 0, 10, 10)] * 2,
                             areas={'park': [{'outer': [[0, 0], [10, 0], [10, 10], [0, 10]], 'holes': []}] * 2})
        geo = adapter.build_geo(twice)
        self.assertEqual(([b['id'] for b in geo['buildings']], len(geo['areas'])), (['w1'], 1))
        self.assertEqual(sum('duplicate' in w or 'more than once' in w for w in geo['metadata']['warnings']), 2)
        self.assertEqual(contract(geo), [])

    def test_osm_origin_only_for_unchanged_tags(self):
        city = minimal_city(buildings=[building('w1', 0, 0, 10, 10), building('w2', 20, 0, 30, 10, h=9.0), building('w3', 40, 0, 50, 10)],
                            roads=[road('w5', [[-50, -20], [50, -20]], 'primary', 22.0)])
        osm = osm_of(w1={'building': 'house', 'height': '9', 'roof:shape': 'gabled', 'roof:height': '3'},
                     w2={'building': 'yes', 'height': '9'}, w3={'building': 'yes', 'building:levels': '2.5'},
                     w5={'highway': 'primary', 'width': '30'})
        geo = adapter.build_geo(city, osm=osm)
        p = {b['id']: b['provenance'] for b in geo['buildings']}
        self.assertEqual(p['w1']['wall_height'], 'derived')                            # 9 m tagged, minus a 3 m roof
        self.assertEqual((p['w2']['wall_height'], p['w2']['levels']), ('osm', 'derived'))
        self.assertEqual((p['w3']['levels'], p['w3']['wall_height']), ('derived', 'derived'))   # 2.5 storeys became 2
        self.assertEqual(geo['roads'][0]['provenance']['width'], 'derived')             # 30 m tagged, clamped to 22

    def test_mapped_look_is_kept_verbatim(self):
        """Wasteland turns these into its palette (falu, wood, brown); OutbreakGeo keeps the mapped values."""
        city = minimal_city(buildings=[building('w1', 0, 0, 10, 10, style='wood', colour='falu', roof_style='tiles', roof_colour='brown')])
        osm = osm_of(w1={'building': 'house', 'building:colour': '#123456', 'building:material': 'timber_framing', 'roof:colour': '#654321'})
        b = adapter.build_geo(city, osm=osm)['buildings'][0]
        self.assertEqual(b['facade'], {'material': 'timber_framing', 'colour': '#123456'})
        self.assertEqual(b['roof'], {'shape': 'pitched', 'material': None, 'colour': '#654321'})
        p = b['provenance']
        self.assertEqual((p['facade_material'], p['facade_colour'], p['roof_colour'], p['roof_material'], p['roof_shape']),
                         ('osm', 'osm', 'osm', 'none', 'inferred'))
        self.assertNotIn('falu', adapter.dumps({'b': b}))

    def test_collision_semantics(self):
        city = minimal_city(
            buildings=[building('w40', 0, 0, 10, 10),                                  # a part on the ground, no main outline
                       building('w41', 20, 0, 40, 20),                                 # a passage runs through it
                       building('w42', -40, 0, -30, 10, h=9.0)],                       # a part high above the ground
            roads=[road('w50', [[30, -10], [30, 30]], 'footway', 2.4), road('w51', [[-80, 50], [80, 50]])],
            walls=[{'kind': 'city_wall', 'poly': [[-90, 60], [120, 60], [120, 63], [-90, 63]]},
                   {'kind': 'hedge', 'poly': [[0, 70], [5, 70], [5, 71]]}])
        osm = osm_of(w40={'building:part': 'yes'}, w41={'building': 'yes'}, w42={'building:part': 'yes', 'min_height': '4'},
                     w50={'highway': 'footway', 'tunnel': 'building_passage'}, w51={'highway': 'residential'})
        geo = adapter.build_geo(city, osm=osm)
        self.assertEqual(contract(geo), [])
        coll = geo['navigation']['collision']
        by_building = defaultdict_list((m, c) for c in coll for m in c['buildings'])
        self.assertEqual(len(by_building['w40']), 1, 'a ground-standing part blocks')
        self.assertNotIn('w42', by_building, 'a part starting at 4 m does not')
        self.assertEqual(len(by_building['w41']), 2, 'the passage splits the block')
        self.assertAlmostEqual(sum(Polygon(c['polygon']['outer']).area for c in by_building['w41']), 400 - 20 * (2.4 + 0.6), places=6)
        walls = [c for c in coll if c['kind'] == 'city_wall']
        self.assertEqual(len(walls), 1, 'the city wall, clipped to the bounds; no hedge')
        self.assertEqual(max(x for x, _ in walls[0]['polygon']['outer']), 100.0)

    def test_only_a_covered_way_at_ground_level_opens_a_building(self):
        city = minimal_city(buildings=[building(f'w{k}', 20 * k - 90, 0, 20 * k - 75, 20) for k in (1, 2, 3, 4)],
                            roads=[road(f'w{10 + k}', [[20 * k - 82.5, -10], [20 * k - 82.5, 30]], 'footway', 2.4) for k in (1, 2, 3, 4)])
        osm = osm_of(**{f'w{k}': {'building': 'yes'} for k in (1, 2, 3, 4)},
                     w11={'highway': 'footway', 'covered': 'yes'},                              # at ground level: cut
                     w12={'highway': 'footway', 'covered': 'yes', 'layer': '2', 'level': '2'},   # a skyway: the house stays closed
                     w13={'highway': 'footway', 'tunnel': 'building_passage', 'level': '0;1'},   # reaches the ground floor: cut
                     w14={'highway': 'footway', 'covered': 'yes', 'level': 'first'})            # unreadable: not cut, reported
        geo = adapter.build_geo(city, osm=osm)
        self.assertEqual(contract(geo), [])
        area = {k: sum(Polygon(c['polygon']['outer'], c['polygon']['holes']).area for c in geo['navigation']['collision'] if f'w{k}' in c['buildings'])
                for k in (1, 2, 3, 4)}
        self.assertAlmostEqual(area[1], 300 - 20 * 3.0, places=6)
        self.assertAlmostEqual(area[2], 300, places=6)
        self.assertAlmostEqual(area[3], 300 - 20 * 3.0, places=6)
        self.assertAlmostEqual(area[4], 300, places=6)
        self.assertTrue(any('road w14' in w and "can't be read" in w for w in geo['metadata']['warnings']))
        # The contract check holds the same line: collision cut open under a skyway is a hole in the building.
        cut = adapter.build_geo(city, osm=osm_of(**{f'w{k}': {'building': 'yes'} for k in (1, 2, 3, 4)}, w12={'highway': 'footway', 'covered': 'yes'}))
        doc = json.loads(adapter.dumps(cut))
        doc['roads'] = [r if r['id'] != 'w12' else {**r, 'tags': {'covered': 'yes', 'layer': '2', 'level': '2'}} for r in doc['roads']]
        self.assertTrue(any('building w2' in e and 'no collision' in e for e in adapter.check_geo(doc)))

    def test_conflicting_osm_entries_are_refused(self):
        city = minimal_city(roads=[road('w5', [[-50, 0], [50, 0]])])
        a = {'type': 'way', 'id': 5, 'tags': {'highway': 'residential', 'oneway': 'yes'}}
        b = {'type': 'way', 'id': 5, 'tags': {'highway': 'residential', 'oneway': 'no'}}
        for elements in ([a, b], [b, a]):
            with self.assertRaises(adapter.AdapterError, msg=elements):
                adapter.build_geo(city, osm={'elements': elements})
        # The same entry twice is one element, and the order changes no byte.
        self.assertEqual(adapter.dumps(adapter.build_geo(city, osm={'elements': [a, copy.deepcopy(a)]})),
                         adapter.dumps(adapter.build_geo(city, osm={'elements': [copy.deepcopy(a), a]})))

    def test_road_topology(self):
        """A bridge over a street, drawn through the street's node position: the same place is not the same node."""
        city = minimal_city(roads=[road('w1', [[-80, 0], [0, 0], [80, 0]]), road('w2', [[0, 0], [-40, -80]]),
                                   road('w3', [[0, -80], [0, 0], [0, 80]], 'primary')])
        tags = {'w1': {'highway': 'residential', 'oneway': 'yes', 'access': 'destination'}, 'w2': {'highway': 'residential'},
                'w3': {'highway': 'primary', 'bridge': 'yes', 'layer': '1', 'motor_vehicle': 'no'}}
        nodes = {'w1': [1, 2, 3], 'w2': [2, 9], 'w3': [4, 5, 6]}
        plane = common.Projection(60.0, 15.0, 'wgs84')
        lines = {r['id']: r['p'] for r in city['roads']}
        with_nodes = {'elements': [{'type': 'way', 'id': int(k[1:]), 'tags': t, 'nodes': nodes[k],
                                    'geometry': [dict(zip(('lat', 'lon'), plane.latlon(*p))) for p in lines[k]]} for k, t in tags.items()]}
        geo = adapter.build_geo(city, osm=with_nodes)
        self.assertEqual(contract(geo), [])
        self.assertEqual(geo['navigation']['topology'], 'osm_nodes')
        self.assertEqual(geo['navigation']['junctions'], [{'id': 'j:n2', 'point': [0.0, 0.0], 'roads': ['w1', 'w2'], 'node': 2, 'basis': 'osm_node'}])
        r = {x['id']: x for x in geo['roads']}
        self.assertEqual((r['w3']['nodes'], r['w3']['layer'], r['w3']['tags']), ([4, 5, 6], 1, {'bridge': 'yes', 'layer': '1', 'motor_vehicle': 'no'}))
        self.assertEqual((r['w1']['layer'], r['w1']['tags']), (0, {'oneway': 'yes', 'access': 'destination'}))
        # Without node ids (Wasteland's old download) the shared position is reported, marked unverified, and only
        # for the roads on one level: the bridge at that position is a separated crossing, never part of it.
        geo = adapter.build_geo(city, osm=osm_of(**tags))
        self.assertEqual(contract(geo), [])
        self.assertEqual(geo['navigation']['topology'], 'positions')
        self.assertEqual(geo['navigation']['junctions'],
                         [{'id': 'j:0.00,0.00', 'point': [0.0, 0.0], 'roads': ['w1', 'w2'], 'node': None, 'basis': 'shared_position'}])
        self.assertEqual([(c['roads'], c['status'], c['differ']) for c in geo['navigation']['crossings']],
                         [(['w1', 'w3'], 'separated', ['bridge', 'layer']), (['w2', 'w3'], 'separated', ['bridge', 'layer'])])
        self.assertTrue(any('shared_position' in w for w in geo['metadata']['warnings']))
        self.assertEqual(adapter.build_geo(city)['roads'][0]['layer'], None, 'unknown without osm.json')

    def test_span_ownership_is_exact(self):
        """The navigator's line grazes chunk corners by millimetres: every piece stays in its own chunk."""
        geo = adapter.build_geo(minimal_city(roads=[road('w1', [[-100, -99.75], [99.75, 99.5]])]))
        self.assertEqual(contract(geo), [])
        spans = geo['roads'][0]['spans']
        grid = chunks.Grid((-100, -100, 100, 100), 60)
        for s in spans:
            x0, y0, x1, y1 = grid.cell_in_world(*chunks.parse_chunk_id(s['chunk']))
            for x, y in chunks.cut_line(geo['roads'][0]['centerline'], s['s0_m'], s['s1_m']):
                self.assertTrue(x0 - 1e-9 <= x <= x1 + 1e-9 and y0 - 1e-9 <= y <= y1 + 1e-9, (s, x, y))
        # A span stretched over a border is caught by the contract check, whatever cut it.
        doc = json.loads(adapter.dumps(geo))
        a, b = doc['roads'][0]['spans'][:2]
        doc['roads'][0]['spans'][:2] = [{'chunk': a['chunk'], 's0_m': a['s0_m'], 's1_m': b['s1_m']}]
        self.assertTrue(any('leaves its chunk' in e for e in adapter.check_geo(doc)))

    def test_a_stretch_along_a_border_belongs_to_the_higher_chunk(self):
        """The navigator's road runs 10 m along x = 60: that stretch is c1_0's, wherever the span's midpoint lies."""
        geo = adapter.build_geo(minimal_city(roads=[road('w1', [[10, 10], [60, 10], [60, 20], [10, 20], [10, 50]])]))
        self.assertEqual(contract(geo), [])
        self.assertEqual([s['chunk'] for s in geo['roads'][0]['spans']], ['c0_0', 'c1_0', 'c0_0'])
        doc = json.loads(adapter.dumps(geo))
        r = doc['roads'][0]
        r['spans'], r['chunks'] = [{'chunk': 'c0_0', 's0_m': 0.0, 's1_m': r['length_m']}], ['c0_0']
        for c in doc['chunks']:
            c['roads'] = ['w1'] if c['id'] == 'c0_0' else []
        self.assertTrue(any('belongs to another chunk' in e for e in adapter.check_geo(doc)))

    def test_decimal_chunk_size_end_to_end(self):
        S = 30.1
        city = minimal_city(half=3 * S, buildings=[building('w1', -3 * S, 0, -3 * S + 10, 10)],
                            roads=[road('w2', [[-3 * S, 1], [-3 * S, 2]]), road('w3', [[S, -20], [S, 20]])])
        geo = adapter.build_geo(city, chunk_size=S)
        self.assertEqual(contract(geo), [])
        self.assertEqual(geo['metadata']['chunking']['count'], 36)
        r = {x['id']: x for x in geo['roads']}
        self.assertEqual([s['chunk'] for s in r['w2']['spans']], ['c-3_0'])
        self.assertEqual([s['chunk'] for s in r['w3']['spans']], ['c1_-1', 'c1_0'], 'a road on a border belongs to the higher chunk')
        # The contract check judges ownership from the published bounds_m, not from the grid code.
        doc = json.loads(adapter.dumps(geo))
        w3 = next(x for x in doc['roads'] if x['id'] == 'w3')
        for s in w3['spans']:
            s['chunk'] = s['chunk'].replace('c1_', 'c0_')
        w3['chunks'] = ['c0_-1', 'c0_0']
        for c in doc['chunks']:
            c['roads'] = sorted({*c['roads'], 'w3'} if c['id'] in w3['chunks'] else set(c['roads']) - {'w3'})
        self.assertTrue(any('belongs to another chunk' in e for e in adapter.check_geo(doc)))

    def test_the_terrain_covers_the_play_area(self):
        att = 'Höjddata: Markhöjdmodell © Lantmäteriet (CC BY 4.0)'
        for x0, step in ((1_000_000, 100), (-50, 50)):            # far away; too small for the ±100 m area
            with self.assertRaises(adapter.AdapterError, msg=(x0, step)):
                adapter.build_geo(minimal_city(terrain={'x0': x0, 'y0': x0, 'step': step, 'n': 3, 'd': [0.0] * 9, 'attribution': att}))
        good = adapter.build_geo(minimal_city(terrain={'x0': -100, 'y0': -100, 'step': 100, 'n': 3, 'd': [0.0] * 9, 'attribution': att}))
        self.assertEqual(contract(good), [])
        doc = json.loads(adapter.dumps(good))
        doc['terrain']['grid']['origin_m'] = [1_000_000.0, 1_000_000.0]
        self.assertTrue(any("doesn't cover" in e for e in adapter.check_geo(doc)))
        # n = 3.0 is a JSON Schema integer: the contract check takes it, it doesn't crash on it.
        doc = json.loads(adapter.dumps(good))
        doc['terrain']['grid']['n'] = 3.0
        self.assertEqual(schema.validate(doc, 'outbreak_geo.v1'), [])
        self.assertEqual(adapter.validate(doc), [])

    def test_overlapping_classes_are_resolved_and_checked(self):
        """Overlapping layers in city.json: the higher layer keeps the ground; water lies only under road classes."""
        sq = lambda x0, y0, x1, y1: {'outer': [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], 'holes': []}       # noqa: E731
        land = [sq(-100, -100, 100, 0)]                                       # the north half is water
        city = minimal_city(land=land, areas={'asphalt': [sq(-50, -20, 50, 10)], 'sidewalk': [sq(-50, 5, 50, 8)],
                                              'park': [sq(-90, -90, -60, 5)]})
        geo = adapter.build_geo(city)
        self.assertEqual(contract(geo), [])
        area = {c: sum(a['area_m2'] for a in geo['areas'] if a['class'] == c) for c in ('carriageway', 'sidewalk', 'park')}
        self.assertEqual(area['carriageway'], 100 * 30, 'the road keeps its ground, over the water too (a deck)')
        self.assertNotIn('sidewalk', {a['class'] for a in geo['areas']}, 'the sidewalk lay wholly under the road')
        self.assertAlmostEqual(area['park'], 30 * 90, places=6, msg='the park is cut to the land')
        doc = json.loads(adapter.dumps(geo))
        park = next(a for a in doc['areas'] if a['class'] == 'park')
        doc['areas'].append({**park, 'id': 'grass-overlap', 'class': 'grass'})
        for c in doc['chunks']:
            c['areas'] = sorted(c['areas'] + (['grass-overlap'] if c['id'] in park['chunks'] else []))
        self.assertTrue(any('overlap' in e for e in adapter.check_geo(doc)))

    def test_a_road_running_back_along_itself(self):
        geo = adapter.build_geo(minimal_city(roads=[road('w1', [[10, 10], [20, 10], [10, 10]])]))
        self.assertEqual(contract(geo), [])
        self.assertEqual(sum(s['s1_m'] - s['s0_m'] for s in geo['roads'][0]['spans']), 20.0)

    def test_order_of_areas_and_walls_changes_no_byte(self):
        a = {'outer': [[0, 0], [10, 0], [10, 10], [0, 10]], 'holes': []}
        b = {'outer': [[20, 0], [30, 0], [30, 10], [20, 10]], 'holes': []}
        wall = {'kind': 'city_wall', 'poly': [[-90, 60], [90, 60], [90, 63], [-90, 63]]}
        one = adapter.build_geo(minimal_city(areas={'park': [a, a, b]}, walls=[wall, wall, {'kind': 'hedge', 'poly': [[0, 0], [1, 0], [1, 1]]}]))
        two = adapter.build_geo(minimal_city(areas={'park': [b, a, a]}, walls=[{'kind': 'hedge', 'poly': [[0, 0], [1, 0], [1, 1]]}, wall, wall]))
        self.assertEqual(adapter.dumps(one), adapter.dumps(two))
        self.assertEqual(sum('more than once' in w for w in one['metadata']['warnings']), 2)

    def test_spans_keep_the_chunk_contract(self):
        rng = random.Random(7)
        lines = [[[-100, -99.98], [99.99, 99.97]],                 # grazes chunk corners by centimetres
                 [[-100, -100], [100, 100]], [[60, -100], [60, 100]], [[-120, 0], [-60, 0], [-60, 60], [120, 60]],
                 [[0.0004, -50], [-0.0004, 50]], [[-100, 0], [99.9999, 0]]]
        for _ in range(150):
            grid_pt = lambda: rng.choice([rng.uniform(-130, 130), rng.randrange(-120, 121, 60) + rng.choice([0, 1e-4, -1e-4])])  # noqa: E731
            lines.append([[grid_pt(), grid_pt()] for _ in range(rng.randint(2, 5))])
        geo = adapter.build_geo(minimal_city(roads=[road(f'w{k}', line) for k, line in enumerate(lines, 1)]))
        self.assertGreater(len(geo['roads']), 100)
        self.assertEqual(contract(geo), [])
        for r in geo['roads']:
            self.assertTrue(all(s['s0_m'] < s['s1_m'] for s in r['spans']), r['id'])

    def test_contract_check_catches_broken_documents(self):
        terrain = {'x0': -100, 'y0': -100, 'step': 100, 'n': 3, 'd': [float(v) for v in range(9)], 'lake': 1.2,
                   'attribution': 'Höjddata: Markhöjdmodell © Lantmäteriet (CC BY 4.0)'}
        city = minimal_city(buildings=[building('w1', -10, 0, 10, 10)], terrain=terrain,
                            roads=[road('w2', [[-50, 5], [0, 5], [50, 5]]), road('w3', [[0, 5], [0, -50]])])
        good = json.loads(adapter.dumps(adapter.build_geo(city, osm=osm_of(w1={'building': 'yes'}, w2={'highway': 'residential'},
                                                                               w3={'highway': 'residential'}))))
        self.assertEqual(adapter.check_geo(good), [])
        self.assertEqual([j['basis'] for j in good['navigation']['junctions']], ['shared_position'])

        def no_collision(g):
            g['navigation']['collision'] = []
            for c in g['chunks']:
                c['collision'] = []

        def owner_moved(g):                       # to another chunk it covers, with the index kept consistent
            b = g['buildings'][0]
            old, new = b['chunk'], next(c for c in b['chunks'] if c != b['chunk'])
            b['chunk'] = new
            for c in g['chunks']:
                c['buildings'] = ['w1'] if c['id'] == new else [x for x in c['buildings'] if c['id'] != old]

        breakages = {
            'no building collision': no_collision,
            'owner in the wrong chunk': owner_moved,
            'junctions deleted': lambda g: g['navigation']['junctions'].clear(),
            'terrain raster cut short': lambda g: g['terrain'].update(heights_m=g['terrain']['heights_m'][:4]),
            'topology overstated': lambda g: g['navigation'].update(topology='osm_nodes'),
            'grid origin moved': lambda g: g['metadata']['chunking'].update(origin_m=[60.0, 0.0]),
            'chunk count': lambda g: g['metadata']['chunking'].update(count=99),
            'bounds not centred': lambda g: g['metadata']['bounds'].update(min_m=[-90.0, -100.0]),
            'wgs84 box off': lambda g: g['metadata']['bounds']['wgs84'].update(north=g['metadata']['bounds']['wgs84']['north'] + 0.001),
            'unknown chunk': lambda g: g['buildings'][0]['chunks'].append('c9_9'),
            'owner outside': lambda g: g['buildings'][0].update(chunk='c1_1'),
            'walls start above the top': lambda g: g['buildings'][0].update(min_height_m=7.0),
            'empty span': lambda g: g['roads'][0]['spans'][0].update(s1_m=g['roads'][0]['spans'][0]['s0_m']),
            'span in the wrong chunk': lambda g: g['roads'][0]['spans'][0].update(chunk='c1_1'),
            'invalid polygon': lambda g: g['buildings'][0]['footprint'].update(outer=[[0, 0], [10, 10], [10, 0], [0, 10]]),
            'collision without its building': lambda g: g['navigation']['collision'][0].update(buildings=['w99']),
            'chunk index': lambda g: g['chunks'][0]['buildings'].append('w1'),
            'drivable list': lambda g: g['navigation']['drivable_roads'].clear(),
            'bounds_m': lambda g: g['chunks'][0].update(bounds_m=[-120.0, -120.0, -60.0, -60.0]),
        }
        for what, breakage in breakages.items():
            doc = copy.deepcopy(good)
            breakage(doc)
            self.assertTrue(adapter.check_geo(doc), what)

    def test_invalid_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(adapter.AdapterError):
                adapter.convert_city_dir(Path(tmp))                               # no city.json
            (Path(tmp) / 'city.json').write_text('{not json', encoding='utf-8')
            with self.assertRaises(adapter.AdapterError):
                adapter.convert_city_dir(Path(tmp))

    def test_contract_examples(self):
        for name, file in (('outbreak_overrides.v1', 'outbreak_overrides.example.json'), ('gameplay_layer.v1', 'gameplay_layer.example.json'),
                           ('asset_manifest.v1', 'asset_manifest.example.json')):
            doc = json.loads((EXAMPLES / file).read_text(encoding='utf-8'))
            self.assertEqual(schema.validate(doc, name), [], file)

        def overrides(**change):
            """The example with w11 changed; None removes a key."""
            ov = json.loads((EXAMPLES / 'outbreak_overrides.example.json').read_text(encoding='utf-8'))
            entry = ov['buildings']['w11']
            for k, v in change.items():
                entry.pop(k) if v is None else entry.__setitem__(k, v)
            return ov

        self.assertTrue(schema.validate(overrides(basis='street_view'), 'outbreak_overrides.v1'), 'Street View is not an allowed basis')
        self.assertTrue(schema.validate(overrides(basis=None), 'outbreak_overrides.v1'), 'every correction states its basis')
        self.assertTrue(schema.validate(overrides(source=None), 'outbreak_overrides.v1'), 'and its source')
        self.assertTrue(schema.validate(overrides(source={'reference': ' ', 'by': 'x', 'date': '2026-10-05'}), 'outbreak_overrides.v1'))
        open_data = {'basis': 'open_data', 'source': {'reference': 'Stadsarkivet byggnadsregister v3', 'by': 'x', 'date': '2026-10-05'}}
        self.assertTrue(schema.validate(overrides(**open_data), 'outbreak_overrides.v1'), 'open data states its licence')
        open_data['source']['license'] = 'CC0-1.0'
        self.assertEqual(schema.validate(overrides(**open_data), 'outbreak_overrides.v1'), [])
        self.assertTrue(schema.validate(overrides(zombie_spawn=True), 'outbreak_overrides.v1'), 'no gameplay in the overrides')

        def assets(**change):
            doc = json.loads((EXAMPLES / 'asset_manifest.example.json').read_text(encoding='utf-8'))
            doc['assets'][0].update(change)
            return doc

        no_licence = assets()
        del no_licence['assets'][0]['license']
        self.assertTrue(schema.validate(no_licence, 'asset_manifest.v1'), 'every asset states its licence')
        self.assertTrue(schema.validate(assets(license={'terms': '', 'author': '', 'source': ''}), 'asset_manifest.v1'), 'not empty')
        approved = {'terms': 'own work, all rights Outbreak', 'author': 'Outbreak team', 'source': 'Studio file plaster-3bay.rbxm'}
        self.assertTrue(schema.validate(assets(status='approved', license=approved), 'asset_manifest.v1'), 'approved needs evidence')
        self.assertEqual(schema.validate(assets(status='approved', license={**approved, 'evidence': 'commit abc123'}), 'asset_manifest.v1'), [])
        gameplay = json.loads((EXAMPLES / 'gameplay_layer.example.json').read_text(encoding='utf-8'))
        gameplay['features'][0]['type'] = 'teleporter'
        self.assertTrue(schema.validate(gameplay, 'gameplay_layer.v1'))

    def test_schema_checker(self):
        self.assertEqual(schema.errors(1, {'type': 'integer'}, {}), [])
        self.assertTrue(schema.errors(True, {'type': 'integer'}, {}))
        self.assertTrue(schema.errors(True, {'enum': [1]}, {}))
        self.assertTrue(schema.errors(float('nan'), {'type': 'number'}, {}))
        with self.assertRaises(schema.SchemaError):
            schema.errors(1, {'type': 'number', 'multipleOf': 2}, {})
        for path in (ROOT / 'pipeline/outbreak/schemas').glob('*.schema.json'):
            doc = json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual(doc['$id'], path.name)
            self.assertIn('schema_version', doc['required'])


def defaultdict_list(pairs):
    out = {}
    for k, v in pairs:
        out.setdefault(k, []).append(v)
    return out


if __name__ == '__main__':
    unittest.main()
