"""P01 tests of the Outbreak real-city pipeline: offline (no network, Blender, Node or Roblox Studio).

    .venv/bin/python -m unittest tests.test_outbreak_p01

- the WGS84 plane Outbreak sources use (common.Projection 'wgs84'), against an independent geodesic
  (Vincenty), and Wasteland's default plane unchanged;
- OSM node ids kept by Wasteland's download (fetch_osm.py) and carried, aligned point for point, into
  OutbreakGeo roads; verified junctions only at shared nodes; crossings that are separated (bridge,
  tunnel, layer, level) or unresolved; the fallback without node ids; the contract catching fabricated
  topology;
- the terrain download recording where every height came from;
- retained snapshots: the real one verifies, tampered or malformed ones are refused;
- the real Borås dataset (pipeline/outbreak/snapshots/boras-p01), built twice from the snapshot, once in
  this process and once in its own process with another hash seed, byte-identical, and checked by
  pipeline/outbreak/realcheck.py against the raw snapshot: chunks, topology, buildings, 20+ scale
  measurements, completeness, terrain, provenance. The two builds take a minute and a half together.
"""
import contextlib
import copy
import gzip
import io
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'pipeline'))
sys.path.insert(0, str(ROOT / 'tests'))

import common  # noqa: E402
import fetch_osm  # noqa: E402
import fetch_terrain  # noqa: E402
import numpy as np  # noqa: E402
import outbreak_fixture as fx  # noqa: E402
import prepare_city  # noqa: E402
from outbreak import adapter, realcheck, snapshot  # noqa: E402
from shapely.geometry import LineString, Polygon  # noqa: E402

from test_outbreak import GEN, MESH_KEYS, contract, keys_in, minimal_city, prepared, road  # noqa: E402

SNAPSHOT = ROOT / 'pipeline/outbreak/snapshots/boras-p01'
PLANE = common.Projection(60.0, 15.0, 'wgs84')          # minimal_city's centre


def topo(ways):
    """A minimal city and its osm.json from {way id: (local points, tags, node ids or None)}: the geometry is the
    points on the WGS84 plane, so node ids line up with the centrelines as they do after prepare_city."""
    city = minimal_city(roads=[road(f'w{i}', pts, tags.get('highway', 'residential')) for i, (pts, tags, _) in ways.items()])
    elements = []
    for i, (pts, tags, nodes) in ways.items():
        el = {'type': 'way', 'id': i, 'tags': tags, 'geometry': [dict(zip(('lat', 'lon'), PLANE.latlon(*p))) for p in pts]}
        if nodes is not None:
            el['nodes'] = nodes
        elements.append(el)
    return city, {'elements': elements}


# ------------------------------------------------------------------------------------------ the plane
class ProjectionTest(unittest.TestCase):
    def test_independent_geodesic_reference(self):
        """Vincenty's own worked example (Flinders Peak to Buninyong): the reference the scale checks use."""
        s, az = realcheck.geodesic(-(37 + 57 / 60 + 3.72030 / 3600), 144 + 25 / 60 + 29.52440 / 3600,
                                   -(37 + 39 / 60 + 10.15610 / 3600), 143 + 55 / 60 + 35.38390 / 3600)
        self.assertAlmostEqual(s, 54972.271, delta=0.001)
        self.assertAlmostEqual(math.degrees(az) % 360, 306 + 52 / 60 + 5.37 / 3600, delta=1e-5)

    def test_wgs84_plane_lengths_are_true(self):
        for lat in (0.0, 30.0, 45.0, 57.72, 65.0):
            p = common.Projection(lat, 12.94, 'wgs84')
            for dx, dy in ((500, 0), (0, 500), (-353.6, 353.6), (-500, -500), (500, 300)):
                la, lo = p.latlon(dx, dy)
                true = realcheck.geodesic(lat, 12.94, la, lo)[0]
                self.assertLess(abs(math.hypot(dx, dy) - true) / true, 1.5e-4, (lat, dx, dy))
                x, y = p.xy(la, lo)
                self.assertAlmostEqual(x, dx, delta=1e-6)
                self.assertAlmostEqual(y, dy, delta=1e-6)

    def test_wasteland_default_plane_is_unchanged(self):
        """Every Wasteland city so far: the same floats as the formula before P01."""
        for lat0, lon0 in ((57.7210839, 12.9390626), (55.43, 13.82), (-33.9, 151.2)):
            p = common.Projection(lat0, lon0)
            kx = common.EARTH * math.cos(math.radians(lat0))
            for lat, lon in ((lat0 + 0.004, lon0 - 0.007), (lat0 - 0.0031, lon0 + 0.0123)):
                self.assertEqual(p.xy(lat, lon), ((lon - lon0) * kx, (lat - lat0) * common.EARTH))
            self.assertEqual(p.latlon(123.4, -56.7), (lat0 + -56.7 / common.EARTH, lon0 + 123.4 / kx))
        self.assertEqual(common.projection_for({'center': [57.7, 12.9]}).model, 'sphere')

    def test_the_spherical_plane_is_refused(self):
        sphere, true = common.Projection(57.72, 12.94), common.Projection(57.72, 12.94, 'wgs84')
        self.assertAlmostEqual(sphere.kx / true.kx - 1, -0.0024, delta=0.0002, msg='0.24 % short east-west in Borås')
        city = minimal_city()
        del city['place']['projection']
        with self.assertRaises(adapter.PolicyError) as cm:
            adapter.build_geo(city)
        self.assertIn('-0.2', str(cm.exception))
        with self.assertRaises(adapter.AdapterError):
            adapter.build_geo({**city, 'place': {**city['place'], 'projection': 'utm'}})
        with self.assertRaises(ValueError):
            common.Projection(57.72, 12.94, 'utm')
        # A terrain grid sampled on another plane would put the heights in the wrong place; a flat city ignores it.
        hilly = minimal_city(terrain={'x0': -100, 'y0': -100, 'step': 100, 'n': 3, 'd': [0.0] * 9,
                                      'attribution': 'Terrain: Copernicus DEM GLO-30 © DLR e.V.'})
        with self.assertRaises(adapter.PolicyError):
            adapter.build_geo(hilly, terrain_src={**fx.terrain(), 'projection': 'sphere'})
        with self.assertRaises(adapter.PolicyError):
            adapter.build_geo(hilly, terrain_src={k: v for k, v in fx.terrain().items() if k != 'projection'})
        self.assertEqual(adapter.validate(adapter.build_geo(hilly, terrain_src=fx.terrain())), [])
        self.assertEqual(adapter.validate(adapter.build_geo(minimal_city(), terrain_src={**fx.terrain(), 'projection': 'sphere'})), [])


# ------------------------------------------------------------------------------------------ downloads
def map_response(points, ways, tagged_nodes=()):
    """An OSM API /map answer: nodes at local points (WGS84 plane round fx.CENTER), ways with node refs."""
    els = [{'type': 'node', 'id': n, 'lat': fx.P.latlon(*p)[0], 'lon': fx.P.latlon(*p)[1], **({'tags': t} if t else {})}
           for n, p, t in [(n, p, dict(tagged_nodes).get(n)) for n, p in points.items()]]
    els += [{'type': 'way', 'id': i, 'nodes': refs, 'tags': tags} for i, (refs, tags) in ways.items()]
    return {'elements': els}


class FetchTest(unittest.TestCase):
    def test_map_ways_keep_their_ordered_node_ids(self):
        raw = map_response({1: (0, 0), 2: (10, 0), 3: (20, 0), 4: (10, 10)},
                           {10: ([3, 1, 2], {'highway': 'residential'}), 11: ([2, 99, 4], {'highway': 'footway'}), 12: ([1, 4], {})})
        record = {'parts': [], 'incomplete_ways': []}
        with mock.patch.object(fetch_osm, 'http_json', return_value=raw), contextlib.redirect_stdout(io.StringIO()):
            els = fetch_osm.osm_api_elements('59.99,14.99,60.01,15.01', record=record)
        out = {el['id']: el for el in els if el['type'] == 'way'}
        self.assertEqual(set(out), {10, 11}, 'untagged ways are not features')
        self.assertEqual(out[10]['nodes'], [3, 1, 2], 'way order kept')
        coords = {n: fx.P.latlon(*p) for n, p in {1: (0, 0), 2: (10, 0), 3: (20, 0), 4: (10, 10)}.items()}
        self.assertEqual([(g['lat'], g['lon']) for g in out[10]['geometry']], [coords[3], coords[1], coords[2]])
        # A node the answer lacks: the way keeps all its node ids, the geometry is a point short, so the gap shows.
        self.assertEqual((out[11]['nodes'], len(out[11]['geometry'])), ([2, 99, 4], 2))
        self.assertEqual(record['incomplete_ways'], [11])
        self.assertEqual(snapshot.incomplete_ways({'elements': els}), ['way 11'])
        city, osm = topo({11: ([(10, 0), (5, 5), (10, 10)], {'highway': 'footway'}, None)})
        osm['elements'][0].update(nodes=[2, 99, 4], geometry=out[11]['geometry'])
        self.assertIsNone(adapter.build_geo(city, osm=osm)['roads'][0]['nodes'], 'never a shortcut with ids')

    def _fetch(self, raw, date, overpass=None):
        """fetch_osm.main on the fixture's place with the OSM API's answer (and its Date header) mocked."""
        def http_json(url, data=None, timeout=180, attempts=3, info=None):
            if 'overpass' in url:
                if overpass is None:
                    raise RuntimeError('busy')
                return overpass
            if info is not None:
                info['date'] = date
            return raw
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / 'town'
            fx.write(folder)
            (folder / 'osm.json').unlink()
            old = common.CITIES
            common.CITIES = Path(tmp)
            try:
                with mock.patch.object(fetch_osm, 'http_json', side_effect=http_json), contextlib.redirect_stdout(io.StringIO()):
                    fetch_osm.main(['town'])
            finally:
                common.CITIES = old
            return json.loads((folder / 'osm.json').read_text(encoding='utf-8'))

    def test_download_and_source_times_are_recorded_apart(self):
        raw = map_response({1: (0, 0), 2: (10, 0)}, {10: ([1, 2], {'highway': 'residential'})})
        before = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        osm = self._fetch(raw, 'Wed, 07 Oct 2026 10:02:05 GMT')
        after = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        f = osm['fetch']
        self.assertTrue(before <= f['started_utc'] <= f['finished_utc'] <= after, 'this machine\'s clock, kept apart')
        self.assertEqual([(p['source'], p['server_time_utc']) for p in f['parts']], [('OpenStreetMap API 0.6 map', '2026-10-07T10:02:05Z')])
        self.assertEqual(osm['osm3s']['timestamp_osm_base'], '2026-10-07T10:02:05Z', 'how current the data is: the source says')
        self.assertEqual((f['incomplete_ways'], f['incomplete_relations']), ([], []))
        # A source that states no time: the data time is unknown, said so, not filled in with the download time.
        self.assertIsNone(self._fetch(raw, None)['osm3s']['timestamp_osm_base'])

    def test_an_overpass_completion_keeps_its_own_time(self):
        """A multipolygon reaching outside the box comes from an Overpass mirror, whose data can lag the API's:
        the oldest stated time is the data's; an area Overpass can't complete is recorded."""
        raw = map_response({1: (0, 0), 2: (10, 0), 3: (10, 10)}, {10: ([1, 2, 3, 1], {'landuse': 'grass'})})
        raw['elements'].append({'type': 'relation', 'id': 70, 'tags': {'type': 'multipolygon', 'natural': 'water'},
                                'members': [{'type': 'way', 'ref': 10, 'role': 'outer'}, {'type': 'way', 'ref': 999, 'role': 'outer'}]})
        whole = {'type': 'relation', 'id': 70, 'tags': {'type': 'multipolygon', 'natural': 'water'},
                 'members': [{'type': 'way', 'ref': 10, 'role': 'outer', 'geometry': [{'lat': 60, 'lon': 15}] * 4},
                             {'type': 'way', 'ref': 999, 'role': 'outer', 'geometry': [{'lat': 60, 'lon': 15}] * 4}]}
        osm = self._fetch(raw, 'Wed, 07 Oct 2026 10:02:05 GMT', {'osm3s': {'timestamp_osm_base': '2026-10-07T09:58:00Z'}, 'elements': [whole]})
        self.assertEqual([p.get('server_time_utc') or p.get('timestamp_osm_base') for p in osm['fetch']['parts']],
                         ['2026-10-07T10:02:05Z', '2026-10-07T09:58:00Z'])
        self.assertEqual(osm['osm3s']['timestamp_osm_base'], '2026-10-07T09:58:00Z')
        busy = self._fetch(raw, 'Wed, 07 Oct 2026 10:02:05 GMT')
        self.assertEqual(busy['fetch']['incomplete_relations'], [70])
        self.assertEqual(snapshot.incomplete_relations(busy), ['relation 70'])

    def test_terrain_records_where_each_height_came_from(self):
        """A ground-model grid that the fallback had to patch, and cells no source covered, are counted, also
        inside the play area: the fallback stays explicit."""
        place = {'center': [57.72, 12.94], 'size_m': 200, 'country_code': 'se', 'projection': 'wgs84'}

        def lm(lat, lon, auth, pixel):
            z = np.full(lat.shape, 150.0, np.float32)
            z[:3, :] = np.nan                                     # Lantmäteriet misses the southern rows
            return z, ['item-1']

        def cop(lat, lon):
            z = np.full(lat.shape, 151.0, np.float32)
            z[:4] = np.nan                                        # and Copernicus a few of those
            return z, ['Copernicus_DSM_COG_10_N57_00_E012_00_DEM']
        with mock.patch.object(fetch_terrain, 'lantmateriet_auth', return_value='Basic x'), \
                mock.patch.object(fetch_terrain, 'lantmateriet', side_effect=lm), \
                mock.patch.object(fetch_terrain, 'copernicus', side_effect=cop), contextlib.redirect_stdout(io.StringIO()):
            t = fetch_terrain.fetch(place)
        cov = t['coverage']
        self.assertEqual((t['kind'], t['projection']), ('dtm', 'wgs84'))
        self.assertEqual(cov['cells'], t['n'] ** 2)
        self.assertEqual(cov['no_data_cells'], 4)
        self.assertEqual(cov['fallback_cells'], 3 * t['n'] - 4)
        self.assertEqual((cov['play_area_fallback_cells'], cov['play_area_no_data_cells']), (0, 0), 'the margin, not the play area')
        self.assertIn('Lantmäteriet item-1', t['tiles'])
        self.assertTrue(all(math.isfinite(v) for v in t['z']))

    def test_a_hole_next_to_the_play_area_is_counted(self):
        """The navigator's case on P01's 30 m grid: no height at (520, 10), 20 m outside a 1000 m square. Filled
        with the lowest height, it would pull down what prepare_city interpolates at (499, 10), so it counts as a
        hole of the support (every cell prepare_city reads), though not of the play area; one beyond that doesn't."""
        place = {'center': [57.72, 12.94], 'size_m': 1000, 'country_code': 'se', 'projection': 'wgs84'}
        proj = common.Projection(57.72, 12.94, 'wgs84')

        def cop(holes_at):
            def run(lat, lon):
                z = np.full(lat.shape, 100.0, np.float32)
                for x, y in holes_at:
                    la, lo = proj.latlon(x, y)
                    z[np.argmin((lat - la) ** 2 + (lon - lo) ** 2)] = np.nan
                return z, ['Copernicus_DSM_COG_10_N57_00_E012_00_DEM']
            return run
        self.assertEqual(fetch_terrain.SUPPORT_M, prepare_city.TERRAIN_MARGIN)
        for holes, support in ((((520, 10),), 1), (((1090, 10),), 0)):
            with mock.patch.object(fetch_terrain, 'lantmateriet_auth', return_value=None), \
                    mock.patch.object(fetch_terrain, 'copernicus', side_effect=cop(holes)), contextlib.redirect_stdout(io.StringIO()):
                cov = fetch_terrain.fetch(place)['coverage']
            self.assertEqual((cov['no_data_cells'], cov['play_area_no_data_cells'], cov['support_no_data_cells']), (1, 0, support), holes)


# ------------------------------------------------------------------------------------------ topology
class TopologyTest(unittest.TestCase):
    def test_a_shared_node_is_a_verified_junction(self):
        city, osm = topo({1: ([(-50, 0), (0, 0), (50, 0)], {'highway': 'residential'}, [11, 12, 13]),
                          2: ([(0, -50), (0, 0)], {'highway': 'service'}, [21, 12])})
        geo = adapter.build_geo(city, osm=osm)
        self.assertEqual(contract(geo), [])
        self.assertEqual(geo['navigation']['junctions'], [{'id': 'j:n12', 'point': [0.0, 0.0], 'roads': ['w1', 'w2'], 'node': 12, 'basis': 'osm_node'}])
        self.assertEqual(geo['navigation']['crossings'], [])
        self.assertEqual({r['id']: r['nodes'] for r in geo['roads']}, {'w1': [11, 12, 13], 'w2': [21, 12]})

    def test_a_crossing_without_a_shared_node_is_not_a_junction(self):
        """Two streets drawn across each other on one level with no common node: OSM says they don't meet. Not
        a junction, but an unresolved crossing, reported."""
        city, osm = topo({1: ([(-50, 0), (50, 0)], {'highway': 'residential'}, [11, 13]),
                          2: ([(0, -50), (0, 50)], {'highway': 'residential'}, [21, 23])})
        geo = adapter.build_geo(city, osm=osm)
        self.assertEqual(contract(geo), [])
        self.assertEqual(geo['navigation']['junctions'], [])
        self.assertEqual(geo['navigation']['crossings'],
                         [{'id': 'x:w1:w2:0.000,0.000', 'point': [0.0, 0.0], 'roads': ['w1', 'w2'], 'status': 'unresolved', 'differ': []}])
        self.assertTrue(any('unresolved' in w for w in geo['metadata']['warnings']))

    def test_bridges_tunnels_layers_and_levels_separate(self):
        street = ([(-90, 0), (90, 0)], {'highway': 'residential'}, [1, 2])
        cases = {
            2: ({'highway': 'primary', 'bridge': 'yes'}, 'separated', ['bridge']),                  # a bridge is layer 1 by default
            3: ({'highway': 'primary', 'layer': '1'}, 'separated', ['layer']),
            4: ({'highway': 'primary', 'tunnel': 'yes'}, 'separated', ['tunnel']),                  # a tunnel is layer -1
            5: ({'highway': 'footway', 'bridge': 'yes', 'layer': '0'}, 'unresolved', ['bridge', 'layer']),   # an explicit layer wins
            6: ({'highway': 'footway', 'covered': 'yes'}, 'unresolved', ['covered']),               # a roof doesn't lift it
            7: ({'highway': 'footway', 'tunnel': 'building_passage'}, 'unresolved', ['tunnel']),   # a passage runs at the ground
            8: ({'highway': 'footway', 'level': '1'}, 'separated', ['level']),
            9: ({'highway': 'footway', 'layer': 'x'}, 'unresolved', ['layer']),                     # unreadable: can't tell
        }
        ways = {1: street, **{i: ([(-100 + 20 * i, -50), (-100 + 20 * i, 50)], tags, [i * 10, i * 10 + 1]) for i, (tags, _, _) in cases.items()}}
        city, osm = topo(ways)
        geo = adapter.build_geo(city, osm=osm)
        self.assertEqual(contract(geo), [])
        self.assertEqual(geo['navigation']['junctions'], [])
        got = {c['roads'][1]: (c['status'], c['differ']) for c in geo['navigation']['crossings']}
        self.assertEqual(got, {f'w{i}': (status, differ) for i, (_, status, differ) in cases.items()})

    def test_the_fallback_without_node_ids(self):
        """Without node ids a shared position is only possibly a junction (shared_position, never osm_node), and
        only among roads on one level."""
        ways = {1: ([(-50, 0), (0, 0), (50, 0)], {'highway': 'residential'}, None),
                2: ([(0, -50), (0, 0)], {'highway': 'service'}, None),
                3: ([(0, 50), (0, 0), (20, -40)], {'highway': 'primary', 'bridge': 'yes', 'layer': '1'}, None)}
        city, osm = topo(ways)
        geo = adapter.build_geo(city, osm=osm)
        self.assertEqual(contract(geo), [])
        self.assertEqual(geo['navigation']['topology'], 'positions')
        self.assertEqual([(j['basis'], j['node'], j['roads']) for j in geo['navigation']['junctions']], [('shared_position', None, ['w1', 'w2'])])
        self.assertEqual({tuple(c['roads']): c['status'] for c in geo['navigation']['crossings']},
                         {('w1', 'w3'): 'separated', ('w2', 'w3'): 'separated'})
        # Some roads with ids, some without: the shared node is verified only between the roads that have it.
        ways[2] = (ways[2][0], ways[2][1], [21, 12])
        ways[1] = (ways[1][0], ways[1][1], [11, 12, 13])
        city, osm = topo(ways)
        geo = adapter.build_geo(city, osm=osm)
        self.assertEqual(contract(geo), [])
        self.assertEqual(geo['navigation']['topology'], 'mixed')
        self.assertEqual([(j['basis'], j['roads']) for j in geo['navigation']['junctions']], [('osm_node', ['w1', 'w2'])])

    def test_node_ids_that_dont_line_up_are_not_used(self):
        """Ids are attached only where the way's geometry is exactly the centreline: a shifted copy, a missing
        point or ids for another way never land on a road."""
        ok = ([(-50, 0), (0, 0), (50, 0)], {'highway': 'residential'}, [11, 12, 13])
        for bad in ([(-50, 1), (0, 1), (50, 1)], [(-50, 0), (50, 0)]):
            city, osm = topo({1: ok, 2: ([(0, -50), (0, 0)], {'highway': 'service'}, [21, 12])})
            osm['elements'][0]['geometry'] = [dict(zip(('lat', 'lon'), PLANE.latlon(*p))) for p in bad]
            osm['elements'][0]['nodes'] = [11, 12, 13][:len(bad)]
            geo = adapter.build_geo(city, osm=osm)
            self.assertEqual(contract(geo), [])
            r = {x['id']: x for x in geo['roads']}
            self.assertIsNone(r['w1']['nodes'])
            self.assertEqual(geo['navigation']['topology'], 'mixed')
            self.assertTrue(any("don't line up" in w and 'w1' in w for w in geo['metadata']['warnings']))
            self.assertNotIn('osm_node', {j['basis'] for j in geo['navigation']['junctions']})

    def test_topology_is_independent_of_input_order(self):
        rng = random.Random(7)
        ways = {}
        for i in range(1, 13):
            a, b = (rng.uniform(-90, 90), rng.uniform(-90, 90)), (rng.uniform(-90, 90), rng.uniform(-90, 90))
            ways[i] = ([a, (0, 0), b] if i % 3 == 0 else [a, b], {'highway': rng.choice(['residential', 'footway']), 'layer': rng.choice(['0', '1'])},
                       [i * 10, 1, i * 10 + 1] if i % 3 == 0 else [i * 10, i * 10 + 1])
        city, osm = topo(ways)
        one = adapter.build_geo(city, osm=osm)
        self.assertEqual(contract(one), [])
        self.assertTrue(one['navigation']['crossings'] and one['navigation']['junctions'])
        for seed in range(3):
            c2, o2 = copy.deepcopy(city), copy.deepcopy(osm)
            random.Random(seed).shuffle(c2['roads'])
            random.Random(seed).shuffle(o2['elements'])
            self.assertEqual(adapter.dumps(adapter.build_geo(c2, osm=o2)), adapter.dumps(one))

    def test_the_contract_catches_fabricated_topology(self):
        city, osm = topo({1: ([(-50, 0), (0, 0), (50, 0)], {'highway': 'residential'}, [11, 12, 13]),
                          2: ([(0, -50), (0, 0)], {'highway': 'service'}, [21, 12]),
                          3: ([(30, -50), (30, 50)], {'highway': 'primary', 'bridge': 'yes', 'layer': '1'}, [31, 32])})
        good = json.loads(adapter.dumps(adapter.build_geo(city, osm=osm)))
        self.assertEqual(adapter.check_geo(good), [])
        tamper = {
            'a junction where no node is shared': lambda g: g['navigation']['junctions'].append(
                {'id': 'j:n32', 'point': [30.0, 50.0], 'roads': ['w1', 'w3'], 'node': 32, 'basis': 'osm_node'}),
            'the bridge joined to the street': lambda g: g['navigation']['junctions'][0]['roads'].append('w3'),
            'one node at two places': lambda g: g['roads'][2]['nodes'].__setitem__(0, 12),
            'a separated crossing called unresolved': lambda g: g['navigation']['crossings'][0].update(status='unresolved'),
            'a crossing left out': lambda g: g['navigation']['crossings'].clear(),
            'a junction downgraded to a guess': lambda g: g['navigation']['junctions'][0].update(basis='shared_position', node=None),
        }
        for what, change in tamper.items():
            doc = copy.deepcopy(good)
            change(doc)
            self.assertTrue(adapter.check_geo(doc), what)

    def test_from_the_download_to_outbreakgeo(self):
        """Node ids from the OSM API's answer through fetch_osm.py, osm.json, prepare_city's city.json and the
        adapter: a junction where two streets share a node, a bridge crossing a street without one, a footway
        crossing a street on its level without one."""
        points = {1: (-120, -20), 2: (0, -20), 3: (120, -20), 4: (0, -100), 5: (0, 100), 6: (60, -100), 7: (60, 100),
                  8: (-80, -60), 9: (-80, 60)}
        ways = {501: ([1, 2, 3], {'highway': 'residential', 'name': 'Ågatan'}), 502: ([4, 2, 5], {'highway': 'tertiary'}),
                503: ([6, 7], {'highway': 'primary', 'bridge': 'yes', 'layer': '1'}), 504: ([8, 9], {'highway': 'footway'})}
        with mock.patch.object(fetch_osm, 'http_json', return_value=map_response(points, ways)), contextlib.redirect_stdout(io.StringIO()):
            elements = fetch_osm.osm_api_elements('59.99,14.99,60.01,15.01')
        osm = {'version': 0.6, 'generator': 'wasteland-builder (test)', 'osm3s': {'timestamp_osm_base': '2026-10-07T00:00:00Z'},
               'elements': fx.elements() + elements}
        with tempfile.TemporaryDirectory() as tmp:
            folder = prepared(Path(tmp) / 'city', osm=osm)
            geo = adapter.convert_city_dir(folder, generator=GEN, synthetic=True)
        self.assertEqual(contract(geo), [])
        r = {x['id']: x for x in geo['roads']}
        self.assertEqual({k: r[k]['nodes'] for k in ('w501', 'w502', 'w503', 'w504')},
                         {'w501': [1, 2, 3], 'w502': [4, 2, 5], 'w503': [6, 7], 'w504': [8, 9]})
        j = {x['id']: x for x in geo['navigation']['junctions']}
        self.assertEqual((j['j:n2']['roads'], j['j:n2']['basis']), (['w501', 'w502'], 'osm_node'))
        x = {tuple(c['roads']): (c['status'], c['differ']) for c in geo['navigation']['crossings']}
        self.assertEqual(x[('w501', 'w503')], ('separated', ['bridge', 'layer']))
        self.assertEqual(x[('w501', 'w504')], ('unresolved', []))
        self.assertEqual(r['w503']['tags'], {'bridge': 'yes', 'layer': '1'})


# ------------------------------------------------------------------------------------------ the ground gate
class GroundGateTest(unittest.TestCase):
    def test_a_narrow_path_lost_entirely_is_caught(self):
        """The navigator's case: a 200 m × 1.4 m path through prepare_city and the adapter. Removed whole, it still
        passes the contract, and eroding the leftover by the simplification tolerance (0.8 m) would make it vanish;
        but no kept edge lies near it, and the rebuilt layer piece is gone: both rules catch it."""
        path = {**fx.way(900, [(-140, 130), (60, 130)], {'highway': 'path', 'width': '1.4', 'surface': 'dirt'}), 'nodes': [9001, 9002]}
        osm = {'version': 0.6, 'generator': 'outbreak test fixture', 'osm3s': {'timestamp_osm_base': None}, 'elements': fx.elements() + [path]}
        with tempfile.TemporaryDirectory() as tmp:
            folder = prepared(Path(tmp) / 'city', osm=osm)
            geo = json.loads(adapter.dumps(adapter.convert_city_dir(folder, generator=GEN, synthetic=True)))
        reader = prepare_city.OSM(osm, fx.P)
        world = Polygon([(-150, -150), (150, -150), (150, 150), (-150, 150)])
        report = realcheck.area_report(geo, reader, world)
        self.assertTrue(report['pass'], report)
        kept = [a for a in geo['areas'] if a['class'] == 'path']
        self.assertGreater(sum(a['area_m2'] for a in kept), 240, 'the path is there (less where the street crosses it)')
        gone = copy.deepcopy(geo)
        ids = {a['id'] for a in kept}
        gone['areas'] = [a for a in gone['areas'] if a['id'] not in ids]
        for c in gone['chunks']:
            c['areas'] = [x for x in c['areas'] if x not in ids]
        self.assertEqual(adapter.check_geo(gone), [], 'the contract alone can\'t tell')
        report = realcheck.area_report(gone, reader, world)
        self.assertFalse(report['pass'])
        self.assertTrue(report['classes']['path']['missing_pieces'])
        self.assertTrue(report['classes']['path']['lost'])
        beyond = sum(x['beyond_edges_m2'] for x in report['classes']['path']['lost'])
        self.assertAlmostEqual(beyond, sum(a['area_m2'] for a in kept) - 2 * 0.8 * 1.4, delta=0.2,
                               msg='all of it but 0.8 m each side of the street it crosses (a kept edge)')


# ------------------------------------------------------------------------------------------ snapshots
def rewrite(folder: Path, place=None, osm=None, terrain=None, manifest_edit=None):
    """Change a copied snapshot's files and update SNAPSHOT.json's checksums to match, so only the change itself
    (not a stale checksum) can make verification fail."""
    m = json.loads((folder / snapshot.MANIFEST).read_text(encoding='utf-8'))
    for name, data in (('place.json', place), ('osm.json', osm), ('terrain.json', terrain)):
        if data is None:
            continue
        raw = json.dumps(data, ensure_ascii=False).encode('utf-8')
        stored = gzip.compress(raw, mtime=0) if snapshot.FILES[name].endswith('.gz') else raw
        (folder / snapshot.FILES[name]).write_bytes(stored)
        m['files'][name] = {'stored': snapshot.FILES[name], 'sha256': snapshot.sha256(raw), 'bytes': len(raw),
                            'stored_sha256': snapshot.sha256(stored), 'stored_bytes': len(stored)}
    if manifest_edit:
        manifest_edit(m)
    (folder / snapshot.MANIFEST).write_text(json.dumps(m, ensure_ascii=False), encoding='utf-8')


class SnapshotTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest, cls.files = snapshot.verify(SNAPSHOT)
        cls.osm = json.loads(cls.files['osm.json'].decode('utf-8'))
        cls.terrain = json.loads(cls.files['terrain.json'].decode('utf-8'))
        cls.place = json.loads(cls.files['place.json'])

    def test_the_real_snapshot_verifies(self):
        m = self.manifest
        for name, data in self.files.items():
            self.assertEqual(snapshot.sha256(data), m['files'][name]['sha256'], name)
        self.assertEqual(m['id'], 'boras-p01')
        self.assertEqual((self.place['center'], self.place['size_m'], self.place['projection']), ([57.7210839, 12.9390626], 1000.0, 'wgs84'))
        self.assertTrue(snapshot.STAMP.match(m['osm']['timestamp_osm_base']), 'the sources stated their times')
        self.assertEqual(m['osm']['sources'][0]['source'], 'OpenStreetMap API 0.6 map')
        self.assertLessEqual(m['osm']['downloaded_utc']['started'], m['osm']['downloaded_utc']['finished'])
        self.assertEqual(m['osm']['incomplete_ways'], [])
        self.assertNotIn(b'\r', self.files['place.json'])
        self.assertEqual(m['osm']['ways'], m['osm']['ways_with_node_ids'], 'every way keeps its node ids')
        self.assertEqual((m['osm']['street_view_tags'], m['osm']['incomplete_relations']), ([], []))
        self.assertEqual(m['osm']['license'], 'ODbL-1.0')
        self.assertEqual(m['terrain']['tiles'], ['Copernicus_DSM_COG_10_N57_00_E012_00_DEM'])
        self.assertIn('Copernicus DEM licence', m['terrain']['license'])
        self.assertIn('none were configured', m['terrain']['primary_source'], 'the fallback is explicit')
        self.assertEqual(m['terrain']['coverage']['play_area_no_data_cells'], 0)
        self.assertEqual(self.terrain['projection'], 'wgs84')
        self.assertTrue((SNAPSHOT / snapshot.NOTICE).exists())

    def test_tampered_or_malformed_snapshots_are_refused(self):
        def element(osm, kind, test):
            return next(el for el in osm['elements'] if el['type'] == kind and test(el))
        cases = {
            'a changed byte': lambda f: (f / 'osm.json.gz').write_bytes(gzip.compress(self.files['osm.json'].replace(b'Bor', b'Bar', 1), mtime=0)),
            'a stale checksum': lambda f: rewrite(f, manifest_edit=lambda m: m['files']['terrain.json'].update(sha256='0' * 64)),
            'another data time': lambda f: rewrite(f, manifest_edit=lambda m: m['osm'].update(timestamp_osm_base='2020-01-01T00:00:00Z')),
            'another download time': lambda f: rewrite(f, manifest_edit=lambda m: m['osm']['downloaded_utc'].update(started='2020-01-01T00:00:00Z')),
            'a data time newer than a source': lambda f: rewrite(f, osm=self._with(lambda o: o['osm3s'].update(timestamp_osm_base='2030-01-01T00:00:00Z')),
                                                                 manifest_edit=lambda m: m['osm'].update(timestamp_osm_base='2030-01-01T00:00:00Z')),
            'no fetch record': lambda f: rewrite(f, osm=self._with(lambda o: o.pop('fetch'))),
            'a way a node short': lambda f: rewrite(f, osm=self._with(lambda o: element(o, 'way', lambda el: 'highway' in el['tags'])['nodes'].append(1))),
            'no terrain tiles': lambda f: rewrite(f, manifest_edit=lambda m: m['terrain'].update(tiles=[])),
            'holes in the play area': lambda f: rewrite(f, manifest_edit=lambda m: m['terrain']['coverage'].update(play_area_no_data_cells=3)),
            'a hole next to the play area': lambda f: rewrite(f, terrain={**self.terrain, 'coverage': {**self.terrain['coverage'], 'support_cells': 4000,
                                                                                                     'support_fallback_cells': 0, 'support_no_data_cells': 1}},
                                                              manifest_edit=lambda m: m['terrain']['coverage'].update(support_cells=4000, support_fallback_cells=0,
                                                                                                                    support_no_data_cells=1)),
            'a hole somewhere, support not counted': lambda f: rewrite(f, terrain={**self.terrain, 'coverage': {**self.terrain['coverage'], 'no_data_cells': 1}},
                                                                       manifest_edit=lambda m: m['terrain']['coverage'].update(no_data_cells=1)),
            'an unknown terrain licence': lambda f: rewrite(f, terrain={**self.terrain, 'attribution': 'Terrain: a mystery source'},
                                                             manifest_edit=lambda m: m['terrain'].update(attribution='Terrain: a mystery source')),
            'a Street View source tag': lambda f: rewrite(f, osm=self._with(lambda o: element(o, 'way', lambda el: 'building' in el['tags'])['tags']
                                                                            .update(source='Google Street View'))),
            'an incomplete area': lambda f: rewrite(f, osm=self._with(lambda o: element(o, 'relation', lambda el: el['tags'].get('type') == 'multipolygon')
                                                                       ['members'][0].pop('geometry'))),
            'node ids dropped': lambda f: rewrite(f, osm=self._with(lambda o: element(o, 'way', lambda el: 'highway' in el['tags']).pop('nodes'))),
            'a moved centre': lambda f: rewrite(f, place={**self.place, 'center': [57.73, 12.94]}),
            'the sphere plane': lambda f: rewrite(f, place={**self.place, 'projection': 'sphere'}),
            'a terrain grid short of the area': lambda f: rewrite(f, terrain={**self.terrain, 'x0': -400.0}),
            'an extra file': lambda f: (f / 'overrides.json').write_text('{}'),
            'no manifest': lambda f: (f / snapshot.MANIFEST).unlink(),
        }
        for what, change in cases.items():
            with tempfile.TemporaryDirectory() as tmp:
                copy_ = Path(tmp) / SNAPSHOT.name
                shutil.copytree(SNAPSHOT, copy_)
                change(copy_)
                with self.assertRaises(snapshot.SnapshotError, msg=what):
                    snapshot.verify(copy_)
        with tempfile.TemporaryDirectory() as tmp:
            renamed = Path(tmp) / 'another-name'
            shutil.copytree(SNAPSHOT, renamed)
            with self.assertRaises(snapshot.SnapshotError):
                snapshot.verify(renamed)

    def _with(self, change):
        osm = copy.deepcopy(self.osm)
        change(osm)
        return osm

    def _create(self, osm, env_text):
        """snapshot.create with the download steps replaced by writing these files (no network) and the
        repository's .env replaced by `env_text`. (manifest or SnapshotError, the environment the steps saw)."""
        seen = {}

        def fake_run(cmd, *a, **k):
            city = Path(cmd[-2]) / cmd[-1]
            seen.update({k_: os.environ.get(k_) for k_ in snapshot.LM_KEYS})
            (city / 'osm.json').write_bytes(json.dumps(osm).encode('utf-8'))
            (city / 'terrain.json').write_bytes(json.dumps(self.terrain).encode('utf-8'))
            return subprocess.CompletedProcess(cmd, 0)
        import wasteland
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {}, clear=False):
            for k in snapshot.LM_KEYS:
                os.environ.pop(k, None)
            root = Path(tmp) / 'repo'
            root.mkdir()
            (root / '.env').write_text(env_text)
            with mock.patch.object(wasteland, 'ROOT', root), mock.patch.object(snapshot.subprocess, 'run', side_effect=fake_run):
                try:
                    result = snapshot.create(Path(tmp) / 'out' / 'boras-p01', name='Borås', center=(57.7210839, 12.9390626), size_m=1000.0,
                                             country='se', selection='test')
                except snapshot.SnapshotError as exc:
                    result = exc
        return result, seen

    def test_create_reads_the_env_file_and_refuses_incomplete_downloads(self):
        m, seen = self._create(self.osm, 'LANTMATERIET_USER=someone\nLANTMATERIET_PASSWORD=secret\n')
        self.assertIsInstance(m, dict)
        self.assertEqual((seen['LANTMATERIET_USER'], seen['LANTMATERIET_PASSWORD']), ('someone', 'secret'), 'the download steps got .env')
        self.assertIn('they were set, yet it gave no data', m['terrain']['primary_source'])
        self.assertNotIn('secret', json.dumps(m))
        m, seen = self._create(self.osm, '# nothing\n')
        self.assertIn('none were configured', m['terrain']['primary_source'])
        multipolygon = lambda o: next(el for el in o['elements'] if el['type'] == 'relation' and el['tags'].get('type') == 'multipolygon')  # noqa: E731
        for what, change in (('a way a node short', lambda o: next(el for el in o['elements'] if el['type'] == 'way')['nodes'].append(1)),
                             ('no fetch record', lambda o: o.pop('fetch')),
                             ('an incomplete area', lambda o: multipolygon(o)['members'][0].pop('geometry'))):
            bad, _ = self._create(self._with(change), '')
            self.assertIsInstance(bad, snapshot.SnapshotError, what)

    def test_street_view_and_incomplete_area_detection(self):
        osm = {'elements': [{'type': 'way', 'id': 1, 'tags': {'building': 'yes', 'source:geometry': 'streetview'}},
                            {'type': 'way', 'id': 2, 'tags': {'building': 'yes', 'source': 'survey;Mapillary'}},
                            {'type': 'relation', 'id': 3, 'tags': {'type': 'multipolygon'}, 'members': [{'type': 'way', 'ref': 9, 'role': 'outer'}]}]}
        self.assertEqual(snapshot.street_view_tags(osm), ['way 1 source:geometry=streetview'])
        self.assertEqual(snapshot.incomplete_relations(osm), ['relation 3'])


# ------------------------------------------------------------------------------------------ the real dataset
class RealBorasTest(unittest.TestCase):
    """The ~1 km² of central Borås, built from the snapshot alone, twice."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix='outbreak-p01-test-'))
        out = cls.tmp / 'other' / 'OutbreakGeo.json'
        env = {**os.environ, 'PYTHONHASHSEED': '4242'}
        other = subprocess.Popen([sys.executable, str(ROOT / 'pipeline/outbreak/snapshot.py'), 'build', str(SNAPSHOT), '--out', str(out)],
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True, encoding='utf-8', errors='replace')
        cls.timings = {}
        cls.geo = snapshot.build(SNAPSHOT, cls.tmp / 'here', timings=cls.timings)
        log, _ = other.communicate()
        if other.returncode:
            raise AssertionError(f'the second build failed: {log[-800:]}')
        cls.text = adapter.dumps(cls.geo).encode('utf-8')
        cls.other = out.read_bytes()
        cls.folder = cls.tmp / 'here' / 'boras-p01'
        cls.manifest, cls.files = snapshot.verify(SNAPSHOT)
        cls.report = realcheck.report(json.loads(cls.text), cls.manifest, cls.files)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_two_independent_clean_builds_are_byte_identical(self):
        self.assertEqual(snapshot.sha256(self.text), snapshot.sha256(self.other))
        self.assertEqual(self.text, self.other)
        doc = json.loads(self.text)
        self.assertEqual(adapter.validate(doc), [])
        self.assertTrue(doc['provenance']['sources']['inputs']['city']['clean_rebuild'])

    def test_metres_bounds_and_no_render_data(self):
        m = self.geo['metadata']
        self.assertEqual((m['bounds']['min_m'], m['bounds']['max_m']), ([-500.0, -500.0], [500.0, 500.0]))
        self.assertEqual(m['coordinates']['projection'], 'equirectangular-wgs84')
        self.assertFalse(m['synthetic'])
        keys = set(keys_in(self.geo))
        self.assertEqual(MESH_KEYS & keys, set(), 'no render triangle soup or decoration')
        self.assertFalse(any('stud' in k for k in keys))
        self.assertLess(len(self.text), (self.folder / 'city.json').stat().st_size / 5)

    def test_chunks(self):
        c = self.report['chunks']
        self.assertTrue(c['pass'])
        self.assertEqual((c['size_m'], c['grid'], c['listed']), (60.0, [18, 18], 324))
        self.assertEqual(c['edge_chunks'], 68, '20 m strips along the edges of a 1000 m square')
        self.assertGreaterEqual(c['road_longest_by_chunks']['chunks'], 10)
        self.assertGreaterEqual(c['building_widest']['chunks'], 4)
        self.assertGreaterEqual(c['area_widest']['chunks'], 50)
        self.assertEqual(len(c['building_vertices_by_quadrant']), 4, 'features at negative and positive coordinates')
        for b in self.geo['chunks']:
            x0, y0, x1, y1 = b['bounds_m']
            self.assertTrue(-500 <= x0 < x1 <= 500 and -500 <= y0 < y1 <= 500, b['id'])

    def test_decimal_chunk_size_on_real_data(self):
        """The chunk rules hold at a size that floor(x / size) gets wrong, on the real city."""
        city = adapter.read_json(self.folder / 'city.json', [])[0]
        osm = json.loads(self.files['osm.json'].decode('utf-8'))
        geo = adapter.build_geo(city, osm=osm, terrain_src=json.loads(self.files['terrain.json']), chunk_size=30.1)
        self.assertEqual(adapter.validate(geo), [])
        self.assertEqual(len(geo['chunks']), 34 * 34)

    def test_road_topology(self):
        t = self.report['topology']
        self.assertTrue(t['pass'], t)
        self.assertEqual((t['roads_with_node_ids'], t['topology']), (t['roads'], 'osm_nodes'))
        self.assertEqual((t['shared_position'], t['unresolved'], t['roads_without_node_ids']), (0, 0, 0))
        self.assertEqual(t['junctions'], t['verified_osm_node'])
        self.assertTrue(t['verified_junctions_match_raw_osm'])
        self.assertEqual((t['node_order_mismatches'], t['verified_junction_mismatches'], t['separated_without_level_difference']), ([], [], []))
        self.assertGreater(t['crossings_separated'], 10, 'bridges over streets in the area')
        self.assertEqual(t['separated_pairs_also_joined'], 0)
        self.assertTrue(t['route']['pass'], t['route'])

    def test_topology_is_independent_of_input_order_on_real_data(self):
        city = adapter.read_json(self.folder / 'city.json', [])[0]
        osm = json.loads(self.files['osm.json'].decode('utf-8'))
        random.Random(11).shuffle(city['roads'])
        random.Random(12).shuffle(osm['elements'])
        geo = adapter.build_geo(city, osm=osm, terrain_src=json.loads(self.files['terrain.json']))
        for key in ('junctions', 'crossings', 'topology'):
            self.assertEqual(geo['navigation'][key], self.geo['navigation'][key], key)
        self.assertEqual([r['nodes'] for r in geo['roads']], [r['nodes'] for r in self.geo['roads']])

    def test_buildings(self):
        b = self.report['buildings']
        self.assertTrue(b['pass'], b)
        self.assertGreater(b['total'], 400)
        self.assertEqual(sum(b['wall_height'].values()), b['total'])
        self.assertIn('inferred', b['wall_height'])
        self.assertIn('osm', b['levels'])
        self.assertTrue(b['courtyards'] and b['courtyards_kept'])
        self.assertEqual(b['raised_in_ground_collision'], [])
        self.assertIn('Caroli kyrka', b['landmark_names'])

    def test_scale(self):
        s = self.report['scale']
        self.assertTrue(s['pass'], [m for m in s['measurements'] if not m['pass']] or s['position'])
        self.assertGreaterEqual(s['count'], 20)
        self.assertEqual({m['kind'] for m in s['measurements']} >= {'road segment', 'road length', 'across the area', 'building edge'}, True)
        quadrants = {m['source'][-3:-1] for m in s['measurements'] if m['kind'] == 'road segment'}
        self.assertEqual(quadrants, {'NE', 'NW', 'SE', 'SW'})
        self.assertGreater(max(m['source_m'] for m in s['measurements']), 1000)
        self.assertLess(abs(s['position']['rotation_rad']), realcheck.THRESHOLDS['rotation_rad'])

    def test_completeness(self):
        c = self.report['completeness']
        self.assertEqual(c['unexplained'], [])
        self.assertTrue(c['roads']['drivable'].get('kept'))
        self.assertTrue(c['buildings']['outlines'].get('kept as a building'))
        self.assertTrue(c['pass'] and c['areas']['pass'], c['areas'])
        water = c['areas']['classes']['water']
        self.assertAlmostEqual(water['outbreakgeo_m2'], water['source_m2'], delta=0.01 * water['source_m2'])
        self.assertTrue(all(not v['lost'] and v.get('fabricated_m2', 0) <= realcheck.FABRICATED_M2 for v in c['areas']['classes'].values()))

    def test_lost_or_fabricated_ground_fails_the_area_gate(self):
        """The navigator's case: ground classes removed (chunk index kept consistent) still pass the contract, but
        not the completeness gate, which checks every mapped area against the source."""
        geo = json.loads(self.text)

        def without(pred):
            g = copy.deepcopy(geo)
            gone = {a['id'] for a in g['areas'] if pred(a)}
            g['areas'] = [a for a in g['areas'] if a['id'] not in gone]
            for c in g['chunks']:
                c['areas'] = [x for x in c['areas'] if x not in gone]
            return g
        biggest = {c: max((a for a in geo['areas'] if a['class'] == c), key=lambda a: a['area_m2'])['id'] for c in ('park', 'parking')}
        relabelled = copy.deepcopy(geo)
        max((a for a in relabelled['areas'] if a['class'] == 'grass'), key=lambda a: a['area_m2'])['class'] = 'park'
        kungsgatan = LineString(next(r for r in geo['roads'] if r['id'] == 'w4265634')['centerline']).buffer(8)
        biggest['carriageway'] = max((a for a in geo['areas'] if a['class'] == 'carriageway'), key=lambda a: a['area_m2'])['id']
        smallest = {c: min((a for a in geo['areas'] if a['class'] == c and a['area_m2'] >= 50), key=lambda a: (a['area_m2'], a['id']))['id']
                    for c in ('sidewalk', 'path')}
        cases = {'park, grass and parking gone': without(lambda a: a['class'] in ('park', 'grass', 'parking')),
                 'one narrow sidewalk piece gone': without(lambda a: a['id'] == smallest['sidewalk']),
                 'one narrow path piece gone': without(lambda a: a['id'] == smallest['path']),
                 'every sidewalk gone': without(lambda a: a['class'] == 'sidewalk'),
                 "Kungsgatan's sidewalks gone": without(lambda a: a['class'] == 'sidewalk' and Polygon(a['polygon']['outer']).intersects(kungsgatan)),
                 'the largest carriageway gone': without(lambda a: a['id'] == biggest['carriageway']),
                 'every path gone': without(lambda a: a['class'] == 'path'),
                 'all pedestrian ground gone': without(lambda a: a['class'] == 'pedestrian'),
                 'the largest park gone': without(lambda a: a['id'] == biggest['park']),
                 'the largest car park gone': without(lambda a: a['id'] == biggest['parking']),
                 'the water gone': without(lambda a: a['class'] == 'water'),
                 'grass called park': relabelled}
        reader = prepare_city.OSM(json.loads(self.files['osm.json'].decode('utf-8')), common.Projection(57.7210839, 12.9390626, 'wgs84'))
        world = Polygon([(-500, -500), (500, -500), (500, 500), (-500, 500)])
        for what, doc in cases.items():
            self.assertEqual(adapter.check_geo(json.loads(adapter.dumps(doc))), [], what)
            self.assertFalse(realcheck.area_report(doc, reader, world)['pass'], what)
        self.assertFalse(realcheck.report(cases['one narrow path piece gone'], self.manifest, self.files)['completeness']['pass'])
        self.assertTrue(self.report['completeness']['areas']['pass'])

    def test_terrain(self):
        t = self.report['terrain']
        self.assertTrue(t['pass'], t)
        self.assertEqual(t['non_finite_heights'], 0)
        self.assertTrue(t['covers_play_area'])
        self.assertEqual(t['kind'], 'dsm')
        self.assertIn('Copernicus', t['attribution'])

    def test_provenance_and_licences(self):
        p = self.report['provenance']
        self.assertTrue(p['pass'], p)
        self.assertEqual(p['licenses']['OpenStreetMap'], 'ODbL-1.0')
        self.assertTrue(all(p['inputs_match_snapshot'].values()))
        self.assertEqual(p['google_mentions_outside_policy'], 0)
        self.assertIn('OpenStreetMap contributors', ' '.join(p['attribution']))
        self.assertGreater(self.timings['prepare_s'], 0)


if __name__ == '__main__':
    unittest.main()
