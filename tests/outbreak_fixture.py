"""The synthetic city for the Outbreak adapter tests (tests/test_outbreak.py): no network, no real place.

"Synthby" is a 300 m square (the size of the later Roblox proof of concept) with:
- two connected streets crossing at (32, 8), where they share an OSM node: Storgatan (east–west,
  residential, running past both edges) and Kyrkvägen (north–south, tertiary, sett), plus a footway,
  Parkstigen. The road ways carry node ids, as Overpass gives them;
- a paved square (Torget) and, from prepare_city, inferred sidewalks along both streets;
- a park and a pond;
- five buildings of different heights: an untagged house (storeys guessed), a five-storey block across
  the x = 0 chunk border, an office tagged height=12.5 with a flat roof and brick walls, the church
  (a landmark, typed only by amenity=place_of_worship) and a courtyard block (a multipolygon relation
  with a hole) across the x = −120 chunk border;
- a ground model rising 2 % to the north.
Positions are local metres; the pipeline's own Projection (the WGS84 plane an Outbreak source uses) turns
them into latitude/longitude, so prepare_city.py gets them back to the centimetre. The centre is an arbitrary point, used only for the
projection.
"""
import json

from common import Projection

SLUG = '_outbreak_selftest'
CENTER = (60.0, 15.0)
SIZE_M = 300
P = Projection(*CENTER, 'wgs84')

# OSM way id → (x0, y0, x1, y1, tags), in local metres
BUILDINGS = {
    10: (-50, 25, -38, 34, {'building': 'house'}),
    11: (-30, -40, 15, -22, {'building': 'apartments', 'building:levels': '5'}),
    12: (40, -45, 70, -25, {'building': 'office', 'height': '12.5', 'roof:shape': 'flat', 'building:material': 'brick'}),
    13: (-100, 30, -70, 46, {'building': 'yes', 'amenity': 'place_of_worship', 'name': 'Synthby kyrka'}),
}
COURTYARD = {'id': 30, 'outer': (-140, 60, -95, 100), 'inner': (-128, 72, -107, 88),
             'tags': {'type': 'multipolygon', 'building': 'apartments', 'building:levels': '4', 'roof:shape': 'flat'}}
ROADS = {
    1: ([(-170, 8), (32, 8), (170, 8)], {'highway': 'residential', 'name': 'Storgatan'}),
    2: ([(32, -170), (32, 8), (32, 170)], {'highway': 'tertiary', 'name': 'Kyrkvägen', 'surface': 'sett'}),
    3: ([(-130, -90), (-20, -90)], {'highway': 'footway', 'name': 'Parkstigen'}),
}
# Node ids of the road ways, as Overpass gives them: Storgatan and Kyrkvägen share node 100 at (32, 8).
ROAD_NODES = {1: [101, 100, 103], 2: [201, 100, 203], 3: [301, 302]}
AREAS = {
    20: ((45, 20, 100, 55), {'highway': 'pedestrian', 'area': 'yes', 'name': 'Torget'}),
    21: ((-115, -125, -45, -60), {'leisure': 'park', 'name': 'Synthparken'}),
    22: ((70, -125, 125, -75), {'natural': 'water', 'name': 'Dammen'}),
}


def ll(x, y):
    lat, lon = P.latlon(x, y)
    return {'lat': lat, 'lon': lon}


def rect(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]


def way(i, pts, tags):
    return {'type': 'way', 'id': i, 'tags': tags, 'geometry': [ll(*p) for p in pts]}


def elements():
    out = [way(i, rect(*r[:4]), r[4]) for i, r in BUILDINGS.items()]
    c = COURTYARD
    out.append({'type': 'relation', 'id': c['id'], 'tags': c['tags'], 'members': [
        {'type': 'way', 'ref': 301, 'role': 'outer', 'geometry': [ll(*p) for p in rect(*c['outer'])]},
        {'type': 'way', 'ref': 302, 'role': 'inner', 'geometry': [ll(*p) for p in rect(*c['inner'])]}]})
    out += [{**way(i, pts, tags), 'nodes': ROAD_NODES[i]} for i, (pts, tags) in ROADS.items()]
    out += [way(i, rect(*r), tags) for i, (r, tags) in AREAS.items()]
    return out


def terrain(step=10.0, n=129):
    """terrain.json in fetch_terrain.py's local-grid form: a ground model rising 2 % to the north."""
    x0 = -step * (n - 1) / 2
    z = [round(25.0 + 0.02 * (x0 + j * step), 2) for j in range(n) for _ in range(n)]
    return {'source': 'synthetic ground model', 'kind': 'dtm', 'attribution': 'Terrain: synthetic test grid',
            'tiles': ['synthetic-1'], 'frame': 'local', 'projection': 'wgs84', 'x0': x0, 'y0': x0, 'step': step, 'n': n, 'z': z}


def write(folder):
    """place.json, osm.json and terrain.json for prepare_city.py."""
    half = SIZE_M / 2
    (s, w), (n, e) = P.latlon(-half, -half), P.latlon(half, half)
    place = {'name': 'Synthby', 'query': 'Synthby', 'display_name': 'Synthby (synthetic test city)', 'country_code': '',
             'center': list(CENTER), 'size_m': SIZE_M, 'projection': 'wgs84', 'bbox': [round(s, 6), round(w, 6), round(n, 6), round(e, 6)],
             'attribution': 'Map data © OpenStreetMap contributors (ODbL)'}
    osm = {'version': 0.6, 'generator': 'outbreak test fixture', 'osm3s': {'timestamp_osm_base': 'synthetic'}, 'elements': elements()}
    folder.mkdir(parents=True, exist_ok=True)
    for name, data in (('place.json', place), ('osm.json', osm), ('terrain.json', terrain())):
        (folder / name).write_text(json.dumps(data))      # ASCII (\u escapes): readable under any locale
