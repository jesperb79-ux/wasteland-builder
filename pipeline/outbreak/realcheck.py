"""P01 validation of an OutbreakGeo built from a real snapshot, from first principles where it matters.

    .venv/bin/python pipeline/outbreak/realcheck.py <snapshot folder> [--builds 2] [--out report.json]

Builds the OutbreakGeo from the snapshot in clean temporary folders (each in its own process, with its own
hash seed; no network), checks the builds are byte-identical, and reports on the result against the
snapshot's raw osm.json and terrain.json:

- chunks: grid size and counts, populated and empty chunks, the widest building, road and area;
- topology: node ids, verified and fallback junctions, crossings, every verified junction and every
  road's node order checked against the raw OSM ways, and a route through verified junctions across the
  area;
- buildings: counts, mapped against inferred heights and storeys, roofs, courtyards, raised parts;
- scale: at least 20 distances between OSM nodes measured on the WGS84 ellipsoid by Vincenty's formula
  (not through the pipeline's projection) against the same distances in OutbreakGeo, and every road
  node's position against an azimuthal-equidistant reference: offset, rotation, scale;
- completeness: what the source has in the play area against what OutbreakGeo keeps, each exclusion
  by a named rule;
- terrain, provenance, reproducibility and the offline pipeline's time, size and memory.

Every check has a pass rule fixed here, before any result is seen (TOLERANCE and the THRESHOLDS).
"""
from __future__ import annotations

import argparse
import heapq
import json
import math
import shutil
import os
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

if __package__ in (None, ''):                                    # run as a script: find pipeline/
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import shapely  # noqa: E402
from shapely.geometry import Polygon, box  # noqa: E402
from shapely.ops import unary_union  # noqa: E402
from shapely.strtree import STRtree  # noqa: E402

import prepare_city  # noqa: E402
from common import ROOT, Projection  # noqa: E402
from outbreak import adapter, snapshot  # noqa: E402
from outbreak.chunks import Grid  # noqa: E402

# ------------------------------------------------------------------------------------------ pass rules (fixed in advance)
def tolerance(d_m: float) -> float:
    """Allowed error of a measured length: 0.10 m, or 0.1 % for lengths over 100 m. The budget it covers:
    city.json rounds every coordinate to 0.01 m (≤ 0.014 m on a distance), the WGS84 plane is true to
    1.3e-4 within 500 m of the centre, and building simplification drops vertices but never moves one."""
    return max(0.10, 0.001 * d_m)


THRESHOLDS = {
    'min_measurements': 20,
    'position_max_m': 0.10,          # any road node's position against the geodesic reference
    'position_mean_offset_m': 0.03,  # the mean residual vector: a systematic shift of the whole map
    'rotation_rad': 2e-5,            # best-fit rotation (≈ 1 cm at 500 m)
    'scale_dev': 2e-4,               # best-fit scale − 1
    'route_min_junctions': 10,       # a route through verified junctions across the area
    'route_min_span_m': 600.0,
}

# ------------------------------------------------------------------------------------------ geodesy (independent of common.Projection)
WGS84_A, WGS84_F = 6378137.0, 1 / 298.257223563


def geodesic(lat1, lon1, lat2, lon2) -> tuple[float, float]:
    """(distance in metres, forward azimuth in radians from north) between two points on the WGS84 ellipsoid,
    by Vincenty's inverse formula (accurate to about 0.5 mm)."""
    a, f = WGS84_A, WGS84_F
    b = a * (1 - f)
    if lat1 == lat2 and lon1 == lon2:
        return 0.0, 0.0
    L = math.radians(lon2 - lon1)
    U1, U2 = math.atan((1 - f) * math.tan(math.radians(lat1))), math.atan((1 - f) * math.tan(math.radians(lat2)))
    sU1, cU1, sU2, cU2 = math.sin(U1), math.cos(U1), math.sin(U2), math.cos(U2)
    lam = L
    for _ in range(200):
        sl, cl = math.sin(lam), math.cos(lam)
        s_sig = math.hypot(cU2 * sl, cU1 * sU2 - sU1 * cU2 * cl)
        if s_sig == 0:
            return 0.0, 0.0
        c_sig = sU1 * sU2 + cU1 * cU2 * cl
        sig = math.atan2(s_sig, c_sig)
        s_alpha = cU1 * cU2 * sl / s_sig
        c2_alpha = 1 - s_alpha ** 2
        c2_sm = c_sig - 2 * sU1 * sU2 / c2_alpha if c2_alpha else 0.0
        C = f / 16 * c2_alpha * (4 + f * (4 - 3 * c2_alpha))
        prev = lam
        lam = L + (1 - C) * f * s_alpha * (sig + C * s_sig * (c2_sm + C * c_sig * (-1 + 2 * c2_sm ** 2)))
        if abs(lam - prev) < 1e-13:
            break
    else:
        raise ValueError('Vincenty did not converge')
    u2 = c2_alpha * (a * a - b * b) / (b * b)
    A = 1 + u2 / 16384 * (4096 + u2 * (-768 + u2 * (320 - 175 * u2)))
    B = u2 / 1024 * (256 + u2 * (-128 + u2 * (74 - 47 * u2)))
    d_sig = B * s_sig * (c2_sm + B / 4 * (c_sig * (-1 + 2 * c2_sm ** 2) - B / 6 * c2_sm * (-3 + 4 * s_sig ** 2) * (-3 + 4 * c2_sm ** 2)))
    az = math.atan2(cU2 * sl, cU1 * sU2 - sU1 * cU2 * cl)
    return b * A * (sig - d_sig), az


def reference_xy(lat0, lon0, lat, lon) -> tuple[float, float]:
    """Azimuthal-equidistant position of a point round the centre: the geodesic distance along its azimuth."""
    s, az = geodesic(lat0, lon0, lat, lon)
    return s * math.sin(az), s * math.cos(az)


# ------------------------------------------------------------------------------------------ the raw source
class Source:
    """The snapshot's raw OSM ways and nodes: where each node is, which ways hold it, in which order."""

    def __init__(self, osm: dict):
        self.elements = osm['elements']
        self.ways = {el['id']: el for el in self.elements if el.get('type') == 'way'}
        self.node_ll, self.conflicts = {}, 0
        for w in self.ways.values():
            for n, g in zip(w.get('nodes') or [], w.get('geometry') or []):
                ll = (g['lat'], g['lon'])
                if self.node_ll.setdefault(n, ll) != ll:
                    self.conflicts += 1
        self.ways_of = defaultdict(set)
        for w in self.ways.values():
            for n in w.get('nodes') or []:
                self.ways_of[n].add(w['id'])


def _num(v):
    return prepare_city.num(v)


# ------------------------------------------------------------------------------------------ chunks
def chunk_report(geo) -> dict:
    meta = geo['metadata']
    S = meta['chunking']['size_m']
    (x0, y0), (x1, y1) = meta['bounds']['min_m'], meta['bounds']['max_m']
    nx = math.ceil(x1 / S) - math.floor(x0 / S)
    ny = math.ceil(y1 / S) - math.floor(y0 / S)
    chunks = geo['chunks']
    populated = [c for c in chunks if c['buildings'] or c['roads'] or c['areas'] or c['collision']]
    edge = [c for c in chunks if abs((c['bounds_m'][2] - c['bounds_m'][0]) - S) > 1e-9 or abs((c['bounds_m'][3] - c['bounds_m'][1]) - S) > 1e-9]

    def widest(items, key_extra):
        return max(items, key=lambda f: (len(f['chunks']), key_extra(f), f['id']))
    wb, wr, wa = widest(geo['buildings'], lambda f: f['area_m2']), widest(geo['roads'], lambda f: f['length_m']), widest(geo['areas'], lambda f: f['area_m2'])
    on_border = sum(1 for r in geo['roads'] for x, y in r['centerline']
                    if x0 <= x < x1 and y0 <= y < y1 and (abs(x / S - round(x / S)) < 1e-12 or abs(y / S - round(y / S)) < 1e-12))
    quadrants = Counter((x >= 0, y >= 0) for b in geo['buildings'] for x, y in b['footprint']['outer'])
    return {
        'size_m': S, 'grid': [nx, ny], 'theoretical': nx * ny, 'listed': len(chunks), 'populated': len(populated),
        'empty': len(chunks) - len(populated), 'edge_chunks': len(edge),
        'building_widest': {'id': wb['id'], 'name': wb['name'], 'chunks': len(wb['chunks']), 'area_m2': wb['area_m2']},
        'road_longest_by_chunks': {'id': wr['id'], 'name': wr['name'], 'class': wr['class'], 'chunks': len(wr['chunks']), 'length_m': round(wr['length_m'], 1)},
        'area_widest': {'id': wa['id'], 'class': wa['class'], 'chunks': len(wa['chunks']), 'area_m2': wa['area_m2']},
        'buildings_over_borders': sum(1 for b in geo['buildings'] if len(b['chunks']) > 1),
        'roads_over_borders': sum(1 for r in geo['roads'] if len(r['chunks']) > 1),
        'road_vertices_on_chunk_borders': on_border,
        'building_vertices_by_quadrant': {f'{"+" if a else "-"}x{"+" if b else "-"}y': n for (a, b), n in sorted(quadrants.items())},
        'pass': nx * ny == len(chunks),
    }


# ------------------------------------------------------------------------------------------ topology
def topology_report(geo, src: Source) -> dict:
    roads = geo['roads']
    nav = geo['navigation']
    with_nodes = [r for r in roads if r['nodes'] is not None]
    by_id = {r['id']: r for r in roads}
    # Node order and identity against the raw ways.
    order_bad = [r['id'] for r in with_nodes if r['osm'] is None or src.ways.get(r['osm']['id'], {}).get('nodes') != r['nodes']]
    # Every verified junction: its node is in each listed road's raw way, and no other OutbreakGeo road's way has it.
    way_to_road = defaultdict(list)
    for r in roads:
        if r['osm'] and r['osm']['type'] == 'way':
            way_to_road[r['osm']['id']].append(r['id'])
    verified = [j for j in nav['junctions'] if j['basis'] == 'osm_node']
    junction_bad = []
    for j in verified:
        holders = sorted(rid for w in src.ways_of.get(j['node'], ()) for rid in way_to_road.get(w, ()))
        if holders != j['roads']:
            junction_bad.append(j['id'])
    # Independently: nodes shared by two or more OutbreakGeo roads' raw ways, inside the half-open bounds.
    (x0, y0), (x1, y1) = geo['metadata']['bounds']['min_m'], geo['metadata']['bounds']['max_m']
    expected = set()
    for n, ways in src.ways_of.items():
        holders = {rid for w in ways for rid in way_to_road.get(w, ())}
        if len(holders) > 1:
            pos = next((p for rid in holders for p, nd in zip(by_id[rid]['centerline'], by_id[rid]['nodes'] or []) if nd == n), None)
            if pos is not None and x0 <= pos[0] < x1 and y0 <= pos[1] < y1:
                expected.add(n)
    crossings = nav['crossings']
    separated = [c for c in crossings if c['status'] == 'separated']
    # Separated crossings must really be on different levels by their tags; never in a junction together.
    junction_pairs = {frozenset((a, b)) for j in nav['junctions'] for a in j['roads'] for b in j['roads'] if a < b}
    sep_bad = [c['id'] for c in separated if not c['differ'] or adapter.vertical_level(by_id[c['roads'][0]]) == adapter.vertical_level(by_id[c['roads'][1]])]
    return {
        'roads': len(roads), 'roads_with_node_ids': len(with_nodes), 'roads_with_node_ids_pct': round(100 * len(with_nodes) / max(1, len(roads)), 2),
        'topology': nav['topology'],
        'junctions': len(nav['junctions']), 'verified_osm_node': len(verified),
        'shared_position': sum(1 for j in nav['junctions'] if j['basis'] == 'shared_position'),
        'roads_without_node_ids': len(roads) - len(with_nodes),
        'unresolved': sum(1 for c in crossings if c['status'] == 'unresolved') + (len(roads) - len(with_nodes)),
        'crossings': len(crossings), 'crossings_separated': len(separated),
        'crossings_separated_by': dict(Counter('+'.join(c['differ']) for c in separated)),
        'crossings_unresolved': sum(1 for c in crossings if c['status'] == 'unresolved'),
        'separated_pairs_also_joined': sum(1 for c in separated if frozenset(c['roads']) in junction_pairs),
        'node_order_mismatches': order_bad, 'verified_junction_mismatches': junction_bad,
        'verified_junctions_expected_from_raw_osm': len(expected),
        'verified_junctions_match_raw_osm': {int(j['node']) for j in verified} == expected,
        'separated_without_level_difference': sep_bad,
        'source_node_position_conflicts': src.conflicts,
        'route': route_check(geo),
        'pass': (not order_bad and not junction_bad and not sep_bad and {int(j['node']) for j in verified} == expected
                 and src.conflicts == 0),
    }


def route_check(geo) -> dict:
    """A graph-level check: the road graph whose vertices are verified junctions (edges: consecutive verified
    junctions along a road), its components, and a shortest route between the junctions nearest two opposite
    corners of the area."""
    verified = {j['node'] for j in geo['navigation']['junctions'] if j['basis'] == 'osm_node'}
    point = {j['node']: j['point'] for j in geo['navigation']['junctions'] if j['basis'] == 'osm_node'}
    adj = defaultdict(dict)
    for r in geo['roads']:
        if r['nodes'] is None:
            continue
        s, last = 0.0, None
        for k, (p, n) in enumerate(zip(r['centerline'], r['nodes'])):
            if k:
                s += math.dist(r['centerline'][k - 1], p)
            if n in verified:
                if last is not None and last[0] != n:
                    w = s - last[1]
                    if w < adj[last[0]].get(n, (math.inf,))[0]:
                        adj[last[0]][n] = adj[n][last[0]] = (w, r['id'])
                last = (n, s)
    seen, comps = set(), []
    for v in sorted(verified):
        if v in seen:
            continue
        comp, queue = [], deque([v])
        seen.add(v)
        while queue:
            u = queue.popleft()
            comp.append(u)
            for w in adj[u]:
                if w not in seen:
                    seen.add(w)
                    queue.append(w)
        comps.append(comp)
    main = max(comps, key=len) if comps else []
    half = geo['metadata']['bounds']['max_m'][0]
    a = min(main, key=lambda n: (math.dist(point[n], (-half, -half)), n))
    b = min(main, key=lambda n: (math.dist(point[n], (half, half)), n))
    # Dijkstra by length.
    dist, prev, heap = {a: 0.0}, {}, [(0.0, a)]
    while heap:
        d, u = heapq.heappop(heap)
        if u == b:
            break
        if d > dist[u]:
            continue
        for w, (length, _) in sorted(adj[u].items()):
            nd = d + length
            if nd < dist.get(w, math.inf):
                dist[w], prev[w] = nd, u
                heapq.heappush(heap, (nd, w))
    path = [b]
    while path[-1] != a:
        path.append(prev[path[-1]])
    path.reverse()
    roads_used = []
    for u, w in zip(path, path[1:]):
        rid = adj[u][w][1]
        if not roads_used or roads_used[-1] != rid:
            roads_used.append(rid)
    span = math.dist(point[a], point[b])
    return {'graph_vertices': len(verified), 'graph_edges': sum(len(v) for v in adj.values()) // 2, 'components': len(comps),
            'largest_component': len(main), 'from': f'j:n{a}', 'to': f'j:n{b}', 'straight_m': round(span, 1),
            'route_m': round(dist[b], 1), 'junctions_passed': len(path), 'roads_used': len(roads_used),
            'pass': len(path) >= THRESHOLDS['route_min_junctions'] and span >= THRESHOLDS['route_min_span_m']}


# ------------------------------------------------------------------------------------------ buildings
def building_report(geo, src: Source, place: dict) -> dict:
    bs = geo['buildings']
    prov = lambda key: Counter(b['provenance'][key] for b in bs)        # noqa: E731
    biggest = max(bs, key=lambda b: (b['area_m2'], b['id']))
    widest = max(bs, key=lambda b: (len(b['chunks']), b['area_m2'], b['id']))
    # Courtyards: source buildings with inner rings (multipolygons), clipped to the play area as prepare_city clips
    # them (a courtyard the edge cuts through is open there, no longer a hole), against OutbreakGeo's holes.
    half = place['size_m'] / 2
    inner = box(-half, -half, half, half).buffer(-0.5)
    reader = prepare_city.OSM({'elements': src.elements}, Projection(*place['center'], 'wgs84'))
    courtyards = []
    for el in src.elements:
        if el.get('type') == 'relation' and 'building' in (el.get('tags') or {}) and any(m.get('role') == 'inner' for m in el.get('members') or []):
            ids = [b for b in bs if b['osm'] == {'type': 'relation', 'id': el['id']}]
            g = reader.area(el)
            if ids and g is not None:
                pieces = prepare_city.clean(g.intersection(inner), 4.0)
                courtyards.append({'id': f'r{el["id"]}', 'holes_in_source_play_area': sum(len(p.interiors) for p in pieces),
                                   'holes_in_outbreakgeo': sum(len(b['footprint']['holes']) for b in ids)})
    raised = [b['id'] for b in bs if b['min_height_m'] is not None and b['min_height_m'] > adapter.FLOATING_M]
    in_collision = {m for c in geo['navigation']['collision'] for m in c['buildings']}
    return {
        'total': len(bs), 'by_part': dict(Counter(b['part'] for b in bs)), 'landmarks': sum(1 for b in bs if b['landmark']),
        'landmark_names': sorted(b['name'] for b in bs if b['landmark']),
        'wall_height': dict(prov('wall_height')), 'levels': dict(prov('levels')), 'roof_shape': dict(prov('roof_shape')),
        'roof_material_mapped': prov('roof_material').get('osm', 0), 'roof_colour_mapped': prov('roof_colour').get('osm', 0),
        'facade_material_mapped': prov('facade_material').get('osm', 0),
        'largest_footprint': {'id': biggest['id'], 'name': biggest['name'], 'kind': biggest['kind'], 'area_m2': biggest['area_m2']},
        'largest_chunk_span': {'id': widest['id'], 'name': widest['name'], 'chunks': len(widest['chunks'])},
        'courtyards': courtyards, 'courtyards_kept': all(c['holes_in_outbreakgeo'] == c['holes_in_source_play_area'] for c in courtyards),
        'raised': raised, 'raised_in_ground_collision': sorted(set(raised) & in_collision),
        'pass': all(c['holes_in_outbreakgeo'] == c['holes_in_source_play_area'] for c in courtyards) and not (set(raised) & in_collision),
    }


# ------------------------------------------------------------------------------------------ scale
def scale_report(geo, src: Source) -> dict:
    lat0, lon0 = geo['metadata']['center']['lat'], geo['metadata']['center']['lon']
    (x0, y0), (x1, y1) = geo['metadata']['bounds']['min_m'], geo['metadata']['bounds']['max_m']
    inside = lambda p: x0 <= p[0] < x1 and y0 <= p[1] < y1          # noqa: E731
    roads = [r for r in geo['roads'] if r['nodes'] is not None]
    node_xy = {}
    for r in roads:
        for p, n in zip(r['centerline'], r['nodes']):
            node_xy[n] = p
    out = []

    def measure(kind, label, a, b, path=None):
        """A distance between OSM nodes a and b (or along the node path): geodesic on the raw lat/lon against
        Euclidean in OutbreakGeo."""
        nodes = path or [a, b]
        src_m = sum(geodesic(*src.node_ll[p], *src.node_ll[q])[0] for p, q in zip(nodes, nodes[1:]))
        geo_m = sum(math.dist(node_xy[p], node_xy[q]) for p, q in zip(nodes, nodes[1:]))
        err = abs(geo_m - src_m)
        out.append({'id': f'm{len(out) + 1:02d}', 'kind': kind, 'source': label, 'source_m': round(src_m, 4), 'outbreakgeo_m': round(geo_m, 4),
                    'abs_error_m': round(err, 4), 'pct_error': round(100 * err / src_m, 5) if src_m else 0.0,
                    'tolerance_m': round(tolerance(src_m), 4), 'pass': err <= tolerance(src_m)})

    def quadrant(p):
        return ('N' if p[1] >= 0 else 'S') + ('E' if p[0] >= 0 else 'W')
    # 1. Road segments: in each quadrant, the longest straight segment of two different road classes.
    segs = defaultdict(list)
    for r in roads:
        for (p, n), (q, m) in zip(zip(r['centerline'], r['nodes']), list(zip(r['centerline'], r['nodes']))[1:]):
            if inside(p) and inside(q) and n != m:
                segs[quadrant(((p[0] + q[0]) / 2, (p[1] + q[1]) / 2))].append((math.dist(p, q), r['class'], r['id'], n, m))
    for qd in ('NE', 'NW', 'SE', 'SW'):
        used = set()
        for length, cls, rid, n, m in sorted(segs[qd], key=lambda s: (-s[0], s[2], s[3])):
            if cls not in used:
                used.add(cls)
                measure('road segment', f'way {rid[1:]} nodes {n}-{m} ({cls}, {qd})', n, m)
            if len(used) == 2:
                break
    # 2. Whole roads along their nodes: the four longest roads lying wholly inside the area.
    whole = sorted((r for r in roads if all(inside(p) for p in r['centerline'])), key=lambda r: (-r['length_m'], r['id']))[:4]
    for r in whole:
        measure('road length', f'way {r["id"][1:]} along {len(r["nodes"])} nodes ({r["class"]})', None, None, path=r['nodes'])
    # 3. Across the area: verified junctions nearest opposite corners and opposite edge midpoints.
    js = [j for j in geo['navigation']['junctions'] if j['basis'] == 'osm_node']
    near = lambda t: min(js, key=lambda j: (math.dist(j['point'], t), j['node']))['node']   # noqa: E731
    h = x1 * 0.9
    for (ax, ay), (bx, by), label in (((-h, -h), (h, h), 'SW-NE'), ((-h, h), (h, -h), 'NW-SE'), ((-h, 0), (h, 0), 'W-E'), ((0, -h), (0, h), 'S-N')):
        a, b = near((ax, ay)), near((bx, by))
        measure('across the area', f'junction nodes {a}-{b} ({label})', a, b)
    # 4. Building edges: six buildings (largest first, at most two per quadrant, distinct kinds), the longest edge
    # whose two OSM nodes are both footprint vertices next to each other in OutbreakGeo.
    picked, per_quadrant, kinds = 0, Counter(), set()
    for b in sorted(geo['buildings'], key=lambda b: (-b['area_m2'], b['id'])):
        if picked == 6:
            break
        if b['osm'] is None or b['osm']['type'] != 'way' or b['part'] != 'building' or '#' in b['id']:
            continue
        way = src.ways.get(b['osm']['id'])
        c = Polygon(b['footprint']['outer']).centroid
        qd = quadrant((c.x, c.y))
        if way is None or per_quadrant[qd] >= 2 or b['kind'] in kinds:
            continue
        ring = b['footprint']['outer']
        ref = [reference_xy(lat0, lon0, *src.node_ll[n]) for n in way['nodes']]
        match = []
        for n, (rx, ry) in zip(way['nodes'], ref):
            k = min(range(len(ring)), key=lambda i: math.dist(ring[i], (rx, ry)))
            match.append((n, k if math.dist(ring[k], (rx, ry)) <= 0.10 else None))
        best = None
        for (n, k), (m, kk) in zip(match, match[1:]):
            if k is not None and kk is not None and n != m and (kk - k) % len(ring) in (1, len(ring) - 1):
                length = math.dist(ring[k], ring[kk])
                if best is None or length > best[0]:
                    best = (length, n, m, k, kk)
        if best is None:
            continue
        _, n, m, k, kk = best
        node_xy.setdefault(n, ring[k])
        node_xy.setdefault(m, ring[kk])
        if node_xy[n] != ring[k] or node_xy[m] != ring[kk]:
            continue                          # a node also on a road elsewhere: keep the road's position
        measure('building edge', f'way {b["osm"]["id"]} nodes {n}-{m} ({b["kind"]}, {qd})', n, m)
        picked += 1
        per_quadrant[qd] += 1
        kinds.add(b['kind'])
    # 5. Widths mapped in OSM (a width tag kept as the road's width: provenance osm).
    for r in sorted((r for r in geo['roads'] if r['provenance']['width'] == 'osm' and 'width' in r['tags']), key=lambda r: r['id'])[:2]:
        src_m, geo_m = _num(r['tags']['width']), r['width_m']
        err = abs(geo_m - src_m)
        out.append({'id': f'm{len(out) + 1:02d}', 'kind': 'mapped width', 'source': f'way {r["id"][1:]} width={r["tags"]["width"]} ({r["class"]})',
                    'source_m': src_m, 'outbreakgeo_m': geo_m, 'abs_error_m': round(err, 4), 'pct_error': round(100 * err / src_m, 5),
                    'tolerance_m': round(tolerance(src_m), 4), 'pass': err <= tolerance(src_m)})
    # Every road node's position against the azimuthal-equidistant reference: offset, rotation, scale.
    res, pairs = [], []
    for n, (x, y) in sorted(node_xy.items()):
        if inside((x, y)):
            rx, ry = reference_xy(lat0, lon0, *src.node_ll[n])
            res.append((x - rx, y - ry))
            pairs.append(((rx, ry), (x, y)))
    sxx = sum(rx * x + ry * y for (rx, ry), (x, y) in pairs)
    sxy = sum(rx * y - ry * x for (rx, ry), (x, y) in pairs)
    srr = sum(rx * rx + ry * ry for (rx, ry), _ in pairs)
    rotation = math.atan2(sxy, sxx)
    scale = math.hypot(sxx, sxy) / srr
    mean = (sum(d[0] for d in res) / len(res), sum(d[1] for d in res) / len(res))
    position = {'points': len(res), 'max_m': round(max(math.hypot(*d) for d in res), 4), 'mean_m': round(sum(math.hypot(*d) for d in res) / len(res), 4),
                'mean_offset_m': [round(mean[0], 4), round(mean[1], 4)], 'rotation_rad': rotation, 'rotation_deg': math.degrees(rotation),
                'scale': scale}
    position['pass'] = (position['max_m'] <= THRESHOLDS['position_max_m'] and math.hypot(*mean) <= THRESHOLDS['position_mean_offset_m']
                        and abs(rotation) <= THRESHOLDS['rotation_rad'] and abs(scale - 1) <= THRESHOLDS['scale_dev'])
    worst_abs = max(out, key=lambda m: m['abs_error_m'])
    worst_pct = max(out, key=lambda m: m['pct_error'])
    return {'tolerance': 'max(0.10 m, 0.1 % of the length)', 'measurements': out, 'count': len(out),
            'worst_abs': {'id': worst_abs['id'], 'abs_error_m': worst_abs['abs_error_m']},
            'worst_pct': {'id': worst_pct['id'], 'pct_error': worst_pct['pct_error']},
            'position': position,
            'pass': len(out) >= THRESHOLDS['min_measurements'] and all(m['pass'] for m in out) and position['pass']}


# ------------------------------------------------------------------------------------------ ground areas
# prepare_city's ground layers that OutbreakGeo carries as area classes, by its own tag rules (the area_union
# calls in prepare_city.main), and their priority there, highest first: where layers meet, the higher keeps the
# ground. Pier (above pedestrian ground) and rail bed and sand (above pitch) rank among them but aren't carried.
GROUND_RULES = {
    'parking': lambda t: t.get('amenity') == 'parking' and t.get('parking') not in ('underground', 'multi-storey', 'rooftop'),
    'pitch': lambda t: t.get('leisure') in ('pitch', 'golf_course', 'track', 'stadium'),
    'park': lambda t: t.get('leisure') in ('park', 'garden', 'common', 'recreation_ground', 'dog_park', 'nature_reserve'),
    'grass': lambda t: (t.get('landuse') in ('grass', 'village_green', 'meadow', 'flowerbed', 'recreation_ground', 'greenfield', 'allotments',
                                            'orchard', 'plant_nursery', 'vineyard') or t.get('natural') in ('grassland', 'heath', 'scrub', 'fell')),
    'cemetery': lambda t: t.get('landuse') == 'cemetery' or t.get('amenity') == 'grave_yard',
    'forest': lambda t: t.get('landuse') == 'forest' or t.get('natural') in ('wood', 'tree_group'),
}
RANK = ('carriageway', 'sidewalk', 'pedestrian', 'path', 'parking', 'pitch', 'park', 'grass', 'cemetery', 'forest')
SIMPLIFY_M = 0.8        # prepare_city simplifies each ground layer by up to this much before writing it ...
DROPPED_M2 = 25.0       # ... and drops pieces smaller than this
FABRICATED_M2 = 1.0     # area of a class allowed outside its expected source (+ SIMPLIFY_M): numerical noise only
COMPONENT_M2 = 2 * DROPPED_M2   # an expected layer piece this large can't be one prepare_city drops ...
COVER_MIN = 0.5         # ... so OutbreakGeo must keep at least this share of it (simplifying moves edges, never drops a ring)


def road_ground(osm_reader, B: Polygon) -> list[tuple]:
    """The ground prepare_city derives from the highway ways, by its own rules (prepare_city.main, "roads"):
    [(class, label, polygon)] for each way's strip (carriageway, pedestrian or path by class and surface), each
    sidewalk band, each mapped road area and square, and each platform."""
    num = _num
    out = []
    for el, t, g in osm_reader.lines(lambda t: 'highway' in t):
        hw, surf = t['highway'], t.get('surface', '')
        if t.get('area') == 'yes' or hw in ('proposed', 'construction', 'corridor', 'elevator', 'bus_stop'):
            continue
        if t.get('tunnel') in ('yes', 'culvert') or (t.get('layer', '0').startswith('-') and t.get('tunnel')):
            continue
        if not g.intersects(B.buffer(20)):
            continue
        label = f'way {el["id"]}'
        if hw in prepare_city.CAR_WIDTH:
            w = num(t.get('width')) or (num(t.get('lanes')) * 3.2 + 0.6 if num(t.get('lanes')) else None) or prepare_city.CAR_WIDTH[hw]
            if hw == 'service':
                w = {'parking_aisle': 5.0, 'driveway': 3.2, 'alley': 3.6}.get(t.get('service'), w)
            w = max(2.6, min(w, 22))
            cls = 'pedestrian' if hw == 'pedestrian' else 'carriageway' if surf in prepare_city.COBBLE_SURFACES else \
                'path' if surf in prepare_city.SOFT_SURFACES or hw == 'track' else 'carriageway'
            out.append((cls, label, g.buffer(w / 2, cap_style=2, join_style=2)))
            sw = t.get('sidewalk', t.get('sidewalk:both', ''))
            if hw in prepare_city.SIDEWALK_CLASSES and sw not in ('no', 'none', 'separate') and t.get('sidewalk:both') != 'separate':
                swidth = num(t.get('sidewalk:width')) or 1.9
                for s in {'left': [1], 'right': [-1]}.get(sw, [1, -1]):
                    try:
                        band = g.buffer(s * (w / 2 + swidth), single_sided=True, cap_style=2, join_style=2)
                    except Exception:
                        continue
                    out.append(('sidewalk', f'{label} sidewalk {"left" if s > 0 else "right"}', band))
        elif hw in prepare_city.SOFT_WIDTH:
            w = max(1.4, min(num(t.get('width')) or prepare_city.SOFT_WIDTH[hw], 8))
            cls = 'path' if surf in prepare_city.SOFT_SURFACES or (hw in ('path', 'bridleway') and not surf) else 'pedestrian'
            out.append((cls, label, g.buffer(w / 2, cap_style=2, join_style=2)))
    for el, t, g in osm_reader.areas(lambda t: (t.get('highway') in ('pedestrian', 'footway', 'service', 'residential', 'unclassified')
                                                and t.get('area') == 'yes') or t.get('place') == 'square' or 'area:highway' in t
                                     or t.get('amenity') == 'marketplace'):
        kind = t.get('area:highway') or t.get('highway') or 'pedestrian'
        car = kind in ('residential', 'unclassified', 'service', 'secondary', 'primary', 'tertiary') and t.get('surface', '') not in prepare_city.COBBLE_SURFACES
        out.append(('carriageway' if car else 'pedestrian', f'{el["type"]} {el["id"]} (road area)', g))
    for el, t, g in osm_reader.areas(lambda t: t.get('railway') == 'platform' or t.get('public_transport') == 'platform'):
        out.append(('pedestrian', f'{el["type"]} {el["id"]} (platform)', g))
    return out


def area_report(geo, osm_reader, B: Polygon) -> dict:
    """Ground areas against the source, both ways, by prepare_city's documented rules.

    Expected: every mapped area of a carried class (parks, grass, car parks, pitches, cemeteries, woods), every
    water area, and the ground prepare_city derives from the highway ways (road_ground: carriageway, sidewalk,
    pedestrian ground and paths). Lost: inside the play area, such an element's ground must be OutbreakGeo ground
    of its class or of a class prepare_city ranks higher, building or city wall collision (buildings take the
    ground), water (land classes are cut to the land) or a higher layer OutbreakGeo doesn't carry (pier, rail
    bed, sand). What is left over is lost unless prepare_city's own rules explain it: simplifying moves a layer's
    edges by at most SIMPLIFY_M (and keeps every ring), so ground it vacates lies within SIMPLIFY_M of an edge
    OutbreakGeo kept (of the class or one ranked higher); and pieces under DROPPED_M2 are dropped. So a leftover
    piece of DROPPED_M2 or more beyond SIMPLIFY_M from every kept edge is ground lost: a narrow path removed whole
    leaves no kept edge near it.
    Whole pieces, a second line: prepare_city's layers are rebuilt from the expected elements by its own priority
    chain, and every rebuilt piece of COMPONENT_M2 or more (twice what it drops) must be kept, at least COVER_MIN of
    it, as its class or one ranked higher.
    Fabricated: no class may cover ground outside its expected elements (grass: plus the lawns prepare_city draws
    round small houses), widened by SIMPLIFY_M, by more than FABRICATED_M2."""
    def union(shapes):
        shapes = [g for g in shapes if g is not None and not g.is_empty]
        return unary_union(shapes) if shapes else Polygon()

    def pieces(g):
        return [] if g is None or g.is_empty else [p for p in shapely.get_parts(g) if p.geom_type == 'Polygon' and p.area > 0]

    # OutbreakGeo's ground and what else takes ground, each piece with its rank in prepare_city's priority (water and
    # collision above every land class; pier, rail bed and sand at their places among the layers), looked up locally.
    pier = union([g for _, _, g in osm_reader.areas(lambda t: t.get('man_made') in ('pier', 'breakwater', 'quay'))] +
                 [g.buffer((_num(t.get('width')) or 3.0) / 2, cap_style=2)
                  for _, t, g in osm_reader.lines(lambda t: t.get('man_made') in ('pier', 'breakwater', 'quay')) if not g.is_closed])
    rail = union(g.buffer(1.7, cap_style=2) for _, t, g in osm_reader.lines(lambda t: t.get('railway') in ('rail', 'light_rail', 'narrow_gauge'))
                 if not t.get('tunnel'))
    sand = union(g for _, _, g in osm_reader.areas(lambda t: t.get('natural') in ('beach', 'sand') or t.get('leisure') == 'playground'
                                                        or t.get('landuse') == 'sand'))
    PIER_R, BED_R = RANK.index('sidewalk') + 0.5, RANK.index('parking') + 0.5
    ranked = [(RANK.index(a['class']) if a['class'] in RANK else -1, a['class'], Polygon(a['polygon']['outer'], a['polygon']['holes']))
              for a in geo['areas']]
    ranked += [(-1, 'collision', Polygon(c['polygon']['outer'], c['polygon']['holes'])) for c in geo['navigation']['collision']]
    ranked += [(PIER_R, 'pier', p) for p in pieces(pier)] + [(BED_R, 'rail', p) for p in pieces(rail)] + [(BED_R, 'sand', p) for p in pieces(sand)]
    tree = STRtree([p for _, _, p in ranked])

    def around(g, test):
        return union(ranked[i][2] for i in tree.query(g) if test(*ranked[i][:2]))

    by_class = defaultdict(list)
    for r, kind, p in ranked:
        by_class[kind].append(p)
    water, collision = union(by_class['water']), union(by_class['collision'])
    # Expected elements, by class.
    elements = defaultdict(list)
    for cls, rule in GROUND_RULES.items():
        elements[cls] += [(f'{el["type"]} {el["id"]}', g) for el, t, g in osm_reader.areas(rule)]
    for cls, label, g in road_ground(osm_reader, B):
        elements[cls].append((label, g))
    elements['water'] += [(f'{el["type"]} {el["id"]}', g) for el, t, g in osm_reader.areas(
        lambda t: t.get('natural') in ('water', 'bay', 'strait') or 'water' in t or t.get('waterway') in ('riverbank', 'dock', 'boatyard')
        or t.get('landuse') in ('reservoir', 'basin')) if not (t.get('tunnel') or t.get('covered') == 'yes' or t.get('natural') == 'wetland')]
    elements['water'] += [(f'{el["type"]} {el["id"]}', g.buffer(
        (_num(t.get('width')) or {'river': 14, 'canal': 9, 'stream': 2.5, 'ditch': 1.4, 'drain': 1.2}[t['waterway']]) / 2, cap_style=2))
        for el, t, g in osm_reader.lines(lambda t: t.get('waterway') in ('river', 'canal', 'stream', 'ditch', 'drain'))
        if not (t.get('tunnel') in ('culvert', 'yes') or t.get('layer', '0').startswith('-'))]
    lawns = [Polygon(b['footprint']['outer'], b['footprint']['holes']).buffer(12.0).intersection(B) for b in geo['buildings']
             if b['kind'] in prepare_city.HOUSE_KINDS or (b['kind'] in ('yes', 'residential') and b['area_m2'] < 221 and b['levels'] <= 2)]
    clipped = {cls: [(label, g.intersection(B)) for label, g in items] for cls, items in elements.items()}
    clipped = {cls: [(label, g) for label, g in items if not g.is_empty and g.area > 0] for cls, items in clipped.items()}
    # prepare_city's layers rebuilt from the expected elements by its own chain (prepare_city.main, "ground layers"):
    # roads first (carriageway, then sidewalk bands outside the roadways, paving outside both, paths outside all
    # hard ground), each layer cut to the land, minus the buildings' ground and every layer ranked above it.
    # OutbreakGeo's water and collision stand in for the land and the buildings, and road layers are cut to the
    # land too (bridge decks over water are left out of the expectation, never wrongly expected).
    src = {c: union(g for _, g in clipped.get(c, [])) for c in RANK}
    roadways = src['carriageway']
    chain = {'carriageway': roadways, 'sidewalk': src['sidewalk'].difference(roadways)}
    chain['pedestrian'] = src['pedestrian'].difference(roadways).difference(chain['sidewalk'])
    chain['path'] = src['path'].difference(roadways).difference(chain['sidewalk']).difference(chain['pedestrian'])
    for c in ('parking', 'pitch', 'park', 'cemetery', 'forest'):
        chain[c] = src[c]
    chain['grass'] = union([src['grass']] + lawns)
    order = [('carriageway', chain['carriageway']), ('sidewalk', chain['sidewalk']), ('pier', pier), ('pedestrian', chain['pedestrian']),
             ('path', chain['path']), ('parking', chain['parking']), ('rail', rail), ('sand', sand)] + \
            [(c, chain[c]) for c in ('pitch', 'park', 'grass', 'cemetery', 'forest')]
    taken = union([water, collision])
    expected, above = {}, []
    for name, g in order:
        g = g.difference(taken)
        for prev in above:
            if g.is_empty:
                break
            g = g.difference(prev)
        expected[name] = g
        above.append(g)
    out, ok = {}, True
    for cls in RANK + ('water',):
        r = -1 if cls == 'water' else RANK.index(cls)
        if cls == 'water':
            def accounted(rank, kind):
                return kind in ('water', 'pier')

            def kept(rank, kind):
                return kind == 'water'
        else:
            def accounted(rank, kind, r=r):
                return kind in ('water', 'collision') or (kind in RANK and rank <= r) or (kind in ('pier', 'rail', 'sand') and rank < r)

            def kept(rank, kind, r=r):
                return kind in RANK and rank <= r
        lost = []
        for label, piece in clipped.get(cls, []):
            # Where simplifying may have moved an edge: within SIMPLIFY_M of an edge OutbreakGeo kept (its class's or a
            # higher class's). Ground missing anywhere else is lost, unless it is a piece prepare_city drops.
            missing = piece.difference(around(piece, accounted))
            if missing.is_empty or missing.area < DROPPED_M2:
                continue                                        # no piece of it can be lost
            reach = missing.buffer(SIMPLIFY_M + 0.01)
            near = around(reach, kept).boundary.intersection(reach)          # the kept edges that could explain it
            rest = missing.difference(near.buffer(SIMPLIFY_M)) if not near.is_empty else missing
            big = [c for c in pieces(rest) if c.area >= DROPPED_M2]
            if big:
                lost.append({'source': label, 'source_m2': round(piece.area, 1), 'missing_m2': round(missing.area, 1),
                             'beyond_edges_m2': round(sum(c.area for c in big), 1)})
        # Fabricated: OutbreakGeo ground of the class outside its expected elements (+ SIMPLIFY_M).
        allowed = [g.buffer(SIMPLIFY_M + 0.2) for _, g in clipped.get(cls, [])] + ([g.buffer(SIMPLIFY_M + 0.2) for g in lawns] if cls == 'grass' else [])
        allowed_tree = STRtree(allowed) if allowed else None
        excess = 0.0
        for p in by_class[cls]:
            near = union(allowed[i] for i in allowed_tree.query(p)) if allowed_tree is not None else Polygon()
            excess += p.difference(near).area
        # Whole pieces: every expected layer piece too large to be dropped must be there, at least COVER_MIN of it
        # (its class or one ranked higher).
        missing_pieces, cover = [], []
        if cls != 'water':
            for piece in pieces(expected[cls]):
                if piece.area < COMPONENT_M2:
                    continue
                share = piece.intersection(around(piece, kept)).area / piece.area
                cover.append(share)
                if share < COVER_MIN:
                    c = piece.representative_point()
                    missing_pieces.append({'at': [round(c.x, 1), round(c.y, 1)], 'expected_m2': round(piece.area, 1), 'kept_share': round(share, 3)})
        out[cls] = {'source_elements': len(clipped.get(cls, [])), 'source_m2': round(src[cls].area if cls in src else union(g for _, g in clipped.get(cls, [])).area),
                    'outbreakgeo_m2': round(sum(p.area for p in by_class[cls])),
                    'lost': lost, 'expected_pieces': len(cover), 'least_kept_share': round(min(cover), 3) if cover else None,
                    'missing_pieces': missing_pieces, 'fabricated_m2': round(excess, 3)}
        ok &= not lost and not missing_pieces and (cls == 'water' or excess <= FABRICATED_M2)
    return {'rule': (f'every expected ground element (mapped areas, water, and the carriageway, sidewalks, pedestrian ground and paths '
                     f'prepare_city derives from the highways) kept by its class or one ranked higher, a leftover piece of {DROPPED_M2:g} m² '
                     f'or more beyond {SIMPLIFY_M} m from every kept edge being lost; every expected layer piece of '
                     f'{COMPONENT_M2:g} m² or more at least {COVER_MIN:.0%} kept; no class outside its expected '
                     f'elements (+{SIMPLIFY_M} m) by more than {FABRICATED_M2:g} m²'),
            'classes': out, 'pass': bool(ok)}


# ------------------------------------------------------------------------------------------ completeness
def completeness_report(geo, src: Source, place: dict) -> dict:
    """What the source has in the play area against what OutbreakGeo keeps; every source feature is either
    kept or excluded by a named rule (prepare_city's or the adapter's)."""
    proj = Projection(*place['center'], 'wgs84')
    half = place['size_m'] / 2
    B = box(-half, -half, half, half)
    osm_reader = prepare_city.OSM({'elements': src.elements}, proj)
    road_ids = {r['osm']['id'] for r in geo['roads'] if r['osm']}
    grid = Grid((-half, -half, half, half), geo['metadata']['chunking']['size_m'])
    fates = defaultdict(Counter)
    unexplained = []
    for el, t, g in osm_reader.lines(lambda t: 'highway' in t):
        if not g.intersects(B):
            continue
        hw = t['highway']
        group = 'drivable' if hw in adapter.DRIVABLE else 'pedestrian_and_other'
        if el['id'] in road_ids:
            fate = 'kept'
        elif t.get('area') == 'yes':
            fate = 'excluded: area=yes (a road area, kept as ground)'
        elif hw in ('proposed', 'construction', 'corridor', 'elevator', 'bus_stop'):
            fate = f'excluded: highway={hw} (not built or indoors)'
        elif t.get('tunnel') in ('yes', 'culvert') or (t.get('layer', '0').startswith('-') and t.get('tunnel')):
            fate = 'excluded: tunnel (underground)'
        elif hw not in prepare_city.CAR_WIDTH and hw not in prepare_city.SOFT_WIDTH:
            fate = f'excluded: highway={hw} (a class prepare_city has no width for)'
        elif not grid.line_spans([[prepare_city.r2(x), prepare_city.r2(y)] for x, y in g.coords]):
            fate = 'excluded: no part inside the half-open play area (it runs along or ends on its edge)'
        else:
            fate = 'unexplained'
            unexplained.append(f'way {el["id"]}')
        fates[group][fate] += 1
    # Buildings: outlines (building=*) and parts (building:part), as prepare_city reads them.
    geo_refs = Counter((b['osm']['type'], b['osm']['id']) for b in geo['buildings'] if b['osm'])
    col_refs = {(c['osm']['type'], c['osm']['id']) for c in geo['navigation']['collision'] if c['osm']}
    inner = B.buffer(-0.5)
    bfates = defaultdict(Counter)
    for el, t, g in osm_reader.areas(lambda t: 'building' in t or 'building:part' in t):
        if not g.intersects(B):
            continue
        group = 'outlines' if 'building' in t else 'parts'
        key = (el['type'], el['id'])
        built = 'building' in t and t['building'] not in adapter.NOT_BUILT or 'building' not in t and t.get('building:part') != 'no'
        if key in geo_refs:
            fate = 'kept as a building'
        elif key in col_refs:
            fate = 'kept as collision (its parts are the buildings)'
        elif not built:
            fate = f'excluded: building={t.get("building")} (not a standing building)'
        elif not prepare_city.clean(g.intersection(inner), 4.0):
            fate = 'excluded: under 4 m² inside the play area (prepare_city)'
        else:
            fate = 'unexplained'
            unexplained.append(f'{el["type"]} {el["id"]}')
        bfates[group][fate] += 1
    # Ground areas, mapped and derived from the roads, against the source both ways.
    areas = area_report(geo, osm_reader, B)
    barriers = Counter(t['barrier'] for el, t, g in osm_reader.lines(lambda t: t.get('barrier')) if g.intersects(B))
    on_roads = {n for r in geo['roads'] if r['nodes'] for n in r['nodes']}
    barrier_nodes = Counter(el['tags']['barrier'] for el in src.elements
                            if el.get('type') == 'node' and (el.get('tags') or {}).get('barrier') and el['id'] in on_roads)
    rails = sum(1 for el, t, g in osm_reader.lines(lambda t: t.get('railway') in ('rail', 'tram', 'light_rail', 'narrow_gauge')) if g.intersects(B))
    sidewalk_ways = sum(1 for el, t, g in osm_reader.lines(lambda t: t.get('footway') == 'sidewalk') if g.intersects(B))
    excluded = {
        'barriers': {'source': dict(barriers), 'outbreakgeo': sum(1 for c in geo['navigation']['collision'] if c['kind'] == 'city_wall'),
                     'rule': 'Only city walls are carried (as collision); mapped walls, fences, hedges and retaining walls are not yet.'},
        'barrier nodes on roads': {'source': dict(barrier_nodes), 'outbreakgeo': 0,
                                   'rule': 'Bollards, gates and the like on a road are nodes of its way; their tags are not carried yet '
                                           '(the road keeps the node id, so a later phase can add them).'},
        'railways': {'source': rails, 'outbreakgeo': 0, 'rule': 'Rails and tram lines are not carried yet (ground under them is).'},
        'trees, street furniture, edge containers, generated hedges': {'rule': 'Wasteland decoration; dropped by the adapter.'},
    }
    return {'roads': {k: dict(v) for k, v in fates.items()}, 'buildings': {k: dict(v) for k, v in bfates.items()},
            'areas': areas, 'separately_mapped_sidewalk_ways': sidewalk_ways, 'not_carried': excluded,
            'unexplained': unexplained, 'pass': not unexplained and areas['pass']}


# ------------------------------------------------------------------------------------------ terrain, provenance
def terrain_report(geo, terrain_src: dict, place: dict) -> dict:
    t = geo['terrain']
    half = place['size_m'] / 2
    g = t['grid']
    n, step, (tx, ty) = int(g['n']), g['step_m'], g['origin_m']
    h = t['heights_m']
    covers = adapter.raster_covers((-half, -half, half, half), tx, ty, step, n)
    zs = terrain_src['z']
    sn, ss, sx0 = terrain_src['n'], terrain_src['step'], terrain_src['x0']
    play = [zs[j * sn + i] for j in range(sn) for i in range(sn)
            if abs(sx0 + i * ss) <= half and abs(terrain_src['y0'] + j * ss) <= half]
    cov = terrain_src.get('coverage') or {}
    return {
        'source': t['source'], 'kind': t['kind'], 'attribution': t['attribution'], 'tiles': terrain_src.get('tiles'),
        'outbreakgeo_grid': {'n': n, 'step_m': step, 'origin_m': [tx, ty], 'extent_m': [tx, tx + (n - 1) * step]},
        'source_grid': {'n': sn, 'step_m': ss, 'extent_m': [sx0, sx0 + (sn - 1) * ss]},
        'covers_play_area': covers, 'non_finite_heights': sum(1 for v in h if not math.isfinite(v)),
        'range_in_bounds_m': t['range_in_bounds_m'], 'source_elevation_in_play_area_m': [min(play), max(play)],
        'vertical_datum': t['vertical_datum'], 'coverage': cov,
        'support_no_data_cells': cov.get('support_no_data_cells', cov.get('no_data_cells')),
        'pass': (covers and all(math.isfinite(v) for v in h) and cov.get('play_area_no_data_cells') == 0 and cov.get('play_area_fallback_cells') == 0
                 and cov.get('support_no_data_cells', cov.get('no_data_cells')) == 0),
    }


def provenance_report(geo, manifest: dict) -> dict:
    p = geo['provenance']
    inputs = p['sources']['inputs']
    pinned = {name: inputs[key]['sha256'] == manifest['files'][name]['sha256']
              for key, name in (('place', 'place.json'), ('osm', 'osm.json'), ('terrain', 'terrain.json'))}
    lic = {x['data']: x['license'] for x in p['licenses']}
    text = adapter.dumps(geo).lower()
    return {
        'attribution': p['attribution'], 'licenses': lic, 'inputs_match_snapshot': pinned,
        'osm_timestamp': inputs['osm']['timestamp_osm_base'], 'terrain_tiles': inputs['terrain']['tiles'],
        'clean_rebuild': inputs['city']['clean_rebuild'], 'street_view_policy': p['policy']['street_view'],
        'snapshot_street_view_tags': manifest['osm']['street_view_tags'],
        'google_mentions_outside_policy': text.count('google') - p['policy']['street_view'].lower().count('google')
                                          - p['policy']['wasteland_refinements'].lower().count('google'),
        'pass': (all(pinned.values()) and lic.get('OpenStreetMap') == 'ODbL-1.0' and inputs['city']['clean_rebuild']
                 and not manifest['osm']['street_view_tags'] and not geo['metadata']['synthetic']),
    }


def report(geo, manifest, files) -> dict:
    place = json.loads(files['place.json'])
    osm = json.loads(files['osm.json'].decode('utf-8'))
    terrain_src = json.loads(files['terrain.json'].decode('utf-8'))
    src = Source(osm)
    nav = geo['navigation']
    return {
        'counts': {'chunks': len(geo['chunks']), 'buildings': len(geo['buildings']), 'roads': len(geo['roads']), 'areas': len(geo['areas']),
                   'collision': len(nav['collision']), 'junctions': len(nav['junctions']), 'crossings': len(nav['crossings'])},
        'warnings': geo['metadata']['warnings'],
        'chunks': chunk_report(geo), 'topology': topology_report(geo, src), 'buildings': building_report(geo, src, place),
        'scale': scale_report(geo, src), 'completeness': completeness_report(geo, src, place),
        'terrain': terrain_report(geo, terrain_src, place), 'provenance': provenance_report(geo, manifest),
    }


# ------------------------------------------------------------------------------------------ builds
def independent_builds(folder: Path, count: int = 2) -> list[dict]:
    """Build the snapshot `count` times at once, each in its own process (own hash seed) and clean temporary
    folder, through snapshot.py's command line. [{path, bytes, sha256, seconds, stdout}] (the files are kept
    in a temporary folder the caller removes: 'tmp')."""
    tmp = Path(tempfile.mkdtemp(prefix='outbreak-p01-'))
    procs = []
    for k in range(count):
        out = tmp / f'build{k + 1}' / 'OutbreakGeo.json'
        env = {**os.environ, 'PYTHONHASHSEED': str(101 + k)}
        procs.append((out, time.perf_counter(), subprocess.Popen(
            [sys.executable, str(ROOT / 'pipeline/outbreak/snapshot.py'), 'build', str(folder), '--out', str(out), '--timings'],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True, encoding='utf-8', errors='replace')))
    results = []
    for out, started, p in procs:
        stdout, _ = p.communicate()
        seconds = time.perf_counter() - started
        if p.returncode:
            raise snapshot.SnapshotError(f'build failed: {stdout[-800:]}')
        data = out.read_bytes()
        timings = json.loads(next(line[8:] for line in stdout.splitlines() if line.startswith('TIMINGS ')))
        results.append({'path': str(out), 'bytes': len(data), 'sha256': snapshot.sha256(data), 'seconds': seconds, 'timings': timings, 'tmp': str(tmp)})
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(prog='pipeline/outbreak/realcheck.py', description='P01 validation of a real OutbreakGeo')
    ap.add_argument('snapshot')
    ap.add_argument('--builds', type=int, default=2)
    ap.add_argument('--out', help='write the report as JSON here')
    args = ap.parse_args(argv)
    folder = Path(args.snapshot)
    manifest, files = snapshot.verify(folder)
    t = time.perf_counter()
    builds = independent_builds(folder, args.builds)
    wall = time.perf_counter() - t
    geo = json.loads(Path(builds[0]['path']).read_bytes().decode('utf-8'))
    rep = report(geo, manifest, files)
    rep['reproducibility'] = {'builds': len(builds), 'sha256': [b['sha256'] for b in builds], 'bytes': [b['bytes'] for b in builds],
                              'identical': len({b['sha256'] for b in builds}) == 1 and len(builds) > 1}
    rep['performance'] = {'wall_s_all_builds_in_parallel': round(wall, 1), 'builds': [{k: v for k, v in b.items() if k in ('seconds', 'timings')} for b in builds],
                          'snapshot_stored_bytes': {n: manifest['files'][n]['stored_bytes'] for n in manifest['files']}}
    rep['pass'] = {k: rep[k]['pass'] for k in ('chunks', 'topology', 'buildings', 'scale', 'completeness', 'terrain', 'provenance')}
    rep['pass']['reproducibility'] = rep['reproducibility']['identical']
    text = json.dumps(rep, ensure_ascii=False, indent=1)
    if args.out:
        Path(args.out).write_text(text, encoding='utf-8')
    adapter._say(text)
    shutil.rmtree(builds[0]['tmp'], ignore_errors=True)


if __name__ == '__main__':
    main()
