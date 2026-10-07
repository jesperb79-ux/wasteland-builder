"""Download the OpenStreetMap features for a city's bounding box through the Overpass API.

Writes cities/<slug>/osm.json (raw Overpass JSON, ODbL). Re-running reuses the file unless --force.
Its `fetch` record says when it was downloaded, from which sources, how current each source's data was
(the OSM API's server time, Overpass's timestamp_osm_base) and which ways or areas came back incomplete.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from datetime import timezone
from email.utils import parsedate_to_datetime

from common import EARTH, city_dir, http_json, load_place, say, step_done

MIRRORS = [
    'https://overpass-api.de/api/interpreter',
    'https://overpass.private.coffee/api/interpreter',
    'https://overpass.kumi.systems/api/interpreter',
]

# Several small queries instead of one big one: public Overpass servers time out (HTTP 504) on
# heavy combined requests. Giant relations (seas, bays, straits) are skipped: the sea comes from
# the coastline ways. Each query falls back to the next mirror on its own.
QUERIES = {
    'buildings': 'way["building"]; relation["building"]; way["building:part"]; relation["building:part"];',
    'streets': 'way["highway"]; way["area:highway"]; way["railway"~"^(rail|tram|light_rail|narrow_gauge|platform)$"]; '
               'way["barrier"~"^(city_wall|wall|hedge|retaining_wall)$"]; way["historic"~"^(city_wall|citywalls|castle|ruins|fort)$"]; '
               'way["man_made"~"^(pier|breakwater|bridge|quay)$"];',
    'areas': 'way["landuse"]; way["leisure"]; way["natural"]; way["amenity"~"^(parking|grave_yard|marketplace)$"]; '
             'way["place"="square"]; way["waterway"]; way["water"];',
    'relations': 'relation["type"="multipolygon"]["landuse"]; relation["type"="multipolygon"]["leisure"]; '
                 'relation["type"="multipolygon"]["natural"]["natural"!~"^(bay|strait|sea|peninsula|cape|isthmus|coastline)$"]; '
                 'relation["type"="multipolygon"]["water"]; relation["type"="multipolygon"]["waterway"="riverbank"]; '
                 'relation["type"="multipolygon"]["amenity"="parking"]; relation["type"="multipolygon"]["place"="square"]; '
                 'relation["type"="multipolygon"]["man_made"~"^(pier|breakwater|bridge)$"];',
    'points': 'node["natural"="tree"]; node["highway"~"^(street_lamp|traffic_signals|bus_stop)$"]; '
              'node["amenity"~"^(bench|fountain|waste_basket|bicycle_parking|telephone|post_box)$"]; '
              'node["name"]["amenity"]; node["name"]["tourism"]; node["name"]["historic"]; '
              'node["place"~"^(suburb|quarter|neighbourhood|square|locality|island|islet|village|hamlet)$"];',
}
TEMPLATE = '[out:json][timeout:{timeout}]{bbox};\n({body});\nout body geom qt;'


def _now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def _http_time(date):
    """An HTTP Date header as 'YYYY-MM-DDTHH:MM:SSZ' (UTC), or None."""
    try:
        return parsedate_to_datetime(date).astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ') if date else None
    except (TypeError, ValueError):
        return None


def overpass(body, bbox, wait=100, record=None):
    """Run one Overpass query on the first mirror that answers; None when all fail. `record` (the fetch
    record main() keeps), when given, gets the mirror and how current its data was (timestamp_osm_base)."""
    query = TEMPLATE.format(timeout=wait - 10, bbox=f'[bbox:{bbox}]' if bbox else '', body=body)
    for url in MIRRORS:
        try:
            data = http_json(url, {'data': query}, timeout=wait, attempts=1)
            if data.get('remark') and ('error' in data['remark'].lower() or 'timed out' in data['remark'].lower()):
                raise RuntimeError(data['remark'])
            if record is not None:
                record['parts'].append({'source': f'Overpass API {url.split("/")[2]}',
                                        'timestamp_osm_base': (data.get('osm3s') or {}).get('timestamp_osm_base'),
                                        'elements': len(data['elements'])})
            return data['elements']
        except Exception as exc:
            say(f'    {url.split("/")[2]}: {str(exc)[:80]}')
    return None


def osm_api_elements(bbox, depth=0, record=None):
    """Fallback: the main OSM API's /map call (complete ways; relations as far as their members are
    inside the box), converted to Overpass-style elements with geometry. Splits the box when the
    API refuses (more than 50 000 nodes). `record`, when given, gets the server time of each answer
    and the ways a node was missing from."""
    s, w, n, e = (float(v) for v in bbox.split(','))
    url = f'https://api.openstreetmap.org/api/0.6/map.json?bbox={w:.6f},{s:.6f},{e:.6f},{n:.6f}'
    info = {}
    try:
        raw = http_json(url, timeout=180, attempts=2, info=info)['elements']
    except Exception as exc:
        if depth >= 3:
            raise SystemExit(f'The OpenStreetMap API failed too: {exc}. Try again later or choose a smaller --size.')
        say(f'    splitting the area ({exc})')
        mlat, mlon = (s + n) / 2, (w + e) / 2
        out, seen = [], {}
        for q in ((s, w, mlat, mlon), (s, mlon, mlat, e), (mlat, w, n, mlon), (mlat, mlon, n, e)):
            for el in osm_api_elements(','.join(f'{v:.6f}' for v in q), depth + 1, record):
                key = (el['type'], el['id'])
                if key in seen and el['type'] == 'relation':
                    # Fill in member geometry the other quarter had.
                    old = seen[key]
                    for a, b in zip(old['members'], el['members']):
                        if not a.get('geometry') and b.get('geometry'):
                            a['geometry'] = b['geometry']
                elif key not in seen:
                    seen[key] = el
                    out.append(el)
        return out
    say(f'    {len(raw)} raw elements from api.openstreetmap.org')
    if record is not None:
        record['parts'].append({'source': 'OpenStreetMap API 0.6 map', 'bbox': bbox, 'server_time_utc': _http_time(info.get('date')),
                                'elements': len(raw)})
    nodes = {el['id']: el for el in raw if el['type'] == 'node'}
    ways = {el['id']: el for el in raw if el['type'] == 'way'}
    geom = lambda refs: [{'lat': nodes[r]['lat'], 'lon': nodes[r]['lon']} for r in refs if r in nodes]
    out = []
    for el in raw:
        if el['type'] == 'node' and el.get('tags'):
            out.append({'type': 'node', 'id': el['id'], 'lat': el['lat'], 'lon': el['lon'], 'tags': el['tags']})
        elif el['type'] == 'way' and el.get('tags'):
            # Every node id beside the geometry (as Overpass gives them): road topology needs them. A node the
            # answer lacks leaves the geometry a point short, so the two lists no longer line up: the gap shows
            # instead of becoming a shortcut nobody can see.
            g = geom(el['nodes'])
            out.append({'type': 'way', 'id': el['id'], 'tags': el['tags'], 'nodes': el['nodes'], 'geometry': g})
            if record is not None and len(g) < len(el['nodes']):
                record['incomplete_ways'].append(el['id'])
        elif el['type'] == 'relation' and el.get('tags'):
            members = []
            for m in el['members']:
                mm = {'type': m['type'], 'ref': m['ref'], 'role': m.get('role', '')}
                if m['type'] == 'way' and m['ref'] in ways:
                    mm['geometry'] = geom(ways[m['ref']]['nodes'])
                    if record is not None and len(mm['geometry']) < len(ways[m['ref']]['nodes']):
                        record['incomplete_ways'].append(m['ref'])
                members.append(mm)
            out.append({'type': 'relation', 'id': el['id'], 'tags': el['tags'], 'members': members})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog='wasteland.py fetch')
    ap.add_argument('slug')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args(argv)
    place = load_place(args.slug)
    out = city_dir(args.slug) / 'osm.json'
    if out.exists() and not args.force:
        say(f'OSM data already downloaded ({out.stat().st_size / 1e6:.1f} MB). Use --force to refresh.')
        return
    s, w, n, e = place['bbox']
    # Fetch a margin around the play area so outlines crossing the edge stay whole.
    pad = 80
    dlat = pad / EARTH
    dlon = pad / (EARTH * math.cos(math.radians((s + n) / 2)))
    bbox = f'{s - dlat:.6f},{w - dlon:.6f},{n + dlat:.6f},{e + dlon:.6f}'
    # 1) The OSM API's map call: fast and complete for ways. 2) Overpass completes multipolygons that
    # reach outside the box. If the API is unavailable, Overpass does everything.
    record = {'started_utc': _now(), 'finished_utc': None, 'parts': [], 'incomplete_ways': [], 'incomplete_relations': []}
    try:
        say('  downloading from api.openstreetmap.org …')
        elements = osm_api_elements(bbox, record=record)
        remark = 'OpenStreetMap API 0.6 map'
        incomplete = [el['id'] for el in elements if el['type'] == 'relation' and el['tags'].get('type') == 'multipolygon'
                      and any(m['type'] == 'way' and not m.get('geometry') for m in el['members'])]
        if incomplete:
            say(f'  completing {len(incomplete)} large areas through Overpass …')
            fixed = None
            for _ in range(3):                    # public Overpass servers answer 504 now and then; a retry often works
                fixed = overpass(f'relation(id:{",".join(map(str, incomplete[:400]))});', bbox=None, wait=40, record=record)
                if fixed:
                    break
            if fixed:
                by_id = {el['id']: el for el in fixed if el['type'] == 'relation'}
                elements = [by_id.get(el['id'], el) if el['type'] == 'relation' else el for el in elements]
            else:
                say('  (Overpass busy — large areas that reach outside the map may be missing; rerun later with --refresh)')
    except SystemExit:
        say('  The OSM API failed — using Overpass instead.')
        elements, seen, remark = [], set(), 'Overpass API'
        record['parts'].clear()
        record['incomplete_ways'].clear()
        for part, body in QUERIES.items():
            data = overpass(body, bbox, record=record)
            if data is None:
                raise SystemExit(f'Could not download {part}: all Overpass servers are busy. Try again in a few minutes.')
            for el in data:
                if (el['type'], el['id']) not in seen:
                    seen.add((el['type'], el['id']))
                    elements.append(el)
    record['finished_utc'] = _now()
    record['incomplete_ways'] = sorted(set(record['incomplete_ways']))
    record['incomplete_relations'] = sorted(el['id'] for el in elements if el['type'] == 'relation'
                                            and el.get('tags', {}).get('type') == 'multipolygon'
                                            and any(m['type'] == 'way' and not m.get('geometry') for m in el['members']))
    if record['incomplete_ways'] or record['incomplete_relations']:
        say(f'  ! incomplete: {len(record["incomplete_ways"])} ways missing a node, {len(record["incomplete_relations"])} areas missing a member')
    # How current the data is: the oldest time a source states for its part (the OSM API's server time, an
    # Overpass mirror's timestamp_osm_base); unknown (null) when a part states none. The download time is in `fetch`.
    times = [p.get('server_time_utc') or p.get('timestamp_osm_base') for p in record['parts']]
    data = {'version': 0.6, 'generator': f'wasteland-builder ({remark})',
            'osm3s': {'timestamp_osm_base': min(times) if times and all(times) else None,
                      'copyright': 'The data included in this document is from www.openstreetmap.org. The data is made available under ODbL.'},
            'fetch': record, 'elements': elements}
    out.write_text(json.dumps(data, ensure_ascii=False))
    counts = {}
    for el in data['elements']:
        t = el.get('tags', {})
        key = next((k for k in ('building', 'building:part', 'highway', 'railway', 'waterway', 'landuse', 'leisure', 'natural', 'barrier') if k in t), 'other')
        counts[key] = counts.get(key, 0) + 1
    say(f'Saved {len(data["elements"])} features ({out.stat().st_size / 1e6:.1f} MB): ' +
        ', '.join(f'{v} {k}' for k, v in sorted(counts.items(), key=lambda kv: -kv[1])))
    step_done(args.slug, 'fetch', elements=len(data['elements']), counts=counts)


if __name__ == '__main__':
    main()
