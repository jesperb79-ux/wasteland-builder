"""Retained real-source snapshots: the place, the OpenStreetMap download and the terrain grid an
OutbreakGeo is built from, pinned by SHA-256, so the same OutbreakGeo can be built again offline,
without downloading whatever OpenStreetMap holds that day.

    .venv/bin/python pipeline/outbreak/snapshot.py create <folder> --name … --center LAT,LON --size M --country CC
    .venv/bin/python pipeline/outbreak/snapshot.py verify <folder>
    .venv/bin/python pipeline/outbreak/snapshot.py build <folder> --out OutbreakGeo.json

`create` runs Wasteland's own download steps (fetch_osm.py, fetch_terrain.py) once, for a place.json on
the WGS84 plane, and keeps what they wrote, byte for byte: place.json, osm.json and terrain.json (the two
large ones gzipped, mtime 0), with SNAPSHOT.json (schemas/real_snapshot.v1) recording their SHA-256,
the download time, terrain tiles, attribution and licences. `verify` refuses a snapshot whose files
don't match it, or whose metadata is malformed or inconsistent. `build` reads nothing else: it writes
the verified files into a fresh city folder, runs prepare_city (UTF-8 mode, a child process) and the
adapter (which rebuilds city.json once more and checks it is identical) and validates the result.
Format and rules: docs/outbreak-real-city.md.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

if __package__ in (None, ''):                                    # run as a script: find pipeline/
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import EARTH, ROOT, Projection  # noqa: E402
from outbreak import adapter, schema  # noqa: E402


def load_env():
    """Wasteland's own .env reading (wasteland.py load_env: KEY=value lines, never printed, the environment wins),
    as `wasteland.py build` does before its terrain step: LANTMATERIET_* credentials live there."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import wasteland
    wasteland.load_env()

MANIFEST = 'SNAPSHOT.json'
NOTICE = 'NOTICE.md'
FILES = {'place.json': 'place.json', 'osm.json': 'osm.json.gz', 'terrain.json': 'terrain.json.gz'}
CODE_FILES = ('pipeline/common.py', 'pipeline/fetch_osm.py', 'pipeline/fetch_terrain.py', 'pipeline/outbreak/snapshot.py')
LM_KEYS = ('LANTMATERIET_USER', 'LANTMATERIET_PASSWORD', 'LANTMATERIET_CONSUMER_KEY', 'LANTMATERIET_CONSUMER_SECRET')
# OSM elements whose source tags name Google or Street View: OSM forbids those sources, and Outbreak never
# uses them, so a snapshot holding one is refused.
STREET_VIEW = re.compile(r'google|street\s*view|gsv\b', re.IGNORECASE)
STAMP = re.compile(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$')

# Wasteland's download steps, with its city folder pointed at a temporary one, in UTF-8 mode so the
# files are UTF-8 whatever the locale.
_FETCH = ('import sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); import common; '
          'common.CITIES = Path(sys.argv[2]); import fetch_osm, fetch_terrain; '
          'fetch_osm.main([sys.argv[3]]); fetch_terrain.main([sys.argv[3]])')
# prepare_city the same way; prints its peak memory last.
_PREPARE = ('import sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); import common; '
            'common.CITIES = Path(sys.argv[2]); import prepare_city; prepare_city.main([sys.argv[3]]); '
            'from outbreak.snapshot import peak_memory; print("PEAK", peak_memory())')


class SnapshotError(ValueError):
    """The snapshot can't be used: missing or changed files, malformed or inconsistent metadata."""


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def peak_memory() -> int | None:
    """This process's peak resident memory in bytes (Windows: peak working set), without extra packages."""
    try:
        if sys.platform == 'win32':
            import ctypes
            from ctypes import wintypes

            class Counters(ctypes.Structure):
                _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + \
                           [(n, ctypes.c_size_t) for n in ('PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage',
                                                           'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
                                                           'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]
            c = Counters()
            c.cb = ctypes.sizeof(c)
            kernel32, psapi = ctypes.WinDLL('kernel32'), ctypes.WinDLL('psapi')
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
            if not psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb):
                return None
            return int(c.PeakWorkingSetSize)
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(peak if sys.platform == 'darwin' else peak * 1024)
    except Exception:                                            # a measurement, never a failure
        return None


def street_view_tags(osm: dict) -> list[str]:
    """'<type> <id> <key>=<value>' for every source tag in osm.json that names Google or Street View."""
    out = []
    for el in osm.get('elements') or []:
        for k, v in sorted((el.get('tags') or {}).items()):
            if k.startswith('source') and STREET_VIEW.search(str(v)):
                out.append(f'{el.get("type")} {el.get("id")} {k}={v}')
    return out


def incomplete_relations(osm: dict) -> list[str]:
    """Multipolygons with a way member that has no geometry: fetch_osm.py completes those that reach outside
    the download through Overpass, and goes on without them when Overpass is busy. Such an area would be
    missing or wrong, so a snapshot holding one is refused."""
    return [f'relation {el.get("id")}' for el in osm.get('elements') or []
            if el.get('type') == 'relation' and (el.get('tags') or {}).get('type') == 'multipolygon'
            and any(m.get('type') == 'way' and not m.get('geometry') for m in el.get('members') or [])]


def incomplete_ways(osm: dict) -> list[str]:
    """Ways whose node ids and geometry don't line up one for one (fetch_osm.py keeps every node id; a node the
    download lacked leaves the geometry short), or which have no node ids at all, plus the ways the download
    itself reported incomplete (its `fetch` record, which also covers multipolygon members)."""
    out = [f'way {el.get("id")}' for el in osm.get('elements') or [] if el.get('type') == 'way'
           and (not isinstance(el.get('nodes'), list) or not isinstance(el.get('geometry'), list)
                or len(el['nodes']) != len(el['geometry']) or not all(el['geometry']))]
    reported = (osm.get('fetch') or {}).get('incomplete_ways') or []
    return sorted(set(out) | {f'way {w}' for w in reported})


def terrain_licence(attribution: str) -> tuple[str, str]:
    """(licence, url) of a terrain attribution from fetch_terrain.py; SnapshotError for an unknown source."""
    names, urls = [], []
    for part in [a.strip() for a in attribution.split(' · ') if a.strip()]:
        known = next((lic for key, lic in adapter.TERRAIN_LICENCES if key in part), None)
        if known is None:
            raise SnapshotError(f'terrain attribution "{part}" is not a known source, so its licence is unknown')
        names.append(f'{known["data"]}: {known["license"]}')
        urls.append(known['url'])
    if not names:
        raise SnapshotError('the terrain has no attribution')
    return '; '.join(names), ' '.join(urls)


def _place_bbox(center, size_m) -> dict:
    proj = Projection(center[0], center[1], 'wgs84')
    (s, w), (n, e) = proj.latlon(-size_m / 2, -size_m / 2), proj.latlon(size_m / 2, size_m / 2)
    return {'south': round(s, 7), 'west': round(w, 7), 'north': round(n, 7), 'east': round(e, 7)}


def _code_pins() -> dict:
    return {f: adapter._sha256_text(ROOT / f) for f in CODE_FILES}


def _now() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


# ------------------------------------------------------------------------------------------ create
def create(folder: Path, *, name: str, center, size_m: float, country: str, query: str | None = None,
           display_name: str | None = None, selection: str = '', purpose: str = '') -> dict:
    """Download once with Wasteland's own steps and keep the result as a snapshot in `folder` (new or empty)."""
    folder = Path(folder)
    sid = folder.name
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]*', sid):
        raise SnapshotError(f'snapshot folder name {sid!r}: lower-case letters, digits and dashes (it becomes the city slug)')
    if folder.exists() and any(folder.iterdir()):
        raise SnapshotError(f'{folder} is not empty: a snapshot is never overwritten')
    lat, lon = (float(v) for v in center)
    bbox = _place_bbox((lat, lon), size_m)
    place = {'name': name, 'query': query or name, 'display_name': display_name or name, 'country_code': country.lower(),
             'region': '', 'center': [lat, lon], 'size_m': size_m, 'projection': 'wgs84',
             'bbox': [bbox['south'], bbox['west'], bbox['north'], bbox['east']], 'osm': None,
             'attribution': 'Map data © OpenStreetMap contributors (ODbL)'}
    load_env()                                    # before the check, and inherited by the download steps
    credentials = any(os.environ.get(k) for k in LM_KEYS)
    with tempfile.TemporaryDirectory() as tmp:
        city = Path(tmp) / sid
        city.mkdir()
        (city / 'place.json').write_bytes((json.dumps(place, indent=2) + '\n').encode('ascii'))   # any locale, any platform
        proc = subprocess.run([sys.executable, '-X', 'utf8', '-c', _FETCH, str(ROOT / 'pipeline'), tmp, sid])
        if proc.returncode:
            raise SnapshotError('the download failed (see the output above)')
        missing = [n for n in FILES if not (city / n).exists()]
        if missing:
            raise SnapshotError(f'the download steps left no {", ".join(missing)}; nothing kept')
        raw = {n: (city / n).read_bytes() for n in FILES}
    osm, terrain = json.loads(raw['osm.json'].decode('utf-8')), json.loads(raw['terrain.json'].decode('utf-8'))
    broken, short = incomplete_relations(osm), incomplete_ways(osm)
    if broken:
        raise SnapshotError(f'Overpass could not complete {len(broken)} areas reaching outside the download ({", ".join(broken)}): '
                            'they would be missing. Nothing kept; try again when Overpass answers.')
    if short:
        raise SnapshotError(f'{len(short)} ways came back without all their nodes ({", ".join(short[:10])}): their geometry would '
                            'skip those points. Nothing kept; try again.')
    fetch = osm.get('fetch')
    if not isinstance(fetch, dict) or not fetch.get('parts'):
        raise SnapshotError('osm.json has no fetch record (download time and sources): not written by this fetch_osm.py')
    licence, licence_url = terrain_licence(terrain['attribution'])
    if terrain['kind'] == 'dtm':
        primary = 'Lantmäteriet Markhöjdmodell (1 m ground model), the preferred source for Swedish places.'
    elif country.lower() == 'se':
        primary = ('Lantmäteriet Markhöjdmodell (1 m ground model) is preferred for Swedish places but needs LANTMATERIET_* '
                   'credentials in .env; ' + ('they were set, yet it gave no data' if credentials else 'none were configured') +
                   ', so fetch_terrain.py used its documented fallback, Copernicus DEM GLO-30 (30 m surface model, which '
                   'prepare_city cleans of mapped buildings and forests).')
    else:
        primary = 'Copernicus DEM GLO-30 (30 m surface model), fetch_terrain.py\'s source outside Sweden.'
    ways = [el for el in osm['elements'] if el.get('type') == 'way']
    s, w, n, e = place['bbox']
    pad_lat, pad_lon = 80 / EARTH, 80 / (EARTH * math.cos(math.radians((s + n) / 2)))     # fetch_osm.py's margin, as it computes it
    manifest = {
        'schema': 'outbreak-real-snapshot', 'schema_version': 1, 'id': sid, 'created_utc': _now(),
        'purpose': purpose or 'Reproducible input for an OutbreakGeo built from real data.',
        'place': {'name': name, 'country_code': country.lower(), 'center': [lat, lon], 'size_m': size_m, 'projection': 'wgs84',
                  'bbox_wgs84': bbox, 'selection': selection},
        'files': {},
        'osm': {'source': 'OpenStreetMap: ' + osm['generator'].replace('wasteland-builder (', '').rstrip(')'),
                'generator': osm['generator'],
                'downloaded_utc': {'started': fetch['started_utc'], 'finished': fetch['finished_utc']},
                'sources': fetch['parts'], 'timestamp_osm_base': osm['osm3s']['timestamp_osm_base'],
                'download_bbox': [round(v, 6) for v in (s - pad_lat, w - pad_lon, n + pad_lat, e + pad_lon)],
                'elements': len(osm['elements']), 'ways': len(ways),
                'ways_with_node_ids': sum(1 for el in ways if len(el.get('nodes') or []) == len(el.get('geometry') or []) > 0),
                'street_view_tags': street_view_tags(osm), 'incomplete_relations': broken, 'incomplete_ways': short, 'license': 'ODbL-1.0',
                'attribution': '© OpenStreetMap contributors', 'license_url': 'https://www.openstreetmap.org/copyright'},
        'terrain': {'source': terrain['source'], 'kind': terrain['kind'], 'tiles': terrain['tiles'], 'attribution': terrain['attribution'],
                    'license': licence, 'license_url': licence_url, 'projection': terrain.get('projection'),
                    'step_m': terrain['step'], 'n': terrain['n'], 'coverage': terrain.get('coverage'), 'primary_source': primary},
        'pipeline': {**adapter.git_generator(), 'code_sha256': _code_pins()},
        'policy': {'street_view': 'Not a source. No Google Street View or Google imagery was read; OSM elements whose source '
                                  'tags name Google or Street View would make the snapshot invalid (street_view_tags).',
                   'network': 'Live OpenStreetMap and terrain services were read once, when this snapshot was made. Builds read '
                              'only these files.'},
    }
    folder.mkdir(parents=True, exist_ok=True)
    for n_, stored in FILES.items():
        data = raw[n_]
        out = gzip.compress(data, compresslevel=9, mtime=0) if stored.endswith('.gz') else data
        (folder / stored).write_bytes(out)
        manifest['files'][n_] = {'stored': stored, 'sha256': sha256(data), 'bytes': len(data),
                                 'stored_sha256': sha256(out), 'stored_bytes': len(out)}
    (folder / NOTICE).write_bytes(notice(manifest).encode('utf-8'))
    (folder / MANIFEST).write_bytes((json.dumps(manifest, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
    verify(folder)
    return manifest


def notice(manifest: dict) -> str:
    t = manifest['terrain']
    return (f'# {manifest["id"]}: licences of the data in this folder\n\n'
            f'- `osm.json.gz`: an extract of OpenStreetMap downloaded {manifest["osm"]["downloaded_utc"]["started"]} '
            f'(data as of {manifest["osm"]["timestamp_osm_base"] or "an unstated time"}), '
            f'{manifest["osm"]["attribution"]}, available under the Open Database License 1.0 (ODbL), '
            f'{manifest["osm"]["license_url"]}. Anything made from it (OutbreakGeo) is a derived database under the ODbL.\n'
            f'- `terrain.json.gz`: {t["source"]} heights, sampled by Wasteland\'s fetch_terrain.py. {t["attribution"]}. '
            f'Licence: {t["license"]} ({t["license_url"]}).\n'
            f'- `place.json`, `SNAPSHOT.json`: written by pipeline/outbreak/snapshot.py (MIT, like the rest of the repository).\n\n'
            'No Google Street View or other Google data is in this folder. Wasteland Builder\'s bundled vehicles, art, audio '
            'and music (THIRD_PARTY.md) are not used.\n')


# ------------------------------------------------------------------------------------------ verify
def verify(folder: Path) -> tuple[dict, dict[str, bytes]]:
    """(manifest, {name: bytes}) of a snapshot whose files and metadata check out; SnapshotError otherwise."""
    folder = Path(folder)
    try:
        manifest = json.loads((folder / MANIFEST).read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise SnapshotError(f'{folder / MANIFEST}: {exc}') from None
    problems = schema.validate(manifest, 'real_snapshot.v1')
    if problems:
        raise SnapshotError(f'{MANIFEST} is malformed: ' + '; '.join(problems[:8]))
    if manifest['id'] != folder.name:
        problems.append(f'id {manifest["id"]!r} is not the folder name {folder.name!r}')
    stray = sorted(p.name for p in folder.iterdir() if p.name not in {MANIFEST, NOTICE, *FILES.values()})
    if stray:
        problems.append(f'unexpected files in the snapshot: {", ".join(stray)}')
    files = {}
    for name, stored in FILES.items():
        rec = manifest['files'][name]
        if rec['stored'] != stored:
            problems.append(f'{name} must be stored as {stored}')
            continue
        try:
            raw = (folder / stored).read_bytes()
            data = gzip.decompress(raw) if stored.endswith('.gz') else raw
        except (OSError, EOFError, gzip.BadGzipFile) as exc:
            problems.append(f'{stored}: {exc}')
            continue
        if (sha256(raw), len(raw)) != (rec['stored_sha256'], rec['stored_bytes']):
            problems.append(f'{stored} has changed (SHA-256 or size differ from {MANIFEST})')
        if (sha256(data), len(data)) != (rec['sha256'], rec['bytes']):
            problems.append(f'{name} has changed (SHA-256 or size of its content differ from {MANIFEST})')
        files[name] = data
    if problems:
        raise SnapshotError('; '.join(problems))
    try:
        place = json.loads(files['place.json'].decode('utf-8'))
        osm = json.loads(files['osm.json'].decode('utf-8'))
        terrain = json.loads(files['terrain.json'].decode('utf-8'))
    except (UnicodeDecodeError, ValueError) as exc:
        raise SnapshotError(f'a snapshot file is not UTF-8 JSON: {exc}') from None
    problems += _consistency(manifest, place, osm, terrain)
    if problems:
        raise SnapshotError('; '.join(problems))
    return manifest, files


def _consistency(manifest, place, osm, terrain) -> list[str]:
    """What SNAPSHOT.json says against what the files hold."""
    out = []
    mp, mo, mt = manifest['place'], manifest['osm'], manifest['terrain']
    for key, want in (('name', mp['name']), ('country_code', mp['country_code']), ('center', mp['center']), ('size_m', mp['size_m']),
                      ('projection', mp['projection'])):
        if place.get(key) != want:
            out.append(f'place.json {key} is {place.get(key)!r}, {MANIFEST} says {want!r}')
    if isinstance(place.get('center'), list) and len(place['center']) == 2 and _num(place.get('size_m')):
        box = _place_bbox(place['center'], place['size_m'])
        if any(abs(box[k] - mp['bbox_wgs84'][k]) > 1e-7 for k in box) or place.get('bbox') != [box[k] for k in ('south', 'west', 'north', 'east')]:
            out.append('the play-area box is not the square of size_m round the centre on the WGS84 plane')
    elements = osm.get('elements') if isinstance(osm, dict) else None
    if not isinstance(elements, list):
        return out + ['osm.json has no elements']
    ways = [el for el in elements if isinstance(el, dict) and el.get('type') == 'way']
    fetch = osm.get('fetch') if isinstance(osm.get('fetch'), dict) else {}
    for key, have in (('generator', osm.get('generator')), ('timestamp_osm_base', (osm.get('osm3s') or {}).get('timestamp_osm_base')),
                      ('downloaded_utc', {'started': fetch.get('started_utc'), 'finished': fetch.get('finished_utc')}),
                      ('sources', fetch.get('parts')),
                      ('elements', len(elements)), ('ways', len(ways)),
                      ('ways_with_node_ids', sum(1 for el in ways if len(el.get('nodes') or []) == len(el.get('geometry') or []) > 0))):
        if have != mo[key]:
            out.append(f'osm.json {key} is {have!r}, {MANIFEST} says {mo[key]!r}')
    if not fetch.get('parts') or not STAMP.match(str(fetch.get('started_utc'))) or not STAMP.match(str(fetch.get('finished_utc'))):
        out.append('osm.json has no fetch record (download time and sources)')
    times = [p.get('server_time_utc') or p.get('timestamp_osm_base') for p in fetch.get('parts') or []]
    if (osm.get('osm3s') or {}).get('timestamp_osm_base') != (min(times) if times and all(times) else None):
        out.append('osm.json timestamp_osm_base is not the oldest time its sources state (or null when one states none)')
    found = street_view_tags(osm)
    if found:
        out.append(f'osm.json has elements whose source names Google or Street View: {", ".join(found[:5])}')
    broken = incomplete_relations(osm) + [f'relation {r}' for r in fetch.get('incomplete_relations') or []]
    if broken:
        out.append(f'osm.json has areas missing a member\'s geometry: {", ".join(sorted(set(broken))[:5])}')
    short = incomplete_ways(osm)
    if short:
        out.append(f'osm.json has ways without all their nodes: {", ".join(short[:5])}')
    for key, have in (('source', terrain.get('source')), ('kind', terrain.get('kind')), ('tiles', terrain.get('tiles')),
                      ('attribution', terrain.get('attribution')), ('projection', terrain.get('projection')),
                      ('step_m', terrain.get('step')), ('n', terrain.get('n')), ('coverage', terrain.get('coverage'))):
        if have != mt[key]:
            out.append(f'terrain.json {key} is {have!r}, {MANIFEST} says {mt[key]!r}')
    try:
        if terrain_licence(str(terrain.get('attribution') or '')) != (mt['license'], mt['license_url']):
            out.append(f'the terrain licence in {MANIFEST} is not the one its attribution names')
    except SnapshotError as exc:
        out.append(str(exc))
    cov = terrain.get('coverage') if isinstance(terrain.get('coverage'), dict) else {}
    holes = cov.get('support_no_data_cells', cov.get('no_data_cells'))     # older files: the whole grid, a superset
    if holes != 0:
        out.append(f'the terrain has {holes} cells without data where prepare_city reads it (the play area, 450 m round it and one '
                   'step): heights there would be made up')
    n, step, z = terrain.get('n'), terrain.get('step'), terrain.get('z')
    if not (isinstance(n, int) and _num(step) and isinstance(z, list) and len(z) == n * n and all(_num(v) for v in z)):
        out.append('terrain.json needs n×n finite heights')
    elif terrain.get('frame') != 'local' or not all(_num(terrain.get(k)) for k in ('x0', 'y0')):
        out.append('terrain.json must be a grid in the local plane (frame local, x0, y0)')
    else:
        half = mp['size_m'] / 2
        if not adapter.raster_covers((-half, -half, half, half), terrain['x0'], terrain['y0'], step, n):
            out.append('the terrain grid does not cover the play area')
    return out


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


# ------------------------------------------------------------------------------------------ build
def build(folder: Path, work: Path, *, generator=None, timings: dict | None = None) -> dict:
    """OutbreakGeo from a verified snapshot only: a fresh city folder under `work` (which must not hold one
    already), prepare_city, then the adapter with its clean-rebuild check, then the contract."""
    timings = timings if timings is not None else {}
    manifest, files = verify(folder)
    city = Path(work) / manifest['id']
    city.mkdir(parents=True)
    for name, data in files.items():
        (city / name).write_bytes(data)
    t = time.perf_counter()
    proc = subprocess.run([sys.executable, '-X', 'utf8', '-c', _PREPARE, str(ROOT / 'pipeline'), str(work), manifest['id']],
                          capture_output=True, text=True, encoding='utf-8', errors='replace')
    timings['prepare_s'] = time.perf_counter() - t
    if proc.returncode or not (city / 'city.json').exists():
        raise SnapshotError(f'prepare_city failed: {(proc.stderr or proc.stdout).strip()[-600:]}')
    peak = re.findall(r'^PEAK (\d+)$', proc.stdout, re.MULTILINE)
    timings['prepare_peak_bytes'] = int(peak[-1]) if peak else None
    t = time.perf_counter()
    geo = adapter.convert_city_dir(city, generator=generator if generator is not None else adapter.git_generator(), timings=timings)
    timings['convert_s'] = time.perf_counter() - t
    t = time.perf_counter()
    problems = adapter.validate(geo)
    timings['validate_s'] = time.perf_counter() - t
    if problems:
        raise SnapshotError('the OutbreakGeo breaks its contract: ' + '; '.join(problems[:10]))
    timings['sizes'] = {'city.json': (city / 'city.json').stat().st_size, **{n: len(d) for n, d in files.items()}}
    return geo


# ------------------------------------------------------------------------------------------ command line
def main(argv=None):
    ap = argparse.ArgumentParser(prog='pipeline/outbreak/snapshot.py', description='retained real-source snapshots (docs/outbreak-real-city.md)')
    sub = ap.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('create', help='download once with Wasteland\'s steps and keep the files')
    c.add_argument('folder')
    c.add_argument('--name', required=True)
    c.add_argument('--center', required=True, help='lat,lon')
    c.add_argument('--size', type=float, required=True, help='side of the square play area in metres')
    c.add_argument('--country', required=True, help='ISO country code (se: Lantmäteriet terrain if credentials are set)')
    c.add_argument('--query')
    c.add_argument('--display-name')
    c.add_argument('--selection', default='')
    c.add_argument('--purpose', default='')
    v = sub.add_parser('verify', help='check the files and metadata')
    v.add_argument('folder')
    b = sub.add_parser('build', help='build OutbreakGeo offline from the snapshot')
    b.add_argument('folder')
    b.add_argument('--out', required=True)
    b.add_argument('--timings', action='store_true', help='print the stage timings, sizes and peak memory as a JSON line')
    args = ap.parse_args(argv)
    try:
        if args.cmd == 'create':
            lat, lon = (float(x) for x in args.center.split(','))
            m = create(Path(args.folder), name=args.name, center=(lat, lon), size_m=args.size, country=args.country, query=args.query,
                       display_name=args.display_name, selection=args.selection, purpose=args.purpose)
            adapter._say(f'snapshot {m["id"]}: OSM as of {m["osm"]["timestamp_osm_base"]}, {m["osm"]["elements"]} elements; '
                         f'terrain {m["terrain"]["source"]} ({", ".join(m["terrain"]["tiles"])})')
        elif args.cmd == 'verify':
            m, files = verify(Path(args.folder))
            adapter._say(f'snapshot {m["id"]}: OK ({", ".join(f"{n} {len(d)} bytes" for n, d in files.items())})')
        else:
            timings = {}
            with tempfile.TemporaryDirectory() as tmp:
                geo = build(Path(args.folder), Path(tmp), timings=timings)
            text = adapter.dumps(geo).encode('utf-8')
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(text)
            adapter._say(f'OutbreakGeo {out} ({len(text)} bytes, SHA-256 {sha256(text)}): {len(geo["buildings"])} buildings, '
                         f'{len(geo["roads"])} roads, {len(geo["chunks"])} chunks; prepare {timings["prepare_s"]:.1f} s, '
                         f'convert {timings["convert_s"]:.1f} s, validate {timings["validate_s"]:.1f} s')
            if args.timings:
                timings['outbreakgeo_bytes'] = len(text)
                timings['adapter_process_peak_bytes'] = peak_memory()
                print('TIMINGS ' + json.dumps(timings, sort_keys=True), flush=True)
    except (SnapshotError, adapter.AdapterError) as exc:
        raise SystemExit(f'error: {exc}') from None


if __name__ == '__main__':
    main()
