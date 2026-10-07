"""Shared paths, place config and the local projection used by every pipeline step.

Coordinates in the city model are metres on a local tangent plane centred on the place:
x = east, y = north, z = up (Blender). The web export converts to three.js Y-up as
(x, z, -y), so a map point (x, y) becomes [x, -y] in the game's map.json.
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CITIES = ROOT / 'cities'
GAME = ROOT / 'game'
USER_AGENT = 'wasteland-builder/0.1 (+https://github.com/; OSM-based game city generator)'
EARTH = 111_320.0


def slugify(text: str) -> str:
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode()
    return re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-') or 'city'


def city_dir(slug: str) -> Path:
    path = CITIES / slug
    if not (path / 'place.json').exists():
        known = ', '.join(sorted(p.name for p in CITIES.glob('*') if (p / 'place.json').exists())) or 'none yet'
        sys.exit(f'No city "{slug}". Create it with: python3 wasteland.py new "<place>"   (existing: {known})')
    return path


def load_place(slug: str) -> dict:
    return json.loads((city_dir(slug) / 'place.json').read_text())


def write_json(path: Path, data, compact=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    if compact:
        tmp.write_text(json.dumps(data, ensure_ascii=False, separators=(',', ':')))
    else:
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    tmp.replace(path)


class Projection:
    """Equirectangular local plane around (lat0, lon0).

    model 'sphere' (the default, every Wasteland city so far): 111 320 m per degree of latitude and
    111 320 · cos lat0 per degree of longitude. Shapes are right, but the scale is off by up to ~0.7 %
    depending on latitude (−0.24 % east–west at 58° N), which doesn't matter for the game.
    model 'wgs84' (place.json "projection": "wgs84"): the WGS84 ellipsoid's meridional and prime-vertical
    radii at lat0, so lengths are true to about 1e-4 within a kilometre of the centre."""

    MODELS = ('sphere', 'wgs84')

    def __init__(self, lat0: float, lon0: float, model: str = 'sphere'):
        if model not in self.MODELS:
            raise ValueError(f'unknown projection {model!r} (known: {", ".join(self.MODELS)})')
        self.lat0, self.lon0, self.model = lat0, lon0, model
        if model == 'sphere':
            self.kx, self.ky = EARTH * math.cos(math.radians(lat0)), EARTH
        else:
            a, f = 6378137.0, 1 / 298.257223563
            e2 = f * (2 - f)
            w = 1 - e2 * math.sin(math.radians(lat0)) ** 2
            self.ky = math.radians(1) * a * (1 - e2) / w ** 1.5
            self.kx = math.radians(1) * a / math.sqrt(w) * math.cos(math.radians(lat0))

    def xy(self, lat: float, lon: float):
        return ((lon - self.lon0) * self.kx, (lat - self.lat0) * self.ky)

    def latlon(self, x: float, y: float):
        return (self.lat0 + y / self.ky, self.lon0 + x / self.kx)


def projection_for(place: dict) -> Projection:
    return Projection(*place['center'], place.get('projection', 'sphere'))


def http_json(url: str, data: dict | None = None, timeout=180, attempts=3, info: dict | None = None):
    """GET (or POST `data`) and parse JSON. `info`, when given, gets the server's Date header of the answer."""
    body = urllib.parse.urlencode(data).encode() if data else None
    last = None
    for attempt in range(attempts):
        try:
            req = urllib.request.Request(url, data=body, headers={'User-Agent': USER_AGENT, 'Accept': 'application/json'})
            with urllib.request.urlopen(req, timeout=timeout) as res:
                if info is not None:
                    info['date'] = res.headers.get('Date')
                return json.loads(res.read().decode('utf-8'))
        except Exception as exc:  # network hiccups, 429/504 from public servers
            last = exc
            time.sleep(2 + attempt * 4)
    raise RuntimeError(f'{url}: {last}')


def step_done(slug: str, step: str, **info):
    path = city_dir(slug) / 'status.json'
    status = json.loads(path.read_text()) if path.exists() else {}
    status[step] = {'done': time.strftime('%Y-%m-%d %H:%M:%S'), **info}
    write_json(path, status)


def say(msg: str):
    print(msg, flush=True)
