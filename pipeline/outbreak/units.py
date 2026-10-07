"""Units for the Outbreak real-city experiment: metres are canonical, studs exist only at the Roblox edge.

OutbreakGeo stores every length in metres (x east, y north, z up, origin = the place centre, like
city.json). Studs never appear inside OutbreakGeo; a future Roblox builder converts with the functions
here, and nowhere else. Scale: 1 stud = 0.28 m, so 1 m = 1 / 0.28 ≈ 3.571428571 studs.

Axes: Roblox is right-handed with Y up. North is Roblox −Z, the same handedness change the Three.js game
uses for city.json points: (x, y, z) → (x, z, −y).
"""
from __future__ import annotations

import math

METRES_PER_STUD = 0.28
STUDS_PER_METRE = 1 / METRES_PER_STUD          # 3.5714285714…


def _finite(value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'expected a finite number, got {value!r}')
    return float(value)


def metres_to_studs(metres: float) -> float:
    return _finite(metres) / METRES_PER_STUD


def studs_to_metres(studs: float) -> float:
    return _finite(studs) * METRES_PER_STUD


def local_to_roblox(x_m: float, y_m: float, z_m: float = 0.0) -> tuple[float, float, float]:
    """OutbreakGeo point (metres; x east, y north, z up) → Roblox position in studs (X east, Y up, Z south)."""
    return (metres_to_studs(x_m), metres_to_studs(z_m), -metres_to_studs(y_m) + 0.0)


def roblox_to_local(x_studs: float, y_studs: float, z_studs: float) -> tuple[float, float, float]:
    """Roblox position in studs → OutbreakGeo point in metres (x east, y north, z up)."""
    return (studs_to_metres(x_studs), -studs_to_metres(z_studs) + 0.0, studs_to_metres(y_studs))
