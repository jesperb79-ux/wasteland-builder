"""Outbreak real-city experiment (P00–P01): Wasteland's geography → OutbreakGeo, a Roblox-neutral format.

    adapter    city.json (+ osm.json, terrain.json), verified by a clean rebuild → OutbreakGeo v1
    chunks     the deterministic chunk grid
    units      metres ↔ studs (1 stud = 0.28 m) and the Roblox axis convention
    schema     checks documents against schemas/*.schema.json
    snapshot   retained real-source snapshots: create once, verify, build offline
    realcheck  P01 validation of a real OutbreakGeo against its snapshot

Run: .venv/bin/python pipeline/outbreak <slug>. See docs/outbreak-real-city.md.
"""
