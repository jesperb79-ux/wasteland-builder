# Third-party data and bundled assets

| What | Where | Source | License / terms |
|---|---|---|---|
| Map data and everything derived from it (generated `cities/*`, game packs) | `cities/` (not in git) | © OpenStreetMap contributors | ODbL 1.0 — attribute "© OpenStreetMap contributors", share-alike for derived databases |
| OpenStreetMap extract of central Borås (Outbreak P01 snapshot) | `pipeline/outbreak/snapshots/boras-p01/osm.json.gz` | © OpenStreetMap contributors | ODbL 1.0 (see the folder's `NOTICE.md`) |
| Terrain heights of central Borås (Outbreak P01 snapshot) | `pipeline/outbreak/snapshots/boras-p01/terrain.json.gz` | Copernicus DEM GLO-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018, provided under COPERNICUS by the European Union and ESA | Copernicus DEM licence: free of charge, attribution required |
| Place search | at runtime | Nominatim (OSMF) | [usage policy](https://operations.osmfoundation.org/policies/nominatim/): ≤1 request/s, identify the app |
| Map download | at runtime | OSM API, Overpass API mirrors | [OSM API usage policy](https://operations.osmfoundation.org/policies/api/), Overpass fair use |
| Vehicles (`game/public/models`, `game/art/battlecars.blend`) | game | Kalmar Wasteland, Anders Bjarby | **author to confirm before publishing** (suggested: CC BY 4.0) |
| Bitmaps (`game/public/art`) | game | generated with OpenAI image models for Kalmar Wasteland | **author to confirm** (suggested: CC BY 4.0) |
| Sound effects and generic narrator lines (`game/public/audio/sfx-*`, `voice-*`) | game | generated with ElevenLabs for Kalmar Wasteland | ElevenLabs terms for the generating account; **author to confirm** |
| Music (`game/public/audio/music*.mp3`) | game | generated with Suno for Kalmar Wasteland | Suno terms of the generating account (paid plans: owner may use commercially) — **author to confirm** |
| Procedural textures (`cache/textures`) | generated locally | `pipeline/make_textures.py` | same as the code (MIT) |
| ElevenLabs sound-effects skill notes | `.agents/skills/elevenlabs/references/sound-effects.md` | elevenlabs/skills | MIT |
| Image API wrapper | `.agents/skills/imagegen/scripts/generate_image.py` | this project | MIT |
| npm and Python dependencies | `game/package.json`, `pipeline/web/package.json`, `requirements.txt` | their authors | their own licenses (MIT, Apache-2.0, BSD …) |

Google Street View is used only as a visual reference during refinement rounds; no Google imagery is
stored in this repository or in generated cities.
