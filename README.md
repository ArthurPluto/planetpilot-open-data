# Planet Pilot open data

[Planet Pilot](https://planetpilot.world) draws its world from open map data. The game is proprietary, but the
map databases it derives from OpenStreetMap (through the Overture Maps Foundation releases) are covered by the
Open Database License (ODbL) 1.0, so they are offered here under the same license, with the scripts that build them.

Map data (c) OpenStreetMap contributors, ODbL 1.0. Overture Maps Foundation. Building footprints also from
Microsoft, Esri Community Maps contributors and Google (CC BY 4.0), via Overture Maps. Building heights also from
3D-GloBFP, Google Open Buildings 2.5D Temporal and GHSL (CC BY 4.0).
The Odesa ground also uses Copernicus DEM GLO-30 (c) DLR e.V. 2010-2014 and (c) Airbus Defence and Space GmbH
2014-2018 provided under COPERNICUS by the European Union and ESA.

## Downloads

Each game release that changes the data has a zip on the [Releases](../../releases) page
(`planetpilot-odbl-v<version>.zip`, about 220 MB) with every derived database (roads, road graph and profiles, far roads,
buildings and green for 73 city and region packs, city water, coast patches, the Great Wall route, the Odesa city pack,
port props, the Lake Ontario patch, landmark positions), the license and a `SOURCES.txt` listing each file's source
release and the command that built it. The latest release is the one that matches the game on Steam.

## What is here

- `scripts/`: the bake scripts as they were at the latest data release (version 73). They need Python 3.12, numpy and the
  Overture release (`s3://overturemaps-us-west-2/release/<release id>/`); some also need scipy, shapely or zstandard,
  and `bake_road_profiles.gd` runs in Godot 4.7. Each script's header describes its binary output format. Several
  import small helper modules of the game's `tools/` folder (for example `region_pack`, `world_scale`) that are not
  published here; the headers say what they do.
- `licenses/`: the ODbL text, the per-file sources, and the CC BY-SA 3.0 notice for the two models below.
- `models/`: the Christ the Redeemer and Mount Rushmore 3D models, adapted from CC BY-SA 3.0 photos and offered
  under CC BY-SA 3.0 (see `licenses/CC_BY_SA_MODELS.txt`).

The game world shown on screen is a Produced Work under the ODbL; only the databases are Derivative Databases.

Contact: privacy@planetpilot.world
