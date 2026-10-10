# Planet Pilot open data

[Planet Pilot](https://planetpilot.world) draws its world from open map data. The game is proprietary, but the
map databases it derives from OpenStreetMap (through the Overture Maps Foundation releases) are covered by the
Open Database License (ODbL) 1.0, so they are offered here under the same license, with the scripts that build them.

Map data (c) OpenStreetMap contributors, ODbL 1.0. Overture Maps Foundation. Building footprints also from
Microsoft, Esri Community Maps contributors and Google (CC BY 4.0), via Overture Maps. Building heights also from
3D-GloBFP, Google Open Buildings 2.5D Temporal and GHSL (CC BY 4.0).

## Downloads

Each game release that changes the data has a zip on the [Releases](../../releases) page
(`planetpilot-odbl-v<version>.zip`, about 200 MB) with every derived database, the license and a
`SOURCES.txt` listing each file's source release and the command that built it.

## What is here

- `scripts/`: the bake scripts as they were at the latest data release. They need Python 3.12, numpy and the
  Overture release (`s3://overturemaps-us-west-2/release/<release id>/`); `bake_road_profiles.gd` runs in Godot 4.7.
  Each script's header describes its binary output format.
- `licenses/`: the ODbL text, the per-file sources, and the CC BY-SA 3.0 notice for the two models below.
- `models/`: the Christ the Redeemer and Mount Rushmore 3D models, adapted from CC BY-SA 3.0 photos and offered
  under CC BY-SA 3.0 (see `licenses/CC_BY_SA_MODELS.txt`).

The game world shown on screen is a Produced Work under the ODbL; only the databases are Derivative Databases.

Contact: privacy@planetpilot.world
