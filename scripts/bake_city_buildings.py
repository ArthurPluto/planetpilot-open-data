"""Detailed cities: real building footprints from Overture Maps (buildings theme, OpenStreetMap + ML footprints,
ODbL) baked into data/world/odbl/buildings/<region>.bin (CBL1, one file per region) plus its index
data/world/odbl/buildings/index.bin (RPX1, tools/region_pack.py) for scripts/world/city_buildings.gd.

    python tools/bake_city_buildings.py fetch [--only israel,nyc]   download the region caches (DuckDB, S3)
    python tools/bake_city_buildings.py bake  [--only ...]           bake the regions' files, then the index
    python tools/bake_city_buildings.py info  [path]                 per-region counts (path: a folder, or an old
                                                                     single buildings.bin)
    python tools/bake_city_buildings.py index                         rebuild the index from the region files there

A bake of some regions (--only) rewrites only their files, their rows of city_towers.json and the index; the other
regions' files stay. CITY_OUT=<folder>: the output base for a trial bake (buildings/, city_towers.json, and
tools/bake_city_green.py's green/ go there; default data/world/odbl; a path ending in .bin means its folder).

Scale (the game planet is 1/5 real size, Geo.SCALE 5): towns.gd draws every house REAL size on the 1/5 ground (a
city footprint shrinks, a house stays a house: aircraft, cars and runways are real size too). The detailed cities
keep that convention: every footprint keeps its real shape, size, orientation and height, and stands at its real
lat / lon (which the game maps 1/5). The 1/5 ground holds about 1/25 of a city's buildings, so the bake SELECTS:
greedy by importance (height x footprint, landmark towers first), a building is kept when its footprint (plus a
street gap) clears every one kept before it, the drawn graph roads and the city water. The tallest towers and the
large blocks stand exactly where they are; the gaps between them fill with the next ones, so a district keeps its
real skyline, grain and orientation at the density the ground can hold.

Towers and landmarks beat roads (v2): a building of YIELD_M or more, Arthur's home and everything in the Ma'agal
HaShalom block (MAAGAL_R round the home) is kept where it really stands even where a drawn road crosses it; its record
carries F_YIELD and the game hides the road ribbon and its cars there (RoadBuild.tower_fix), so no road runs through
a tower. The greedy order puts the named (known_towers.json) and the tallest first, then the rest.

Sources: Overture Maps buildings release OVERTURE (theme=buildings/type=building), read with DuckDB straight from
the public S3 bucket, one bbox query per region; caches data/world/src/overture/buildings/<region>.parquet
(gitignored, never exported). Heights: Overture height, else num_floors x FLOOR_M, else the GHSL average net building
height of the 100 m cell with a hash spread, else a class default; then raised where the floors say taller
(bad Overture data) and to the Wikidata height of a known tower (data/world/known_towers.json, CC0) when the
Overture height is missing or far under it. Arthur's home: HOME_H. The cache's box sits beside it (<cache>.box.json):
a region whose box grew is fetched again (the old cache moved aside, never deleted).

Bake quality (tools/tests/test_bake_rules.py): a known tower raises, of the footprints holding its point, the one
whose own height is nearest, else the SMALLEST, never one over BIG_FP (podiums, malls, complexes), and not when a
tower already stands beside it (NEXT_R) or a twin of its height stands within PHANTOM_R (a duplicate point); a
mosque's or monument's Wikidata height is its minaret's; masts and TV towers (NOT_A_BUILDING, the Wikidata kind
"mast") and the registered landmarks' own footprints (data/landmarks: by name, or at the real spot) are left out;
sanity: an Overture height over 2 x its floors x 3.3 + 10 m, a 150 m+ height on a field / industrial footprint, an
unnamed needle, a lone spike the GHSL cell does not back, the GHSL guess and unbacked Overture heights on huge
footprints. Per-region fixes: region_overrides() (height_fixes / tower_fixes, towers, drop / drops, qid_fixes).

CBL1 (little endian; version 3: one file per region, n_regions 1; version 2 was one file for every region):
  header 32 B : "CBL1", u32 version, u32 n_tiles, u32 n_regions, u32 regions_off, u32 cover_off, u32 n_cover, u32 0
  tiles  n x 32 B (sorted by key): u32 key ((ty * TPD_LON) + tx, 1/TPD degree tiles, ty from the south pole),
           u32 off, u32 bytes (zlib), u32 n_bld, u32 n_pts, u16 region, u16 flags, f32 max_h, u32 0
  regions n x 64 B: char[24] name, f32 lat0, lon0, lat1, lon1, u32 kept, u32 source, u32 0, u32 0
  cover  n_cover x u32: sorted keys of covered 1/COVER_PD degree cells (gy * 360 * COVER_PD + gx, gy from the south)
  tile payload (zlib): buildings n x 12 B: u16 first point, u8 n_pts, u8 kind, u16 height (0.1 m), u16 min_h (0.1 m),
           u8 roof, u8 shade, u16 flags (F_YIELD 1: roads yield to it, F_BANK 2: a tower on the water edge); then points n_pts x 4 B: i16 east, i16 north (0.25 toy m from the tile's SW
           corner, counter-clockwise, not closed)
  kind: 0 residential, 1 commercial, 2 industrial, 3 civic, 4 landmark-tall (>= TALL_M)
  roof: 0 flat, 1 gabled, 2 hipped, 3 pyramidal, 4 dome
"""
import argparse
import hashlib
import json
import math
import os
import re
import struct
import sys
import time
import zlib

import region_pack  # noqa: E402 (tools/region_pack.py)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.environ.get("CITY_SRC") or os.path.join(ROOT, "data", "world", "src")   # (CITY_SRC: a shared raw cache)
CACHE = os.path.join(SRC, "overture", "buildings")
OUT_BASE = os.path.join(ROOT, "data", "world", "odbl")
TOWNS_BIN = os.path.join(ROOT, "data", "world", "towns", "towns.bin")
ROADGRAPH = os.path.join(ROOT, "data", "world", "odbl", "roadgraph.bin")
CITY_WATER = os.path.join(ROOT, "data", "world", "odbl", "city_water.bin")

OVERTURE = "2026-09-23.1"
BUILDINGS = "s3://overturemaps-us-west-2/release/%s/theme=buildings/type=building/*.parquet" % OVERTURE

SCALE = 5.0                       # Geo.SCALE
R_TOY = 6371000.0 / SCALE         # Geo.R (toy m)
M_PER_DEG = R_TOY * math.pi / 180.0   # 22,239 toy m per degree of latitude
TPD = 32                          # tiles per degree (lat and lon): 1/32 deg, ~695 toy m N-S
COVER_PD = 240                    # cover cells per degree (TownData.CELLS: ~93 toy m)
Q = 0.25                          # point quantum (toy m)
TALL_M = 120.0                    # landmark-tall: drawn from far, a slimmer gap
YIELD_M = 60.0                    # a building this tall is kept on a drawn road: the road yields (F_YIELD)
F_YIELD = 1
F_BANK = 2                        # stands on the city water's edge (a bank tower): the game checks only its centre
BANK_NUDGE = 24.0                 # toy m: a footprint touching the city water moves at most this far landward
BANK_DRY = 3.0                    # toy m: a bank tower that cannot clear stays when its centre is this far on land
FLOOR_M = 3.3                     # storey height (num_floors x FLOOR_M)
HOME_H = 52.0                     # Ma'agal HaShalom 1: 16 storeys, 52 m
MAAGAL_R = 45.0                   # toy m round the home: the Ma'agal HaShalom block (every building there yields)
KNOWN = os.path.join(ROOT, "data", "world", "known_towers.json")   # (tools/fetch_known_towers.py)
TOWER_LIST_M = 100.0
TALL_GAP = 2.0                    # toy m between two kept towers (F_YIELD both)
MAX_FLOORS = 110                  # a floor count over this is a typo (Israel's "189 floors", London's 156)
# not a building's mass: lattice masts, antennas, water towers, TV and radio towers (a solid block of their height
# would be wrong; the famous ones are landmark-model candidates), in the local scripts too
NOT_A_BUILDING = re.compile(
    r"(?i)\b(mast|antenna|transmitter|transmission tower|radio tower|(?<!at )water tower(?! place)|sutro tower|tv tower|"
    r"television tower|telecom(munications?)? tower|communications? tower|broadcasting tower|radio (and|&) tv tower|"
    r"radio (and|&) television tower|fernsehturm|funkturm|sendeturm|torre de (tv|televis[aã]o|televisi[oó]n|"
    r"comunica[cç][aãoõ](es)?)|torre de transmiss[aã]o|tour hertzienne|tour de t[eé]l[eé]vision|"
    r"tower of (radio|tv|television)|milad tower|baghdad tower|canton tower|tokyo skytree|tokyo tower|"
    r"pitampura tv tower|tour de l.[eé]changeur|torrespa[nñ]a|torre entel|[cç]aml[iı]ca (tower|kulesi)|panchdeep tower|[eé]metteur|kyoto tower|yokohama marine tower|beijing olympic tower)\b"
    r"|奥林匹克塔|广州塔|廣州塔|京都タワー$|电视塔|电视发射塔|发射塔|广播电视塔|电视台发射塔|廣播電視塔|電視塔|電波塔|テレビ塔|送信所|"
    r"برج میلاد|برج مخابرات|برج تلویزیون|برج بغداد|برج القاهرة|برج الاتصالات|برج التلفزيون|"
    r"टीवी टावर|दूरदर्शन टावर|टेलीविजन टावर|송신소|방송탑|"
    r"телебашня|радиобашня|телевизионная башня")
# under construction, proposed, stalled or cancelled (2026-10): their Overture / Wikidata height is the plan, not
# what stands
UNBUILT = re.compile(
    r"(?i)^(spiral tower|burj binghatti.*|rama ix super tower|first ?bank (of )?nigeria (headquarters|hq).*|"
    r"pertamina energy tower|peruri 88|riverview plaza.*|world trade cent(re|er) kolkata|saman faraz.*|"
    r"supernova spira|ddt sky tower|the stratford residences|dream tower|seoul lite|tower infinity|"
    r"india tower( mumbai)?|diamond of istanbul|(tour )?triangle|old chicago main post office twin towers?|"
    r"tribune east tower|waldorf.astoria hotel (and|&) residence tower.*|park michigan|american commerce center|"
    r"waldorf astoria philadelphia|one bayfront plaza|1101 brickell|infinity ii)$")
# a Wikidata height of a structure of worship or a monument is its spire, dome or minaret, not the footprint's
WORSHIP = re.compile(r"(?i)\b(mosque|masjid|musalla|minaret|temple|church|cathedral|basilica|chapel|pagoda|mandir|"
                     r"gurdwara|monument|memorial|mausoleum|shrine|obelisk|stupa)\b|清真寺|教堂|寺$|مسجد|مصلى|جامع|"
                     r"मंदिर|मस्जिद|গির্জা|মসজিদ|catedral|igreja|mesquita|iglesia|mezquita")
BIG_FP = 8000.0                   # m2: a Wikidata height of 60 m and up never raises a footprint this big (podiums,
                                  # malls, mosques, complexes: the tower's point lands in its podium)
PHANTOM_R = 600.0                 # m: a Wikidata item whose height a footprint within this already has is a duplicate
MAST_PARTS_R = 40.0               # m
MAST_FP = 3000.0                  # m2: the biggest footprint a Wikidata mast's point drops (its own shaft or base)
NEXT_FP = 3000.0                  # m2: a footprint this big holding a tower point next to a tower is its podium
NEXT_R = 80.0                     # m: a footprint this near with 0.6 x the item's height or more is the item's tower
SLAB_A = (6000.0, 15000.0)        # m2: GHSL guesses capped to SLAB_H on footprints this big, and an unsupported
SLAB_H = (50.0, 30.0)             # Overture height on the biggest capped at SLAB_OV_H
SLAB_OV_H = 45.0
# Wikidata items with a known-bad height or spot: qid -> metres (0 = skip the item) or {"h", "lat", "lon"} (any of
# them; the region JSON's "qid_fixes" adds to these)
QID_FIXES = {
    "Q24942115": 0,                                    # Torres Carlos Antunez, Santiago: "448 m" on ~30 floors
    "Q649218": 0,                                      # India Tower, Mumbai: a stalled foundation pit
    "Q8772052": {"lat": 19.44993, "lon": -99.13756},   # Edificio Tlatelolco: its point sits on the SRE at the Alameda
    "Q497536": {"h": 47.0},                            # Jatiya Sangsad Bhaban, Dhaka: P2048 155 is feet
    "Q6150039": {"h": 235.0},                          # Torre BBVA Mexico: 157 floors in Wikidata, 50 real
}
# height source index (city_towers.json height_source): 4 = the 3D-GloBFP footprint dataset, 5 = an Overture building
# part, 6 = Google Open Buildings 2.5D (tools/fetch_heights.py)
HEIGHT_SOURCES = ["overture", "floors", "ghsl", "wikidata", "3d-globfp", "overture-part", "google-2.5d"]
CRLF = chr(13) + chr(10)
VERSION = 3
MAGIC = b"CBL1"

# Region boxes (lat0, lon0, lat1, lon1) fetched from Overture. "poly" clips the kept set (lon, lat ring); "zones"
# raise the detail (a smaller street gap, smaller buildings kept) inside boxes: (lat0, lon0, lat1, lon1, level)
# with level 2 = the highest (Arthur's neighbourhood), 1 = high.
REGIONS = {
    "israel": {
        "box": (29.45, 34.20, 33.35, 35.92),
        # Israel with Jerusalem and the West Bank hills, the Golan; not Gaza, Sinai, Lebanon, Syria or Jordan
        "poly": [(34.27, 31.22), (34.49, 31.59), (34.57, 31.68), (34.75, 32.10), (34.89, 32.60), (34.96, 32.83),
                 (35.07, 33.09), (35.10, 33.09), (35.50, 33.10), (35.57, 33.27), (35.62, 33.29), (35.82, 33.33),
                 (35.89, 32.94), (35.80, 32.71), (35.57, 32.64), (35.55, 32.39), (35.57, 32.10), (35.53, 31.75),
                 (35.40, 31.50), (35.47, 31.10), (35.33, 30.55), (35.10, 29.95), (34.96, 29.49), (34.88, 29.52),
                 (34.27, 31.22)],
        # Arthur's home: Ma'agal HaShalom 1, Rishon LeZion (OSM way 371637662); its footprint is always kept
        "home": (31.9695, 34.7684),
        "zones": [(31.9545, 34.7504, 31.9845, 34.7864, 2),   # Rishon LeZion: Ma'agal HaShalom, centred on home
                  (31.935, 34.73, 32.00, 34.83, 1),      # Rishon LeZion
                  (32.03, 34.74, 32.15, 34.85, 1),       # Tel Aviv
                  (31.74, 35.17, 31.82, 35.25, 1),       # Jerusalem
                  (32.76, 34.95, 32.84, 35.05, 1)],      # Haifa
    },
    "moscow": {"box": (55.55, 37.30, 55.95, 37.95)},
    "nyc": {"box": (40.55, -74.10, 40.92, -73.70)},
    "dubai": {"box": (24.95, 55.05, 25.35, 55.45)},
    "london": {"box": (51.38, -0.35, 51.62, 0.15)},
    "lasvegas": {"box": (36.00, -115.35, 36.32, -115.00)},
    "sf": {"box": (37.69, -122.53, 37.86, -122.20)},
    "bangkok": {"box": (13.60, 100.40, 13.92, 100.75)},
}

# More regions, one JSON file per city in tools/city_regions/<name>.json (so parallel workers never edit the same
# file): {"box": [lat0, lon0, lat1, lon1], "poly": [[lon, lat], ...], "zones": [[lat0, lon0, lat1, lon1, level], ...]}
# ("poly" and "zones" optional; any other keys are notes for people). A name already above is never overridden.
REGIONS_DIR = os.path.join(ROOT, "tools", "city_regions")
OVERRIDE_KEYS = ("height_fixes", "tower_fixes", "towers", "drop", "drops", "qid_fixes")
if os.path.isdir(REGIONS_DIR):
    for _f in sorted(os.listdir(REGIONS_DIR)):
        if _f.endswith(".json") and _f[:-5] not in REGIONS:
            with open(os.path.join(REGIONS_DIR, _f), encoding="utf-8") as _h:
                _cfg = json.load(_h)
            _r = {"box": tuple(_cfg["box"])}
            if _cfg.get("poly"):
                _r["poly"] = [tuple(p) for p in _cfg["poly"]]
            if _cfg.get("zones"):
                _r["zones"] = [tuple(z) for z in _cfg["zones"]]
            for _k in OVERRIDE_KEYS:
                if _cfg.get(_k):
                    _r[_k] = _cfg[_k]
            REGIONS[_f[:-5]] = _r
# Per-region bake overrides (region_overrides()), from the region JSON or from tools/city_regions/fixes/<id>.json
# (same keys, merged after; a fixes file may name a built-in region too):
#   "height_fixes" / "tower_fixes": [[lat, lon, h, "note"], ...] or [{"lat", "lon", "h", "note"}]: the footprint at
#       the point (of those holding it, else the nearest within 30 m: the tallest when h lowers it, else the
#       smallest) gets h m; h 0 = no trusted height (the
#       Overture / Wikidata height is dropped: the floors, else the GHSL guess). A plain string is a note (ignored).
#   "towers": [{"name", "lat", "lon", "h"}]: a tower Overture and Wikidata miss, raised like a known tower (the
#       footprint whose own height is nearest h, else the smallest holding the point; never one over BIG_FP unless
#       "any_size": true); "add": [w, d] (m) puts a w x d footprint there first (Overture has only the site outline)
#   "drop" / "drops": [[lat, lon, "note"], ...] or [{"lat", "lon", "r", "all", "reason"}]: the footprint at the point
#       is removed (r m: every footprint with its centroid within r; "all": every footprint holding the point)
#   "qid_fixes": {"Q123": h}: a Wikidata item's height (0 = skip the item)
FIXES_DIR = os.path.join(REGIONS_DIR, "fixes")
if os.path.isdir(FIXES_DIR):
    for _f in sorted(os.listdir(FIXES_DIR)):
        if _f.endswith(".json") and _f[:-5] in REGIONS:
            with open(os.path.join(FIXES_DIR, _f), encoding="utf-8") as _h:
                _cfg = json.load(_h)
            for _k in OVERRIDE_KEYS:
                if _cfg.get(_k):
                    _o = REGIONS[_f[:-5]].get(_k)
                    REGIONS[_f[:-5]][_k] = dict(_o or {}, **_cfg[_k]) if _k == "qid_fixes" else list(_o or []) + list(_cfg[_k])

# CITY_OUT: write the bake somewhere else (a worker's trial bake): the output base folder
if os.environ.get("CITY_OUT"):
    OUT_BASE = os.environ["CITY_OUT"]
    if OUT_BASE.lower().endswith(".bin"):
        OUT_BASE = os.path.dirname(OUT_BASE)
OUT_DIR = os.path.join(OUT_BASE, "buildings")
TOWERS_OUT = os.path.join(OUT_BASE, "city_towers.json")
SOURCES_TXT = os.path.join(OUT_BASE, "SOURCES.txt")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def cache_path(name):
    return os.path.join(CACHE, name + ".parquet")


# The fetched box beside each cache (<cache>.box.json): a region whose box grew is fetched again instead of baking
# from the smaller cache. A cache from before the sidecar is judged by its data's extent (the footprints reach
# within BOX_TOL of every side of the box it was fetched for).
BOX_TOL = 0.02


def write_box(path, box):
    with open(path + ".box.json", "w", encoding="utf-8") as f:
        json.dump({"box": [float(v) for v in box]}, f)


def box_holds(outer, inner, tol=1e-6):
    return (outer[0] <= inner[0] + tol and outer[1] <= inner[1] + tol and outer[2] >= inner[2] - tol
            and outer[3] >= inner[3] - tol)


def cache_box(con, path):
    """The box the cache at path was fetched for: the sidecar's, else its data extent padded by BOX_TOL (written to
    nothing written), None when unknown."""
    side = path + ".box.json"
    if os.path.exists(side):
        with open(side, encoding="utf-8") as f:
            return tuple(json.load(f)["box"])
    if con is None:
        return None
    y0, x0, y1, x1 = con.execute("SELECT min(y0), min(x0), max(y1), max(x1) FROM read_parquet('%s')"
                                 % path.replace(chr(92), "/")).fetchone()
    if y0 is None:
        return None
    return (y0 - BOX_TOL, x0 - BOX_TOL, y1 + BOX_TOL, x1 + BOX_TOL)


def cache_covers(con, path, box):
    b = cache_box(con, path)
    return b is not None and box_holds(b, box)


def set_aside(path):
    """Moves a stale cache (and its sidecar) aside, never deleting: <cache>.oldbox<n>."""
    k = 1
    while os.path.exists("%s.oldbox%d" % (path, k)):
        k += 1
    dst = "%s.oldbox%d" % (path, k)
    os.replace(path, dst)
    if os.path.exists(path + ".box.json"):
        os.replace(path + ".box.json", dst + ".box.json")
    return dst


# ---------------------------------------------------------------- fetch

def fetch(names):
    import duckdb
    os.makedirs(CACHE, exist_ok=True)
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs; INSTALL spatial; LOAD spatial;")
    con.execute("SET s3_region='us-west-2'; SET enable_http_metadata_cache=true; SET parquet_metadata_cache=true;")
    for name in names:
        path = cache_path(name)
        if os.path.exists(path):
            if cache_covers(con, path, REGIONS[name]["box"]):
                log("fetch %s: cached (%.1f MB)" % (name, os.path.getsize(path) / 1e6))
                continue
            aside = set_aside(path)
            log("fetch %s: the cache was fetched for a smaller box, moved to %s; fetching again" % (name, aside))
        la0, lo0, la1, lo1 = REGIONS[name]["box"]
        t0 = time.time()
        q = f"""
        COPY (
          SELECT id, height, num_floors, class, subtype, names.primary AS name, roof_shape, roof_height, min_height,
                 has_parts, ST_AsWKB(geometry) AS wkb,
                 bbox.xmin AS x0, bbox.ymin AS y0, bbox.xmax AS x1, bbox.ymax AS y1
          FROM read_parquet('{BUILDINGS}', hive_partitioning = 0)
          WHERE bbox.xmin >= {lo0} AND bbox.xmax <= {lo1} AND bbox.ymin >= {la0} AND bbox.ymax <= {la1}
            AND coalesce(is_underground, false) = false
        ) TO '{path.replace(chr(92), '/')}.tmp' (FORMAT parquet, COMPRESSION zstd)"""
        con.execute(q)
        os.replace(path + ".tmp", path)
        write_box(path, REGIONS[name]["box"])
        n = con.execute("SELECT count(*) FROM read_parquet('%s')" % path.replace(chr(92), '/')).fetchone()[0]
        log("fetch %s: %d buildings, %.1f MB, %.0f s" % (name, n, os.path.getsize(path) / 1e6, time.time() - t0))




# ---------------------------------------------------------------- GHSL building heights (100 m, Mollweide tiles)

GHSL_DIR = os.path.join(SRC, "ghsl", "built_h_100")
GHSL_URL = ("https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_BUILT_H_GLOBE_R2023A/"
            "GHS_BUILT_H_ANBH_E2018_GLOBE_R2023A_54009_100/V1-0/tiles/"
            "GHS_BUILT_H_ANBH_E2018_GLOBE_R2023A_54009_100_V1_0_R%d_C%d.zip")
MOLL_R = 6378137.0                # ESRI:54009 (GHSL) spherical Mollweide on the WGS84 semi-major axis (6371007.181 sampled 3 to 7 km off)
MOLL_X0 = -18041000.0
MOLL_Y0 = 9000000.0
GHSL_TILE = 1000000.0
GHSL_PX = 100.0


def moll(lat, lon):
    """Mollweide (ESRI:54009) of numpy lat / lon arrays (deg)."""
    import numpy as np
    phi = np.radians(np.asarray(lat, np.float64))
    lam = np.radians(np.asarray(lon, np.float64))
    th = phi.copy()
    k = math.pi * np.sin(phi)
    for _ in range(30):
        f = 2 * th + np.sin(2 * th) - k
        d = 2 + 2 * np.cos(2 * th)
        safe = np.abs(d) > 1e-9
        th = th - np.where(safe, f / np.where(safe, d, 1.0), 0.0)
    x = MOLL_R * 2 * math.sqrt(2) / math.pi * lam * np.cos(th)
    y = MOLL_R * math.sqrt(2) * np.sin(th)
    return x, y


def ghsl_tiles(box):
    import numpy as np
    la0, lo0, la1, lo1 = box
    la, lo = np.meshgrid(np.linspace(la0, la1, 12), np.linspace(lo0, lo1, 12))
    x, y = moll(la.ravel(), lo.ravel())
    rows = np.floor((MOLL_Y0 - y) / GHSL_TILE).astype(int) + 1
    cols = np.floor((x - MOLL_X0) / GHSL_TILE).astype(int) + 1
    return sorted(set(zip(rows.tolist(), cols.tolist())))


def fetch_ghsl(names):
    import requests
    os.makedirs(GHSL_DIR, exist_ok=True)
    for name in names:
        for r, c in ghsl_tiles(REGIONS[name]["box"]):
            path = os.path.join(GHSL_DIR, "R%d_C%d.zip" % (r, c))
            if os.path.exists(path):
                continue
            t0 = time.time()
            with requests.get(GHSL_URL % (r, c), stream=True, timeout=120) as resp:
                resp.raise_for_status()
                with open(path + ".tmp", "wb") as f:
                    for chunk in resp.iter_content(1 << 20):
                        f.write(chunk)
            os.replace(path + ".tmp", path)
            log("ghsl R%d_C%d: %.1f MB, %.0f s" % (r, c, os.path.getsize(path) / 1e6, time.time() - t0))


class Ghsl:
    """Average net building height (m) at 100 m, uint8 per tile (0 = none; nodata 255 folded to 0)."""

    def __init__(self):
        self.tiles = {}

    def _tile(self, r, c):
        import io
        import zipfile
        import numpy as np
        import tifffile
        key = (r, c)
        if key not in self.tiles:
            path = os.path.join(GHSL_DIR, "R%d_C%d.zip" % key)
            a = None
            if os.path.exists(path):
                z = zipfile.ZipFile(path)
                n = [x for x in z.namelist() if x.endswith(".tif")][0]
                pg = tifffile.TiffFile(io.BytesIO(z.read(n))).pages[0]
                tp = pg.tags["ModelTiepointTag"].value
                f = pg.asarray()
                f[(f >= 254.5) | ~np.isfinite(f)] = 0.0
                # (edge tiles are smaller: their own origin)
                a = (np.clip(np.round(f), 0, 254).astype(np.uint8), float(tp[3]), float(tp[4]))
            self.tiles[key] = a
        return self.tiles[key]

    def sample(self, lat, lon):
        """(centre cell, 3 x 3 max) in m, for numpy arrays."""
        import numpy as np
        x, y = moll(lat, lon)
        gx = np.floor((x - MOLL_X0) / GHSL_PX).astype(np.int64)
        gy = np.floor((MOLL_Y0 - y) / GHSL_PX).astype(np.int64)
        cen = np.zeros(len(gx), np.float32)
        mx = np.zeros(len(gx), np.float32)
        tr = gy // 10000
        tc = gx // 10000
        for key in sorted(set(zip(tr.tolist(), tc.tolist()))):
            sel = np.nonzero((tr == key[0]) & (tc == key[1]))[0]
            t = self._tile(key[0] + 1, key[1] + 1)
            if t is None:
                continue
            a, tx0, ty0 = t
            hh, ww = a.shape
            py = gy[sel] - int(round((MOLL_Y0 - ty0) / GHSL_PX))
            px = gx[sel] - int(round((tx0 - MOLL_X0) / GHSL_PX))
            inside = (py >= 0) & (py < hh) & (px >= 0) & (px < ww)
            py = np.clip(py, 0, hh - 1)
            px = np.clip(px, 0, ww - 1)
            cen[sel] = np.where(inside, a[py, px], 0)
            m = np.zeros(len(sel), np.float32)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    m = np.maximum(m, a[np.clip(py + dy, 0, hh - 1), np.clip(px + dx, 0, ww - 1)])
            mx[sel] = np.where(inside, m, 0)
        return cen, mx


# ---------------------------------------------------------------- keep-outs known at bake time

def road_lines(box):
    """The drawn graph roads (every class but ferries) near box: [(N x 2 lat, lon array, half width m)].
    A minimal RGR1 v2 reader (RoadGraph._decode layout)."""
    import numpy as np
    import zstandard
    data = open(ROADGRAPH, "rb").read()
    assert data[:4] == b"RGR1"
    ver, flags, nt, toff = struct.unpack_from("<4I", data, 4)
    recs = [struct.unpack_from("<IIIIIIIHH", data, toff + 32 * t) for t in range(nt)]
    la0, lo0, la1, lo1 = box
    pays = {}

    def cell(key):
        k = (key - 65536) // 16 if key >= 65536 else key
        return k // 360 - 90, k % 360 - 180

    def payload(t):
        if t not in pays:
            k, off, nb, nn, ne, na, npt, nst, nsk = recs[t]
            raw = 16 * nn + 8 * na + 32 * ne + 4 * npt + 16 * nst + 16 * nsk
            pay = data[off:off + nb]
            if flags & 4:
                pay = zstandard.ZstdDecompressor().decompress(pay, max_output_size=raw)
            pays[t] = pay
        return pays[t]

    def node(t, i):
        return struct.unpack_from("<ii", payload(t), 16 * i)

    out = []
    for t in range(nt):
        cla, clo = cell(recs[t][0])
        if cla + 1 < la0 - 0.3 or cla > la1 + 0.3 or clo + 1 < lo0 - 0.3 or clo > lo1 + 0.3:
            continue
        k, off, nb, nn, ne, na, npt, nst, nsk = recs[t]
        pay = payload(t)
        eo = 16 * nn + 8 * na
        po = eo + 32 * ne
        for e in range(ne):
            a, b, bt, cls, fl, lanes, wm, nid, pt_off, n_pts = struct.unpack_from("<IIHBBBBHIH", pay, eo + 32 * e)
            if fl & 32:
                continue
            qy, qx = node(t, a)
            pts = [(qy, qx)]
            for p in range(n_pts):
                dy, dx = struct.unpack_from("<hh", pay, po + 4 * (pt_off + p))
                qy += dy * 16
                qx += dx * 16
                pts.append((qy, qx))
            pts.append(node(bt, b))
            arr = np.array(pts, np.float64) * 1e-6
            if arr[:, 0].max() < la0 or arr[:, 0].min() > la1 or arr[:, 1].max() < lo0 or arr[:, 1].min() > lo1:
                continue
            out.append((arr, max(wm, 6) * 0.5))
    return out


class CityWater:
    """city_water.bin query raster (CWQ1): signed toy m distance to the city water (+ outside, 1e9 = none)."""

    def __init__(self):
        import numpy as np
        self.tiles = {}
        self.index = {}
        self.data = b""
        if not os.path.exists(CITY_WATER):
            return
        b = open(CITY_WATER, "rb").read()
        o = b.find(b"CWQ1")
        if o < 0:
            return
        nt, rng, qn = struct.unpack_from("<IfI", b, o + 4)
        self.qn = qn
        for i in range(nt):
            k, off, sz = struct.unpack_from("<III", b, o + 16 + 12 * i)
            self.index[k] = (off, sz)
        self.data = b
        t = (np.arange(256) - 127.5) / 127.5
        self.lut = (np.sign(t) * t * t * rng).astype(np.float32)

    def dist(self, lat, lon):
        import numpy as np
        lat = np.asarray(lat, np.float64)
        lon = np.asarray(lon, np.float64)
        out = np.full(len(lat), 1e9, np.float32)
        if not self.index:
            return out
        tx = np.floor((lon + 180.0) / 0.1).astype(np.int64)
        ty = np.floor((90.0 - lat) / 0.1).astype(np.int64)
        key = ty * 3600 + tx
        for k in np.unique(key):
            k = int(k)
            if k not in self.index:
                continue
            if k not in self.tiles:
                off, sz = self.index[k]
                blob = self.data[off:off + sz]
                try:
                    raw = zlib.decompress(blob)
                except zlib.error:
                    raw = zlib.decompress(blob, -15)
                self.tiles[k] = np.frombuffer(raw, np.uint8).reshape(self.qn, self.qn)
            a = self.tiles[k]
            sel = np.nonzero(key == k)[0]
            u = (lon[sel] + 180.0 - tx[sel] * 0.1) * 2400.0
            v = (90.0 - lat[sel] - ty[sel] * 0.1) * 2400.0
            i = np.clip(np.floor(u).astype(np.int64), 0, self.qn - 2)
            j = np.clip(np.floor(v).astype(np.int64), 0, self.qn - 2)
            fx = np.clip(u - i, 0, 1)
            fy = np.clip(v - j, 0, 1)
            L = self.lut
            d = (L[a[j, i]] * (1 - fx) + L[a[j, i + 1]] * fx) * (1 - fy) + \
                (L[a[j + 1, i]] * (1 - fx) + L[a[j + 1, i + 1]] * fx) * fy
            none = (a[j, i] == 255) & (a[j, i + 1] == 255) & (a[j + 1, i] == 255) & (a[j + 1, i + 1] == 255)
            out[sel] = np.where(none, 1e9, d)
        return out


# ---------------------------------------------------------------- known towers (Wikidata, CC0)

def feet_fix(h, fl):
    """A height that reads as feet against its floor count (over 7.5 m a storey, 2.6 to 5.5 as feet): metres."""
    if h and fl and h / fl > 7.5 and 2.6 <= h * 0.3048 / fl <= 5.5:
        return round(h * 0.3048, 1)
    return h


def known_towers(region, qid_fixes=None, skip_qids=(), log_to=None):
    """[{name, qid, lat, lon, h, fl, worship}] of data/world/known_towers.json for a region (height, else floors x
    FLOOR_M). Left out: masts and TV towers (kind "mast" or NOT_A_BUILDING), unbuilt towers, the registered landmarks
    (skip_qids: their model stands for them) and QID_FIXES / qid_fixes 0. A floor count over MAX_FLOORS is a typo; a
    height far over the item's own floors (6 m a storey + 30) is too: the floors."""
    if not os.path.exists(KNOWN):
        return []
    with open(KNOWN, encoding="utf-8") as f:
        d = json.load(f)
    fixes = dict(QID_FIXES, **(qid_fixes or {}))
    out = []
    for row in d["rows"]:
        reg, qid, nm, lat, lon, hm, fl, sl = row[:8]
        kd = row[8] if len(row) > 8 else ""
        if reg != region:
            continue
        fx = fixes.get(qid)
        fx = fx if isinstance(fx, dict) else {} if fx is None else {"h": fx}
        why = ("mast" if kd == "mast" or NOT_A_BUILDING.search(nm) else "unbuilt" if UNBUILT.match(nm)
               else "landmark" if qid in skip_qids else "qid_fix 0" if fx.get("h") == 0 else "")
        if why:
            if log_to is not None:
                log_to.append("wikidata %s %s: skipped (%s)" % (qid, nm, why))
            continue
        if fl and fl > MAX_FLOORS:
            fl = None
        hm = feet_fix(hm, fl)
        if hm and fl and hm > fl * 6.0 + 30.0:
            hm = None
        wh = float(fx.get("h") or hm or (fl * FLOOR_M if fl else 0.0))
        if wh >= 20.0:
            out.append({"name": nm, "qid": qid, "lat": float(fx.get("lat", lat)), "lon": float(fx.get("lon", lon)),
                        "h": wh, "fl": fl or 0,
                        "worship": kd == "worship" or bool(WORSHIP.search(nm))})
    return out


def region_overrides(cfg):
    """The region's fixes (OVERRIDE_KEYS) normalised: (heights [(lat, lon, h)], towers [(name, lat, lon, h, any)],
    drops [(lat, lon, r)])."""
    def pts(lst):
        for e in lst or []:
            if isinstance(e, str):
                continue # a note
            if isinstance(e, dict):
                yield e
            else:
                yield {"lat": e[0], "lon": e[1], "h": e[2] if len(e) > 2 and not isinstance(e[2], str) else None,
                       "r": e[2] if len(e) > 2 and not isinstance(e[2], str) else None}
    heights = [(float(e["lat"]), float(e["lon"]), float(e.get("h") or 0.0))
               for k in ("height_fixes", "tower_fixes") for e in pts(cfg.get(k))]
    towers = [(str(e.get("name", "")), float(e["lat"]), float(e["lon"]), float(e["h"]), bool(e.get("any_size")))
              for e in pts(cfg.get("towers"))]
    drops = [(float(e["lat"]), float(e["lon"]), -1.0 if e.get("all") else float(e.get("r") or 0.0))
             for k in ("drop", "drops") for e in pts(cfg.get(k))]
    return heights, towers, drops


def add_footprints(cfg, ids, hgt, flo, cls, sub, roofs, names, clon, clat, loc, area, zl, npts):
    """The region's "towers" with an "add" size: a w x d m footprint (Overture height h) appended at the point."""
    import numpy as np
    import shapely
    adds = [e for e in cfg.get("towers") or [] if isinstance(e, dict) and e.get("add")]
    if not adds:
        return ids, hgt, flo, cls, sub, roofs, names, clon, clat, loc, area, zl, npts
    w = np.array([float(e["add"][0]) * 0.5 for e in adds])
    d = np.array([float(e["add"][1]) * 0.5 for e in adds])
    new = shapely.orient_polygons(shapely.box(-w, -d, w, d))

    def cat(a, b):
        return np.concatenate([a, np.asarray(b, dtype=a.dtype)])
    return (cat(ids, ["fix:%s:%d" % (e.get("name", ""), k) for k, e in enumerate(adds)]),
            cat(hgt, [float(e["h"]) for e in adds]), cat(flo, [np.nan] * len(adds)), cat(cls, [None] * len(adds)),
            cat(sub, [None] * len(adds)), cat(roofs, [None] * len(adds)), cat(names, [e.get("name") for e in adds]),
            cat(clon, [float(e["lon"]) for e in adds]), cat(clat, [float(e["lat"]) for e in adds]), cat(loc, new),
            cat(area, shapely.area(new)), cat(zl, [0] * len(adds)), cat(npts, [4] * len(adds)))


LANDMARKS_DIR = os.path.join(ROOT, "data", "landmarks")
LANDMARK_FP_MAX = 300.0           # m: a bigger landmark (a park, a palace, an island) drops only its named copies
LANDMARK_R_MAX = 150.0


def landmarks():
    """The registered landmarks: [{id, names (casefolded), qid, lat, lon (the real spot: real.json, else the
    registry), r (m: the footprint radius, capped), big (too big for the radius rule), site (the model's spot:
    sites.json, else the registry)}]."""
    if not os.path.isdir(LANDMARKS_DIR):
        return []
    real = {}
    rp = os.path.join(LANDMARKS_DIR, "real.json")
    if os.path.exists(rp):
        with open(rp, encoding="utf-8") as f:
            real = json.load(f).get("landmarks", {})
    sites = {}
    sp = os.path.join(LANDMARKS_DIR, "sites.json")
    if os.path.exists(sp):
        with open(sp, encoding="utf-8") as f:
            sites = json.load(f)
    out = []
    for fn in sorted(os.listdir(LANDMARKS_DIR)):
        if not fn.endswith(".json") or fn in ("real.json", "sites.json"):
            continue
        with open(os.path.join(LANDMARKS_DIR, fn), encoding="utf-8") as f:
            e = json.load(f)
        if not isinstance(e, dict) or "lat" not in e or "id" not in e:
            continue
        r = real.get(e["id"], {})
        fp = float(e.get("fp") or 0.0)
        if r.get("fp_obb"):
            fp = max(r["fp_obb"]) * 0.5
        names = {str(n).casefold().strip() for n in (e.get("name"), r.get("wd_label")) if n}
        out.append({"id": e["id"], "names": names, "qid": r.get("qid", ""), "lat": float(r.get("lat", e["lat"])),
                    "lon": float(r.get("lon", e["lon"])), "r": min(fp, LANDMARK_R_MAX), "big": fp > LANDMARK_FP_MAX,
                    "site": (float(sites.get(e["id"], e)["lat"]), float(sites.get(e["id"], e)["lon"]))})
    return out


def holding(lat, lon, clat, clon, loc, area, kr, near_m=0.0, min_area=0.0):
    """Indices of the footprints holding (lat, lon), smallest first; none: the nearest within near_m (of min_area
    m2 or more), else []."""
    import numpy as np
    import shapely
    near = np.nonzero((np.abs(clat - lat) < 0.004) & (np.abs(clon - lon) < 0.005))[0]
    if len(near) == 0:
        return near
    px = (lon - clon[near]) * kr * np.cos(np.radians(clat[near]))
    py = (lat - clat[near]) * kr
    inside = near[shapely.contains_xy(loc[near], px, py)]
    if len(inside):
        return inside[np.argsort(area[inside], kind="stable")]
    if near_m > 0:
        d = shapely.distance(loc[near], shapely.points(px, py))
        j = int(np.argmin(d))
        if d[j] <= near_m and area[near[j]] >= min_area:
            return near[j:j + 1]
    return near[:0]


def dist_m(clat, clon, lat, lon, kr):
    import numpy as np
    return np.hypot((clat - lat) * kr, (clon - lon) * kr * math.cos(math.radians(lat)))


def mast_points(region):
    """[(name, lat, lon, h)] of the region's Wikidata masts and TV towers (kind "mast" or NOT_A_BUILDING names)."""
    if not os.path.exists(KNOWN):
        return []
    with open(KNOWN, encoding="utf-8") as f:
        d = json.load(f)
    return [(r[2], float(r[3]), float(r[4]), float(r[5] or 0.0)) for r in d["rows"]
            if r[0] == region and ((len(r) > 8 and r[8] == "mast") or NOT_A_BUILDING.search(r[2]))]


def drop_mask(cfg, names, clat, clon, loc, area, kr, log_to=None, hgt=None, region="", overrides=True):
    """Footprints the bake leaves out: copies of a registered landmark (named like it within 3 km of its real spot
    or its model's site; at its real spot the smallest footprint holding it and, for a landmark of up to
    LANDMARK_FP_MAX, every footprint centred within its radius and not much bigger than it: Tokyo Tower's 333 m
    block, Skytree's 634 m), and the region's "drop" points."""
    import numpy as np
    out = np.zeros(len(clat), bool)
    la0, lo0, la1, lo1 = cfg["box"]
    low = np.array([str(n).casefold().strip() if n else "" for n in names], dtype=object)
    for lm in landmarks():
        pts = [(lm["lat"], lm["lon"]), lm["site"]]
        if not any(la0 - 0.03 <= a <= la1 + 0.03 and lo0 - 0.03 <= o <= lo1 + 0.03 for a, o in pts):
            continue
        n0 = int(out.sum())
        if lm["names"]:
            named = np.isin(low, list(lm["names"]))
            for (a, o) in pts:
                out |= named & (dist_m(clat, clon, a, o, kr) < 3000.0)
        h = holding(lm["lat"], lm["lon"], clat, clon, loc, area, kr)
        if len(h) and math.sqrt(area[h[0]] / math.pi) <= max(3.0 * lm["r"], 60.0):
            out[h[0]] = True
        if not lm["big"] and lm["r"] > 0:
            out |= (dist_m(clat, clon, lm["lat"], lm["lon"], kr) < lm["r"]) & (np.sqrt(area / math.pi) <= 1.5 * lm["r"])
        if log_to is not None and out.sum() > n0:
            log_to.append("landmark %s: %d footprints dropped" % (lm["id"], int(out.sum()) - n0))
    # a Wikidata mast's own footprint (whatever its Overture name: Canton Tower as 广州塔): up to MAST_FP m2 and
    # as tall as the mast (half its height, else 60 m)
    mast = []
    for (mn, a, o, mh) in mast_points(region) if hgt is not None else []:
        hold = holding(a, o, clat, clon, loc, area, kr)
        hold = [i for i in hold if area[i] <= MAST_FP and np.isfinite(hgt[i]) and hgt[i] >= (0.5 * mh if mh else 60.0)]
        if not hold and mh:
            # (the mast's own shaft beside the point: Kyoto Tower on its hotel, a part footprint)
            d = dist_m(clat, clon, a, o, kr)
            near = np.nonzero((d < MAST_PARTS_R * 1.5) & (area <= MAST_FP) & np.isfinite(hgt) & (hgt >= 0.5 * mh))[0]
            hold = [int(near[np.argmin(d[near])])] if len(near) else []
        if hold:
            mast.append(hold[0])
            if log_to is not None:
                log_to.append("mast %s: its footprint (%.0f m2, %.0f m) dropped" % (mn, area[hold[0]], hgt[hold[0]]))
    # and the unnamed parts of a mast within MAST_PARTS_R as tall as half of it (Beijing Olympic Tower's five shafts)
    if hgt is not None:
        mast += [i for i in np.nonzero(np.isfinite(hgt) & (hgt >= 60.0))[0]
                 if names[i] and NOT_A_BUILDING.search(str(names[i]))]
        for i in mast:
            out[i] = True
            d = dist_m(clat, clon, clat[i], clon[i], kr)
            part = (d < MAST_PARTS_R) & np.isfinite(hgt) & (hgt >= 0.5 * hgt[i]) & (area <= MAST_FP) & np.array([not n for n in names])
            out |= part
            if log_to is not None and part.any():
                log_to.append("mast parts near %.5f, %.5f: %d dropped" % (clat[i], clon[i], int(part.sum())))
    for (a, o, r) in region_overrides(cfg)[2] if overrides else []:
        if r < 0:
            h = holding(a, o, clat, clon, loc, area, kr)
            out[h] = True
            n = len(h)
        elif r > 0:
            m = dist_m(clat, clon, a, o, kr) < r
            out |= m
            n = int(m.sum())
        else:
            h = holding(a, o, clat, clon, loc, area, kr, 15.0)
            out[h[:1]] = True
            n = min(1, len(h))
        if log_to is not None:
            log_to.append("drop %.5f, %.5f r %.0f: %d footprints" % (a, o, r, n))
    return out


def own_heights(hgt, flo, known_h, has_f):
    """A footprint's own height (Overture, else its floors; nan = none)."""
    import numpy as np
    return np.where(known_h, hgt, np.where(has_f, flo * FLOOR_M, np.nan))


def isolated_spikes(h, src, gm, clat, clon, named, kr, plain=None):
    """Overture / floors heights of 80 m and up that the GHSL cell does not back (its 3 x 3 max under 1/6 of it)
    with no building of 0.4 x that height within 400 m and no Wikidata name: a typo (Lagos "93" at 99 m,
    Tiantongyuan's 66 floors). plain (unnamed, no floors): 150 m and up with nothing of half its height within 400 m
    is one too whatever the GHSL cell says (Sao Paulo's unnamed 219 m in Santo Amaro)."""
    import numpy as np
    plain = np.zeros(len(h), bool) if plain is None else plain
    lone = (src <= 1) & (named == None) & (((h >= 80.0) & (gm > 0) & (gm * 6.0 < h)) | (plain & (src == 0) & (h >= 150.0)))  # noqa: E711
    cand = np.nonzero(lone)[0]
    out = np.zeros(len(h), bool)
    if len(cand) == 0:
        return out
    tall_i = np.nonzero(h >= 32.0)[0]
    for i in cand:
        d = dist_m(clat[tall_i], clon[tall_i], clat[i], clon[i], kr)
        if not ((d < 400.0) & (h[tall_i] >= (0.5 if plain[i] else 0.4) * h[i]) & (tall_i != i)).any():
            out[i] = True
    return out


def same_name(a, b):
    """Footprint name a and tower name b name the same building (one holds the other, casefolded)."""
    a = str(a or "").casefold().strip()
    b = str(b or "").casefold().strip()
    return len(a) >= 4 and len(b) >= 4 and (a in b or b in a)


def match_tower(wlat, wlon, wh, clat, clon, loc, area, own, kr, any_size=False, names=None, tname=""):
    """The footprint a tower point (Wikidata, a region's "towers") raises, (index, None, named) or (-1, why,
    False): of the footprints holding the point (else the nearest within 30 m of 300 m2 and up) that can carry wh
    (150 m2 and up, wh under 25 x their side): the one named like the tower, else the one whose own height is
    nearest wh (within 30 %), else the smallest; never one over BIG_FP for a 60 m+ height (Metro Town's 28,900 m2
    mall is named like its tower too) unless any_size."""
    import numpy as np
    cands = holding(wlat, wlon, clat, clon, loc, area, kr, 30.0, 300.0)
    if len(cands) == 0:
        return -1, "no footprint", False
    fit = cands[(area[cands] >= 150.0) & (wh <= 25.0 * np.sqrt(area[cands]))]
    if len(fit) == 0:
        return -1, "footprint %.0f m2: a mast on a roof, a monument on a plaza" % area[cands[-1]], False
    if names is not None and tname:
        nm = [i for i in fit if same_name(names[i], tname) and (area[i] <= BIG_FP or any_size or wh < 60.0)]
        if nm:
            return int(nm[0]), None, True
    oh = own[fit]
    close = np.isfinite(oh) & (np.abs(oh - wh) <= 0.3 * wh)
    if close.any():
        k = np.nonzero(close)[0]
        return int(fit[k[np.argmin(np.abs(oh[k] - wh))]]), None, False
    i = int(fit[0])
    if wh >= 60.0 and area[i] > BIG_FP and not any_size:
        return -1, "smallest footprint %.0f m2 > %.0f" % (area[i], BIG_FP), False
    return i, None, False


def phantom(i, wh, clat, clon, own, kr, podium=True):
    """Another footprint within PHANTOM_R already stands at about wh (its own height within 5 %): the Wikidata
    point is a duplicate of that tower (Tianjin's phantom Bohai Bank Tower); or one within NEXT_R has 0.6 x wh or more:
    the point landed on a podium (over NEXT_FP) beside its tower (c14's rule: Two Liberty). The index of that
    footprint, else -1."""
    import numpy as np
    dla = PHANTOM_R / kr
    m = (np.abs(clat - clat[i]) < dla) & (np.abs(clon - clon[i]) < dla / math.cos(math.radians(clat[i])))
    m[i] = False
    j = np.nonzero(m & np.isfinite(own) & (own >= 0.6 * wh) & (own >= 40.0))[0]
    if len(j) == 0:
        return -1
    d = dist_m(clat[j], clon[j], clat[i], clon[i], kr)
    j = j[((d < NEXT_R) & podium) | ((d < PHANTOM_R) & (own[j] >= 60.0) & (np.abs(own[j] - wh) <= 0.05 * wh))]
    return int(j[0]) if len(j) else -1


def verified_tallest(hgt, flo, known_h, has_f, items):
    """The city's tallest height a floor count backs (an Overture height with its floors, a Wikidata height with
    its floors); 0 = none."""
    ok = known_h & has_f & (hgt <= 6.0 * flo + 30.0) & (hgt >= 2.5 * flo)
    v = float(hgt[ok].max()) if ok.any() else 0.0
    for t in items:
        if t["fl"] and 2.5 * t["fl"] <= t["h"] <= 6.0 * t["fl"] + 30.0:
            v = max(v, t["h"])
    return v


# ---------------------------------------------------------------- classification

K_RES, K_COM, K_IND, K_CIV, K_TALL = 0, 1, 2, 3, 4
R_FLAT, R_GABLE, R_HIP, R_PYR, R_DOME = 0, 1, 2, 3, 4
CLASS_KIND = {}
for _k, _names in ((K_RES, "house apartments residential detached semidetached_house terrace dormitory bungalow cabin "
                    "hut static_caravan houseboat farm trullo stilt_house"),
                   (K_COM, "retail commercial office hotel supermarket kiosk bank mall shop"),
                   (K_IND, "industrial warehouse factory manufacture greenhouse hangar storage_tank silo service garage "
                    "garages shed barn cowshed farm_auxiliary digester roof parking carport boathouse stable sty "
                    "allotment_house glasshouse"),
                   (K_CIV, "school kindergarten university college hospital church mosque synagogue temple cathedral "
                    "chapel shrine religious government civic public fire_station police stadium grandstand "
                    "sports_hall sports_centre museum library townhall train_station transportation military "
                    "pavilion monastery bunker")):
    for _n in _names.split():
        CLASS_KIND[_n] = _k
SUBTYPE_KIND = {"residential": K_RES, "commercial": K_COM, "industrial": K_IND, "agricultural": K_IND,
                "outbuilding": K_IND, "service": K_IND, "civic": K_CIV, "education": K_CIV, "medical": K_CIV,
                "religious": K_CIV, "entertainment": K_CIV, "transportation": K_CIV, "military": K_CIV}
ROOF_MAP = {"flat": R_FLAT, "gabled": R_GABLE, "saltbox": R_GABLE, "gambrel": R_GABLE, "skillion": R_GABLE,
            "hipped": R_HIP, "half_hipped": R_HIP, "mansard": R_HIP, "side_hipped": R_HIP,
            "pyramidal": R_PYR, "dome": R_DOME, "onion": R_DOME, "round": R_DOME, "spherical": R_DOME}
# share of small houses with a pitched roof (Overture rarely says), and gable vs hip
PITCHED = {"israel": (0.12, R_GABLE), "moscow": (0.35, R_HIP), "nyc": (0.55, R_GABLE), "dubai": (0.03, R_HIP),
           "london": (0.9, R_GABLE), "lasvegas": (0.85, R_HIP), "sf": (0.45, R_GABLE), "bangkok": (0.55, R_HIP)}
FIELD_CLASSES = ["stadium", "grandstand", "sports_hall", "sports_centre", "pavilion", "parking", "carport", "roof",
                 "hangar", "storage_tank", "silo", "greenhouse", "shelter", "toilets"]
MIN_AREA = (50.0, 35.0, 25.0)     # m2, by zone level
GAP = (6.0, 4.5, 3.5)             # street gap between kept footprints (toy m), by zone level
ROAD_MARGIN = 3.0                 # past a drawn deck edge
WATER_MARGIN = 2.0                # off the city water (toy m)
MAX_PTS = 40
NUDGE_R = (5.0, 10.0, 16.0)       # toy m: a building on a drawn road tries these moves off it
HOUSE_CLASSES = ["house", "detached", "semidetached_house", "terrace", "bungalow", "cabin", "hut"]


def hash01(ids):
    import numpy as np
    out = np.empty(len(ids), np.float64)
    for i, s in enumerate(ids):
        out[i] = int.from_bytes(hashlib.blake2b(s.encode(), digest_size=4).digest(), "little") / 4294967296.0
    return out


def largest_polygons(geoms):
    """Per input geometry: its largest polygon part (None when it has none)."""
    import numpy as np
    import shapely
    parts, pidx = shapely.get_parts(geoms, return_index=True)
    isp = shapely.get_type_id(parts) == 3
    parts, pidx = parts[isp], pidx[isp]
    out = np.full(len(geoms), None, dtype=object)
    if len(parts) == 0:
        return out
    pa = shapely.area(parts)
    order = np.lexsort((-pa, pidx))
    first = order[np.r_[True, pidx[order][1:] != pidx[order][:-1]]]
    out[pidx[first]] = parts[first]
    return out


# ---------------------------------------------------------------- bake one region

def water_banks(water, X, Y, lc, li, area, lat_ref, lon_ref, K, bad, tower, toy, tree):
    """(wet, bank, moved): the buildings on the city water (wet), the towers kept at its edge (bank, F_BANK) and how
    many moved off it. X, Y and toy (the toy footprints) are updated in place for the moved ones. A footprint is wet
    when a vertex is within WATER_MARGIN toy m of the water or its centre within WATER_MARGIN + 0.3 sqrt(area);
    one not already dropped (bad) steps landward along the water distance gradient (at most BANK_NUDGE in all);
    a moved one that is not a tower must not land on a drawn road (tree); a tower that still touches stays (at the
    better of its real and moved spot) when its centre is BANK_DRY on land."""
    import numpy as np
    import shapely
    n = len(X)

    def state(sub):
        sv = sub[li]
        vy = lc[sv, 1] + Y[li[sv]]
        vx = lc[sv, 0] + X[li[sv]]
        vla = lat_ref + vy / K
        dv = water.dist(vla, lon_ref + vx / (K * np.cos(np.radians(vla))))
        mind = np.full(n, 1e9, np.float32)
        np.minimum.at(mind, li[sv], dv)
        dc = np.full(n, 1e9, np.float32)
        cla = lat_ref + Y[sub] / K
        dc[sub] = water.dist(cla, lon_ref + X[sub] / (K * np.cos(np.radians(cla))))
        return mind, dc

    def d_at(x, y):
        la = lat_ref + y / K
        return water.dist(la, lon_ref + x / (K * np.cos(np.radians(la))))

    need_c = WATER_MARGIN + np.sqrt(area) * 0.3
    mind, dc = state(np.ones(n, bool))
    wet = (mind < WATER_MARGIN) | (dc < need_c)
    X0, Y0, dc0 = X.copy(), Y.copy(), dc.copy()
    moved = np.zeros(n)
    try_ = wet & ~bad & (dc < 1e8)
    for _ in range(5):
        idx = np.nonzero(try_ & wet)[0]
        if len(idx) == 0:
            break
        e = 3.0
        gx = d_at(X[idx] + e, Y[idx]) - d_at(X[idx] - e, Y[idx])
        gy = d_at(X[idx], Y[idx] + e) - d_at(X[idx], Y[idx] - e)
        gn = np.hypot(gx, gy)
        ok = (gn > 1e-3) & (np.abs(gx) < 1e8) & (np.abs(gy) < 1e8)
        step = np.maximum(np.maximum(WATER_MARGIN - mind[idx], need_c[idx] - dc[idx]), 0.0) + 1.0
        step = np.minimum(step, BANK_NUDGE - moved[idx])
        ok &= step > 0.25
        idx, gx, gy, gn, step = idx[ok], gx[ok], gy[ok], gn[ok], step[ok]
        if len(idx) == 0:
            break
        X[idx] += gx / gn * step
        Y[idx] += gy / gn * step
        moved[idx] += step
        sub = np.zeros(n, bool)
        sub[idx] = True
        m2, d2 = state(sub)
        mind[idx], dc[idx] = m2[idx], d2[idx]
        wet[idx] = (mind[idx] < WATER_MARGIN) | (dc[idx] < need_c[idx])
    shifted = (X != X0) | (Y != Y0)
    # a tower that still touches: the better of its real and its moved spot, kept when the centre is dry enough
    bank = wet & tower & ~bad & (np.maximum(dc, dc0) >= BANK_DRY)
    back = shifted & ((wet & ~bank) | (bank & (dc0 >= dc)))
    if tree is not None:
        # a moved building that is no tower must not land on a drawn road
        mv = np.nonzero(shifted & ~back & ~tower)[0]
        if len(mv):
            off = np.column_stack([X[mv] - X0[mv], Y[mv] - Y0[mv]])
            geo = np.array([shapely.transform(toy[i], lambda c, o=o: c + o) for i, o in zip(mv, off)], dtype=object)
            hit = tree.query(geo, predicate="intersects")
            onroad = np.zeros(n, bool)
            onroad[mv[np.unique(hit[0])]] = True
            back |= onroad
            wet |= onroad
    X[back], Y[back] = X0[back], Y0[back]
    wet[back & ~bank] = True
    keep = shifted & ~back
    if keep.any():
        ki = np.nonzero(keep)[0]
        toy[ki] = np.array([shapely.transform(toy[i], lambda c, o=(X[i] - X0[i], Y[i] - Y0[i]): c + np.array(o))
                            for i in ki], dtype=object)
    wet &= ~bank
    return wet, bank, int((keep & ~bank).sum())


def bake_region(name, ghsl, water):
    import duckdb
    import numpy as np
    import shapely
    from shapely import STRtree
    cfg = REGIONS[name]
    la0, lo0, la1, lo1 = cfg["box"]
    lat_ref = (la0 + la1) * 0.5
    lon_ref = (lo0 + lo1) * 0.5
    K = M_PER_DEG
    t0 = time.time()
    con = duckdb.connect()
    tb = con.execute("SELECT id, height, num_floors, class, subtype, roof_shape, name, wkb FROM read_parquet('%s') ORDER BY id"
                     % cache_path(name).replace("\\", "/")).fetchnumpy()
    ids = np.asarray(tb["id"], dtype=object)
    n_src = len(ids)
    hgt = np.asarray(tb["height"].astype(np.float64).filled(np.nan) if hasattr(tb["height"], "filled") else tb["height"], np.float64)
    flo = np.asarray(tb["num_floors"].astype(np.float64).filled(np.nan) if hasattr(tb["num_floors"], "filled") else tb["num_floors"], np.float64)
    cls = np.asarray(tb["class"], dtype=object)
    sub = np.asarray(tb["subtype"], dtype=object)
    roofs = np.asarray(tb["roof_shape"], dtype=object)
    names = np.asarray(tb["name"], dtype=object)
    wkb = np.array([bytes(w) for w in tb["wkb"]], dtype=object)
    del tb
    polys = largest_polygons(shapely.from_wkb(wkb))
    del wkb
    ok = polys != None  # noqa: E711
    polys[ok] = shapely.polygons(shapely.get_exterior_ring(polys[ok].astype(object)))
    cen = shapely.centroid(polys[ok].astype(object))
    clon = np.full(len(polys), np.nan)
    clat = np.full(len(polys), np.nan)
    clon[ok] = shapely.get_x(cen)
    clat[ok] = shapely.get_y(cen)
    ok &= np.isfinite(clon) & np.isfinite(clat)
    ok &= (clat >= la0) & (clat <= la1) & (clon >= lo0) & (clon <= lo1)   # (a cache fetched for a wider box)
    if "poly" in cfg:
        ok[ok] &= shapely.contains_xy(shapely.Polygon(cfg["poly"]), clon[ok], clat[ok])
    sel = np.nonzero(ok)[0]
    ids, hgt, flo, cls, sub, roofs, names, polys, clon, clat = (a[sel] for a in (ids, hgt, flo, cls, sub, roofs, names, polys, clon, clat))
    polys = polys.astype(object)
    # local metres round each centroid (real size), simplified
    kr = 6371000.0 * math.pi / 180.0
    coords, cidx = shapely.get_coordinates(polys, return_index=True)
    lx = (coords[:, 0] - clon[cidx]) * kr * np.cos(np.radians(clat[cidx]))
    ly = (coords[:, 1] - clat[cidx]) * kr
    loc = shapely.polygons(shapely.linearrings(np.column_stack([lx, ly]), indices=cidx))
    loc = largest_polygons(shapely.make_valid(shapely.simplify(loc, 0.75, preserve_topology=True)))
    sel = np.nonzero(loc != None)[0]  # noqa: E711
    ids, hgt, flo, cls, sub, roofs, names, clon, clat = (a[sel] for a in (ids, hgt, flo, cls, sub, roofs, names, clon, clat))
    loc = shapely.orient_polygons(shapely.polygons(shapely.get_exterior_ring(loc[sel].astype(object))))
    # too many points: a coarser simplification, then the hull
    npts = shapely.get_num_coordinates(loc) - 1
    big = npts > MAX_PTS
    if big.any():
        loc[big] = shapely.orient_polygons(shapely.simplify(loc[big], 2.5, preserve_topology=True))
        npts = shapely.get_num_coordinates(loc) - 1
        big = npts > MAX_PTS
        if big.any():
            loc[big] = shapely.orient_polygons(shapely.simplify(shapely.convex_hull(loc[big]), 4.0))
            npts = shapely.get_num_coordinates(loc) - 1
    area = shapely.area(loc)
    zl = np.zeros(len(ids), np.int64)
    for (za0, zo0, za1, zo1, lv) in cfg.get("zones", []):
        inz = (clat >= za0) & (clat <= za1) & (clon >= zo0) & (clon <= zo1)
        zl = np.where(inz, np.maximum(zl, lv), zl)
    ok = (area >= np.array(MIN_AREA)[zl]) & (npts >= 3) & (npts <= MAX_PTS) & shapely.is_valid(loc)
    ok &= np.array([not (bool(n) and NOT_A_BUILDING.search(str(n))) for n in names])
    notes = []   # (CITY_DEBUG: what the tower and override rules did)
    ok &= ~drop_mask(cfg, names, clat, clon, loc, area, kr, notes, hgt, name)
    sel = np.nonzero(ok)[0]
    ids, hgt, flo, cls, sub, roofs, names, clon, clat, loc, area, zl, npts = (
        a[sel] for a in (ids, hgt, flo, cls, sub, roofs, names, clon, clat, loc, area, zl, npts))
    ids, hgt, flo, cls, sub, roofs, names, clon, clat, loc, area, zl, npts = add_footprints(
        cfg, ids, hgt, flo, cls, sub, roofs, names, clon, clat, loc, area, zl, npts)
    hsh = hash01(ids)
    # the home footprint (the building that holds the "home" point) is kept before everything else
    home = np.zeros(len(ids), bool)
    if "home" in cfg:
        hla, hlo = cfg["home"]
        near = np.nonzero((np.abs(clat - hla) < 0.002) & (np.abs(clon - hlo) < 0.002))[0]
        if len(near):
            hx = (hlo - clon[near]) * kr * np.cos(np.radians(clat[near]))
            hy = (hla - clat[near]) * kr
            home[near[shapely.contains_xy(loc[near], hx, hy)]] = True
    # heights: Overture height, else floors, else GHSL (100 m average net height) with a spread
    gc, gm = ghsl.sample(clat, clon)
    est = np.where(gc > 0, gc, gm * 0.6).astype(np.float64)
    est = np.where((area > 1500.0) & (gm > 18.0), np.maximum(est, gm * 0.75), est)
    kind = np.array([CLASS_KIND.get(c, SUBTYPE_KIND.get(s, -1)) for c, s in zip(cls, sub)], np.int64)
    dflt = np.where(kind == K_IND, 7.0, np.where(kind == K_COM, 12.0, 8.0))
    est = np.where(est > 0, est, dflt) * (0.75 + 0.5 * hsh)
    house = np.isin(cls, HOUSE_CLASSES) | ((kind == K_RES) & (area < 160.0))
    est = np.where(house, np.minimum(est, 7.5 + 3.0 * hsh), est)
    known_h = np.isfinite(hgt) & (hgt > 2.0)
    has_f = np.isfinite(flo) & (flo > 0) & (flo <= MAX_FLOORS)
    # an Overture height far over its floors is a typo (Bangkok's 36-storey "314 m", Guntur's 520 m on 2): the floors
    known_h &= ~(has_f & (hgt > 2.0 * flo * FLOOR_M + 10.0))
    # 150 m and up with no floors on a field, sports, parking or industrial footprint, or a needle (unnamed, no floors,
    # over 8 x its footprint's side: Camden's 434 m house): a typo, the GHSL guess
    noname = np.array([not n for n in names])
    known_h &= ~(~has_f & (hgt >= 150.0) & ((kind == K_IND) | np.isin(cls, FIELD_CLASSES) |
                                            (noname & (hgt > 8.0 * np.sqrt(area)))))
    known_f = ~known_h & has_f
    h = np.where(known_h, hgt, np.where(known_f, flo * FLOOR_M, est))
    # the floors say far taller than the Overture height (a podium or eave height): the floors
    h = np.where(known_h & has_f & (flo * FLOOR_M > 1.5 * hgt), flo * FLOOR_M, h)
    # a tower still being built: the GHSL guess, not the planned height
    unbuilt = np.array([bool(n) and bool(UNBUILT.match(str(n))) for n in names])
    known_h &= ~unbuilt
    h = np.where(unbuilt, est, h)
    src = np.where(known_h, 0, np.where(known_f, 1, 2)).astype(np.int64)   # (HEIGHT_SOURCES index)
    # more heights (tools/fetch_heights.py): Overture building parts (the tower over its podium, its footprint), and
    # 3D-GloBFP / Google 2.5D per-footprint heights in place of the GHSL guess; CITY_HEIGHTS=off bakes without them
    if os.environ.get("CITY_HEIGHTS", "") != "off":
        import fetch_heights
        h, src, loc, area, n_fp, n_gg, n_pt, n_tw = fetch_heights.apply(name, cfg["box"], ids, loc, area, clat, clon, h,
                                                                        src, known_h, known_f, unbuilt)
        log("%s heights: %d from 3D-GloBFP, %d from Google 2.5D, %d raised to a building part, %d tower footprints from "
            "parts" % (name, n_fp, n_gg, n_pt, n_tw))
    # known towers (Wikidata): the footprint that holds the item's point takes its height when Overture has none
    # or one far under it (the floors and GHSL are guesses: a known height beats them upward)
    named = np.full(len(ids), None, dtype=object)
    # (v4 matching, tools/tests/test_bake_rules.py): of the footprints holding the point, the one whose own height is
    # nearest the item's, else the SMALLEST (a tower's point lands in its podium, mall or complex outline too); no
    # 60 m+ raise on a footprint over BIG_FP; a tower beside the point (NEXT_R) or a twin of its height near it
    # (PHANTOM_R) makes it a duplicate; a structure of worship's height is its spire or minaret (no raise); a
    # height 1.6 x over the city's floor-backed tallest with no floors anywhere is a typo (Santiago's "448 m")
    hfix, tlist, _ = region_overrides(cfg)
    own = own_heights(hgt, flo, known_h, has_f & ~unbuilt)
    own = np.where(src == 5, h, own)   # (a building part's height is the footprint's own too)
    # masts and landmark copies again with the part heights and tower footprints (Oriental Pearl, Torrespana raised
    # by their parts): left out at the keep-outs below
    late = drop_mask(cfg, names, clat, clon, loc, area, kr, notes, h, name, overrides=False)
    items = known_towers(name, cfg.get("qid_fixes"), {lm["qid"] for lm in landmarks() if lm["qid"]}, notes)
    # a building part's height can include the antenna (Willis 527 m, 875 N Michigan 457 m): a part-raised footprint
    # holding a known tower's point stands at the item's (roof) height when it is clearly over it
    for t in items:
        wh = float(t["h"])
        if wh <= 0.0:
            continue
        dy = (clat - t["lat"]) * 111320.0
        dx = (clon - t["lon"]) * 111320.0 * math.cos(math.radians(t["lat"]))
        over = (src == 5) & (h > 1.08 * wh) & (dx * dx + dy * dy < np.maximum(60.0, np.sqrt(area)) ** 2)
        if over.any():
            notes.append("wikidata %s %s %.0f m: part height %.0f m capped (antenna)" % (t["qid"], t["name"], wh,
                         float(h[over].max())))
            h = np.where(over, wh, h)
            own = np.where(over, wh, own)
    ref = verified_tallest(hgt, flo, known_h, has_f, items)
    rpoly = shapely.Polygon(cfg["poly"]) if "poly" in cfg else None
    for t in items:
        wh = t["h"]
        tag = "wikidata %s %s %.0f m" % (t["qid"], t["name"], wh)
        if rpoly is not None and not rpoly.contains(shapely.Point(t["lon"], t["lat"])):
            continue # (another region's tower in an overlapping box)
        i, why, byname = match_tower(t["lat"], t["lon"], wh, clat, clon, loc, area, own, kr, names=names, tname=t["name"])
        if i < 0:
            notes.append("%s: skipped (%s)" % (tag, why))
            continue
        close = np.isfinite(own[i]) and abs(own[i] - wh) <= 0.3 * wh
        if t["worship"] and not close:
            notes.append("%s: no raise (worship / monument: a spire or minaret height)" % tag)
            continue
        if not (close or byname):
            if ref > 0 and wh > 1.6 * ref and not t["fl"] and not np.isfinite(own[i]):
                notes.append("%s: skipped (over 1.6 x the floor-backed tallest %.0f m, no floors)" % (tag, ref))
                continue
            j = phantom(i, wh, clat, clon, own, kr, area[i] > NEXT_FP)
            if j >= 0:
                notes.append("%s: skipped (footprint %.0f m off already %.0f m: a duplicate point)" % (
                    tag, float(dist_m(clat[j], clon[j], clat[i], clon[i], kr)), own[j]))
                continue
        if not known_h[i] or hgt[i] < 0.6 * wh:
            if wh > h[i]:
                h[i] = wh
                src[i] = 3
        notes.append("%s: footprint %.0f m2 (own %s) -> %.0f m" % (tag, area[i], "%.0f" % own[i] if np.isfinite(own[i]) else "-", h[i]))
        named[i] = t["name"]
    # the region's own fixes (region_overrides): towers the data misses, then heights by point
    for (tn, tla, tlo, th, anys) in tlist:
        i, why, _ = match_tower(tla, tlo, th, clat, clon, loc, area, own, kr, anys)
        if i < 0:
            notes.append("tower %s: skipped (%s)" % (tn, why))
            continue
        h[i] = th
        src[i] = 3
        named[i] = tn or named[i]
        notes.append("tower %s: footprint %.0f m2 -> %.0f m" % (tn, area[i], th))
    for (fla, flo_, fh) in hfix:
        hold = holding(fla, flo_, clat, clon, loc, area, kr, 30.0)
        if len(hold) == 0:
            notes.append("height fix %.5f, %.5f: no footprint" % (fla, flo_))
            continue
        # (a fix that lowers: the tallest holder, the one carrying the bad height; a raise: the smallest)
        i = int(hold[np.argmax(h[hold])]) if fh <= float(h[hold].max()) else int(hold[0])
        if fh > 0:
            h[i], src[i] = fh, 3
        else: # no trusted height: the floors (when they agree with a building), else the GHSL guess
            okf = has_f[i] and not unbuilt[i]
            h[i], src[i] = (flo[i] * FLOOR_M, 1) if okf else (est[i], 2)
            named[i] = None
        notes.append("height fix %.5f, %.5f: footprint %.0f m2 -> %.0f m" % (fla, flo_, area[i], h[i]))
    # sanity: a lone Overture / floors spike the GHSL cell does not back; the GHSL guess on a huge footprint (a 70,000
    # m2 block at 69 m) and an Overture height no floors back on the biggest (a mall at 60 m)
    sp = isolated_spikes(h, src, gm, clat, clon, named, kr, noname & ~has_f)
    h = np.where(sp, est, h)
    src = np.where(sp, 2, src)
    for k in range(2):
        h = np.where(np.isin(src, (2, 4, 6)) & (area >= SLAB_A[k]), np.minimum(h, SLAB_H[k]), h)
    slab = (src == 0) & (area >= SLAB_A[1]) & (h > SLAB_OV_H) & ~(has_f & (flo * FLOOR_M >= 0.6 * h)) & (named == None)  # noqa: E711
    h = np.where(slab, SLAB_OV_H, h)
    notes.append("sanity: %d lone spikes to the GHSL guess, %d huge Overture slabs capped" % (int(sp.sum()), int(slab.sum())))
    if os.environ.get("CITY_DEBUG"):
        os.makedirs(os.environ["CITY_DEBUG"], exist_ok=True)
        with open(os.path.join(os.environ["CITY_DEBUG"], name + "_rules.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(notes) + "\n")
    h = np.clip(h, 3.5, 900.0)
    if home.any():
        h = np.where(home, HOME_H, h)
    kind = np.where(kind >= 0, kind, np.where(h >= 30.0, K_COM, np.where((area > 2500.0) & (h < 14.0), K_IND, K_RES)))
    kind = np.where(h >= TALL_M, K_TALL, kind).astype(np.uint8)
    pshare, pkind = PITCHED.get(name, (0.3, R_GABLE))
    roof = np.array([ROOF_MAP.get(r, 255) if r is not None else 255 for r in roofs], np.int64)
    small = (kind == K_RES) & (area < 320.0) & (h < 14.0)
    roof = np.where(roof != 255, roof, np.where(small & (np.modf(hsh * 7.31)[0] < pshare), pkind, R_FLAT)).astype(np.uint8)
    # toy position of the centroid (sinusoidal round the region centre: true scale locally)
    X = (clon - lon_ref) * K * np.cos(np.radians(clat))
    Y = (clat - lat_ref) * K
    lc, li = shapely.get_coordinates(loc, return_index=True)
    toy = shapely.polygons(shapely.linearrings(lc + np.column_stack([X[li], Y[li]]), indices=li))
    n_cand = len(ids)
    # roads yield to the towers, the home and the Ma'agal HaShalom block (kept where they really stand)
    yld = (h >= YIELD_M) | home
    if "home" in cfg:
        hla, hlo = cfg["home"]
        hX = (hlo - lon_ref) * K * math.cos(math.radians(hla))
        hY = (hla - lat_ref) * K
        yld |= (X - hX) ** 2 + (Y - hY) ** 2 < MAAGAL_R * MAAGAL_R
    # --- keep-outs: the drawn graph roads, the city water
    bad = np.zeros(n_cand, bool)
    lines = road_lines(cfg["box"])
    if lines:
        geos = []
        for arr, half in lines:
            xs = (arr[:, 1] - lon_ref) * K * np.cos(np.radians(arr[:, 0]))
            ys = (arr[:, 0] - lat_ref) * K
            geos.append(shapely.buffer(shapely.linestrings(np.column_stack([xs, ys])), half + ROAD_MARGIN, quad_segs=2))
        tree = STRtree(np.array(geos, dtype=object))
        hit = tree.query(toy, predicate="intersects")
        bad[np.unique(hit[0])] = True
        n_yield_road = int((bad & yld).sum())
        bad &= ~yld # (the towers stay where they are: the road is cut there at runtime)
        # a building on a road moves off it when a nudge of up to NUDGE_MAX clears it (the roads are drawn
        # real width on the 1/5 ground, so they cover far more of a city than they do on Earth)
        todo = np.nonzero(bad & ((area >= 80.0) | (h >= 12.0)))[0]
        for r in NUDGE_R:
            for a8 in range(8):
                if len(todo) == 0:
                    break
                ang = (a8 + (0.5 if r == NUDGE_R[1] else 0.0)) * math.pi / 4.0
                off = np.array([math.cos(ang) * r, math.sin(ang) * r])
                moved = shapely.transform(toy[todo], lambda c, o=off: c + o)
                hh = tree.query(moved, predicate="intersects")
                clear = np.ones(len(todo), bool)
                clear[np.unique(hh[0])] = False
                ok_i = todo[clear]
                toy[ok_i] = moved[clear]
                X[ok_i] += off[0]
                Y[ok_i] += off[1]
                bad[ok_i] = False
                todo = todo[~clear]
        clat = lat_ref + Y / K
        clon = lon_ref + X / (K * np.cos(np.radians(clat)))
        lc, li = shapely.get_coordinates(toy, return_index=True)
        lc = lc - np.column_stack([X[li], Y[li]])
    else:
        n_yield_road = 0
    n_road = int(bad.sum())
    # --- the city water: footprints are real size on the 1/5 ground, so one standing on a real bank reaches over
    # the drawn water (the Nile Corniche, Raffles Place, Puerto Madero). One that touches it moves landward (down
    # the water's distance gradient) up to BANK_NUDGE toy m; a tower (YIELD_M, a known or the home) that still
    # touches stays where its centre is BANK_DRY on land: the water yields (F_BANK, the game checks its centre only)
    wet, bank, n_nudged = water_banks(water, X, Y, lc, li, area, lat_ref, lon_ref, K, bad,
                                      yld | (named != None) | home, toy, tree if lines else None)  # noqa: E711
    n_water = int((wet & ~bad).sum())
    bad |= wet
    clat = lat_ref + Y / K
    clon = lon_ref + X / (K * np.cos(np.radians(clat)))
    # --- greedy selection: the known (named) and landmark towers first, then the ones roads yield to, then by
    # height x footprint
    prio = h * np.sqrt(area) * np.where((kind == K_TALL) | (named != None), 1000.0, np.where(yld, 100.0, 1.0))  # noqa: E711
    # the towers of TALL_M and up by height alone (a wide 228 m block must not beat the 310 m tower beside it)
    prio = np.where(h >= TALL_M, 1e8 * h, prio)
    prio = np.where(home, 1e12, prio)
    bad &= ~home
    bad |= late
    order = np.lexsort((ids, -prio))
    gap = np.array(GAP)[zl]
    C = 32.0
    grid = {}
    kept = []
    blocker = {}   # (CITY_DEBUG: the kept footprint that took a skipped one's place)
    bx = shapely.bounds(toy)
    for i in order:
        if bad[i]:
            continue
        g = gap[i]
        x0, y0, x1, y1 = bx[i]
        near = set()
        for cy in range(int(math.floor((y0 - g) / C)), int(math.floor((y1 + g) / C)) + 1):
            for cx in range(int(math.floor((x0 - g) / C)), int(math.floor((x1 + g) / C)) + 1):
                lst = grid.get((cx, cy))
                if lst:
                    near.update(lst)
        if near:
            cand = [j for j in near if bx[j][0] - g <= x1 and bx[j][2] + g >= x0 and bx[j][1] - g <= y1 and bx[j][3] + g >= y0]
            if cand:
                ca = np.array(cand, np.int64)
                # two towers roads yield to keep TALL_GAP (a skyline packs tighter than a street of houses)
                gg = np.where(yld[ca] & yld[i], min(g, TALL_GAP), g)
                hit = shapely.distance(toy[ca], toy[i]) < gg
                if hit.any():
                    blocker[i] = int(ca[np.argmax(hit)])
                    continue
        kept.append(i)
        for cy in range(int(math.floor(y0 / C)), int(math.floor(y1 / C)) + 1):
            for cx in range(int(math.floor(x0 / C)), int(math.floor(x1 / C)) + 1):
                grid.setdefault((cx, cy), []).append(i)
    kept = np.array(sorted(kept, key=lambda i: (-prio[i], ids[i])), np.int64)
    if os.environ.get("CITY_DEBUG"):
        kk = np.zeros(n_cand, bool)
        kk[kept] = True
        lost = [i for i in np.nonzero(~kk & ((h >= 150.0) | (named != None)))[0]]  # noqa: E711
        with open(os.path.join(os.environ["CITY_DEBUG"], name + "_rules.txt"), "a", encoding="utf-8") as f:
            for i in sorted(lost, key=lambda i: -h[i]):
                f.write("not kept: %s %.0f m %.0f m2 at %.5f, %.5f (%s)\n" % (
                    named[i] or names[i] or "-", h[i], area[i], clat[i], clon[i],
                    "road" if bad[i] and not wet[i] else "water" if wet[i] else "spacing: %s %.0f m %.0f m2" % (
                        named[blocker[i]] or names[blocker[i]] or "-", h[blocker[i]], area[blocker[i]])
                    if i in blocker else "spacing"))
    log("%s: %d source, %d candidates, %d on roads, %d towers kept on roads, %d in water (%d moved off it, %d bank towers "
        "kept), %d kept (%d yield, %d >= 120 m, %d named), home %d (%.0f s)" % (
            name, n_src, n_cand, n_road, n_yield_road, n_water, n_nudged, int(bank[kept].sum()) if len(kept) else 0,
            len(kept), int(yld[kept].sum()), int((h[kept] >= TALL_M).sum()),
            int((named[kept] != None).sum()), int(home[kept].sum()) if len(kept) else 0, time.time() - t0))  # noqa: E711
    return {"name": name, "kept": kept, "home": home, "bank": bank, "clat": clat, "clon": clon, "loc": loc, "h": h, "kind": kind, "roof": roof,
            "hsh": hsh, "n_src": n_src, "yield": yld, "named": named, "names": names, "src": src}


# ---------------------------------------------------------------- write CBL1 (one file per region) and the index

def tile_of(lat, lon):
    return int(math.floor((lat + 90.0) * TPD)), int(math.floor((lon + 180.0) * TPD))


def write_region(R):
    """Write OUT_DIR/<region>.bin (CBL1 v3, one region); returns its tower rows (TOWER_LIST_M and up)."""
    import numpy as np
    import shapely
    tiles = {}   # key -> [buildings, max h]
    cover = set()
    towers = []
    cfg = REGIONS[R["name"]]
    n = 0
    for i in R["kept"]:
        lat, lon = float(R["clat"][i]), float(R["clon"][i])
        ty, tx = tile_of(lat, lon)
        key = ty * (360 * TPD) + tx
        lat_t = ty / TPD - 90.0
        lon_t = tx / TPD - 180.0
        cos_t = math.cos(math.radians(lat_t + 0.5 / TPD))
        e0 = (lon - lon_t) * M_PER_DEG * cos_t
        n0 = (lat - lat_t) * M_PER_DEG
        pts = shapely.get_coordinates(R["loc"][i])[:-1]
        q = np.round((pts + np.array([e0, n0])) / Q).astype(np.int64)
        if q.min() < -32768 or q.max() > 32767:
            continue
        t = tiles.setdefault(key, [[], 0.0])
        hq = int(min(65535, round(float(R["h"][i]) * 10.0)))
        t[0].append((len(q), int(R["kind"][i]), hq, int(R["roof"][i]), int(R["hsh"][i] * 255.0) & 255,
                     (F_YIELD if R["yield"][i] else 0) | (F_BANK if R["bank"][i] else 0), q.astype("<i2").tobytes()))
        t[1] = max(t[1], float(R["h"][i]))
        n += 1
        gy = int(math.floor((lat + 90.0) * COVER_PD))
        gx = int(math.floor((lon + 180.0) * COVER_PD))
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                cover.add((gy + dy) * 360 * COVER_PD + (gx + dx))
        if R["h"][i] >= TOWER_LIST_M:
            nm = R["named"][i] or R["names"][i] or ""
            towers.append([R["name"], nm, round(lat, 6), round(lon, 6), round(float(R["h"][i]), 1),
                           HEIGHT_SOURCES[int(R["src"][i])]])
    keys = sorted(tiles)
    blobs = []
    recs = []
    for key in keys:
        blds, mh = tiles[key]
        head = bytearray()
        pts = bytearray()
        first = 0
        for (np_, kd, hq, rf, sh, fl, pb) in blds:
            head += struct.pack("<HBBHHBBH", first, np_, kd, hq, 0, rf, sh, fl)
            pts += pb
            first += np_
        blob = zlib.compress(bytes(head + pts), 9)
        recs.append((key, len(blob), len(blds), first, mh))
        blobs.append(blob)
    cover = sorted(cover)
    n_t = len(keys)
    regions_off = 32 + 32 * n_t
    cover_off = regions_off + 64
    data_off = cover_off + 4 * len(cover)
    out = bytearray(struct.pack("<4sIIIIIII", MAGIC, VERSION, n_t, 1, regions_off, cover_off, len(cover), 0))
    pos = data_off
    for (key, nb, nbld, npt, mh) in recs:
        out += struct.pack("<IIIIIHHfI", key, pos, nb, nbld, npt, 0, 0, mh, 0)
        pos += nb
    la0, lo0, la1, lo1 = cfg["box"]
    out += struct.pack("<24sffffIIII8x", R["name"].encode()[:24], la0, lo0, la1, lo1, n, R["n_src"], 0, 0)
    out += np.array(cover, "<u4").tobytes()
    assert len(out) == data_off
    for b in blobs:
        out += b
    path = region_pack.region_path(OUT_DIR, R["name"])
    region_pack.write_atomic(path, bytes(out))
    log("wrote %s: %d buildings, %d tiles, %d cover cells, %.2f MB, sha256 %s" % (
        path, n, n_t, len(cover), len(out) / 1e6, hashlib.sha256(out).hexdigest()[:16]))
    return towers


def write_index_and_towers(towers):
    """The index from every region file in OUT_DIR, then city_towers.json."""
    metas = region_pack.write_index(OUT_DIR, MAGIC, TPD, list(REGIONS), COVER_PD, log,
                                    {n: c.get("poly") for n, c in REGIONS.items()})
    log("wrote %s: %d regions, %.1f MB in all, biggest %s" % (
        os.path.join(OUT_DIR, region_pack.INDEX), len(metas), sum(m["bytes"] for m in metas) / 1e6,
        max(((m["bytes"] / 1e6, m["name"]) for m in metas), default=(0, "-"))))
    write_towers(towers, {m["name"] for m in metas})


def write_towers(new, indexed):
    """city_towers.json: the kept towers (TOWER_LIST_M and up) of every region in the index: name (Wikidata, else
    Overture), position, height, height source. The rows of the regions not baked now are kept."""
    rows = []
    if os.path.exists(TOWERS_OUT):
        with open(TOWERS_OUT, encoding="utf-8") as f:
            rows = [r for r in json.load(f).get("rows", []) if r[0] not in new and r[0] in indexed]
    for v in new.values():
        rows += v
    rows.sort(key=lambda t: (t[0], -t[4], t[2], t[3]))
    lines = ['{', ' "source": "tools/bake_city_buildings.py: the kept city buildings of %d m and more (names: Wikidata CC0, '
             'Overture Maps ODbL)",' % TOWER_LIST_M, ' "columns": ["region", "name", "lat", "lon", "height_m", "height_source"],',
             ' "rows": [']
    for k, t in enumerate(rows):
        lines.append('  ' + json.dumps(t, ensure_ascii=False) + (',' if k < len(rows) - 1 else ''))
    lines += [' ]', '}']
    os.makedirs(os.path.dirname(TOWERS_OUT), exist_ok=True)
    with open(TOWERS_OUT, "w", encoding="utf-8", newline="") as f:
        f.write(CRLF.join(lines) + CRLF)
    log("wrote %s: %d towers" % (TOWERS_OUT, len(rows)))


def write_sources():
    line = ("buildings/*.bin: Overture Maps buildings release %s (OpenStreetMap and other ODbL building footprints and "
            "heights; height estimates from 3D-GloBFP, Che et al. 2024, CC BY 4.0, Google Open Buildings 2.5D Temporal, "
            "CC BY 4.0, and GHSL GHS-BUILT-H R2023A, CC BY 4.0), python tools/fetch_heights.py fetch, "
            "python tools/bake_city_buildings.py, "
            "baked %s" % (OVERTURE, time.strftime("%Y-%m-%d")))
    lines = []
    if os.path.exists(SOURCES_TXT):
        with open(SOURCES_TXT, encoding="utf-8", newline="") as f:
            lines = [l for l in f.read().replace("\r\n", "\n").split("\n")
                     if l and not l.startswith(("buildings.bin:", "buildings/*.bin:"))]
    lines.append(line)
    with open(SOURCES_TXT, "w", encoding="utf-8", newline="") as f:
        f.write("\r\n".join(lines) + "\r\n")


def region_stats(data):
    """Per region index of a CBL1 file: {region index: [tiles, buildings, >= TALL_M, F_YIELD]}."""
    magic, ver, nt, nr, roff, coff, nc, _ = struct.unpack_from("<4sIIIIIII", data, 0)
    st = {}
    for i in range(nt):
        key, off, nb, n, npt, ri = struct.unpack_from("<IIIIIH", data, 32 + 32 * i)
        p = zlib.decompress(data[off:off + nb])
        s = st.setdefault(ri, [0, 0, 0, 0])
        s[0] += 1
        s[1] += n
        for k in range(n):
            _f, _np, _kd, hq, _mh, _rf, _sh, fl = struct.unpack_from("<HBBHHBBH", p, 12 * k)
            s[2] += hq >= TALL_M * 10
            s[3] += fl & F_YIELD
    return st


def info(path=None):
    """Per-region counts of a folder of region files (default OUT_DIR) or of an old single CBL1 v2 file."""
    path = path or OUT_DIR
    if os.path.isfile(path):
        files = [path]
    else:
        files = [region_pack.region_path(path, m["name"]) for m in region_pack.read_index(path)]
    tot = [0, 0, 0, 0, 0, 0]
    print("%-12s %8s %9s %6s %7s %6s %6s %7s" % ("region", "kept", "source", "tiles", "cover", "120m+", "yield", "MB"))
    for fp in files:
        data = open(fp, "rb").read()
        magic, ver, nt, nr, roff, coff, nc, _ = struct.unpack_from("<4sIIIIIII", data, 0)
        st = region_stats(data)
        one = nr == 1
        for r in range(nr):
            nm, la0, lo0, la1, lo1, n, ns, _a, _b = struct.unpack_from("<24sffffIIII8x", data, roff + 64 * r)
            s = st.get(r, [0, 0, 0, 0])
            print("%-12s %8d %9d %6d %7s %6d %6d %7s" % (nm.rstrip(b"\0").decode(), n, ns, s[0], nc if one else "-",
                                                       s[2], s[3], "%.2f" % (len(data) / 1e6) if one else "-"))
            for j, v in enumerate((n, ns, s[0], nc if one else 0, s[2], s[3])):
                tot[j] += v
        if not one:
            print("(%s: CBL1 v%d, %d regions in one file, %d cover cells, %.2f MB)" % (fp, ver, nr, nc, len(data) / 1e6))
    print("%-12s %8d %9d %6d %7d %6d %6d" % ("total", tot[0], tot[1], tot[2], tot[3], tot[4], tot[5]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fetch", "bake", "info", "index"])
    ap.add_argument("path", nargs="?", default=None)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    names = [n for n in a.only.split(",") if n] or list(REGIONS)
    for n in names:
        if n not in REGIONS:
            raise SystemExit("no region %r (tools/city_regions/<name>.json)" % n)
    if a.cmd == "fetch":
        fetch(names)
        fetch_ghsl(names)
    elif a.cmd == "bake":
        fetch_ghsl(names)
        ghsl = Ghsl()
        water = CityWater()
        # one region at a time: its arrays are freed before the next one's are made
        towers = {}
        for n in names:
            towers[n] = write_region(bake_region(n, ghsl, water))
        write_index_and_towers(towers)
        write_sources()
        info()
    elif a.cmd == "index":
        write_index_and_towers({})
        info()
    else:
        info(a.path)


if __name__ == "__main__":
    main()
