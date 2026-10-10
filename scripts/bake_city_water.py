"""Real-width city rivers for the real-placement water (docs/REAL_PLACEMENT.md
section 4): Overture Maps base.water polygons (OpenStreetMap water areas:
natural=water + water=river, waterway=riverbank, canals, docks, city lakes),
generalised for the 1/5 planet and baked into data/world/odbl/city_water.bin
(CWT1, ODbL: everything here is derived from OSM). Read and drawn by
scripts/world/rivers.gd; queried through Rivers.water_at / river_at and the
Water class.

Windows: every Natural Earth place over MIN_POP people or in AUDIT_CITIES
(15 km real around it) and every landmark (5 km real), merged where they
overlap. One DuckDB bbox query per window against the Overture release,
cached under data/world/src/overture/water/<window>.parquet.

Per window, on a 1/4800 deg raster (4.6 toy m):
  1. polygons rasterised by kind: river (river, stream), canal, dock, lake
     (lake, pond, reservoir, water; only when at least MIN_LAKE_M2 real m2),
     opened with a 2 x 2 cell square (water under ~46 real m wide goes)
  2. landmark keep-out (bake_water_ov.landmark_shift): water inside a dry
     landmark's structure hull (at its baked site) is shifted out, both
     banks, at most SHIFT_MAX toy m (not narrowed); water landmarks keep it
  3. islands (land enclosed by water, the sea included) narrower than
     ISLAND_KEEP toy m are flooded, wider ones (Ile de la Cite, Manhattan,
     Museum Island) protected
  4. widening: rivers and canals of real width w are drawn at least
     clamp(w / 2.2, 30, 120) toy m wide, around their own skeleton (lat / lon
     stay real), never into protected land, runway cores or the sea
  5. tidal city water (its source cell's DEM at most LOW_M + 1 real m, joined
     to the sea, outside the runway keep-out) becomes sea instead: COV1 carve
     patches in data/world/odbl/coast_ov_osm.bin (ODbL), so a widened tidal
     river (the Thames in London) is one body of water with the harbour carve
  6. the sea (Copernicus WBM ocean plus the coast_ov.bin carve, which must be
     baked first, plus step 5) and the Natural Earth lakes (lakes.bin) are cut
     out: those draw themselves
  7. whole lakes go (lake_drops): any within LAKE_RUNWAY toy m of a kept
     runway strip (no half-faded ghost ponds by the pads) and isolated small
     ones outside towns (rural reservoirs, fishponds); then everything within
     PAD_CUT toy m of a strip is cut (the pad blend)
  8. slivers under SLIVER toy m and bits under MIN_AREA toy m2 go; signed
     distance field (toy m)
Outputs:
  - mesh pieces: the water outline (contour of the distance field),
    Douglas-Peucker DP toy m, clipped to PIECE_DEG cells (one piece per
    cell), ear-clipped, refined
    to MAX_EDGE toy m edges and further where the per-vertex shore distance
    would interpolate badly
  - query tiles: the signed distance on 0.1 deg tiles (every kind is water
    for the queries: Water.kind_at reads city lakes as rivers)
  - Natural Earth river segments (rivers.bin) within NE_NEAR toy m of a city
    river, or out at sea, are dropped (the city water replaces them)

city_water.bin (CWT1, little endian):
  header 16 B: "CWT1", u32 version 1, u32 n_pieces, u32 n_ne_drops
  pieces n x 40 B: i32 lat0, lon0, lat1, lon1 (1e-5 deg bbox), u32 v_off,
    u32 nv, u32 i_off, u32 ni, u16 kind (0 river, 1 canal, 2 dock, 3 lake),
    u16 flags (0), u32 reserved (0)
  ne_drops n x 12 B: u32 Rivers piece id, u64 segment mask (bit k: the
    segment from the piece's point k to k + 1 is not drawn; all set = the
    whole piece)
  verts nv_total x 6 B: u16 lat, u16 lon (0..65535 across the piece bbox),
    u16 shore distance (0.1 toy m)
  indices ni_total x 2 B: u16, local to the piece
  query section: "CWQ1", u32 n_tiles, f32 RANGE (64 toy m), u32 samples
    per side (241), then n x 12 B {u32 key (ty * 3600 + tx, 0.1 deg tiles,
    tx = floor((lon + 180) / 0.1), ty = floor((90 - lat) / 0.1)), u32 offset
    (from file start), u32 size}, then the blobs: zlib of 241 x 241 u8,
    row 0 = north, sample (i, j) at lon = -180 + tx * 0.1 + i / 2400,
    lat = 90 - ty * 0.1 - j / 2400: the signed distance to the city water
    (sqrt code as COV1: v = round(127.5 + 127.5 sign(d) sqrt(|d| / RANGE)),
    + outside, 255 = none, saturated at Q_INSIDE inside the water)

Also writes data/world/odbl/SOURCES.txt (the Overture release and the bake
command) and data/world/odbl/LICENSE.txt (ODbL notice), and prints the river
offset before (Natural Earth ribbons) and after (city water) per city.

Run:  python tools/bake_city_water.py [--jobs N] [--only name,name] [--fetch-only]
      (the original bake: the audit-city and landmark windows, every one re-baked)
      CITY_SRC=C:/tmp/city_src python tools/bake_city_water.py --regions all|id,id [--jobs N] [--no-merge | --merge]
      (region mode, the detailed cities: see "region mode" below; parts are reused, the merge keeps the rest)
Needs: duckdb (pip install duckdb), shapely, contourpy, mapbox_earcut,
numpy, scipy, Pillow, and tools/bake_water_ov.py's output first.
"""
import argparse
import math
import os
import re
import struct
import sys
import threading
import time
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(__file__))
from world_scale import R_REAL, SCALE  # noqa: E402
import bake_water_ov as wov  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# CITY_SRC (a shared raw cache, as bake_city_buildings): the Overture water, the Copernicus tiles and the region
# parts go under CITY_SRC/water instead of data/world/src
_CS = os.environ.get("CITY_SRC")
WSRC = os.path.join(_CS, "water") if _CS else os.path.join(ROOT, "data", "world", "src", "water")
CACHE = os.path.join(WSRC, "overture") if _CS else os.path.join(ROOT, "data", "world", "src", "overture", "water")
COP_SRC = os.path.join(WSRC, "detail") if _CS else None  # bake_detail.SRC (cop90/) when set
PARTS = os.path.join(WSRC, "parts")
TOWNS_BIN = os.path.join(ROOT, "data", "world", "towns", "towns.bin")
ODBL = os.path.join(ROOT, "data", "world", "odbl")
OUT = os.path.join(ODBL, "city_water.bin")
OSM_COV = os.path.join(ODBL, "coast_ov_osm.bin")
LOG_DIR = "C:/tmp/realplace"
RELEASE = "2026-09-23.0"
S3 = f"s3://overturemaps-us-west-2/release/{RELEASE}/theme=base/type=water/*"
M_PER_DEG = R_REAL * math.pi / 180.0
TOY_PER_DEG = M_PER_DEG / SCALE

PPD = 4800
TILE_DEG = 0.1  # query tiles
Q_PPD = 2400  # query samples per degree (9.3 toy m)
TILE_C = 240  # cells per tile side (samples 241, edges shared)
TILE_W = 3600
Q_RANGE = 40.0
PIECE_DEG = 0.05
DP = 8.0
MAX_EDGE = 150.0
SHORE_TOL = 8.0
SHORE_MIN_EDGE = 30.0
SHORE_BAND = 30.0
Q_INSIDE = -8.0  # toy m: the query raster saturates deeper inside the water
MIN_LAKE_M2 = 250000.0  # real m2: the city lakes drawn (Central Park Reservoir, the Alster)
SLIVER = 10.0  # toy m: water narrower than this after the cuts goes
MIN_AREA = 3000.0  # toy m2: smaller separate bits of water go
ISLAND_KEEP = 40.0
MOUTH_RATIO = 1.5  # region mode: no widening from river cells whose width with the sea is over this x their own
NE_NEAR = 300.0  # toy m
NE_GAP = 5000.0  # toy m: region mode, the longest run of Natural Earth segments dropped between two dropped ones
RUNWAY_CORE = wov.CORE_M + 30.0  # toy m from a runway strip: the widening stops here
# toy m: the city water ends here (Airports' pad weight is 0.8 at CORE_M + 0.276 BLEND_M = 264 m); the river
# shader fades only the Natural Earth ribbons over pads, so a polygon must never reach into the pad blend
PAD_CUT = 270.0
LAKE_RUNWAY = 300.0  # toy m: a lake that comes this close to a kept runway strip goes whole (no ghost ponds)
LAKE_RUNWAY_BIG_M2 = 5e6  # real m2: unless it is bigger than this (then the pad cut clips it like a river)
LAKE_ALONE_M2 = 1e6  # real m2: smaller lakes go unless they touch a river / canal / dock or lie in a town
LAKE_RING = 100.0  # toy m (500 real m): the ring round a lake whose mean built share decides "in a town"
LAKE_LANDMARK_M = 1500.0  # real m: an isolated lake this close to a landmark site stays (Lake Hollywood)
LAKE_BUILT = 0.4  # TownData built share (GHSL + WorldCover, bake_towns' tiles15 raster) that counts as a town
TOWN_TILES = os.path.join(ROOT, "data", "world", "src", "ghsl", "tiles15")
MIN_POP = 0
PLACE_R = 15000.0  # real m
LANDMARK_R = 5000.0
K_RIVER, K_CANAL, K_DOCK, K_LAKE = 0, 1, 2, 3
# the 66 cities of the position audit (C:/tmp/pos_research/audit.md): with the
# landmark windows they are the city set (every place over MIN_POP too when
# MIN_POP > 0; 216 places over 2 M would cost ~10 MB, over the Phase 1 budget)
AUDIT_CITIES = {
    "London", "Paris", "New York", "Tokyo", "Los Angeles", "Chicago", "San Francisco", "Seattle", "Miami", "Toronto",
    "Vancouver", "Mexico City", "Rio de Janeiro", "Sao Paulo", "Buenos Aires", "Lima", "Bogota", "Madrid",
    "Barcelona", "Lisbon", "Rome", "Milan", "Naples", "Venice", "Berlin", "Munich", "Hamburg", "Amsterdam",
    "Brussels", "Vienna", "Prague", "Warsaw", "Stockholm", "Oslo", "Kobenhavn", "Helsinki", "Athens", "Istanbul",
    "Moscow", "Cairo", "Tel Aviv-Yafo", "Jerusalem", "Haifa", "Dubai", "Mumbai", "New Delhi", "Bangkok",
    "Singapore", "Hong Kong", "Shanghai", "Beijing", "Seoul", "Osaka", "Sydney", "Melbourne", "Auckland",
    "Cape Town", "Johannesburg", "Lagos", "Nairobi", "Zurich", "Geneva", "Nice", "Marseille", "Edinburgh", "Dublin",
}


# --------------------------------------------------------------------------- windows

def windows():
    """Merged boxes: [(name, lat0, lon0, lat1, lon1, members)] in tile-aligned degrees."""
    import json as _j  # noqa: PLC0415
    d = _j.load(open(os.path.join(ROOT, "data", "places.json"), encoding="utf-8"))
    W = []
    biggest = {}  # an audit name means its most populous place (not London, Ontario)
    for p in d["places"]:
        if p[0] in AUDIT_CITIES and p[3] > biggest.get(p[0], (0, 0, 0, -1))[3]:
            biggest[p[0]] = p
    for p in d["places"]:
        if (MIN_POP > 0 and p[3] >= MIN_POP) or biggest.get(p[0]) is p:
            W.append((p[0], float(p[1]), float(p[2]), PLACE_R))
    for k, L in wov.load_landmarks().items():
        W.append((k, L["lat"], L["lon"], LANDMARK_R))
    arenas = wov.load_arenas()
    boxes = []
    for n, la, lo, r in W:
        if any(wov.toy_dist(la, lo, a, b) < wov.ARENA_KEEP_M + 2.0 * r / SCALE + 2000.0 for a, b in arenas):
            print(f"window {n} skipped: within the PvP arena keep-out")
            continue
        dl = r / M_PER_DEG
        de = dl / math.cos(math.radians(la))
        boxes.append([la - dl, lo - de, la + dl, lo + de, [n]])
    changed = True
    while changed:
        changed = False
        out = []
        for b in boxes:
            for o in out:
                if b[0] <= o[2] and b[2] >= o[0] and b[1] <= o[3] and b[3] >= o[1]:
                    o[0], o[1], o[2], o[3] = min(o[0], b[0]), min(o[1], b[1]), max(o[2], b[2]), max(o[3], b[3])
                    o[4] += b[4]
                    changed = True
                    break
            else:
                out.append(b)
        boxes = out
    res = []
    for b in boxes:
        # tile aligned
        la0 = math.floor(b[0] / TILE_DEG) * TILE_DEG
        lo0 = math.floor(b[1] / TILE_DEG) * TILE_DEG
        la1 = math.ceil(b[2] / TILE_DEG) * TILE_DEG
        lo1 = math.ceil(b[3] / TILE_DEG) * TILE_DEG
        name = re.sub(r"[^a-z0-9]+", "_", b[4][0].lower()).strip("_")
        res.append((name, round(la0, 4), round(lo0, 4), round(la1, 4), round(lo1, 4), b[4]))
    res.sort()
    return res


def _detail():
    """bake_detail (mosaic_cop, fetch_cop) reading its Copernicus tiles from COP_SRC when CITY_SRC is set."""
    import bake_detail  # noqa: PLC0415
    if COP_SRC:
        bake_detail.SRC = COP_SRC
    return bake_detail


def cop_degrees(la0, lo0, la1, lo1):
    return [(L, G) for L in range(int(math.floor(la0)), int(math.ceil(la1)))
            for G in range(int(math.floor(lo0)), int(math.ceil(lo1)))]


# --------------------------------------------------------------------------- region mode (detailed cities)
#
# Every detailed-city region (bake_city_buildings.REGIONS: the built-in ones plus tools/city_regions/*.json) gets
# city water over its whole box, baked per region into part files and merged into the existing city_water.bin /
# coast_ov_osm.bin, so regions bake in parallel and only new or changed ones re-bake:
#   - a region owns the 0.1 deg query tiles of its box grown by REGION_PAD; a tile whose centre lies in a window of
#     the original bake (windows(): the audit cities and landmarks) stays the original's unless the region is in
#     REGION_FORCE (its original water was wrong); a tile two regions claim goes to the first by name
#   - the owned tiles are baked in chunks of at most CHUNK_TILES x CHUNK_TILES tiles, each with REGION_MARGIN deg
#     of context round it (lakes, islands and the widening see across the chunk edge), output clipped to the owned
#     tiles (pieces, query tiles, Natural Earth drops, the OSM carve)
#   - part file PARTS/r_<region>__<chunk window>__<hash>.pkl (hash: the window, the owned tiles, PART_VERSION, the
#     release); an existing part is reused
#   - merge: the pieces and query tiles of every owned tile are replaced, the Natural Earth drops OR-ed, the OSM
#     carve patches min-ed; idempotent (re-merging the same parts gives the same file)

REGION_PAD = 0.02
REGION_MARGIN = 0.1
CHUNK_TILES = 5
# original window water replaced: Singapore (the river mouth widening flooded Raffles Place), Toronto (baked against
# the Natural Earth Lake Ontario before tools/patch_lake_osm.py: city water left in the lake, none in the harbour)
REGION_FORCE = {"singapore", "toronto"}
PART_VERSION = 5  # v5: Natural Earth chords dropped by their ends, city lakes count (drops below; v4 was a trial)
SUNK_M = -5.0  # real m: a Copernicus lake cell under this is a below-sea-level lake (drawn as sea)
SUNK_GROW = 12  # cells (~280 real m): and its OSM shore round it


def _regions():
    import bake_city_buildings as cb  # noqa: PLC0415
    return cb.REGIONS


def water_boxes(rid):
    """A region's extra water boxes ("water_boxes": [[lat0, lon0, lat1, lon1], ...] in its tools/city_regions JSON):
    tiles it also owns for the city water only, outside its box and poly (Kinshasa: the Pool Malebo, whose north
    half lies outside the clipped city; the Natural Earth Congo crossed it as one straight ribbon over the fields)."""
    import json as _j  # noqa: PLC0415
    p = os.path.join(ROOT, "tools", "city_regions", rid + ".json")
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as f:
        return [tuple(b) for b in _j.load(f).get("water_boxes", [])]


def _tile_range(la0, lo0, la1, lo1):
    """(ty0, ty1, tx0, tx1) of the 0.1 deg query tiles a box covers (inclusive)."""
    return (int(math.floor((90.0 - la1) / TILE_DEG)), int(math.floor((90.0 - la0) / TILE_DEG)),
            int(math.floor((lo0 + 180.0) / TILE_DEG)), int(math.floor((lo1 + 180.0) / TILE_DEG)))


def region_plan(ids=None):
    """{region: [(chunk window, owned tile keys)]} for the regions (all when ids is None); the claims are resolved
    over ALL regions, so one region's plan never depends on which ones are asked for."""
    import shapely  # noqa: PLC0415
    from shapely.geometry import box as sbox  # noqa: PLC0415
    regs = _regions()
    base = windows()
    arenas = wov.load_arenas()
    arena_r = wov.ARENA_KEEP_M + TILE_DEG * 0.75 * TOY_PER_DEG + 2000.0
    claimed = {}
    out = {}
    for rid in sorted(regs):
        la0, lo0, la1, lo1 = regs[rid]["box"]
        poly = shapely.Polygon(regs[rid]["poly"]).buffer(REGION_PAD) if regs[rid].get("poly") else None
        ty0, ty1, tx0, tx1 = _tile_range(la0 - REGION_PAD, lo0 - REGION_PAD, la1 + REGION_PAD, lo1 + REGION_PAD)
        box_r = (ty0, ty1, tx0, tx1)
        # extra water boxes (tiles strictly inside: a box edge on a tile edge takes no neighbour)
        extra = set()
        for b in water_boxes(rid):
            ea, eb, ec, ed = _tile_range(b[0] + 1e-6, b[1] + 1e-6, b[2] - 1e-6, b[3] - 1e-6)
            extra |= {ty * TILE_W + tx for ty in range(ea, eb + 1) for tx in range(ec, ed + 1)}
            ty0, ty1, tx0, tx1 = min(ty0, ea), max(ty1, eb), min(tx0, ec), max(tx1, ed)
        keys = set()
        for ty in range(ty0, ty1 + 1):
            for tx in range(tx0, tx1 + 1):
                k = ty * TILE_W + tx
                if k in claimed:
                    continue
                cla = 90.0 - (ty + 0.5) * TILE_DEG
                clo = (tx + 0.5) * TILE_DEG - 180.0
                if rid not in REGION_FORCE and any(w[1] <= cla <= w[3] and w[2] <= clo <= w[4] for w in base):
                    continue
                if k not in extra and not (box_r[0] <= ty <= box_r[1] and box_r[2] <= tx <= box_r[3]):
                    continue
                if k not in extra and poly is not None and not poly.intersects(sbox(clo - TILE_DEG * 0.5, cla - TILE_DEG * 0.5,
                                                                                  clo + TILE_DEG * 0.5, cla + TILE_DEG * 0.5)):
                    continue
                if any(wov.toy_dist(cla, clo, aa, ab) < arena_r for aa, ab in arenas):
                    continue  # (as windows(): no city water within the PvP arena keep-out)
                claimed[k] = rid
                keys.add(k)
        chunks = []
        for cy in range(ty0, ty1 + 1, CHUNK_TILES):
            for cx in range(tx0, tx1 + 1, CHUNK_TILES):
                ck = {k for k in keys if cy <= k // TILE_W < cy + CHUNK_TILES and cx <= k % TILE_W < cx + CHUNK_TILES}
                if not ck:
                    continue
                cty = [k // TILE_W for k in ck]
                ctx = [k % TILE_W for k in ck]
                cla1 = round(90.0 - min(cty) * TILE_DEG + REGION_MARGIN, 4)
                cla0 = round(90.0 - (max(cty) + 1) * TILE_DEG - REGION_MARGIN, 4)
                clo0 = round(min(ctx) * TILE_DEG - 180.0 - REGION_MARGIN, 4)
                clo1 = round((max(ctx) + 1) * TILE_DEG - 180.0 + REGION_MARGIN, 4)
                chunks.append(((f"r_{rid}", cla0, clo0, cla1, clo1, [rid]), ck))
        out[rid] = chunks
    if ids is None:
        return out
    bad = [r for r in ids if r not in out]
    if bad:
        raise SystemExit("unknown regions: " + ",".join(bad))
    return {r: out[r] for r in ids}


_INPUTS = []


def job_inputs(win):
    """(landmarks, runways) a chunk window's bake sees (kept runways after the airport moves and thinning)."""
    if not _INPUTS:
        _INPUTS.extend((wov.load_landmarks(), wov.load_runways()))
    lms, runways = _INPUTS
    bb = (win[1] - 0.2, win[2] - 0.3, win[3] + 0.2, win[4] + 0.3)
    lm = {n: v for n, v in lms.items() if bb[0] <= v["lat"] <= bb[2] and bb[1] <= v["lon"] <= bb[3]}
    rw = [r for r in runways if bb[0] - 0.5 <= r[0] <= bb[2] + 0.5 and bb[1] - 0.8 <= r[1] <= bb[3] + 0.8]
    return lm, rw


def part_path(win, keys):
    """The part file of a chunk: keyed by the window, the owned tiles, PART_VERSION, the release and the inputs
    (landmark sites, kept runways), so an airport move or a landmark shift re-bakes the chunks it reaches."""
    import hashlib  # noqa: PLC0415
    lm, rw = job_inputs(win)
    lm_sig = sorted((n, round(float(v["lat"]), 5), round(float(v["lon"]), 5)) for n, v in lm.items())
    h = hashlib.sha1(repr((win[:5], sorted(keys), PART_VERSION, RELEASE, lm_sig, rw)).encode()).hexdigest()[:10]
    return os.path.join(PARTS, f"{win[0]}__{win[1]:.2f}_{win[2]:.2f}_{win[3]:.2f}_{win[4]:.2f}__{h}.pkl")


def read_cwt1(path):
    """(pieces [(head 40 B, verts bytes, idx bytes)], drops {pid: mask}, tiles {key: zlib blob})."""
    if not os.path.exists(path):
        return [], {}, {}
    b = open(path, "rb").read()
    assert b[:4] == b"CWT1"
    _, npc, nd = struct.unpack_from("<III", b, 4)
    heads = [b[16 + 40 * i:56 + 40 * i] for i in range(npc)]
    o = 16 + 40 * npc
    drops = {}
    for i in range(nd):
        pid, m = struct.unpack_from("<IQ", b, o + 12 * i)
        drops[pid] = m
    vo = o + 12 * nd
    nv = sum(struct.unpack_from("<iiiiIIII", h)[5] for h in heads)
    ni = sum(struct.unpack_from("<iiiiIIII", h)[7] for h in heads)
    io = vo + nv * 6
    pieces = []
    for h in heads:
        _, _, _, _, v_off, n_v, i_off, n_i = struct.unpack_from("<iiiiIIII", h)
        pieces.append((h, b[vo + v_off * 6:vo + (v_off + n_v) * 6], b[io + i_off * 2:io + (i_off + n_i) * 2]))
    q = io + ni * 2
    assert b[q:q + 4] == b"CWQ1"
    n, _, _ = struct.unpack_from("<IfI", b, q + 4)
    tiles = {}
    for i in range(n):
        k, off, size = struct.unpack_from("<III", b, q + 16 + 12 * i)
        tiles[k] = b[off:off + size]
    return pieces, drops, tiles


def write_cwt1_raw(path, pieces, drops, tiles):
    heads, vbuf, ibuf = [], bytearray(), bytearray()
    v_off = i_off = 0
    for h, vb, ib in pieces:
        a0, b0, a1, b1, _, n_v, _, n_i, kd, fl, rs = struct.unpack("<iiiiIIIIHHI", h)
        heads.append(struct.pack("<iiiiIIIIHHI", a0, b0, a1, b1, v_off, n_v, i_off, n_i, kd, fl, rs))
        vbuf += vb
        ibuf += ib
        v_off += n_v
        i_off += n_i
    keys = sorted(tiles)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(b"CWT1" + struct.pack("<III", 1, len(heads), len(drops)))
        for h in heads:
            f.write(h)
        for pid in sorted(drops):
            f.write(struct.pack("<IQ", pid, drops[pid]))
        f.write(vbuf)
        f.write(ibuf)
        base = f.tell()
        f.write(b"CWQ1" + struct.pack("<IfI", len(keys), Q_RANGE, TILE_C + 1))
        off = base + 16 + 12 * len(keys)
        for k in keys:
            f.write(struct.pack("<III", k, off, len(tiles[k])))
            off += len(tiles[k])
        for k in keys:
            f.write(tiles[k])
    os.replace(tmp, path)


def piece_key(h):
    """The 0.1 deg tile key of a piece (its bbox centre: a piece lies in one 0.05 deg cell)."""
    a0, b0, a1, b1 = struct.unpack_from("<iiii", h)
    la = (a0 + a1) * 0.5e-5
    lo = (b0 + b1) * 0.5e-5
    return int(math.floor((90.0 - la) / TILE_DEG)) * TILE_W + int(math.floor((lo + 180.0) / TILE_DEG))


def merge_parts(plan, out_path, cov_out):
    """Merge the region parts of `plan` into out_path / cov_out (both read, rewritten whole)."""
    import pickle  # noqa: PLC0415
    pieces, drops, tiles = read_cwt1(out_path)
    osm = {k: v[0] for k, v in wov.read_cov1(cov_out).items()} if os.path.exists(cov_out) else {}
    owned, parts, missing = set(), [], []
    for rid, chunks in sorted(plan.items()):
        for win, keys in chunks:
            pp = part_path(win, keys)
            if not os.path.exists(pp):
                missing.append(rid)
                continue
            owned |= set(keys)
            with open(pp, "rb") as f:
                parts.append(pickle.load(f))
    n0 = len(pieces)
    pieces = [p for p in pieces if piece_key(p[0]) not in owned]
    n_base = len(pieces)
    for k in list(tiles):
        if k in owned:
            del tiles[k]
    for res in parts:
        for lat, lon, shore, tris, kd in res["pieces"]:
            a0 = int(math.floor(lat.min() * 1e5))
            a1 = int(math.ceil(lat.max() * 1e5))
            b0 = int(math.floor(lon.min() * 1e5))
            b1 = int(math.ceil(lon.max() * 1e5))
            ql = np.rint((lat * 1e5 - a0) / max(a1 - a0, 1) * 65535).clip(0, 65535).astype(np.uint16)
            qo = np.rint((lon * 1e5 - b0) / max(b1 - b0, 1) * 65535).clip(0, 65535).astype(np.uint16)
            qs = np.rint(shore * 10.0).clip(0, 65535).astype(np.uint16)
            h = struct.pack("<iiiiIIIIHHI", a0, b0, a1, b1, 0, len(lat), 0, tris.size, kd, 0, 0)
            pieces.append((h, np.column_stack([ql, qo, qs]).astype("<u2").tobytes(), tris.astype("<u2").ravel().tobytes()))
        for k, v in res["tiles"].items():
            tiles[k] = zlib.compress(v.tobytes(), 9)
        for k, v in res["drops"].items():
            drops[k] = drops.get(k, 0) | v
        for k, v in res["osm"].items():
            osm[k] = np.minimum(osm[k], v) if k in osm else v
    write_cwt1_raw(out_path, pieces, drops, tiles)
    wov.write_cov1(cov_out, {k: (wov.enc(v), None) for k, v in osm.items()})
    print(f"merged {len(parts)} parts ({len(owned)} owned tiles): pieces {n0} -> {len(pieces)} ({n0 - n_base} replaced, "
          f"{len(pieces) - n_base} from the regions), {len(tiles)} query tiles, {len(drops)} NE drops, {len(osm)} carve "
          f"patches; {os.path.getsize(out_path) / 1e6:.2f} MB + {os.path.getsize(cov_out) / 1e6:.2f} MB", flush=True)
    if missing:
        print("regions with missing parts (bake them first): " + ",".join(sorted(set(missing))), flush=True)
    return missing


def coverage(plan, path):
    """Per region: its box tiles, the tiles it owns, the box tiles with city water (owned or kept from the original
    bake) and the water area inside the box tiles (query < 0, real km2)."""
    q = read_query(path)
    regs = _regions()
    lines = ["region\tbox tiles\towned\twater tiles\twater km2"]
    cell = (TILE_DEG / TILE_C) ** 2
    for rid in sorted(plan):
        la0, lo0, la1, lo1 = regs[rid]["box"]
        allk = [ty * TILE_W + tx for ty in range(int(math.floor((90.0 - la1) / TILE_DEG)), int(math.floor((90.0 - la0) / TILE_DEG)) + 1)
                for tx in range(int(math.floor((lo0 + 180.0) / TILE_DEG)), int(math.floor((lo1 + 180.0) / TILE_DEG)) + 1)]
        own = set()
        for _, k in plan[rid]:
            own |= set(k)
        wk = [k for k in allk if k in q]
        km2 = 0.0
        for k in wk:
            lat = 90.0 - (k // TILE_W + 0.5) * TILE_DEG
            km2 += float((q[k][:-1, :-1] < 0).sum()) * cell * (M_PER_DEG / 1e3) ** 2 * math.cos(math.radians(lat))
        lines.append(f"{rid}\t{len(allk)}\t{len(own)}\t{len(wk)}\t{km2:.1f}")
    return lines


_DUCK_LOCK = threading.Lock()


def fetch(win):
    name, la0, lo0, la1, lo1, _ = win
    # the Copernicus DEM / WBM under the window (a missing tile reads as ocean: no city water at all)
    bd = _detail()
    for L, G in cop_degrees(la0 - 2.0 / PPD, lo0, la1, lo1 + 2.0 / PPD):  # (the grid's last row / column too)
        bd.fetch_cop(L, G)
    p = os.path.join(CACHE, f"{name}_{la0:.2f}_{lo0:.2f}_{la1:.2f}_{lo1:.2f}.parquet")
    if os.path.exists(p):
        return p
    import duckdb  # noqa: PLC0415
    os.makedirs(CACHE, exist_ok=True)
    con = duckdb.connect()
    with _DUCK_LOCK:  # (parallel INSTALLs race on the extension file: "Could not move file")
        con.execute("INSTALL httpfs; INSTALL spatial;")
    con.execute("LOAD httpfs; LOAD spatial; SET s3_region='us-west-2';")
    part = p + f".{os.getpid()}.part"
    q = f"""COPY (SELECT id, subtype, class, names.primary AS name, is_intermittent, ST_AsWKB(geometry) AS wkb
        FROM read_parquet('{S3}', hive_partitioning=1)
        WHERE bbox.xmin < {lo1} AND bbox.xmax > {lo0} AND bbox.ymin < {la1} AND bbox.ymax > {la0}
        AND ST_GeometryType(geometry) IN ('POLYGON', 'MULTIPOLYGON')
        AND subtype NOT IN ('ocean', 'human_made', 'wastewater', 'physical', 'spring'))
        TO '{part.replace(os.sep, '/')}' (FORMAT parquet)"""
    for attempt in range(4):
        try:
            con.execute(q)
            break
        except Exception as e:  # noqa: BLE001 (network)
            print(f"  {name}: query failed ({e}), retry {attempt + 1}", flush=True)
            time.sleep(5 + attempt * 10)
    else:
        raise RuntimeError(f"Overture query failed for {name}")
    os.replace(part, p)
    return p


def kind_of(subtype, cls):
    cls = cls or ""
    if cls in ("swimming_pool", "fountain", "reflecting_pool", "basin", "moat", "wastewater", "fish_pass"):
        return None
    if subtype in ("river", "stream") or cls in ("river", "riverbank", "stream"):
        return K_RIVER
    if cls in ("dock", "harbour", "marina", "lock"):
        return K_DOCK
    if subtype == "canal" or cls == "canal":
        return K_CANAL
    if subtype in ("lake", "pond", "reservoir", "water", "lagoon"):
        return K_LAKE
    return None


# --------------------------------------------------------------------------- Natural Earth inputs

def read_rivers():
    b = open(os.path.join(ROOT, "data", "world", "rivers.bin"), "rb").read()
    assert b[:4] == b"RIV1"
    npc, nd = struct.unpack_from("<II", b, 4)
    head = np.frombuffer(b, np.int32, npc * 9, 12).reshape(npc, 9)
    deltas = np.frombuffer(b, np.int16, nd * 2, 12 + npc * 36).reshape(nd, 2)
    return head, deltas


def piece_points(head, deltas, pid):
    o = head[pid]
    cnt = int(o[7]) & 0xFFFF
    first = int(o[6])
    d = deltas[first:first + cnt - 1].astype(np.int64)
    la = np.concatenate([[int(o[0])], int(o[0]) + np.cumsum(d[:, 0])]) * 1e-5
    lo = np.concatenate([[int(o[1])], int(o[1]) + np.cumsum(d[:, 1])]) * 1e-5
    return la, lo


def read_lakes():
    b = open(os.path.join(ROOT, "data", "world", "lakes.bin"), "rb").read()
    assert b[:4] == b"LAK1"
    nl, nv, ni = struct.unpack_from("<III", b, 4)
    head = np.frombuffer(b, np.int32, nl * 9, 16).reshape(nl, 9)
    vo = 16 + nl * 36
    vert = np.frombuffer(b, np.uint16, nv * 3, vo).reshape(nv, 3)
    idx = np.frombuffer(b, np.uint16, ni, vo + nv * 6)
    return head, vert, idx


# --------------------------------------------------------------------------- per window

def raster_polys(polys, g, img_size):
    """Fill mask (ny, nx) of shapely polygons (lon / lat) on the point grid."""
    from PIL import Image, ImageDraw  # noqa: PLC0415
    ext = Image.new("1", img_size, 0)
    hole = Image.new("1", img_size, 0)
    de, dh = ImageDraw.Draw(ext), ImageDraw.Draw(hole)

    def tr(coords):
        a = np.asarray(coords)
        return [((x - g.lon0) * g.ppd + 0.5, (g.lat1 - y) * g.ppd + 0.5) for x, y in a[:, :2]]

    for p in polys:
        if len(p.exterior.coords) >= 3:
            de.polygon(tr(p.exterior.coords), fill=1)
        for r in p.interiors:
            if len(r.coords) >= 3:
                dh.polygon(tr(r.coords), fill=1)
    return np.array(ext, bool) & ~np.array(hole, bool)


def sea_mask(g, cov):
    """WBM ocean plus the coast_ov.bin carve on the window's grid, and the
    WBM classes and DEM (real m) there (nearest 3" sample)."""
    bd = _detail()
    la0 = int(math.floor(g.lat1 - g.ny / g.ppd))
    la1 = int(math.ceil(g.lat1))
    lo0 = int(math.floor(g.lon0))
    lo1 = int(math.ceil(g.lon0 + g.nx / g.ppd))
    for L, G in cop_degrees(la0, lo0, la1, lo1):
        n, pd, pw = bd.cop_paths(L, G)
        if n in bd.cop_names() and not (os.path.exists(pd) and os.path.exists(pw)):
            raise RuntimeError(f"Copernicus tile {n} missing (run the fetch first): it would read as ocean")
    dem, wbm = bd.mosaic_cop(la0, la1, lo0, lo1)
    lats = g.lats()
    lons = g.lons()
    r = np.clip(np.rint((la1 - lats) * 1200).astype(np.int64), 0, wbm.shape[0] - 1)
    c = np.clip(np.rint((lons - lo0) * 1200).astype(np.int64), 0, wbm.shape[1] - 1)
    ocean = wbm[r[:, None], c[None, :]] == 1
    LO, LA = np.meshgrid(lons, lats)
    carve = wov.carve_at(cov, LA, LO) < 0.0
    return ocean | carve, wbm[r[:, None], c[None, :]], dem[r[:, None], c[None, :]]


def ne_lake_mask(g, lakes):
    from PIL import Image, ImageDraw  # noqa: PLC0415
    head, vert, idx = lakes
    lat0 = g.lat1 - g.ny / g.ppd
    lon1 = g.lon0 + g.nx / g.ppd
    img = Image.new("1", (g.nx, g.ny), 0)
    dr = ImageDraw.Draw(img)
    hit = False
    for li in range(head.shape[0]):
        o = head[li]
        a0, b0, a1, b1 = o[0] * 1e-5, o[1] * 1e-5, o[2] * 1e-5, o[3] * 1e-5
        if a1 < lat0 or a0 > g.lat1 or b1 < g.lon0 or b0 > lon1:
            continue
        v0, nv, i0, ni = int(o[5]), int(o[6]), int(o[7]), int(o[8])
        vv = vert[v0:v0 + nv].astype(np.float64)
        la = a0 + vv[:, 0] / 65535.0 * (a1 - a0)
        lo = b0 + vv[:, 1] / 65535.0 * (b1 - b0)
        x = (lo - g.lon0) * g.ppd + 0.5
        y = (g.lat1 - la) * g.ppd + 0.5
        tri = idx[i0:i0 + ni].reshape(-1, 3)
        for t in tri:
            dr.polygon([(x[t[0]], y[t[0]]), (x[t[1]], y[t[1]]), (x[t[2]], y[t[2]])], fill=1)
        hit = True
    return np.array(img, bool) if hit else np.zeros((g.ny, g.nx), bool)


def shore_refine(va, tris, dist_fn, passes=8):
    """bake_rivers.shore_refine with the shore distance from the raster: the
    river shader tints by the interpolated distance, so edges whose midpoint
    lies further from the bank than the interpolation says are bisected."""
    sd = dist_fn(va)
    tris = [tuple(t) for t in tris]
    for _ in range(passes):
        use = {}
        for a, b, c in tris:
            for e in ((a, b), (b, c), (c, a)):
                k = (min(e), max(e))
                use[k] = use.get(k, 0) + 1
        cand = [k for k, n in use.items() if n == 2]
        if not cand:
            break
        ca = np.array(cand, dtype=np.int64)
        pa, pb = va[ca[:, 0]], va[ca[:, 1]]
        ln = np.hypot(pa[:, 0] - pb[:, 0], pa[:, 1] - pb[:, 1])
        idx = np.nonzero(ln > SHORE_MIN_EDGE)[0]
        split = {}
        if idx.size:
            mid = (pa[idx] + pb[idx]) * 0.5
            md = dist_fn(mid)
            ip = (sd[ca[idx, 0]] + sd[ca[idx, 1]]) * 0.5
            err = md - ip
            for j in np.nonzero((err > np.maximum(SHORE_TOL, md * 0.35)) & (ip < SHORE_BAND))[0]:
                split[cand[idx[j]]] = len(va) + len(split)
        if not split:
            break
        new = np.zeros((len(split), 2))
        for k, i in split.items():
            new[i - len(va)] = (va[k[0]] + va[k[1]]) * 0.5
        va = np.vstack([va, new])
        sd = np.concatenate([sd, dist_fn(new)])
        out = []
        for t in tris:
            a, b, c = t
            mab = split.get((min(a, b), max(a, b)))
            mbc = split.get((min(b, c), max(b, c)))
            mca = split.get((min(c, a), max(c, a)))
            n = (mab is not None) + (mbc is not None) + (mca is not None)
            if n == 0:
                out.append(t)
            elif n == 3:
                out += [(a, mab, mca), (mab, b, mbc), (mca, mbc, c), (mab, mbc, mca)]
            else:
                while mab is None or (n == 2 and mbc is None):
                    a, b, c = b, c, a
                    mab, mbc, mca = mbc, mca, mab
                if n == 1:
                    out += [(a, mab, c), (mab, b, c)]
                else:
                    out += [(mab, b, mbc), (a, mab, mbc), (a, mbc, c)]
        tris = out
    return va, tris, sd


def built_grid(g):
    """TownData's built share (0..1: bake_towns' 240 per degree tiles15 raster,
    GHSL BUILT-S + WorldCover) at the window's cells, nearest sample."""
    j = np.floor((90.0 - g.lats()) * 240.0).astype(np.int64)
    i = np.floor((g.lons() + 180.0) * 240.0).astype(np.int64)
    out = np.zeros((g.ny, g.nx), np.float32)
    tw = None if os.path.isdir(TOWN_TILES) else towns_tiles()
    for ty in np.unique(j // 240):
        rs = np.nonzero(j // 240 == ty)[0]
        for tx in np.unique(i // 240):
            if tw is not None:
                # no tiles15 cache on this machine: the same plane from towns.bin (TWN1 built share)
                a = tw.get(int(ty) * 360 + int(tx % 360))
                if a is None:
                    continue
            else:
                p = os.path.join(TOWN_TILES, "%03d_%03d.npz" % (ty, tx % 360))
                if not os.path.exists(p):
                    continue
                a = np.load(p)["a"]
            if a.ndim != 3:
                continue
            cs = np.nonzero(i // 240 == tx)[0]
            out[np.ix_(rs, cs)] = a[0][np.ix_(j[rs] - ty * 240, i[cs] - tx * 240)] / 255.0
    return out


class _TownTiles:
    """towns.bin (TWN1) tiles: key (ty * 360 + tx) -> 3 x 240 x 240 u8 (built, height, nonres)."""

    def __init__(self):
        self.b = open(TOWNS_BIN, "rb").read() if os.path.exists(TOWNS_BIN) else b""
        self.index = {}
        if self.b[:4] == b"TWN1":
            _, _, nt, _, _, toff, _ = struct.unpack_from("<7I", self.b, 4)
            for k in range(nt):
                key, off, size, _ = struct.unpack_from("<4I", self.b, toff + 16 * k)
                self.index[key] = (off, size)

    def get(self, key):
        if key not in self.index:
            return None
        off, size = self.index[key]
        return np.frombuffer(zlib.decompress(self.b[off:off + size]), np.uint8).reshape(3, 240, 240)


_TW = []


def towns_tiles():
    if not _TW:
        _TW.append(_TownTiles())
    return _TW[0]


_SITES = []


def landmark_sites():
    """[(id, lat, lon)] of data/landmarks/sites.json (the baked sites)."""
    if not _SITES:
        import json as _j  # noqa: PLC0415
        d = _j.load(open(os.path.join(ROOT, "data", "landmarks", "sites.json"), encoding="utf-8"))
        _SITES.extend((k, float(v["lat"]), float(v["lon"])) for k, v in d.items() if "lat" in v)
    return _SITES


def lake_drops(water, kind, g, runways, sea, name, log):
    """(cells of the city lakes that go, counts). Whole connected lakes go:
    (a) one within LAKE_RUNWAY toy m of a kept runway strip, unless over
    LAKE_RUNWAY_BIG_M2 real m2 (the pad cut then clips it like a river): the
    pad blend would leave a half-transparent ghost pond beside the runway;
    (b) an isolated small one: under LAKE_ALONE_M2 real m2, touching no river,
    canal or dock and not in a town (mean built share in a LAKE_RING ring at
    most LAKE_BUILT): rural reservoirs, fishponds and retention basins, flat
    stickers at 1/5 scale. City park lakes (Central Park Reservoir) and lakes
    on a river (the Alster) stay, and so does an isolated lake within
    LAKE_LANDMARK_M real m of a landmark site (Lake Hollywood by the sign)."""
    lake = water & (kind == K_LAKE)
    out = np.zeros(water.shape, bool)
    n_all = {"lakes": 0, "runway": 0, "alone": 0, "clipped": 0, "landmark": 0}
    if not lake.any():
        return out, n_all
    lab, n = ndimage.label(lake, structure=np.ones((3, 3), bool))
    n_all["lakes"] = int(n)
    area = np.bincount(lab.ravel(), minlength=n + 1).astype(np.float64) * g.sy * g.sx * SCALE * SCALE
    near = np.zeros(n + 1, bool)
    near[np.unique(lab[wov.runway_mask(g, runways, keep=LAKE_RUNWAY) & lake])] = True
    touch = np.zeros(n + 1, bool)
    flow = water & ~lake
    if flow.any():
        adj = ndimage.binary_dilation(flow, structure=np.ones((3, 3), bool), iterations=2)
        touch[np.unique(lab[adj & lake])] = True
    small = ~near & ~touch & (area < LAKE_ALONE_M2)
    small[0] = False
    built = np.full(n + 1, np.nan)
    if small.any():
        d, (ir, ic) = ndimage.distance_transform_edt(~lake, sampling=g.samp, return_indices=True)
        ring = (d <= LAKE_RING) & ~water & ~sea
        del d
        lr = lab[ir[ring], ic[ring]]
        del ir, ic
        bg = built_grid(g)[ring]
        cnt = np.bincount(lr, minlength=n + 1)
        built = np.bincount(lr, weights=bg, minlength=n + 1) / np.maximum(cnt, 1)
    gone_rw = near & (area < LAKE_RUNWAY_BIG_M2)
    gone_alone = small & (built <= LAKE_BUILT)
    by_lm = {}
    if gone_alone.any():
        lat0 = g.lat1 - g.ny / g.ppd
        for sid, sla, slo in landmark_sites():
            if not (lat0 - 0.05 <= sla <= g.lat1 + 0.05 and g.lon0 - 0.08 <= slo <= g.lon0 + g.nx / g.ppd + 0.08):
                continue
            m = np.zeros(lake.shape, bool)
            g.disc(m, sla, slo, LAKE_LANDMARK_M / SCALE)
            for k in np.unique(lab[m & lake]):
                if k and gone_alone[k]:
                    by_lm.setdefault(int(k), sid)
        for k in by_lm:
            gone_alone[k] = False
    gone_rw[0] = gone_alone[0] = False
    rows = ndimage.mean(np.broadcast_to(np.arange(g.ny)[:, None], lake.shape), lab, np.arange(1, n + 1))
    cols = ndimage.mean(np.broadcast_to(np.arange(g.nx)[None, :], lake.shape), lab, np.arange(1, n + 1))
    for k in range(1, n + 1):
        why = "by a runway" if gone_rw[k] else ("alone" if gone_alone[k] else ("big, by a runway: clipped" if near[k] else (
            f"isolated, near landmark {by_lm[k]}: kept" if k in by_lm else None)))
        if why is None:
            continue
        la = g.lat1 - rows[k - 1] / g.ppd
        lo = g.lon0 + cols[k - 1] / g.ppd
        log.append(f"city {name}: lake at {la:.4f},{lo:.4f} {area[k] / 1e6:.3f} km2 real, built {built[k]:.2f}: "
                   f"{'kept, ' if why.endswith(('clipped', 'kept')) else 'dropped, '}{why}")
    out = (gone_rw | gone_alone)[lab]
    n_all["runway"] = int(gone_rw.sum())
    n_all["alone"] = int(gone_alone.sum())
    n_all["landmark"] = len(by_lm)
    n_all["clipped"] = int((near & ~gone_rw).sum() - near[0])
    return out, n_all


def bake_window(args):
    win, parquet, cov_path, lms, runways = args[:5]
    own = args[5] if len(args) > 5 else None  # region mode: the 0.1 deg tile keys this window outputs
    import duckdb  # noqa: PLC0415
    import shapely  # noqa: PLC0415
    from shapely.geometry import box as sbox  # noqa: PLC0415
    import contourpy  # noqa: PLC0415
    import mapbox_earcut as earcut  # noqa: PLC0415
    sys.path.insert(0, os.path.dirname(__file__))
    from bake_rivers import refine  # noqa: PLC0415
    t0 = time.time()
    name, la0, lo0, la1, lo1, members = win
    log = []
    ny = int(round((la1 - la0) * PPD)) + 1
    nx = int(round((lo1 - lo0) * PPD)) + 1
    g = wov.Grid(la1, lo0, ny, nx, PPD)
    if own is not None:
        cty = np.floor((90.0 - g.lats()) / TILE_DEG + 1e-6).astype(np.int64)
        ctx_ = np.floor((g.lons() + 180.0) / TILE_DEG + 1e-6).astype(np.int64)
        own_mask = np.isin(cty[:, None] * TILE_W + ctx_[None, :], np.array(sorted(own), np.int64))
    else:
        own_mask = np.ones((ny, nx), bool)
    rows = duckdb.connect().execute(f"SELECT subtype, class, is_intermittent, wkb FROM '{parquet.replace(os.sep, '/')}'").fetchall()
    by_kind = {K_RIVER: [], K_CANAL: [], K_DOCK: [], K_LAKE: []}
    for st, cl, inter, wkb in rows:
        k = kind_of(st, cl)
        if k is None or (inter and k == K_LAKE):
            continue
        geom = shapely.from_wkb(bytes(wkb))
        parts = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
        for p in parts:
            if k == K_LAKE:
                a = p.area * M_PER_DEG * M_PER_DEG * math.cos(math.radians(p.centroid.y))
                if a < MIN_LAKE_M2:
                    continue
            by_kind[k].append(p)
    masks = {k: raster_polys(v, g, (nx, ny)) if v else np.zeros((ny, nx), bool) for k, v in by_kind.items()}
    # water under ~2 cells (MIN_REAL_W) goes: canal belts and small streams would
    # otherwise widen into one sheet of water (Amsterdam, Venice)
    for k in masks:
        masks[k] = ndimage.binary_opening(masks[k], structure=np.ones((2, 2), bool))
    water = masks[K_RIVER] | masks[K_CANAL] | masks[K_DOCK] | masks[K_LAKE]
    kind = np.full((ny, nx), 255, np.uint8)
    for k in (K_LAKE, K_DOCK, K_CANAL, K_RIVER):
        kind[masks[k]] = k
    cov = wov.read_cov1(cov_path) if os.path.exists(cov_path) else {}
    sea, wbm, dem = sea_mask(g, cov)
    ne_lakes = ne_lake_mask(g, read_lakes())
    if own is not None:
        # lakes below sea level (the Dead Sea, Kinneret: Copernicus lake cells under SUNK_M) are the game's sea sheet
        # (bake_coast LOW_LAKES), never city water; region mode only (the original windows never reached one)
        sunk = (wbm == 2) & (dem < SUNK_M)
        if sunk.any():
            sunk = ndimage.binary_dilation(sunk, iterations=SUNK_GROW)
        ne_lakes = ne_lakes | sunk
    core = wov.runway_mask(g, runways, keep=RUNWAY_CORE)
    # landmark keep-out (the river shifted out of a structure, both banks),
    # islands (the sea counts as water for enclosure), widening
    prot = wov.landmark_shift(water, g, lms, log, f"city {name}", extra=(kind,))
    narrow, wide = wov.island_masks(water | sea, g, ISLAND_KEEP, adjacent_to=water)
    fill = narrow & ~sea & ~prot
    water |= fill
    flowing = (kind == K_RIVER) | (kind == K_CANAL)
    may_grow = ~prot & ~wide & ~sea & ~core & ~ne_lakes
    seeds = flowing & water
    if own is not None:
        # region mode (v2): a river cell whose width is the sea's (a mouth on a bay or harbour carve: the Singapore
        # River on Marina Bay) is no ridge to widen from; it was pushing WIDEN_MAX discs into the bank
        d_own = ndimage.distance_transform_edt(water, sampling=g.samp)
        d_all = ndimage.distance_transform_edt(water | sea, sampling=g.samp)
        seeds &= d_all <= d_own * MOUTH_RATIO + min(g.samp)
        del d_own, d_all
    grown = wov.widen(water, g, may_grow, seeds_in=seeds, ctx=sea)
    # the kind of grown / filled cells: the nearest source cell's
    src = water.copy()
    water |= grown
    _, (ir, ic) = ndimage.distance_transform_edt(~src, sampling=g.samp, return_indices=True)
    kind = kind[ir, ic]
    # tidal city water (at sea level by its source cell's DEM, joined to the
    # sea) is drawn as sea: the OSM carve (coast_ov_osm.bin), so a widened
    # tidal river is one body of water with the harbour carve
    low = water & (dem[ir, ic] <= wov.LOW_M + 1.0) & ~sea
    lab, _ = ndimage.label(low | sea, structure=np.ones((3, 3), bool))
    ids = np.unique(lab[sea])
    osm_carve = low & np.isin(lab, ids[ids > 0]) & ~wov.runway_mask(g, runways) & own_mask
    del lab, low
    osm = {}
    if osm_carve.any():
        sdc = wov.signed_dist(osm_carve, g)
        sdc = np.where((sdc > wov.LAND_KEEP) | (sea & (sdc > wov.OCEAN_KEEP)), 1e9, sdc).astype(np.float32)
        for py in range(int(math.floor((90.0 - la1) / wov.PATCH_DEG)), int(math.ceil((90.0 - la0) / wov.PATCH_DEG))):
            for px in range(int(math.floor((lo0 + 180.0) / wov.PATCH_DEG)), int(math.ceil((lo1 + 180.0) / wov.PATCH_DEG))):
                plat = 90.0 - py * wov.PATCH_DEG - np.arange(wov.PATCH_N) / wov.PPD
                plon = -180.0 + px * wov.PATCH_DEG + np.arange(wov.PATCH_N) / wov.PPD
                rr = (la1 - plat) * PPD
                cc = (plon - lo0) * PPD
                v = ndimage.map_coordinates(sdc, [np.repeat(rr, wov.PATCH_N), np.tile(cc, wov.PATCH_N)], order=1,
                                            mode="constant", cval=1e9).reshape(wov.PATCH_N, wov.PATCH_N)
                if v.min() < wov.RANGE:
                    osm[py * wov.PATCH_W + px] = v.astype(np.float32)
    sea = sea | osm_carve
    # cut what other layers draw, the lakes that go whole (ponds by runways,
    # small isolated rural lakes), the runway pads, then the slivers (not
    # inside the landmark discs: the widening never reaches there, so a narrow
    # arm is kept)
    water &= ~sea & ~ne_lakes
    gone, lake_n = lake_drops(water, kind, g, runways, sea, name, log)
    water &= ~gone & ~core
    del gone
    pad = wov.runway_mask(g, runways, keep=PAD_CUT)
    # (the Natural Earth drops below still see the rivers inside the pad cut)
    riv_pad = water & pad & ((kind == K_RIVER) | (kind == K_CANAL))
    water &= ~pad
    del pad
    rad = max(1, int(round(SLIVER * 0.5 / g.sx)))
    st = ndimage.generate_binary_structure(2, 1)
    water = (ndimage.binary_opening(water, structure=st, iterations=rad) | prot) & water
    lab, nl = ndimage.label(water, structure=np.ones((3, 3), bool))
    if nl:
        area = np.bincount(lab.ravel())[1:] * g.sy * g.sx
        water &= np.concatenate([[False], area >= MIN_AREA])[lab]
    del lab
    sd = wov.signed_dist(water, g)

    # ---- query tiles
    tiles = {}
    tx0 = int(round((lo0 + 180.0) / TILE_DEG))
    ty0 = int(round((90.0 - la1) / TILE_DEG))
    st = PPD // Q_PPD
    sdq = sd[::st, ::st]
    for j in range(int(round((la1 - la0) / TILE_DEG))):
        for i in range(int(round((lo1 - lo0) / TILE_DEG))):
            blk = sdq[j * TILE_C:j * TILE_C + TILE_C + 1, i * TILE_C:i * TILE_C + TILE_C + 1]
            if blk.shape != (TILE_C + 1, TILE_C + 1) or blk.min() >= Q_RANGE:
                continue
            if own is not None and (ty0 + j) * TILE_W + tx0 + i not in own:
                continue
            tiles[(ty0 + j) * TILE_W + tx0 + i] = enc_q(np.maximum(blk, Q_INSIDE))

    # ---- mesh pieces
    pieces = []
    if water.any():
        sx, sy = g.sx, g.sy
        gen = contourpy.contour_generator(z=sd.astype(np.float64), fill_type=contourpy.FillType.OuterOffset)
        pts_list, off_list = gen.filled(-1e12, 0.0)
        polys = []
        for pts, offs in zip(pts_list, off_list):
            rings = [pts[offs[k]:offs[k + 1]] for k in range(len(offs) - 1)]
            rings = [np.column_stack([r[:, 0] * sx, -r[:, 1] * sy]) for r in rings if len(r) >= 4]
            if not rings:
                continue
            p = shapely.Polygon(rings[0], rings[1:]).buffer(0)
            p = p.simplify(DP, preserve_topology=True)
            if not p.is_empty:
                polys.append(p)
        if polys:
            allp = shapely.union_all(polys)

            def sd_at(xy):
                return np.maximum(-ndimage.map_coordinates(sd, [-xy[:, 1] / sy, xy[:, 0] / sx], order=1, mode="nearest"), 0.0)

            for pj in range(int(math.ceil((la1 - la0) / PIECE_DEG))):
                for pi in range(int(math.ceil((lo1 - lo0) / PIECE_DEG))):
                    if own is not None:
                        ck = (int(math.floor((90.0 - la1 + (pj + 0.5) * PIECE_DEG) / TILE_DEG)) * TILE_W
                              + int(math.floor((lo0 + (pi + 0.5) * PIECE_DEG + 180.0) / TILE_DEG)))
                        if ck not in own:
                            continue
                    bx = sbox(pi * PIECE_DEG * PPD * sx, -(pj + 1) * PIECE_DEG * PPD * sy,
                              (pi + 1) * PIECE_DEG * PPD * sx, -pj * PIECE_DEG * PPD * sy)
                    part = allp.intersection(bx)
                    if part.is_empty:
                        continue
                    geoms = [q for q in getattr(part, "geoms", [part]) if q.geom_type == "Polygon" and q.area > 1.0]
                    # every polygon of the cell in one piece (one record, one bbox)
                    pv, pt, ps = [], [], []
                    base = 0
                    for q in geoms:
                        verts = []
                        rings = [np.asarray(q.exterior.coords)[:-1]] + [np.asarray(r.coords)[:-1] for r in q.interiors]
                        ends = []
                        for r in rings:
                            verts.extend(map(tuple, r[:, :2]))
                            ends.append(len(verts))
                        if len(verts) < 3:
                            continue
                        tri = earcut.triangulate_float64(np.asarray(verts, np.float64), np.asarray(ends, np.uint32))
                        if len(tri) == 0:
                            continue
                        vl = list(verts)
                        tris = refine(vl, tri.reshape(-1, 3).tolist(), MAX_EDGE)
                        va, tris, shore = shore_refine(np.asarray(vl, np.float64), tris, sd_at)
                        pv.append(va)
                        ps.append(shore)
                        pt.append(np.asarray(tris, np.int64) + base)
                        base += len(va)
                    if not pv:
                        continue
                    va = np.vstack(pv)
                    shore = np.concatenate(ps)
                    tris = np.vstack(pt)
                    if len(va) > 65535:
                        log.append(f"city {name}: piece over 65535 verts dropped")
                        continue
                    # kind: the majority of the triangle centroids' kinds
                    cen = va[tris].mean(axis=1)
                    kc = kind[np.clip(np.rint(-cen[:, 1] / sy).astype(np.int64), 0, ny - 1),
                              np.clip(np.rint(cen[:, 0] / sx).astype(np.int64), 0, nx - 1)]
                    kc = kc[kc < 4]
                    kd = int(np.bincount(kc, minlength=4).argmax()) if kc.size else K_RIVER
                    lat = la1 - (-va[:, 1] / sy) / PPD
                    lon = lo0 + (va[:, 0] / sx) / PPD
                    pieces.append((lat, lon, shore, tris, kd))

    # ---- Natural Earth segments the city water replaces
    head, deltas = read_rivers()
    riv = (water & ((kind == K_RIVER) | (kind == K_CANAL))) | riv_pad
    if own is not None:
        riv |= water  # region mode (v4): a river lake too (the Pool Malebo is an OSM lake on the Congo)
    d_riv = ndimage.distance_transform_edt(~riv, sampling=g.samp) if riv.any() else None
    drops = {}
    sel = np.nonzero((head[:, 4] * 1e-5 >= la0) & (head[:, 2] * 1e-5 <= la1) & (head[:, 5] * 1e-5 >= lo0) & (head[:, 3] * 1e-5 <= lo1))[0]
    for pid in sel:
        la, lo = piece_points(head, deltas, int(pid))
        mask = 0
        inside = []  # region mode: the segments wholly in the owned tiles
        for k in range(len(la) - 1):
            pla = np.array([la[k], (la[k] + la[k + 1]) * 0.5, la[k + 1]])
            plo = np.array([lo[k], (lo[k] + lo[k + 1]) * 0.5, lo[k + 1]])
            r = np.rint((la1 - pla) * PPD).astype(np.int64)
            c = np.rint((plo - lo0) * PPD).astype(np.int64)
            if (r < 0).any() or (r >= ny).any() or (c < 0).any() or (c >= nx).any():
                continue
            if not own_mask[r, c].all():
                continue
            inside.append(k)
            # region mode (v4): both ends near the city river is enough: a Natural Earth chord across an island or a
            # bend (the Congo across Mbamu island in the Pool Malebo) ran as a straight ribbon over the fields
            near = d_riv is not None and bool((d_riv[r[::2] if own is not None else r, c[::2] if own is not None else c]
                                               <= NE_NEAR).all())
            at_sea = bool(sea[r, c].all())
            if near or at_sea:
                mask |= 1 << k
        if mask and own is not None:
            # region mode (v5): a run of owned segments between two dropped ones, at most NE_GAP toy m long, goes too
            # (the Congo's Natural Earth line across Mbamu island, 15 real km from the Pool Malebo's shores)
            run = []
            for k in range(len(la) - 1):
                if mask >> k & 1:
                    if run and k - 1 in run and min(run) - 1 >= 0 and mask >> (min(run) - 1) & 1:
                        ln = sum(wov.toy_dist(la[q], lo[q], la[q + 1], lo[q + 1]) for q in run)
                        if ln <= NE_GAP:
                            for q in run:
                                mask |= 1 << q
                    run = []
                elif k in inside:
                    run.append(k)
                else:
                    run = []
        if mask:
            drops[int(pid)] = mask
    stats = {"name": name, "members": members[:6], "water_km2": float((water & own_mask).sum() * g.sy * g.sx * SCALE * SCALE / 1e6),
             "grown_km2": float((grown & water & own_mask).sum() * g.sy * g.sx * SCALE * SCALE / 1e6),
             "pieces": len(pieces), "verts": sum(len(p[0]) for p in pieces),
             "tile_kb": sum(len(zlib.compress(v.tobytes(), 6)) for v in tiles.values()) / 1024.0,
             "tiles": len(tiles), "drops": len(drops), "s": time.time() - t0}
    stats["osm_carve_km2"] = float(osm_carve.sum() * g.sy * g.sx * SCALE * SCALE / 1e6)
    stats["lakes"] = lake_n
    return {"pieces": pieces, "tiles": tiles, "drops": drops, "osm": osm, "log": log, "stats": stats}


def enc_q(d):
    t = np.sqrt(np.clip(np.abs(d) / Q_RANGE, 0.0, 1.0)) * np.sign(d)
    b = np.rint(127.5 + 127.5 * t)
    b = np.where(d >= 0.0, np.maximum(b, 128), np.minimum(b, 127))
    return b.astype(np.uint8)


def dec_q(b):
    t = (b.astype(np.float32) - 127.5) / 127.5
    return np.sign(t) * t * t * Q_RANGE


# --------------------------------------------------------------------------- output

def write_cwt1(path, pieces, drops, tiles):
    vbuf = bytearray()
    ibuf = bytearray()
    heads = []
    v_off = 0
    i_off = 0
    for lat, lon, shore, tris, kd in pieces:
        a0 = int(math.floor(lat.min() * 1e5))
        a1 = int(math.ceil(lat.max() * 1e5))
        b0 = int(math.floor(lon.min() * 1e5))
        b1 = int(math.ceil(lon.max() * 1e5))
        ql = np.rint((lat * 1e5 - a0) / max(a1 - a0, 1) * 65535).clip(0, 65535).astype(np.uint16)
        qo = np.rint((lon * 1e5 - b0) / max(b1 - b0, 1) * 65535).clip(0, 65535).astype(np.uint16)
        qs = np.rint(shore * 10.0).clip(0, 65535).astype(np.uint16)
        vbuf += np.column_stack([ql, qo, qs]).astype("<u2").tobytes()
        ibuf += tris.astype("<u2").ravel().tobytes()
        heads.append(struct.pack("<iiiiIIIIHHI", a0, b0, a1, b1, v_off, len(lat), i_off, tris.size, kd, 0, 0))
        v_off += len(lat)
        i_off += tris.size
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = sorted(tiles)
    blobs = [zlib.compress(tiles[k].tobytes(), 9) for k in keys]
    with open(path, "wb") as f:
        f.write(b"CWT1" + struct.pack("<III", 1, len(heads), len(drops)))
        for h in heads:
            f.write(h)
        for pid in sorted(drops):
            f.write(struct.pack("<IQ", pid, drops[pid]))
        f.write(vbuf)
        f.write(ibuf)
        base = f.tell()
        f.write(b"CWQ1" + struct.pack("<IfI", len(keys), Q_RANGE, TILE_C + 1))
        off = base + 16 + 12 * len(keys)
        for k, b in zip(keys, blobs):
            f.write(struct.pack("<III", k, off, len(b)))
            off += len(b)
        for b in blobs:
            f.write(b)


def read_query(path):
    """key -> signed distance float32 toy m."""
    b = open(path, "rb").read()
    assert b[:4] == b"CWT1"
    q = b.index(b"CWQ1")
    n, rng, ns = struct.unpack_from("<IfI", b, q + 4)
    out = {}
    for i in range(n):
        k, o, s = struct.unpack_from("<III", b, q + 16 + 12 * i)
        out[k] = dec_q(np.frombuffer(zlib.decompress(b[o:o + s]), np.uint8).reshape(ns, ns))
    return out


def write_notes():
    os.makedirs(ODBL, exist_ok=True)
    lic = (
        "Files in this folder (data/world/odbl/) are derived from OpenStreetMap data through the\r\n"
        "Overture Maps Foundation releases and are made available under the Open Database License (ODbL)\r\n"
        "v1.0: https://opendatacommons.org/licenses/odbl/1-0/\r\n"
        "Map data (c) OpenStreetMap contributors, ODbL; Overture Maps Foundation.\r\n"
        "Each file's source snapshot and the command that rebuilds it are listed in SOURCES.txt.\r\n"
    )
    open(os.path.join(ODBL, "LICENSE.txt"), "w", encoding="utf-8", newline="").write(lic)
    sp = os.path.join(ODBL, "SOURCES.txt")
    lines = open(sp, encoding="utf-8").read().splitlines() if os.path.exists(sp) else []
    lines = [ln for ln in lines if not ln.startswith(("city_water.bin", "coast_ov_osm.bin"))]
    for f in ("city_water.bin", "coast_ov_osm.bin"):
        lines.append(f"{f}: Overture Maps base.water release {RELEASE} (OpenStreetMap water areas), "
                     f"python tools/bake_city_water.py, baked {time.strftime('%Y-%m-%d')}")
    open(sp, "w", encoding="utf-8", newline="").write("\r\n".join(lines) + "\r\n")


# --------------------------------------------------------------------------- offsets (report)

OFFSET_CITIES = [("paris", 48.857, 2.35), ("london", 51.507, -0.12), ("prague", 50.08, 14.42),
                 ("cairo", 30.05, 31.24), ("bangkok", 13.75, 100.50), ("budapest", 47.50, 19.05)]


def offsets(cwt_path, wins):
    """Real river centre points (the skeleton of the OSM river polygons at
    least 60 real m wide) vs the nearest drawn water: Natural Earth ribbons
    (before) and the city water or the harbour carve (after), real m."""
    import duckdb  # noqa: PLC0415
    import shapely  # noqa: PLC0415
    head, deltas = read_rivers()
    q = read_query(cwt_path)
    cov = wov.read_cov1(wov.OUT) if os.path.exists(wov.OUT) else {}
    if os.path.exists(OSM_COV):
        for k, v in wov.read_cov1(OSM_COV).items():
            cov[k] = (np.minimum(cov[k][0], v[0]), cov[k][1]) if k in cov else v
    lines = ["city\tbefore (NE ribbon) median / p90 real m\tafter (city water or carve) median / p90 real m\tpoints"]
    for cname, lat, lon in OFFSET_CITIES:
        w = [x for x in wins if x[1] <= lat <= x[3] and x[2] <= lon <= x[4]]
        if not w:
            continue
        w = w[0]
        p = fetch(w)
        g = wov.Grid(w[3], w[2], int(round((w[3] - w[1]) * PPD)) + 1, int(round((w[4] - w[2]) * PPD)) + 1, PPD)
        polys = []
        for st, cl, wkb in duckdb.connect().execute(f"SELECT subtype, class, wkb FROM '{p.replace(os.sep, '/')}'").fetchall():
            if kind_of(st, cl) == K_RIVER:
                geom = shapely.from_wkb(bytes(wkb))
                polys += list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
        m = raster_polys(polys, g, (g.nx, g.ny))
        # 8 km real around the centre only
        r0 = int((g.lat1 - lat) * PPD)
        c0 = int((lon - g.lon0) * PPD)
        hr = int(8000.0 / M_PER_DEG * PPD)
        hc = int(hr / math.cos(math.radians(lat)))
        sub = np.zeros_like(m)
        sub[max(r0 - hr, 0):r0 + hr, max(c0 - hc, 0):c0 + hc] = True
        dt = ndimage.distance_transform_edt(m, sampling=g.samp)
        ridge = m & sub & (dt >= ndimage.maximum_filter(dt, size=5) - 1e-3) & (dt * SCALE > 30.0)
        rr, cc = np.nonzero(ridge)
        if rr.size == 0:
            continue
        sel = np.random.default_rng(1).choice(rr.size, min(rr.size, 3000), replace=False)
        rr, cc = rr[sel], cc[sel]
        plat = g.lat1 - rr / PPD
        plon = g.lon0 + cc / PPD
        # before: distance to the NE centrelines minus the ribbon half width
        segs = []
        for pid in np.nonzero((head[:, 4] * 1e-5 >= g.lat1 - g.ny / PPD) & (head[:, 2] * 1e-5 <= g.lat1)
                              & (head[:, 5] * 1e-5 >= g.lon0) & (head[:, 3] * 1e-5 <= g.lon0 + g.nx / PPD))[0]:
            la, lo = piece_points(head, deltas, int(pid))
            rank = (int(head[pid, 7]) >> 16) & 0xFF
            hw = [48.0, 44.0, 40.0, 36.0, 32.0, 28.0, 25.0, 22.0, 19.0, 16.0, 13.0, 11.0, 9.0][min(rank, 12)]
            for k in range(len(la) - 1):
                segs.append((la[k], lo[k], la[k + 1], lo[k + 1], hw))
        cosl = math.cos(math.radians(lat))
        if segs:
            S = np.array(segs)
            ax, ay = (S[:, 1] - lon) * TOY_PER_DEG * cosl, (S[:, 0] - lat) * TOY_PER_DEG
            bx, by = (S[:, 3] - lon) * TOY_PER_DEG * cosl, (S[:, 2] - lat) * TOY_PER_DEG
            px, py = (plon - lon) * TOY_PER_DEG * cosl, (plat - lat) * TOY_PER_DEG
            dx, dy = bx - ax, by - ay
            L = dx * dx + dy * dy + 1e-9
            best = np.full(px.size, 1e9)
            for s in range(0, px.size, 500):
                t = np.clip(((px[s:s + 500, None] - ax) * dx + (py[s:s + 500, None] - ay) * dy) / L, 0, 1)
                d = np.hypot(ax + t * dx - px[s:s + 500, None], ay + t * dy - py[s:s + 500, None]) - S[:, 4]
                best[s:s + 500] = np.maximum(d.min(axis=1), 0.0)
            before = best * SCALE
        else:
            before = np.full(px.size, np.nan)
        # after: the city water's signed distance at the real centre point
        after = np.zeros(plat.size)
        for i in range(plat.size):
            tx = int(math.floor((plon[i] + 180.0) / TILE_DEG))
            ty = int(math.floor((90.0 - plat[i]) / TILE_DEG))
            t = q.get(ty * TILE_W + tx)
            if t is None:
                after[i] = Q_RANGE
                continue
            u = (plon[i] + 180.0 - tx * TILE_DEG) * Q_PPD
            v = (90.0 - plat[i] - ty * TILE_DEG) * Q_PPD
            after[i] = max(float(ndimage.map_coordinates(t, [[v], [u]], order=1)[0]), 0.0)
        after = np.minimum(after, np.maximum(wov.carve_at(cov, plat, plon), 0.0)) * SCALE
        lines.append(f"{cname}\t{np.nanmedian(before):.0f} / {np.nanpercentile(before, 90):.0f}\t"
                     f"{np.median(after):.0f} / {np.percentile(after, 90):.0f}\t{plat.size}")
    return lines


def main_regions(a):
    import pickle  # noqa: PLC0415
    ids = None if a.regions == "all" else [r for r in a.regions.split(",") if r]
    plan = region_plan(ids)
    t0 = time.time()
    if not a.merge and not a.coverage:
        todo = [(rid, w, k) for rid, ch in sorted(plan.items()) for w, k in ch
                if a.force or not os.path.exists(part_path(w, k))]
        print(f"{len(plan)} regions, {sum(len(c) for c in plan.values())} chunks, {len(todo)} to bake", flush=True)
        from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415
        with ThreadPoolExecutor(max_workers=4) as ex:
            paths = list(ex.map(fetch, [w for _, w, _ in todo]))
        print(f"fetched in {time.time() - t0:.0f} s", flush=True)
        if a.fetch_only:
            return
        jobs = []
        for (rid, w, k), p in zip(todo, paths):
            lm, rw = job_inputs(w)
            jobs.append((w, p, a.cov, lm, rw, frozenset(k)))
        os.makedirs(PARTS, exist_ok=True)
        logs = []
        with ProcessPoolExecutor(max_workers=max(1, a.jobs)) as ex:
            for i, (job, res) in enumerate(zip(jobs, ex.map(bake_window, jobs))):
                s = res["stats"]
                print(f"[{i + 1}/{len(jobs)}] {s['name']} {job[0][1:5]}: {s['water_km2']:.1f} km2 real-eq water "
                      f"({s['grown_km2']:.1f} widened), {s['pieces']} pieces / {s['verts']} verts, {s['tiles']} tiles, "
                      f"{s['drops']} NE pieces cut, carve {s['osm_carve_km2']:.1f} km2 ({s['s']:.0f} s)", flush=True)
                pp = part_path(job[0], job[5])
                with open(pp + ".tmp", "wb") as f:
                    pickle.dump(res, f, protocol=4)
                os.replace(pp + ".tmp", pp)
                logs += res["log"]
        with open(os.path.join(LOG_DIR, "water_pushes_regions.txt"), "a", encoding="utf-8") as f:
            f.write("\n".join(logs) + "\n")
        if a.no_merge:
            return
    if not a.coverage:
        # the merge always covers every region with parts on disk, so a partial run never drops the others
        missing = merge_parts(region_plan(None), a.out, OSM_COV)
        write_notes()
        if ids is not None:
            missing = [r for r in missing if r in ids]
        if missing:
            print("WARNING: missing parts for " + ",".join(sorted(set(missing))))
    lines = coverage(plan, a.out)
    print("\n".join(lines))
    with open(os.path.join(LOG_DIR, "city_water_regions.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"{time.time() - t0:.0f} s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--only", default="", help="comma separated window names (tests)")
    ap.add_argument("--fetch-only", action="store_true")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--cov", default=wov.OUT)
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--regions", default="", help="region mode: all, or comma separated detailed-city regions")
    ap.add_argument("--no-merge", action="store_true", help="region mode: bake the parts only (parallel runs)")
    ap.add_argument("--merge", action="store_true", help="region mode: merge the existing parts only")
    ap.add_argument("--force", action="store_true", help="region mode: re-bake parts that exist")
    ap.add_argument("--coverage", action="store_true", help="region mode: print the per-region water table")
    a = ap.parse_args()
    os.makedirs(LOG_DIR, exist_ok=True)
    if a.regions:
        return main_regions(a)
    wins = windows()
    if a.report_only:
        print("\n".join(offsets(a.out, wins)))
        return
    if a.only:
        keep = set(a.only.split(","))
        wins = [w for w in wins if w[0] in keep or keep & set(w[5])]
    print(f"{len(wins)} windows", flush=True)
    t0 = time.time()
    from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415
    with ThreadPoolExecutor(max_workers=4) as ex:
        paths = list(ex.map(fetch, wins))
    print(f"fetched in {time.time() - t0:.0f} s", flush=True)
    if a.fetch_only:
        return
    lms = wov.load_landmarks()
    runways = wov.load_runways()
    jobs = []
    for w, p in zip(wins, paths):
        bb = (w[1] - 0.2, w[2] - 0.3, w[3] + 0.2, w[4] + 0.3)
        lm = {k: v for k, v in lms.items() if bb[0] <= v["lat"] <= bb[2] and bb[1] <= v["lon"] <= bb[3]}
        rw = [r for r in runways if bb[0] - 0.5 <= r[0] <= bb[2] + 0.5 and bb[1] - 0.8 <= r[1] <= bb[3] + 0.8]
        jobs.append((w, p, a.cov, lm, rw))
    pieces, tiles, drops, logs, osm = [], {}, {}, [], {}
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for i, res in enumerate(ex.map(bake_window, jobs)):
            s = res["stats"]
            print(f"[{i + 1}/{len(jobs)}] {s['name']} {s['members']}: {s['water_km2']:.1f} km2 real-eq water "
                  f"({s['grown_km2']:.1f} widened), {s['pieces']} pieces / {s['verts']} verts, {s['tiles']} tiles / {s['tile_kb']:.0f} KB, "
                  f"{s['drops']} NE pieces cut ({s['s']:.0f} s)", flush=True)
            pieces += res["pieces"]
            tiles.update(res["tiles"])
            for k, v in res["drops"].items():
                drops[k] = drops.get(k, 0) | v
            logs += res["log"]
            for k, v in res["osm"].items():
                osm[k] = np.minimum(osm[k], v) if k in osm else v
    write_cwt1(a.out, pieces, drops, tiles)
    wov.write_cov1(OSM_COV, {k: (wov.enc(v), None) for k, v in osm.items()})
    write_notes()
    with open(os.path.join(LOG_DIR, "water_pushes.txt"), "a", encoding="utf-8") as f:
        f.write("\n".join(logs) + "\n")
    nv = sum(len(p[0]) for p in pieces)
    ni = sum(p[3].size for p in pieces)
    lines = [f"CWT1 {len(pieces)} pieces ({nv} verts, {ni // 3} tris), {len(tiles)} query tiles, "
             f"{len(drops)} NE pieces cut, {os.path.getsize(a.out) / 1e6:.2f} MB; coast_ov_osm.bin {len(osm)} patches, "
             f"{os.path.getsize(OSM_COV) / 1e6:.2f} MB; {time.time() - t0:.0f} s"]
    if not a.only:
        lines += offsets(a.out, wins)
    print("\n".join(lines))
    with open(os.path.join(LOG_DIR, "city_water_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
