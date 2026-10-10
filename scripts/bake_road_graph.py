"""Bake the connected road graph: data/world/odbl/roadgraph.bin (RGR1).

docs/ROAD_NETWORK.md sections 3 and 5 (owner RB). Source: Overture Maps
transportation/segment (OpenStreetMap, ODbL), release OVERTURE below, read with
DuckDB straight from the public S3 bucket, one pass per source file (only the
columns needed). The output is an ODbL Derivative Database and
lives under data/world/odbl/ (see data/world/odbl/SOURCES.txt).

Scope (section 5): motorway and trunk worldwide; primary within 40 km real of
every TownData town of 100k+ (towns.bin) and inside the 246 roads.bin city boxes;
secondary and the named avenues of tools/roads_iconic.json inside those boxes.

Pipeline (deterministic: rows sorted by Overture id, no time stamps, no set
iteration order in the output):
1. topology: segments split at connectors shared with another kept segment;
2. dual carriageways: antiparallel one-way pieces of a class within DUAL_R
   (DUAL_R_FAST for motorway / trunk) are paired, one of each pair is dropped and
   the kept one moves onto the mean line (F_DUAL);
3. slip roads: each cluster of link segments becomes a junction where its main
   roads cross (the edges there get F_LINK), or one kept LINK path between two
   main roads that do not cross; the rest goes;
4. cleanup: edges under CONTRACT collapse into a point (roundabouts, turning
   loops), short self loops and parallel duplicates go;
5. repair until stable: snap ends to an edge within SNAP_TOY, prune spurs under
   PRUNE_TOY, every remaining end becomes a sink (TownData town extent, a kept
   airport within AIRPORT_TOY, a ferry end) or is extended up to EXTEND_TOY
   (F_SYNTH feeder), else a dead end over PARK_TOY ends at a car park sink, else
   it is deleted back to its junction;
6. degree-2 collapse, island removal, bridge / tunnel structure records from
   road_flags (F_SEA_OK where the game coast raster says sea), Douglas-Peucker
   DP_TOY, edges split at EDGE_MAX_TOY;
7. tiles (RGR1 version 2, see encode() in tests/fixtures/roads/
   make_roadgraph_fixture.py): a 1-degree tile over TILE_EDGES edges becomes
   4 x 4 quarter-degree sub-tiles, and every tile payload is one zstd frame
   (header flags bit 2). The fixtures stay version 1.

The world does not fit in memory at once (12.5M segments: 40 GB+), so phase A
runs the steps 1 to 4 per 10 x 10 degree block (MARGIN of context, RB_WORKERS
processes) and clips the result to the block; phase B joins the blocks,
stitches the border crossings (mutual nearest within STITCH) and runs 5 to 7
on the whole graph.

Caches (data/world/src/overture/roadgraph/, gitignored, junctioned in worktrees,
never deleted through the junction): part_<nnn>.parquet, one per Overture segment
file (a per-bbox query cannot prune on class, so it re-reads the bbox column of
the whole release: one pass per file is ~66 GB once instead).

Usage:
  python tools/bake_road_graph.py                fetch the missing caches, then bake
  python tools/bake_road_graph.py --fetch-only   fill the caches
  python tools/bake_road_graph.py --no-fetch     bake from the caches only
  python tools/bake_road_graph.py --no-fetch --only=31.9,34.6,32.3,35.0 --name ta
                                                 a bbox (lat0,lon0,lat1,lon1) to C:/tmp/roadgraph_ta.bin
  python tools/bake_road_graph.py --no-fetch --check   decode the file (A1, layout) and rebake (sha256)
Dev knobs (no effect on the output): RB_WORKERS, RB_REUSE_A=<pickle> (cache
phase A), RB_PARTS=<glob> (other caches), RB_SPLIT=1 (block-split an --only).
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import math
import os
import struct
import sys
import threading
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "data", "world", "src")
CACHE = os.path.join(SRC, "overture", "roadgraph")
TOWNS_BIN = os.path.join(ROOT, "data", "world", "towns", "towns.bin")
ROADS_BIN = os.path.join(ROOT, "data", "world", "odbl", "roads.bin")
ICONIC_JSON = os.path.join(ROOT, "tools", "roads_iconic.json")
AIRPORTS = os.path.join(ROOT, "data", "airports.json")
MOVES = os.path.join(ROOT, "data", "airports", "moves.json")
KEEP = os.path.join(ROOT, "data", "airports", "keep.json")
OUT = os.path.join(ROOT, "data", "world", "odbl", "roadgraph.bin")
SOURCES = os.path.join(ROOT, "data", "world", "odbl", "SOURCES.txt")

OVERTURE = "2026-09-23.1"
SEGMENTS = ("s3://overturemaps-us-west-2/release/%s/theme=transportation/type=segment/*.parquet" % OVERTURE)

R_REAL = 6371008.8
SCALE = 5.0
TOY_DEG = R_REAL / SCALE * math.pi / 180.0  # toy m per degree of latitude (22,239)
KM_DEG = R_REAL * math.pi / 180.0 / 1000.0  # real km per degree
CHUNK = 10                                   # degrees per motorway / trunk cache chunk
BIG_POP = 100000
PRIMARY_KM = 40.0                            # real km round every 100k+ town


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------- inputs

def load_towns():
    """TownData towns (towns.bin TWN1): list of dicts lat, lon, pop, gid, ext (toy m), name, cc."""
    with open(TOWNS_BIN, "rb") as f:
        b = f.read()
    assert b[:4] == b"TWN1"
    _ver, n, _nt, _ns, towns_off, _to, names_off = struct.unpack_from("<7I", b, 4)
    out = []
    for i in range(n):
        o = towns_off + i * 32
        lat, lon, pop, gid = struct.unpack_from("<iiII", b, o)
        ext, = struct.unpack_from("<H", b, o + 20)
        noff, cc = struct.unpack_from("<IH", b, o + 24)
        e = b.index(b"\0", names_off + noff)
        out.append({"i": i, "lat": lat * 1e-6, "lon": lon * 1e-6, "pop": pop, "gid": gid, "ext": float(ext),
                    "name": b[names_off + noff:e].decode("utf-8"),
                    "cc": bytes([cc & 255, cc >> 8]).decode("ascii", "replace")})
    return out


def load_boxes(towns):
    """The 246 roads.bin (RDN1) city boxes: list of dicts gid, lat, lon, box (lon0, lat0, lon1, lat1),
    icon (iconic names of that city)."""
    with open(ROADS_BIN, "rb") as f:
        b = f.read()
    assert b[:4] == b"RDN1"
    _v, n, coff, _no, _nn = struct.unpack_from("<5I", b, 4)
    icon = json.load(open(ICONIC_JSON, encoding="utf-8"))["cities"]
    by_gid = {t["gid"]: t for t in towns if t["gid"]}
    out = []
    for i in range(n):
        gid, lat, lon, _bear, rad, _lo, _nl, _z = struct.unpack_from("<IddffIII", b, coff + i * 40)
        r = (rad + 1000.0) / TOY_DEG
        rx = r / max(0.05, math.cos(math.radians(lat)))
        t = by_gid.get(gid)
        names = []
        if t is not None:
            for c in icon:
                if c["cc"] == t["cc"] and c["city"].lower() == t["name"].lower():
                    names = list(c["names"])
        out.append({"gid": gid, "lat": lat, "lon": lon, "box": (lon - rx, lat - r, lon + rx, lat + r), "icon": names,
                    "name": t["name"] if t else str(gid)})
    out.sort(key=lambda c: c["gid"])
    # iconic cities absent from roads.bin under that exact name still match by nearest box
    return out


# ---------------------------------------------------------------- fetch

_con = None
_con_lock = threading.Lock()


def _duck():
    global _con
    with _con_lock:
        if _con is None:
            import duckdb
            _con = duckdb.connect()
            _con.execute("INSTALL httpfs; LOAD httpfs; INSTALL spatial; LOAD spatial;")
            _con.execute("SET s3_region='us-west-2'; SET enable_http_metadata_cache=true; "
                         "SET parquet_metadata_cache=true;")
        return _con.cursor()


def _sql_str(s):
    return "'" + s.replace("'", "''") + "'"


# heading 'backward' denied for everyone, everywhere on the segment = one way along the geometry
_OW = ("CASE WHEN len(list_filter(access_restrictions, x -> x.access_type = 'denied' AND x.\"between\" IS NULL "
       "AND x.\"when\".heading = '%s' AND x.\"when\".\"using\" IS NULL AND x.\"when\".recognized IS NULL "
       "AND x.\"when\".\"mode\" IS NULL AND x.\"when\".vehicle IS NULL AND x.\"when\".during IS NULL)) > 0")
COLS = ("id, subtype, class, subclass, names.primary AS name, list_extract(routes, 1).\"ref\" AS ref, "
        "connectors, list_filter(road_flags, x -> list_has_any(x.\"values\", ['is_bridge', 'is_tunnel'])) "
        "AS road_flags, level_rules, " + (_OW % "backward") + " THEN 1 " + (_OW % "forward")[4:] +
        " THEN -1 ELSE 0 END AS ow, ST_AsWKB(geometry) AS wkb")


def part_path(i):
    return os.path.join(CACHE, "part_%03d.parquet" % i)


def source_files():
    con = _duck()
    files = [r[0] for r in con.execute("SELECT file FROM glob('%s') ORDER BY file" % SEGMENTS).fetchall()]
    con.close()
    return files


def fetch_part(i, url, icon_names):
    """One Overture segment file: every motorway, trunk, primary, secondary, every named avenue of
    tools/roads_iconic.json and every ferry (subtype water). One pass over the file: the class filter cannot
    prune row groups, so per-bbox queries would re-read the bbox column of the whole release each time."""
    path = part_path(i)
    if os.path.exists(path):
        return path, 0.0
    t0 = time.time()
    exact = sorted({n.lower() for n in icon_names if "%" not in n})
    pats = sorted({n for n in icon_names if "%" in n})
    cond = ["lower(names.primary) IN (%s)" % ", ".join(_sql_str(n) for n in exact)] if exact else []
    cond += ["names.primary ILIKE %s" % _sql_str(n) for n in pats]
    icon = "(" + " OR ".join(cond) + ")" if cond else "FALSE"
    q = f"""
    COPY (
      SELECT {COLS}, bbox.xmin AS x0, bbox.ymin AS y0, bbox.xmax AS x1, bbox.ymax AS y1
      FROM read_parquet('{url}', hive_partitioning = 0)
      WHERE (subtype = 'road' AND (class IN ('motorway', 'trunk', 'primary', 'secondary') OR {icon}))
         OR subtype = 'water'
    ) TO '{path.replace(chr(92), '/')}.tmp' (FORMAT parquet)"""
    cur = _duck()
    cur.execute(q)
    cur.close()
    os.replace(path + ".tmp", path)
    return path, time.time() - t0


def run_fetch(jobs):
    os.makedirs(CACHE, exist_ok=True)
    files = source_files()
    icon = []
    for c in json.load(open(ICONIC_JSON, encoding="utf-8"))["cities"]:
        icon += c["names"]
    todo = [i for i in range(len(files)) if not os.path.exists(part_path(i))]
    log("fetch: %d of %d Overture %s segment files to do" % (len(todo), len(files), OVERTURE))
    t0 = time.time()
    done = 0
    with cf.ThreadPoolExecutor(jobs) as ex:
        futs = {ex.submit(fetch_part, i, files[i], icon): i for i in todo}
        for f in cf.as_completed(futs):
            done += 1
            try:
                path, dt = f.result()
                log("  [%d/%d] part %d %.0f s %.1f MB (%.0f s total)" % (
                    done, len(todo), futs[f], dt, os.path.getsize(path) / 1e6, time.time() - t0))
            except Exception as ex_:
                log("  [%d/%d] part %d FAILED: %s" % (done, len(todo), futs[f], ex_))



# ---------------------------------------------------------------- bake constants

sys.path.insert(0, os.path.join(ROOT, "tests", "fixtures", "roads"))
import make_roadgraph_fixture as rgf  # noqa: E402  (encode(): the RGR1 reference encoder)

MOTORWAY, TRUNK, PRIMARY, SECONDARY, ICONIC, FEEDER, LINK = range(7)
F_ONEWAY, F_DUAL, F_LINK, F_BRIDGE, F_TUNNEL, F_FERRY, F_SYNTH, F_SEA_OK = 1, 2, 4, 8, 16, 32, 64, 128
S_TOWN, S_AIRPORT, S_PORT, S_FERRY, S_PARKING, S_LANDMARK = range(1, 7)
ST_BRIDGE, ST_TUNNEL = 1, 2
F_SLIP = 256          # internal (never written): a source slip-road edge awaiting the link collapse
CLS_OF = {"motorway": MOTORWAY, "trunk": TRUNK, "primary": PRIMARY, "secondary": SECONDARY}
CLS_KM_NAMES = ["motorway", "trunk", "primary", "secondary", "iconic", "feeder", "link"]
WIDTH = {MOTORWAY: 22, TRUNK: 18, PRIMARY: 16, SECONDARY: 12, ICONIC: 20, FEEDER: 8, LINK: 8}
LANES_DUAL = {MOTORWAY: 3, TRUNK: 2, PRIMARY: 2, SECONDARY: 2, ICONIC: 2}
LANES_ONE = {MOTORWAY: 2, TRUNK: 1, PRIMARY: 1, SECONDARY: 1, ICONIC: 1, FEEDER: 1, LINK: 1}
SINK_KM_NAMES = ["", "town", "airport", "port", "ferry", "parking", "landmark"]

M_DEG = R_REAL * math.pi / 180.0   # real m per degree of latitude
TOY = 1.0 / SCALE                  # toy m per real m
DUAL_R = 45.0          # real m: antiparallel one-way carriageways this close are one road
DUAL_R_FAST = 150.0    # real m, motorway / trunk (the Ayalon runs a railway and a river down its median)
CONTRACT = 100.0       # real m (20 toy m): shorter edges collapse into a point (roundabouts, turning loops,
                       # the two crossings of a dual carriageway)
ISLAND_M = 3000.0      # real m: a whole component shorter than this is a fragment
NODE_MERGE = 30.0      # real m: nodes this close are one place (carriageway seams the dual merge leaves)
LOOP_MAX = 400.0       # real m: a self loop shorter than this is a turning circle, removed
DUAL_STEP = 25.0       # real m between pairing samples
DUAL_MIN = 0.6         # a carriageway is merged away when this share of it runs beside a kept one
DUAL_DOT = -0.8        # antiparallel: cos of the angle between the two travel directions
JUNC_R = 300.0         # real m round a slip-road cluster where its main roads may cross
SNAP_TOY = 60.0        # repair: snap an end to an edge this close (toy m)
MERGE_TOY = 12.0       # a snap this short moves the end onto the edge instead of adding a link
PRUNE_TOY = 400.0      # repair: dead-end spurs shorter than this are pruned (toy m)
EXTEND_TOY = 300.0     # repair: extend an end this far toward the nearest edge (toy m)
AIRPORT_TOY = 3000.0   # an end this close to an airport is its landside sink (toy m)
FERRY_TOY = 120.0      # an end this close to a ferry route end is a ferry sink (toy m)
PARK_TOY = 2000.0      # a dead end this far from its junction ends at a car park sink (toy m)
TOWN_MIN_EXT = 200.0   # toy m: the smallest town extent a road end may fall in
STRUCT_MIN = 80.0      # real m (16 toy m, under one 20 m profile step): shorter bridge / tunnel ranges are not
                       # written (A12: 30 MB; RS can still tell a grade separation by two drawn roads crossing
                       # without a shared node)
STRUCT_GAP = 20.0      # real m: ranges of one kind closer than this are one structure
DP_TOY = 8.0           # Douglas-Peucker tolerance (toy m; 4 gives 37.8 MB, over A12)
EDGE_MAX_TOY = 20000.0 # edges are split so every point stays within EDGE_SPAN 25 km of node a's tile
PT_MAX_DEG = 0.4       # interior point deltas stay inside i16 x PQ (0.52 deg)
CELL = 0.02            # deg: the repair pass's edge grid
DUCK_TMP = "C:/tmp/rb_duck"
TILE_EDGES = 2000     # RG: a GDScript tile decode costs ~5 us per edge; above this a tile misses A11's 15 ms
SUB_SPAN_TOY = 5000.0  # v2 sub-tiles: every point of an edge stays this close to its node a's sub-tile
SUB_EDGE_TOY = 4000.0  # so sub-tile edges are cut to this length


def ecef(lon, lat):
    lo = np.radians(lon)
    la = np.radians(lat)
    c = np.cos(la)
    return np.stack([c * np.cos(lo), c * np.sin(lo), np.sin(la)], axis=-1) * R_REAL


def step_m(xy):
    """Real m of each step of an (n, 2) lon, lat polyline."""
    d = np.diff(xy, axis=0)
    dlon = (d[:, 0] + 540.0) % 360.0 - 180.0
    c = np.cos(np.radians((xy[1:, 1] + xy[:-1, 1]) * 0.5))
    return np.hypot(dlon * c, d[:, 1]) * M_DEG


def poly_len(xy):
    return float(step_m(xy).sum()) if len(xy) > 1 else 0.0


# ---------------------------------------------------------------- load

class Src:
    pass


def cache_files():
    import glob
    return sorted(glob.glob(os.environ.get("RB_PARTS") or os.path.join(CACHE, "part_*.parquet")))


def scope_sql(towns, boxes):
    """A coarse scope filter for DuckDB (scope() decides exactly): motorway, trunk and ferries everywhere;
    primary in quarter-degree cells near a 100k+ town; anything inside a roads.bin box."""
    cells = set()
    pad = PRIMARY_KM / KM_DEG + 0.3
    for t in towns:
        if t["pop"] < BIG_POP:
            continue
        padx = pad / max(0.05, math.cos(math.radians(t["lat"])))
        for cy in range(int(math.floor((t["lat"] - pad) * 4)), int(math.floor((t["lat"] + pad) * 4)) + 1):
            for cx in range(int(math.floor((t["lon"] - padx) * 4)), int(math.floor((t["lon"] + padx) * 4)) + 1):
                cells.add((cy, (cx + 720) % 1440 - 720))
    box = " OR ".join("(x0 < %r AND x1 > %r AND y0 < %r AND y1 > %r)" % (b["box"][2], b["box"][0], b["box"][3],
                                                                        b["box"][1]) for b in boxes)
    return sorted(cells), "(class IN ('motorway', 'trunk') OR subtype = 'water' OR (class = 'primary' AND "         "(floor(y0 * 4)::INT * 2000 + floor(x0 * 4)::INT) IN (SELECT c FROM pcells)) OR %s)" % (box or "FALSE")


def load_source(only, boxes, towns):
    import duckdb
    import shapely
    parts = cache_files()
    if not parts:
        raise SystemExit("no cached parts: run with fetch first")
    log("load: %d cached parts" % len(parts))
    os.makedirs(DUCK_TMP, exist_ok=True)
    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial;")
    con.execute("SET memory_limit='4GB'; SET temp_directory='%s'; SET preserve_insertion_order=true;" % DUCK_TMP)
    lst = "[" + ", ".join(_sql_str(p.replace("\\", "/")) for p in parts) + "]"
    cells, cond = scope_sql(towns, boxes)
    con.execute("CREATE TEMP TABLE pcells AS SELECT unnest(?::INTEGER[]) AS c", [[cy * 2000 + cx for cy, cx in cells]])
    if only:
        s, w, n, e = only
        cond += " AND x0 < %r AND x1 > %r AND y0 < %r AND y1 > %r" % (e, w, n, s)
    con.execute(f"""CREATE TEMP TABLE seg AS
        SELECT (row_number() OVER (ORDER BY id) - 1)::BIGINT AS si, * FROM
          (SELECT * FROM read_parquet({lst}) WHERE {cond})""")
    r = con.execute("SELECT subtype, class, subclass, name, ref, ow, wkb FROM seg ORDER BY si").fetchnumpy()
    c = con.execute("""SELECT si, (dense_rank() OVER (ORDER BY cid) - 1)::BIGINT AS ci, pos FROM
        (SELECT si, c.connector_id AS cid, c."at" AS pos FROM (SELECT si, unnest(connectors) AS c FROM seg))
        ORDER BY si, pos, ci""").fetchnumpy()
    fl = con.execute("SELECT si, road_flags, level_rules FROM seg WHERE len(road_flags) > 0 ORDER BY si").fetchall()
    con.close()
    if len(r["subtype"]) == 0:
        return None
    o = Src()
    o.n = len(r["subtype"])
    o.subtype = np.asarray(r["subtype"], dtype=object)
    o.cls_s = np.asarray(r["class"], dtype=object)
    o.subclass = np.asarray(r["subclass"], dtype=object)
    o.name = np.asarray(r["name"], dtype=object)
    o.ref = np.asarray(r["ref"], dtype=object)
    o.ow = np.asarray(r["ow"], dtype=np.int8)
    geoms = shapely.from_wkb(np.array([bytes(w) for w in r["wkb"]], dtype=object))
    del r
    coords, gi = shapely.get_coordinates(geoms, return_index=True)
    del geoms
    o.xy = np.ascontiguousarray(coords, dtype=np.float64)
    o.off = np.searchsorted(gi, np.arange(o.n + 1)).astype(np.int64)
    o.csi = np.asarray(c["si"], dtype=np.int64)
    o.cci = np.asarray(c["ci"], dtype=np.int64)
    o.cat = np.asarray(c["pos"], dtype=np.float64)
    o.nconn = int(o.cci.max()) + 1 if len(o.cci) else 0
    o.flags = {}
    for si, rf, lv in fl:
        recs = []
        for f in rf or []:
            vals = f.get("values") or []
            b = f.get("between") or [0.0, 1.0]
            kind = ST_BRIDGE if "is_bridge" in vals else ST_TUNNEL
            level = 0
            for L in lv or []:
                lb = L.get("between") or [0.0, 1.0]
                if lb[0] < b[1] and lb[1] > b[0] and L.get("value") is not None:
                    level = int(L["value"])
            if level == 0:
                level = 1 if kind == ST_BRIDGE else -1
            recs.append((kind, max(-128, min(127, level)), float(b[0]), float(b[1])))
        if recs:
            o.flags[int(si)] = recs
    log("  %d segments, %d vertices, %d connector refs, %d with bridge / tunnel ranges" % (
        o.n, len(o.xy), len(o.cci), len(o.flags)))
    return o


def iconic_matcher(names):
    import re
    exact = set()
    pats = []
    for nm in names:
        if "%" in nm:
            pats.append(re.compile("^" + ".*".join(re.escape(p) for p in nm.lower().split("%")) + "$"))
        else:
            exact.add(nm.lower())

    def f(s):
        if not s:
            return False
        s = s.lower()
        return s in exact or any(p.match(s) for p in pats)
    return f


def scope(src, towns, boxes):
    """Which source segments the graph keeps, and their graph class."""
    n = src.n
    road = src.subtype == "road"
    cls = np.full(n, -1, dtype=np.int8)
    for k, v in CLS_OF.items():
        cls[road & (src.cls_s == k)] = v
    link = src.subclass == "link"
    mid = src.xy[(src.off[:-1] + src.off[1:] - 1) // 2]
    inbox = np.zeros(n, dtype=bool)
    icon = np.zeros(n, dtype=bool)
    for b in boxes:
        lon0, lat0, lon1, lat1 = b["box"]
        idx = np.flatnonzero((mid[:, 0] > lon0) & (mid[:, 0] < lon1) & (mid[:, 1] > lat0) & (mid[:, 1] < lat1))
        inbox[idx] = True
        if b["icon"]:
            f = iconic_matcher(b["icon"])
            for i in idx:
                if road[i] and not icon[i] and f(src.name[i]):
                    icon[i] = True
    from scipy.spatial import cKDTree
    big = [t for t in towns if t["pop"] >= BIG_POP]
    tree = cKDTree(ecef(np.array([t["lon"] for t in big]), np.array([t["lat"] for t in big])))
    d, _ = tree.query(ecef(mid[:, 0], mid[:, 1]), distance_upper_bound=PRIMARY_KM * 1000.0)
    near = np.isfinite(d)
    keep = road & (((cls >= 0) & (cls <= TRUNK)) | ((cls == PRIMARY) & (near | inbox)) |
                   ((cls == SECONDARY) & inbox) | icon)
    gcls = cls.copy()
    gcls[icon & ((cls < 0) | (cls >= PRIMARY))] = ICONIC
    water = np.flatnonzero(src.subtype == "water")
    ends = []
    for i in water:
        ends.append(src.xy[src.off[i]])
        ends.append(src.xy[src.off[i + 1] - 1])
    ferry = np.array(ends) if ends else np.zeros((0, 2))
    log("scope: %d kept (motorway %d, trunk %d, primary %d, secondary %d, iconic %d; links %d), %d ferry ends" % (
        int(keep.sum()), int((keep & (gcls == 0)).sum()), int((keep & (gcls == 1)).sum()),
        int((keep & (gcls == 2)).sum()), int((keep & (gcls == 3)).sum()), int((keep & (gcls == 4)).sum()),
        int((keep & link).sum()), len(ferry)))
    return keep, gcls, link & keep, ferry


# ---------------------------------------------------------------- topology

class Pieces:
    pass


def topology(src, keep):
    """Split every kept segment at its connectors shared with another kept segment (and at both ends)."""
    n = src.n
    xy = src.xy
    off = src.off
    nv = len(xy)
    seg_of = np.repeat(np.arange(n), np.diff(off))
    st = np.zeros(nv)
    if nv > 1:
        sm = step_m(xy)
        same = seg_of[1:] == seg_of[:-1]
        st[1:] = np.where(same, sm, 0.0)
    cum = np.cumsum(st)
    base = cum[off[:-1]]
    tot = cum[np.maximum(off[1:] - 1, off[:-1])] - base
    frac = (cum - base[seg_of]) / np.maximum(tot[seg_of], 1e-9)
    frac = np.clip(frac, 0.0, 1.0)
    G = seg_of + frac * 0.999
    # connectors of kept segments
    m = keep[src.csi]
    csi, cci, cat = src.csi[m], src.cci[m], src.cat[m]
    key = csi * max(1, src.nconn) + cci
    _, first = np.unique(key, return_index=True)
    first.sort()
    csi, cci, cat = csi[first], cci[first], cat[first]
    count = np.bincount(cci, minlength=src.nconn)
    tgt = csi + np.clip(cat, 0.0, 1.0) * 0.999
    j = np.searchsorted(G, tgt)
    lo = off[csi]
    hi = off[csi + 1] - 1
    j = np.clip(j, lo, hi)
    jm = np.clip(j - 1, lo, hi)
    use_m = np.abs(G[jm] - tgt) < np.abs(G[j] - tgt)
    v = np.where(use_m, jm, j)
    is_end = (v == lo) | (v == hi)
    nodec = (count[cci] >= 2) | is_end
    rs, rv, rn = csi[nodec], v[nodec], cci[nodec]
    # synthetic end nodes where a kept segment has no connector at an end
    K = np.flatnonzero(keep)
    has0 = np.zeros(n, dtype=bool)
    has1 = np.zeros(n, dtype=bool)
    has0[rs[rv == off[rs]]] = True
    has1[rs[rv == off[rs + 1] - 1]] = True
    s0 = K[~has0[K]]
    s1 = K[~has1[K]]
    NC = src.nconn
    rs = np.concatenate([rs, s0, s1])
    rv = np.concatenate([rv, off[s0], off[s1 + 1] - 1])
    rn = np.concatenate([rn, NC + 2 * s0, NC + 2 * s1 + 1])
    order = np.lexsort((rn, rv, rs))
    rs, rv, rn = rs[order], rv[order], rn[order]
    # two connectors on one vertex: one node (alias the later onto the first)
    dup = np.flatnonzero((rs[1:] == rs[:-1]) & (rv[1:] == rv[:-1])) + 1
    parent = {}

    def root(a):
        while a in parent:
            a = parent[a]
        return a
    for d in dup:
        a, b = root(int(rn[d - 1])), root(int(rn[d]))
        if a != b:
            parent[max(a, b)] = min(a, b)
    if parent:
        rn = np.array([root(int(x)) for x in rn], dtype=np.int64)
    keepr = np.ones(len(rs), dtype=bool)
    keepr[dup] = False
    rs, rv, rn = rs[keepr], rv[keepr], rn[keepr]
    # pieces between consecutive node vertices of a segment
    same = (rs[1:] == rs[:-1]) & (rv[1:] > rv[:-1])
    pi = np.flatnonzero(same)
    P = Pieces()
    P.si = rs[pi]
    P.v0 = rv[pi]
    P.v1 = rv[pi + 1]
    P.a = rn[pi]
    P.b = rn[pi + 1]
    # node coordinates: the vertex of the first segment that has the node
    uniq, fi = np.unique(rn, return_index=True)
    P.node_ids = uniq
    P.node_xy = xy[rv[fi]].copy()
    # antimeridian: drop pieces with a step across it
    jump = np.zeros(nv, dtype=bool)
    if nv > 1:
        jump[1:] = np.abs(np.diff(xy[:, 0])) > 180.0
    cj = np.cumsum(jump)
    bad = (cj[P.v1] - cj[P.v0]) > 0
    if bad.any():
        for k in ("si", "v0", "v1", "a", "b"):
            setattr(P, k, getattr(P, k)[~bad])
    P.frac = frac
    P.n = len(P.si)
    log("topology: %d pieces, %d nodes (%d antimeridian pieces dropped)" % (P.n, len(uniq), int(bad.sum())))
    return P


def node_index(P, ids):
    return np.searchsorted(P.node_ids, ids)


def piece_xy(src, P, i, xy=None):
    xy = src.xy if xy is None else xy
    a = xy[P.v0[i]:P.v1[i] + 1].copy()
    a[0] = P.node_xy[node_index(P, P.a[i])]
    a[-1] = P.node_xy[node_index(P, P.b[i])]
    return a


# ---------------------------------------------------------------- dual carriageways

def local_dir(p, q, lat):
    d = q - p
    dx = ((d[:, 0] + 540.0) % 360.0 - 180.0) * np.cos(np.radians(lat))
    dy = d[:, 1]
    L = np.hypot(dx, dy)
    L[L == 0] = 1.0
    return np.stack([dx / L, dy / L], axis=1)


def dual_merge(src, P, gcls, link):
    """Pair antiparallel one-way carriageways of a class, drop one of each pair, move the kept one onto the
    mean line. Returns (state per piece: 0 plain, 1 kept dual, 2 dropped; pair list for the road keys)."""
    from scipy.spatial import cKDTree
    pc = gcls[P.si]
    ow = src.ow[P.si]
    cand = np.flatnonzero((ow != 0) & ~link[P.si] & (pc >= 0) & (pc <= ICONIC))
    state = np.zeros(P.n, dtype=np.int8)
    if len(cand) == 0:
        return state, []
    # vertex steps of candidate pieces (with node coordinates at the ends)
    xy = src.xy
    lens = P.v1[cand] - P.v0[cand]
    tot = int(lens.sum())
    pid = np.repeat(cand, lens)
    start = np.repeat(P.v0[cand], lens) + (np.arange(tot) - np.repeat(np.cumsum(lens) - lens, lens))
    p0 = xy[start].copy()
    p1 = xy[start + 1].copy()
    isf = start == np.repeat(P.v0[cand], lens)
    isl = start + 1 == np.repeat(P.v1[cand], lens)
    p0[isf] = P.node_xy[node_index(P, P.a[pid[isf]])]
    p1[isl] = P.node_xy[node_index(P, P.b[pid[isl]])]
    sgn = np.where(ow[pid] < 0, -1.0, 1.0)
    L = step_m(np.stack([p0, p1], axis=1).reshape(-1, 2))[::2] if tot else np.zeros(0)
    k = np.maximum(1, np.ceil(L / DUAL_STEP).astype(np.int64))
    sp = np.repeat(np.arange(tot), k)
    t = (np.arange(int(k.sum())) - np.repeat(np.cumsum(k) - k, k) + 0.5) / np.repeat(k, k)
    d01 = p1 - p0
    d01[:, 0] = (d01[:, 0] + 540.0) % 360.0 - 180.0
    spt = p0[sp] + d01[sp] * t[:, None]
    sdir = local_dir(p0, p1, (p0[:, 1] + p1[:, 1]) * 0.5)[sp] * sgn[sp, None]
    spid = pid[sp]
    scls = pc[spid]
    X = ecef(spt[:, 0], spt[:, 1])
    tree = cKDTree(X)
    KQ = 24  # own-piece samples fill the first ~2 r / DUAL_STEP neighbours
    dd, ii = tree.query(X, k=KQ, distance_upper_bound=DUAL_R_FAST)
    lim = np.where(scls <= TRUNK, DUAL_R_FAST, DUAL_R)
    N = len(X)
    partner = np.full(N, -1, dtype=np.int64)
    for col in range(1, KQ):
        j = ii[:, col]
        ok = (j < N) & (partner < 0) & (dd[:, col] <= lim)
        jj = np.where(ok, j, 0)
        ok &= spid[jj] != spid
        ca, cb = scls, scls[jj]
        ok &= (ca == cb) | ((ca <= TRUNK) & (cb <= TRUNK))
        ok &= (sdir[:, 0] * sdir[jj, 0] + sdir[:, 1] * sdir[jj, 1]) < DUAL_DOT
        partner[ok] = jj[ok]
    nsamp = np.bincount(spid, minlength=P.n)
    has = partner >= 0
    pp = spid[has]
    qq = spid[partner[has]]
    M = np.int64(P.n)
    code = pp * M + qq
    uc, inv, cnt = np.unique(code, return_inverse=True, return_counts=True)
    dsum = np.zeros((len(uc), 2))
    np.add.at(dsum, inv, sdir[has])
    up = uc // M
    uq = uc % M
    pair = {}
    for a, b, c, dx, dy in zip(up.tolist(), uq.tolist(), cnt.tolist(), dsum[:, 0].tolist(), dsum[:, 1].tolist()):
        lo_, hi_ = (a, b) if a < b else (b, a)
        rec = pair.setdefault((lo_, hi_), [0, 0.0, 0.0])
        rec[0] += c
        if a == lo_:
            rec[1] += dx
            rec[2] += dy
        else:
            rec[1] -= dx
            rec[2] -= dy
    pairs = sorted(((v[0], a, b, v[1], v[2]) for (a, b), v in pair.items() if v[0] >= 3),
                   key=lambda r: (-r[0], r[1], r[2]))
    for c, a, b, dx, dy in pairs:
        sa, sb = state[a], state[b]
        if sa == 2 or sb == 2:
            if sa == 0:
                state[a] = 1
            if sb == 0:
                state[b] = 1
        elif sa == 1 and sb == 0:
            state[b] = 2
        elif sb == 1 and sa == 0:
            state[a] = 2
        elif sa == 0 and sb == 0:
            east = dx > 0 or (dx == 0 and dy > 0)
            state[a], state[b] = (1, 2) if east else (2, 1)
    # a dropped carriageway must run beside kept ones for most of its length
    beside = np.zeros(P.n)
    for (a, b), v in pair.items():
        pass
    cnt_ab = {}
    for a, b, c in zip(up.tolist(), uq.tolist(), cnt.tolist()):
        cnt_ab[(a, b)] = c
    for (a, b), c in cnt_ab.items():
        if state[b] == 1:
            beside[a] += c
    drop = np.flatnonzero(state == 2)
    undo = drop[beside[drop] < DUAL_MIN * nsamp[drop]]
    state[undo] = 1
    # kept carriageways that have no dropped partner left are plain one-way roads again
    kept_with = np.zeros(P.n, dtype=bool)
    for (a, b), c in cnt_ab.items():
        if state[a] == 1 and state[b] == 2 and c >= 3:
            kept_with[a] = True
    state[(state == 1) & ~kept_with] = 0
    # move kept carriageways onto the mean line: half way toward the nearest dropped antiparallel sample
    dsel = np.flatnonzero(state[spid] == 2)
    if len(dsel):
        dtree = cKDTree(X[dsel])
        global _SHIFT_R
        _SHIFT_R = DUAL_R_FAST
        kp = np.flatnonzero(state == 1)
        # interior vertices of kept pieces
        lens = P.v1[kp] - P.v0[kp] - 1
        lens = np.maximum(lens, 0)
        vid = np.repeat(P.v0[kp], lens) + 1 + (np.arange(int(lens.sum())) - np.repeat(np.cumsum(lens) - lens, lens))
        vpid = np.repeat(kp, lens)
        newxy = src.xy.copy()
        if len(vid):
            vdir = local_dir(src.xy[vid - 1], src.xy[vid + 1], src.xy[vid, 1]) * np.where(
                ow[vpid] < 0, -1.0, 1.0)[:, None]
            sh = _half_shift(dtree, X, dsel, spt, sdir, src.xy[vid], vdir)
            newxy[vid] += sh
        # nodes of kept pieces
        na = np.concatenate([P.a[kp], P.b[kp]])
        nxt = np.concatenate([src.xy[np.minimum(P.v0[kp] + 1, P.v1[kp])], src.xy[np.maximum(P.v1[kp] - 1, P.v0[kp])]])
        sgnn = np.concatenate([np.where(ow[kp] < 0, -1.0, 1.0), -np.where(ow[kp] < 0, -1.0, 1.0)])
        ni = node_index(P, na)
        nxy = P.node_xy[ni]
        ndir = local_dir(nxy, nxt, nxy[:, 1]) * sgnn[:, None]
        sh = _half_shift(dtree, X, dsel, spt, sdir, nxy, ndir)
        uni, fi = np.unique(ni, return_index=True)
        P.node_xy[uni] += sh[fi]
        src.xy = newxy
    log("dual: %d one-way pieces, %d pairs, %d kept as merged centre lines, %d carriageways merged away" % (
        len(cand), len(pairs), int((state == 1).sum()), int((state == 2).sum())))
    return state, [(a, b) for c, a, b, _, _ in pairs]


_SHIFT_R = DUAL_R_FAST


def _half_shift(dtree, X, dsel, spt, sdir, pts, dirs):
    q = ecef(pts[:, 0], pts[:, 1])
    dd, jj = dtree.query(q, k=8, distance_upper_bound=_SHIFT_R)
    out = np.zeros((len(pts), 2))
    done = np.zeros(len(pts), dtype=bool)
    nd = len(dsel)
    for col in range(8):
        j = jj[:, col]
        ok = (j < nd) & ~done
        js = dsel[np.where(ok, j, 0)]
        ok &= (dirs[:, 0] * sdir[js, 0] + dirs[:, 1] * sdir[js, 1]) < DUAL_DOT
        d = spt[js] - pts
        d[:, 0] = (d[:, 0] + 540.0) % 360.0 - 180.0
        out[ok] = 0.5 * d[ok]
        done |= ok
    return out


# ---------------------------------------------------------------- road keys (which pieces are one road)

def road_keys(src, P, gcls, link, pairs, names):
    parent = np.arange(P.n)

    def find(a):
        r = a
        while parent[r] != r:
            r = parent[r]
        while parent[a] != r:
            parent[a], a = r, parent[a]
        return r

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb
    nl = np.flatnonzero(~link[P.si])
    ends = np.concatenate([P.a[nl], P.b[nl]])
    who = np.concatenate([nl, nl])
    o = np.lexsort((who, ends))
    ends, who = ends[o], who[o]
    bounds = np.flatnonzero(np.diff(ends)) + 1
    groups = np.split(who, bounds)
    pc = gcls[P.si]
    pname = names[P.si]
    for g in groups:
        if len(g) == 2:
            if pc[g[0]] == pc[g[1]]:
                union(int(g[0]), int(g[1]))
        if len(g) >= 2:
            byn = {}
            for x in g.tolist():
                nm = pname[x]
                if nm:
                    if nm in byn:
                        union(byn[nm], x)
                    else:
                        byn[nm] = x
    for a, b in pairs:
        union(a, b)
    return np.array([find(i) for i in range(P.n)], dtype=np.int64)


# ---------------------------------------------------------------- the mutable graph

class Edge:
    __slots__ = ("a", "b", "xy", "cls", "flags", "lanes", "name", "structs", "len", "key")

    def __init__(self, a, b, xy, cls, flags, lanes, name, structs, key):
        self.a, self.b, self.xy = a, b, xy
        self.cls, self.flags, self.lanes, self.name = cls, flags, lanes, name
        self.structs = structs  # [kind, level, s0, s1] real m from a
        self.key = key
        self.len = poly_len(xy)


class Graph:
    def __init__(self):
        self.nodes = {}   # id -> [lon, lat]
        self.adj = {}     # id -> list of edge ids
        self.edges = {}
        self.sink = {}    # node -> (kind, name, ref, radius)
        self.grid = {}
        self.ecells = {}
        self.next_node = 0
        self.next_edge = 0

    # nodes and edges
    def add_node(self, xy, nid=None):
        if nid is None:
            nid = self.next_node
        self.next_node = max(self.next_node, nid + 1)
        self.nodes[nid] = [float(xy[0]), float(xy[1])]
        self.adj[nid] = []
        return nid

    def add_edge(self, a, b, xy, cls, flags, lanes, name, structs, key):
        xy = np.array(xy, dtype=np.float64)
        xy[0] = self.nodes[a]
        xy[-1] = self.nodes[b]
        e = Edge(a, b, xy, cls, flags, lanes, name, structs, key)
        eid = self.next_edge
        self.next_edge += 1
        self.edges[eid] = e
        self.adj[a].append(eid)
        self.adj[b].append(eid)
        self._index(eid, e)
        return eid

    def remove_edge(self, eid):
        e = self.edges.pop(eid)
        self.adj[e.a].remove(eid)
        self.adj[e.b].remove(eid)
        for c in self.ecells.pop(eid, ()):
            s = self.grid.get(c)
            if s is not None:
                s.discard(eid)
        return e

    def remove_node_if_free(self, n):
        if n in self.adj and not self.adj[n]:
            del self.adj[n]
            del self.nodes[n]
            self.sink.pop(n, None)

    def degree(self, n):
        return len(self.adj[n])

    def other(self, eid, n):
        e = self.edges[eid]
        return e.b if e.a == n else e.a

    # spatial index
    def _cells(self, xy):
        lo = xy[:, 0]
        la = xy[:, 1]
        if len(xy) > 1:
            st = np.hypot(np.diff(lo), np.diff(la))
            k = np.maximum(1, np.ceil(st / (CELL * 0.5)).astype(np.int64))
            if k.max() > 1:
                t = (np.arange(int(k.sum())) - np.repeat(np.cumsum(k) - k, k)) / np.repeat(k, k)
                i = np.repeat(np.arange(len(st)), k)
                lo = np.concatenate([lo[i] + np.diff(xy[:, 0])[i] * t, xy[-1:, 0]])
                la = np.concatenate([la[i] + np.diff(xy[:, 1])[i] * t, xy[-1:, 1]])
        cx = np.floor(lo / CELL).astype(np.int64)
        cy = np.floor(la / CELL).astype(np.int64)
        return set(zip(cx.tolist(), cy.tolist()))

    def _index(self, eid, e):
        cs = self._cells(e.xy)
        self.ecells[eid] = cs
        for c in cs:
            self.grid.setdefault(c, set()).add(eid)

    def edges_in(self, lon, lat, r_m):
        dla = r_m / M_DEG
        dlo = dla / max(0.05, math.cos(math.radians(lat)))
        out = set()
        for cx in range(int(math.floor((lon - dlo) / CELL)), int(math.floor((lon + dlo) / CELL)) + 1):
            for cy in range(int(math.floor((lat - dla) / CELL)), int(math.floor((lat + dla) / CELL)) + 1):
                s = self.grid.get((cx, cy))
                if s:
                    out |= s
        return sorted(out)

    def nearest_on(self, lon, lat, r_m, skip=()):
        """(dist real m, eid, seg index, t, point) of the nearest point on an edge within r_m, or None."""
        best = None
        c = math.cos(math.radians(lat))
        for eid in self.edges_in(lon, lat, r_m):
            if eid in skip:
                continue
            xy = self.edges[eid].xy
            px = ((xy[:, 0] - lon + 540.0) % 360.0 - 180.0) * c * M_DEG
            py = (xy[:, 1] - lat) * M_DEG
            ax, ay = px[:-1], py[:-1]
            dx, dy = px[1:] - ax, py[1:] - ay
            L2 = dx * dx + dy * dy
            L2[L2 == 0] = 1e-12
            t = np.clip(-(ax * dx + ay * dy) / L2, 0.0, 1.0)
            qx, qy = ax + t * dx, ay + t * dy
            d2 = qx * qx + qy * qy
            i = int(np.argmin(d2))
            d = math.sqrt(float(d2[i]))
            if d <= r_m and (best is None or d < best[0]):
                ti = float(t[i])
                pt = xy[i] + (xy[i + 1] - xy[i]) * ti
                best = (d, eid, i, ti, pt)
        return best

    def split(self, eid, i, t, pt):
        """Cut edge eid at pt (on its step i, fraction t); returns the node there."""
        e = self.edges[eid]
        xy = e.xy
        sm = step_m(xy)
        s_at = float(sm[:i].sum() + sm[i] * t) if len(sm) else 0.0
        if s_at < 1.0:
            return e.a
        if e.len - s_at < 1.0:
            return e.b
        self.remove_edge(eid)
        n = self.add_node(pt)
        xa = np.vstack([xy[:i + 1], [pt]])
        xb = np.vstack([[pt], xy[i + 1:]])
        sa, sb = [], []
        for kind, lev, s0, s1 in e.structs:
            if s0 < s_at:
                sa.append([kind, lev, s0, min(s1, s_at)])
            if s1 > s_at:
                sb.append([kind, lev, max(0.0, s0 - s_at), s1 - s_at])
        fa = e.flags & ~(F_BRIDGE | F_TUNNEL)
        self.add_edge(e.a, n, xa, e.cls, fa | _sflags(sa), e.lanes, e.name, sa, e.key)
        self.add_edge(n, e.b, xb, e.cls, fa | _sflags(sb), e.lanes, e.name, sb, e.key)
        return n

    def move_node(self, n, xy):
        self.nodes[n] = [float(xy[0]), float(xy[1])]
        for eid in list(self.adj[n]):
            e = self.edges[eid]
            if e.a == n:
                e.xy[0] = self.nodes[n]
            if e.b == n:
                e.xy[-1] = self.nodes[n]
            e.len = poly_len(e.xy)
            for c in self.ecells.pop(eid, ()):
                self.grid[c].discard(eid)
            self._index(eid, e)

    def merge_into(self, n, m):
        """Node n disappears into node m (its edges now end at m)."""
        if n == m:
            return
        for eid in list(self.adj[n]):
            e = self.edges[eid]
            if e.a == n:
                e.a = m
                e.xy[0] = self.nodes[m]
            if e.b == n:
                e.b = m
                e.xy[-1] = self.nodes[m]
            e.len = poly_len(e.xy)
            self.adj[m].append(eid)
            for c in self.ecells.pop(eid, ()):
                self.grid[c].discard(eid)
            self._index(eid, e)
        self.adj[n] = []
        if n in self.sink and m not in self.sink:
            self.sink[m] = self.sink[n]
        self.remove_node_if_free(n)
        # an edge from m to m of almost no length is noise
        for eid in list(self.adj[m]):
            e = self.edges.get(eid)
            if e is not None and e.a == e.b == m and e.len < 50.0:
                self.remove_edge(eid)

    def chain(self, n):
        """From end n back along degree-2 nodes: (edge ids, root node, real m)."""
        out = []
        total = 0.0
        cur = n
        prev = None
        seen = {n}
        while True:
            nxt = [x for x in self.adj[cur] if x != prev]
            if cur != n and (len(self.adj[cur]) != 2 or cur in self.sink):
                return out, cur, total
            if not nxt:
                return out, cur, total
            eid = nxt[0]
            out.append(eid)
            total += self.edges[eid].len
            prev = eid
            cur = self.other(eid, cur)
            if cur in seen:
                return out, cur, total
            seen.add(cur)


def _sflags(structs):
    f = 0
    for s in structs:
        f |= F_BRIDGE if s[0] == ST_BRIDGE else F_TUNNEL
    return f


def piece_structs(src, P, i, length):
    recs = src.flags.get(int(P.si[i]))
    if not recs:
        return []
    f0 = float(P.frac[P.v0[i]])
    f1 = float(P.frac[P.v1[i]])
    span = max(f1 - f0, 1e-9)
    out = []
    for kind, level, b0, b1 in recs:
        lo = max(b0, f0)
        hi = min(b1, f1)
        if hi > lo:
            out.append([kind, level, (lo - f0) / span * length, (hi - f0) / span * length])
    return out


def build_graph(src, P, gcls, link, state, keys, names):
    g = Graph()
    for k, nid in enumerate(P.node_ids.tolist()):
        pass
    used = np.flatnonzero(state != 2)
    nodes_used = np.unique(np.concatenate([P.a[used], P.b[used]]))
    ni = node_index(P, nodes_used)
    for nid, xy in zip(nodes_used.tolist(), P.node_xy[ni]):
        g.add_node(xy, nid)
    ow = src.ow[P.si]
    is_link = link[P.si]
    for i in used.tolist():
        xy = piece_xy(src, P, i)
        c = int(gcls[P.si[i]])
        a, b = int(P.a[i]), int(P.b[i])
        flags = 0
        if state[i] == 1:
            flags |= F_DUAL
        elif ow[i] != 0:
            flags |= F_ONEWAY
            if ow[i] < 0:
                xy = xy[::-1].copy()
                a, b = b, a
        if is_link[i]:
            flags |= F_SLIP
        st = piece_structs(src, P, i, poly_len(xy))
        if (flags & F_ONEWAY) and ow[i] < 0:
            L = poly_len(xy)
            st = [[k_, lv, L - s1, L - s0] for k_, lv, s0, s1 in st]
        lanes = LANES_DUAL.get(c, 1) if flags & F_DUAL else LANES_ONE.get(c, 1)
        g.add_edge(a, b, xy, c, flags | _sflags(st), lanes, names[P.si[i]] or "", st, int(keys[i]))
    log("graph: %d nodes, %d edges" % (len(g.nodes), len(g.edges)))
    return g


# ---------------------------------------------------------------- slip roads

def collapse_links(g, P, link, keys):
    """Each cluster of slip-road edges becomes a junction where its main roads cross, or one kept LINK path
    between two main roads that do not cross; the rest of the cluster goes."""
    # main-road keys at every source node (before the dual merge dropped carriageways)
    nl = np.flatnonzero(~link[P.si])
    at = {}
    for i in nl.tolist():
        k = int(keys[i])
        for n in (int(P.a[i]), int(P.b[i])):
            s = at.get(n)
            if s is None:
                at[n] = {k}
            else:
                s.add(k)
    link_edges = sorted(eid for eid, e in g.edges.items() if e.flags & F_SLIP)
    junctions = []
    seen = set()
    kept = set()
    n_junc = n_path = n_none = 0
    for e0 in link_edges:
        if e0 in seen or e0 not in g.edges:
            continue
        comp = []
        stack = [e0]
        seen.add(e0)
        while stack:
            eid = stack.pop()
            comp.append(eid)
            e = g.edges[eid]
            for n in (e.a, e.b):
                for x in g.adj[n]:
                    if x not in seen and g.edges[x].flags & F_SLIP:
                        seen.add(x)
                        stack.append(x)
        comp.sort()
        cnodes = sorted({g.edges[x].a for x in comp} | {g.edges[x].b for x in comp})
        attach = {}
        for n in cnodes:
            for k in sorted(at.get(n, ())):
                attach.setdefault(k, []).append(n)
        if len(attach) < 2:
            n_none += 1
            continue
        pts = np.array([g.nodes[n] for n in cnodes])
        clon, clat = float(pts[:, 0].mean()), float(pts[:, 1].mean())
        ext = float(np.max(np.hypot((pts[:, 0] - clon) * math.cos(math.radians(clat)), pts[:, 1] - clat))) * M_DEG
        reach = ext + JUNC_R
        ks = sorted(attach)
        joined = set()
        for x in range(len(ks)):
            for y in range(x + 1, len(ks)):
                if _connect_roads(g, ks[x], ks[y], clon, clat, reach, junctions):
                    joined.add(ks[x])
                    joined.add(ks[y])
                    n_junc += 1
        # roads of the cluster that do not cross another: keep the shortest slip path to one that does
        lonely = [k for k in ks if k not in joined]
        for k in lonely:
            src_nodes = attach[k]
            dst = set()
            for k2 in ks:
                if k2 != k:
                    dst.update(attach[k2])
            path = _shortest(g, src_nodes, dst, set(comp))
            if path:
                for eid in path:
                    kept.add(eid)
                n_path += 1
    # turn the kept paths into two-way LINK edges, drop every other slip edge
    for eid in sorted(kept):
        if eid in g.edges:
            e = g.edges[eid]
            e.cls = LINK
            e.flags = e.flags & (F_BRIDGE | F_TUNNEL)
            e.lanes = 1
    drop = sorted(eid for eid, e in g.edges.items() if e.flags & F_SLIP)
    for eid in drop:
        e = g.remove_edge(eid)
        g.remove_node_if_free(e.a)
        g.remove_node_if_free(e.b)
    for n in junctions:
        for eid in g.adj.get(n, ()):
            if g.edges[eid].cls != LINK:
                g.edges[eid].flags |= F_LINK
    log("links: %d clusters without two roads, %d junctions made, %d slip paths kept, %d slip edges removed" % (
        n_none, n_junc, n_path, len(drop)))


def _connect_roads(g, k1, k2, lon, lat, reach, junctions):
    """Make (or find) the junction where roads k1 and k2 cross near (lon, lat). True when joined."""
    from shapely.geometry import LineString
    cand = g.edges_in(lon, lat, reach)
    e1 = [x for x in cand if g.edges[x].key == k1 and not g.edges[x].flags & F_SLIP]
    e2 = [x for x in cand if g.edges[x].key == k2 and not g.edges[x].flags & F_SLIP]
    if not e1 or not e2:
        return False
    c = math.cos(math.radians(lat))
    # already meeting at a node near here
    n1 = {g.edges[x].a for x in e1} | {g.edges[x].b for x in e1}
    n2 = {g.edges[x].a for x in e2} | {g.edges[x].b for x in e2}
    for n in sorted(n1 & n2):
        p = g.nodes[n]
        if math.hypot((p[0] - lon) * c, p[1] - lat) * M_DEG < reach:
            return True
    best = None
    for x in e1:
        l1 = LineString(g.edges[x].xy)
        for y in e2:
            if x == y:
                continue
            inter = l1.intersection(LineString(g.edges[y].xy))
            if inter.is_empty:
                continue
            geoms = getattr(inter, "geoms", [inter])
            for q in geoms:
                if q.geom_type != "Point":
                    q = q.representative_point()
                d = math.hypot((q.x - lon) * c, q.y - lat) * M_DEG
                if d < reach and (best is None or (d, x, y) < best[:3]):
                    best = (d, x, y, q.x, q.y)
    if best is None:
        return False
    _, x, y, qx, qy = best
    n_a = _split_at(g, x, qx, qy)
    n_b = _split_at(g, y, qx, qy)
    if n_a is None or n_b is None:
        return False
    if n_a != n_b:
        g.merge_into(n_b, n_a)
    junctions.append(n_a)
    return True


def _split_at(g, eid, lon, lat):
    e = g.edges.get(eid)
    if e is None:
        return None
    # the nearest point on this edge
    c = math.cos(math.radians(lat))
    xy = e.xy
    px = ((xy[:, 0] - lon + 540.0) % 360.0 - 180.0) * c * M_DEG
    py = (xy[:, 1] - lat) * M_DEG
    ax, ay = px[:-1], py[:-1]
    dx, dy = px[1:] - ax, py[1:] - ay
    L2 = dx * dx + dy * dy
    L2[L2 == 0] = 1e-12
    t = np.clip(-(ax * dx + ay * dy) / L2, 0.0, 1.0)
    d2 = (ax + t * dx) ** 2 + (ay + t * dy) ** 2
    i = int(np.argmin(d2))
    hit = (i, float(t[i]))
    pt = xy[hit[0]] + (xy[hit[0] + 1] - xy[hit[0]]) * hit[1]
    return g.split(eid, hit[0], hit[1], pt)


def _shortest(g, srcs, dsts, allowed):
    import heapq
    dist = {}
    prev = {}
    h = []
    for s in sorted(set(srcs)):
        dist[s] = 0.0
        heapq.heappush(h, (0.0, s))
    while h:
        d, n = heapq.heappop(h)
        if d > dist.get(n, 1e30):
            continue
        if n in dsts and d > 0:
            path = []
            while n in prev:
                eid, n = prev[n]
                path.append(eid)
            return path
        for eid in sorted(g.adj[n]):
            if eid not in allowed:
                continue
            m = g.other(eid, n)
            nd = d + g.edges[eid].len
            if nd < dist.get(m, 1e30):
                dist[m] = nd
                prev[m] = (eid, n)
                heapq.heappush(h, (nd, m))
    return None


# ---------------------------------------------------------------- cleanup

def cleanup(g):
    """Collapse very short edges into a point (roundabouts, turning loops, merge slivers), drop short self loops
    and parallel duplicates between the same two nodes."""
    n_c = n_l = n_d = 0
    from scipy.spatial import cKDTree
    ids = sorted(g.nodes)
    if len(ids) > 1:
        xy = np.array([g.nodes[n] for n in ids])
        pairs = cKDTree(ecef(xy[:, 0], xy[:, 1])).query_pairs(NODE_MERGE, output_type="ndarray")
        par = {}

        def root(x):
            while x in par:
                x = par[x]
            return x
        for i, j in sorted(map(tuple, pairs.tolist())):
            a, b = root(ids[i]), root(ids[j])
            if a != b:
                par[max(a, b)] = min(a, b)
        groups = {}
        for n in sorted(par):
            groups.setdefault(root(n), []).append(n)
        for r, members in groups.items():
            allm = [r] + members
            # the place keeps a sink, else its busiest node
            tgt = min(allm, key=lambda n: (n not in g.sink, -len(g.adj[n]), n))
            for n in allm:
                if n != tgt and n in g.nodes:
                    g.merge_into(n, tgt)
                    n_c += 1
    for eid in sorted(g.edges):
        e = g.edges.get(eid)
        if e is None or e.len >= CONTRACT or e.a == e.b:
            continue
        a, b = e.a, e.b
        if b in g.sink and a not in g.sink:
            a, b = b, a
        if len(g.adj[b]) > len(g.adj[a]) and a not in g.sink:
            a, b = b, a
        g.remove_edge(eid)
        g.merge_into(b, a)
        n_c += 1
    for eid in sorted(g.edges):
        e = g.edges.get(eid)
        if e is not None and e.a == e.b and e.len < LOOP_MAX:
            g.remove_edge(eid)
            n_l += 1
    by_pair = {}
    for eid in sorted(g.edges):
        e = g.edges[eid]
        if e.a != e.b:
            by_pair.setdefault((min(e.a, e.b), max(e.a, e.b)), []).append(eid)
    for pr, ids in by_pair.items():
        if len(ids) < 2:
            continue
        ids.sort(key=lambda x: (g.edges[x].cls, -(g.edges[x].flags & F_DUAL), g.edges[x].len, x))
        keep = g.edges[ids[0]]
        for x in ids[1:]:
            o = g.edges[x]
            if o.len < 1.5 * keep.len + 200.0 and o.len < 5000.0:
                if keep.flags & F_ONEWAY and o.flags & F_ONEWAY and o.a == keep.b:
                    keep.flags = (keep.flags & ~F_ONEWAY) | F_DUAL
                    keep.lanes = LANES_DUAL.get(keep.cls, keep.lanes)
                g.remove_edge(x)
                n_d += 1
    for n in sorted(g.nodes):
        if n in g.adj and not g.adj[n]:
            g.remove_node_if_free(n)
    return n_c, n_l, n_d


# ---------------------------------------------------------------- degree-2 collapse

CORE = F_ONEWAY | F_FERRY
CLS_ABSORB = 500.0     # real m: a piece this short between two edges of another class takes their class


def collapse_deg2(g):
    """Merge the two edges at every degree-2 node of the same one-way / ferry kind. A change of name, lanes,
    dual carriageway or synthetic link alone does not keep a node (64 B each: half the file); neither does a
    class change when the shorter part is under CLS_ABSORB. The merged edge takes the longer part's class,
    name, lanes and F_DUAL; F_SYNTH, F_LINK and the structure flags are OR'ed."""
    n_m = 0
    for n in sorted(g.nodes):
        if n not in g.adj or len(g.adj[n]) != 2 or n in g.sink:
            continue
        e1, e2 = g.adj[n]
        if e1 == e2:
            continue
        A, B = g.edges[e1], g.edges[e2]
        if (A.flags & F_FERRY) != (B.flags & F_FERRY):
            continue
        if (A.flags & F_ONEWAY) != (B.flags & F_ONEWAY):
            # a short one-way piece (a carriageway seam by an interchange) joins the two-way road
            short = A if A.flags & F_ONEWAY else B
            if short.len >= CLS_ABSORB:
                continue
            short.flags &= ~F_ONEWAY
        if A.cls != B.cls and (min(A.len, B.len) >= CLS_ABSORB or FEEDER in (A.cls, B.cls)):
            continue
        if A.flags & F_ONEWAY:
            # travel order: the edge arriving at n first
            if A.b != n:
                A, B = B, A
            if A.b != n or B.a != n:
                continue
        else:
            if A.b != n:
                A = _rev(A)
            if B.a != n:
                B = _rev(B)
        if A.a == n or B.b == n:
            continue
        if A.len + B.len > EDGE_MAX_TOY / TOY:
            continue
        xy = np.vstack([A.xy, B.xy[1:]])
        st = [list(s) for s in A.structs] + [[k, lv, s0 + A.len, s1 + A.len] for k, lv, s0, s1 in B.structs]
        main, other = (A, B) if A.len >= B.len else (B, A)
        flags = (A.flags | B.flags) & ~F_DUAL | (main.flags & F_DUAL)
        name = main.name or other.name
        a, b = A.a, B.b
        g.remove_edge(e1)
        g.remove_edge(e2)
        g.add_edge(a, b, xy, main.cls, flags, main.lanes, name, st, main.key)
        g.remove_node_if_free(n)
        n_m += 1
    return n_m


def _rev(e):
    r = Edge(e.b, e.a, e.xy[::-1].copy(), e.cls, e.flags, e.lanes, e.name,
             [[k, lv, e.len - s1, e.len - s0] for k, lv, s0, s1 in e.structs], e.key)
    return r


# ---------------------------------------------------------------- sinks and repair

class Sinks:
    def __init__(self, towns, airports, ferry):
        from scipy.spatial import cKDTree
        self.towns = towns
        self.t_tree = cKDTree(ecef(np.array([t["lon"] for t in towns]), np.array([t["lat"] for t in towns])))
        self.t_ext = np.array([max(TOWN_MIN_EXT, t["ext"]) for t in towns]) / TOY
        self.max_ext = float(self.t_ext.max())
        self.ap = airports
        self.a_tree = cKDTree(ecef(np.array([a[1] for a in airports]), np.array([a[2] for a in airports])))
        self.f_tree = cKDTree(ecef(ferry[:, 0], ferry[:, 1])) if len(ferry) else None

    def find(self, lon, lat):
        q = ecef(np.array([lon]), np.array([lat]))[0]
        idx = self.t_tree.query_ball_point(q, self.max_ext)
        best = None
        for i in sorted(idx):
            d = float(np.linalg.norm(self.t_tree.data[i] - q))
            if d <= self.t_ext[i]:
                r = d / self.t_ext[i]
                if best is None or r < best[0]:
                    best = (r, i)
        if best is not None:
            t = self.towns[best[1]]
            return (S_TOWN, t["name"], t["gid"], float(max(TOWN_MIN_EXT, t["ext"])))
        d, i = self.a_tree.query(q, distance_upper_bound=AIRPORT_TOY / TOY)
        if np.isfinite(d):
            return (S_AIRPORT, self.ap[i][0], 0, 300.0)
        if self.f_tree is not None:
            d, i = self.f_tree.query(q, distance_upper_bound=FERRY_TOY / TOY)
            if np.isfinite(d):
                return (S_FERRY, "", 0, 100.0)
        return None


def load_airports():
    keep = None
    if os.path.exists(KEEP):
        keep = set(json.load(open(KEEP, encoding="utf-8"))["keep"])
    else:
        import subprocess
        try:
            out = subprocess.run(["git", "-C", ROOT, "show", "w1-apsel:data/airports/keep.json"],
                                 capture_output=True, check=True).stdout
            keep = set(json.loads(out.decode("utf-8"))["keep"])
            log("airports: keep.json read from branch w1-apsel")
        except Exception:
            log("airports: no keep.json, every airport counts")
    d = json.load(open(AIRPORTS, encoding="utf-8"))
    f = d["fields"]
    ic, il, io = f.index("code"), f.index("lat"), f.index("lon")
    moves = json.load(open(MOVES, encoding="utf-8")).get("moves", {}) if os.path.exists(MOVES) else {}
    out = []
    for a in d["airports"]:
        code = a[ic]
        if keep is not None and code not in keep:
            continue
        lat, lon = float(a[il]), float(a[io])
        mv = moves.get(code)
        if mv:
            lat += float(mv.get("dlat", 0.0))
            lon += float(mv.get("dlon", 0.0))
        out.append((code, lon, lat))
    out.sort()
    log("airports: %d" % len(out))
    return out


def repair(g, sinks):
    stats = {"snap": 0, "merge": 0, "prune": 0, "sink": 0, "extend": 0, "park": 0, "delete": 0}
    for rnd in range(12):
        changed = 0
        if rnd:
            cleanup(g)
        for n in sorted(g.nodes):
            if n in g.adj and not g.adj[n]:
                g.remove_node_if_free(n)
        ends = [n for n in sorted(g.nodes) if len(g.adj[n]) == 1 and n not in g.sink]
        # 1 snap
        for n in ends:
            if n not in g.nodes or len(g.adj[n]) != 1 or n in g.sink:
                continue
            ch, root, L = g.chain(n)
            skip = set(ch)
            p = g.nodes[n]
            hit = g.nearest_on(p[0], p[1], SNAP_TOY / TOY, skip)
            if hit is None:
                continue
            d, eid, i, t, pt = hit
            if root in g.nodes and _near_root(g, root, pt, eid):
                continue
            e0 = g.edges[g.adj[n][0]]
            m = g.split(eid, i, t, pt)
            if d < MERGE_TOY / TOY:
                g.merge_into(n, m)
                stats["merge"] += 1
            else:
                g.add_edge(n, m, [g.nodes[n], g.nodes[m]], e0.cls, F_SYNTH, e0.lanes, "", [], e0.key)
                stats["snap"] += 1
            changed += 1
        ends = [n for n in sorted(g.nodes) if len(g.adj[n]) == 1 and n not in g.sink]
        # 2 prune short spurs (an end that is a sink candidate stays)
        for n in ends:
            if n not in g.nodes or len(g.adj[n]) != 1 or n in g.sink:
                continue
            ch, root, L = g.chain(n)
            if L * TOY >= PRUNE_TOY or root == n or len(g.adj.get(root, ())) < 3:
                continue
            p = g.nodes[n]
            s = sinks.find(p[0], p[1])
            if s is not None and s[0] != S_TOWN:
                continue
            _delete_chain(g, ch)
            stats["prune"] += 1
            changed += 1
        ends = [n for n in sorted(g.nodes) if len(g.adj[n]) == 1 and n not in g.sink]
        # 3 sinks, 4 extend, 5 car park, 6 delete back
        for n in ends:
            if n not in g.nodes or len(g.adj[n]) != 1 or n in g.sink:
                continue
            p = g.nodes[n]
            s = sinks.find(p[0], p[1])
            if s is not None:
                g.sink[n] = s
                stats["sink"] += 1
                continue
            ch, root, L = g.chain(n)
            hit = g.nearest_on(p[0], p[1], EXTEND_TOY / TOY, set(ch))
            if hit is not None and not (root in g.nodes and _near_root(g, root, hit[4], hit[1])):
                d, eid, i, t, pt = hit
                m = g.split(eid, i, t, pt)
                if d < MERGE_TOY / TOY:
                    g.merge_into(n, m)
                    stats["merge"] += 1
                else:
                    g.add_edge(n, m, [g.nodes[n], g.nodes[m]], FEEDER, F_SYNTH, 1, "", [], -1)
                    stats["extend"] += 1
                changed += 1
                continue
            if L * TOY >= PARK_TOY and root != n:
                g.sink[n] = (S_PARKING, "", 0, 60.0)
                stats["park"] += 1
                continue
            _delete_chain(g, ch)
            stats["delete"] += 1
            changed += 1
        log("  repair round %d: %s" % (rnd + 1, stats))
        if not changed:
            break
    for n in sorted(g.nodes):
        if n in g.adj and not g.adj[n]:
            g.remove_node_if_free(n)
    return stats


def _near_root(g, root, pt, eid):
    """A snap target right beside the junction the spur starts from would only make a little loop."""
    e = g.edges[eid]
    if root not in (e.a, e.b):
        return False
    r = g.nodes[root]
    c = math.cos(math.radians(r[1]))
    return math.hypot((pt[0] - r[0]) * c, pt[1] - r[1]) * M_DEG < 1000.0


def _delete_chain(g, ch):
    for eid in ch:
        if eid in g.edges:
            e = g.remove_edge(eid)
            g.remove_node_if_free(e.a)
            g.remove_node_if_free(e.b)


def drop_islands(g):
    """Components of the graph shorter than ISLAND_M in all are fragments, not a network: removed."""
    seen = set()
    n_d = 0
    for n0 in sorted(g.nodes):
        if n0 in seen:
            continue
        comp = [n0]
        seen.add(n0)
        k = 0
        es = set()
        while k < len(comp):
            n = comp[k]
            k += 1
            for eid in g.adj[n]:
                es.add(eid)
                m = g.other(eid, n)
                if m not in seen:
                    seen.add(m)
                    comp.append(m)
        if sum(g.edges[x].len for x in es) < ISLAND_M:
            for eid in sorted(es):
                g.remove_edge(eid)
            for n in comp:
                g.remove_node_if_free(n)
            n_d += 1
    log("islands: %d small components removed" % n_d)


# ---------------------------------------------------------------- output

def coast_sea():
    p = os.path.join(ROOT, "data", "world", "coast.bin")
    if not os.path.exists(p):
        return None
    a = np.fromfile(p, dtype=np.uint8).reshape(2048, 4096)

    def is_water(lon, lat):
        x = int(((lon + 180.0) / 360.0 * 4096) % 4096)
        y = int(min(2047, max(0, (90.0 - lat) / 180.0 * 2048)))
        return a[y, x] < 120
    return is_water


def finish(g):
    """Sea bridges, Douglas-Peucker, long edges split (point deltas within i16 at write time)."""
    sea = coast_sea()
    n_sea = 0
    for eid in sorted(g.edges):
        e = g.edges[eid]
        if sea is not None and e.flags & F_BRIDGE:
            for kind, lv, s0, s1 in e.structs:
                if kind == ST_BRIDGE and s1 - s0 >= 400.0:
                    pts = [_at(e.xy, s0 + (s1 - s0) * f) for f in (0.25, 0.5, 0.75)]
                    if all(sea(p[0], p[1]) for p in pts):
                        e.flags |= F_SEA_OK
                        n_sea += 1
                        break
    simplify(g)
    n_split = sum(split_long(g, eid, EDGE_MAX_TOY) for eid in sorted(g.edges))
    log("finish: %d sea bridges, %d long-edge splits" % (n_sea, n_split))


def simplify(g):
    """Douglas-Peucker DP_TOY in a local metric frame; structure ranges kept as shares of the length."""
    import shapely
    from shapely.geometry import LineString
    for eid in sorted(g.edges):
        e = g.edges[eid]
        c = math.cos(math.radians(float(e.xy[:, 1].mean())))
        loc = np.stack([e.xy[:, 0] * c, e.xy[:, 1]], axis=1)
        if len(loc) > 2:
            simp = np.asarray(shapely.simplify(LineString(loc), DP_TOY / TOY / M_DEG, preserve_topology=False).coords)
            if len(simp) >= 2:
                xy = np.stack([simp[:, 0] / c, simp[:, 1]], axis=1)
                xy[0], xy[-1] = e.xy[0], e.xy[-1]
                L0 = e.len
                L1 = poly_len(xy)
                k = L1 / L0 if L0 > 0 else 1.0
                e.structs = [[kd, lv, s0 * k, s1 * k] for kd, lv, s0, s1 in e.structs]
                e.xy = xy
                e.len = L1


def split_long(g, eid, max_toy):
    """Cut edge eid into equal parts of at most max_toy at PASS nodes; returns the number of cuts."""
    e = g.edges.get(eid)
    if e is None or e.len * TOY <= max_toy:
        return 0
    parts = int(math.ceil(e.len * TOY / max_toy))
    seg = e.len / parts
    cur = eid
    cuts = 0
    for k in range(1, parts):
        ee = g.edges[cur]
        sm = step_m(ee.xy)
        cs = np.concatenate([[0.0], np.cumsum(sm)])
        i = int(np.clip(np.searchsorted(cs, seg) - 1, 0, len(sm) - 1))
        t = min(1.0, max(0.0, (seg - cs[i]) / max(sm[i], 1e-9)))
        pt = ee.xy[i] + (ee.xy[i + 1] - ee.xy[i]) * t
        a0 = ee.a
        n = g.split(cur, i, float(t), pt)
        if n == a0 or n == ee.b:
            break
        nxt = [x for x in g.adj[n] if g.edges[x].a == n and x > cur]
        if not nxt:
            break
        cur = max(nxt)
        cuts += 1
    return cuts


def subdivide(g):
    """Tiles over TILE_EDGES edges (RoadGraph decodes a tile in one go, ~5 us per edge: A11 15 ms) become 4 x 4
    quarter-degree sub-tiles (RGR1 v2) whose edges stay within SUB_SPAN_TOY of their sub-tile."""
    from collections import Counter
    cnt = Counter(rgf.tile_key(g.nodes[e.a][1], g.nodes[e.a][0]) for e in g.edges.values())
    dense = {k for k, c in cnt.items() if c > TILE_EDGES}
    if not dense:
        return dense, cnt
    n = 0
    for eid in sorted(g.edges):
        e = g.edges.get(eid)
        if e is not None and rgf.tile_key(g.nodes[e.a][1], g.nodes[e.a][0]) in dense:
            n += split_long(g, eid, SUB_EDGE_TOY)
    key_of = _key_of(dense)
    sub = Counter(key_of(g.nodes[e.a][1], g.nodes[e.a][0]) for e in g.edges.values())
    log("subdivide: %d dense tiles (max %d edges) split into %d sub-tiles (max %d edges), %d edge cuts" % (
        len(dense), max(cnt[k] for k in dense), sum(1 for k in sub if k >= 65536),
        max(c for k, c in sub.items() if k >= 65536), n))
    return dense, sub


def _key_of(dense):
    def f(lat, lon):
        k = rgf.tile_key(lat, lon)
        return rgf.sub_key(lat, lon) if k in dense else k
    return f


def _at(xy, s):
    sm = step_m(xy)
    cs = np.concatenate([[0.0], np.cumsum(sm)])
    i = int(np.clip(np.searchsorted(cs, s) - 1, 0, len(sm) - 1))
    t = (s - cs[i]) / max(sm[i], 1e-9)
    return xy[i] + (xy[i + 1] - xy[i]) * min(1.0, max(0.0, t))


def densify_pts(xy):
    out = [xy[0]]
    for i in range(1, len(xy)):
        d = xy[i] - out[-1]
        k = int(math.ceil(max(abs(d[0]), abs(d[1])) / PT_MAX_DEG))
        for j in range(1, k):
            out.append(out[-1] + d * (j / k) if j == 1 else xy[i - 1] + (xy[i] - xy[i - 1]) * (j / k))
        out.append(xy[i])
    return np.array(out)


F_FILE_ZSTD = 4        # header flags bit 2 (v2): every tile payload is one zstd frame


def to_bytes(g, flags=0, dense=()):
    nodes = []
    key_of = _key_of(set(dense))
    order = sorted(g.nodes, key=lambda n: (key_of(g.nodes[n][1], g.nodes[n][0]), g.nodes[n][1], g.nodes[n][0], n))
    rank = {n: i for i, n in enumerate(order)}
    for n in order:
        lon, lat = g.nodes[n]
        nodes.append({"name": n, "lat": lat, "lon": lon, "sink": g.sink.get(n)})
    keep_names = name_budget(g)
    edges = []
    for eid in sorted(g.edges, key=lambda x: (rank[g.edges[x].a], rank[g.edges[x].b], x)):
        e = g.edges[eid]
        xy = e.xy
        if len(xy) > 2 and np.abs(np.diff(xy, axis=0)).max() > PT_MAX_DEG:
            xy = densify_pts(xy)
        L = max(e.len, 1e-6)
        st = [(k, lv, max(0.0, s0 / L), min(1.0, s1 / L)) for k, lv, s0, s1 in tidy_structs(e.structs)]
        fl = (e.flags & ~(F_BRIDGE | F_TUNNEL)) | _sflags(st)
        if not fl & F_BRIDGE:
            fl &= ~F_SEA_OK
        edges.append({"a": e.a, "b": e.b, "cls": e.cls, "flags": fl & 255, "lanes": e.lanes,
                      "width": WIDTH.get(e.cls, 12), "name": e.name if e.name in keep_names else "", "pts": [(p[1], p[0]) for p in xy[1:-1]],
                      "structs": st})
    import zstandard
    zc = zstandard.ZstdCompressor(level=19, write_checksum=False, write_content_size=True)
    return rgf.encode(nodes, edges, release=OVERTURE, flags=flags | F_FILE_ZSTD, key_of=key_of,
                      sub_span=int(SUB_SPAN_TOY), compress=zc.compress)


def tidy_structs(structs):
    """Ranges of one kind closer than STRUCT_GAP merge; then ranges under STRUCT_MIN (culverts, the bridge of a
    stream crossing: below the 20 toy m profile step) go."""
    out = []
    for k, lv, s0, s1 in sorted(structs, key=lambda s: (s[2], s[0])):
        if out and out[-1][0] == k and s0 - out[-1][3] < STRUCT_GAP:
            out[-1][3] = max(out[-1][3], s1)
            if abs(lv) > abs(out[-1][1]):
                out[-1][1] = lv
            continue
        out.append([k, lv, s0, s1])
    return [r for r in out if r[3] - r[2] >= STRUCT_MIN]


def name_budget(g, limit=65000):
    """RGR1 name ids are u16: every sink name, then road names by class and total length until the budget."""
    sink_names = {s[1] for s in g.sink.values() if s[1]}
    score = {}
    for e in g.edges.values():
        if e.name:
            c, L = score.get(e.name, (99, 0.0))
            score[e.name] = (min(c, e.cls), L + e.len)
    order = sorted((v[0], -v[1], k) for k, v in score.items() if k not in sink_names)
    room = max(0, limit - len(sink_names))
    keep = {k for _, _, k in order[:room]} | sink_names
    if len(order) > room:
        log("names: %d road names over the u16 budget dropped" % (len(order) - room))
    return keep


def report(g, data):
    km = [0.0] * 7
    for e in g.edges.values():
        km[e.cls] += e.len / 1000.0
    sk = {}
    for s in g.sink.values():
        sk[SINK_KM_NAMES[s[0]]] = sk.get(SINK_KM_NAMES[s[0]], 0) + 1
    tidy = [s for e in g.edges.values() for s in tidy_structs(e.structs)]
    nb = sum(1 for s in tidy if s[0] == ST_BRIDGE)
    nt = sum(1 for s in tidy if s[0] == ST_TUNNEL)
    dang = sum(1 for n in g.nodes if len(g.adj[n]) <= 1 and n not in g.sink)
    log("report: real km " + ", ".join("%s %.0f" % (CLS_KM_NAMES[i], km[i]) for i in range(7)))
    log("  nodes %d, edges %d, sinks %s, bridges %d, tunnels %d, dangling %d, %.2f MB" % (
        len(g.nodes), len(g.edges), sk, nb, nt, dang, len(data) / 1e6))
    return {"km": km, "sinks": sk, "bridges": nb, "tunnels": nt, "dangling": dang}


def seg_names(src):
    out = np.empty(src.n, dtype=object)
    for i in range(src.n):
        nm, rf = src.name[i], src.ref[i]
        if src.cls_s[i] in ("motorway", "trunk"):
            out[i] = rf or nm or ""
        else:
            out[i] = nm or rf or ""
    return out


REGION = 10            # degrees: phase A bakes the world in blocks this size (memory: the world at once is 40 GB+)
MARGIN = 0.3           # degrees of context loaded round a block
STITCH = 150.0         # real m: border nodes of two blocks this close are one crossing
WORKERS = int(os.environ.get("RB_WORKERS", "3"))


def bake_region(job):
    """Phase A for one block: everything up to the repair, simplified, then clipped to the block (core).
    Returns (name, nodes [(id, lon, lat, border)], edges [(a, b, xy, cls, flags, lanes, name, structs)], s)."""
    name, core, towns, boxes = job
    s, w, n, e = core
    t0 = time.time()
    src = load_source((s - MARGIN, w - MARGIN, n + MARGIN, e + MARGIN), boxes, towns)
    if src is None:
        return name, [], [], 0.0
    keep, gcls, link, _ferry = scope(src, towns, boxes)
    if not keep.any():
        return name, [], [], 0.0
    names = seg_names(src)
    P = topology(src, keep)
    if P.n == 0:
        return name, [], [], 0.0
    state, pairs = dual_merge(src, P, gcls, link)
    keys = road_keys(src, P, gcls, link, pairs, names)
    g = build_graph(src, P, gcls, link, state, keys, names)
    del src
    collapse_links(g, P, link, keys)
    del P
    for _ in range(2):
        log("  cleanup: %d short edges contracted, %d loops, %d duplicates" % cleanup(g))
        log("  collapse: %d degree-2 nodes merged" % collapse_deg2(g))
    simplify(g)
    border = clip_core(g, core)
    nodes = [(nid, g.nodes[nid][0], g.nodes[nid][1], nid in border) for nid in sorted(g.nodes)]
    edges = [(x.a, x.b, x.xy, x.cls, x.flags, x.lanes, x.name, x.structs) for _, x in sorted(g.edges.items())]
    log("block %s: %d nodes, %d edges, %d border nodes, %.0f s" % (name, len(nodes), len(edges), len(border),
                                                                   time.time() - t0))
    return name, nodes, edges, time.time() - t0


def clip_core(g, core):
    """Cut every edge where it crosses the block border and keep what lies inside; returns the border nodes."""
    s, w, n, e = core
    border = set()
    for eid in sorted(g.edges):
        cur = eid
        for _ in range(256):
            ed = g.edges.get(cur)
            if ed is None:
                break
            xy = ed.xy
            hit = None
            for i in range(len(xy) - 1):
                (x0, y0), (x1, y1) = xy[i], xy[i + 1]
                ts = []
                for v, a0, a1 in ((s, y0, y1), (n, y0, y1), (w, x0, x1), (e, x0, x1)):
                    if (a0 - v) * (a1 - v) < 0:
                        ts.append((v - a0) / (a1 - a0))
                ts = [t for t in ts if 1e-9 < t < 1 - 1e-9]
                if ts:
                    hit = (i, min(ts))
                    break
            if hit is None:
                break
            i, t = hit
            pt = xy[i] + (xy[i + 1] - xy[i]) * t
            a0, b0 = ed.a, ed.b
            m = g.split(cur, i, t, pt)
            if m in (a0, b0):
                break
            border.add(m)
            nxt = [x for x in g.adj[m] if x > cur and g.edges[x].a == m]
            if not nxt:
                break
            cur = max(nxt)
    for eid in sorted(g.edges):
        ed = g.edges[eid]
        mid = _at(ed.xy, ed.len * 0.5) if ed.len > 0 else ed.xy[0]
        if not (s <= mid[1] < n and w <= mid[0] < e):
            g.remove_edge(eid)
    for nid in sorted(g.nodes):
        g.remove_node_if_free(nid)
    return {b for b in border if b in g.nodes}


def regions(only):
    if only and not os.environ.get("RB_SPLIT"):
        return [("only", tuple(only))]
    if only:
        # test the block split and stitch on a bbox: its blocks, clipped to it
        s, w, n, e = only
        out = []
        for nm, (bs, bw, bn, be) in regions(None):
            if bs < n and bn > s and bw < e and be > w:
                out.append((nm, (max(bs, s), max(bw, w), min(bn, n), min(be, e))))
        return out
    out = []
    for la in range(-90, 90, REGION):
        for lo in range(-180, 180, REGION):
            out.append(("%d_%d" % (la, lo), (float(la), float(lo), float(la + REGION), float(lo + REGION))))
    return out


def stitch(g, border):
    """Pair the border nodes of neighbouring blocks (mutual nearest, same class, within STITCH) into one node."""
    from scipy.spatial import cKDTree
    ids = sorted(b for b in border if b in g.nodes and len(g.adj[b]) == 1)
    if len(ids) < 2:
        return 0
    xy = np.array([g.nodes[b] for b in ids])
    blk = np.array([b >> 40 for b in ids])
    cls = np.array([g.edges[g.adj[b][0]].cls for b in ids])
    X = ecef(xy[:, 0], xy[:, 1])
    dd, ii = cKDTree(X).query(X, k=8, distance_upper_bound=STITCH)
    best = np.full(len(ids), -1)
    for i in range(len(ids)):
        for d, j in zip(dd[i], ii[i]):
            if j >= len(ids) or j == i or blk[j] == blk[i]:
                continue
            if cls[j] != cls[i] and not (cls[i] <= TRUNK and cls[j] <= TRUNK):
                continue
            best[i] = j
            break
    n_s = 0
    for i in range(len(ids)):
        j = best[i]
        if j > i and best[j] == i:
            a, b = ids[i], ids[j]
            if a in g.nodes and b in g.nodes:
                pa, pb = g.nodes[a], g.nodes[b]
                g.move_node(a, [(pa[0] + pb[0]) * 0.5, (pa[1] + pb[1]) * 0.5])
                g.merge_into(b, a)
                n_s += 1
    return n_s


def bake(only=None):
    t0 = time.time()
    towns = load_towns()
    boxes = load_boxes(towns)
    regs = regions(only)
    jobs = [(nm, core, towns, boxes) for nm, core in regs]
    results = {}
    import pickle
    cache = os.environ.get("RB_REUSE_A")
    if cache and os.path.exists(cache):
        with open(cache, "rb") as f:
            results = pickle.load(f)
        log("phase A: %d blocks from %s" % (len(results), cache))
    elif len(jobs) == 1:
        nm, nodes, edges, dt = bake_region(jobs[0])
        results[nm] = (nodes, edges)
    else:
        with cf.ProcessPoolExecutor(WORKERS) as ex:
            futs = {ex.submit(bake_region, j): j[0] for j in jobs}
            done = 0
            for f in cf.as_completed(futs):
                nm, nodes, edges, dt = f.result()
                done += 1
                if nodes:
                    results[nm] = (nodes, edges)
                    log("[%d/%d] block %s: %d edges, %.0f s (%.0f s total)" % (
                        done, len(jobs), nm, len(edges), dt, time.time() - t0))
    if cache and not os.path.exists(cache):
        with open(cache, "wb") as f:
            pickle.dump(results, f, protocol=4)
    # phase B: one graph; node ids made unique per block (block number << 40)
    g = Graph()
    border = set()
    for bi, (nm, _) in enumerate(regs):
        if nm not in results:
            continue
        nodes, edges = results.pop(nm)
        base = (bi + 1) << 40
        for nid, lon, lat, isb in nodes:
            g.add_node((lon, lat), base + nid)
            if isb:
                border.add(base + nid)
        for a, b, xy, c, fl, ln, name, st in edges:
            g.add_edge(base + a, base + b, xy, c, fl, ln, name, st, -1)
    log("joined: %d nodes, %d edges, %d border nodes" % (len(g.nodes), len(g.edges), len(border)))
    log("stitch: %d block-border crossings joined" % stitch(g, border))
    log("  cleanup: %d short edges contracted, %d loops, %d duplicates" % cleanup(g))
    log("  collapse: %d degree-2 nodes merged" % collapse_deg2(g))
    sinks = Sinks(towns, load_airports(), load_ferry_ends())
    repair(g, sinks)
    log("  collapse: %d degree-2 nodes merged" % collapse_deg2(g))
    drop_islands(g)
    finish(g)
    dense, _ = subdivide(g)
    data = to_bytes(g, dense=dense)
    rep = report(g, data)
    log("bake: %.0f s" % (time.time() - t0))
    return data, rep


def load_ferry_ends():
    """Both ends of every Overture ferry route (subtype water): S_FERRY sinks."""
    import duckdb
    import shapely
    con = duckdb.connect()
    lst = "[" + ", ".join(_sql_str(p.replace(chr(92), "/")) for p in cache_files()) + "]"
    r = con.execute(f"SELECT wkb FROM read_parquet({lst}) WHERE subtype = 'water' ORDER BY id").fetchall()
    con.close()
    pts = []
    for (w,) in r:
        c = shapely.get_coordinates(shapely.from_wkb(bytes(w)))
        if len(c):
            pts.append(c[0])
            pts.append(c[-1])
    log("ferries: %d route ends" % len(pts))
    return np.array(pts) if pts else np.zeros((0, 2))


# ---------------------------------------------------------------- check (decode the file)

def decode_check(data):
    """A1 and structure sanity straight from the bytes (the RoadGraph._decode layout)."""
    assert data[:4] == b"RGR1", "magic"
    ver, flags, nt, toff, noff, nn, tdeg = struct.unpack_from("<7I", data, 4)
    tn, te = struct.unpack_from("<II", data, 48)
    dang = 0
    nodes = edges = sinks = structs = 0
    kinds = [0, 0, 0, 0]
    prev = -1
    for t in range(nt):
        k, off, nb, n_nodes, n_edges, n_adj, n_pts, n_st, n_sk = struct.unpack_from("<IIIIIIIHH", data, toff + 32 * t)
        assert k > prev
        prev = k
        raw = 16 * n_nodes + 8 * n_adj + 32 * n_edges + 4 * n_pts + 16 * n_st + 16 * n_sk
        if flags & F_FILE_ZSTD:
            import zstandard
            pay = zstandard.ZstdDecompressor().decompress(data[off:off + nb], max_output_size=raw)
        else:
            pay = data[off:off + nb]
        assert len(pay) == raw
        for i in range(n_nodes):
            _la, _lo, _ao, deg, kind, sk = struct.unpack_from("<iiIBBH", pay, 16 * i)
            kinds[kind] += 1
            if kind == 3 or (deg <= 1 and sk == 0xFFFF):
                dang += 1
        nodes += n_nodes
        edges += n_edges
        sinks += n_sk
        structs += n_st
    assert nodes == tn and edges == te
    return {"tiles": nt, "nodes": nodes, "edges": edges, "sinks": sinks, "structs": structs, "dangling": dang,
            "kinds": kinds, "names": nn, "bytes": len(data)}


def write_sources():
    line = ("roadgraph.bin: Overture Maps transportation/segment release %s (OpenStreetMap roads: motorway, trunk, "
            "primary, secondary, named avenues, as a connected graph), python tools/bake_road_graph.py, baked %s"
            % (OVERTURE, time.strftime("%Y-%m-%d")))
    with open(SOURCES, "rb") as f:
        txt = f.read().decode("utf-8")
    eol = "\r\n" if "\r\n" in txt else "\n"
    lines = [x for x in txt.split(eol) if x and not x.startswith("roadgraph.bin:")]
    lines.append(line)
    with open(SOURCES, "wb") as f:
        f.write((eol.join(lines) + eol).encode("utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch-only", action="store_true", help="fill the caches and stop")
    ap.add_argument("--no-fetch", action="store_true", help="bake from the caches only")
    ap.add_argument("--only", default="", help="lat0,lon0,lat1,lon1: a bbox, written to C:/tmp/roadgraph_<name>.bin")
    ap.add_argument("--name", default="only")
    ap.add_argument("--out", default="", help="write here instead (experiments; SOURCES.txt untouched)")
    ap.add_argument("--check", action="store_true", help="decode the output (A1) and rebake to compare sha256")
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()
    only = tuple(float(v) for v in a.only.split(",")) if a.only else None
    out = a.out or ("C:/tmp/roadgraph_%s.bin" % a.name if only else OUT)
    if a.check:
        with open(out, "rb") as f:
            have = f.read()
        c = decode_check(have)
        log("check %s: %s" % (out, c))
        data, _ = bake(only)
        h0, h1 = hashlib.sha256(have).hexdigest(), hashlib.sha256(data).hexdigest()
        log("sha256 file %s\nsha256 bake %s\n%s" % (h0, h1, "IDENTICAL" if h0 == h1 else "DIFFERENT"))
        ok = c["dangling"] == 0 and h0 == h1 and (only or len(have) <= 30 * 1024 * 1024)
        log("A1 dangling %d, size %.2f MB: %s" % (c["dangling"], len(have) / 1e6, "PASS" if ok else "FAIL"))
        sys.exit(0 if ok else 1)
    if not a.no_fetch:
        run_fetch(a.jobs)
    if a.fetch_only:
        return
    data, rep = bake(only)
    c = decode_check(data)
    log("decoded: %s" % c)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "wb") as f:
        f.write(data)
    log("wrote %s (%.2f MB, sha256 %s)" % (out, len(data) / 1e6, hashlib.sha256(data).hexdigest()))
    if not only and not a.out:
        write_sources()
    if c["dangling"]:
        log("A1 FAILED: %d dangling" % c["dangling"])
        sys.exit(1)


if __name__ == "__main__":
    main()
