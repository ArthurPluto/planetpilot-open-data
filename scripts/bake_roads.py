"""Bake the real main roads of the big cities: data/world/odbl/roads.bin (RDN1).

docs/REAL_PLACEMENT.md section 7. Source: Overture Maps transportation/segment
(OpenStreetMap, ODbL), release OVERTURE below, read with DuckDB straight from
the public S3 bucket (only the row groups of each city's bbox). The output is
an ODbL Derivative Database and lives under data/world/odbl/ (published as is
with this script and the release date; see data/world/odbl/SOURCES.txt).

Cities: the ~200 biggest GeoNames cities5000 places (one per agglomeration),
the 66 cities of the position audit, every landmark city and every city of
tools/roads_iconic.json. Each city's query box is its GHS-UCDB R2024A urban
centre polygon (the one holding the town point) plus 2 km, clamped to
8..35 km real around the town point.

Kept per city (generalised for the 1/5 planet, toy m = real m / 5):
- motorway and trunk always (dual carriageways merged: a stroke parallel to a
  kept one within 25 toy m is dropped), slip roads (subclass link) never;
- primary, secondary and trunks under 10 km real only when at least 60% of
  their length runs within 25 deg of the city's street grid (the towns snap
  their grid streets onto those; a diagonal one would only clear a corridor);
- primary and secondary by stroke length, longest first, a stroke's runs that
  lie within 300 toy m of a parallel kept road dropped, so parallel arterials
  end up >= 300 toy m apart; runs under 300 toy m dropped;
- the named avenues of tools/roads_iconic.json whatever their class, within
  20 km real of the town point, parallel ones 160 toy m apart (list order);
- Douglas-Peucker 3 toy m; widths real (motorway 22 m, trunk 18, primary 16,
  secondary 12, iconic >= 20).
Street bearing: the length-weighted mode (mod 90 deg, 3 deg smoothing) of every
road segment (residential and up) within BEARING_R real m of the town point.

roads.bin (RDN1, little endian, internal to TR: scripts/world/road_net.gd):
  header 24 B : "RDN1", u32 version (1), u32 n_cities, u32 cities_off,
                u32 names_off, u32 names_count
  cities n x 40 B (sorted by lat): u32 geonameid, f64 lat, f64 lon (origin),
                f32 bearing (deg 0..90 from east, counter-clockwise, NaN unknown),
                f32 radius (toy m the lines reach from the origin),
                u32 line_off (from file start), u32 n_lines, u32 0
  lines        : u8 cls (RoadNet.MOTORWAY..ICONIC), u8 width_m, u16 name_id
                (0xFFFF none; iconic names only), u16 n_pts, then n_pts x
                (i16 east, i16 north) toy m from the origin
  names        : names_count UTF-8 strings, each 0-terminated

Caches (data/world/src/overture/roads/, gitignored, junctioned in worktrees):
  <geonameid>.parquet  the city's raw segments (id, class, subclass, name, wkb)

Usage:
  python tools/bake_roads.py                 fetch the missing caches, then bake
  python tools/bake_roads.py --no-fetch      bake from the caches only
  python tools/bake_roads.py --only=Paris,Tel Aviv   these cities only (no write)
  python tools/bake_roads.py --list          print the city list and exit
"""
import argparse
import concurrent.futures as cf
import glob
import io
import json
import math
import os
import sqlite3
import struct
import sys
import threading
import time
import zipfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "data", "world", "src")
GEONAMES = os.path.join(SRC, "geonames", "cities5000.zip")
UCDB = os.path.join(SRC, "ghsl", "GHS_UCDB_GLOBE_R2024A.gpkg")
CACHE = os.path.join(SRC, "overture", "roads")
TOWNS_BIN = os.path.join(ROOT, "data", "world", "towns", "towns.bin")
ICONIC = os.path.join(ROOT, "tools", "roads_iconic.json")
AUDIT_CITIES = "C:/tmp/posaudit/cities.json"
LANDMARKS = os.path.join(ROOT, "data", "landmarks")
OUT = os.path.join(ROOT, "data", "world", "odbl", "roads.bin")

OVERTURE = "2026-09-23.1"
SEGMENTS = ("s3://overturemaps-us-west-2/release/%s/theme=transportation/type=segment/*.parquet" % OVERTURE)

R_REAL = 6371008.8
SCALE = 5.0
TOY_DEG = R_REAL / SCALE * math.pi / 180.0  # toy m per degree of latitude (22,239)
KM_DEG = R_REAL * math.pi / 180.0 / 1000.0  # real km per degree

N_TOP = 200
SAME_CITY_KM = 25.0      # a smaller GeoNames place this close to a kept one is the same agglomeration
BOX_MARGIN_KM = 2.0
BOX_MIN_KM = 8.0
BOX_MAX_KM = 35.0
BEARING_R = 2000.0       # real m around the town point for the street bearing
MAJOR = ("motorway", "trunk", "primary", "secondary")
MINOR = ("tertiary", "residential", "unclassified", "living_street")
CLS = {"motorway": 0, "trunk": 1, "primary": 2, "secondary": 3}
ICONIC_CLS = 4
WIDTH = {0: 22, 1: 18, 2: 16, 3: 12, 4: 20}
SPACING = 300.0          # toy m between parallel primaries / secondaries
DUP = 25.0               # toy m: a parallel stroke this close is the other carriageway
MIN_RUN = 300.0          # toy m: shorter primary / secondary runs dropped
MIN_RUN_FAST = 120.0     # toy m: motorway / trunk / iconic
DP_TOL = 3.0             # toy m
ALIGN_DEG = 25.0         # a grid street the towns can snap onto runs within this of the grid's axes
ALIGN_MIN = 0.6          # primaries / secondaries (and short trunks) kept only when this share of them is
TRUNK_LONG = 2000.0      # toy m: a trunk this long is kept whatever its bearing
ICON_R = 4000.0          # toy m: named avenues only this close to the origin (Broadway in Brooklyn is not it)
ICON_SPACING = 160.0     # toy m between parallel named avenues
PAR_COS = 0.85           # "parallel": |cos| of the angle between the two directions


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------- inputs

def load_geonames():
    rows = []
    with zipfile.ZipFile(GEONAMES) as z:
        with z.open("cities5000.txt") as f:
            for line in io.TextIOWrapper(f, encoding="utf-8"):
                p = line.rstrip("\n").split("\t")
                rows.append({"gid": int(p[0]), "name": p[1], "ascii": p[2], "lat": float(p[4]), "lon": float(p[5]),
                             "fcode": p[7], "cc": p[8], "pop": int(p[14] or 0)})
    return rows


def km(a_lat, a_lon, b_lat, b_lon):
    dlat = (b_lat - a_lat)
    dlon = (b_lon - a_lon + 540.0) % 360.0 - 180.0
    return math.hypot(dlat, dlon * math.cos(math.radians((a_lat + b_lat) * 0.5))) * KM_DEG


def towns_bin_origins():
    """geonameid -> (lat, lon) of the TownData town (towns.bin), when the file exists."""
    if not os.path.exists(TOWNS_BIN):
        return {}
    with open(TOWNS_BIN, "rb") as f:
        b = f.read()
    if b[:4] != b"TWN1":
        return {}
    _ver, n, _nt, _ns, towns_off, _to, _no = struct.unpack_from("<7I", b, 4)
    out = {}
    for i in range(n):
        lat, lon, _pop, gid = struct.unpack_from("<iiII", b, towns_off + i * 32)
        out[gid] = (lat * 1e-6, lon * 1e-6)
    return out


def pick_cities(gn):
    by_pop = sorted(gn, key=lambda r: -r["pop"])
    chosen = []

    def covered(r):
        for c in chosen:
            if c["pop"] >= r["pop"] and km(c["lat"], c["lon"], r["lat"], r["lon"]) < SAME_CITY_KM:
                return True
        return False

    for r in by_pop:
        if len(chosen) >= N_TOP:
            break
        if r["fcode"] in ("PPLX", "PPLH", "PPLQ", "PPLW"):
            continue
        if not covered(r):
            chosen.append(dict(r, why="top"))
    have = {c["gid"] for c in chosen}

    def add(r, why):
        if r is not None and r["gid"] not in have:
            have.add(r["gid"])
            chosen.append(dict(r, why=why))

    def nearest(lat, lon, reach_km, min_pop, cc=None, name=None):
        best = None
        bd = 1e9
        for r in gn:
            if r["pop"] < min_pop or (cc and r["cc"] != cc):
                continue
            if abs(r["lat"] - lat) > reach_km / 100.0:
                continue
            d = km(lat, lon, r["lat"], r["lon"])
            if name and (r["name"].lower() == name.lower() or r["ascii"].lower() == name.lower()):
                d -= 1000.0
            if d < reach_km and d < bd:
                bd = d
                best = r
        return best

    if os.path.exists(AUDIT_CITIES):
        for c in json.load(open(AUDIT_CITIES, encoding="utf-8")):
            r = nearest(c["lat"], c["lon"], 15.0, 50000, c.get("cc"), c.get("gn") or c["name"])
            add(r, "audit")
    for c in json.load(open(ICONIC, encoding="utf-8"))["cities"]:
        nm = c["city"].lower()
        hits = [r for r in gn if r["cc"] == c["cc"] and (r["name"].lower() == nm or r["ascii"].lower() == nm)]
        if not hits:
            log("  iconic city not found:", c["city"], c["cc"])
            continue
        add(max(hits, key=lambda r: r["pop"]), "iconic")
    for p in sorted(glob.glob(os.path.join(LANDMARKS, "*.json"))):
        try:
            d = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        if "lat" not in d or d.get("peak") or d.get("kind") == "peak":
            continue
        r = nearest(float(d["lat"]), float(d["lon"]), 30.0, 100000)
        if r is not None and not any(km(c["lat"], c["lon"], r["lat"], r["lon"]) < SAME_CITY_KM for c in chosen):
            add(r, "landmark")
    return chosen


# ---------------------------------------------------------------- GHS-UCDB boxes

def _inv_mollweide(x, y):
    R = 6378137.0
    th = np.arcsin(np.clip(y / (math.sqrt(2.0) * R), -1.0, 1.0))
    lat = np.degrees(np.arcsin(np.clip((2.0 * th + np.sin(2.0 * th)) / math.pi, -1.0, 1.0)))
    lon = np.degrees(math.pi * x / (2.0 * math.sqrt(2.0) * R * np.maximum(np.cos(th), 1e-9)))
    return lon, lat


def _fwd_mollweide(lon, lat):
    R = 6378137.0
    phi = math.radians(lat)
    th = phi
    for _ in range(30):
        f = 2 * th + math.sin(2 * th) - math.pi * math.sin(phi)
        d = 2 + 2 * math.cos(2 * th)
        if abs(d) < 1e-12:
            break
        th -= f / d
    return 2 * math.sqrt(2) / math.pi * R * math.radians(lon) * math.cos(th), math.sqrt(2) * R * math.sin(th)


def ucdb_boxes(cities):
    """gid -> (lon0, lat0, lon1, lat1) of the urban centre holding the town point (None: none)."""
    out = {c["gid"]: None for c in cities}
    if not os.path.exists(UCDB):
        log("  no GHS-UCDB, fixed boxes")
        return out
    import shapely.wkb
    from shapely.geometry import Point
    con = sqlite3.connect("file:%s?mode=ro" % UCDB.replace("\\", "/"), uri=True)
    rows = con.execute("SELECT geom FROM GHS_UCDB_THEME_GENERAL_CHARACTERISTICS_GLOBE_R2024A").fetchall()
    polys = []
    for (g,) in rows:
        flags = g[3]
        env = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}.get((flags >> 1) & 7, 0)
        geom = shapely.wkb.loads(bytes(g[8 + env:]))
        polys.append(geom)
    from shapely.strtree import STRtree
    tree = STRtree(polys)
    for c in cities:
        x, y = _fwd_mollweide(c["lon"], c["lat"])
        pt = Point(x, y)
        cand = tree.query(pt.buffer(5000.0))
        best = None
        bd = 1e18
        for k in cand:
            d = polys[k].distance(pt)
            if d < bd:
                bd = d
                best = polys[k]
        if best is None or bd > 5000.0:
            continue
        xs, ys = [], []
        for part in getattr(best, "geoms", [best]):
            a = np.asarray(part.exterior.coords)
            xs.append(a[:, 0])
            ys.append(a[:, 1])
        lon, lat = _inv_mollweide(np.concatenate(xs), np.concatenate(ys))
        out[c["gid"]] = (float(lon.min()), float(lat.min()), float(lon.max()), float(lat.max()))
    return out


def query_box(c, ub):
    lat0, lon0 = c["lat"], c["lon"]
    kx = KM_DEG * math.cos(math.radians(lat0))
    if ub is None:
        e = w = n = s = BOX_MIN_KM + 4.0
    else:
        w = (lon0 - ub[0]) * kx + BOX_MARGIN_KM
        e = (ub[2] - lon0) * kx + BOX_MARGIN_KM
        s = (lat0 - ub[1]) * KM_DEG + BOX_MARGIN_KM
        n = (ub[3] - lat0) * KM_DEG + BOX_MARGIN_KM
    cl = lambda v: min(max(v, BOX_MIN_KM), BOX_MAX_KM)
    return (lon0 - cl(w) / kx, lat0 - cl(s) / KM_DEG, lon0 + cl(e) / kx, lat0 + cl(n) / KM_DEG)


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
            _con.execute("SET s3_region='us-west-2'; SET enable_http_metadata_cache=true; SET parquet_metadata_cache=true;")
        return _con.cursor()


def _sql_str(s):
    return "'" + s.replace("'", "''") + "'"


def fetch(c, box, iconic_names):
    path = os.path.join(CACHE, "%d.parquet" % c["gid"])
    if os.path.exists(path):
        return path, 0.0
    t0 = time.time()
    lon0, lat0, lon1, lat1 = box
    br = BEARING_R / 1000.0 / KM_DEG
    brx = br / max(0.05, math.cos(math.radians(c["lat"])))
    cond_icon = "FALSE"
    if iconic_names:
        parts = []
        for nm in iconic_names:
            if "%" in nm:
                parts.append("names.primary ILIKE %s" % _sql_str(nm))
            else:
                parts.append("lower(names.primary) = lower(%s)" % _sql_str(nm))
        cond_icon = "(" + " OR ".join(parts) + ")"
    q = f"""
    COPY (
      SELECT id, class, subclass, names.primary AS name, ST_AsWKB(geometry) AS wkb
      FROM read_parquet('{SEGMENTS}', hive_partitioning=0)
      WHERE bbox.xmin < {lon1} AND bbox.xmax > {lon0} AND bbox.ymin < {lat1} AND bbox.ymax > {lat0}
        AND subtype = 'road'
        AND (class IN {MAJOR}
             OR {cond_icon}
             OR (class IN {MINOR} AND bbox.xmin < {c['lon'] + brx} AND bbox.xmax > {c['lon'] - brx}
                 AND bbox.ymin < {c['lat'] + br} AND bbox.ymax > {c['lat'] - br}))
    ) TO '{path.replace(chr(92), '/')}.tmp' (FORMAT parquet)"""
    cur = _duck()
    cur.execute(q)
    cur.close()
    os.replace(path + ".tmp", path)
    return path, time.time() - t0


# ---------------------------------------------------------------- generalise

def to_toy(coords, lat0, lon0):
    a = np.asarray(coords, dtype=np.float64)
    dlon = (a[:, 0] - lon0 + 540.0) % 360.0 - 180.0
    return np.stack([dlon * TOY_DEG * math.cos(math.radians(lat0)), (a[:, 1] - lat0) * TOY_DEG], axis=1)


def strokes(segs):
    """Chain segments of one (class, name) group into long strokes through their
    shared ends (smallest turn, under 50 deg). segs: list of (N, 2) arrays."""
    key = lambda p: (round(p[0] * 2.0), round(p[1] * 2.0))  # 0.5 toy m
    ends = {}
    for i, s in enumerate(segs):
        ends.setdefault(key(s[0]), []).append((i, 0))
        ends.setdefault(key(s[-1]), []).append((i, 1))
    used = [False] * len(segs)
    out = []

    def dir_out(s, at_end):
        if at_end:
            d = s[-1] - s[max(0, len(s) - 3)]
        else:
            d = s[0] - s[min(len(s) - 1, 2)]
        n = np.hypot(*d)
        return d / n if n > 1e-9 else d

    def extend(line, at_end):
        while True:
            tip = line[-1] if at_end else line[0]
            d0 = dir_out(line, at_end)
            best = None
            bc = math.cos(math.radians(50.0))
            for (j, e) in ends.get(key(tip), []):
                if used[j]:
                    continue
                s = segs[j]
                # continuing into segment j from its end e
                d1 = -dir_out(s, e == 1)
                c = float(np.dot(d0, d1))
                if c > bc:
                    bc = c
                    best = (j, e)
            if best is None:
                return line
            j, e = best
            used[j] = True
            if at_end:
                # oriented to start at the tip
                s = segs[j] if e == 0 else segs[j][::-1]
                line = np.vstack([line, s[1:]])
            else:
                # oriented to end at the tip
                s = segs[j][::-1] if e == 0 else segs[j]
                line = np.vstack([s[:-1], line])

    for i in range(len(segs)):
        if used[i]:
            continue
        used[i] = True
        line = segs[i]
        line = extend(line, True)
        line = extend(line, False)
        out.append(line)
    return out


class Cover:
    """Kept road samples on a grid: is a point near a parallel kept road?"""

    def __init__(self, cell):
        self.cell = cell
        self.g = {}

    def add(self, pts, dirs):
        for p, d in zip(pts, dirs):
            k = (int(math.floor(p[0] / self.cell)), int(math.floor(p[1] / self.cell)))
            self.g.setdefault(k, []).append((p[0], p[1], d[0], d[1]))

    def near(self, p, d, dist):
        cx = int(math.floor(p[0] / self.cell))
        cy = int(math.floor(p[1] / self.cell))
        r = int(math.ceil(dist / self.cell))
        d2 = dist * dist
        for i in range(cx - r, cx + r + 1):
            for j in range(cy - r, cy + r + 1):
                for (x, y, dx, dy) in self.g.get((i, j), ()):
                    if (x - p[0]) ** 2 + (y - p[1]) ** 2 <= d2 and abs(dx * d[0] + dy * d[1]) >= PAR_COS:
                        return True
        return False


def resample(line, step):
    seg = np.diff(line, axis=0)
    l = np.hypot(seg[:, 0], seg[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(l)])
    total = cum[-1]
    if total < 1e-6:
        return line[:1], np.array([[1.0, 0.0]]), np.array([0.0])
    s = np.arange(0.0, total + 1e-9, step)
    if s[-1] < total - 1e-6:
        s = np.append(s, total)
    k = np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(l) - 1)
    t = (s - cum[k]) / np.maximum(l[k], 1e-9)
    pts = line[k] + seg[k] * t[:, None]
    dirs = seg[k] / np.maximum(l[k], 1e-9)[:, None]
    return pts, dirs, s


def runs_free(line, cover, dist, min_run, step=20.0):
    """Pieces of a line (arrays of points) that lie farther than dist from any
    parallel kept road, each at least min_run long."""
    pts, dirs, s = resample(line, step)
    free = [not cover.near(p, d, dist) for p, d in zip(pts, dirs)]
    out = []
    i = 0
    n = len(pts)
    while i < n:
        if not free[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and free[j + 1]:
            j += 1
        if s[j] - s[i] >= min_run:
            out.append(pts[i:j + 1])
        i = j + 1
    return out


def simplify(line, tol):
    from shapely.geometry import LineString
    if len(line) <= 2:
        return line
    g = LineString(line).simplify(tol, preserve_topology=False)
    return np.asarray(g.coords)


def aligned(line, bearing):
    """Length share of a line within ALIGN_DEG of the street grid's axes."""
    if math.isnan(bearing):
        return 1.0
    d = np.diff(line, axis=0)
    l = np.hypot(d[:, 0], d[:, 1])
    a = (np.degrees(np.arctan2(d[:, 1], d[:, 0])) - bearing) % 90.0
    off = np.minimum(a, 90.0 - a)
    return float(l[off <= ALIGN_DEG].sum() / max(l.sum(), 1e-9))


def bearing_mode(segs_minor_major):
    """Length-weighted mode of segment bearings mod 90 (deg from east, ccw)."""
    hist = np.zeros(90)
    for line in segs_minor_major:
        d = np.diff(line, axis=0)
        l = np.hypot(d[:, 0], d[:, 1])
        a = np.degrees(np.arctan2(d[:, 1], d[:, 0])) % 90.0
        np.add.at(hist, np.floor(a).astype(int) % 90, l)
    if hist.sum() <= 0:
        return float("nan"), 0.0
    k = np.exp(-0.5 * (np.arange(-6, 7) / 3.0) ** 2)
    sm = np.zeros(90)
    for o, w in zip(range(-6, 7), k):
        sm += w * np.roll(hist, o)
    i = int(np.argmax(sm))
    # sub-degree peak: weighted mean over +-3 deg (circular)
    idx = np.arange(i - 3, i + 4)
    w = hist[idx % 90]
    peak = float((idx * w).sum() / max(w.sum(), 1e-9) + 0.5) % 90.0
    share = float(sm[i] / max(sm.sum(), 1e-9))
    return peak, share


def clip_box(c):
    """The city's query box in toy m around its origin (shapely box)."""
    from shapely.geometry import box
    lon0, lat0, lon1, lat1 = c["box"]
    a = to_toy([(lon0, lat0), (lon1, lat1)], c["olat"], c["olon"])
    return box(a[0, 0], a[0, 1], a[1, 0], a[1, 1])


def bake_city(c, path, iconic_names):
    import pyarrow.parquet as pq
    import shapely.wkb
    from shapely.geometry import LineString
    t = pq.read_table(path).to_pydict()
    lat0, lon0 = c["olat"], c["olon"]
    clip = clip_box(c)
    icon = {n.lower() for n in iconic_names if "%" not in n}
    icon_pat = [n.lower().replace("%", "") for n in iconic_names if "%" in n]
    groups = {}
    bear = []
    rb = BEARING_R / SCALE
    for cls, sub, name, wkb in zip(t["class"], t["subclass"], t["name"], t["wkb"]):
        g = shapely.wkb.loads(bytes(wkb))
        parts = getattr(g, "geoms", [g])
        for part in parts:
            a = to_toy(part.coords, lat0, lon0)
            if len(a) < 2:
                continue
            near = np.hypot(a[:, 0], a[:, 1]).min() < rb
            if near and cls in MAJOR + MINOR:
                bear.append(a)
            if sub == "link":
                continue
            nm = (name or "")
            is_icon = nm.lower() in icon or any(p in nm.lower() for p in icon_pat)
            if is_icon:
                k = (ICONIC_CLS, nm)
            elif cls in CLS:
                k = (CLS[cls], nm if cls in ("primary", "secondary") or nm else "")
            else:
                continue
            groups.setdefault(k, []).append(a)
    bearing, share = bearing_mode(bear)
    # strokes, by class then length
    cands = []
    for (cl, nm), segs in groups.items():
        for st in strokes(segs):
            L = float(np.hypot(*np.diff(st, axis=0).T).sum())
            cands.append((cl, nm, L, st))
    order = {0: 0, 1: 1, ICONIC_CLS: 2, 2: 3, 3: 4}
    irank = {n.lower(): k for k, n in enumerate(iconic_names)}
    cands.sort(key=lambda x: (order[x[0]], irank.get(x[1].lower(), 99) if x[0] == ICONIC_CLS else 0, -x[2]))
    cover = Cover(SPACING)
    dup = Cover(DUP * 2.0)
    lines = []
    for cl, nm, L, st in cands:
        if cl == 1 and L < TRUNK_LONG and aligned(st, bearing) < ALIGN_MIN:
            continue  # a short trunk across the grid: a primary
        if cl in (2, 3):
            if L < MIN_RUN or aligned(st, bearing) < ALIGN_MIN:
                continue
            pieces = []
            for r in runs_free(st, cover, SPACING, MIN_RUN):
                pieces += runs_free(r, dup, DUP, MIN_RUN)
        elif cl == ICONIC_CLS:
            # named avenues: central ones only, and parallel ones apart (the
            # 1/5 planet fits one or two of Manhattan's avenues, not eight)
            if float(np.hypot(st[:, 0], st[:, 1]).min()) > ICON_R:
                continue
            pieces = []
            for r in runs_free(st, cover, ICON_SPACING, MIN_RUN_FAST):
                pieces += runs_free(r, dup, DUP, MIN_RUN_FAST)
        else:
            pieces = runs_free(st, dup, DUP, MIN_RUN_FAST)
        for p in pieces:
            pts, dirs, _ = resample(p, 20.0)
            cover.add(pts, dirs)
            dup.add(pts, dirs)
            g = LineString(p).intersection(clip)
            for part in getattr(g, "geoms", [g]):
                if part.is_empty or part.geom_type != "LineString" or part.length < 60.0:
                    continue
                sp = simplify(np.asarray(part.coords), DP_TOL)
                lines.append((cl, nm if cl == ICONIC_CLS else "", sp))
    radius = 0.0
    for _cl, _nm, sp in lines:
        radius = max(radius, float(np.hypot(sp[:, 0], sp[:, 1]).max()))
    return {"bearing": bearing, "share": share, "lines": lines, "radius": radius, "n_seg": len(t["class"])}


# ---------------------------------------------------------------- write

def write(cities, out_path):
    names = []
    name_id = {}
    blob = bytearray()
    recs = []
    cities = sorted(cities, key=lambda c: (c["olat"], c["olon"]))
    head = 24
    city_sz = 40
    lines_off = head + city_sz * len(cities)
    for c in cities:
        r = c["res"]
        off = lines_off + len(blob)
        n = 0
        for cl, nm, sp in r["lines"]:
            q = np.clip(np.round(sp), -32767, 32767).astype(np.int16)
            # drop repeated points after rounding
            keep = [0] + [i for i in range(1, len(q)) if (q[i] != q[i - 1]).any()]
            q = q[keep]
            if len(q) < 2:
                continue
            nid = 0xFFFF
            if nm:
                if nm not in name_id:
                    name_id[nm] = len(names)
                    names.append(nm)
                nid = name_id[nm]
            w = WIDTH[cl] if cl != ICONIC_CLS else max(WIDTH[ICONIC_CLS], 20)
            blob += struct.pack("<BBHH", cl, w, nid, len(q))
            blob += q.astype("<i2").tobytes()
            n += 1
        recs.append(struct.pack("<IddffIII", c["gid"], c["olat"], c["olon"], r["bearing"], r["radius"], off, n, 0))
    names_off = lines_off + len(blob)
    nb = b"".join(s.encode("utf-8") + b"\0" for s in names)
    with open(out_path + ".tmp", "wb") as f:
        f.write(b"RDN1" + struct.pack("<5I", 1, len(cities), head, names_off, len(names)))
        for r in recs:
            f.write(r)
        f.write(blob)
        f.write(nb)
    os.replace(out_path + ".tmp", out_path)
    return os.path.getsize(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fetch", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    os.makedirs(CACHE, exist_ok=True)
    t0 = time.time()
    gn = load_geonames()
    cities = pick_cities(gn)
    log("%d cities (%s)" % (len(cities), ", ".join("%s %d" % (w, sum(1 for c in cities if c["why"] == w))
                                                     for w in ("top", "audit", "iconic", "landmark"))))
    only = {s.strip().lower() for s in a.only.split(",") if s.strip()}
    if only:
        cities = [c for c in cities if c["name"].lower() in only or c["ascii"].lower() in only]
    if a.list:
        for c in cities:
            log("%-9d %-28s %s %9d %s" % (c["gid"], c["name"], c["cc"], c["pop"], c["why"]))
        return
    origins = towns_bin_origins()
    for c in cities:
        o = origins.get(c["gid"])
        c["olat"], c["olon"] = (o if o else (c["lat"], c["lon"]))
    icon = {}
    for e in json.load(open(ICONIC, encoding="utf-8"))["cities"]:
        icon[(e["city"].lower(), e["cc"])] = e["names"]
    for c in cities:
        c["icon"] = icon.get((c["name"].lower(), c["cc"]), icon.get((c["ascii"].lower(), c["cc"]), []))
    boxes = ucdb_boxes(cities)
    for c in cities:
        c["box"] = query_box(c, boxes[c["gid"]])
    if not a.no_fetch:
        todo = [c for c in cities if not os.path.exists(os.path.join(CACHE, "%d.parquet" % c["gid"]))]
        log("fetching %d cities from Overture %s" % (len(todo), OVERTURE))
        with cf.ThreadPoolExecutor(a.jobs) as ex:
            futs = {ex.submit(fetch, c, c["box"], c["icon"]): c for c in todo}
            for k, f in enumerate(cf.as_completed(futs)):
                c = futs[f]
                try:
                    _p, dt = f.result()
                    log("  [%d/%d] %s %.0f s" % (k + 1, len(todo), c["name"], dt))
                except Exception as e:
                    log("  [%d/%d] %s FAILED %s" % (k + 1, len(todo), c["name"], e))
    done = []
    for c in cities:
        p = os.path.join(CACHE, "%d.parquet" % c["gid"])
        if not os.path.exists(p):
            log("  no cache for", c["name"])
            continue
        c["res"] = bake_city(c, p, c["icon"])
        r = c["res"]
        cnt = {}
        for cl, _n, _s in r["lines"]:
            cnt[cl] = cnt.get(cl, 0) + 1
        log("  %-24s seg %6d lines %4d %s bearing %5.1f (%.2f) r %5.0f" % (
            c["name"], r["n_seg"], len(r["lines"]), cnt, r["bearing"], r["share"], r["radius"]))
        done.append(c)
    if only:
        log("(--only: nothing written)")
        return
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    size = write(done, OUT)
    log("%d cities -> %s (%.2f MB) in %.0f s" % (len(done), OUT, size / 1e6, time.time() - t0))


if __name__ == "__main__":
    main()
