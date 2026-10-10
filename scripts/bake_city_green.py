"""Detailed cities, GREEN: real parks, gardens, woods and street trees from Overture Maps (base theme land_use and
land, OpenStreetMap, ODbL) baked into data/world/odbl/green/<region>.bin (CGR1, one file per region) plus its index
data/world/odbl/green/index.bin (RPX1, tools/region_pack.py) for scripts/world/city_green.gd.

    python tools/bake_city_green.py fetch [--only london,nyc]   download the region caches (DuckDB, S3)
    python tools/bake_city_green.py bake  [--only ...]           bake the regions' files, then the index
    python tools/bake_city_green.py info  [path]                 per-region counts (path: a folder, or an old
                                                                 single city_green.bin)
    python tools/bake_city_green.py index                         rebuild the index from the region files there

Bake the buildings first: a region's green reads its building files (data/world/odbl/buildings/, the cover cells
and the footprints the trees keep off). --only rewrites only those regions' files and the index. CITY_OUT: as
tools/bake_city_buildings.py (the output base: green/ next to buildings/).

Regions: the boxes of tools/bake_city_buildings.py (REGIONS; Israel clipped to its polygon).
Sources: Overture release OVERTURE, theme=base: type=land_use (park, garden, grass, cemetery, golf, recreation
ground, pitch, village green, nature reserve) and type=land (wood, forest, grass, meadow, scrub polygons; tree
points and tree rows), type=water (lakes and ponds inside the green). Caches data/world/src/overture/green/<region>_<type>.parquet (gitignored, never exported).

Scale: the game planet is 1/5 real size; a park keeps its real lat / lon outline (so it is 1/5 across, like the
city), its trees stay REAL size (like the buildings), so the bake thins the trees: a tree point is kept when no
kept one stands within TREE_GAP toy m; park interiors get scattered trees at the runtime (city_green.gd) where
the data has none.

CGR1 (little endian; version 2: one file per region, n_regions 1; version 1 was one file for every region):
  header 32 B : "CGR1", u32 version, u32 n_tiles, u32 n_regions, u32 regions_off, u32 0, u32 0, u32 0
  tiles  n x 32 B (sorted by key): u32 key ((ty * 360 * TPD) + tx, 1/TPD degree tiles from the south pole / -180),
           u32 off, u32 bytes (zlib), u32 n_poly, u32 n_pts, u32 n_trees, u32 0, u32 0
  regions n x 64 B: char[24] name, f32 lat0, lon0, lat1, lon1, u32 n_poly, u32 n_trees, u32 0, u32 0
  tile payload (zlib): polygons n_poly x 8 B: u32 first point, u16 n_pts, u8 kind, u8 flags;
           points n_pts x 4 B: u16 east, u16 north (TILE_Q of the tile's extent from its SW corner);
           trees n_trees x 4 B: u16 east, u16 north (same quantum)
  A polygon is the CLIPPED part inside its tile (rings simplified, holes dropped: a lake inside a park is the
  game's water), counter-clockwise in (east, north), not closed.
  kind: 0 park / garden / recreation, 1 grass / pitch / meadow, 2 wood / forest, 3 cemetery, 4 golf, 5 park lake / pond
"""
import argparse
import math
import os
import struct
import sys
import time
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import bake_city_buildings as cb  # noqa: E402
import region_pack  # noqa: E402

SRC = cb.SRC
CACHE = os.path.join(SRC, "overture", "green")
OUT_DIR = os.path.join(cb.OUT_BASE, "green")
MAGIC = b"CGR1"
OVERTURE = cb.OVERTURE
BASE = "s3://overturemaps-us-west-2/release/%s/theme=base/type=%%s/*.parquet" % OVERTURE
TPD = 32
VERSION = 2
CRLF = chr(13) + chr(10)

LU_SUBTYPES = ("park", "recreation", "cemetery", "horticulture", "managed", "protected", "agriculture")
LAND_SUBTYPES = ("forest", "grass", "tree", "shrub")
WATER_SUBTYPES = ("lake", "pond", "water", "reservoir")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def cache_path(name, typ):
    return os.path.join(CACHE, "%s_%s.parquet" % (name, typ))


def fetch(names):
    import duckdb
    os.makedirs(CACHE, exist_ok=True)
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs; INSTALL spatial; LOAD spatial;")
    con.execute("SET s3_region='us-west-2'; SET enable_http_metadata_cache=true; SET parquet_metadata_cache=true;")
    for name in names:
        la0, lo0, la1, lo1 = cb.REGIONS[name]["box"]
        for typ, subs in (("land_use", LU_SUBTYPES), ("land", LAND_SUBTYPES), ("water", WATER_SUBTYPES)):
            path = cache_path(name, typ)
            if os.path.exists(path):
                log("fetch %s %s: cached (%.1f MB)" % (name, typ, os.path.getsize(path) / 1e6))
                continue
            t0 = time.time()
            sl = ", ".join("'%s'" % s for s in subs)
            q = f"""
            COPY (
              SELECT id, subtype, class, names.primary AS name, ST_AsWKB(geometry) AS wkb
              FROM read_parquet('{BASE % typ}', hive_partitioning = 0)
              WHERE bbox.xmax >= {lo0} AND bbox.xmin <= {lo1} AND bbox.ymax >= {la0} AND bbox.ymin <= {la1}
                AND subtype IN ({sl})
            ) TO '{path.replace(chr(92), '/')}.tmp' (FORMAT parquet, COMPRESSION zstd)"""
            con.execute(q)
            os.replace(path + ".tmp", path)
            n = con.execute("SELECT count(*) FROM read_parquet('%s')" % path.replace(chr(92), '/')).fetchone()[0]
            log("fetch %s %s: %d features, %.1f MB, %.0f s" % (name, typ, n, os.path.getsize(path) / 1e6, time.time() - t0))


# ---------------------------------------------------------------- classification

K_PARK, K_GRASS, K_WOOD, K_CEM, K_GOLF, K_POND = 0, 1, 2, 3, 4, 5
POND_MIN_M2 = 1500.0       # a park lake or pond (city_water.bin draws the rivers and the big lakes only)
# class -> (kind, min real m2); everything else is not drawn
LU_CLASS = {
    "park": (K_PARK, 5000.0), "village_green": (K_PARK, 2000.0), "recreation_ground": (K_PARK, 4000.0),
    "garden": (K_PARK, 20000.0), "common": (K_PARK, 4000.0),
    "grass": (K_GRASS, 12000.0), "pitch": (K_GRASS, 8000.0),
    "cemetery": (K_CEM, 4000.0), "grave_yard": (K_CEM, 4000.0),
    "golf_course": (K_GOLF, 20000.0),
}
LAND_CLASS = {
    "wood": (K_WOOD, 3000.0), "forest": (K_WOOD, 3000.0),
    "grass": (K_GRASS, 12000.0), "grassland": (K_GRASS, 12000.0), "heath": (K_GRASS, 8000.0),
}
PRIORITY = [K_POND, K_WOOD, K_CEM, K_GOLF, K_GRASS, K_PARK]   # an overlap goes to the first
SIMPLIFY_M = 12.0         # real m (2.4 toy m)
COVER_GROW = 8            # cover cells (1/240 deg) round the city's buildings: green only there
COVER_GROW_REGION = {"israel": 3}   # (Israel: the whole country is one region, its towns close together)
# a wood or park bigger than this is countryside (Forest draws the forests from the tree cover), by region
MAX_M2 = {"israel": {K_WOOD: 4.0e5, K_PARK: 3.0e6}, "*": {K_WOOD: 3.0e6, K_PARK: 1.2e7}}
MASK_N = 32               # park mask cells per tile side
# trees (toy m gaps between kept trees; scatter density in trees per toy m2 before the gap)
TREE_GAP_STREET = 11.0
TREE_GAP = {K_POND: 99.0, K_PARK: 11.0, K_GRASS: 22.0, K_WOOD: 8.0, K_CEM: 11.0, K_GOLF: 20.0}
SCATTER = {K_POND: 0.0, K_PARK: 1.0 / 200.0, K_GRASS: 1.0 / 2600.0, K_WOOD: 1.0 / 110.0, K_CEM: 1.0 / 220.0, K_GOLF: 1.0 / 900.0}
BLD_CLEAR = 2.5           # toy m: a tree keeps this far off a kept building footprint
T_STREET, T_PARK, T_WOOD = 0, 1, 2
M_PER_DEG = cb.M_PER_DEG  # toy m per degree


def building_files(name):
    """The building region files (CBL1) whose tiles reach within a tile of the region's box: its own and any
    neighbour's (the old single file gave every region's cells and footprints)."""
    la0, lo0, la1, lo1 = cb.REGIONS[name]["box"]
    ty0, tx0 = int(math.floor((la0 + 90.0) * TPD)) - 1, int(math.floor((lo0 + 180.0) * TPD)) - 1
    ty1, tx1 = int(math.floor((la1 + 90.0) * TPD)) + 1, int(math.floor((lo1 + 180.0) * TPD)) + 1
    out = []
    for m in region_pack.read_index(cb.OUT_DIR):
        if m["t0"][0] <= ty1 and m["t1"][0] >= ty0 and m["t0"][1] <= tx1 and m["t1"][1] >= tx0:
            out.append(open(region_pack.region_path(cb.OUT_DIR, m["name"]), "rb").read())
    if name not in [m["name"] for m in region_pack.read_index(cb.OUT_DIR)]:
        raise SystemExit("%s: no building file in %s (bake_city_buildings.py bake --only %s first)" % (name, cb.OUT_DIR, name))
    return out


def cover_cells(files):
    """The CBL1 cover cells (sorted keys) of the building files."""
    import numpy as np
    cells = []
    for b in files:
        magic, ver, nt, nr, roff, coff, nc, _ = struct.unpack_from("<4sIIIIIII", b, 0)
        cells.append(np.frombuffer(b, "<u4", nc, coff).astype(np.int64))
    return np.unique(np.concatenate(cells)) if cells else np.zeros(0, np.int64)


def building_rings(files):
    """{tile key: [ring (n, 2) lat / lon]} of every kept building in the building files."""
    out = {}
    for b in files:
        for key, rings in _rings(b).items():
            out.setdefault(key, []).extend(rings)
    return out


def _rings(b):
    import numpy as np
    magic, ver, nt, nr, roff, coff, nc, _ = struct.unpack_from("<4sIIIIIII", b, 0)
    out = {}
    for i in range(nt):
        key, off, nb, n, npt = struct.unpack_from("<IIIII", b, 32 + 32 * i)
        p = zlib.decompress(b[off:off + nb])
        ty, tx = key // (360 * TPD), key % (360 * TPD)
        lat_t, lon_t = ty / TPD - 90.0, tx / TPD - 180.0
        cos_t = math.cos(math.radians(lat_t + 0.5 / TPD))
        pts = np.frombuffer(p, "<i2", npt * 2, n * 12).reshape(-1, 2).astype(np.float64) * cb.Q
        rings = []
        for k in range(n):
            first, np_ = struct.unpack_from("<HB", p, 12 * k)
            q = pts[first:first + np_]
            rings.append(np.column_stack([lat_t + q[:, 1] / M_PER_DEG, lon_t + q[:, 0] / (M_PER_DEG * cos_t)]))
        out[key] = rings
    return out


def load_region(name, cover):
    """Green polygons by kind (lon / lat shapely geometries, overlaps resolved), tree points (n, 3) lat, lon, type,
    and the city area."""
    import duckdb
    import numpy as np
    import shapely
    cfg = cb.REGIONS[name]
    la0, lo0, la1, lo1 = cfg["box"]
    clip = shapely.Polygon(cfg["poly"]) if "poly" in cfg else shapely.box(lo0, la0, lo1, la1)
    # the city: cover cells of the region's buildings grown by COVER_GROW
    W = 360 * cb.COVER_PD
    gy = cover // W
    gx = cover % W
    inb = (gy >= (la0 + 90) * cb.COVER_PD - 2) & (gy <= (la1 + 90) * cb.COVER_PD + 2) & \
          (gx >= (lo0 + 180) * cb.COVER_PD - 2) & (gx <= (lo1 + 180) * cb.COVER_PD + 2)
    gy, gx = gy[inb], gx[inb]
    d = 1.0 / cb.COVER_PD
    cells = shapely.box(gx * d - 180.0, gy * d - 90.0, (gx + 1) * d - 180.0, (gy + 1) * d - 90.0)
    city = shapely.union_all(cells).buffer(COVER_GROW_REGION.get(name, COVER_GROW) * d, quad_segs=2)
    city = city.intersection(clip)
    lat_ref = (la0 + la1) * 0.5
    kx = 111320.0 * math.cos(math.radians(lat_ref))   # real m per degree of longitude
    ky = 111320.0
    by_kind = {k: [] for k in PRIORITY}
    con = duckdb.connect()
    trees = []
    for typ, table in (("land_use", LU_CLASS), ("land", LAND_CLASS)):
        path = cache_path(name, typ).replace("\\", "/")
        tb = con.execute("SELECT subtype, class, wkb FROM read_parquet('%s')" % path).fetchnumpy()
        cls = np.asarray(tb["class"], dtype=object)
        sub = np.asarray(tb["subtype"], dtype=object)
        geoms = shapely.from_wkb(np.array([bytes(w) for w in tb["wkb"]], dtype=object))
        gt = shapely.get_type_id(geoms)
        if typ == "land":
            # tree points, tree rows (a point every 10 real m)
            pt = (sub == "tree") & (gt == 0)
            xy = shapely.get_coordinates(geoms[pt])
            trees += [(float(y), float(x), T_STREET) for x, y in xy]
            rows = geoms[(sub == "tree") & (gt == 1)]
            if len(rows):
                xy = shapely.get_coordinates(shapely.segmentize(rows, 10.0 / ky))
                trees += [(float(y), float(x), T_STREET) for x, y in xy]
        poly = (gt == 3) | (gt == 6)
        for i in np.nonzero(poly)[0]:
            c = cls[i]
            if c not in table:
                continue
            kind, min_m2 = table[c]
            g = geoms[i]
            am2 = shapely.area(g) * kx * ky
            if am2 < min_m2 or am2 > MAX_M2.get(name, MAX_M2["*"]).get(kind, 1e12):
                continue
            by_kind[kind].append(g)
    # park lakes and ponds: water polygons that lie in the green (the Serpentine, the Round Pond)
    tb = con.execute("SELECT wkb FROM read_parquet('%s')" % cache_path(name, "water").replace(chr(92), "/")).fetchnumpy()
    wg = shapely.from_wkb(np.array([bytes(w) for w in tb["wkb"]], dtype=object))
    wg = wg[np.isin(shapely.get_type_id(wg), [3, 6])]
    wg = wg[shapely.area(wg) * kx * ky >= POND_MIN_M2]
    # (a park's polygon often has its lake as a hole: the green with its holes filled)
    land = [g for k in (K_PARK, K_GRASS, K_WOOD, K_CEM, K_GOLF) for g in by_kind[k]]
    if len(wg) and land:
        lp = shapely.get_parts(shapely.make_valid(np.array(land, dtype=object)))
        lp = lp[shapely.get_type_id(lp) == 3]
        filled = shapely.polygons(shapely.get_exterior_ring(lp))
        lt = shapely.STRtree(filled)
        hit = lt.query(shapely.point_on_surface(wg), predicate="within")
        if hit.size:
            by_kind[K_POND] = list(wg[np.unique(hit[0])])
    log("%s: %d tree points, polygons %s" % (name, len(trees), {k: len(v) for k, v in by_kind.items()}))
    # union per kind, simplify, clip to the city; an overlap goes to the higher priority kind
    tol = SIMPLIFY_M / ky
    taken = None
    out = {}
    for k in PRIORITY:
        if not by_kind[k]:
            continue
        u = shapely.union_all(shapely.make_valid(np.array(by_kind[k], dtype=object)))
        u = shapely.make_valid(shapely.simplify(u, tol, preserve_topology=True))
        u = u.intersection(city)
        if taken is not None:
            u = u.difference(taken)
        u = shapely.make_valid(u)
        parts = shapely.get_parts(u)
        parts = parts[shapely.get_type_id(parts) == 3]
        parts = parts[shapely.area(parts) * kx * ky >= 1200.0]   # slivers go
        if len(parts) == 0:
            continue
        g = shapely.MultiPolygon(list(parts)) if len(parts) > 1 else parts[0]
        out[k] = g
        taken = g if taken is None else shapely.union(taken, g)
    tr = np.array(trees, np.float64).reshape(-1, 3)
    if len(tr):
        tr = tr[shapely.contains_xy(city, tr[:, 1], tr[:, 0])]
    if len(tr) and K_POND in out:
        tr = tr[~shapely.contains_xy(out[K_POND], tr[:, 1], tr[:, 0])]
    return out, tr, city


def scatter_trees(green, seed):
    """Random tree candidates inside the green polygons by kind: [(lat, lon, type, kind)]."""
    import numpy as np
    import shapely
    rng = np.random.default_rng(seed)
    out = []
    for k, g in sorted(green.items()):
        for poly in shapely.get_parts(g):
            x0, y0, x1, y1 = poly.bounds
            lat_c = (y0 + y1) * 0.5
            kx = M_PER_DEG * math.cos(math.radians(lat_c))
            area_toy = shapely.area(poly) * kx * M_PER_DEG
            n = rng.poisson(area_toy * SCATTER[k])
            if n == 0:
                continue
            m = int(n * (x1 - x0) * (y1 - y0) / max(shapely.area(poly), 1e-12) * 1.3) + 4
            xs = rng.uniform(x0, x1, m)
            ys = rng.uniform(y0, y1, m)
            ok = shapely.contains_xy(poly, xs, ys)
            xs, ys = xs[ok][:n], ys[ok][:n]
            t = T_WOOD if k == K_WOOD else T_PARK
            if k == K_POND:
                continue
            out += [(float(y), float(x), t, k) for x, y in zip(xs, ys)]
    return out


def thin(points, lat_ref):
    """Greedy: a point is kept when no kept one is within the smaller of the two gaps (toy m).
    points: [(lat, lon, type, gap)] in priority order."""
    kx = M_PER_DEG * math.cos(math.radians(lat_ref))
    C = 8.0
    grid = {}
    kept = []
    for (lat, lon, t, gap) in points:
        x = lon * kx
        y = lat * M_PER_DEG
        cx, cy = int(math.floor(x / C)), int(math.floor(y / C))
        r = int(math.ceil(gap / C))
        ok = True
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                for (px, py, pg) in grid.get((cx + dx, cy + dy), ()):
                    g = min(gap, pg)
                    if (px - x) ** 2 + (py - y) ** 2 < g * g:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            grid.setdefault((cx, cy), []).append((x, y, gap))
            kept.append((lat, lon, t))
    return kept


def bake_region(name, cover, brings):
    import numpy as np
    import shapely
    t0 = time.time()
    green, street, city = load_region(name, cover)
    la0, lo0, la1, lo1 = cb.REGIONS[name]["box"]
    lat_ref = (la0 + la1) * 0.5
    # trees: the real ones first (street trees and the parks' own), then the scatter, each with its gap
    allg = shapely.union_all(list(green.values())) if green else None
    pts = []
    if len(street):
        ig_all = shapely.contains_xy(allg, street[:, 1], street[:, 0]) if allg is not None else np.zeros(len(street), bool)
        for (lat, lon, _t), ig in zip(street, ig_all):
            pts.append((float(lat), float(lon), T_PARK if ig else T_STREET, TREE_GAP[K_PARK] if ig else TREE_GAP_STREET))
    sc = scatter_trees(green, sum(map(ord, name)))
    pts += [(lat, lon, t, TREE_GAP[k]) for (lat, lon, t, k) in sc]
    # off the kept buildings (BLD_CLEAR), tile by tile
    keep = np.ones(len(pts), bool)
    arr = np.array([(p[0], p[1]) for p in pts], np.float64).reshape(-1, 2)
    if len(arr):
        ty = np.floor((arr[:, 0] + 90.0) * TPD).astype(np.int64)
        tx = np.floor((arr[:, 1] + 180.0) * TPD).astype(np.int64)
        keys = ty * 360 * TPD + tx
        for key in np.unique(keys):
            sel = np.nonzero(keys == key)[0]
            kty, ktx = int(key) // (360 * TPD), int(key) % (360 * TPD)
            near = []
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    near += brings.get((kty + dy) * 360 * TPD + ktx + dx, [])
            if not near:
                continue
            lat_c = (kty + 0.5) / TPD - 90.0
            kxx = M_PER_DEG * math.cos(math.radians(lat_c))
            polys = [shapely.Polygon(np.column_stack([r[:, 1] * kxx, r[:, 0] * M_PER_DEG])) for r in near]
            tree = shapely.STRtree(polys)
            px = shapely.points(arr[sel, 1] * kxx, arr[sel, 0] * M_PER_DEG)
            hit = tree.query(px, predicate="dwithin", distance=BLD_CLEAR)
            if hit.size:
                keep[sel[np.unique(hit[0])]] = False
    n_cand = len(pts)
    pts = [p for p, k in zip(pts, keep) if k]
    trees = thin(pts, lat_ref)
    km2 = {k: round(shapely.area(g) * 111320.0 ** 2 * math.cos(math.radians(lat_ref)) / 1e6, 2) for k, g in green.items()}
    log("%s: green real km2 by kind %s, %d trees kept of %d candidates (%d off buildings, %d real points) (%.0f s)" % (
        name, km2, len(trees), n_cand, n_cand - len(pts), len(street), time.time() - t0))
    return {"name": name, "green": green, "trees": trees, "box": (la0, lo0, la1, lo1)}


def write_region(R):
    """Write OUT_DIR/<region>.bin (CGR1 v2, one region)."""
    import numpy as np
    import shapely
    results = [R]
    tiles = {}   # key -> {"tris": [(coords (3, 2) lon / lat, kind)], "trees": [(lat, lon, type)], "mask": bytes}
    regions = []
    d = 1.0 / TPD
    for R in results:
        n_poly = 0
        mine = set()
        for k, g in sorted(R["green"].items()):
            x0, y0, x1, y1 = g.bounds
            for ty in range(int(math.floor((y0 + 90.0) * TPD)), int(math.floor((y1 + 90.0) * TPD)) + 1):
                for tx in range(int(math.floor((x0 + 180.0) * TPD)), int(math.floor((x1 + 180.0) * TPD)) + 1):
                    box = shapely.box(tx * d - 180.0, ty * d - 90.0, (tx + 1) * d - 180.0, (ty + 1) * d - 90.0)
                    part = shapely.make_valid(g.intersection(box))
                    if part.is_empty:
                        continue
                    polys = [p for p in shapely.get_parts(part) if shapely.get_type_id(p) == 3 and p.area > 0]
                    if not polys:
                        continue
                    key = ty * 360 * TPD + tx
                    mine.add(key)
                    t = tiles.setdefault(key, {"tris": [], "trees": [], "mask": None})
                    for p in polys:
                        for tg in shapely.get_parts(shapely.constrained_delaunay_triangles(p)):
                            t["tris"].append((shapely.get_coordinates(tg)[:3], k))
                        n_poly += 1
        for (lat, lon, tt) in R["trees"]:
            key = int(math.floor((lat + 90.0) * TPD)) * 360 * TPD + int(math.floor((lon + 180.0) * TPD))
            mine.add(key)
            tiles.setdefault(key, {"tris": [], "trees": [], "mask": None})["trees"].append((lat, lon, tt))
        # the park mask: cells whose centre is green (any kind)
        allg = shapely.union_all(list(R["green"].values())) if R["green"] else None
        for key in mine:
            t = tiles[key]
            if allg is None or not t["tris"]:
                continue
            ty, tx = key // (360 * TPD), key % (360 * TPD)
            lat_t, lon_t = ty * d - 90.0, tx * d - 180.0
            u = (np.arange(MASK_N) + 0.5) / MASK_N * d
            xx, yy = np.meshgrid(lon_t + u, lat_t + u)
            m = shapely.contains_xy(allg, xx.ravel(), yy.ravel())
            t["mask"] = np.packbits(m.astype(np.uint8), bitorder="little").tobytes()
        la0, lo0, la1, lo1 = R["box"]
        regions.append((R["name"], la0, lo0, la1, lo1, n_poly, len(R["trees"])))
        log("%s: %d polygon parts, %d trees written" % (R["name"], n_poly, len(R["trees"])))
    keys = sorted(tiles)
    blobs = []
    recs = []
    n_tri_all = 0
    for key in keys:
        t = tiles[key]
        ty, tx = key // (360 * TPD), key % (360 * TPD)
        lat_t, lon_t = ty * d - 90.0, tx * d - 180.0
        vmap = {}
        verts = bytearray()
        tris = bytearray()
        kinds = bytearray()
        for (c, k) in t["tris"]:
            ids = []
            for (x, y) in c:
                qx = int(round(min(max((x - lon_t) / d, 0.0), 1.0) * 65535))
                qy = int(round(min(max((y - lat_t) / d, 0.0), 1.0) * 65535))
                v = vmap.get((qx, qy))
                if v is None:
                    v = len(vmap)
                    vmap[(qx, qy)] = v
                    verts += struct.pack("<HH", qx, qy)
                ids.append(v)
            if len(set(ids)) < 3:
                continue
            tris += struct.pack("<HHH", *ids)
            kinds.append(k)
        if len(vmap) > 65535:
            raise SystemExit("tile %d: %d vertices" % (key, len(vmap)))
        tb = bytearray()
        for (lat, lon, tt) in t["trees"]:
            qx = int(round(min(max((lon - lon_t) / d, 0.0), 1.0) * 65535))
            qy = int(round(min(max((lat - lat_t) / d, 0.0), 1.0) * 65535))
            tb += struct.pack("<HHB", qx, qy, tt)
        mask = t["mask"] if t["mask"] is not None else bytes(MASK_N * MASK_N // 8)
        nt = len(kinds)
        n_tri_all += nt
        blob = zlib.compress(bytes(mask) + bytes(verts) + bytes(tris) + bytes(kinds) + bytes(tb), 9)
        recs.append((key, len(blob), len(vmap), nt, len(t["trees"])))
        blobs.append(blob)
    n_t = len(keys)
    regions_off = 32 + 32 * n_t
    data_off = regions_off + 64 * len(regions)
    out = bytearray(struct.pack("<4sIIIIIII", MAGIC, VERSION, n_t, len(regions), regions_off, 0, 0, 0))
    pos = data_off
    for (key, nb, nv, nt, ntr) in recs:
        out += struct.pack("<IIIIIIII", key, pos, nb, nv, nt, ntr, 0, 0)
        pos += nb
    for (nm, la0, lo0, la1, lo1, n, ntr) in regions:
        out += struct.pack("<24sffffIIII8x", nm.encode()[:24], la0, lo0, la1, lo1, n, ntr, 0, 0)
    assert len(out) == data_off
    for b in blobs:
        out += b
    path = region_pack.region_path(OUT_DIR, R["name"])
    region_pack.write_atomic(path, bytes(out))
    log("wrote %s: %d tiles, %d triangles, %.2f MB" % (path, n_t, n_tri_all, len(out) / 1e6))


def write_sources():
    line = ("green/*.bin: Overture Maps base land_use and land release %s (OpenStreetMap parks, gardens, grass, "
            "woods, cemeteries and trees), python tools/bake_city_green.py, baked %s" % (OVERTURE, time.strftime("%Y-%m-%d")))
    lines = []
    if os.path.exists(cb.SOURCES_TXT):
        with open(cb.SOURCES_TXT, encoding="utf-8", newline="") as f:
            lines = [l for l in f.read().replace("\r\n", "\n").split("\n")
                     if l and not l.startswith(("city_green.bin:", "green/*.bin:"))]
    lines.append(line)
    with open(cb.SOURCES_TXT, "w", encoding="utf-8", newline="") as f:
        f.write(CRLF.join(lines) + CRLF)


def bake(names):
    for n in names:
        files = building_files(n)
        cover = cover_cells(files)
        brings = building_rings(files)
        log("%s: buildings %d cover cells, %d tiles (%d region files)" % (n, len(cover), len(brings), len(files)))
        write_region(bake_region(n, cover, brings))
    metas = region_pack.write_index(OUT_DIR, MAGIC, TPD, list(cb.REGIONS), None, log,
                                    {n: c.get("poly") for n, c in cb.REGIONS.items()})
    log("wrote %s: %d regions, %.1f MB in all" % (os.path.join(OUT_DIR, region_pack.INDEX), len(metas),
                                                 sum(m["bytes"] for m in metas) / 1e6))
    write_sources()
    info()


def info(path=None):
    """Per-region counts of a folder of region files (default OUT_DIR) or of an old single CGR1 v1 file."""
    path = path or OUT_DIR
    if os.path.isfile(path):
        files = [path]
    else:
        files = [region_pack.region_path(path, m["name"]) for m in region_pack.read_index(path)]
    tot = [0, 0, 0, 0]
    for fp in files:
        data = open(fp, "rb").read()
        magic, ver, nt, nr, roff, _a, _b, _c = struct.unpack_from("<4sIIIIIII", data, 0)
        tiles = {}
        for i in range(nt):
            key, off, nb, nv, ntri, ntr = struct.unpack_from("<IIIIII", data, 32 + 32 * i)
            tiles[key] = (ntri, ntr)
        for r in range(nr):
            nm, la0, lo0, la1, lo1, n, ntr, _x, _y = struct.unpack_from("<24sffffIIII8x", data, roff + 64 * r)
            if nr == 1:
                print("  %-12s %6d polygon parts %7d trees %5d tiles %7d triangles %6.2f MB" % (
                    nm.rstrip(b"\0").decode(), n, ntr, nt, sum(t[0] for t in tiles.values()), len(data) / 1e6))
            else:
                print("  %-12s %6d polygon parts %7d trees" % (nm.rstrip(b"\0").decode(), n, ntr))
            tot[0] += n
            tot[1] += ntr
        tot[2] += nt
        tot[3] += sum(t[0] for t in tiles.values())
        if nr != 1:
            print("  (%s: CGR1 v%d, %d regions in one file, %.2f MB)" % (fp, ver, nr, len(data) / 1e6))
    print("  %-12s %6d polygon parts %7d trees %5d tiles %7d triangles" % ("total", tot[0], tot[1], tot[2], tot[3]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fetch", "bake", "info", "index"])
    ap.add_argument("path", nargs="?", default=None)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    for n in a.only.split(","):
        if n and n not in cb.REGIONS:
            raise SystemExit("no region %r (tools/city_regions/<name>.json)" % n)
    names = [n for n in cb.REGIONS if not a.only or n in a.only.split(",")]
    if a.cmd == "fetch":
        fetch(names)
    elif a.cmd == "bake":
        bake(names)
    elif a.cmd == "index":
        region_pack.write_index(OUT_DIR, MAGIC, TPD, list(cb.REGIONS), None, log,
                                    {n: c.get("poly") for n, c in cb.REGIONS.items()})
        info()
    else:
        info(a.path)


if __name__ == "__main__":
    main()
