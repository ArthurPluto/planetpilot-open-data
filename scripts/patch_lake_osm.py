"""OpenStreetMap shores for a Natural Earth lake beside a detailed city
(docs/REAL_PLACEMENT.md section 13.21). The 1:10m Lake Ontario outline runs
straight past downtown Toronto: no Toronto Islands, no Inner Harbour, no Leslie
Spit or Port Lands, the shore 300 to 900 real m off. And bake_rivers drops a
lake's triangles over the global coast mask's sea (coast.bin), which holds
Ontario's deep middle (its DEM is lake-bed bathymetry under 0 m), while the
near terrain (the Copernicus detail tiles: lake, not ocean) is land there: a
field-covered hole 10 real km off downtown. Inside each PATCHES window the
lake's outline in data/world/lakes.bin is replaced by the Overture
(OpenStreetMap) lake polygon, run through the same outline rules as
tools/bake_rivers.py (Douglas-Peucker SIMPLIFY, two Chaikin passes, a finer
simplify), and the whole lake is triangulated again (earcut with its islands,
refine to MAX_EDGE, shore_refine); no triangle inside the window is dropped as
sea. Outside the window the Natural Earth outline and the sea drop stay; the
lake keeps its index (lake_field.bin refers to it) and its level.

Then re-bake the lake field tiles it reaches:
    python tools/bake_lake_field.py --lakes <index>

Licence: a patched lake record holds OpenStreetMap geometry (ODbL, through the
Overture Maps base.water release); data/world/odbl/SOURCES.txt says which.

Run:  CITY_SRC=C:/tmp/city_src python tools/patch_lake_osm.py [--dry-run]
      (reads the region-mode Overture water parquet of bake_city_water.py,
      CITY_SRC/water/overture/<file>; idempotent: outside the window the
      outline comes from data/world/src/ne_10m_lakes.geojson when present,
      else from the current record, which a patch leaves as it was there)
"""
import argparse
import glob
import math
import os
import struct
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import bake_rivers as br  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LAKES = os.path.join(ROOT, "data", "world", "lakes.bin")
SOURCES = os.path.join(ROOT, "data", "world", "odbl", "SOURCES.txt")
_CS = os.environ.get("CITY_SRC")
OV_DIR = os.path.join(_CS, "water", "overture") if _CS else os.path.join(ROOT, "data", "world", "src", "overture", "water")
RELEASE = "2026-09-23.0"  # bake_city_water.RELEASE (the parquet's release)
# name: the Overture water name; probe: a point inside the lake (picks the record); window: lat0, lon0, lat1, lon1;
# parquet: the file. Lake Ontario: the whole lake west of Kingston (the seam crosses open water at 76.3 W)
PATCHES = [
    {"name": "Lake Ontario", "probe": (43.58, -79.40), "window": (43.10, -79.95, 44.40, -76.30),
     "parquet": "r_toronto_43.40_-79.70_44.00_-79.00.parquet"},
]
# The lake field samples every 1/256 deg (87 toy m): water channels and islands much narrower cannot be carved, so the
# sheet cut across the island lagoons in shards. Before the outline rules the OSM water is opened by OPEN_DEG (channels
# under ~2 x OPEN_DEG go: the Toronto Island lagoons, the Western Gap) and closed by CLOSE_DEG (islets go)
OPEN_DEG = 0.0012
CLOSE_DEG = 0.0008
SLIVER_DEG2 = 2e-7  # polygons / holes smaller than this (deg2, ~0.2 real ha) after the clip go (seam slivers)


def read_lakes(path=LAKES):
    b = open(path, "rb").read()
    assert b[:4] == b"LAK1"
    nl, nv, ni = struct.unpack_from("<III", b, 4)
    heads = [list(struct.unpack_from("<iiiifIIII", b, 16 + 36 * i)) for i in range(nl)]
    vo = 16 + 36 * nl
    io_ = vo + nv * 6
    recs = []
    for h in heads:
        v0, nvv, i0, nii = h[5], h[6], h[7], h[8]
        recs.append([h, b[vo + v0 * 6:vo + (v0 + nvv) * 6], b[io_ + i0 * 2:io_ + (i0 + nii) * 2]])
    return recs


def write_lakes(recs, path=LAKES):
    heads, vb, ib = [], bytearray(), bytearray()
    nv = ni = 0
    for h, v, i in recs:
        h = list(h)
        h[5], h[6], h[7], h[8] = nv, len(v) // 6, ni, len(i) // 2
        heads.append(h)
        vb += v
        ib += i
        nv += len(v) // 6
        ni += len(i) // 2
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(b"LAK1" + struct.pack("<III", len(heads), nv, ni))
        for h in heads:
            f.write(struct.pack("<iiiifIIII", *h))
        f.write(vb)
        f.write(ib)
    os.replace(tmp, path)


def rec_geom(rec):
    """(lat, lon arrays, triangles) of a lakes.bin record."""
    h, v, i = rec
    vv = np.frombuffer(v, "<u2").reshape(-1, 3).astype(np.float64)
    a0, b0, a1, b1 = h[0] * 1e-5, h[1] * 1e-5, h[2] * 1e-5, h[3] * 1e-5
    lat = a0 + vv[:, 0] / 65535.0 * (a1 - a0)
    lon = b0 + vv[:, 1] / 65535.0 * (b1 - b0)
    return lat, lon, np.frombuffer(i, "<u2").reshape(-1, 3).astype(np.int64)


def rec_outline(rec):
    """The record's water as a shapely (Multi)Polygon (lon, lat): the union of its triangles."""
    import shapely  # noqa: PLC0415
    lat, lon, tri = rec_geom(rec)
    tris = shapely.polygons(np.stack([np.stack([lon[tri], lat[tri]], -1)[:, k] for k in (0, 1, 2, 0)], 1))
    return shapely.union_all(tris).buffer(0)


def ne_outline(probe):
    """The Natural Earth lake polygon holding the probe, through bake_rivers' outline rules (lake_polys); None when
    the source geojson is not cached here."""
    import shapely  # noqa: PLC0415
    if not os.path.exists(os.path.join(br.SRC, br.LAKE_FILES[0] + ".geojson")):
        return None
    pt = shapely.Point(probe[1], probe[0])
    for rings in br.lake_polys():
        p = shapely.Polygon(rings[0], rings[1:]).buffer(0)
        if p.contains(pt):
            return p
    return None


def osm_outline(parquet, name):
    """The Overture lake polygon of that name through bake_rivers' outline rules (simplify, chaikin, simplify)."""
    import duckdb  # noqa: PLC0415
    import shapely  # noqa: PLC0415
    import shapely.affinity  # noqa: PLC0415
    rows = duckdb.connect().execute(f"SELECT wkb FROM '{parquet.replace(os.sep, '/')}' WHERE name = ? AND subtype = 'lake'",
                                    [name]).fetchall()
    if not rows:
        raise SystemExit(f"{name}: not in {parquet}")
    g = max((shapely.from_wkb(bytes(r[0])) for r in rows), key=lambda q: q.area)
    # opening and closing in a local frame (lon scaled by cos lat: round buffers)
    kx = math.cos(math.radians((g.bounds[1] + g.bounds[3]) * 0.5))
    g = shapely.affinity.scale(g, kx, 1.0, origin=(0, 0))
    g = g.buffer(-OPEN_DEG, join_style="round").buffer(OPEN_DEG + CLOSE_DEG).buffer(-CLOSE_DEG)
    g = shapely.affinity.scale(g, 1.0 / kx, 1.0, origin=(0, 0))
    out = []
    for p in getattr(g, "geoms", [g]):
        rings = []
        for k, ring in enumerate([p.exterior] + list(p.interiors)):
            pts = [(float(x), float(y)) for x, y in np.asarray(ring.coords)[:-1, :2]]
            if len(pts) >= 3:
                pts = br.simplify(br.chaikin(br.simplify(pts, br.SIMPLIFY), 2), br.SIMPLIFY * 0.5)
            if len(pts) >= 3:
                rings.append(pts)
            elif k == 0:
                break
        if rings:
            out.append(shapely.Polygon(rings[0], rings[1:]).buffer(0))
    return shapely.union_all(out)


def clean(g):
    """Polygons of g without slivers and tiny holes (the window seams)."""
    import shapely  # noqa: PLC0415
    out = []
    for p in getattr(g, "geoms", [g]):
        if p.geom_type != "Polygon" or p.area < SLIVER_DEG2:
            continue
        holes = [r for r in p.interiors if shapely.Polygon(r).area >= SLIVER_DEG2]
        out.append(shapely.Polygon(p.exterior, holes).simplify(1e-7, preserve_topology=True))
    return [p for p in out if not p.is_empty]


def build_record(polys, level, window):
    """A lakes.bin record (head, verts bytes, index bytes) of polygons (lon, lat), as bake_rivers.bake_lakes (no
    sea drop inside window)."""
    import mapbox_earcut as earcut  # noqa: PLC0415
    minx = min(p.bounds[0] for p in polys)
    miny = min(p.bounds[1] for p in polys)
    maxx = max(p.bounds[2] for p in polys)
    maxy = max(p.bounds[3] for p in polys)
    kx = math.cos(math.radians((miny + maxy) * 0.5))
    verts, tris, sx = [], [], []
    for p in polys:
        rings = [np.asarray(p.exterior.coords)[:-1, :2]] + [np.asarray(r.coords)[:-1, :2] for r in p.interiors]
        loc, ends = [], []
        for r in rings:
            loc.extend(((x - minx) * br.DEG_M * kx, (y - miny) * br.DEG_M) for x, y in r)
            ends.append(len(loc))
        t = np.asarray(earcut.triangulate_float64(np.array(loc), np.array(ends, dtype=np.uint32))).reshape(-1, 3)
        if len(t) == 0:
            continue
        base = len(verts)
        vl = list(loc)
        t = br.refine(vl, t.tolist(), br.MAX_EDGE)
        verts.extend(vl)
        tris.extend((a + base, b + base, c + base) for a, b, c in t)
        for k0, k1 in zip([0] + ends[:-1], ends):
            for k in range(k0, k1):
                a = loc[k]
                b = loc[k + 1 if k + 1 < k1 else k0]
                sx.append((a[0], a[1], b[0], b[1]))
    sg = np.array(sx)
    va, tri, sd = br.shore_refine(verts, tris, sg)
    t = np.array(tri, dtype=np.int64)
    cxm = va[t, 0].mean(axis=1)
    cym = va[t, 1].mean(axis=1)
    clat = miny + cym / br.DEG_M
    clon = minx + cxm / (br.DEG_M * kx)
    la0, lo0, la1, lo1 = window
    keep = (br.coast_dist(clat, clon) > -60.0) | ((clat >= la0) & (clat <= la1) & (clon >= lo0) & (clon <= lo1))
    t = t[keep]
    used = np.unique(t)
    remap = np.full(len(va), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    t = remap[t]
    va = va[used]
    sd = sd[used]
    if len(va) > 65535:
        raise SystemExit(f"patched lake has {len(va)} verts (u16 indices)")
    vlat = miny + va[:, 1] / br.DEG_M
    vlon = minx + va[:, 0] / (br.DEG_M * kx)
    a0, b0, a1, b1 = int(round(miny * br.Q)), int(round(minx * br.Q)), int(round(maxy * br.Q)), int(round(maxx * br.Q))
    vb = np.zeros(len(va), dtype=[("la", "<u2"), ("lo", "<u2"), ("d", "<u2")])
    vb["la"] = np.clip(np.round((vlat - a0 / br.Q) / ((a1 - a0) / br.Q) * 65535.0), 0, 65535)
    vb["lo"] = np.clip(np.round((vlon - b0 / br.Q) / ((b1 - b0) / br.Q) * 65535.0), 0, 65535)
    vb["d"] = np.clip(np.round(sd), 0, 65535)
    return [a0, b0, a1, b1, level, 0, len(va), 0, t.size], vb.tobytes(), t.reshape(-1).astype("<u2").tobytes()


def find_lake(recs, probe):
    best = None
    for li, (h, _, _) in enumerate(recs):
        if not (h[0] * 1e-5 <= probe[0] <= h[2] * 1e-5 and h[1] * 1e-5 <= probe[1] <= h[3] * 1e-5):
            continue
        lat, lon, tri = rec_geom(recs[li])
        a, b, c = (np.stack([lon[tri[:, k]], lat[tri[:, k]]], 1) for k in range(3))
        p = np.array([probe[1], probe[0]])

        def side(u, v):
            return (v[:, 0] - u[:, 0]) * (p[1] - u[:, 1]) - (v[:, 1] - u[:, 1]) * (p[0] - u[:, 0])
        s1, s2, s3 = side(a, b), side(b, c), side(c, a)
        if (((s1 >= 0) & (s2 >= 0) & (s3 >= 0)) | ((s1 <= 0) & (s2 <= 0) & (s3 <= 0))).any():
            area = (h[2] - h[0]) * (h[3] - h[1])
            if best is None or area > best[1]:
                best = (li, area)
    return None if best is None else best[0]


def main():
    import shapely  # noqa: PLC0415
    from shapely.geometry import box as sbox  # noqa: PLC0415
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    br.coast = np.fromfile(os.path.join(br.OUT, "coast.bin"), dtype=np.uint8).reshape(br.SH, br.SW).astype(np.float32)
    t0 = time.time()
    recs = read_lakes()
    done = []
    for P in PATCHES:
        li = find_lake(recs, P["probe"])
        if li is None:
            raise SystemExit(f"{P['name']}: no lake at {P['probe']}")
        pq = glob.glob(os.path.join(OV_DIR, P["parquet"]))
        if not pq:
            raise SystemExit(f"{P['name']}: {P['parquet']} not in {OV_DIR} (bake_city_water.py --regions fetches it)")
        la0, lo0, la1, lo1 = P["window"]
        win = sbox(lo0, la0, lo1, la1)
        ne = ne_outline(P["probe"])
        cur = rec_outline(recs[li])
        if ne is None:
            # no Natural Earth source here: the current record outside the window (unchanged by a patch)
            ne = cur
        osm = osm_outline(pq[0], P["name"])
        new = shapely.union_all([ne.difference(win), osm.intersection(win)]).buffer(0)
        polys = clean(new)
        h, v, i = build_record(polys, recs[li][0][4], P["window"])
        old_v, old_t = len(recs[li][1]) // 6, len(recs[li][2]) // 6
        print(f"{P['name']}: lake {li}, {old_v} -> {len(v) // 6} verts, {old_t} -> {len(i) // 6} tris, "
              f"{len(polys)} polygons, {sum(len(p.interiors) for p in polys)} islands; window water "
              f"{cur.intersection(win).area * 1e4:.1f} -> {new.intersection(win).area * 1e4:.1f} (1e-4 deg2)", flush=True)
        recs[li] = [h, v, i]
        done.append((P["name"], li))
    if a.dry_run:
        return
    write_lakes(recs)
    lines = open(SOURCES, encoding="utf-8").read().splitlines() if os.path.exists(SOURCES) else []
    lines = [ln for ln in lines if not ln.startswith("lakes.bin")]
    lines.append("lakes.bin (data/world/, records " + ", ".join(f"{li} ({n})" for n, li in done)
                 + f") and the lake_field.bin tiles round them: shores inside tools/patch_lake_osm.py PATCHES windows "
                 f"from Overture Maps base.water release {RELEASE} (OpenStreetMap, ODbL), python tools/patch_lake_osm.py, "
                 f"patched {time.strftime('%Y-%m-%d')}")
    open(SOURCES, "w", encoding="utf-8", newline="").write("\r\n".join(lines) + "\r\n")
    print(f"lakes.bin {os.path.getsize(LAKES) / 1e6:.2f} MB ({time.time() - t0:.0f} s); next: "
          f"python tools/bake_lake_field.py --lakes {','.join(str(li) for _, li in done)}")


if __name__ == "__main__":
    main()
