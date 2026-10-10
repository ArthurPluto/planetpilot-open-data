"""Per-building heights for the detailed-city bake (tools/bake_city_buildings.py), beyond Overture's own.

    python tools/fetch_heights.py download [--only ...]           download the 3D-GloBFP zips only (network, no CPU)
    python tools/fetch_heights.py fetch [--only shanghai,lagos]   download + cut the caches per region (idempotent)
    python tools/fetch_heights.py plan  [--only ...]              list the 3D-GloBFP tiles and MB each region needs
    python tools/fetch_heights.py stats [--only ...]              height counts of the cut caches

Two sources (docs/BUILDING_HEIGHTS.md has the research and the licences):

1. 3D-GloBFP (Che et al. 2024, ESSD 16, 5357; CC BY 4.0): global building footprints (Microsoft ML footprints, Shi et
   al. 2024 for East Asia) with a per-footprint height (XGBoost on Sentinel-1/2, DEMs, night lights; reference year
   2020). Distributed as zipped shapefiles per grid cell on figshare (10 articles, 2,554 cells, about 200 GB). The bake
   joins it onto the Overture footprints by overlap (footprint_heights) and uses it between floors and GHSL.
2. Overture Maps building_part (ODbL, same release as the buildings): a building with has_parts often carries only its
   podium height in the building row while its tower is a part (Singapore's Guoco Tower). part_info returns, per
   building id, the tallest part's height and footprint.

Caches (CITY_SRC, default data/world/src; never deleted):
  heights/globfp/<cell file>.zip                      the figshare downloads (kept: several regions share a cell)
  heights/globfp/index.json                           the figshare file list (ids, sizes, cell bounds)
  heights/globfp_cut/<region>_<box>.parquet           the region's footprints: h, wkb (box in the name: a box change
                                                      re-cuts instead of silently reusing an old cache)
  overture/building_part/<region>_<box>.parquet       the region's Overture building parts
"""
import argparse
import json
import math
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.environ.get("CITY_SRC") or os.path.join(ROOT, "data", "world", "src")
GLOBFP_DIR = os.path.join(SRC, "heights", "globfp")
GLOBFP_CUT = os.path.join(SRC, "heights", "globfp_cut")
PART_DIR = os.path.join(SRC, "overture", "building_part")
OVERTURE = "2026-09-23.1"
PARTS = "s3://overturemaps-us-west-2/release/%s/theme=buildings/type=building_part/*.parquet" % OVERTURE
# the 10 figshare articles of 3D-GloBFP v2 (zenodo 10.5281/zenodo.15487037 data_links.txt), CC BY 4.0
GLOBFP_ARTICLES = (28879733, 28881749, 28882700, 28889813, 28890593, 28891631, 28903454, 28903853, 28904453, 28906499)
FLOOR_M = 3.3
# join: an Overture footprint takes the 3D-GloBFP height when matched footprints cover this share of it
MIN_COVER = 0.3
# the height whose taller matched parts cover this share of the matched area (a tower on a slab reads as the tower
# only when it is a real share of the footprint, not one corner)
TALL_SHARE = 0.3
# tower footprint: the tallest part replaces the building's footprint when it is this much taller than the lowest
# part (or the building row) and covers at most this share of the building
PART_TALLER = 1.5
PART_MAX_SHARE = 0.6
PART_MIN_H = 40.0
TOWER_SHARE = 0.5


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def box_tag(box):
    return "_".join("%.3f" % v for v in box)


def cut_path(name, box):
    return os.path.join(GLOBFP_CUT, "%s_%s.parquet" % (name, box_tag(box)))


def part_path(name, box):
    return os.path.join(PART_DIR, "%s_%s.parquet" % (name, box_tag(box)))


def regions(extra=""):
    """{name: box} of the bake's regions (built-in + tools/city_regions/*.json), plus region JSON files matching the
    glob extra (other worktrees' boxes, for a download ahead of a merge)."""
    import glob
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import bake_city_buildings as B
    out = {n: tuple(r["box"]) for n, r in B.REGIONS.items()}
    for f in sorted(glob.glob(extra)) if extra else []:
        n = os.path.basename(f)[:-5]
        if n not in out:
            with open(f, encoding="utf-8") as h:
                out[n] = tuple(json.load(h)["box"])
    return out


# ---------------------------------------------------------------- 3D-GloBFP

def globfp_index():
    """[(name, url, MB, lon0, lat0, lon1, lat1)] of every 3D-GloBFP cell file (cached figshare listing)."""
    path = os.path.join(GLOBFP_DIR, "index.json")
    if not os.path.exists(path):
        import requests
        os.makedirs(GLOBFP_DIR, exist_ok=True)
        rows = []
        for a in GLOBFP_ARTICLES:
            d = requests.get("https://api.figshare.com/v2/articles/%d" % a, timeout=60).json()
            assert d["license"]["name"] == "CC BY 4.0", d["license"]
            for f in d["files"]:
                m = re.match(r"(\d+)_([-\d.]+)_([-\d.]+)_([-\d.]+)_([-\d.]+)_.*\.zip$", f["name"])
                if m:
                    rows.append([f["name"], f["download_url"], round(f["size"] / 1e6, 1)] + [float(m.group(k)) for k in range(2, 6)])
        with open(path + ".tmp", "w", encoding="utf-8") as h:
            json.dump({"source": "3D-GloBFP v2 figshare articles %s, CC BY 4.0" % (GLOBFP_ARTICLES,), "rows": rows}, h)
        os.replace(path + ".tmp", path)
    with open(path, encoding="utf-8") as h:
        return [tuple(r) for r in json.load(h)["rows"]]


def globfp_cells(box):
    la0, lo0, la1, lo1 = box
    return [r for r in globfp_index() if r[3] < lo1 and r[5] > lo0 and r[4] < la1 and r[6] > la0]


def download(url, path):
    import requests
    t0 = time.time()
    for attempt in range(4):
        try:
            with requests.get(url, stream=True, timeout=120, headers={"User-Agent": "skybound-bake/1"}) as resp:
                resp.raise_for_status()
                with open(path + ".tmp", "wb") as f:
                    for chunk in resp.iter_content(1 << 20):
                        f.write(chunk)
            os.replace(path + ".tmp", path)
            return time.time() - t0
        except Exception as e:  # noqa: BLE001
            log("  retry %d: %s" % (attempt + 1, e))
            time.sleep(5 + 10 * attempt)
    raise RuntimeError("download failed: " + url)


def download_globfp(name, box):
    """The region's 3D-GloBFP cell zips (network only: run it outside the CPU queue)."""
    os.makedirs(GLOBFP_DIR, exist_ok=True)
    for (fn, url, mb, *_b) in globfp_cells(box):
        zp = os.path.join(GLOBFP_DIR, fn)
        if not os.path.exists(zp):
            dt = download(url, zp)
            log("globfp %s: %s %.0f MB, %.0f s" % (name, fn, mb, dt))


def fetch_globfp(name, box):
    import duckdb
    out = cut_path(name, box)
    if os.path.exists(out):
        log("globfp %s: cached (%.1f MB)" % (name, os.path.getsize(out) / 1e6))
        return
    os.makedirs(GLOBFP_DIR, exist_ok=True)
    os.makedirs(GLOBFP_CUT, exist_ok=True)
    cells = globfp_cells(box)
    t0 = time.time()
    download_globfp(name, box)
    la0, lo0, la1, lo1 = box
    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial;")
    parts = []
    for (fn, *_r) in cells:
        zp = os.path.join(GLOBFP_DIR, fn).replace("\\", "/")
        import zipfile
        shp = [x for x in zipfile.ZipFile(zp).namelist() if x.endswith(".shp")]
        for s in shp:
            parts.append(f"""SELECT Height AS h, ST_AsWKB(geom) AS wkb FROM ST_Read('/vsizip/{zp}/{s}')
              WHERE Height IS NOT NULL AND Height > 0 AND ST_XMin(geom) < {lo1} AND ST_XMax(geom) > {lo0}
                AND ST_YMin(geom) < {la1} AND ST_YMax(geom) > {la0}""")
    if not parts:
        log("globfp %s: no cells" % name)
        return
    con.execute("COPY (%s) TO '%s.tmp' (FORMAT parquet, COMPRESSION zstd)" % (" UNION ALL ".join(parts), out.replace("\\", "/")))
    os.replace(out + ".tmp", out)
    n = con.execute("SELECT count(*) FROM read_parquet('%s')" % out.replace("\\", "/")).fetchone()[0]
    log("globfp %s: %d footprints from %d cells (%.0f MB zips), cut %.1f MB, %.0f s" % (
        name, n, len(cells), sum(c[2] for c in cells), os.path.getsize(out) / 1e6, time.time() - t0))


# ---------------------------------------------------------------- Overture building_part

def fetch_parts(name, box):
    import duckdb
    out = part_path(name, box)
    if os.path.exists(out):
        log("parts %s: cached (%.1f MB)" % (name, os.path.getsize(out) / 1e6))
        return
    os.makedirs(PART_DIR, exist_ok=True)
    la0, lo0, la1, lo1 = box
    t0 = time.time()
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs; INSTALL spatial; LOAD spatial;")
    con.execute("SET s3_region='us-west-2'; SET enable_http_metadata_cache=true; SET parquet_metadata_cache=true;")
    con.execute(f"""
    COPY (
      SELECT id, building_id, height, num_floors, min_height, min_floor, ST_AsWKB(geometry) AS wkb
      FROM read_parquet('{PARTS}', hive_partitioning = 0)
      WHERE bbox.xmin >= {lo0 - 0.01} AND bbox.xmax <= {lo1 + 0.01} AND bbox.ymin >= {la0 - 0.01} AND bbox.ymax <= {la1 + 0.01}
        AND coalesce(is_underground, false) = false
    ) TO '{out.replace(chr(92), '/')}.tmp' (FORMAT parquet, COMPRESSION zstd)""")
    os.replace(out + ".tmp", out)
    n = con.execute("SELECT count(*) FROM read_parquet('%s')" % out.replace("\\", "/")).fetchone()[0]
    log("parts %s: %d building parts, %.1f MB, %.0f s" % (name, n, os.path.getsize(out) / 1e6, time.time() - t0))


# ---------------------------------------------------------------- Google Open Buildings 2.5D Temporal (Global South)

G25_DIR = os.path.join(SRC, "heights", "gob25d")
G25_BUCKET = "https://storage.googleapis.com/open-buildings-temporal-data/"
G25_LIST = "https://storage.googleapis.com/storage/v1/b/open-buildings-temporal-data/o"
G25_YEAR = "2023_06_30"
G25_LEVEL_PX = 4.0          # the overview read: 4 m pixels (the data's effective resolution; files are 0.5 m)
G25_SAMPLE_PCT = 75.0       # a footprint's height: this percentile of its samples
G25_THREADS = 6


def utm(lat, lon, epsg):
    """WGS84 lat / lon (deg, numpy) to UTM easting / northing of EPSG 326zz / 327zz (Snyder's series, mm level)."""
    import numpy as np
    zone = epsg % 100
    south = epsg // 100 == 327
    a, f, k0 = 6378137.0, 1 / 298.257223563, 0.9996
    e2 = f * (2 - f)
    ep2 = e2 / (1 - e2)
    phi = np.radians(np.asarray(lat, np.float64))
    lam = np.radians(np.asarray(lon, np.float64)) - math.radians(zone * 6 - 183)
    n = a / np.sqrt(1 - e2 * np.sin(phi) ** 2)
    t = np.tan(phi) ** 2
    c = ep2 * np.cos(phi) ** 2
    A = np.cos(phi) * lam
    m = a * ((1 - e2 / 4 - 3 * e2 ** 2 / 64 - 5 * e2 ** 3 / 256) * phi
             - (3 * e2 / 8 + 3 * e2 ** 2 / 32 + 45 * e2 ** 3 / 1024) * np.sin(2 * phi)
             + (15 * e2 ** 2 / 256 + 45 * e2 ** 3 / 1024) * np.sin(4 * phi) - (35 * e2 ** 3 / 3072) * np.sin(6 * phi))
    x = k0 * n * (A + (1 - t + c) * A ** 3 / 6 + (5 - 18 * t + t * t + 72 * c - 58 * ep2) * A ** 5 / 120) + 500000.0
    y = k0 * (m + n * np.tan(phi) * (A * A / 2 + (5 - t + 9 * c + 4 * c * c) * A ** 4 / 24
                                     + (61 - 58 * t + t * t + 600 * c - 330 * ep2) * A ** 6 / 720))
    return x, (y + 10000000.0 if south else y)


def g25_epsgs(box):
    la0, lo0, la1, lo1 = box
    zones = range(int(math.floor((lo0 + 180) / 6)) + 1, int(math.floor((lo1 + 180) / 6)) + 2)
    hemi = ([326] if la1 > 0 else []) + ([327] if la0 < 0 else [])
    return [h * 100 + z for h in hemi for z in zones]


def g25_index(epsgs):
    """[(uri, epsg, x0, y1, size px, scale)] of the 2023 tiles in these UTM zones (manifests cached)."""
    import requests
    os.makedirs(G25_DIR, exist_ok=True)
    lp = os.path.join(G25_DIR, "manifests_%s.json" % G25_YEAR)
    if not os.path.exists(lp):
        names, tok = [], None
        while True:
            prm = {"prefix": "v1/manifests/", "maxResults": 1000, "fields": "items(name),nextPageToken"}
            if tok:
                prm["pageToken"] = tok
            d = requests.get(G25_LIST, params=prm, timeout=60).json()
            names += [i["name"] for i in d.get("items", []) if i["name"].endswith(G25_YEAR + ".json")]
            tok = d.get("nextPageToken")
            if not tok:
                break
        with open(lp + ".tmp", "w", encoding="utf-8") as h:
            json.dump(names, h)
        os.replace(lp + ".tmp", lp)
    with open(lp, encoding="utf-8") as h:
        names = json.load(h)
    out = []
    for nm in names:
        epsg = int(nm.split("_")[2])
        if epsg not in epsgs:
            continue
        mp = os.path.join(G25_DIR, os.path.basename(nm))
        if not os.path.exists(mp):
            r = requests.get(G25_BUCKET + nm, timeout=120)
            r.raise_for_status()
            with open(mp + ".tmp", "wb") as h:
                h.write(r.content)
            os.replace(mp + ".tmp", mp)
        with open(mp, encoding="utf-8") as h:
            m = json.load(h)
        pre = m["uriPrefix"].replace("gs://open-buildings-temporal-data/", "")
        for ts in m["tilesets"]:
            for src in ts["sources"]:
                at = src["affineTransform"]
                out.append((pre + src["uris"][0], epsg, at["translateX"], at["translateY"], src["dimensions"]["width"],
                            at["scaleX"]))
    return out


def g25_tiles(box):
    la0, lo0, la1, lo1 = box
    lat = [la0, la0, la1, la1, (la0 + la1) / 2, la0, la1, (la0 + la1) / 2, (la0 + la1) / 2]
    lon = [lo0, lo1, lo0, lo1, (lo0 + lo1) / 2, (lo0 + lo1) / 2, (lo0 + lo1) / 2, lo0, lo1]
    out = []
    for t in g25_index(g25_epsgs(box)):
        uri, epsg, x0, y1, npx, sc = t
        x, y = utm(lat, lon, epsg)
        ext = npx * sc
        if min(x) < x0 + ext and max(x) > x0 and min(y) < y1 and max(y) > y1 - ext:
            out.append(t)
    return out


class _HttpFile:
    """A read-only, seekable file over HTTP range requests (1 MB blocks, cached): tifffile reads one overview of a
    cloud-optimised GeoTIFF without the whole file."""
    BLOCK = 1 << 20

    def __init__(self, url):
        import requests
        self.s = requests.Session()
        self.url = url
        self.pos = 0
        self.blocks = {}
        r = self.s.head(url, timeout=60)
        r.raise_for_status()
        self.size = int(r.headers["Content-Length"])
        self.fetched = 0

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def tell(self):
        return self.pos

    def seekable(self):
        return True

    def readable(self):
        return True

    def _get(self, b0, b1):
        need = [b for b in range(b0, b1 + 1) if b not in self.blocks]
        while need:
            run = [need[0]]
            while len(run) < 64 and len(need) > len(run) and need[len(run)] == run[-1] + 1:
                run.append(need[len(run)])
            lo = run[0] * self.BLOCK
            hi = min(self.size, (run[-1] + 1) * self.BLOCK) - 1
            for attempt in range(4):
                try:
                    r = self.s.get(self.url, headers={"Range": "bytes=%d-%d" % (lo, hi)}, timeout=120)
                    r.raise_for_status()
                    break
                except Exception:  # noqa: BLE001
                    if attempt == 3:
                        raise
                    time.sleep(5 + 10 * attempt)
            data = r.content
            self.fetched += len(data)
            for k, b in enumerate(run):
                self.blocks[b] = data[k * self.BLOCK:(k + 1) * self.BLOCK]
            need = need[len(run):]

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos))
        if n == 0:
            return b""
        b0, b1 = self.pos // self.BLOCK, (self.pos + n - 1) // self.BLOCK
        self._get(b0, b1)
        buf = b"".join(self.blocks[b] for b in range(b0, b1 + 1))
        o = self.pos - b0 * self.BLOCK
        self.pos += n
        return buf[o:o + n]

    def readinto(self, b):
        d = self.read(len(b))
        b[:len(d)] = d
        return len(d)

    def close(self):
        self.s.close()


def g25_tile_path(uri):
    return os.path.join(G25_DIR, "h4m_" + uri.replace("/", "_").replace(".tif", ".npz"))


def _g25_tile(t):
    """Read one tile's 4 m height overview into its npz cache; returns the bytes read."""
    import numpy as np
    import tifffile
    uri, epsg, x0, y1, npx, sc = t
    p = g25_tile_path(uri)
    if os.path.exists(p):
        return 0
    fh = _HttpFile(G25_BUCKET + uri)
    with tifffile.TiffFile(fh) as tf:
        want = int(round(npx * sc / G25_LEVEL_PX))
        pg = min(tf.pages, key=lambda q: abs(q.shape[-1] - want))
        a = pg.asarray()
    hb = a[1] if a.ndim == 3 and a.shape[0] == 3 else a[..., 1]
    q = np.where(hb >= 0, np.clip(np.round(hb * 2.0), 0, 254), 255).astype(np.uint8)
    np.savez_compressed(p[:-4] + ".tmp.npz", h=q, epsg=epsg, x0=x0, y1=y1, px=npx * sc / q.shape[1])
    os.replace(p[:-4] + ".tmp.npz", p)
    fh.close()
    return fh.fetched


def fetch_g25(name, box):
    """The 4 m overview of the building-height band of every 2023 tile over the box (uint8, 0.5 m steps, 255 none),
    read with HTTP range requests (G25_THREADS tiles at once: network bound)."""
    from concurrent.futures import ThreadPoolExecutor
    tiles = g25_tiles(box)
    t0 = time.time()
    with ThreadPoolExecutor(G25_THREADS) as ex:
        got = sum(ex.map(_g25_tile, tiles))
    log("g25 %s: %d tiles, %.0f MB read, %.0f s" % (name, len(tiles), got / 1e6, time.time() - t0))


def g25_heights(name, box, loc, clat, clon):
    """Google 2.5D (2023) height per bake footprint: G25_SAMPLE_PCT of samples at the centroid and halfway to each
    vertex (4 m pixels); NaN where no tile or no building pixels."""
    import numpy as np
    import shapely
    n = len(loc)
    out = np.full(n, np.nan)
    tiles = [t for t in g25_tiles(box) if os.path.exists(g25_tile_path(t[0]))] if n else []
    if not tiles:
        return out
    kr = 6371000.0 * math.pi / 180.0
    lc, li = shapely.get_coordinates(loc, return_index=True)
    sx = np.r_[lc[:, 0] * 0.5, np.zeros(n)]
    sy = np.r_[lc[:, 1] * 0.5, np.zeros(n)]
    si = np.r_[li, np.arange(n)]
    slat = clat[si] + sy / kr
    slon = clon[si] + sx / (kr * np.cos(np.radians(clat[si])))
    val = np.full(len(si), np.nan)
    done = np.zeros(len(si), bool)
    for epsg in sorted(set(t[1] for t in tiles)):
        x, y = utm(slat, slon, epsg)
        for (uri, ep, x0, y1, npx, sc) in tiles:
            if ep != epsg:
                continue
            ext = npx * sc
            inside = ~done & (x >= x0) & (x < x0 + ext) & (y <= y1) & (y > y1 - ext)
            if not inside.any():
                continue
            z = np.load(g25_tile_path(uri))
            a, px = z["h"], float(z["px"])
            c = np.clip(((x[inside] - x0) / px).astype(np.int64), 0, a.shape[1] - 1)
            r = np.clip(((y1 - y[inside]) / px).astype(np.int64), 0, a.shape[0] - 1)
            v = a[r, c].astype(np.float64)
            idx = np.nonzero(inside)[0]
            val[idx] = np.where(v == 255, np.nan, v * 0.5)
            done[idx] = True
    ok = np.isfinite(val) & (val > 0.0)
    if not ok.any():
        return out
    si, val = si[ok], val[ok]
    order = np.lexsort((val, si))
    si, val = si[order], val[order]
    start = np.r_[0, np.nonzero(np.diff(si))[0] + 1]
    cnt = np.diff(np.r_[start, len(si)])
    k = start + np.minimum(cnt - 1, np.floor(cnt * G25_SAMPLE_PCT / 100.0).astype(np.int64))
    out[si[start]] = val[k]
    return out


# ---------------------------------------------------------------- readers (the bake's hook)

def _local(lon, lat, lon_ref, lat_ref):
    kr = 6371000.0 * math.pi / 180.0
    return (lon - lon_ref) * kr * math.cos(math.radians(lat_ref)), (lat - lat_ref) * kr


def footprint_heights(name, box, loc, clat, clon):
    """3D-GloBFP height (m) for each bake footprint (loc: shapely polygons in local metres round (clat, clon)), NaN
    where the cache is missing or nothing matches. Matched = the 3D-GloBFP footprints that overlap it; the height is
    the tallest one whose taller matches still cover TALL_SHARE of the matched area."""
    import numpy as np
    import shapely
    out = np.full(len(loc), np.nan)
    path = cut_path(name, box)
    if not os.path.exists(path) or len(loc) == 0:
        return out
    import duckdb
    tb = duckdb.connect().execute("SELECT h, wkb FROM read_parquet('%s')" % path.replace("\\", "/")).fetchnumpy()
    gh = np.asarray(tb["h"], np.float64)
    gg = shapely.from_wkb(np.array([bytes(w) for w in tb["wkb"]], dtype=object))
    la0, lo0, la1, lo1 = box
    lat_ref, lon_ref = (la0 + la1) * 0.5, (lo0 + lo1) * 0.5
    kr = 6371000.0 * math.pi / 180.0
    cx = kr * math.cos(math.radians(lat_ref))
    gg = shapely.transform(gg, lambda c: np.column_stack([(c[:, 0] - lon_ref) * cx, (c[:, 1] - lat_ref) * kr]))
    ok = shapely.is_valid(gg)
    gg, gh = gg[ok], gh[ok]
    ox = (clon - lon_ref) * cx
    oy = (clat - lat_ref) * kr
    lc, li = shapely.get_coordinates(loc, return_index=True)
    og = shapely.polygons(shapely.linearrings(lc + np.column_stack([ox[li], oy[li]]), indices=li))
    bad = ~shapely.is_valid(og)
    og[bad] = shapely.make_valid(og[bad])
    tree = shapely.STRtree(gg)
    oi, gi = tree.query(og, predicate="intersects")
    if len(oi) == 0:
        return out
    inter = shapely.area(shapely.intersection(og[oi], gg[gi]))
    keep = inter > 0.5
    oi, gi, inter = oi[keep], gi[keep], inter[keep]
    oarea = shapely.area(og)
    # per footprint: by height descending, the first height where the cumulative overlap reaches TALL_SHARE
    order = np.lexsort((-gh[gi], oi))
    oi, gi, inter = oi[order], gi[order], inter[order]
    cov = np.bincount(oi, weights=inter, minlength=len(og))
    start = np.r_[0, np.nonzero(np.diff(oi))[0] + 1]
    csum = np.cumsum(inter)
    base = np.repeat(csum[start] - inter[start], np.diff(np.r_[start, len(oi)]))
    frac = (csum - base) / cov[oi]
    hit = frac >= TALL_SHARE
    first = np.full(len(og), -1, np.int64)
    idx = np.nonzero(hit)[0]
    # the first hit per footprint (idx is ascending, oi grouped)
    u, k = np.unique(oi[idx], return_index=True)
    first[u] = idx[k]
    good = (first >= 0) & (cov >= MIN_COVER * oarea)
    out[good] = gh[gi[first[good]]]
    return out


def part_info(name, box, ids, clat, clon):
    """Per bake building (Overture ids): (max part height m or NaN, min part height m or NaN, the tower part's polygon
    in local metres round (clat, clon) or None)."""
    import numpy as np
    import shapely
    n = len(ids)
    hmax = np.full(n, np.nan)
    hmin = np.full(n, np.nan)
    poly = np.full(n, None, dtype=object)
    path = part_path(name, box)
    if not os.path.exists(path) or n == 0:
        return hmax, hmin, poly
    import duckdb
    tb = duckdb.connect().execute("SELECT building_id, height, num_floors, wkb FROM read_parquet('%s') WHERE building_id IS NOT NULL"
        % path.replace("\\", "/")).fetchnumpy()
    bid = np.asarray(tb["building_id"], dtype=object)
    ph = tb["height"]
    ph = np.asarray(ph.astype(np.float64).filled(np.nan) if hasattr(ph, "filled") else ph, np.float64)
    pf = tb["num_floors"]
    pf = np.asarray(pf.astype(np.float64).filled(np.nan) if hasattr(pf, "filled") else pf, np.float64)
    h = np.where(np.isfinite(ph) & (ph > 2.0), ph, np.where(np.isfinite(pf) & (pf > 0) & (pf <= 110), pf * FLOOR_M, np.nan))
    pos = {s: i for i, s in enumerate(ids)}
    row = np.array([pos.get(s, -1) for s in bid], np.int64)
    ok = (row >= 0) & np.isfinite(h)
    if not ok.any():
        return hmax, hmin, poly
    wkb = tb["wkb"]
    sel = np.nonzero(ok)[0]
    geo = shapely.from_wkb(np.array([bytes(wkb[i]) for i in sel], dtype=object))
    row, h = row[sel], h[sel]
    np.fmax.at(hmax, row, h)
    hmin_init = np.full(n, np.inf)
    np.minimum.at(hmin_init, row, h)
    hmin = np.where(np.isfinite(hmin_init), hmin_init, np.nan)
    # the tower's footprint: the largest part of TOWER_SHARE of the tallest part's height or more (a tower is often
    # stacked sections, its top a small crown or a spire: Shanghai Tower, Jin Mao)
    ar = shapely.area(geo)
    cand = h >= TOWER_SHARE * hmax[row]
    order = np.lexsort((-ar, ~cand, row))
    first = order[np.r_[True, row[order][1:] != row[order][:-1]]]
    kr = 6371000.0 * math.pi / 180.0
    for j in first:
        r = row[j]
        g = geo[j]
        if shapely.get_type_id(g) == 6:
            parts = shapely.get_parts(g)
            g = parts[np.argmax(shapely.area(parts))]
        if shapely.get_type_id(g) != 3:
            continue
        c = shapely.get_coordinates(shapely.get_exterior_ring(g))
        lx = (c[:, 0] - clon[r]) * kr * math.cos(math.radians(clat[r]))
        ly = (c[:, 1] - clat[r]) * kr
        poly[r] = shapely.Polygon(np.column_stack([lx, ly]))
    return hmax, hmin, poly


def apply(name, box, ids, loc, area, clat, clon, h, src, known_h, known_f, skip):
    """The bake's height hook (tools/bake_city_buildings.py, after Overture / floors / GHSL are chosen): returns
    (h, src, loc, area, n_globfp, n_google, n_parts, n_tower_fp) with
      - Overture building parts: a has_parts building lower than its tallest part takes the part height (src 5), and
        a tower on a podium (or a whole complex at tower height) takes the tallest part's footprint;
      - 3D-GloBFP: a footprint with no Overture height and no floors takes the 3D-GloBFP height instead of the GHSL
        guess (src 4);
      - Google Open Buildings 2.5D (2023, Global South only) before 3D-GloBFP where it has building pixels (src 6).
    skip: footprints left alone (the bake's UNBUILT towers: a part's height is the plan). A missing cache changes
    nothing."""
    import numpy as np
    import shapely
    hmax, hmin, tpoly = part_info(name, box, ids, clat, clon)
    h = h.copy()
    src = src.copy()
    loc = loc.copy()
    area = area.copy()
    raise_p = np.isfinite(hmax) & (hmax > h + 1.0) & ~skip
    h = np.where(raise_p, hmax, h)
    src = np.where(raise_p, 5, src)
    # tower footprint: the tallest part when it stands well over the rest of the complex and is a small share of it
    low = np.where(known_h, np.fmin(hmin, h), hmin)
    tower = ~skip & np.isfinite(hmax) & (hmax >= PART_MIN_H) & (hmax >= PART_TALLER * np.where(np.isfinite(low), low, 0.0)) & \
        (tpoly != None)  # noqa: E711
    n_fp = 0
    for i in np.nonzero(tower)[0]:
        p = shapely.make_valid(tpoly[i])
        if shapely.get_type_id(p) != 3:
            continue
        p = shapely.orient_polygons(shapely.simplify(p, 0.75, preserve_topology=True))
        a = shapely.area(p)
        npts = shapely.get_num_coordinates(p) - 1
        if a < 80.0 or a > PART_MAX_SHARE * area[i] or npts < 3 or npts > 40 or not shapely.is_valid(p):
            continue
        loc[i] = p
        area[i] = a
        h[i] = max(h[i], hmax[i])
        n_fp += 1
    free = ~known_h & ~known_f & ~raise_p & ~skip
    gf = footprint_heights(name, box, loc, clat, clon)
    gf = np.where(gf >= 2.5, gf, np.nan)
    gg = g25_heights(name, box, loc, clat, clon) if free.any() else np.full(len(h), np.nan)
    gg = np.where(gg >= 2.5, gg, np.nan)
    # Google 2.5D first where it exists (2023, 4 m: closer than 3D-GloBFP in every height band but 20 to 40 m in the
    # docs/BUILDING_HEIGHTS.md checks; 3D-GloBFP flattens Global South towers: Lagos 60 to 100 m read 17 m), then
    # 3D-GloBFP
    use_g = free & np.isfinite(gg)
    use_f = free & ~use_g & np.isfinite(gf)
    h = np.where(use_g, gg, np.where(use_f, gf, h))
    src = np.where(use_g, 6, np.where(use_f, 4, src))
    return h, src, loc, area, int(use_f.sum()), int(use_g.sum()), int(raise_p.sum()), n_fp


# ---------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["download", "fetch", "plan", "stats"])
    ap.add_argument("--only", default="")
    ap.add_argument("--no-parts", action="store_true")
    ap.add_argument("--no-google", action="store_true")
    ap.add_argument("--regions-glob", default="", help="more region JSON files (e.g. other worktrees' boxes)")
    a = ap.parse_args()
    regs = regions(a.regions_glob)
    names = [n for n in a.only.split(",") if n] or list(regs)
    if a.cmd == "plan":
        tot = {}
        for n in names:
            cells = globfp_cells(regs[n])
            for c in cells:
                tot[c[0]] = c[2]
            print("%-18s %6.0f MB  %s" % (n, sum(c[2] for c in cells), " ".join(c[0].split("_")[0] for c in cells)))
        have = sum(mb for f, mb in tot.items() if os.path.exists(os.path.join(GLOBFP_DIR, f)))
        print("unique cells %d, %.0f MB (%.0f MB already downloaded)" % (len(tot), sum(tot.values()), have))
    elif a.cmd == "download":
        for n in names:
            download_globfp(n, regs[n])
    elif a.cmd == "fetch":
        for n in names:
            if not a.no_parts:
                fetch_parts(n, regs[n])
            fetch_globfp(n, regs[n])
            if not a.no_google:
                fetch_g25(n, regs[n])
    else:
        import duckdb
        for n in names:
            p = cut_path(n, regs[n])
            if not os.path.exists(p):
                print("%-18s (no cache)" % n)
                continue
            r = duckdb.connect().execute("SELECT count(*), median(h), sum((h >= 60)::int), sum((h >= 100)::int), max(h) "
                "FROM read_parquet('%s')" % p.replace("\\", "/")).fetchone()
            print("%-18s %8d footprints, median %.1f m, %6d >= 60 m, %5d >= 100 m, max %.0f m" % ((n,) + r))


if __name__ == "__main__":
    main()
