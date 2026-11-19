"""City near packs: bundled near-tier detail tiles for a detailed city whose coast,
quays, beaches and estuaries come from OpenStreetMap and whose ground comes
from Copernicus GLO-30 (docs/REAL_PLACEMENT.md 13.27).

Why: the global mid layer (7.5", about 46 toy m) smooths a city's port into a
beach, rounds thin moles off and blurs the bluffs; the HD packs (3", optional
download) take their coast from the Copernicus water mask. A city near pack is
the near tier (0.5 deg tiles, 601 x 601 samples of 3", DetailTiles.NEAR) baked
for one city, bundled with the game (desktop and web), and its near tiles win
over the HD packs (DetailTiles.CITY_DIR).

Sources (cached in data/world/src/city_near/<city>/, not shipped):
  - Copernicus DEM GLO-30 (1") tiles, AWS open data bucket copernicus-dem-30m
    (a surface model: the ground is taken as a grey opening, CITY.open_m wide,
    which removes buildings, trees, ships and cranes narrower than that).
  - OpenStreetMap via the Overpass API (ODbL, so the pack lives under
    data/world/odbl/detail/): natural=coastline, natural=water areas (limans,
    harbours), natural=beach, man_made=pier / breakwater / quay, the named
    promenades (CITY.promenade_re).
  - The global mid layer's tiles over the pack's mid box, dumped raw by
    tools/detail_tile_dump.gd (feathering, the sea floor away from the shore,
    the land cover class, the pack's own mid tier for the web).

Per pack:
  1. Sea on a fine grid (2400 samples per degree): the OSM coastline faces
     (land on the left of a coastline way), plus every OSM water area the mid
     layer already draws as sea (limans at sea level), plus the harbours;
     moles, piers and breakwaters are added as land at least CITY.mole_m wide
     (a thinner mole falls between the 3" samples and shows as a dotted line).
  2. Signed sea distance (toy m, + land) by EDT on the fine grid, read at the
     3" samples: the per-pixel shoreline (bilinear of the samples) then follows
     the OSM coast to a few toy metres.
  3. Ground: the GLO-30 opening, smoothed, read at the samples; land at least
     1 m; quays CITY.quay_h; the sea floor from the mid layer, never deeper
     ramping out from the shore; harbours (class 15, deep water) and named waters
     their own (CITY.depths; a shallow salt liman reads pale through the water).
  4. Land cover: the mid layer's class; water 8; and the shore kinds the
     terrain shader reads (coast_px_k): 12 quay (the port's low land, moles,
     piers, promenades), 14 beach (OSM beaches), 13 bare shore (any other land
     within CITY.shore_m of the sea or of a CITY.bare_waters water): no sand.
  5. Feathering: the outer NEAR_FEATHER deg of the near box fade into the mid
     layer (bake_detail.py's rule).
Output: data/world/odbl/detail/<city>.bin (SBDT version 1, codec 0 lossless
WebP, the region pack format of tools/bake_detail.py) and packs.json there.

Run:  python tools/bake_city_near.py [city ...]          (default: every city)
      python tools/bake_city_near.py --fetch [city ...]  (download the sources)
Needs: numpy, scipy, Pillow, tifffile, shapely; the mid dump first:
  godot --headless --path . -s res://tools/detail_tile_dump.gd -- tier=mid box=<mid box, inset> out=data/world/src/city_near/<city>/mid
"""
import io
import json
import math
import os
import re
import struct
import sys
import time
import urllib.parse
import urllib.request

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from shapely.geometry import LineString, Point, Polygon, MultiPolygon, box as sbox
from shapely.ops import linemerge, polygonize, unary_union
from shapely import prepared

sys.path.insert(0, os.path.dirname(__file__))
from bake_detail import enc_coast, dec_coast, encode_tile, box_weight, COAST_RANGE  # noqa: E402
from world_scale import R_REAL, SCALE  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "data", "world", "odbl", "detail")
SRC = os.path.join(ROOT, "data", "world", "src", "city_near")
M_PER_DEG = R_REAL * math.pi / 180.0
NPPD = 1200          # near samples per degree (3")
FPPD = 2400          # fine grid
MPPD = 480           # mid samples per degree
NEAR_FEATHER = 0.15  # deg (bake_detail.py)
C_DEEP = 15          # class: deep water (dredged harbour basins)
COP30 = "https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N{lat:02d}_00_E{lon:03d}_00_DEM/Copernicus_DSM_COG_10_N{lat:02d}_00_E{lon:03d}_00_DEM.tif"
OVERPASS = "https://overpass-api.de/api/interpreter"
C_WATER, C_QUAY, C_BARE, C_BEACH = 8, 12, 13, 14

CITIES = {
    "odesa": {
        "label": "Odesa (city near pack)",
        "near": (46.0, 47.0, 30.0, 31.5),
        "mid": (45.0, 48.0, 29.0, 32.0),
        "dem": [(46, 30)],             # GLO-30 tiles (lat, lon of the SW corner); elsewhere the mid layer
        # the shore kinds are drawn inside this box only (lat0, lat1, lon0, lon1)
        "zone": (46.28, 46.70, 30.55, 31.05),
        # the port: low land in here within port_m of the sea is quay
        "port": [(46.4835, 30.7600), (46.4890, 30.7700), (46.4985, 30.7640), (46.5060, 30.7470), (46.5150, 30.7430),
                 (46.5290, 30.7480), (46.5300, 30.7300), (46.5130, 30.7240), (46.4985, 30.7290), (46.4905, 30.7380),
                 (46.4860, 30.7470), (46.4835, 30.7540)],
        "port_m": 450.0,
        "port_low": 9.0,               # real m: port land below this (the ground model) is quay
        # Chornomorsk port on the Sukhyi liman: quays
        "port2": [(46.2950, 30.6500), (46.3000, 30.6700), (46.3350, 30.6700), (46.3520, 30.6650), (46.3520, 30.6280),
                  (46.3380, 30.6250), (46.3120, 30.6450)],
        "quay_h": (2.0, 4.0),          # real m, the quay deck
        "mole_m": 120.0,               # real m, the narrowest mole / pier the pack draws
        "pier_grow_m": 25.0,           # real m, an OSM pier area grows by this
        "mole_min_len": 250.0,
        "mesh_min_len": 40.0,
        "mole_h": 2.6, "groyne_w": 3.0, "groyne_h": 1.0,  # toy m: mole deck over the water, groyne width and height          # real m: breakwaters and groynes this long are drawn as meshes
        # boxes (lat0, lat1, lon0, lon1) whose thin coastline land is a mesh mole, not grown land:
        # the Reid mole out to the Vorontsov Lighthouse
        "no_thin": [(46.49360, 46.49700, 30.75880, 30.76250)],         # real m: shorter piers and breakwaters stay as they are (not land, not quay)
        "thin_m": 50.0,                # real m: land of the coastline itself thinner than 2 x this (moles drawn in the
                                       # coastline) grows by it on each side, so the samples hold it
        # quay land added by hand (lat, lon corners): what a real-size landmark needs at 1/5
        "fills": [
            {"why": "the port road at the foot of the Potemkin Stairs (the landmarks agent: the 142 m flights run past the 1/5 bluff foot)",
             "poly": [(46.490793, 30.741948), (46.487950, 30.745142), (46.492074, 30.752882), (46.494916, 30.749688)]},
            {"why": "the passenger pier (Marine Station) on from the Stairs along their axis (bearing 52.3), 84 toy m wide, 138 m out: the real-size terminal (116 x 38 m) 15 m off the water",
             "poly": [(46.493558, 30.746926), (46.490573, 30.750277), (46.494913, 30.758433), (46.497898, 30.755082)]},
        ],
        # water kept round water landmarks (lat, lon, real m): Landmarks.water_problem wants their
        # fp / 2 ring wet (the Vorontsov Lighthouse at the head of the Reid mole: fp 20 toy m)
        "keep_wet": [(46.496533, 30.760034, 58.0)],
        "promenade_re": "здоров",      # named promenades (the Trasa Zdorovya): quay
        "promenade_m": 12.0,           # real m, half width
        "kind_m": 50.0,                # real m: a sample this close to a beach / quay feature takes its kind
        "shore_m": 1500.0,             # real m: land this close to the sea is bare shore unless beach or quay
        "pale_waters": ["Куяльницький"],  # salt pan water: pale (the water shader)
        "bare_waters": ["Хаджибейський", "Тилігульський", "Дністровський"],  # their shores too (Kuyalnyk keeps its pale flats)
        # the sea floor (real m): shelf_d0 at the waterline, shelf m per m out, the mid layer's
        # bathymetry where deeper (to shelf_max); the dredged port basins harbour_d (and class
        # 15, deep water); named waters their own (Kuyalnyk: a salt pan a few decimetres deep,
        # pale; Khadzhibey and the Dniester liman shallow; the Sukhyi liman Chornomorsk's port)
        "deep_m": 500.0,               # real m: the port's waters this far out of its polygons are deep (class 15)
        "shelf_d0": 1.5, "shelf": 0.012, "shelf_max": 60.0, "harbour_d": 14.0,
        "depths": {"Куяльницький": 0.3, "Хаджибейський": 3.0, "Сухий": 12.0, "Дністровський": 2.0,
                   "Тилігульський": 4.0, "Будацький": 1.2},
        "open_m": 150.0,               # real m, the ground opening window
        # relief gain (toy m per real m) of the land relief finer than the global grid inside the
        # zone, instead of the game's TILE_GAIN 0.18 for gentle land: the bluff under Primorsky
        # Boulevard, the balkas and the liman cliffs read at about the global grid's own gain
        "relief_gain": 0.7,
        "relief_fade": 0.03,           # deg, the gain fades to the game's own over this at the zone edge
        "overpass": """[out:json][timeout:180];
(
  way["natural"="coastline"]({b});
  relation["natural"="water"]({b});
  way["natural"="water"]["water"~"^(lagoon|lake|reservoir|river|harbour|basin|oxbow)$"]({b});
  way["natural"="beach"]({b});
  relation["natural"="beach"]({b});
  way["man_made"~"^(pier|breakwater|groyne|quay)$"]({b});
  relation["man_made"~"^(pier|breakwater)$"]({b});
  way["landuse"="port"]({b});
  relation["landuse"="port"]({b});
  way["highway"]["name"~"здоров",i](46.38,30.70,46.50,30.80);
);
out geom;
""",
    },
}


def log(*a):
    print(*(str(x).encode("ascii", "backslashreplace").decode() for x in a), flush=True)


# --------------------------------------------------------------------------- sources

def src_dir(city):
    return os.path.join(SRC, city)


def fetch(city, c):
    d = src_dir(city)
    os.makedirs(os.path.join(d, "mid"), exist_ok=True)
    for la, lo in c["dem"]:
        p = os.path.join(d, f"cop30_N{la:02d}E{lo:03d}.tif")
        if not os.path.exists(p):
            log("GET", COP30.format(lat=la, lon=lo))
            urllib.request.urlretrieve(COP30.format(lat=la, lon=lo), p + ".part")
            os.replace(p + ".part", p)
    p = os.path.join(d, "osm_coast.json")
    if not os.path.exists(p):
        n0, n1, e0, e1 = c["near"]
        q = c["overpass"].replace("{b}", f"{n0},{e0},{n1},{e1}")
        for wait in (0, 30, 60, 120, 240):
            time.sleep(wait)
            req = urllib.request.Request(OVERPASS, data=urllib.parse.urlencode({"data": q}).encode(),
                                         headers={"User-Agent": "planet-pilot-city-near/1.0"})
            try:
                b = urllib.request.urlopen(req, timeout=300).read()
            except Exception as e:  # noqa: BLE001
                log("overpass:", e)
                continue
            if b.lstrip().startswith(b"{"):
                open(p, "wb").write(b)
                break
        with open(os.path.join(d, "osm_coast.overpass"), "w", encoding="utf-8") as f:
            f.write(q)
    log("sources in", d)


def read_cop30(path):
    import tifffile
    with tifffile.TiffFile(path) as t:
        a = t.pages[0].asarray().astype(np.float32)
    return a  # pixel (r, c) = (lat_n - r / 3600, lon_w + c / 3600), PixelIsPoint


def mid_mosaic(city, c):
    """The mid layer over the mid box: h (real m), class, coast (toy m), row 0 north."""
    lat0, lat1, lon0, lon1 = c["mid"]
    ny, nx = int(round((lat1 - lat0) * MPPD)) + 1, int(round((lon1 - lon0) * MPPD)) + 1
    h = np.full((ny, nx), -40.0, np.float32)
    cls = np.full((ny, nx), C_WATER, np.uint8)
    co = np.full((ny, nx), -COAST_RANGE, np.float32)
    raws = {}
    for ty in range(int(round(90 - lat1)), int(round(90 - lat0))):
        for tx in range(int(round(lon0 + 180)), int(round(lon1 + 180))):
            p = os.path.join(src_dir(city), "mid", f"mid_{tx}_{ty}.raw")
            if not os.path.exists(p):
                continue
            b = np.fromfile(p, np.uint8).reshape(481, 481, 4)
            raws[(tx, ty)] = b
            r = (ty - int(round(90 - lat1))) * MPPD
            q = (tx - int(round(lon0 + 180))) * MPPD
            h[r:r + 481, q:q + 481] = b[..., :2].copy().view("<i2")[..., 0]
            cls[r:r + 481, q:q + 481] = b[..., 2] >> 4
            co[r:r + 481, q:q + 481] = dec_coast(b[..., 3])
    return h, cls, co, raws


# --------------------------------------------------------------------------- OSM geometry

def _ring(geom):
    return [(p["lon"], p["lat"]) for p in geom]


def _assemble(lines):
    """Closed polygons from (possibly split) member ways."""
    if not lines:
        return []
    merged = unary_union([LineString(l) for l in lines if len(l) >= 2])
    return [p for p in polygonize(merged) if p.is_valid and p.area > 0]


def area_of(e):
    """Polygon (or None) of an OSM way / multipolygon relation with geometry."""
    if e["type"] == "way":
        g = e.get("geometry") or []
        if len(g) >= 4 and g[0] == g[-1]:
            p = Polygon(_ring(g))
            return p if p.is_valid else p.buffer(0)
        return None
    outer = [_ring(m["geometry"]) for m in e.get("members", []) if m.get("role") in ("outer", "") and m.get("geometry")]
    inner = [_ring(m["geometry"]) for m in e.get("members", []) if m.get("role") == "inner" and m.get("geometry")]
    po = _assemble(outer)
    if not po:
        return None
    a = unary_union(po)
    if inner:
        a = a.difference(unary_union(_assemble(inner)))
    return a


def deg_buf(geom, m, lat):
    """Buffer by m real metres (an ellipse in degrees, good enough for a city box)."""
    k = math.cos(math.radians(lat))
    from shapely import affinity
    g = affinity.scale(geom, xfact=k, yfact=1.0, origin=(0, 0))
    g = g.buffer(m / M_PER_DEG, quad_segs=6)
    return affinity.scale(g, xfact=1.0 / k, yfact=1.0, origin=(0, 0))


def load_osm(city):
    d = json.load(open(os.path.join(src_dir(city), "osm_coast.json"), encoding="utf-8"))
    return d["elements"]


# --------------------------------------------------------------------------- raster helpers

class Grid:
    """A point grid over (lat0, lat1, lon0, lon1), ppd samples per degree, row 0 north."""

    def __init__(self, bx, ppd):
        self.lat0, self.lat1, self.lon0, self.lon1 = bx
        self.ppd = ppd
        self.ny = int(round((self.lat1 - self.lat0) * ppd)) + 1
        self.nx = int(round((self.lon1 - self.lon0) * ppd)) + 1

    def px(self, lon, lat):
        return ((lon - self.lon0) * self.ppd, (self.lat1 - lat) * self.ppd)

    def draw(self, geoms, val, img=None):
        if img is None:
            img = Image.new("L", (self.nx, self.ny), 0)
        dr = ImageDraw.Draw(img)
        for g in geoms:
            polys = g.geoms if isinstance(g, MultiPolygon) else [g]
            for p in polys:
                if p.is_empty or not isinstance(p, Polygon):
                    continue
                dr.polygon([self.px(x, y) for x, y in p.exterior.coords], fill=val)
                for r in p.interiors:
                    dr.polygon([self.px(x, y) for x, y in r.coords], fill=0 if val else 1)
        return img

    def mask(self, geoms):
        return np.asarray(self.draw(geoms, 1), np.uint8) > 0


def sample_bilinear(a, rows, cols):
    return ndimage.map_coordinates(a, [rows, cols], order=1, mode="nearest")


# --------------------------------------------------------------------------- bake

def bake(city, c):
    t0 = time.time()
    lat0, lat1, lon0, lon1 = c["near"]
    latc = 0.5 * (lat0 + lat1)
    F = Grid(c["near"], FPPD)
    N = Grid(c["near"], NPPD)
    m_h, m_cls, m_co, raws = mid_mosaic(city, c)
    ml0, ml1, mo0, mo1 = c["mid"]
    els = load_osm(city)
    # (the near box itself: the coastline ways Overpass returns run a little past it, so they cut it into faces)
    clip = sbox(lon0, lat0, lon1, lat1)

    # ---- sea faces from the coastline (land on the left of every way)
    coast_ways = [e for e in els if e["type"] == "way" and e.get("tags", {}).get("natural") == "coastline"]
    lines = [LineString(_ring(e["geometry"])) for e in coast_ways if len(e.get("geometry", [])) >= 2]
    merged = unary_union(lines).intersection(clip)
    faces = list(polygonize(unary_union([merged, clip.boundary])))
    probes = []  # (point, is_sea)
    eps = 2e-5
    for ln in lines:
        cs = list(ln.coords)
        for i in range(0, len(cs) - 1, max(1, (len(cs) - 1) // 6 or 1)):
            (x0, y0), (x1, y1) = cs[i], cs[i + 1]
            dx, dy = x1 - x0, y1 - y0
            L = math.hypot(dx, dy)
            if L < 1e-9:
                continue
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            nx_, ny_ = -dy / L, dx / L  # left normal
            probes.append((Point(mx + nx_ * eps, my + ny_ * eps), False))
            probes.append((Point(mx - nx_ * eps, my - ny_ * eps), True))
    sea_faces = []
    for f in faces:
        pf = prepared.prep(f)
        votes = [s for p, s in probes if pf.contains(p)]
        if votes:
            sea = sum(votes) * 2 > len(votes)
        else:  # bounded by the box only: ask the mid layer
            rp = f.representative_point()
            r, q = (ml1 - rp.y) * MPPD, (rp.x - mo0) * MPPD
            sea = float(sample_bilinear(m_co, np.array([r]), np.array([q]))[0]) < 0
        if sea:
            sea_faces.append(f)
    log(f"[{city}] coastline: {len(coast_ways)} ways, {len(faces)} faces, {len(sea_faces)} sea ({time.time() - t0:.0f} s)")
    open_sea = unary_union(sea_faces)

    # ---- water areas: limans the mid layer draws as sea, harbours
    named = {}
    harbours = []
    extra_sea = []
    for e in els:
        t = e.get("tags", {})
        if t.get("natural") != "water":
            continue
        a = area_of(e)
        if a is None or a.is_empty:
            continue
        a = a.intersection(clip)
        if a.is_empty:
            continue
        nm = t.get("name", "")
        if t.get("water") == "harbour":
            harbours.append(a)
            extra_sea.append(a)
            continue
        if a.area < 2e-4:  # about 2 km2: ponds and reservoirs stay the lakes' and the city water's
            continue
        rp = a.representative_point()
        r, q = (ml1 - rp.y) * MPPD, (rp.x - mo0) * MPPD
        mco = float(sample_bilinear(m_co, np.array([r]), np.array([q]))[0])
        if mco < 0 or any(k in nm for k in c["depths"]):
            extra_sea.append(a)
            for k in c["depths"]:
                if k in nm:
                    named.setdefault(k, []).append(a)
    log(f"[{city}] waters: {len(extra_sea)} drawn as sea ({', '.join(sorted(named))}), {len(harbours)} harbours")

    # ---- moles, piers, breakwaters, quays (land, at least mole_m wide)
    moles = []
    mesh_moles = []  # breakwater areas (PortProps meshes)
    mesh_lines = []  # groynes and breakwater lines (PortProps meshes)
    quay_feats = []
    beach_feats = []
    for e in els:
        t = e.get("tags", {})
        mm = t.get("man_made")
        if mm in ("pier", "breakwater", "quay", "groyne"):
            # (the short groynes and breakwaters that make the pocket beaches are
            # left out: grown to a sample's width they would close the beaches)
            a = area_of(e)
            g = e.get("geometry") or []
            from shapely import affinity
            k = math.cos(math.radians(latc))
            geo = a if a is not None and not a.is_empty else (LineString(_ring(g)) if len(g) >= 2 else None)
            if geo is None:
                continue
            gm = affinity.scale(geo, xfact=k * M_PER_DEG, yfact=M_PER_DEG, origin=(0, 0))
            mrr = gm.minimum_rotated_rectangle
            xs = list(mrr.exterior.coords) if mrr.geom_type == "Polygon" else list(gm.coords)
            span = max(math.dist(p, q) for p in xs for q in xs)
            z0_, z1_, zo0_, zo1_ = c["zone"]
            rp_ = geo.representative_point() if a is not None else geo.interpolate(0.5, normalized=True)
            in_zone = z0_ <= rp_.y <= z1_ and zo0_ <= rp_.x <= zo1_
            # breakwaters and groynes: thin stone drawn as meshes over the water
            # (PortProps "moles" / "groynes"), never land: at 1/5 they are a few
            # toy m wide, far thinner than the samples
            if mm == "breakwater" and a is not None and not a.is_empty:
                if in_zone and span >= c["mesh_min_len"]:
                    mesh_moles.append(a)
                continue
            if mm in ("groyne", "breakwater"):
                if in_zone and span >= c["mesh_min_len"]:
                    mesh_lines.append(geo if a is None or a.is_empty else a.exterior)
                continue
            if span < c["mole_min_len"]:
                # short piers (the beaches' groynes and jetties): meshes too
                if mm == "pier" and in_zone and span >= c["mesh_min_len"]:
                    if a is not None and not a.is_empty:
                        mesh_moles.append(a)
                    else:
                        mesh_lines.append(geo)
                continue
            if a is not None and not a.is_empty:
                quay_feats.append(a)
                if mm == "pier":
                    # a pier thinner than mole_m grows all round to that width
                    # (its mean width: 2 area / perimeter, in real m): it carries sheds
                    wid = 2.0 * gm.area / max(gm.length, 1e-6)
                    moles.append(deg_buf(a, max(c["pier_grow_m"], 0.5 * (c["mole_m"] - wid)), latc))
            else:
                quay_feats.append(deg_buf(geo, 8.0, latc))
                if mm == "pier":
                    moles.append(deg_buf(geo, c["mole_m"] / 2, latc))
        if t.get("natural") == "beach":
            a = area_of(e)
            if a is not None and not a.is_empty:
                beach_feats.append(a)
        if "highway" in t and re.search(c["promenade_re"], t.get("name", ""), re.I) and len(e.get("geometry") or []) >= 2:
            quay_feats.append(deg_buf(LineString(_ring(e["geometry"])), c["promenade_m"], latc))
    log(f"[{city}] {len(moles)} mole shapes, {len(quay_feats)} quay features, {len(beach_feats)} beaches ({time.time() - t0:.0f} s)")

    # ---- fine masks and the signed distance
    sea_img = F.draw([open_sea] + extra_sea, 1)
    sea_img = F.draw([unary_union(moles)], 0, sea_img) if moles else sea_img
    sea_f = np.asarray(sea_img, np.uint8) > 0
    # thin land in the coastline (moles, the Reid breakwater): grow it to the width the samples hold
    rpx = max(1, int(round(c["thin_m"] / (M_PER_DEG / FPPD))))
    yy, xx = np.mgrid[-rpx:rpx + 1, -rpx:rpx + 1]
    disk = (yy * yy + (xx * math.cos(math.radians(latc))) ** 2) <= rpx * rpx
    land_f = ~sea_f
    thin = land_f & ~ndimage.binary_opening(land_f, structure=disk)
    grown = ndimage.binary_dilation(thin, structure=disk)
    for b0, b1, b2, b3 in c.get("no_thin", []):
        # (a mole of the coastline drawn as a mesh instead: its outline from the land faces)
        x0_, y0_ = F.px(b2, b1)
        x1_, y1_ = F.px(b3, b0)
        grown[int(y0_):int(y1_) + 1, int(x0_):int(x1_) + 1] = False
        bx_ = sbox(b2, b0, b3, b1)
        for f_ in faces:
            if f_ not in sea_faces and f_.intersects(bx_):
                g_ = f_.intersection(bx_)
                for p_ in (g_.geoms if hasattr(g_, "geoms") else [g_]):
                    if isinstance(p_, Polygon) and not p_.is_empty:
                        mesh_moles.append(p_)
                        sea_f |= F.mask([p_])  # (water under the mesh)
    sea_f &= ~grown
    fills = [Polygon([(lo, la) for la, lo in f_["poly"]]) for f_ in c.get("fills", [])]
    if fills:
        sea_f &= ~F.mask(fills)
    for la, lo, rm in c.get("keep_wet", []):
        cx, cy = F.px(lo, la)
        ry = rm / (M_PER_DEG / FPPD)
        rx = ry / math.cos(math.radians(la))
        y0, y1 = int(cy - ry) - 1, int(cy + ry) + 2
        x0, x1 = int(cx - rx) - 1, int(cx + rx) + 2
        Y, X = np.mgrid[y0:y1, x0:x1]
        sea_f[y0:y1, x0:x1] |= ((Y - cy) / ry) ** 2 + ((X - cx) / rx) ** 2 <= 1.0
    log(f"[{city}] thin land grown: {int(thin.sum())} px")
    # the open sea alone (the shore kinds' distance), the bare waters
    open_f = F.mask([open_sea] + harbours) & sea_f
    bare_f = F.mask([a for k in c["bare_waters"] for a in named.get(k, [])]) & sea_f
    dy = M_PER_DEG / FPPD / SCALE
    dx = dy * math.cos(math.radians(latc))
    d_land = ndimage.distance_transform_edt(~sea_f, sampling=(dy, dx))  # land px: distance to the sea
    d_sea = ndimage.distance_transform_edt(sea_f, sampling=(dy, dx))    # sea px: distance to land
    half = 0.5 * math.sqrt(dx * dy)
    sd_f = np.where(sea_f, -(d_sea - half), d_land - half).astype(np.float32)
    del d_land, d_sea
    d_open = ndimage.distance_transform_edt(~(open_f | bare_f), sampling=(dy, dx)) * SCALE  # real m to the open sea / bare waters
    log(f"[{city}] fine grid {F.ny}x{F.nx}, sea {sea_f.mean() * 100:.1f}% ({time.time() - t0:.0f} s)")

    # ---- samples
    rows = np.arange(N.ny)[:, None] * (FPPD // NPPD)
    cols = np.arange(N.nx)[None, :] * (FPPD // NPPD)
    sd = sd_f[rows, cols]
    near_sea = d_open[rows, cols]
    del d_open
    lat_s = lat1 - np.arange(N.ny) / NPPD
    lon_s = lon0 + np.arange(N.nx) / NPPD
    LAT, LON = np.meshgrid(lat_s, lon_s, indexing="ij")
    mr, mq = (ml1 - LAT) * MPPD, (LON - mo0) * MPPD
    mid_h = sample_bilinear(m_h, mr, mq)
    mid_co = sample_bilinear(m_co, mr, mq)
    mid_cls = m_cls[np.clip(np.rint(mr).astype(int), 0, m_cls.shape[0] - 1), np.clip(np.rint(mq).astype(int), 0, m_cls.shape[1] - 1)]

    # ---- ground: GLO-30 opening where we have it, the mid layer elsewhere
    ground = mid_h.copy()
    have = np.zeros_like(ground, bool)
    for la, lo in c["dem"]:
        a = read_cop30(os.path.join(src_dir(city), f"cop30_N{la:02d}E{lo:03d}.tif"))
        sea_dem = a <= 0.5
        w = max(3, int(round(c["open_m"] / (M_PER_DEG / 3600 * math.cos(math.radians(la + 0.5))))) | 1)
        op = ndimage.grey_opening(a, size=(w, w))
        op = np.where(sea_dem, a, op)  # (the open sea and the limans keep their surface)
        op = ndimage.gaussian_filter(op, 1.0)
        r = (la + 1 - LAT) * 3600
        q = (LON - lo) * 3600
        inside = (r >= 0) & (r <= 3599) & (q >= 0) & (q <= 3599)
        g = sample_bilinear(op, np.clip(r, 0, 3599), np.clip(q, 0, 3599))
        # a short blend into the mid layer at the DEM tile's own edges
        edge = np.minimum.reduce([r, 3599 - r, q, 3599 - q]) / 3600.0
        wgt = np.clip(edge / 0.02, 0, 1) * inside
        ground = np.where(inside, ground * (1 - wgt) + g * wgt, ground)
        have |= inside
    log(f"[{city}] ground model ({time.time() - t0:.0f} s)")

    # ---- shore kinds (inside the zone)
    z0, z1, zo0, zo1 = c["zone"]
    zone = (LAT >= z0) & (LAT <= z1) & (LON >= zo0) & (LON <= zo1)
    km = c["kind_m"]
    def feat_dist(geoms):
        if not geoms:
            return np.full(LAT.shape, 1e9, np.float32)
        m = F.mask(geoms)
        dfine = ndimage.distance_transform_edt(~m, sampling=(dy, dx)) * SCALE
        return dfine[rows, cols]
    d_beach = feat_dist(beach_feats)
    d_quay = feat_dist(quay_feats)
    port_m = F.mask([Polygon([(lo, la) for la, lo in c["port"]])] + ([Polygon([(lo, la) for la, lo in c["port2"]])] if c.get("port2") else []))[rows, cols]
    land = sd > 0
    # (a promenade runs right behind most beaches: the beach keeps the samples it shares)
    is_beach = land & zone & (d_beach < km) & (d_beach <= d_quay + km)
    fill_m = F.mask(fills)[rows, cols] if fills else np.zeros_like(land)
    is_quay = land & zone & ~is_beach & ((d_quay < km) | fill_m | (port_m & (ground < c["port_low"]) & (near_sea < c["port_m"])))
    is_bare = land & zone & ~is_beach & ~is_quay & (near_sea < c["shore_m"])
    cls = np.where(land, mid_cls, C_WATER).astype(np.uint8)
    cls = np.where(cls == C_WATER, np.where(land, 5, C_WATER), cls).astype(np.uint8)  # (land the mid layer called water)
    cls[is_bare] = C_BARE
    cls[is_quay] = C_QUAY
    cls[is_beach] = C_BEACH
    log(f"[{city}] kinds: quay {int(is_quay.sum())}, beach {int(is_beach.sum())}, bare {int(is_bare.sum())} samples")

    # ---- heights
    h = ground.copy()
    h = np.where(land, np.maximum(h, 1.0), h)
    qh0, qh1 = c["quay_h"]
    h = np.where(is_quay, np.clip(h, qh0, qh1), h)
    # moles over the sea in the DEM: their deck
    # The sea floor in real m. The water shader colours by it (vdepth: turquoise and
    # see-through sand under about 2.5 m, mid blue to 14 m) but never deeper than 4% of the
    # distance from the shore, so the shelf still ramps out from every beach; the harbour
    # basins are class 15 (deep water), which lifts that ramp right off their quay walls.
    dist_r = np.maximum(-sd, 0) * SCALE  # real m from the shore
    dep = np.clip(c["shelf_d0"] + c["shelf"] * dist_r, 1.0, None)
    dep = np.maximum(dep, np.minimum(-mid_h, c["shelf_max"]))  # (the real bathymetry where it is deeper)
    harb = F.mask(harbours)[rows, cols] if harbours else np.zeros_like(land)
    dep = np.where(harb, c["harbour_d"], dep)
    for k, geoms in named.items():
        m = F.mask(geoms)[rows, cols]
        dep = np.where(m, np.minimum(c["depths"][k], 0.2 + 0.01 * dist_r), dep)
    # deep water: the harbours and the port's outer basins inside its moles (the port
    # polygons grown by deep_m)
    pp = [Polygon([(lo, la) for la, lo in c["port"]])] + ([Polygon([(lo, la) for la, lo in c["port2"]])] if c.get("port2") else [])
    port_wet = F.mask([deg_buf(p_, c["deep_m"], latc) for p_ in pp])[rows, cols]
    deep = (harb | port_wet) & ~land
    dep = np.where(port_wet & ~land, np.maximum(dep, c["harbour_d"]), dep)
    h = np.where(land, h, np.minimum(-dep, -0.3))
    cls[deep] = C_DEEP
    # pale salt water (class 8 + a tree nibble, the water shader's cpx_water)
    pale = np.zeros_like(land)
    for k in c.get("pale_waters", []):
        for g_ in named.get(k, []):
            pale |= F.mask([g_])[rows, cols]
    pale &= ~land

    # ---- relief gain: Heightfield._tile_base adds k (rd - raw_g) to the global base, k =
    # TILE_GAIN by the global ruggedness (0.18 on gentle land); store the relief so the game
    # draws it at c["relief_gain"] (land only, at least 1 m: lower land is the floor anyway)
    from bake_detail import global_raw  # noqa: PLC0415
    rg = np.maximum(global_raw(LAT, LON), 0.0)
    aux = np.fromfile(os.path.join(ROOT, "data", "world", "shape_aux.bin"), np.uint8).reshape(2048, 4096, 4)
    rug = sample_bilinear(aux[..., 2].astype(np.float32) / 255.0, (90.0 - LAT) / 180.0 * 2048 - 0.5, (LON + 180.0) / 360.0 * 4096 - 0.5)
    del aux
    t_ = np.clip((rug - 0.05) / 0.35, 0, 1)
    k = 0.18 + (0.45 - 0.18) * t_ * t_ * (3 - 2 * t_)
    gw = box_weight(LAT, LON, c["zone"], c["relief_fade"])
    f = 1.0 + (c["relief_gain"] / k - 1.0) * gw
    h = np.where(land, np.maximum(rg + (np.maximum(h, 0.0) - rg) * f, 1.0), h)
    log(f"[{city}] relief gain {c['relief_gain']} (game k {float(k[gw > 0.5].min()):.2f} to {float(k[gw > 0.5].max()):.2f})")

    # ---- feather into the mid layer
    w = box_weight(LAT, LON, c["near"], NEAR_FEATHER)
    m_h_q = np.rint(mid_h)
    m_co_q = dec_coast(enc_coast(mid_co))
    h = m_h_q * (1 - w) + h * w
    coast = m_co_q * (1 - w) + sd * w
    cls = np.where(w < 0.5, np.where(mid_co >= 0, mid_cls, C_WATER), cls).astype(np.uint8)
    log(f"[{city}] near grid {N.ny}x{N.nx} ({time.time() - t0:.0f} s)")

    # ---- tiles
    blobs = []
    keys = []
    cover = (cls.astype(np.uint16) * 16 + np.where(pale & (cls == C_WATER), 1, 0)).astype(np.uint8)
    coast8 = enc_coast(coast)
    for ty in range(int(round((90 - lat1) / 0.5)), int(round((90 - lat0) / 0.5))):
        for tx in range(int(round((lon0 + 180) / 0.5)), int(round((lon1 + 180) / 0.5))):
            r = int(round(((90 - ty * 0.5) - lat1) * -NPPD))
            q = int(round((-180 + tx * 0.5 - lon0) * NPPD))
            sl = (slice(r, r + 601), slice(q, q + 601))
            if (coast8[sl] < 128).all():
                continue
            blobs.append(encode_tile((h[sl], cover[sl], coast8[sl])))
            keys.append((0, tx, ty))
    n_near = len(blobs)
    for (tx, ty), b in sorted(raws.items()):  # the mid layer itself (the web has no global layer)
        hh = b[..., :2].copy().view("<i2")[..., 0].astype(np.float32)
        blobs.append(encode_tile((hh, b[..., 2].copy(), b[..., 3].copy())))
        keys.append((1, tx, ty))
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, city + ".bin")
    off = 12 + 16 * len(blobs)
    with open(path + ".part", "wb") as f:
        f.write(b"SBDT" + struct.pack("<II", 1, len(blobs)))
        for (tier, tx, ty), b in zip(keys, blobs):
            f.write(struct.pack("<BBHHHII", tier, 0, tx, ty, 0, off, len(b)))
            off += len(b)
        for b in blobs:
            f.write(b)
    os.replace(path + ".part", path)
    nb = sum(len(b) for b in blobs[:n_near])
    mb = sum(len(b) for b in blobs[n_near:])
    log(f"[{city}] {path}: {os.path.getsize(path) / 1e6:.2f} MB, near {n_near} tiles {nb / 1e6:.2f} MB, mid {len(blobs) - n_near} tiles {mb / 1e6:.2f} MB ({time.time() - t0:.0f} s)")
    # preview: hillshade, sea, kinds
    gy, gx = np.gradient(h)
    shade = np.clip(0.6 + (gx - gy) * 0.05, 0, 1)
    rgb = np.stack([shade * 225, shade * 220, shade * 200], -1)
    rgb[coast < 0] = (60, 110, 170)
    rgb[cls == C_QUAY] = (150, 150, 150)
    rgb[cls == C_BEACH] = (240, 210, 120)
    rgb[cls == C_BARE] = rgb[cls == C_BARE] * np.array([0.7, 1.0, 0.7])
    Image.fromarray(rgb.astype(np.uint8)).save(os.path.join(src_dir(city), f"preview_{city}.png"))
    # the mesh moles and groynes into PortProps' file (its other keys kept)
    pp_path = os.path.join(ROOT, "data", "world", "odbl", "port_props.json")
    pp = json.load(open(pp_path, encoding="utf-8")) if os.path.exists(pp_path) else {"version": 1, "ports": [], "berths": []}
    def ring_ll(r):
        return [[round(y, 6), round(x, 6)] for x, y in list(r.coords)[:-1]]
    pp["moles"] = [{"ring": ring_ll(m.exterior), "h": c["mole_h"]} for m in mesh_moles if m.area > 0]
    pp["groynes"] = [{"pts": [[round(y, 6), round(x, 6)] for x, y in ln.coords], "w": c["groyne_w"], "h": c["groyne_h"]} for ln in mesh_lines]
    # storage tanks and silos (OSM, cache osm_port2.json): centre, real diameter (m), kind
    tanks = []
    t2 = os.path.join(src_dir(city), "osm_port2.json")
    if os.path.exists(t2):
        z0_, z1_, zo0_, zo1_ = c["zone"]
        for e in json.load(open(t2, encoding="utf-8"))["elements"]:
            t = e.get("tags", {})
            kind = "silo" if (t.get("man_made") == "silo" or t.get("building") == "silo") else ("tank" if (t.get("man_made") == "storage_tank" or t.get("building") in ("storage_tank", "tank")) else "")
            if not kind:
                continue
            if e["type"] == "node":
                la_, lo_, dm = e["lat"], e["lon"], 12.0
            else:
                a_ = area_of(e)
                if a_ is None or a_.is_empty:
                    continue
                cp = a_.centroid
                la_, lo_ = cp.y, cp.x
                dm = 2.0 * math.sqrt(a_.area * M_PER_DEG * M_PER_DEG * math.cos(math.radians(la_)) / math.pi)
            if z0_ <= la_ <= z1_ and zo0_ <= lo_ <= zo1_ and dm >= 4.0:
                tanks.append([round(la_, 6), round(lo_, 6), round(min(dm, 60.0), 1), kind])
    tanks.sort(key=lambda r: -r[2])  # (the biggest first: they win the spacing)
    pp["tanks"] = tanks
    pp["moles_source"] = "tools/bake_city_near.py %s: OpenStreetMap man_made=breakwater areas, the Reid mole of the coastline, man_made=groyne and breakwater lines" % city
    with open(pp_path, "w", encoding="utf-8", newline=chr(10)) as f:
        f.write(json.dumps(pp, ensure_ascii=False, indent=1) + chr(10))
    log(f"[{city}] port_props.json: {len(pp['moles'])} mesh moles, {len(pp['groynes'])} groynes, {len(pp['tanks'])} tanks and silos")
    np.savez_compressed(os.path.join(src_dir(city), "near_grid.npz"), h=h.astype(np.float32), coast=coast.astype(np.float32), cls=cls)
    return {"label": c["label"], "file": city + ".bin", "near": list(c["near"]), "mid": list(c["mid"]),
            "bytes": os.path.getsize(path), "tiles": {"near": n_near, "mid": len(blobs) - n_near},
            "tile_bytes": {"near": nb, "mid": mb}, "baker": "tools/bake_city_near.py"}


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    names = args or list(CITIES)
    if "--fetch" in sys.argv:
        for n in names:
            fetch(n, CITIES[n])
        return
    man_path = os.path.join(OUT, "packs.json")
    man = json.load(open(man_path)) if os.path.exists(man_path) else {}
    man.setdefault("packs", {})
    man.update({"version": 1, "kind": "city_near",
                "encoding": "RGBA WebP lossless: R,G int16 LE real m; B class*16+tree (12 quay, 13 bare shore, 14 beach, 15 deep harbour water; water 8 with a nibble: pale salt water); A sqrt coast (+-2500 toy m)",
                "coast_range": COAST_RANGE})
    for n in names:
        man["packs"][n] = bake(n, CITIES[n])
        with open(man_path, "w") as f:
            json.dump(man, f, indent=1)


if __name__ == "__main__":
    main()
