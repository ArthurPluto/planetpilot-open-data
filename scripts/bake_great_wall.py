"""Bakes the Ming Great Wall route for the streamed wall (scripts/world/great_wall.gd).

Source: OpenStreetMap (ODbL 1.0) through the Overpass API, the ways tagged
historic=citywalls / barrier=wall / barrier=city_wall named 长城 or
"Great Wall" between 35-43.5 N and 97-120.5 E. Fetch (one query, ~18 MB):

    [out:json][timeout:300];
    (
      way["historic"~"citywalls|wall|fortification"]["name"~"长城|Great Wall"](35.0,97.0,43.5,120.5);
      way["barrier"="wall"]["name"~"长城|Great Wall"](35.0,97.0,43.5,120.5);
      way["historic"="citywalls"]["wikidata"="Q12501"](35.0,97.0,43.5,120.5);
      way["wall"="great_wall"](35.0,97.0,43.5,120.5);
    );
    out tags geom;

    python -I tools/bake_great_wall.py <overpass.json>... [--png map.png]

(v55 used three queries: the one above, every historic=citywalls /
barrier=city_wall way and relation member in the box, and the ways tagged
highway / historic / man_made=wall named 长城; unnamed open walls join when
they lie within NEAR_REAL of the named route.)

Keeps the Ming wall: drops the ways named or described as another dynasty's
wall (Qin, Han, Northern Qi, Song, Jin border trenches), walkways and areas,
anything north of 41.6 N (the Han / Jin lines of Inner Mongolia) and east of
Shanhaiguan (the Liaodong wall). Joins the ways into chains, simplifies them
(Douglas-Peucker, 150 m real = 30 m on the 1/5 planet), drops loose bits under
1 km real, and cuts the chains out of the Mutianyu hero model's corridor
(data/landmarks/great_wall.json, drawn by Landmarks) with a short link from
each hero end to the nearest cut end. Kind per chain: rammed earth where OSM
says material=soil or, untagged, west of 110.8 E (Shaanxi, Ningxia, Gansu);
brick / stone elsewhere.

Writes data/world/odbl/great_wall.bin (GWL1, little endian):
    "GWL1", u32 chains, u32 points
    per chain: u8 kind (0 brick, 1 earth, 2 hero link), u8 0, u16 0, u32 n,
               n x (i32 lat, i32 lon) in 1e-6 degrees
and appends its line to data/world/odbl/SOURCES.txt.
"""
import argparse
import datetime
import json
import math
import os
import re
import struct
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "world", "odbl", "great_wall.bin")
SOURCES = os.path.join(ROOT, "data", "world", "odbl", "SOURCES.txt")
HERO = os.path.join(ROOT, "data", "landmarks", "great_wall.json")
SITES = os.path.join(ROOT, "data", "landmarks", "sites.json")

R_REAL = 6371008.8
SCALE = 5.0
TOL_REAL = 150.0       # Douglas-Peucker tolerance, real m (30 m game)
LAT_MAX = 41.6
LON_MAX = 119.85       # Old Dragon's Head is at 119.794 E
EARTH_LON = 110.8
SNAP_REAL = 60.0      # way ends closer than this (real m) join
GAP_REAL = 1500.0     # mapping gaps up to this (real m) get a straight link
MIN_PIECE_REAL = 150.0
HERO_CUT = 160.0       # game m each side of the hero path
HERO_LINK = 2500.0     # game m: the farthest cut end a hero end links to

NOT_MING = re.compile(r"(Jin dynasty|Han dynasty|Qin dynasty|Qin Great|Song dynasty|Northern Qi|"
                      r"金界壕|金长城|汉长城|秦长城|宋长城|北齐|北魏|战国|燕长城|赵长城|魏长城|隋长城|界壕|Genghis)")
KIND_BRICK, KIND_EARTH, KIND_LINK = 0, 1, 2
WALL_NAME = re.compile(r"(长城|Great Wall)")
DRAGON_HEAD = (39.9662, 119.7966)  # the tip of Old Dragon's Head, in the sea
NEAR_REAL = 2500.0     # an unnamed wall this close to the named route joins it
# route clean-up (v55 review: no spaghetti of parallel / side pieces)
DUP_REAL = 300.0       # a stretch this close to a longer chain is a duplicate
SAMPLE_REAL = 60.0     # sample spacing of the duplicate / spur tests
KEEP_RUN_REAL = 800.0  # a chain's stretch left after the duplicate cut keeps over this
SPUR_REAL = 3000.0     # a dead-end side piece under this is a spur
SPUR_FREE = 500.0      # an end with no other chain this close is loose
ISLAND_REAL = 1000.0   # a lone bit (both ends loose) under this goes
T_LINK_REAL = 450.0    # a cut-back branch end links to the side of a chain this close


def hav(a, b):
    """Great-circle distance (real m) between (lat, lon) pairs."""
    p1, p2 = math.radians(a[0]), math.radians(b[0])
    dp = p2 - p1
    dl = math.radians(b[1] - a[1])
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R_REAL * math.asin(min(1.0, math.sqrt(h)))


def length(pts):
    return sum(hav(a, b) for a, b in zip(pts, pts[1:]))


def keep_way(e):
    t = e.get("tags", {})
    g = e.get("geometry") or []
    if len(g) < 2:
        return False
    if t.get("area") == "yes" or "building" in t:
        return False
    # the walkway on top of a wall section is kept (it is the wall); roads
    # named after the wall (长城路) are not walls at all
    if not ("historic" in t or "barrier" in t or t.get("man_made") == "wall"):
        return False
    if "highway" in t and t["highway"] not in ("path", "footway", "steps", "pedestrian"):
        return False
    text = " ".join(t.get(k, "") for k in ("name", "name:en", "name:zh", "alt_name", "description", "old_name"))
    if NOT_MING.search(text):
        return False
    for p in g:
        if p["lat"] > LAT_MAX or p["lon"] > LON_MAX:
            return False
    return True


def kind_of(t, pts):
    m = t.get("material", "")
    if m in ("soil", "earth", "clay", "rammed_earth", "adobe"):
        return KIND_EARTH
    if m in ("stone", "brick", "masonry"):
        return KIND_BRICK
    lon = sum(p[1] for p in pts) / len(pts)
    return KIND_EARTH if lon < EARTH_LON else KIND_BRICK


def _clusters(chains, reach):
    """Endpoint cluster id per (chain, end): ends closer than reach (real m) share one."""
    cell = reach / 111000.0
    grid = {}
    ends = []
    for i, c in enumerate(chains):
        for e in (0, 1):
            ends.append((i, e, c[0] if e == 0 else c[-1]))
    parent = list(range(len(ends)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for n, (_, _, p) in enumerate(ends):
        gk = (int(p[0] // cell), int(p[1] // cell))
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                for m in grid.get((gk[0] + dy, gk[1] + dx), []):
                    if hav(p, ends[m][2]) < reach:
                        parent[find(n)] = find(m)
        grid.setdefault(gk, []).append(n)
    return {(i, e): find(n) for n, (i, e, _) in enumerate(ends)}


def join(ways, snap=SNAP_REAL):
    """Joins ways (lists of (lat, lon)) end to end into chains of one kind,
    through every point where exactly two ends meet (within snap real m)."""
    chains = [list(w[1]) for w in ways]
    kinds = [w[0] for w in ways]
    alive = [True] * len(chains)
    changed = True
    while changed:
        changed = False
        cl = _clusters(chains, snap)
        members = {}
        for (i, e), c in cl.items():
            if alive[i]:
                members.setdefault(c, []).append((i, e))
        used = set()
        for c, m in members.items():
            if len(m) != 2:
                continue
            (a, ea), (b, eb) = m
            if a == b or a in used or b in used or kinds[a] != kinds[b]:
                continue
            ca = chains[a] if ea == 1 else chains[a][::-1]
            cb = chains[b] if eb == 0 else chains[b][::-1]
            chains[a] = ca + cb[1:] if hav(ca[-1], cb[0]) < 1.0 else ca + cb
            alive[b] = False
            used.update((a, b))
            changed = True
        chains = [c for i, c in enumerate(chains) if alive[i]]
        kinds = [k for i, k in enumerate(kinds) if alive[i]]
        alive = [True] * len(chains)
    return list(zip(kinds, chains))


class Pts:
    """A spatial hash of sample points (local metres, cell = reach), tagged."""

    def __init__(self, reach):
        self.reach = reach
        self.g = {}

    @staticmethod
    def xy(p):
        # metres east / north, east scaled by cos(40 N): plenty for 300 m tests
        return (p[1] * 111195.0 * 0.766, p[0] * 111195.0)

    def add(self, p, tag):
        x, y = self.xy(p)
        self.g.setdefault((int(x // self.reach), int(y // self.reach)), []).append((x, y, tag))

    def nearest(self, p, reach=None, skip=None):
        """(distance, tag) of the nearest point within reach (real m) whose tag
        is not skip (a chain index, or the first item of a tuple tag)."""
        r = reach or self.reach
        x, y = self.xy(p)
        cx, cy = int(x // self.reach), int(y // self.reach)
        n = int(math.ceil(r / self.reach))
        best, bt = r, None
        for dy in range(-n, n + 1):
            for dx in range(-n, n + 1):
                for qx, qy, t in self.g.get((cx + dx, cy + dy), ()):
                    if skip is not None and (t[0] if isinstance(t, tuple) else t) == skip:
                        continue
                    d = math.hypot(qx - x, qy - y)
                    if d < best:
                        best, bt = d, t
        return best, bt

    def near(self, p, reach=None, skip=None):
        return self.nearest(p, reach, skip)[1] is not None


def densify(pts, step):
    out = [pts[0]]
    for a, b in zip(pts, pts[1:]):
        n = max(1, int(hav(a, b) / step))
        for i in range(1, n + 1):
            t = i / n
            out.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
    return out


def dedupe(chains):
    """Drops near-parallel duplicates: longest chains first, every stretch of a
    chain that runs within DUP_REAL of a chain already kept is cut out (OSM maps
    both faces of a wall, an old and a restored line, the walkway, short second
    lines beside the main one); what is left keeps when over KEEP_RUN_REAL."""
    order = sorted(range(len(chains)), key=lambda i: -length(chains[i][1]))
    grid = Pts(DUP_REAL)
    out = []
    for i in order:
        kind, pts = chains[i]
        dense = densify(pts, SAMPLE_REAL)
        runs, run = [], []
        for p in dense:
            if not grid.near(p):
                run.append(p)
            elif run:
                runs.append(run)
                run = []
        if run:
            runs.append(run)
        for r in runs:
            if len(r) > 1 and length(r) >= KEEP_RUN_REAL:
                out.append((kind, dp(r, TOL_REAL)))
        for p in dense:
            grid.add(p, i)
    return out


def prune_spurs(chains):
    """Drops short dead-end spurs: a chain under SPUR_REAL with one end loose
    (no other chain within SPUR_FREE) and the other against another chain's
    side (not its end: an end-to-end continuation is the main line), and lone
    bits under ISLAND_REAL with both ends loose. Two rounds."""
    for _round in range(2):
        allp = Pts(250.0)
        ends = Pts(250.0)
        for ci, (_, pts) in enumerate(chains):
            for p in densify(pts, SAMPLE_REAL):
                allp.add(p, ci)
            ends.add(pts[0], ci)
            ends.add(pts[-1], ci)
        keep = []
        for ci, (kind, pts) in enumerate(chains):
            L = length(pts)
            if L >= SPUR_REAL:
                keep.append((kind, pts))
                continue
            st = []
            for p in (pts[0], pts[-1]):
                if ends.near(p, GAP_REAL, ci):
                    st.append("end")
                elif allp.near(p, SPUR_FREE, ci):
                    st.append("side")
                else:
                    st.append("loose")
            if "loose" in st and "side" in st:
                continue
            if st == ["loose", "loose"] and L < ISLAND_REAL:
                continue
            keep.append((kind, pts))
        chains = keep
    return chains


def bridge(chains, gap=GAP_REAL):
    """Short straight links over the mapping gaps: loose chain ends paired end
    to end, nearest pairs first, each end used once (no fans of links to one
    point); an end left over gets a short link to the side of the nearest
    chain within T_LINK_REAL (a branch dedupe() cut back from the main line)."""
    snap = Pts(SNAP_REAL)
    for ci, (_, pts) in enumerate(chains):
        snap.add(pts[0], ci)
        snap.add(pts[-1], ci)
    loose = []
    for ci, (kind, pts) in enumerate(chains):
        for e in (0, 1):
            p = pts[0] if e == 0 else pts[-1]
            if not snap.near(p, SNAP_REAL, ci):
                loose.append((ci, kind, p))
    grid = Pts(gap)
    for n, (ci, _, p) in enumerate(loose):
        grid.add(p, n)
    pairs = []
    for n, (ci, _, p) in enumerate(loose):
        x, y = Pts.xy(p)
        cx, cy = int(x // gap), int(y // gap)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                for qx, qy, m in grid.g.get((cx + dx, cy + dy), ()):
                    if m > n and loose[m][0] != ci:
                        d = math.hypot(qx - x, qy - y)
                        if d < gap:
                            pairs.append((d, n, m))
    pairs.sort()
    used = set()
    links = []
    for d, n, m in pairs:
        if n in used or m in used:
            continue
        used.update((n, m))
        links.append((loose[n][1], [loose[n][2], loose[m][2]]))
    side = Pts(T_LINK_REAL)
    for ci, (_, pts) in enumerate(chains):
        for p in densify(pts, SAMPLE_REAL):
            side.add(p, (ci, p))
    for n, (ci, kind, p) in enumerate(loose):
        if n in used:
            continue
        d, t = side.nearest(p, T_LINK_REAL, ci)
        if t is not None and d > SNAP_REAL * 0.5:
            links.append((kind, [p, t[1]]))
    return links


def to_xy(p, lat0):
    k = R_REAL * math.pi / 180.0
    return (p[1] * k * math.cos(math.radians(lat0)), p[0] * k)


def dp(pts, tol):
    """Douglas-Peucker on (lat, lon) with a local equirectangular metric."""
    if len(pts) < 3:
        return pts
    lat0 = pts[0][0]
    xy = [to_xy(p, lat0) for p in pts]
    keep = [False] * len(pts)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        i, j = stack.pop()
        ax, ay = xy[i]
        bx, by = xy[j]
        dx, dy = bx - ax, by - ay
        ll = dx * dx + dy * dy
        best, bi = -1.0, -1
        for k in range(i + 1, j):
            px, py = xy[k]
            if ll < 1e-9:
                d = math.hypot(px - ax, py - ay)
            else:
                t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / ll))
                d = math.hypot(px - ax - t * dx, py - ay - t * dy)
            if d > best:
                best, bi = d, k
        if best > tol:
            keep[bi] = True
            stack.append((i, bi))
            stack.append((bi, j))
    return [p for p, k in zip(pts, keep) if k]


def hero_path():
    """The Mutianyu hero model's path in (lat, lon) (Landmarks.drape_point)."""
    h = json.load(open(HERO, encoding="utf-8"))
    s = json.load(open(SITES, encoding="utf-8")).get("great_wall", {})
    lat0, lon0 = float(s.get("lat", h["lat"])), float(s.get("lon", h["lon"]))
    face = math.radians(float(s.get("turn", 0.0)))
    k = (R_REAL / SCALE) * math.pi / 180.0
    out = []
    for x, z in h["path"]:
        # Basis(UP, -face) * (x, 0, z): east = x cos + z sin, south = -x sin + z cos
        e = x * math.cos(face) + z * math.sin(face)
        so = -x * math.sin(face) + z * math.cos(face)
        out.append((lat0 - so / k, lon0 + e / (k * math.cos(math.radians(lat0)))))
    return out


def game_xy(p, ref):
    k = (R_REAL / SCALE) * math.pi / 180.0
    return ((p[1] - ref[1]) * k * math.cos(math.radians(ref[0])), (p[0] - ref[0]) * k)


def seg_dist(q, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    ll = dx * dx + dy * dy
    t = 0.0 if ll < 1e-9 else max(0.0, min(1.0, ((q[0] - a[0]) * dx + (q[1] - a[1]) * dy) / ll))
    return math.hypot(q[0] - a[0] - t * dx, q[1] - a[1] - t * dy)


def cut_hero(chains, hero):
    """Splits the chains where they run inside the hero corridor; links the hero ends."""
    ref = hero[len(hero) // 2]
    hxy = [game_xy(p, ref) for p in hero]
    reach = max(math.hypot(*p) for p in hxy) + HERO_CUT

    def inside(p):
        q = game_xy(p, ref)
        if math.hypot(*q) > reach:
            return False
        return min(seg_dist(q, a, b) for a, b in zip(hxy, hxy[1:])) < HERO_CUT

    out = []
    cut_ends = []
    for kind, pts in chains:
        # densify near the hero so the cut is clean
        if any(math.hypot(*game_xy(p, ref)) < reach + 2000.0 for p in pts):
            dense = [pts[0]]
            for a, b in zip(pts, pts[1:]):
                n = max(1, int(hav(a, b) / SCALE / 20.0))
                for i in range(1, n + 1):
                    t = i / n
                    dense.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
            run = []
            was_in = False
            for p in dense:
                if inside(p):
                    if run:
                        cut_ends.append(run[-1])
                    if len(run) > 1:
                        out.append((kind, dp(run, TOL_REAL)))
                    run = []
                    was_in = True
                else:
                    if was_in:
                        cut_ends.append(p)
                    was_in = False
                    run.append(p)
            if len(run) > 1:
                out.append((kind, dp(run, TOL_REAL)))
        else:
            out.append((kind, pts))
    links = []
    for end in (hero[0], hero[-1]):
        best, bp = HERO_LINK, None
        for c in cut_ends:
            d = math.hypot(*[a - b for a, b in zip(game_xy(c, ref), game_xy(end, ref))])
            if d < best:
                best, bp = d, c
        if bp is not None:
            links.append((KIND_LINK, [end, bp]))
    return out, links


def write_bin(chains, path):
    body = bytearray()
    n_pts = 0
    for kind, pts in chains:
        body += struct.pack("<BBHI", kind, 0, 0, len(pts))
        for la, lo in pts:
            body += struct.pack("<ii", int(round(la * 1e6)), int(round(lo * 1e6)))
        n_pts += len(pts)
    with open(path, "wb") as f:
        f.write(b"GWL1" + struct.pack("<II", len(chains), n_pts) + bytes(body))
    return n_pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", nargs="+")
    ap.add_argument("--png", default="")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    ways_all = {}
    for src in a.src:
        d = json.load(open(src, encoding="utf-8"))
        for e in d.get("elements", []):
            if e.get("type") == "way":
                ways_all[e["id"]] = e
    kept = []
    dropped = []
    cand = []
    for e in ways_all.values():
        g = e.get("geometry") or []
        pts = [(float(p["lat"]), float(p["lon"])) for p in g if p]
        t = e.get("tags", {})
        if not WALL_NAME.search(" ".join(t.get(k, "") for k in ("name", "name:en", "name:zh", "alt_name"))):
            # an unnamed wall (no name at all, open): a Great Wall piece when it
            # lies along the named route (below); named others are town walls
            if "name" not in t and len(pts) > 1 and hav(pts[0], pts[-1]) > 1.0 and keep_way(e):
                cand.append(pts)
            elif len(pts) > 1:
                dropped.append(pts)
            continue
        if keep_way(e):
            kept.append((kind_of(t, pts), pts))
        elif len(pts) > 1:
            dropped.append(pts)
    # unnamed pieces within NEAR_REAL of the named route, grown in rounds
    cell = NEAR_REAL / 111000.0
    grid = set()
    def mark(pts):
        for p in pts:
            grid.add((int(p[0] // cell), int(p[1] // cell)))
    def near(pts):
        for p in pts[:: max(1, len(pts) // 8)] + [pts[-1]]:
            gk = (int(p[0] // cell), int(p[1] // cell))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if (gk[0] + dy, gk[1] + dx) in grid:
                        return True
        return False
    for _, pts in kept:
        mark(pts)
    added = 0
    for _round in range(4):
        rest = []
        for pts in cand:
            if near(pts):
                kept.append((kind_of({}, pts), pts))
                mark(pts)
                added += 1
            else:
                rest.append(pts)
        cand = rest
    dropped += cand
    print("unnamed wall pieces along the route: %d added, %d left out" % (added, len(cand)))
    chains = join(kept)
    raw_km = sum(length(c[1]) for c in chains) / 1000.0
    out = []
    for kind, pts in chains:
        L = length(pts)
        closed = hav(pts[0], pts[-1]) < 1.0
        if L < MIN_PIECE_REAL or (closed and L < 3000.0):
            continue
        out.append((kind, dp(pts, TOL_REAL)))
    n0, km0 = len(out), sum(length(c[1]) for c in out) / 1000.0
    out = prune_spurs(dedupe(out))
    print("clean-up: %d chains %.0f km real -> %d chains %.0f km (duplicates and spurs dropped)" % (
        n0, km0, len(out), sum(length(c[1]) for c in out) / 1000.0))
    gaps = bridge(out)
    print("gap links: %d, %.0f km real" % (len(gaps), sum(length(c[1]) for c in gaps) / 1000.0))
    out += gaps
    # Old Dragon's Head: the wall walks out into the Bohai sea; OSM's line
    # stops on the beach, so the nearest chain end within 2 km gets the tip
    best, bi = 2000.0, None
    for ci, (kind, pts) in enumerate(out):
        for e in (0, -1):
            dd = hav(pts[e], DRAGON_HEAD)
            if dd < best:
                best, bi = dd, (ci, e)
    if bi is not None:
        pts = out[bi[0]][1]
        if bi[1] == 0:
            pts.insert(0, DRAGON_HEAD)
        else:
            pts.append(DRAGON_HEAD)
    out, links = cut_hero(out, hero_path())
    out = [c for c in out if length(c[1]) >= 200.0] + links
    n_pts = write_bin(out, a.out)
    km = {KIND_BRICK: 0.0, KIND_EARTH: 0.0, KIND_LINK: 0.0}
    for kind, pts in out:
        km[kind] += length(pts) / 1000.0
    tot = sum(km.values())
    print("ways %d kept %d -> chains %d (%d after filters), points %d, %.0f KB" % (
        len(ways_all), len(kept), len(chains), len(out), n_pts, os.path.getsize(a.out) / 1024.0))
    print("real km: %.0f joined, %.0f kept (brick %.0f, earth %.0f, hero links %.1f); game km %.0f" % (
        raw_km, tot, km[KIND_BRICK], km[KIND_EARTH], km[KIND_LINK], tot / SCALE))
    for nm, p in (("Old Dragon's Head", (39.9686, 119.7944)), ("Jiayuguan", (39.8013, 98.2167)), ("Mutianyu", (40.4319, 116.5704))):
        best = min(hav(p, q) for _, pts in out for q in pts)
        print("  nearest wall point to %s: %.1f km real" % (nm, best / 1000.0))
    if a.png:
        from PIL import Image, ImageDraw
        W, H = 2400, 900
        lo0, lo1, la0, la1 = 97.0, 120.5, 35.0, 43.5
        img = Image.new("RGB", (W, H), (250, 250, 247))
        dr = ImageDraw.Draw(img)
        px = lambda p: ((p[1] - lo0) / (lo1 - lo0) * W, (la1 - p[0]) / (la1 - la0) * H)
        for pts in dropped:
            dr.line([px(p) for p in pts], fill=(215, 215, 215), width=1)
        col = {KIND_BRICK: (70, 70, 70), KIND_EARTH: (200, 140, 60), KIND_LINK: (230, 40, 40)}
        for kind, pts in out:
            dr.line([px(p) for p in pts], fill=col[kind], width=2)
        dr.text((10, 10), "Ming Great Wall (OSM, ODbL): dark brick/stone, tan rammed earth, red hero links, grey dropped", fill=(0, 0, 0))
        img.save(a.png)
    stamp = datetime.date.today().isoformat()
    line = ("great_wall.bin: OpenStreetMap via the Overpass API (ways historic=citywalls / barrier=wall named "
            "长城 / Great Wall, Ming lines Jiayuguan to Shanhaiguan), python tools/bake_great_wall.py, baked %s" % stamp)
    lines = open(SOURCES, encoding="utf-8").read().splitlines() if os.path.exists(SOURCES) else []
    lines = [ln for ln in lines if not ln.startswith("great_wall.bin:")] + [line]
    with open(SOURCES, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
