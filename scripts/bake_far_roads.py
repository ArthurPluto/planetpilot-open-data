"""Bake the far-road line data: data/world/odbl/farroads.bin (FRD1).

Source: data/world/odbl/roadgraph.bin (RGR1, tools/bake_road_graph.py; Overture
Maps / OpenStreetMap, ODbL), so the output is an ODbL derivative and lives under
data/world/odbl/. FarRoads (scripts/world/far_roads.gd) rasterises it on the GPU
into a road clipmap the terrain shader draws (shaders/include/far_roads.gdshaderinc).

Every drawn edge (no ferries; source tunnel ranges cut out) is split at the
1-degree cell borders and stored per cell in three STREAMS (levels of detail):
  0  every class, the graph's own points (DP 8 toy m): the near clipmap levels
  1  motorway, trunk, primary, secondary, iconic, Douglas-Peucker DP1 toy m
  2  motorway and trunk, DP2 toy m: the far level (to the horizon from cruise)
A stream payload is exactly the bytes of an RGBA8 image W texels wide (one zstd
frame), so the game turns it into a texture without a per-point loop:
  rows [0, R0)        point i at texel i: (x lo, x hi, y lo, y hi), x and y the
                      u16 fraction (65535 = 1 degree) of lon / lat from the
                      cell's south-west corner
  rows [R0, R0 + R1)  the attribute byte of point i at byte i (4 per texel):
                      the class (RoadGraph.MOTORWAY..LINK) of the segment from
                      point i to point i + 1, 255 = none (a line ends at i)
  R0 = ceil(n / W), R1 = ceil(n / (4 W)). Within a cell the lines are sorted
  from the least to the most important class (painter's order).

File: header 32 bytes "FRD1", u32 version (1), u32 cells, u32 streams (2),
u32 index offset, u32 W, 8 bytes zero; index: per cell (sorted by key =
(floor(lat) + 90) * 360 + floor(lon) + 180) u32 key then per stream u32
offset, u32 compressed bytes, u32 points (40 bytes); then the payloads.

Usage: python tools/bake_far_roads.py [--in PATH] [--out PATH]
"""
import argparse
import math
import os
import struct
import sys
import time

import numpy as np
import shapely
import zstandard

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rgr1_read  # noqa: E402

R = 6371000.0 / 5.0          # Geo.R (toy m)
K = R * math.pi / 180.0      # toy m per degree of latitude
W = 1024                     # payload image width (texels)
F_FERRY = 32
ST_TUNNEL = 2
DP1 = 60.0
STREAMS = 2
# painter's order: least important first (feeder, link, secondary, primary, iconic, trunk, motorway)
RANK = {5: 0, 6: 1, 3: 2, 2: 3, 4: 4, 1: 5, 0: 6}
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def log(*a):
    print(*a, flush=True)


def cut_tunnels(ll, structs):
    """Pieces of the (lat, lon) polyline outside every tunnel range (toy m from a)."""
    tun = sorted((s0, s1) for kind, _lv, s0, s1 in structs if kind == ST_TUNNEL and s1 > s0)
    if not tun:
        return [ll]
    lat0 = ll[:, 0].mean()
    xy = np.column_stack([ll[:, 1] * K * math.cos(math.radians(lat0)), ll[:, 0] * K])
    d = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))])
    keep = []
    t = 0.0
    for s0, s1 in tun + [(d[-1] + 1.0, d[-1] + 1.0)]:
        if s0 > t:
            keep.append((t, min(s0, d[-1])))
        t = max(t, s1)
    out = []
    for a, b in keep:
        if b - a < 1.0:
            continue
        inner = (d > a) & (d < b)
        pa = np.array([np.interp(a, d, ll[:, 0]), np.interp(a, d, ll[:, 1])])
        pb = np.array([np.interp(b, d, ll[:, 0]), np.interp(b, d, ll[:, 1])])
        out.append(np.vstack([pa[None], ll[inner], pb[None]]))
    return out


def simplify(ll, tol):
    if tol <= 0.0 or len(ll) <= 2:
        return ll
    c = math.cos(math.radians(ll[:, 0].mean()))
    xy = np.column_stack([ll[:, 1] * K * c, ll[:, 0] * K])
    s = np.asarray(shapely.simplify(shapely.linestrings(xy), tol, preserve_topology=False).coords)
    if len(s) < 2:
        return ll[[0, -1]]
    return np.column_stack([s[:, 1] / K, s[:, 0] / (K * c)])


def split_cells(ll):
    """The polyline cut at every 1-degree cell border: [(cell (ilat, ilon), (lat, lon) array)]."""
    out = []
    cur = [ll[0]]
    cell = (math.floor(ll[0][0]), math.floor(ll[0][1]))
    for i in range(1, len(ll)):
        p = ll[i - 1]
        q = ll[i]
        guard = 0
        while True:
            qc = (math.floor(q[0]), math.floor(q[1]))
            guard += 1
            if qc == cell or guard > 64:
                cur.append(q)
                break
            # the first border crossed from p toward q (inside the current cell)
            best_t = 2.0
            nxt = None
            for ax in (0, 1):
                dv = q[ax] - p[ax]
                if dv > 0 and qc[ax] > cell[ax]:
                    t = (cell[ax] + 1 - p[ax]) / dv
                    step = (1 if ax == 0 else 0, 1 if ax == 1 else 0)
                elif dv < 0 and qc[ax] < cell[ax]:
                    t = (cell[ax] - p[ax]) / dv
                    step = (-1 if ax == 0 else 0, -1 if ax == 1 else 0)
                else:
                    continue
                if t < best_t:
                    best_t = t
                    nxt = step
            if nxt is None:
                cur.append(q)
                break
            x = p + (q - p) * min(max(best_t, 0.0), 1.0)
            cur.append(x)
            if len(cur) >= 2:
                out.append((cell, np.array(cur)))
            cell = (cell[0] + nxt[0], cell[1] + nxt[1])
            cur = [x]
            p = x
    if len(cur) >= 2:
        out.append((cell, np.array(cur)))
    return out


def stream_of(cls):
    """The streams an edge class goes into."""
    return [0, 1] if cls in (0, 1) else [0]


def encode(lines, cell):
    """lines: [(cls, (lat, lon) array)] of one cell and stream -> (payload bytes, points)."""
    lines.sort(key=lambda x: RANK.get(x[0], 0))
    la0, lo0 = cell
    n = sum(len(l) for _c, l in lines)
    pts = np.zeros((n, 2), np.uint16)
    att = np.full(n, 255, np.uint8)
    i = 0
    for c, l in lines:
        m = len(l)
        fx = np.clip(np.round((l[:, 1] - lo0) * 65535.0), 0, 65535)
        fy = np.clip(np.round((l[:, 0] - la0) * 65535.0), 0, 65535)
        pts[i:i + m, 0] = fx
        pts[i:i + m, 1] = fy
        att[i:i + m - 1] = c
        i += m
    r0 = (n + W - 1) // W
    r1 = (n + 4 * W - 1) // (4 * W)
    img = np.zeros((r0 + r1) * W * 4, np.uint8)
    img[:n * 4] = pts.view(np.uint8).reshape(-1)  # little endian u16: lo, hi
    img[r0 * W * 4:r0 * W * 4 + n] = att
    return img.tobytes(), n


def bake(src, dst):
    t0 = time.time()
    edges = rgr1_read.read(src)
    log("read %d edges in %.1f s" % (len(edges), time.time() - t0))
    cells = {}  # (ilat, ilon) -> [list per stream of (cls, ll)]
    for k, e in enumerate(edges):
        if e["flags"] & F_FERRY:
            continue
        cls = e["cls"]
        for piece in cut_tunnels(e["ll"], e["structs"]):
            for s in stream_of(cls):
                pl = simplify(piece, (0.0, DP1)[s])
                for cell, part in split_cells(pl):
                    cells.setdefault(cell, [[] for _ in range(STREAMS)])[s].append((cls, part))
        if k % 100000 == 0:
            log("  %d / %d edges, %d cells, %.0f s" % (k, len(edges), len(cells), time.time() - t0))
    zc = zstandard.ZstdCompressor(level=19, write_checksum=False, write_content_size=True)
    keys = sorted(((c[0] + 90) * 360 + (c[1] + 180) % 360, c) for c in cells)
    head = 32
    index_n = len(keys) * (4 + 12 * STREAMS)
    body = bytearray()
    index = bytearray()
    npts = [0] * STREAMS
    raw = [0] * STREAMS
    cmax = [0] * STREAMS
    for key, cell in keys:
        index += struct.pack("<I", key)
        for s in range(STREAMS):
            lines = cells[cell][s]
            if not lines:
                index += struct.pack("<III", 0, 0, 0)
                continue
            pay, n = encode(lines, cell)
            z = zc.compress(pay)
            index += struct.pack("<III", head + index_n + len(body), len(z), n)
            body += z
            npts[s] += n
            raw[s] += len(pay)
            cmax[s] = max(cmax[s], n)
    hdr = b"FRD1" + struct.pack("<IIIII", 1, len(keys), STREAMS, head, W) + bytes(8)
    data = hdr + index + body
    tmp = dst + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, dst)
    log("wrote %s: %d cells, %.1f MB (raw %s MB), points per stream %s, max per cell %s, %.0f s" % (
        dst, len(keys), len(data) / 1e6, [round(r / 1e6, 1) for r in raw], npts, cmax, time.time() - t0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", default=os.path.join(ROOT, "data/world/odbl/roadgraph.bin"))
    ap.add_argument("--out", dest="dst", default=os.path.join(ROOT, "data/world/odbl/farroads.bin"))
    a = ap.parse_args()
    bake(a.src, a.dst)


if __name__ == "__main__":
    main()
