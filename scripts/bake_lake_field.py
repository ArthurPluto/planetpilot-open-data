"""Bake the lake field: where the ground must meet a lake, for the terrain
height function (scripts/world/lake_field.gd, Heightfield.land_h, and its GPU
twin shaders/include/terrain_fn.gdshaderinc).

Input: data/world/lakes.bin (tools/bake_rivers.py), i.e. the very triangles the
lake surfaces are drawn with, so the carved shore follows the drawn water.

The world is cut into 0.5 x 0.5 degree tiles; every tile a lake reaches
(within RANGE toy metres) gets N x N samples (N = 129, edge samples shared
with the neighbours, sample (i, j) at lon0 + i / 256, lat0 + j / 256 degrees,
j = 0 on the south edge, about 87 toy m apart at 1/5 scale), stored as an image 2N wide:
  left half, per sample:
    R    signed distance to the nearest lake shore, linear:
         sd = (R / 255 * 2 - 1) * RANGE toy m (+ land, - water), clamped
    G, B u16 LE: 1 + index (in lakes.bin) of that lake, 0 = no lake within RANGE
  right half, the same sample's second nearest lake (another one):
    R    its distance, R / 255 * RANGE toy m;  G, B its 1 + index, 0 = none
  A = 255
(the second lake lets the ground between two lakes of different levels ease
from one to the other instead of stepping where they are equally far).
The water level of each lake is chosen at run time from the ground itself
(LakeField.level), so it is not stored here.

Shore fit: the runtime reads the field bilinearly, so its zero line runs
straight across a sample cell while the drawn outline curves inside it; where
the zero line lands outside the drawn water the ground dips under the lake
level on dry land (a pit strip along the shore). The samples next to a shore
are therefore least-squares fitted (fit_shore) so that the bilinear field
matches the true signed distance at points on the drawn outline (0) and
FIT_H either side of it, with a pull back to the plain distances (the
samples on a tile edge, shared with the neighbour, stay plain). At 1/5 scale
(87 toy m cells) this keeps the shore as flush as the plain 43 m cells did.

Output: data/world/lake_field.bin (little endian)
  "LKF2", u32 tile count, u32 N, f32 RANGE,
  tiles x 12 B: u16 tx, u16 ty (tx = (lon0 + 180) * 2, ty = (lat0 + 90) * 2),
                u32 offset, u32 size   (sorted by ty * 1024 + tx)
  then the tiles, each a lossless WebP (RGBA, 2N x N, row 0 = south edge).

Run: python tools/bake_lake_field.py     (numpy, scipy, Pillow with WebP)
     python tools/bake_lake_field.py --lakes 10,12   (only the tiles those lakes reach, and the file's tiles
     within a degree of them, re-baked into the existing file: tools/patch_lake_osm.py)
"""
import math
import os
import struct
import sys
import time
import io

import numpy as np
from PIL import Image
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "data", "world", "lake_field.bin")
from world_scale import R_TOY  # noqa: E402 (Geo.R)
DEG_M = R_TOY * math.pi / 180.0
TILE_DEG = 0.5
N = 129
STEP = TILE_DEG / (N - 1)  # 1/256 degree
RANGE = 300.0  # toy metres (LakeField.RANGE)
DENSE = 2.5  # outline densified to this spacing (toy m) for the distance query
FIT_STEP = 20.0  # shore fit: points along the outline this far apart (toy m)
FIT_H = 30.0  # ... and this far either side of it, on the water and the land side
FIT_W_OFF = 0.5  # weight of the side points (the outline points weigh 1)
FIT_REG = 0.05  # pull of every fitted sample toward its plain distance
FINE = 4  # the side points' land / water test: labels on a grid FINE x denser


def load_lakes():
    b = open(os.path.join(ROOT, "data", "world", "lakes.bin"), "rb").read()
    assert b[:4] == b"LAK1"
    nl, nv, ni = struct.unpack_from("<III", b, 4)
    heads = [struct.unpack_from("<iiiifIIII", b, 16 + i * 36) for i in range(nl)]
    vo = 16 + nl * 36
    verts = np.frombuffer(b, dtype=[("la", "<u2"), ("lo", "<u2"), ("d", "<u2")], count=nv, offset=vo)
    idx = np.frombuffer(b, dtype="<u2", count=ni, offset=vo + nv * 6)
    lakes = []
    for li, h in enumerate(heads):
        la0, lo0, la1, lo1 = h[0] * 1e-5, h[1] * 1e-5, h[2] * 1e-5, h[3] * 1e-5
        v = verts[h[5]:h[5] + h[6]]
        lat = la0 + v["la"].astype(np.float64) / 65535.0 * (la1 - la0)
        lon = lo0 + v["lo"].astype(np.float64) / 65535.0 * (lo1 - lo0)
        t = idx[h[7]:h[7] + h[8]].astype(np.int64).reshape(-1, 3)
        # outline = edges used by one triangle (opp: that triangle's third
        # vertex, on the water side of the edge)
        e = np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]])
        opp = np.concatenate([t[:, 2], t[:, 0], t[:, 1]])
        k = np.sort(e, axis=1)
        key = k[:, 0] * 70000 + k[:, 1]
        u, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
        one = cnt[inv] == 1
        lakes.append({"box": (la0, lo0, la1, lo1), "lat": lat, "lon": lon, "edges": e[one], "opp": opp[one]})
    return lakes


def tile_of(lat, lon):
    return int(math.floor((lon + 180.0) / TILE_DEG)), int(math.floor((lat + 90.0) / TILE_DEG))


def inside_labels(gx, gy, A, B, SID):
    """Even-odd scanline fill per lake over the grid (columns gx, rows gy):
    1 + lake index where a sample lies inside that lake, else 0 (islands are
    outline loops too)."""
    nx = len(gx)
    ny = len(gy)
    lab = np.zeros((ny, nx), np.int64)
    y0 = A[:, 1]
    y1 = B[:, 1]
    lo_y = np.minimum(y0, y1)
    hi_y = np.maximum(y0, y1)
    dy = gy[1] - gy[0]
    r0 = np.clip(np.ceil((lo_y - gy[0]) / dy).astype(np.int64), 0, ny)
    r1 = np.clip(np.ceil((hi_y - gy[0]) / dy).astype(np.int64), 0, ny)  # rows r0 .. r1 - 1: lo <= y < hi
    n_r = np.maximum(r1 - r0, 0)
    if n_r.sum() > 0:
        eidx = np.repeat(np.arange(len(A)), n_r)
        rows = r0[eidx] + (np.arange(n_r.sum()) - np.repeat(np.cumsum(n_r) - n_r, n_r))
        yr = gy[rows]
        t = (yr - A[eidx, 1]) / (B[eidx, 1] - A[eidx, 1])
        xr = A[eidx, 0] + t * (B[eidx, 0] - A[eidx, 0])
        lid = SID[eidx]
        order = np.lexsort((xr, lid, rows))
        rows, xr, lid = rows[order], xr[order], lid[order]
        # pairs (in, out) within each (row, lake) group
        grp = rows * 100000 + lid
        start = np.r_[True, grp[1:] != grp[:-1]]
        pos = np.arange(len(grp)) - np.maximum.accumulate(np.where(start, np.arange(len(grp)), 0))
        ins = pos % 2 == 0
        ok = ins & (np.r_[grp[1:], -1] == grp)
        i_in = np.nonzero(ok)[0]
        xa = xr[i_in]
        xb = xr[i_in + 1]
        ra = rows[i_in]
        la = lid[i_in]
        dx = gx[1] - gx[0]
        c0 = np.clip(np.ceil((xa - gx[0]) / dx).astype(np.int64), 0, nx)
        c1 = np.clip(np.ceil((xb - gx[0]) / dx).astype(np.int64), 0, nx)
        n_c = np.maximum(c1 - c0, 0)
        if n_c.sum() > 0:
            sidx = np.repeat(np.arange(len(c0)), n_c)
            cols = c0[sidx] + (np.arange(n_c.sum()) - np.repeat(np.cumsum(n_c) - n_c, n_c))
            lab[ra[sidx], cols] = la[sidx]
    return lab


def fit_shore(sd, gx, gy, A, B, C, SID, tree):
    """Least-squares touch-up of the samples around the shores (see the
    module notes): the bilinear field should read the true signed distance
    on the drawn outline (0) and FIT_H either side of it. Returns the new
    sd (N x N); samples no fit point touches keep their plain distance."""
    x0, x1, y0, y1 = gx[0], gx[-1], gy[0], gy[-1]
    m = (np.maximum(A[:, 0], B[:, 0]) >= x0 - FIT_H) & (np.minimum(A[:, 0], B[:, 0]) <= x1 + FIT_H) & \
        (np.maximum(A[:, 1], B[:, 1]) >= y0 - FIT_H) & (np.minimum(A[:, 1], B[:, 1]) <= y1 + FIT_H)
    A2, B2, C2 = A[m], B[m], C[m]
    d = B2 - A2
    L = np.hypot(d[:, 0], d[:, 1])
    ok = L > 1e-6
    A2, B2, C2, d, L = A2[ok], B2[ok], C2[ok], d[ok], L[ok]
    if len(A2) == 0:
        return sd
    nrm = np.stack([-d[:, 1], d[:, 0]], 1) / L[:, None]
    side = np.sign(((C2 - A2) * nrm).sum(1))
    nrm *= np.where(side == 0, 1.0, side)[:, None]  # toward the water
    k = np.maximum(1, np.ceil(L / FIT_STEP).astype(np.int64))
    si = np.repeat(np.arange(len(A2)), k)
    tt = (np.arange(k.sum()) - np.repeat(np.cumsum(k) - k, k) + 0.5) / np.repeat(k, k)
    P = A2[si] + d[si] * tt[:, None]
    n = nrm[si]
    # land / water of the side points: labels on a finer grid; a side point
    # closer to some shore than a fine cell's half diagonal is left out
    gxf = np.linspace(x0, x1, (N - 1) * FINE + 1)
    gyf = np.linspace(y0, y1, (N - 1) * FINE + 1)
    lab_f = inside_labels(gxf, gyf, A, B, SID)
    half = 0.5 * math.hypot(gxf[1] - gxf[0], gyf[1] - gyf[0])
    X = [P]
    T = [np.zeros(len(P))]
    Wt = [np.ones(len(P))]
    for sgn in (1.0, -1.0):
        O = P + sgn * FIT_H * n
        dist, _ = tree.query(O, distance_upper_bound=RANGE)
        ci = np.clip(np.round((O[:, 0] - x0) / (gxf[1] - gxf[0])).astype(np.int64), 0, len(gxf) - 1)
        ri = np.clip(np.round((O[:, 1] - y0) / (gyf[1] - gyf[0])).astype(np.int64), 0, len(gyf) - 1)
        wet = lab_f[ri, ci] > 0
        good = np.isfinite(dist) & (dist > half)
        X.append(O[good])
        T.append(np.where(wet[good], -dist[good], dist[good]))
        Wt.append(np.full(int(good.sum()), FIT_W_OFF))
    X = np.concatenate(X)
    T = np.concatenate(T)
    Wt = np.concatenate(Wt)
    u = (X[:, 0] - x0) / (gx[1] - gx[0])
    v = (X[:, 1] - y0) / (gy[1] - gy[0])
    inb = (u >= 0) & (u <= N - 1) & (v >= 0) & (v <= N - 1)
    if not inb.any():
        return sd
    u, v, T, Wt = u[inb], v[inb], T[inb], Wt[inb]
    i = np.minimum(np.floor(u).astype(np.int64), N - 2)
    j = np.minimum(np.floor(v).astype(np.int64), N - 2)
    fx = u - i
    fy = v - j
    cols = np.stack([j * N + i, j * N + i + 1, (j + 1) * N + i, (j + 1) * N + i + 1], 1)
    w4 = np.stack([(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy], 1) * Wt[:, None]
    used, inv = np.unique(cols, return_inverse=True)
    M = sparse.csr_matrix((w4.ravel(), (np.repeat(np.arange(len(u)), 4), inv.ravel())), shape=(len(u), len(used)))
    flat = sd.ravel().copy()
    s0 = flat[used]
    # the edge samples are shared with the neighbour tiles: they keep their
    # plain distance, so both tiles hold the same bytes there
    ui = used % N
    uj = used // N
    reg = np.where((ui == 0) | (ui == N - 1) | (uj == 0) | (uj == N - 1), 1e6, FIT_REG)
    lhs = (M.T @ M + sparse.diags(reg)).tocsc()
    rhs = M.T @ (T * Wt) + reg * s0
    s = spsolve(lhs, rhs)
    lim = 0.5 * (gx[1] - gx[0])
    flat[used] = np.clip(s, s0 - lim, s0 + lim)
    return flat.reshape(N, N)


def bake_tile(tx, ty, ids, lakes):
    lon0 = tx * TILE_DEG - 180.0
    lat0 = ty * TILE_DEG - 90.0
    latc = lat0 + TILE_DEG * 0.5
    lonc = lon0 + TILE_DEG * 0.5
    kx = math.cos(math.radians(latc))
    gx = (lon0 + np.arange(N) * STEP - lonc) * DEG_M * kx  # sample columns (toy m)
    gy = (lat0 + np.arange(N) * STEP - latc) * DEG_M  # sample rows
    ex0, ex1 = gx[0] - RANGE - 50, gx[-1] + RANGE + 50
    ey0, ey1 = gy[0] - RANGE - 50, gy[-1] + RANGE + 50
    seg_a = []
    seg_b = []
    seg_c = []
    seg_id = []
    for li in ids:
        L = lakes[li]
        dlon = L["lon"] - lonc
        dlon = np.where(dlon > 180, dlon - 360, np.where(dlon < -180, dlon + 360, dlon))
        x = dlon * DEG_M * kx
        y = (L["lat"] - latc) * DEG_M
        e = L["edges"]
        ax, ay, bx, by = x[e[:, 0]], y[e[:, 0]], x[e[:, 1]], y[e[:, 1]]
        seg_a.append(np.stack([ax, ay], 1))
        seg_b.append(np.stack([bx, by], 1))
        seg_c.append(np.stack([x[L["opp"]], y[L["opp"]]], 1))
        seg_id.append(np.full(len(e), li + 1, np.int64))
    A = np.concatenate(seg_a)
    B = np.concatenate(seg_b)
    C = np.concatenate(seg_c)
    SID = np.concatenate(seg_id)
    # ---- inside: even-odd scanline fill per lake (islands are outline loops too)
    lab = inside_labels(gx, gy, A, B, SID)
    # ---- distance to each lake's outline (densified segments, one KD tree
    # per lake): the nearest lake and the second nearest one
    gxx, gyy = np.meshgrid(gx, gy)
    d1 = np.full((N, N), np.inf)
    i1 = np.zeros((N, N), np.int64)
    d2 = np.full((N, N), np.inf)
    i2 = np.zeros((N, N), np.int64)
    own = {}
    allq = []
    reach = RANGE * 1.05
    for li in ids:
        m = (SID == li + 1) & (np.maximum(A[:, 0], B[:, 0]) > ex0) & (np.minimum(A[:, 0], B[:, 0]) < ex1) &             (np.maximum(A[:, 1], B[:, 1]) > ey0) & (np.minimum(A[:, 1], B[:, 1]) < ey1)
        if not m.any():
            continue
        A2, B2 = A[m], B[m]
        c0 = max(0, int(np.searchsorted(gx, min(A2[:, 0].min(), B2[:, 0].min()) - reach)))
        c1 = min(N, int(np.searchsorted(gx, max(A2[:, 0].max(), B2[:, 0].max()) + reach)) + 1)
        r0 = max(0, int(np.searchsorted(gy, min(A2[:, 1].min(), B2[:, 1].min()) - reach)))
        r1 = min(N, int(np.searchsorted(gy, max(A2[:, 1].max(), B2[:, 1].max()) + reach)) + 1)
        if c1 <= c0 or r1 <= r0:
            continue
        L = np.hypot(B2[:, 0] - A2[:, 0], B2[:, 1] - A2[:, 1])
        k = np.maximum(1, np.ceil(L / DENSE).astype(np.int64))
        si = np.repeat(np.arange(len(A2)), k + 1)
        tt = (np.arange((k + 1).sum()) - np.repeat(np.cumsum(k + 1) - (k + 1), k + 1)) / np.repeat(k, k + 1)
        Q = A2[si] + (B2[si] - A2[si]) * tt[:, None]
        allq.append(Q)
        P = np.stack([gxx[r0:r1, c0:c1].ravel(), gyy[r0:r1, c0:c1].ravel()], 1)
        dist, _ = cKDTree(Q).query(P, distance_upper_bound=reach)
        dL = np.full((N, N), np.inf)
        dL[r0:r1, c0:c1] = dist.reshape(r1 - r0, c1 - c0)
        own[li + 1] = dL
        closer = dL < d1
        second = ~closer & (dL < d2)
        d2 = np.where(closer, d1, np.where(second, dL, d2))
        i2 = np.where(closer, i1, np.where(second, li + 1, i2))
        d1 = np.where(closer, dL, d1)
        i1 = np.where(closer, li + 1, i1)
    inside = lab > 0
    # inside a lake that lake is the nearest one (overlapping outlines of the
    # three Natural Earth files: the other one becomes the second)
    fix = inside & (i1 != lab)
    if fix.any():
        for li1 in np.unique(lab[fix]):
            f = fix & (lab == li1)
            dl = own.get(int(li1))
            if dl is None:
                continue
            d2 = np.where(f, d1, d2)
            i2 = np.where(f, i1, i2)
            d1 = np.where(f, dl, d1)
            i1 = np.where(f, li1, i1)
    far1 = ~np.isfinite(d1)
    i1 = np.where(far1 & ~inside, 0, i1)
    d1 = np.where(far1, RANGE, d1)
    far2 = ~np.isfinite(d2)
    i2 = np.where(far2, 0, i2)
    d2 = np.where(far2, RANGE, d2)
    if not i1.any():
        return None
    sd = np.where(inside, -d1, d1)
    if allq:
        sd = fit_shore(sd, gx, gy, A, B, C, SID, cKDTree(np.concatenate(allq)))
    out = np.zeros((N, 2 * N, 4), np.uint8)
    out[:, :N, 0] = np.round((np.clip(sd, -RANGE, RANGE) / RANGE * 0.5 + 0.5) * 255.0).astype(np.uint8)
    out[:, :N, 1] = (i1 & 255).astype(np.uint8)
    out[:, :N, 2] = (i1 >> 8).astype(np.uint8)
    out[:, N:, 0] = np.round(np.clip(d2, 0.0, RANGE) / RANGE * 255.0).astype(np.uint8)
    out[:, N:, 1] = (i2 & 255).astype(np.uint8)
    out[:, N:, 2] = (i2 >> 8).astype(np.uint8)
    out[..., 3] = 255
    buf = io.BytesIO()
    Image.fromarray(out, "RGBA").save(buf, "WEBP", lossless=True, quality=100, method=4, exact=True)
    return buf.getvalue()


def read_field(path=OUT):
    """{(tx, ty): webp bytes} of an existing lake_field.bin."""
    b = open(path, "rb").read()
    assert b[:4] == b"LKF2"
    n = struct.unpack_from("<I", b, 4)[0]
    out = {}
    for k in range(n):
        tx, ty, off, size = struct.unpack_from("<HHII", b, 16 + 12 * k)
        out[(tx, ty)] = b[off:off + size]
    return out


def main():
    t0 = time.time()
    only = None
    if "--lakes" in sys.argv:
        only = {int(x) for x in sys.argv[sys.argv.index("--lakes") + 1].split(",") if x}
    lakes = load_lakes()
    tiles = {}
    m = (RANGE + 60.0) / DEG_M
    for li, L in enumerate(lakes):
        la0, lo0, la1, lo1 = L["box"]
        ml = m / max(0.05, math.cos(math.radians(max(abs(la0), abs(la1)))))
        tx0, ty0 = tile_of(la0 - m, lo0 - ml)
        tx1, ty1 = tile_of(la1 + m, lo1 + ml)
        for ty in range(ty0, ty1 + 1):
            for tx in range(tx0, tx1 + 1):
                tiles.setdefault((tx % 720, ty), []).append(li)
    print("lakes %d, candidate tiles %d (%.0f s)" % (len(lakes), len(tiles), time.time() - t0), flush=True)
    keep = {}
    if only is not None:
        # patch mode: the other tiles stay byte for byte
        keep = read_field()
        redo = {k for k, ids in tiles.items() if only & set(ids)}
        for li in only:
            la0, lo0, la1, lo1 = lakes[li]["box"]
            tx0, ty0 = tile_of(la0 - 1.0, lo0 - 1.0)
            tx1, ty1 = tile_of(la1 + 1.0, lo1 + 1.0)
            redo |= {k for k in keep if tx0 <= k[0] <= tx1 and ty0 <= k[1] <= ty1}
        for k in redo:
            keep.pop(k, None)
        tiles = {k: tiles.get(k, []) for k in redo}
        print("patch: %d tiles re-baked, %d kept" % (len(tiles), len(keep)), flush=True)
    blobs = list(keep.items())
    for n, (key, ids) in enumerate(sorted(tiles.items(), key=lambda kv: kv[0][1] * 1024 + kv[0][0])):
        raw = bake_tile(key[0], key[1], ids, lakes) if ids else None
        if raw is not None:
            blobs.append((key, raw))
        if n % 500 == 0:
            print("  %d / %d tiles, %d kept (%.0f s)" % (n, len(tiles), len(blobs), time.time() - t0), flush=True)
    blobs.sort(key=lambda kv: kv[0][1] * 1024 + kv[0][0])
    with open(OUT, "wb") as f:
        f.write(b"LKF2")
        f.write(struct.pack("<IIf", len(blobs), N, RANGE))
        off = 16 + 12 * len(blobs)
        for (tx, ty), z in blobs:
            f.write(struct.pack("<HHII", tx, ty, off, len(z)))
            off += len(z)
        for _, z in blobs:
            f.write(z)
    print("lake field: %d tiles, %.2f MB (%.0f s)" % (len(blobs), os.path.getsize(OUT) / 1e6, time.time() - t0))


if __name__ == "__main__":
    main()
