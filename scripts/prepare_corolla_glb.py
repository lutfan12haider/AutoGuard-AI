"""
prepare_corolla_glb.py
----------------------
Converts the real CC BY 4.0 "2023 Toyota Corolla Hybrid" GLB
(corolla_uncompressed.glb) into an uncompressed standard IEEE float32 glTF 2.0
GLB (`frontend/public/models/autoguard_corolla.glb`) tailored for AutoGuard AI's
3D diagnostic animation sequence:

1. Evaluates all node transforms into metric world space (Y=0 tires, +Z front,
   ~4.65m length x 2.09m width x 1.46m height).
2. Dequantizes all vertex attributes (removes KHR_mesh_quantization &
   EXT_meshopt_compression) to standard float32 + uint32 indices.
3. Surgically slices the real sculpted Corolla hood panel using exact
   multi-plane triangle clipping along the natural windshield cowl and
   headlight-swept fender shut lines, placing it (plus an opaque black matte
   inner hood frame Liner) under `Bonnet_Hood_Pivot` pivoted at [0, 0.962, 1.34].
4. Builds a dark matte Corolla engine compartment tub (firewall, radiator core
   support & cooling fan, left/right strut towers, battery/airbox trim) under
   the hood opening so the cutaway bay looks realistic when opened.
5. Repaints exterior bodywork (`Carro_Pintura`) in metallic Toyota Classic Silver.
6. Preserves CC BY 4.0 attribution metadata from original author tonielpro520.
"""

from __future__ import annotations

import json
import os
import struct
import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_GLB = os.path.join(ROOT, "frontend", "public", "models", "corolla_uncompressed.glb")
OUT_GLB = os.path.join(ROOT, "frontend", "public", "models", "autoguard_corolla.glb")

HINGE_PIVOT = np.array([0.0, 0.962, 1.340], dtype=np.float64)


# ── Convex Hood Boundary Half-Spaces (n . p + d >= 0 means INSIDE hood) ──────
def make_hood_planes() -> list[tuple[np.ndarray, float]]:
    planes = [
        # 1. Rear windshield cowl hinge edge: Z >= 1.340
        (np.array([0.0, 0.0, 1.0]), -1.340),
        # 2. Right fender seam: X <= 0.585 - 0.18*(Z - 1.340) => -X - 0.18*Z + 0.8262 >= 0
        (np.array([-1.0, 0.0, -0.18]), 0.585 + 0.18 * 1.340),
        # 3. Left fender seam: -X <= 0.585 - 0.18*(Z - 1.340) => +X - 0.18*Z + 0.8262 >= 0
        (np.array([1.0, 0.0, -0.18]), 0.585 + 0.18 * 1.340),
        # 4. Right swept front nose seam above headlight: 2.135 - Z - 0.36*X >= 0
        (np.array([-0.36, 0.0, -1.0]), 2.135),
        # 5. Left swept front nose seam above headlight: 2.135 - Z + 0.36*X >= 0
        (np.array([0.36, 0.0, -1.0]), 2.135),
    ]
    normed = []
    for n, d in planes:
        L = float(np.linalg.norm(n))
        normed.append((n / L, d / L))
    return normed


HOOD_PLANES = make_hood_planes()


def hood_y_at(x: float, z: float) -> float:
    """Smooth approximate Y height of the Corolla hood contour along (X, Z)."""
    t = np.clip((z - 1.34) / (2.13 - 1.34), 0.0, 1.0)
    center_y = 0.962 - 0.19 * (t ** 1.15)
    return float(center_y - 0.06 * (x * x))


# ── GLB Reader Helpers ────────────────────────────────────────────────────────
def read_glb(path: str):
    data = open(path, "rb").read()
    pos = 12
    json_chunk = bin_chunk = b""
    while pos < len(data):
        clen, ctype = struct.unpack("<II", data[pos:pos + 8])
        chunk = data[pos + 8:pos + 8 + clen]
        if ctype == 0x4E4F534A:
            json_chunk = chunk
        elif ctype == 0x004E4942:
            bin_chunk = chunk
        pos += 8 + clen
    return json.loads(json_chunk.decode("utf-8")), bin_chunk


def read_accessor(g, bin_chunk: bytes, acc_idx: int) -> np.ndarray:
    acc = g["accessors"][acc_idx]
    bv = g["bufferViews"][acc["bufferView"]]
    offset = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
    count = acc["count"]
    ctype = acc["componentType"]
    dtype = {
        5126: np.float32,
        5123: np.uint16,
        5125: np.uint32,
        5122: np.int16,
        5120: np.int8,
        5121: np.uint8,
    }[ctype]
    ncomp = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[acc["type"]]
    item_sz = np.dtype(dtype).itemsize * ncomp
    stride = bv.get("byteStride", item_sz)
    if stride == item_sz:
        arr = np.frombuffer(bin_chunk, dtype=dtype, count=count * ncomp, offset=offset).reshape((count, ncomp))
    else:
        arr = np.zeros((count, ncomp), dtype=dtype)
        for i in range(count):
            arr[i] = np.frombuffer(bin_chunk, dtype=dtype, count=ncomp, offset=offset + i * stride)
    arr = arr.astype(np.float64)
    if acc.get("normalized", False):
        if ctype == 5122:
            arr = np.maximum(arr / 32767.0, -1.0)
        elif ctype == 5120:
            arr = np.maximum(arr / 127.0, -1.0)
        elif ctype == 5123:
            arr = arr / 65535.0
        elif ctype == 5121:
            arr = arr / 255.0
    return arr


def quat_to_mat(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), 0.0],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), 0.0],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float64)


def node_local_mat(n):
    if "matrix" in n:
        return np.array(n["matrix"], dtype=np.float64).reshape((4, 4)).T
    S = np.diag([*n["scale"], 1.0]) if "scale" in n else np.eye(4)
    R = quat_to_mat(n["rotation"]) if "rotation" in n else np.eye(4)
    T = np.eye(4)
    if "translation" in n:
        T[:3, 3] = n["translation"]
    return T @ R @ S


# ── Exact Triangle Split Against Convex Hood Polygon ──────────────────────────
def clip_poly_plane(poly: list[tuple[np.ndarray, np.ndarray]], n: np.ndarray, d: float, keep_inside: bool):
    """Clip polygon (list of (pos3, nrm3)) against half-space n.p + d >= 0 (or <= 0)."""
    if not poly:
        return []
    sign = 1.0 if keep_inside else -1.0
    out = []
    m = len(poly)
    for i in range(m):
        pA, nA = poly[i]
        pB, nB = poly[(i + 1) % m]
        dA = sign * (float(np.dot(n, pA)) + d)
        dB = sign * (float(np.dot(n, pB)) + d)
        inA = dA >= -1e-6
        inB = dB >= -1e-6
        if inA and inB:
            out.append((pB, nB))
        elif inA and not inB:
            t = dA / (dA - dB + 1e-12)
            pI = pA + t * (pB - pA)
            nI = nA + t * (nB - nA)
            nl = np.linalg.norm(nI)
            if nl > 1e-8:
                nI = nI / nl
            out.append((pI, nI))
        elif not inA and inB:
            t = dA / (dA - dB + 1e-12)
            pI = pA + t * (pB - pA)
            nI = nA + t * (nB - nA)
            nl = np.linalg.norm(nI)
            if nl > 1e-8:
                nI = nI / nl
            out.append((pI, nI))
            out.append((pB, nB))
    return out


def fan_triangulate(poly, out_pos: list, out_nrm: list):
    if len(poly) < 3:
        return
    p0, n0 = poly[0]
    for i in range(1, len(poly) - 1):
        p1, n1 = poly[i]
        p2, n2 = poly[i + 1]
        if np.linalg.norm(np.cross(p1 - p0, p2 - p0)) < 1e-10:
            continue
        out_pos.extend([p0, p1, p2])
        out_nrm.extend([n0, n1, n2])


def split_hood_and_body(pos: np.ndarray, nrm: np.ndarray, tris: np.ndarray):
    """
    Separates upper-surface Corolla hood triangles into `hood` and leaves the
    complement seamlessly in `body`.
    """
    hood_p, hood_n = [], []
    body_p, body_n = [], []

    for tri in tris:
        i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
        p0, p1, p2 = pos[i0], pos[i1], pos[i2]
        n0, n1, n2 = nrm[i0], nrm[i1], nrm[i2]

        c = (p0 + p1 + p2) / 3.0
        fn = np.cross(p1 - p0, p2 - p0)
        fl = np.linalg.norm(fn)
        if fl < 1e-10:
            continue
        fn = fn / fl

        # Candidate filter: only upper front exterior surface above headlights/bumper
        min_z = min(p0[2], p1[2], p2[2])
        max_z = max(p0[2], p1[2], p2[2])
        min_x = min(p0[0], p1[0], p2[0])
        max_x = max(p0[0], p1[0], p2[0])
        max_y = max(p0[1], p1[1], p2[1])

        in_band = (
            max_z > 1.33
            and min_z < 2.15
            and max_x > -0.62
            and min_x < 0.62
            and max_y > 0.725
            and c[1] > 0.715
            and abs(fn[1]) > 0.25
        )
        if not in_band:
            body_p.extend([p0, p1, p2])
            body_n.extend([n0, n1, n2])
            continue

        # Check all 3 vertices against all 5 planes quickly
        pts = [p0, p1, p2]
        dists = [[float(np.dot(pn, pt)) + pd for pt in pts] for pn, pd in HOOD_PLANES]
        if all(min(row) >= -1e-5 for row in dists):
            hood_p.extend([p0, p1, p2])
            hood_n.extend([n0, n1, n2])
            continue
        if any(max(row) <= 1e-5 for row in dists):
            body_p.extend([p0, p1, p2])
            body_n.extend([n0, n1, n2])
            continue

        # Split exact polygon: remainder in body + inner piece in hood
        poly = [(p0, n0), (p1, n1), (p2, n2)]
        inner = list(poly)
        rem = list(poly)
        for pn, pd in HOOD_PLANES:
            outside_piece = clip_poly_plane(rem, pn, pd, keep_inside=False)
            fan_triangulate(outside_piece, body_p, body_n)
            rem = clip_poly_plane(rem, pn, pd, keep_inside=True)
            inner = clip_poly_plane(inner, pn, pd, keep_inside=True)
        fan_triangulate(inner, hood_p, hood_n)

    def to_mesh(plist, nlist):
        P = np.asarray(plist, dtype=np.float32)
        N = np.asarray(nlist, dtype=np.float32)
        I = np.arange(len(P), dtype=np.uint32)
        return P, N, I

    return to_mesh(hood_p, hood_n), to_mesh(body_p, body_n)


# ── Procedural Engine Bay Structural Geometry Builders ────────────────────────
class TriBuilder:
    def __init__(self):
        self.v = []
        self.n = []
        self.i = []

    def add_box(self, cx, cy, cz, sx, sy, sz):
        hx, hy, hz = sx * 0.5, sy * 0.5, sz * 0.5
        faces = [
            ((0, 0, 1), [(-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz)]),
            ((0, 0, -1), [(hx, -hy, -hz), (-hx, -hy, -hz), (-hx, hy, -hz), (hx, hy, -hz)]),
            ((1, 0, 0), [(hx, -hy, hz), (hx, -hy, -hz), (hx, hy, -hz), (hx, hy, hz)]),
            ((-1, 0, 0), [(-hx, -hy, -hz), (-hx, -hy, hz), (-hx, hy, hz), (-hx, hy, -hz)]),
            ((0, 1, 0), [(-hx, hy, hz), (hx, hy, hz), (hx, hy, -hz), (-hx, hy, -hz)]),
            ((0, -1, 0), [(-hx, -hy, -hz), (hx, -hy, -hz), (hx, -hy, hz), (-hx, -hy, hz)]),
        ]
        for norm, quad in faces:
            base = len(self.v)
            for qx, qy, qz in quad:
                self.v.append([cx + qx, cy + qy, cz + qz])
                self.n.append(list(norm))
            self.i.extend([base, base + 1, base + 2, base, base + 2, base + 3])

    def add_cyl_y(self, cx, y0, y1, cz, rad, segs=20):
        thetas = np.linspace(0, 2 * np.pi, segs, endpoint=False)
        base = len(self.v)
        for th in thetas:
            nx, nz = np.cos(th), np.sin(th)
            self.v.append([cx + rad * nx, y0, cz + rad * nz])
            self.n.append([nx, 0.0, nz])
            self.v.append([cx + rad * nx, y1, cz + rad * nz])
            self.n.append([nx, 0.0, nz])
        for k in range(segs):
            k2 = (k + 1) % segs
            b0, t0 = base + 2 * k, base + 2 * k + 1
            b1, t1 = base + 2 * k2, base + 2 * k2 + 1
            self.i.extend([b0, b1, t1, b0, t1, t0])
        # Top cap
        tc = len(self.v)
        self.v.append([cx, y1, cz])
        self.n.append([0.0, 1.0, 0.0])
        for th in thetas:
            self.v.append([cx + rad * np.cos(th), y1, cz + rad * np.sin(th)])
            self.n.append([0.0, 1.0, 0.0])
        for k in range(segs):
            k2 = (k + 1) % segs
            self.i.extend([tc, tc + 1 + k2, tc + 1 + k])

    def add_tapered_prism(self, x0_f, x1_f, x0_r, x1_r, y_bot, y_top_f, y_top_r, z_r, z_f):
        """Adds an axis-tapered 6-face closed prism from rear z_r to front z_f."""
        v_r = [(x0_r, y_bot, z_r), (x1_r, y_bot, z_r), (x1_r, y_top_r, z_r), (x0_r, y_top_r, z_r)]
        v_f = [(x0_f, y_bot, z_f), (x1_f, y_bot, z_f), (x1_f, y_top_f, z_f), (x0_f, y_top_f, z_f)]
        quads = [
            (v_f[0], v_f[1], v_f[2], v_f[3]),  # +Z front
            (v_r[1], v_r[0], v_r[3], v_r[2]),  # -Z rear
            (v_f[1], v_r[1], v_r[2], v_f[2]),  # +X right
            (v_r[0], v_f[0], v_f[3], v_r[3]),  # -X left
            (v_r[3], v_r[2], v_f[2], v_f[3]),  # +Y top
            (v_f[0], v_f[1], v_r[1], v_r[0]),  # -Y bottom
        ]
        for a, b, c, d in quads:
            pa, pb, pc, pd = [np.asarray(p, dtype=np.float64) for p in (a, b, c, d)]
            fn = np.cross(pb - pa, pc - pa)
            fl = float(np.linalg.norm(fn))
            norm = (fn / fl).tolist() if fl > 1e-8 else [0.0, 1.0, 0.0]
            base = len(self.v)
            for p in (a, b, c, d):
                self.v.append(list(p))
                self.n.append(norm)
            self.i.extend([base, base + 1, base + 2, base, base + 2, base + 3])

    def finish(self):
        return (
            np.asarray(self.v, dtype=np.float32),
            np.asarray(self.n, dtype=np.float32),
            np.asarray(self.i, dtype=np.uint32),
        )


def build_engine_bay_parts():
    """
    Form-fitted dark satin Corolla engine compartment tub strictly below
    hood_y_at(x, z) - 0.05m so ZERO geometry can ever protrude through the
    sloping front hood or fenders.
    """
    tub = TriBuilder()
    # Floor pan & longitudinal frame rails (Z=1.35..1.94, Y=0.34..0.44)
    tub.add_tapered_prism(-0.36, 0.36, -0.44, 0.44, 0.34, 0.38, 0.38, 1.35, 1.94)
    tub.add_box(-0.27, 0.41, 1.65, 0.06, 0.08, 0.56)
    tub.add_box(0.27, 0.41, 1.65, 0.06, 0.08, 0.56)

    # Rear firewall bulkhead directly beneath cowl (Z=1.34..1.38, top Y=0.84)
    tub.add_tapered_prism(-0.52, 0.52, -0.54, 0.54, 0.36, 0.83, 0.85, 1.335, 1.375)

    # Front radiator core support & electric cooling fan frame (Z=1.90..1.95, Y_top=0.69)
    tub.add_box(0.0, 0.51, 1.925, 0.72, 0.34, 0.045)
    tub.add_box(0.0, 0.695, 1.925, 0.74, 0.03, 0.055)
    tub.add_cyl_y(-0.17, 0.50, 0.67, 1.88, 0.11, segs=18)
    tub.add_cyl_y(0.17, 0.50, 0.67, 1.88, 0.11, segs=18)

    # Tapered left & right fender apron inner walls sloping strictly under hood contour
    # Rear Z=1.36: |X| in [0.44, 0.49], Y_top=0.80 -> Front Z=1.93: |X| in [0.35, 0.40], Y_top=0.68
    tub.add_tapered_prism(0.35, 0.40, 0.44, 0.49, 0.36, 0.68, 0.80, 1.36, 1.93)
    tub.add_tapered_prism(-0.40, -0.35, -0.49, -0.44, 0.36, 0.68, 0.80, 1.36, 1.93)

    # Left & Right MacPherson strut towers (capped neatly at Y=0.71, well under hood Y=0.90)
    tub.add_cyl_y(-0.39, 0.36, 0.71, 1.54, 0.095, segs=20)
    tub.add_cyl_y(0.39, 0.36, 0.71, 1.54, 0.095, segs=20)
    # Compact upper strut top caps & aluminum bay cross-brace bar (Z=1.48, Y=0.735)
    tub.add_box(0.0, 0.735, 1.49, 0.76, 0.02, 0.03)

    # Driver-side cold airbox & passenger-side battery/fuse box (strictly Y_top <= 0.63)
    tub.add_box(-0.31, 0.49, 1.75, 0.15, 0.20, 0.21)
    tub.add_box(0.31, 0.49, 1.74, 0.15, 0.20, 0.22)
    # Coolant overflow tank near rear right tower (Y_top <= 0.66)
    tub.add_cyl_y(0.35, 0.50, 0.65, 1.43, 0.055, segs=16)
    return tub.finish()


# ── Binary GLB Writer ─────────────────────────────────────────────────────────
def main():
    g, bin_chunk = read_glb(SRC_GLB)

    def get_world_mat(target_idx: int) -> np.ndarray:
        def _w(idx, M):
            n = g["nodes"][idx]
            cM = M @ node_local_mat(n)
            if idx == target_idx:
                return cM
            for ch in n.get("children", []):
                r = _w(ch, cM)
                if r is not None:
                    return r
            return None
        for r in g["scenes"][g.get("scene", 0)]["nodes"]:
            res = _w(r, np.eye(4))
            if res is not None:
                return res
        return np.eye(4)

    # Extract all embedded WebP images untouched
    new_bin = bytearray()
    new_bvs = []
    new_imgs = []

    def push_bytes(raw: bytes, target: int | None = None) -> int:
        while len(new_bin) % 4 != 0:
            new_bin.append(0)
        off = len(new_bin)
        new_bin.extend(raw)
        bv = {"buffer": 0, "byteOffset": off, "byteLength": len(raw)}
        if target is not None:
            bv["target"] = target
        idx = len(new_bvs)
        new_bvs.append(bv)
        return idx

    for im in g.get("images", []):
        sbv = g["bufferViews"][im["bufferView"]]
        s0 = sbv.get("byteOffset", 0)
        img_bytes = bin_chunk[s0:s0 + sbv["byteLength"]]
        bv_idx = push_bytes(img_bytes)
        new_imgs.append({"mimeType": im["mimeType"], "bufferView": bv_idx})

    # Tune materials for realistic metallic silver Corolla + add bay liner material
    new_mats = []
    for m in g["materials"]:
        mc = json.loads(json.dumps(m))
        name = mc.get("name", "")
        pbr = mc.setdefault("pbrMetallicRoughness", {})
        if name == "Carro_Pintura":
            # Realistic Toyota Corolla Classic Silver Metallic with crisp reflections
            pbr["baseColorFactor"] = [0.76, 0.79, 0.83, 1.0]
            pbr["metallicFactor"] = 0.82
            pbr["roughnessFactor"] = 0.22
            mc["doubleSided"] = True
        elif name == "Carro_Roda":
            pbr["baseColorFactor"] = [0.80, 0.82, 0.85, 1.0]
            pbr["metallicFactor"] = 0.85
            pbr["roughnessFactor"] = 0.25
        elif name in ("Carro_Plastico", "Carro_Metal_Preto", "Carro_Metal_Preto_1"):
            pbr["metallicFactor"] = 0.20
            pbr["roughnessFactor"] = 0.65
        elif name == "Carro_Vidros":
            pbr["baseColorFactor"] = [0.06, 0.09, 0.11, 0.55]
            pbr["metallicFactor"] = 0.15
            pbr["roughnessFactor"] = 0.08
        new_mats.append(mc)

    mat_bay_idx = len(new_mats)
    new_mats.append({
        "name": "Corolla_Engine_Bay_Matte",
        "pbrMetallicRoughness": {
            "baseColorFactor": [0.11, 0.12, 0.14, 1.0],
            "metallicFactor": 0.25,
            "roughnessFactor": 0.78,
        },
        "doubleSided": False,
    })

    new_accessors = []
    new_meshes = []
    new_nodes = []

    def add_acc(arr: np.ndarray, acc_type: str, comp_type: int, with_minmax: bool = False) -> int:
        bv_target = 34963 if acc_type == "SCALAR" else 34962
        bv_idx = push_bytes(arr.tobytes(), bv_target)
        acc = {
            "bufferView": bv_idx,
            "byteOffset": 0,
            "componentType": comp_type,
            "count": int(len(arr)),
            "type": acc_type,
        }
        if with_minmax:
            acc["min"] = [float(x) for x in arr.min(axis=0)]
            acc["max"] = [float(x) for x in arr.max(axis=0)]
        idx = len(new_accessors)
        new_accessors.append(acc)
        return idx

    def make_mesh_from_arrays(name: str, P: np.ndarray, N: np.ndarray, I: np.ndarray, mat_idx: int, UV: np.ndarray | None = None) -> int:
        P32 = np.ascontiguousarray(P, dtype=np.float32)
        N32 = np.ascontiguousarray(N, dtype=np.float32)
        I32 = np.ascontiguousarray(I, dtype=np.uint32)
        attrs = {
            "POSITION": add_acc(P32, "VEC3", 5126, with_minmax=True),
            "NORMAL": add_acc(N32, "VEC3", 5126, with_minmax=False),
        }
        if UV is not None:
            attrs["TEXCOORD_0"] = add_acc(np.ascontiguousarray(UV, dtype=np.float32), "VEC2", 5126, with_minmax=False)
        prim = {"attributes": attrs, "indices": add_acc(I32, "SCALAR", 5125, with_minmax=False), "material": mat_idx}
        mid = len(new_meshes)
        new_meshes.append({"name": name, "primitives": [prim]})
        return mid

    hood_cut_arrays = None
    body_children = []

    # Process all 61 mesh nodes, baking transforms & extracting the Corolla hood
    for nidx, n in enumerate(g["nodes"]):
        if "mesh" not in n:
            continue
        M = get_world_mat(nidx)
        R3 = np.linalg.inv(M[:3, :3]).T
        m = g["meshes"][n["mesh"]]
        for pi, prim in enumerate(m["primitives"]):
            pts = read_accessor(g, bin_chunk, prim["attributes"]["POSITION"])
            wpts = (M @ np.hstack([pts, np.ones((len(pts), 1))]).T).T[:, :3]
            if "NORMAL" in prim["attributes"]:
                nrms = read_accessor(g, bin_chunk, prim["attributes"]["NORMAL"])
                wnrms = (R3 @ nrms.T).T
                nl = np.linalg.norm(wnrms, axis=1, keepdims=True) + 1e-12
                wnrms = wnrms / nl
            else:
                wnrms = np.tile([0.0, 1.0, 0.0], (len(wpts), 1))

            uv = None
            if "TEXCOORD_0" in prim["attributes"]:
                uv = read_accessor(g, bin_chunk, prim["attributes"]["TEXCOORD_0"])

            tris = read_accessor(g, bin_chunk, prim["indices"]).astype(np.int64).reshape(-1, 3)
            mat_idx = prim.get("material", 0)
            obj_name = n.get("name", f"Mesh_{nidx}")

            if nidx == 38 and pi == 0:
                hood_arrs, body_arrs = split_hood_and_body(wpts, wnrms, tris)
                hood_cut_arrays = hood_arrs
                bP, bN, bI = body_arrs
                mid = make_mesh_from_arrays("Corolla_Exterior_Shell", bP, bN, bI, mat_idx)
            else:
                mid = make_mesh_from_arrays(obj_name, wpts, wnrms, tris.ravel(), mat_idx, uv)

            nid = len(new_nodes)
            new_nodes.append({"name": f"Node_{obj_name}", "mesh": mid})
            body_children.append(nid)

    # Add realistic matte Corolla Engine Bay Tub under the cut opening
    tub_P, tub_N, tub_I = build_engine_bay_parts()
    tub_mid = make_mesh_from_arrays("Corolla_Engine_Bay_Tub", tub_P, tub_N, tub_I, mat_bay_idx)
    tub_nid = len(new_nodes)
    new_nodes.append({"name": "Corolla_Engine_Bay_Tub", "mesh": tub_mid})
    body_children.append(tub_nid)

    # Create articulated Bonnet_Hood_Pivot containing top painted Corolla hood + dark underside liner
    hP, hN, hI = hood_cut_arrays
    hP_local = (hP - HINGE_PIVOT).astype(np.float32)
    hood_paint_mid = make_mesh_from_arrays("Corolla_Bonnet_Paint", hP_local, hN, hI, 10)

    # Back/underliner shell offset -0.012m down with inverted winding for structural depth when lifted
    uP_local = hP_local.copy()
    uP_local[:, 1] -= 0.012
    uN = -hN
    uI = hI.reshape(-1, 3)[:, [0, 2, 1]].ravel()
    hood_liner_mid = make_mesh_from_arrays("Corolla_Bonnet_Underliner", uP_local, uN, uI, mat_bay_idx)

    hood_paint_nid = len(new_nodes)
    new_nodes.append({"name": "Corolla_Bonnet_Skin", "mesh": hood_paint_mid})
    hood_liner_nid = len(new_nodes)
    new_nodes.append({"name": "Corolla_Bonnet_Liner", "mesh": hood_liner_mid})

    pivot_nid = len(new_nodes)
    new_nodes.append({
        "name": "Bonnet_Hood_Pivot",
        "translation": [float(HINGE_PIVOT[0]), float(HINGE_PIVOT[1]), float(HINGE_PIVOT[2])],
        "children": [hood_paint_nid, hood_liner_nid],
    })

    root_nid = len(new_nodes)
    new_nodes.append({
        "name": "Toyota_Corolla_2023",
        "children": body_children + [pivot_nid],
    })

    out_gltf = {
        "asset": {
            "version": "2.0",
            "generator": "AutoGuard AI Corolla GLB Pipeline (glTF 2.0 Standard)",
            "extras": {
                "title": "2023 Toyota Corolla Hybrid (AutoGuard Articulated Edition)",
                "author": "tonielpro520 (https://sketchfab.com/tonielpro520)",
                "license": "CC-BY-4.0 (http://creativecommons.org/licenses/by/4.0/)",
                "source": "https://sketchfab.com/3d-models/2023-toyota-corolla-hybrid-cd2f6b34664442ad906a40bd00136881",
                "modifications": "Dequantized to float32, repainted Classic Silver Metallic, sliced Corolla hood into Bonnet_Hood_Pivot, added structural engine bay tub.",
            },
        },
        "scene": 0,
        "scenes": [{"name": "Corolla_Diagnostic_Scene", "nodes": [root_nid]}],
        "nodes": new_nodes,
        "meshes": new_meshes,
        "materials": new_mats,
        "accessors": new_accessors,
        "bufferViews": new_bvs,
        "buffers": [{"byteLength": len(new_bin)}],
    }
    if new_imgs:
        out_gltf["images"] = new_imgs
        out_gltf["textures"] = g["textures"]
        out_gltf["samplers"] = g["samplers"]
        out_gltf["extensionsUsed"] = ["EXT_texture_webp"]
        out_gltf["extensionsRequired"] = ["EXT_texture_webp"]

    json_bytes = json.dumps(out_gltf, separators=(",", ":")).encode("utf-8")
    while len(json_bytes) % 4 != 0:
        json_bytes += b" "
    while len(new_bin) % 4 != 0:
        new_bin.append(0)

    total_len = 12 + 8 + len(json_bytes) + 8 + len(new_bin)
    with open(OUT_GLB, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, total_len))
        f.write(struct.pack("<II", len(json_bytes), 0x4E4F534A))
        f.write(json_bytes)
        f.write(struct.pack("<II", len(new_bin), 0x004E4942))
        f.write(new_bin)

    print(f"Generated {OUT_GLB} ({os.path.getsize(OUT_GLB)/1024/1024:.2f} MB)")
    print(f"  Hood tris: {len(hI)//3}  Body shell tris: {len(bI)//3}")
    print(f"  Hood local bbox: X=[{hP_local[:,0].min():+.3f},{hP_local[:,0].max():+.3f}] Y=[{hP_local[:,1].min():+.3f},{hP_local[:,1].max():+.3f}] Z=[{hP_local[:,2].min():+.3f},{hP_local[:,2].max():+.3f}]")


if __name__ == "__main__":
    main()
