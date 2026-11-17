"""
generate_engine_glb.py
----------------------
Generates a genuine, standards-compliant glTF 2.0 binary (.glb) 3D model of a
4-Cylinder DOHC Internal Combustion Engine (`frontend/public/models/autoguard_i4_engine.glb`)
with 34 individually named, raycast-selectable, and kinematically animatable 3D nodes.

Every node in the GLB scene graph corresponds to an inspectable mechanical component
mapped to AutoGuard AI's three supported ML classes (`healthy`, `misfire`, `engine_knock`).
"""

from __future__ import annotations

import json
import math
import os
import struct
from typing import List, Tuple
import numpy as np


# ── Geometry Primitives (returns vertices [N,3], normals [N,3], indices [M,3]) ──

def _transform_verts(v: np.ndarray, scale=(1.0, 1.0, 1.0), rot=(0.0, 0.0, 0.0), pos=(0.0, 0.0, 0.0)) -> Tuple[np.ndarray, np.ndarray]:
    sx, sy, sz = scale
    rx, ry, rz = rot
    px, py, pz = pos

    v_s = v * np.array([sx, sy, sz], dtype=np.float32)

    cx, sx_ = math.cos(rx), math.sin(rx)
    cy, sy_ = math.cos(ry), math.sin(ry)
    cz, sz_ = math.cos(rz), math.sin(rz)

    Rx = np.array([[1, 0, 0], [0, cx, -sx_], [0, sx_, cx]], dtype=np.float32)
    Ry = np.array([[cy, 0, sy_], [0, 1, 0], [-sy_, 0, cy]], dtype=np.float32)
    Rz = np.array([[cz, -sz_, 0], [sz_, cz, 0], [0, 0, 1]], dtype=np.float32)
    R = Rz @ Ry @ Rx

    v_r = v_s @ R.T
    v_out = v_r + np.array([px, py, pz], dtype=np.float32)
    return v_out, R


def make_box(size=(1.0, 1.0, 1.0), pos=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0)):
    hx, hy, hz = size[0] * 0.5, size[1] * 0.5, size[2] * 0.5
    faces = [
        # +Z
        ([(-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz)], (0, 0, 1)),
        # -Z
        ([(hx, -hy, -hz), (-hx, -hy, -hz), (-hx, hy, -hz), (hx, hy, -hz)], (0, 0, -1)),
        # +X
        ([(hx, -hy, hz), (hx, -hy, -hz), (hx, hy, -hz), (hx, hy, hz)], (1, 0, 0)),
        # -X
        ([(-hx, -hy, -hz), (-hx, -hy, hz), (-hx, hy, hz), (-hx, hy, -hz)], (-1, 0, 0)),
        # +Y
        ([(-hx, hy, hz), (hx, hy, hz), (hx, hy, -hz), (-hx, hy, -hz)], (0, 1, 0)),
        # -Y
        ([(-hx, -hy, -hz), (hx, -hy, -hz), (hx, -hy, hz), (-hx, -hy, hz)], (0, -1, 0)),
    ]
    verts = []
    norms = []
    tris = []
    for quad, n in faces:
        base = len(verts)
        for pt in quad:
            verts.append(pt)
            norms.append(n)
        tris.append((base, base + 1, base + 2))
        tris.append((base, base + 2, base + 3))

    v_arr = np.array(verts, dtype=np.float32)
    n_arr = np.array(norms, dtype=np.float32)
    v_out, R = _transform_verts(v_arr, (1.0, 1.0, 1.0), rot, pos)
    n_out = n_arr @ R.T
    return v_out, n_out, np.array(tris, dtype=np.uint32)


def make_cylinder(radius_top=0.5, radius_bottom=0.5, height=1.0, segs=24, pos=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0)):
    verts = []
    norms = []
    tris = []
    half_h = height * 0.5
    slope = (radius_bottom - radius_top) / max(height, 1e-6)

    # Side wall
    for i in range(segs):
        a0 = (i / segs) * 2.0 * math.pi
        a1 = ((i + 1) / segs) * 2.0 * math.pi
        c0, s0 = math.cos(a0), math.sin(a0)
        c1, s1 = math.cos(a1), math.sin(a1)

        base = len(verts)
        verts.extend([
            (radius_bottom * c0, -half_h, radius_bottom * s0),
            (radius_bottom * c1, -half_h, radius_bottom * s1),
            (radius_top * c1, half_h, radius_top * s1),
            (radius_top * c0, half_h, radius_top * s0),
        ])
        n0 = np.array([c0, slope, s0], dtype=np.float32)
        n0 /= np.linalg.norm(n0) + 1e-8
        n1 = np.array([c1, slope, s1], dtype=np.float32)
        n1 /= np.linalg.norm(n1) + 1e-8
        norms.extend([n0, n1, n1, n0])
        tris.append((base, base + 2, base + 1))
        tris.append((base, base + 3, base + 2))

    # Top cap
    top_center = len(verts)
    verts.append((0.0, half_h, 0.0))
    norms.append((0.0, 1.0, 0.0))
    for i in range(segs):
        a0 = (i / segs) * 2.0 * math.pi
        a1 = ((i + 1) / segs) * 2.0 * math.pi
        b = len(verts)
        verts.append((radius_top * math.cos(a0), half_h, radius_top * math.sin(a0)))
        verts.append((radius_top * math.cos(a1), half_h, radius_top * math.sin(a1)))
        norms.extend([(0.0, 1.0, 0.0), (0.0, 1.0, 0.0)])
        tris.append((top_center, b + 1, b))

    # Bottom cap
    bot_center = len(verts)
    verts.append((0.0, -half_h, 0.0))
    norms.append((0.0, -1.0, 0.0))
    for i in range(segs):
        a0 = (i / segs) * 2.0 * math.pi
        a1 = ((i + 1) / segs) * 2.0 * math.pi
        b = len(verts)
        verts.append((radius_bottom * math.cos(a0), -half_h, radius_bottom * math.sin(a0)))
        verts.append((radius_bottom * math.cos(a1), -half_h, radius_bottom * math.sin(a1)))
        norms.extend([(0.0, -1.0, 0.0), (0.0, -1.0, 0.0)])
        tris.append((bot_center, b, b + 1))

    v_arr = np.array(verts, dtype=np.float32)
    n_arr = np.array(norms, dtype=np.float32)
    v_out, R = _transform_verts(v_arr, (1.0, 1.0, 1.0), rot, pos)
    n_out = n_arr @ R.T
    return v_out, n_out, np.array(tris, dtype=np.uint32)


def make_tube(r_outer=0.5, r_inner=0.4, height=1.0, segs=24, pos=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0)):
    verts = []
    norms = []
    tris = []
    hh = height * 0.5

    for i in range(segs):
        a0 = (i / segs) * 2.0 * math.pi
        a1 = ((i + 1) / segs) * 2.0 * math.pi
        c0, s0 = math.cos(a0), math.sin(a0)
        c1, s1 = math.cos(a1), math.sin(a1)

        # Outer wall
        b = len(verts)
        verts.extend([
            (r_outer * c0, -hh, r_outer * s0),
            (r_outer * c1, -hh, r_outer * s1),
            (r_outer * c1, hh, r_outer * s1),
            (r_outer * c0, hh, r_outer * s0),
        ])
        norms.extend([(c0, 0, s0), (c1, 0, s1), (c1, 0, s1), (c0, 0, s0)])
        tris.extend([(b, b + 2, b + 1), (b, b + 3, b + 2)])

        # Inner wall
        b = len(verts)
        verts.extend([
            (r_inner * c0, -hh, r_inner * s0),
            (r_inner * c1, -hh, r_inner * s1),
            (r_inner * c1, hh, r_inner * s1),
            (r_inner * c0, hh, r_inner * s0),
        ])
        norms.extend([(-c0, 0, -s0), (-c1, 0, -s1), (-c1, 0, -s1), (-c0, 0, -s0)])
        tris.extend([(b, b + 1, b + 2), (b, b + 2, b + 3)])

        # Top ring
        b = len(verts)
        verts.extend([
            (r_inner * c0, hh, r_inner * s0),
            (r_outer * c0, hh, r_outer * s0),
            (r_outer * c1, hh, r_outer * s1),
            (r_inner * c1, hh, r_inner * s1),
        ])
        norms.extend([(0, 1, 0)] * 4)
        tris.extend([(b, b + 2, b + 1), (b, b + 3, b + 2)])

        # Bottom ring
        b = len(verts)
        verts.extend([
            (r_inner * c0, -hh, r_inner * s0),
            (r_outer * c0, -hh, r_outer * s0),
            (r_outer * c1, -hh, r_outer * s1),
            (r_inner * c1, -hh, r_inner * s1),
        ])
        norms.extend([(0, -1, 0)] * 4)
        tris.extend([(b, b + 1, b + 2), (b, b + 2, b + 3)])

    v_arr = np.array(verts, dtype=np.float32)
    n_arr = np.array(norms, dtype=np.float32)
    v_out, R = _transform_verts(v_arr, (1.0, 1.0, 1.0), rot, pos)
    n_out = n_arr @ R.T
    return v_out, n_out, np.array(tris, dtype=np.uint32)


def merge_meshes(parts: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]):
    all_v = []
    all_n = []
    all_t = []
    offset = 0
    for v, n, t in parts:
        all_v.append(v)
        all_n.append(n)
        all_t.append(t + offset)
        offset += len(v)
    return (
        np.vstack(all_v).astype(np.float32),
        np.vstack(all_n).astype(np.float32),
        np.vstack(all_t).astype(np.uint32),
    )


# ── Build Inline-4 DOHC Engine Assemblies ──────────────────────────────────────

CYL_X = [-1.35, -0.45, 0.45, 1.35]  # 4 cylinder centers along X axis
CRANK_PHASES = [0.0, math.pi, math.pi, 0.0]  # 1-3-4-2 flat-plane inline-4 crank throws


def build_engine_nodes():
    """
    Returns list of dicts:
        name, mesh (v, n, t), material_idx, translation
    """
    nodes = []

    # Materials palette:
    # 0: Cast Aluminum Block (translucent-capable metallic)
    # 1: Polished Chrome / Forged Steel (Crank, Cams, Rods, Valves)
    # 2: Machined Piston Alloy
    # 3: Ceramic / Spark Plug White
    # 4: Ignition Coil Matte Carbon / Anodized Red-Top
    # 5: Fuel Injector Anodized Blue / Steel Rail
    # 6: Combustion Chamber Amber / Volumetric Glow
    # 7: Dark Rubber / Timing Belt
    # 8: Cast Iron / Bronze Exhaust Header
    # 9: Deep Gunmetal Oil Sump

    # 1. ENGINE BLOCK (4 cylinder liner tubes + side crankcase walls + cooling fins)
    block_parts = []
    for cx in CYL_X:
        block_parts.append(make_tube(0.42, 0.36, 1.45, segs=28, pos=(cx, 0.05, 0.0)))
    # Deck plates & crankcase rails
    block_parts.append(make_box((3.9, 0.10, 1.12), pos=(0.0, 0.76, 0.0)))
    block_parts.append(make_box((3.9, 0.12, 1.18), pos=(0.0, -0.68, 0.0)))
    # Side cooling ribs
    for rib_y in [-0.35, -0.10, 0.15, 0.40]:
        block_parts.append(make_box((3.85, 0.04, 1.06), pos=(0.0, rib_y, 0.0)))
    # Main bearing webs
    for wx in [-1.80, -0.90, 0.0, 0.90, 1.80]:
        block_parts.append(make_box((0.10, 0.75, 1.12), pos=(wx, -0.85, 0.0)))
    nodes.append({
        "name": "Engine_Block",
        "mesh": merge_meshes(block_parts),
        "material": 0,
        "translation": [0.0, 0.0, 0.0],
    })

    # 2. CYLINDER HEAD (DOHC upper housing)
    head_parts = [
        make_box((3.88, 0.56, 1.16), pos=(0.0, 1.12, 0.0)),
        make_box((3.96, 0.08, 1.24), pos=(0.0, 0.84, 0.0)),
        make_box((3.94, 0.08, 1.22), pos=(0.0, 1.40, 0.0)),
    ]
    for cx in CYL_X:
        # Spark plug wells
        head_parts.append(make_tube(0.14, 0.09, 0.62, segs=18, pos=(cx, 1.15, 0.0)))
    nodes.append({
        "name": "Cylinder_Head",
        "mesh": merge_meshes(head_parts),
        "material": 0,
        "translation": [0.0, 0.0, 0.0],
    })

    # 3. VALVE COVER
    vc_parts = [
        make_box((3.76, 0.24, 1.08), pos=(0.0, 1.56, 0.0)),
        make_box((3.56, 0.08, 0.42), pos=(0.0, 1.70, 0.30)),
        make_box((3.56, 0.08, 0.42), pos=(0.0, 1.70, -0.30)),
        make_cylinder(0.16, 0.16, 0.14, segs=20, pos=(-1.45, 1.78, 0.30)),  # Oil cap
    ]
    nodes.append({
        "name": "Valve_Cover",
        "mesh": merge_meshes(vc_parts),
        "material": 0,
        "translation": [0.0, 0.0, 0.0],
    })

    # 4. OIL PAN (Lower lubrication sump)
    pan_parts = [
        make_box((3.86, 0.08, 1.18), pos=(0.0, -1.28, 0.0)),
        make_box((3.68, 0.42, 1.04), pos=(0.0, -1.52, 0.0)),
        make_box((2.40, 0.24, 0.88), pos=(0.35, -1.80, 0.0)),
        make_cylinder(0.08, 0.08, 0.12, segs=12, pos=(1.35, -1.86, 0.0), rot=(0, 0, math.pi / 2)),
    ]
    nodes.append({
        "name": "Oil_Pan",
        "mesh": merge_meshes(pan_parts),
        "material": 9,
        "translation": [0.0, 0.0, 0.0],
    })

    # 5. CRANKSHAFT (Centered at Y = -0.95 so rotation around X is pure crank rotation!)
    crank_parts = [
        # Main journal shaft along X
        make_cylinder(0.11, 0.11, 4.35, segs=24, pos=(0.0, 0.0, 0.0), rot=(0.0, 0.0, math.pi / 2)),
        # Rear flywheel
        make_cylinder(0.56, 0.56, 0.14, segs=32, pos=(2.18, 0.0, 0.0), rot=(0.0, 0.0, math.pi / 2)),
        make_tube(0.60, 0.53, 0.16, segs=32, pos=(2.18, 0.0, 0.0), rot=(0.0, 0.0, math.pi / 2)),
        # Front crank pulley
        make_cylinder(0.28, 0.28, 0.16, segs=24, pos=(-2.18, 0.0, 0.0), rot=(0.0, 0.0, math.pi / 2)),
    ]
    throw_r = 0.26
    for idx, cx in enumerate(CYL_X):
        ph = CRANK_PHASES[idx]
        ty = math.cos(ph) * throw_r
        tz = math.sin(ph) * throw_r
        # Crankpin journal
        crank_parts.append(make_cylinder(0.095, 0.095, 0.34, segs=18, pos=(cx, ty, tz), rot=(0, 0, math.pi / 2)))
        # Pair of sculpted crank webs / counterweights
        for wx in [cx - 0.21, cx + 0.21]:
            crank_parts.append(make_box((0.08, 0.52, 0.28), pos=(wx, ty * 0.35, tz * 0.35)))
            crank_parts.append(make_cylinder(0.24, 0.24, 0.08, segs=18, pos=(wx, -ty * 0.45, -tz * 0.45), rot=(0, 0, math.pi / 2)))

    nodes.append({
        "name": "Crankshaft",
        "mesh": merge_meshes(crank_parts),
        "material": 1,
        "translation": [0.0, -0.95, 0.0],
    })

    # 6. PISTONS, CONNECTING RODS & COMBUSTION CHAMBERS (1..4)
    for idx, cx in enumerate(CYL_X):
        c_num = idx + 1
        ph = CRANK_PHASES[idx]
        piston_y = 0.05 + math.cos(ph) * 0.26

        # Piston crown + ring lands + wrist pin
        p_parts = [
            make_cylinder(0.345, 0.345, 0.34, segs=28, pos=(0.0, 0.0, 0.0)),
            make_tube(0.352, 0.335, 0.025, segs=28, pos=(0.0, 0.11, 0.0)),
            make_tube(0.352, 0.335, 0.025, segs=28, pos=(0.0, 0.05, 0.0)),
            make_tube(0.352, 0.335, 0.025, segs=28, pos=(0.0, -0.01, 0.0)),
            make_cylinder(0.075, 0.075, 0.58, segs=16, pos=(0.0, -0.04, 0.0), rot=(math.pi / 2, 0, 0)),
        ]
        nodes.append({
            "name": f"Piston_{c_num}",
            "mesh": merge_meshes(p_parts),
            "material": 2,
            "translation": [cx, piston_y, 0.0],
        })

        # Connecting Rod
        r_parts = [
            make_box((0.09, 0.78, 0.11), pos=(0.0, -0.38, 0.0)),
            make_tube(0.16, 0.10, 0.14, segs=20, pos=(0.0, -0.76, 0.0), rot=(0, 0, math.pi / 2)),
            make_tube(0.11, 0.07, 0.12, segs=18, pos=(0.0, 0.0, 0.0), rot=(0, 0, math.pi / 2)),
        ]
        nodes.append({
            "name": f"Connecting_Rod_{c_num}",
            "mesh": merge_meshes(r_parts),
            "material": 1,
            "translation": [cx, piston_y, 0.0],
        })

        # Combustion Chamber Volumetric Dome (top of cylinder bore)
        cc_parts = [
            make_cylinder(0.31, 0.34, 0.28, segs=24, pos=(0.0, 0.0, 0.0)),
        ]
        nodes.append({
            "name": f"Combustion_Chamber_{c_num}",
            "mesh": merge_meshes(cc_parts),
            "material": 6,
            "translation": [cx, 0.56, 0.0],
        })

        # Spark Plug (1..4) — ceramic body + hex nut + electrode tip
        sp_parts = [
            make_cylinder(0.055, 0.055, 0.36, segs=16, pos=(0.0, 0.18, 0.0)),
            make_cylinder(0.085, 0.085, 0.09, segs=6, pos=(0.0, 0.0, 0.0)),
            make_cylinder(0.042, 0.042, 0.16, segs=14, pos=(0.0, -0.12, 0.0)),
            make_cylinder(0.016, 0.016, 0.07, segs=10, pos=(0.0, -0.22, 0.0)),
        ]
        nodes.append({
            "name": f"Spark_Plug_{c_num}",
            "mesh": merge_meshes(sp_parts),
            "material": 3,
            "translation": [cx, 0.94, 0.0],
        })

        # Ignition Coil (1..4) — Coil-on-Plug module above spark plug
        ic_parts = [
            make_cylinder(0.075, 0.065, 0.34, segs=16, pos=(0.0, -0.12, 0.0)),
            make_box((0.22, 0.12, 0.24), pos=(0.0, 0.08, 0.0)),
            make_box((0.14, 0.07, 0.18), pos=(0.0, 0.14, 0.12)),
        ]
        nodes.append({
            "name": f"Ignition_Coil_{c_num}",
            "mesh": merge_meshes(ic_parts),
            "material": 4,
            "translation": [cx, 1.48, 0.0],
        })

        # Fuel Injector (1..4) — angled into intake runner
        fi_parts = [
            make_cylinder(0.052, 0.052, 0.28, segs=14, pos=(0.0, 0.0, 0.0), rot=(0.48, 0.0, 0.0)),
            make_cylinder(0.028, 0.022, 0.14, segs=12, pos=(0.0, -0.16, -0.08), rot=(0.48, 0.0, 0.0)),
            make_box((0.09, 0.08, 0.10), pos=(0.0, 0.10, 0.06)),
        ]
        nodes.append({
            "name": f"Fuel_Injector_{c_num}",
            "mesh": merge_meshes(fi_parts),
            "material": 5,
            "translation": [cx, 1.08, 0.56],
        })

    # 7. FUEL RAIL (Common rail feeding all 4 injectors)
    rail_parts = [
        make_cylinder(0.065, 0.065, 3.45, segs=20, pos=(0.0, 0.0, 0.0), rot=(0, 0, math.pi / 2)),
        make_cylinder(0.11, 0.11, 0.14, segs=16, pos=(-1.78, 0.0, 0.0), rot=(0, 0, math.pi / 2)),
    ]
    for cx in CYL_X:
        rail_parts.append(make_cylinder(0.045, 0.045, 0.14, segs=12, pos=(cx, -0.08, -0.04), rot=(0.48, 0, 0)))
    nodes.append({
        "name": "Fuel_Rail",
        "mesh": merge_meshes(rail_parts),
        "material": 5,
        "translation": [0.0, 1.24, 0.64],
    })

    # 8. DUAL OVERHEAD CAMSHAFTS (Intake & Exhaust)
    for side, z_pos, name in [("Intake", 0.30, "Camshaft_Intake"), ("Exhaust", -0.30, "Camshaft_Exhaust")]:
        cam_parts = [
            make_cylinder(0.068, 0.068, 4.15, segs=20, pos=(0.0, 0.0, 0.0), rot=(0, 0, math.pi / 2)),
            # Front cam sprocket
            make_cylinder(0.25, 0.25, 0.09, segs=28, pos=(-2.12, 0.0, 0.0), rot=(0, 0, math.pi / 2)),
            make_tube(0.28, 0.22, 0.11, segs=28, pos=(-2.12, 0.0, 0.0), rot=(0, 0, math.pi / 2)),
        ]
        for idx, cx in enumerate(CYL_X):
            lobe_angle = CRANK_PHASES[idx] * 0.5 + (0.4 if side == "Intake" else -0.4)
            ly = math.cos(lobe_angle) * 0.055
            lz = math.sin(lobe_angle) * 0.055
            for lx in [cx - 0.14, cx + 0.14]:
                cam_parts.append(make_cylinder(0.11, 0.11, 0.065, segs=16, pos=(lx, ly, lz), rot=(0, 0, math.pi / 2)))
        nodes.append({
            "name": name,
            "mesh": merge_meshes(cam_parts),
            "material": 1,
            "translation": [0.0, 1.34, z_pos],
        })

    # 9. INTAKE & EXHAUST VALVES (16 valves total)
    for v_name, z_offset, tilt in [
        ("Intake_Valves", 0.19, -0.22),
        ("Exhaust_Valves", -0.19, 0.22),
    ]:
        v_parts = []
        for cx in CYL_X:
            for vx in [cx - 0.14, cx + 0.14]:
                # Valve stem
                v_parts.append(make_cylinder(0.022, 0.022, 0.44, segs=12, pos=(vx, 0.0, 0.0), rot=(tilt, 0, 0)))
                # Valve head disc
                v_parts.append(make_cylinder(0.105, 0.03, 0.035, segs=18, pos=(vx, -0.21, -math.sin(tilt) * 0.21), rot=(tilt, 0, 0)))
                # Valve spring cage
                v_parts.append(make_tube(0.068, 0.045, 0.22, segs=14, pos=(vx, 0.05, math.sin(tilt) * 0.05), rot=(tilt, 0, 0)))
        nodes.append({
            "name": v_name,
            "mesh": merge_meshes(v_parts),
            "material": 1,
            "translation": [0.0, 0.96, z_offset],
        })

    # 10. TIMING DRIVE BELT & TENSIONERS
    belt_parts = [
        make_box((0.12, 2.35, 0.06), pos=(0.0, 0.20, 0.34)),
        make_box((0.12, 2.35, 0.06), pos=(0.0, 0.20, -0.34)),
        make_box((0.12, 0.06, 0.68), pos=(0.0, 1.42, 0.0)),
        make_cylinder(0.14, 0.14, 0.12, segs=20, pos=(0.0, 0.35, 0.18), rot=(0, 0, math.pi / 2)),
    ]
    nodes.append({
        "name": "Timing_Drive_Belt",
        "mesh": merge_meshes(belt_parts),
        "material": 7,
        "translation": [-2.14, 0.0, 0.0],
    })

    # 11. EXHAUST MANIFOLD (4-into-1 tubular header)
    ex_parts = [
        make_box((3.50, 0.22, 0.08), pos=(0.0, 0.0, 0.0)),
        make_cylinder(0.15, 0.15, 1.35, segs=20, pos=(0.0, -0.38, -0.38), rot=(0, 0, math.pi / 2)),
    ]
    for cx in CYL_X:
        ex_parts.append(make_cylinder(0.095, 0.095, 0.54, segs=16, pos=(cx, -0.18, -0.22), rot=(0.75, 0, 0)))
    nodes.append({
        "name": "Exhaust_Manifold",
        "mesh": merge_meshes(ex_parts),
        "material": 8,
        "translation": [0.0, 1.02, -0.62],
    })

    return nodes


def export_glb(output_path: str):
    nodes_data = build_engine_nodes()

    materials = [
        # 0: Cast Aluminum Block
        {
            "name": "Cast_Aluminum_Block",
            "pbrMetallicRoughness": {
                "baseColorFactor": [0.72, 0.77, 0.84, 1.0],
                "metallicFactor": 0.82,
                "roughnessFactor": 0.28,
            },
        },
        # 1: Polished Forged Steel (Crank, Cams, Rods, Valves)
        {
            "name": "Forged_Polished_Steel",
            "pbrMetallicRoughness": {
                "baseColorFactor": [0.88, 0.91, 0.95, 1.0],
                "metallicFactor": 0.94,
                "roughnessFactor": 0.16,
            },
        },
        # 2: Machined Piston Alloy
        {
            "name": "Piston_Billet_Alloy",
            "pbrMetallicRoughness": {
                "baseColorFactor": [0.82, 0.85, 0.90, 1.0],
                "metallicFactor": 0.88,
                "roughnessFactor": 0.22,
            },
        },
        # 3: Spark Plug Ceramic / Nickel
        {
            "name": "Spark_Plug_Ceramic",
            "pbrMetallicRoughness": {
                "baseColorFactor": [0.96, 0.97, 0.99, 1.0],
                "metallicFactor": 0.25,
                "roughnessFactor": 0.20,
            },
        },
        # 4: Ignition Coil Pack
        {
            "name": "Ignition_Coil_Pack",
            "pbrMetallicRoughness": {
                "baseColorFactor": [0.22, 0.26, 0.34, 1.0],
                "metallicFactor": 0.45,
                "roughnessFactor": 0.42,
            },
        },
        # 5: Fuel Injector & High-Pressure Rail
        {
            "name": "Fuel_Injection_System",
            "pbrMetallicRoughness": {
                "baseColorFactor": [0.35, 0.68, 0.94, 1.0],
                "metallicFactor": 0.78,
                "roughnessFactor": 0.25,
            },
        },
        # 6: Combustion Chamber Dome
        {
            "name": "Combustion_Chamber_Zone",
            "pbrMetallicRoughness": {
                "baseColorFactor": [0.98, 0.65, 0.15, 0.45],
                "metallicFactor": 0.1,
                "roughnessFactor": 0.3,
            },
            "alphaMode": "BLEND",
        },
        # 7: Timing Drive Belt
        {
            "name": "Timing_Belt_Reinforced",
            "pbrMetallicRoughness": {
                "baseColorFactor": [0.16, 0.19, 0.24, 1.0],
                "metallicFactor": 0.20,
                "roughnessFactor": 0.65,
            },
        },
        # 8: Exhaust Header Bronze-Steel
        {
            "name": "Exhaust_Manifold_Alloy",
            "pbrMetallicRoughness": {
                "baseColorFactor": [0.68, 0.56, 0.46, 1.0],
                "metallicFactor": 0.85,
                "roughnessFactor": 0.34,
            },
        },
        # 9: Deep Gunmetal Oil Sump
        {
            "name": "Oil_Pan_Gunmetal",
            "pbrMetallicRoughness": {
                "baseColorFactor": [0.32, 0.37, 0.45, 1.0],
                "metallicFactor": 0.78,
                "roughnessFactor": 0.35,
            },
        },
    ]

    bin_chunks = bytearray()
    buffer_views = []
    accessors = []
    meshes = []
    gltf_nodes = []

    def append_bytes(raw: bytes, target: int) -> int:
        # 4-byte align
        while len(bin_chunks) % 4 != 0:
            bin_chunks.append(0)
        offset = len(bin_chunks)
        bin_chunks.extend(raw)
        while len(bin_chunks) % 4 != 0:
            bin_chunks.append(0)
        bv_idx = len(buffer_views)
        buffer_views.append({
            "buffer": 0,
            "byteOffset": offset,
            "byteLength": len(raw),
            "target": target,
        })
        return bv_idx

    for nd in nodes_data:
        v, n, t = nd["mesh"]
        idx_flat = t.flatten().astype(np.uint32)

        bv_pos = append_bytes(v.tobytes(), 34962)  # ARRAY_BUFFER
        bv_nrm = append_bytes(n.tobytes(), 34962)
        bv_idx = append_bytes(idx_flat.tobytes(), 34963)  # ELEMENT_ARRAY_BUFFER

        acc_pos = len(accessors)
        accessors.append({
            "bufferView": bv_pos,
            "componentType": 5126,  # FLOAT
            "count": int(len(v)),
            "type": "VEC3",
            "min": [float(x) for x in v.min(axis=0)],
            "max": [float(x) for x in v.max(axis=0)],
        })

        acc_nrm = len(accessors)
        accessors.append({
            "bufferView": bv_nrm,
            "componentType": 5126,
            "count": int(len(n)),
            "type": "VEC3",
        })

        acc_idx = len(accessors)
        accessors.append({
            "bufferView": bv_idx,
            "componentType": 5125,  # UNSIGNED_INT
            "count": int(len(idx_flat)),
            "type": "SCALAR",
        })

        mesh_idx = len(meshes)
        meshes.append({
            "name": f"{nd['name']}_Mesh",
            "primitives": [
                {
                    "attributes": {
                        "POSITION": acc_pos,
                        "NORMAL": acc_nrm,
                    },
                    "indices": acc_idx,
                    "material": nd["material"],
                }
            ],
        })

        gltf_nodes.append({
            "name": nd["name"],
            "mesh": mesh_idx,
            "translation": nd["translation"],
        })

    root_node_idx = len(gltf_nodes)
    gltf_nodes.append({
        "name": "AutoGuard_I4_DOHC_Engine",
        "children": list(range(root_node_idx)),
    })

    gltf_json = {
        "asset": {
            "version": "2.0",
            "generator": "AutoGuard AI Procedural CAD GLB Builder v2.0",
            "copyright": "CC0 1.0 Universal - AutoGuard AI Open CAD Asset",
        },
        "scene": 0,
        "scenes": [
            {
                "name": "EngineLabScene",
                "nodes": [root_node_idx],
            }
        ],
        "nodes": gltf_nodes,
        "meshes": meshes,
        "materials": materials,
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [
            {
                "byteLength": len(bin_chunks),
            }
        ],
    }

    json_bytes = json.dumps(gltf_json, separators=(",", ":")).encode("utf-8")
    while len(json_bytes) % 4 != 0:
        json_bytes += b" "

    total_length = 12 + 8 + len(json_bytes) + 8 + len(bin_chunks)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        # GLB Header: magic ('glTF' = 0x46546C67), version (2), total_length
        f.write(struct.pack("<III", 0x46546C67, 2, total_length))
        # JSON Chunk: length, type ('JSON' = 0x4E4F534A)
        f.write(struct.pack("<II", len(json_bytes), 0x4E4F534A))
        f.write(json_bytes)
        # BIN Chunk: length, type ('BIN\0' = 0x004E4942)
        f.write(struct.pack("<II", len(bin_chunks), 0x004E4942))
        f.write(bin_chunks)

    print(f"Generated {output_path}: {total_length / 1024:.1f} KB, {len(nodes_data)} named 3D component nodes.")


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(root, "frontend", "public", "models", "autoguard_i4_engine.glb")
    export_glb(out)
