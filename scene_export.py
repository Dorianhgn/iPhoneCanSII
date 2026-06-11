"""
scene_export.py — Bake du nuage de points au format du pitch (spec §4)
-----------------------------------------------------------------------
Produit le **contrat de données** attendu par la web-app de présentation :
3 états couleur partageant **exactement la même géométrie / le même ordre de
sommets**, plus des labels pour l'état segmentation.

États (mêmes positions, seules les couleurs changent) :
  - neutral      : reconstruction brute (RGB capteur)
  - obstacles    : points "obstacle" en rouge (normales ~ horizontales),
                   le reste atténué
  - segmentation : couleurs par classe ICANSII sur les objets reconnus,
                   le reste en gris neutre

Sorties dans `out_dir/scene/` :
  - scene.json                      (recommandé §4 : positions + 3 colors + labels)
  - neutral.ply / obstacles.ply / segmentation.ply   (fallback, même ordre de sommets)

Ce module n'utilise **que numpy + json** (pas d'open3d) → importable et testable
isolément. Les .ply sont écrits en binaire little-endian à la main.
"""

import json
import os
import numpy as np


# Couleurs par défaut (RGB 0-255)
OBSTACLE_RED   = [220,  38,  38]
NEUTRAL_GRAY   = [ 90,  95, 105]   # "reste en gris neutre" (état segmentation)


def _obstacle_mask(normals, angle_axis, angle_threshold_deg):
    """
    Reproduit la convention de view_ply.py : un point est "obstacle" si sa normale
    s'écarte de l'axe vertical de plus de `angle_threshold_deg` (surface ~verticale
    : mur, objet debout). Le sol (normale ~ // axe) n'est PAS un obstacle.
    """
    axis = np.asarray(angle_axis, dtype=np.float64)
    axis /= (np.linalg.norm(axis) + 1e-12)
    cos = np.abs(normals @ axis)
    cos_thr = np.cos(np.deg2rad(angle_threshold_deg))
    return cos < cos_thr


def _write_ply_binary(path, positions, colors_uint8, normals=None):
    """Écrit un .ply binaire little-endian (x y z [nx ny nz] red green blue)."""
    n = len(positions)
    has_normals = normals is not None
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {n}",
              "property float x", "property float y", "property float z"]
    if has_normals:
        header += ["property float nx", "property float ny", "property float nz"]
    header += ["property uchar red", "property uchar green", "property uchar blue",
               "end_header\n"]
    header_bytes = ("\n".join(header)).encode("ascii")

    cols = ["x", "y", "z"]
    fmt = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
    if has_normals:
        fmt += [("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4")]
    fmt += [("red", "u1"), ("green", "u1"), ("blue", "u1")]

    arr = np.empty(n, dtype=fmt)
    arr["x"], arr["y"], arr["z"] = positions[:, 0], positions[:, 1], positions[:, 2]
    if has_normals:
        arr["nx"], arr["ny"], arr["nz"] = normals[:, 0], normals[:, 1], normals[:, 2]
    arr["red"], arr["green"], arr["blue"] = (colors_uint8[:, 0], colors_uint8[:, 1],
                                             colors_uint8[:, 2])
    with open(path, "wb") as f:
        f.write(header_bytes)
        f.write(arr.tobytes())


def bake_scene(
    positions,
    neutral_rgb,
    *,
    normals=None,
    class_ids=None,
    taxonomy=None,
    angle_axis=(0.0, 1.0, 0.0),
    angle_threshold_deg=30.0,
    obstacle_dim=0.30,
    max_points=200_000,
    recenter=True,
    min_points_per_label=40,
    out_dir=".",
    seed=0,
):
    """
    Construit les 3 états couleur + labels et les écrit (scene.json + 3 .ply).

    positions     : (N,3) float
    neutral_rgb   : (N,3) uint8 (0-255)
    normals       : (N,3) float ou None (si None → obstacles = neutral)
    class_ids     : (N,) int (id ICANSII, -1 = aucun) ou None (→ pas de segmentation)
    taxonomy      : segmentation.Taxonomy (requis si class_ids fourni)

    Retourne un dict de stats.
    """
    positions = np.ascontiguousarray(positions, dtype=np.float64)
    neutral_rgb = np.ascontiguousarray(neutral_rgb)
    if neutral_rgb.dtype != np.uint8:
        neutral_rgb = np.clip(neutral_rgb * (255.0 if neutral_rgb.max() <= 1.0 else 1.0),
                              0, 255).astype(np.uint8)
    N0 = len(positions)

    # ── Budget points : sous-échantillonnage UNIQUE (préserve l'ordre partagé) ──
    if max_points and N0 > max_points:
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(N0, size=max_points, replace=False))
        positions = positions[keep]
        neutral_rgb = neutral_rgb[keep]
        if normals is not None:
            normals = np.asarray(normals)[keep]
        if class_ids is not None:
            class_ids = np.asarray(class_ids)[keep]
    N = len(positions)

    # ── Recentrage (cosmétique : auto-rotation centrée côté web) ────────────────
    centroid = positions.mean(axis=0)
    if recenter:
        positions = positions - centroid

    # ── État neutral ────────────────────────────────────────────────────────────
    col_neutral = neutral_rgb.astype(np.uint8)

    # ── État obstacles ──────────────────────────────────────────────────────────
    n_obstacles = 0
    if normals is not None:
        normals = np.ascontiguousarray(normals, dtype=np.float64)
        obs_mask = _obstacle_mask(normals, angle_axis, angle_threshold_deg)
        n_obstacles = int(obs_mask.sum())
        col_obstacles = (col_neutral.astype(np.float32) * float(obstacle_dim))
        col_obstacles = np.clip(col_obstacles, 0, 255).astype(np.uint8)
        col_obstacles[obs_mask] = np.array(OBSTACLE_RED, dtype=np.uint8)
    else:
        col_obstacles = col_neutral.copy()

    # ── État segmentation ───────────────────────────────────────────────────────
    labels = []
    n_segmented = 0
    if class_ids is not None and taxonomy is not None:
        class_ids = np.asarray(class_ids).astype(np.int32)
        col_seg = np.tile(np.array(NEUTRAL_GRAY, dtype=np.uint8), (N, 1))
        seg_mask = class_ids >= 0
        n_segmented = int(seg_mask.sum())
        if n_segmented:
            lut = taxonomy.color_lut(max_id=int(class_ids.max()))
            col_seg[seg_mask] = lut[class_ids[seg_mask]]
            # Labels : centroïde par classe ICANSII présente
            for cid in np.unique(class_ids[seg_mask]):
                pts = positions[class_ids == cid]
                if len(pts) < min_points_per_label:
                    continue
                labels.append({
                    "text": taxonomy.name(int(cid)),
                    "class": int(cid),
                    "position": [round(float(v), 4) for v in pts.mean(axis=0)],
                })
    else:
        col_seg = np.tile(np.array(NEUTRAL_GRAY, dtype=np.uint8), (N, 1))

    # ── Écriture ──────────────────────────────────────────────────────────────
    scene_dir = os.path.join(out_dir, "scene")
    os.makedirs(scene_dir, exist_ok=True)

    scene = {
        "meta": {
            "n_points": N,
            "n_points_before_budget": N0,
            "recentered": bool(recenter),
            "centroid": [round(float(v), 6) for v in centroid],
            "states": ["neutral", "obstacles", "segmentation"],
            "obstacle": {
                "angle_axis": list(map(float, angle_axis)),
                "angle_threshold_deg": float(angle_threshold_deg),
                "n_obstacle_points": n_obstacles,
            },
            "n_segmented_points": n_segmented,
        },
        "positions": np.round(positions, 4).astype(np.float32).flatten().tolist(),
        "colors": {
            "neutral":      col_neutral.flatten().tolist(),
            "obstacles":    col_obstacles.flatten().tolist(),
            "segmentation": col_seg.flatten().tolist(),
        },
        "labels": labels,
    }
    json_path = os.path.join(scene_dir, "scene.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(scene, f, separators=(",", ":"), ensure_ascii=False)

    # Fallback : 3 .ply au même ordre de sommets
    pos32 = positions.astype(np.float32)
    nrm32 = normals.astype(np.float32) if normals is not None else None
    _write_ply_binary(os.path.join(scene_dir, "neutral.ply"),      pos32, col_neutral,   nrm32)
    _write_ply_binary(os.path.join(scene_dir, "obstacles.ply"),    pos32, col_obstacles, nrm32)
    _write_ply_binary(os.path.join(scene_dir, "segmentation.ply"), pos32, col_seg,        nrm32)

    stats = {
        "n_points": N,
        "n_points_before_budget": N0,
        "n_obstacle_points": n_obstacles,
        "n_segmented_points": n_segmented,
        "n_labels": len(labels),
        "scene_json": json_path,
        "scene_json_mb": round(os.path.getsize(json_path) / 1e6, 2),
        "scene_dir": scene_dir,
    }
    return stats
