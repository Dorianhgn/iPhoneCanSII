"""
Tri-plane Probe: Outil d'analyse géométrique pour enregistrements Record3D (.npz).

Ce script projette une accumulation de nuages de points 3D sur trois plans orthogonaux (Top, Front, Side)
pour évaluer la densité de surface et la couverture spatiale.

Usage:
    python triplane_probe.py <recording.npz> [options]

Arguments:
    recording           : Chemin vers le fichier .npz (capturé via --save-npz dans record_reconstruct.py).

Options:
    --n-frames N        : Nombre de frames à accumuler autour de l'index de départ (défaut: 10).
    --start-index I     : Index de la frame pivot (défaut: 0).
    --res R             : Résolution de la grille des tri-planes (défaut: 128x128).
    --extent E          : Taille de la boîte englobante en mètres (défaut: 4.0m, centré sur la frame pivot).
    --past-mode         : Si activé, accumule de [start-index - N : start-index]. 
                          Sinon, accumule de [start-index : start-index + N] (futur).
    --skip-normals      : Désactive l'estimation des normales (accélère le rendu, enlève l'overlay rouge).
    --no-interactive    : Désactive le mode GUI interactif (affiche juste un snapshot statique).

Contrôles (Mode Interactif) :
    LEFT / RIGHT        : Index de départ -1 / +1.
    SHIFT + LEFT/RIGHT  : Index de départ -10 / +10.
    UP / DOWN           : Nombre de frames accumulées -1 / +1.
    SHIFT + UP/DOWN     : Nombre de frames accumulées -5 / +5.
    SPACE               : Alterne entre le mode Mono (Hauteur + Overlay Rouge) et le mode RGB.
"""
import argparse
import time
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

try:
    import yaml
except ImportError:
    yaml = None

from record_reconstruct import Record3DRecorder, pose_to_matrix


# ══════════════════════════════════════════════════════════════════════════════
#  HYPERPARAMÈTRES (VALEURS PAR DÉFAUT)
#  Peuvent être overridés via --config/-c (YAML type config_record_reconstruct).
# ══════════════════════════════════════════════════════════════════════════════

# ── Tri-plane / fenêtre temporelle ───────────────────────────────────────────
TRIPLANE_RES = 128
SPATIAL_EXTENT = 4.0  # m ; taille de la zone projetée tri-plane (indépendant de MAX_DEPTH).
N_FRAMES = 10
START_INDEX = 0

# ── Overlay de normales (compat view_ply) ────────────────────────────────────
ARROW_ANGLE_DEG_THRESHOLD = 30.0   # deg ; seuil angle pour classer "steep".
ARROW_ANGLE_AXIS = [0.0, 1.0, 0.0] # axe monde de référence (Y par défaut).
ARROW_STEP = 2                     # sous-échantillonnage des points pour normales.

# ── Filtrage / estimation des normales ───────────────────────────────────────
CONFIDENCE_MIN = 1                 # filtre confidence LiDAR mini (0/1/2).
NORMAL_VOXEL_SIZE = 0.05           # m ; voxel downsample avant estimation normales.
NORMAL_PROPAGATE_RADIUS = 0.1      # m ; rayon max de propagation au nuage complet.
NORMAL_KNN = 30                    # nb voisins max pour estimation normales.
NORMAL_RADIUS = 0.12               # m ; rayon de recherche pour estimation normales.
FRAME_SKIP = 1                     # stride temporel d'accumulation des frames.


def _coerce_float(value, key):
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{key} doit être un float (reçu: {value})")


def _coerce_int(value, key, min_value=None):
    try:
        out = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{key} doit être un int (reçu: {value})")

    if min_value is not None and out < min_value:
        raise ValueError(f"{key} doit être >= {min_value} (reçu: {out})")
    return out


def _coerce_axis3(value, key):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{key} doit être une liste de 3 valeurs")
    axis = np.asarray(value, dtype=np.float64)
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        raise ValueError(f"{key} ne peut pas être nul")
    return axis.tolist()


def load_hyperparams_from_yaml(config_path):
    """
    Charge des hyperparamètres depuis un YAML de reconstruction.

        Clés mappées principales :
      - VOXEL_SIZE -> NORMAL_VOXEL_SIZE
      - NORMAL_KNN -> NORMAL_KNN
      - NORMAL_RADIUS -> NORMAL_RADIUS
      - FRAME_SKIP -> FRAME_SKIP
      - CONFIDENCE_MIN -> CONFIDENCE_MIN
      - ARROW_ANGLE_DEG_THRESHOLD, ARROW_ANGLE_AXIS, ARROW_STEP

        Note: MAX_DEPTH (profondeur capteur) et SPATIAL_EXTENT (taille de la boîte
        d'affichage tri-plane) ne sont pas mappés automatiquement car ce ne sont
        pas la même grandeur.
    """
    if not config_path:
        return

    if yaml is None:
        print("⚠️  PyYAML non installé. --config ignoré (installe: pip install pyyaml)")
        return

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except FileNotFoundError:
        print(f"⚠️  Fichier config introuvable: {config_path} (defaults conservés)")
        return
    except Exception as e:
        print(f"⚠️  Erreur lecture YAML ({config_path}): {e} (defaults conservés)")
        return

    if not isinstance(data, dict):
        print(f"⚠️  YAML invalide (mapping attendu): {config_path} (defaults conservés)")
        return

    mapping = {
        "VOXEL_SIZE": "NORMAL_VOXEL_SIZE",
        "NORMAL_KNN": "NORMAL_KNN",
        "NORMAL_RADIUS": "NORMAL_RADIUS",
        "FRAME_SKIP": "FRAME_SKIP",
        "CONFIDENCE_MIN": "CONFIDENCE_MIN",
        "ARROW_ANGLE_DEG_THRESHOLD": "ARROW_ANGLE_DEG_THRESHOLD",
        "ARROW_ANGLE_AXIS": "ARROW_ANGLE_AXIS",
        "ARROW_STEP": "ARROW_STEP",
        "NORMAL_PROPAGATE_RADIUS": "NORMAL_PROPAGATE_RADIUS",
        # Accepte aussi des clés natives triplane si présentes dans le YAML.
        "TRIPLANE_RES": "TRIPLANE_RES",
        "SPATIAL_EXTENT": "SPATIAL_EXTENT",
        "N_FRAMES": "N_FRAMES",
        "START_INDEX": "START_INDEX",
        # Alias rétrocompat.
        "NORMAL_ANGLE_DEG": "ARROW_ANGLE_DEG_THRESHOLD",
    }

    if "MAX_DEPTH" in data and "SPATIAL_EXTENT" not in data:
        print(
            "ℹ️  MAX_DEPTH détecté dans la config mais non appliqué à SPATIAL_EXTENT "
            "(paramètres distincts)."
        )

    updated = []
    for src_key, dst_key in mapping.items():
        if src_key not in data:
            continue
        raw = data[src_key]
        try:
            if dst_key in {
                "SPATIAL_EXTENT",
                "NORMAL_VOXEL_SIZE",
                "NORMAL_RADIUS",
                "ARROW_ANGLE_DEG_THRESHOLD",
                "NORMAL_PROPAGATE_RADIUS",
            }:
                value = _coerce_float(raw, src_key)
            elif dst_key in {"TRIPLANE_RES", "N_FRAMES", "START_INDEX"}:
                value = _coerce_int(raw, src_key, min_value=0)
            elif dst_key in {"CONFIDENCE_MIN", "NORMAL_KNN", "FRAME_SKIP", "ARROW_STEP"}:
                value = _coerce_int(raw, src_key, min_value=1)
            elif dst_key == "ARROW_ANGLE_AXIS":
                value = _coerce_axis3(raw, src_key)
            else:
                value = raw
        except ValueError as e:
            print(f"⚠️  {e} -> clé ignorée")
            continue

        globals()[dst_key] = value
        updated.append((src_key, dst_key, value))

    if updated:
        print(f"✅ Config chargée depuis {config_path}")
        for src_key, dst_key, value in updated:
            print(f"   {src_key} -> {dst_key} = {value}")
    else:
        print(f"ℹ️  Aucune clé utile trouvée dans {config_path} (defaults conservés)")


def map_to_pixels(coords, half_extent, res):
    min_c = -half_extent
    max_c = half_extent
    span = max_c - min_c
    t = (coords - min_c) / span
    px = np.floor(t * res).astype(np.int32)
    valid = (px >= 0) & (px < res)
    return px, valid


def normalize_plane(values, occupied):
    out = np.zeros_like(values, dtype=np.float32)
    if not np.any(occupied):
        return out

    vals = values[occupied]
    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    if vmax - vmin < 1e-8:
        out[occupied] = 1.0
        return out

    out[occupied] = (values[occupied] - vmin) / (vmax - vmin)
    return out


def project_triplanes(points, colors, steep_mask, res, extent):
    half_extent = extent * 0.5
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    plane_xy_vals = np.full((res, res), -np.inf, dtype=np.float32)
    plane_xy_occ = np.zeros((res, res), dtype=bool)

    plane_yz_vals = np.full((res, res), np.inf, dtype=np.float32)
    plane_yz_occ = np.zeros((res, res), dtype=bool)

    plane_zx_vals = np.full((res, res), -np.inf, dtype=np.float32)
    plane_zx_occ = np.zeros((res, res), dtype=bool)

    overlay_xy = np.zeros((res, res), dtype=bool)
    overlay_yz = np.zeros((res, res), dtype=bool)
    overlay_zx = np.zeros((res, res), dtype=bool)

    plane_xy_rgb_sum = np.zeros((res, res, 3), dtype=np.float32)
    plane_xy_rgb_cnt = np.zeros((res, res), dtype=np.int32)

    plane_yz_rgb_sum = np.zeros((res, res, 3), dtype=np.float32)
    plane_yz_rgb_cnt = np.zeros((res, res), dtype=np.int32)

    plane_zx_rgb_sum = np.zeros((res, res, 3), dtype=np.float32)
    plane_zx_rgb_cnt = np.zeros((res, res), dtype=np.int32)

    px_x, ok_x = map_to_pixels(x, half_extent, res)
    px_y, ok_y = map_to_pixels(y, half_extent, res)
    px_z, ok_z = map_to_pixels(z, half_extent, res)

    valid_xy = ok_x & ok_y
    if np.any(valid_xy):
        ix = px_x[valid_xy]
        iy = px_y[valid_xy]
        vz = z[valid_xy]
        c_xy = colors[valid_xy]
        np.maximum.at(plane_xy_vals, (iy, ix), vz)
        plane_xy_occ[iy, ix] = True
        np.add.at(plane_xy_rgb_sum, (iy, ix), c_xy)
        np.add.at(plane_xy_rgb_cnt, (iy, ix), 1)

        steep_xy = valid_xy & steep_mask
        if np.any(steep_xy):
            overlay_xy[px_y[steep_xy], px_x[steep_xy]] = True

    valid_yz = ok_y & ok_z
    if np.any(valid_yz):
        iy = px_y[valid_yz]
        iz = px_z[valid_yz]
        vx = x[valid_yz]
        c_yz = colors[valid_yz]
        np.minimum.at(plane_yz_vals, (iz, iy), vx)
        plane_yz_occ[iz, iy] = True
        np.add.at(plane_yz_rgb_sum, (iz, iy), c_yz)
        np.add.at(plane_yz_rgb_cnt, (iz, iy), 1)

        steep_yz = valid_yz & steep_mask
        if np.any(steep_yz):
            overlay_yz[px_z[steep_yz], px_y[steep_yz]] = True

    valid_zx = ok_z & ok_x
    if np.any(valid_zx):
        iz = px_z[valid_zx]
        ix = px_x[valid_zx]
        vy = y[valid_zx]
        c_zx = colors[valid_zx]
        np.maximum.at(plane_zx_vals, (ix, iz), vy)
        plane_zx_occ[ix, iz] = True
        np.add.at(plane_zx_rgb_sum, (ix, iz), c_zx)
        np.add.at(plane_zx_rgb_cnt, (ix, iz), 1)

        steep_zx = valid_zx & steep_mask
        if np.any(steep_zx):
            overlay_zx[px_x[steep_zx], px_z[steep_zx]] = True

    plane_xy = normalize_plane(plane_xy_vals, plane_xy_occ)
    plane_yz = normalize_plane(plane_yz_vals, plane_yz_occ)
    plane_zx = normalize_plane(plane_zx_vals, plane_zx_occ)

    plane_xy_rgb = np.zeros((res, res, 3), dtype=np.float32)
    plane_yz_rgb = np.zeros((res, res, 3), dtype=np.float32)
    plane_zx_rgb = np.zeros((res, res, 3), dtype=np.float32)

    mask_xy_rgb = plane_xy_rgb_cnt > 0
    mask_yz_rgb = plane_yz_rgb_cnt > 0
    mask_zx_rgb = plane_zx_rgb_cnt > 0

    plane_xy_rgb[mask_xy_rgb] = (
        plane_xy_rgb_sum[mask_xy_rgb]
        / plane_xy_rgb_cnt[mask_xy_rgb, None].astype(np.float32)
    )
    plane_yz_rgb[mask_yz_rgb] = (
        plane_yz_rgb_sum[mask_yz_rgb]
        / plane_yz_rgb_cnt[mask_yz_rgb, None].astype(np.float32)
    )
    plane_zx_rgb[mask_zx_rgb] = (
        plane_zx_rgb_sum[mask_zx_rgb]
        / plane_zx_rgb_cnt[mask_zx_rgb, None].astype(np.float32)
    )

    return {
        "plane_front": plane_xy,
        "plane_side": plane_yz,
        "plane_top": plane_zx,
        "plane_front_rgb": np.clip(plane_xy_rgb, 0.0, 1.0),
        "plane_side_rgb": np.clip(plane_yz_rgb, 0.0, 1.0),
        "plane_top_rgb": np.clip(plane_zx_rgb, 0.0, 1.0),
        "occ_front": plane_xy_occ,
        "occ_side": plane_yz_occ,
        "occ_top": plane_zx_occ,
        "overlay_front": overlay_xy,
        "overlay_side": overlay_yz,
        "overlay_top": overlay_zx,
    }


def render_with_overlay(base_plane, overlay):
    rgb = np.repeat(base_plane[:, :, None], 3, axis=2)
    rgb[overlay] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return rgb


def coverage_percent(occupied):
    return 100.0 * float(np.count_nonzero(occupied)) / float(occupied.size)


def compute_plane_metrics(result):
    metrics = {}
    for suffix in ("front", "side", "top"):
        occ = result[f"occ_{suffix}"]
        count = int(np.count_nonzero(occ))
        cov = coverage_percent(occ)
        metrics[suffix] = (count, cov)
    return metrics


def resolve_window_indices(total_frames, start_index, n_frames, past_mode=False):
    """Return a safe list of frame indices for the requested temporal window."""
    if total_frames <= 0:
        return []

    start_index = int(np.clip(start_index, 0, total_frames - 1))
    n_frames = max(1, int(n_frames))
    frame_skip = max(1, int(FRAME_SKIP))

    if past_mode:
        # Prend n_frames échantillons vers le passé, en incluant start_index.
        begin = max(0, start_index - (n_frames - 1) * frame_skip)
        idx_desc = list(range(start_index, begin - 1, -frame_skip))
        return idx_desc[::-1]  # ordre chronologique (ancien -> récent)

    # Prend n_frames échantillons vers le futur, en incluant start_index.
    end = min(total_frames - 1, start_index + (n_frames - 1) * frame_skip)
    return list(range(start_index, end + 1, frame_skip))


def frame_points_in_frame0(frame, t0_inv):
    depth = frame["depth"]
    rgb = frame["rgb"]
    k = frame["intrinsic"]
    confidence = frame["confidence"]

    h, w = depth.shape
    yv, xv = np.indices((h, w))

    z = depth.reshape(-1)
    xv = xv.reshape(-1)
    yv = yv.reshape(-1)

    valid = (z > 0.0) & np.isfinite(z)
    if confidence is not None:
        conf = confidence.reshape(-1)
        valid &= conf >= CONFIDENCE_MIN

    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)

    z = z[valid]
    xv = xv[valid]
    yv = yv[valid]
    colors = rgb[yv, xv].astype(np.float32) / 255.0

    fx = float(k[0, 0])
    cx = float(k[0, 2])
    fy = float(k[1, 1])
    cy = float(k[1, 2])

    # Match reconstruct_fast convention: ARKit camera frame (X right, Y up, Z toward viewer).
    local_pts = np.stack(
        [
            (xv - cx) * z / fx,
            -(yv - cy) * z / fy,
            -z,
        ],
        axis=-1,
    )

    p = frame["pose"]
    t_cam_world = pose_to_matrix(
        p["qx"],
        p["qy"],
        p["qz"],
        p["qw"],
        p["tx"],
        p["ty"],
        p["tz"],
    )

    world_pts = (t_cam_world[:3, :3] @ local_pts.T).T + t_cam_world[:3, 3]

    world_h = np.concatenate(
        [world_pts, np.ones((world_pts.shape[0], 1), dtype=np.float64)], axis=1
    )
    pts_frame0 = (t0_inv @ world_h.T).T[:, :3]
    return pts_frame0.astype(np.float32), colors.astype(np.float32)


def accumulate_points(frames, n_frames, start_index=0, past_mode=False):
    frame_indices = resolve_window_indices(
        total_frames=len(frames),
        start_index=start_index,
        n_frames=n_frames,
        past_mode=past_mode,
    )

    if not frame_indices:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
        )

    anchor_idx = frame_indices[-1] if past_mode else frame_indices[0]
    p0 = frames[anchor_idx]["pose"]
    t0 = pose_to_matrix(
        p0["qx"], p0["qy"], p0["qz"], p0["qw"], p0["tx"], p0["ty"], p0["tz"]
    )
    t0_inv = np.linalg.inv(t0)

    axis_world = np.asarray(ARROW_ANGLE_AXIS, dtype=np.float64)
    axis_world /= np.linalg.norm(axis_world) + 1e-12
    axis_frame0 = t0_inv[:3, :3] @ axis_world
    axis_frame0 /= np.linalg.norm(axis_frame0) + 1e-12

    chunks = []
    color_chunks = []
    for i in frame_indices:
        pts, cols = frame_points_in_frame0(frames[i], t0_inv)
        if pts.shape[0] > 0:
            chunks.append(pts)
            color_chunks.append(cols)

    if not chunks:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
            axis_frame0.astype(np.float32),
        )

    return np.vstack(chunks), np.vstack(color_chunks), axis_frame0.astype(np.float32)


def compute_steep_mask(points, vertical_axis, skip_normals=False):
    t0 = time.perf_counter()

    if skip_normals or points.shape[0] < 8:
        return np.zeros(points.shape[0], dtype=bool), (time.perf_counter() - t0) * 1000.0

    sample_step = max(1, int(ARROW_STEP))
    sampled_points = points[::sample_step] if sample_step > 1 else points

    pcd_full = o3d.geometry.PointCloud()
    pcd_full.points = o3d.utility.Vector3dVector(sampled_points.astype(np.float64))

    pcd_ds = pcd_full.voxel_down_sample(NORMAL_VOXEL_SIZE)
    ds_points = np.asarray(pcd_ds.points)
    if ds_points.shape[0] == 0:
        return np.zeros(points.shape[0], dtype=bool), (time.perf_counter() - t0) * 1000.0

    pcd_ds.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=NORMAL_RADIUS,
            max_nn=max(3, int(NORMAL_KNN)),
        )
    )
    ds_normals = np.asarray(pcd_ds.normals)
    if ds_normals.shape[0] == 0:
        return np.zeros(points.shape[0], dtype=bool), (time.perf_counter() - t0) * 1000.0

    vertical = vertical_axis.astype(np.float64)
    vertical /= np.linalg.norm(vertical) + 1e-12

    cos_abs_ds = np.abs(ds_normals @ vertical)
    threshold = np.cos(np.deg2rad(ARROW_ANGLE_DEG_THRESHOLD))
    steep_ds = cos_abs_ds < threshold

    steep_full = np.zeros(points.shape[0], dtype=bool)
    max_radius2 = NORMAL_PROPAGATE_RADIUS * NORMAL_PROPAGATE_RADIUS

    # Fast vectorized nearest-neighbor propagation via Open3D tensor NNS.
    try:
        ds_tensor = o3d.core.Tensor(ds_points.astype(np.float32))
        q_tensor = o3d.core.Tensor(points.astype(np.float32))
        nns = o3d.core.nns.NearestNeighborSearch(ds_tensor)
        nns.knn_index()
        indices, dists2 = nns.knn_search(q_tensor, 1)
        nn_idx = indices.numpy().reshape(-1)
        nn_dist2 = dists2.numpy().reshape(-1)
        valid = np.isfinite(nn_dist2) & (nn_dist2 <= max_radius2)
        steep_full[valid] = steep_ds[nn_idx[valid]]
    except Exception:
        # Fallback when tensor NNS is unavailable.
        kdtree = o3d.geometry.KDTreeFlann(pcd_ds)
        for i, p in enumerate(points):
            k, idx, d2 = kdtree.search_knn_vector_3d(p.astype(np.float64), 1)
            if k > 0 and d2[0] <= max_radius2:
                steep_full[i] = steep_ds[idx[0]]

    return steep_full, (time.perf_counter() - t0) * 1000.0


def draw_triplanes(axes, result, res, render_mode="mono_red"):
    ordered = [
        ("plane_top", "occ_top", "overlay_top", "Top View (ZX)"),
        ("plane_front", "occ_front", "overlay_front", "Front View (XY)"),
        ("plane_side", "occ_side", "overlay_side", "Side View (YZ)"),
    ]

    for ax, (plane_key, occ_key, overlay_key, name) in zip(axes, ordered):
        plane = result[plane_key]
        occ = result[occ_key]
        overlay = result[overlay_key]
        cov = coverage_percent(occ)
        count = int(np.count_nonzero(occ))

        if render_mode == "rgb":
            rgb = result[f"{plane_key}_rgb"]
        else:
            rgb = render_with_overlay(plane, overlay)
        ax.clear()
        ax.imshow(rgb, origin="lower", interpolation="nearest")
        ax.set_title(f"{name}\nnon-zero: {count}/{res*res} ({cov:.1f}%)")
        ax.axis("off")


def show_triplanes(result, res):
    _, axes = plt.subplots(1, 3, figsize=(16, 5))
    draw_triplanes(axes, result, res, render_mode="mono_red")
    plt.tight_layout()


def density_series(frames, max_n, res, extent, start_index=0, past_mode=False):
    if start_index < 0:
        start_index = 0
    if start_index >= len(frames):
        return [], []

    frame_skip = max(1, int(FRAME_SKIP))
    if past_mode:
        available = (start_index // frame_skip) + 1
    else:
        available = ((len(frames) - 1 - start_index) // frame_skip) + 1

    nmax = min(max_n, available)
    xs = list(range(1, nmax + 1))
    ys = []

    for n in xs:
        pts, cols, vertical = accumulate_points(
            frames,
            n,
            start_index=start_index,
            past_mode=past_mode,
        )
        if pts.shape[0] == 0:
            ys.append(0.0)
            continue

        # Density does not depend on normal-based overlay.
        steep = np.zeros(pts.shape[0], dtype=bool)
        tri = project_triplanes(pts, cols, steep, res=res, extent=extent)

        cov_front = coverage_percent(tri["occ_front"])
        cov_side = coverage_percent(tri["occ_side"])
        cov_top = coverage_percent(tri["occ_top"])
        ys.append((cov_front + cov_side + cov_top) / 3.0)

    return xs, ys


def draw_density_curve(ax, xs, ys, n_frames, start_index, past_mode=False):
    ax.clear()
    ax.plot(xs, ys, marker="o", linewidth=2)
    ax.set_xlabel("N_FRAMES accumulees")
    ax.set_ylabel("Densite pixels non-nuls (%)")
    mode = "past" if past_mode else "future"
    ax.set_title(
        f"Densite moyenne des Tri-planes vs N_FRAMES (start={start_index}, window={n_frames}, mode={mode})"
    )
    ax.grid(True, alpha=0.3)


def density_curve(frames, max_n, res, extent, start_index=0, past_mode=False):
    xs, ys = density_series(
        frames,
        max_n,
        res,
        extent,
        start_index=start_index,
        past_mode=past_mode,
    )

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    draw_density_curve(ax, xs, ys, max_n, start_index, past_mode=past_mode)
    plt.tight_layout()


def compute_probe(
    frames,
    start_index,
    n_frames,
    res,
    extent,
    skip_normals=False,
    past_mode=False,
):
    points, colors, vertical = accumulate_points(
        frames,
        n_frames,
        start_index=start_index,
        past_mode=past_mode,
    )
    if points.shape[0] == 0:
        return None

    steep_mask, normals_ms = compute_steep_mask(
        points,
        vertical,
        skip_normals=skip_normals,
    )
    t_scatter0 = time.perf_counter()
    tri = project_triplanes(points, colors, steep_mask, res=res, extent=extent)
    scatter_ms = (time.perf_counter() - t_scatter0) * 1000.0
    return {
        "tri": tri,
        "metrics": compute_plane_metrics(tri),
        "n_points": int(points.shape[0]),
        "timing_ms": {
            "scatter": scatter_ms,
            "normals": normals_ms,
        },
    }


def interactive_viewer(
    frames,
    start_index,
    n_frames,
    res,
    extent,
    skip_normals=False,
    past_mode=False,
):
    state = {
        "start": max(0, min(start_index, len(frames) - 1)),
        "n": max(1, n_frames),
        "render_mode": "mono_red",
    }

    fig_planes, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig_curve, ax_curve = plt.subplots(1, 1, figsize=(8, 5))

    def redraw():
        t_plot0 = time.perf_counter()
        probe = compute_probe(
            frames,
            state["start"],
            state["n"],
            res,
            extent,
            skip_normals=skip_normals,
            past_mode=past_mode,
        )
        if probe is None:
            for ax in axes:
                ax.clear()
                ax.text(0.5, 0.5, "No valid points", ha="center", va="center")
                ax.axis("off")
            fig_planes.suptitle(
                f"start={state['start']} n={state['n']}  |  No valid points"
            )
        else:
            draw_triplanes(axes, probe["tri"], res, render_mode=state["render_mode"])
            end = min(state["start"] + state["n"] - 1, len(frames) - 1)
            fig_planes.suptitle(
                f"Frames [{state['start']}..{end}] / {len(frames)-1}  |  "
                f"points={probe['n_points']}  |  mode={state['render_mode']}"
            )

        xs, ys = density_series(
            frames,
            state["n"],
            res,
            extent,
            start_index=state["start"],
            past_mode=past_mode,
        )
        draw_density_curve(
            ax_curve,
            xs,
            ys,
            state["n"],
            state["start"],
            past_mode=past_mode,
        )
        fig_curve.suptitle(
            "LEFT/RIGHT: start +-1 | SHIFT+LEFT/RIGHT: start +-10 | "
            "UP/DOWN: N +-1 | SHIFT+UP/DOWN: N +-5 | SPACE: color mode"
        )

        fig_planes.canvas.draw_idle()
        fig_curve.canvas.draw_idle()
        plot_ms = (time.perf_counter() - t_plot0) * 1000.0

        scatter_ms = probe["timing_ms"]["scatter"] if probe is not None else 0.0
        normals_ms = probe["timing_ms"]["normals"] if probe is not None else 0.0
        print(
            f"[timing] scatter={scatter_ms:.1f} ms | normals={normals_ms:.1f} ms | plot={plot_ms:.1f} ms"
        )

    def on_key(event):
        key = event.key
        max_start = max(0, len(frames) - 1)

        if key == " ":
            state["render_mode"] = "rgb" if state["render_mode"] == "mono_red" else "mono_red"
            redraw()
            return

        if key == "shift+right":
            state["start"] = min(state["start"] + 10, max_start)
            redraw()
            return
        if key == "shift+left":
            state["start"] = max(state["start"] - 10, 0)
            redraw()
            return
        if key == "shift+up":
            state["n"] = min(state["n"] + 5, len(frames))
            redraw()
            return
        if key == "shift+down":
            state["n"] = max(state["n"] - 5, 1)
            redraw()
            return

        if key == "right":
            state["start"] = min(state["start"] + 1, max_start)
            redraw()
        elif key == "left":
            state["start"] = max(state["start"] - 1, 0)
            redraw()
        elif key == "up":
            state["n"] = min(state["n"] + 1, len(frames))
            redraw()
        elif key == "down":
            state["n"] = max(state["n"] - 1, 1)
            redraw()

    fig_planes.canvas.mpl_connect("key_press_event", on_key)
    fig_curve.canvas.mpl_connect("key_press_event", on_key)

    redraw()
    plt.show()


def main():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        help="Chemin vers un YAML de config (ex: config_record_reconstruct.yaml)",
    )
    pre_args, _ = pre_parser.parse_known_args()
    load_hyperparams_from_yaml(pre_args.config)

    parser = argparse.ArgumentParser(
        description="Probe tri-plane from a Record3D .npz recording.",
        parents=[pre_parser],
    )
    parser.add_argument("recording", help="Path to .npz recording")
    parser.add_argument(
        "--n-frames",
        type=int,
        default=N_FRAMES,
        help=f"Number of frames to accumulate (default: {N_FRAMES})",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=START_INDEX,
        help=f"First frame index for accumulation window (default: {START_INDEX})",
    )
    parser.add_argument(
        "--res",
        type=int,
        default=TRIPLANE_RES,
        help=f"Tri-plane resolution (default: {TRIPLANE_RES})",
    )
    parser.add_argument(
        "--extent",
        type=float,
        default=SPATIAL_EXTENT,
        help=f"Spatial extent in meters, centered at origin (default: {SPATIAL_EXTENT})",
    )
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Disable keyboard interactive viewer (left/right/up/down).",
    )
    parser.add_argument(
        "--skip-normals",
        action="store_true",
        default=False,
        help="Disable normal estimation and red overlay for faster iteration.",
    )
    parser.add_argument(
        "--past-mode",
        action="store_true",
        default=False,
        help="Use past accumulation window [start_index - n_frames : start_index + 1] anchored at present.",
    )
    args = parser.parse_args()

    frames = Record3DRecorder.load_raw_recording(args.recording)
    if len(frames) == 0:
        print("No frame in recording.")
        return

    max_start = max(0, len(frames) - 1)
    start_index = int(np.clip(args.start_index, 0, max_start))
    frame_skip = max(1, int(FRAME_SKIP))
    if args.past_mode:
        n_available = (start_index // frame_skip) + 1
    else:
        n_available = ((len(frames) - 1 - start_index) // frame_skip) + 1
    n_used = min(args.n_frames, n_available)
    if n_used < args.n_frames:
        print(
            f"Requested {args.n_frames} frames from start={start_index}, "
            f"only {n_used} available in range (past_mode={args.past_mode})."
        )

    if not args.no_interactive:
        interactive_viewer(
            frames,
            start_index=start_index,
            n_frames=max(1, n_used),
            res=args.res,
            extent=args.extent,
            skip_normals=args.skip_normals,
            past_mode=args.past_mode,
        )
        return

    t_plot0 = time.perf_counter()
    probe = compute_probe(
        frames,
        start_index=start_index,
        n_frames=max(1, n_used),
        res=args.res,
        extent=args.extent,
        skip_normals=args.skip_normals,
        past_mode=args.past_mode,
    )
    if probe is None:
        print("No valid 3D points extracted from selected frames.")
        return

    show_triplanes(probe["tri"], args.res)
    density_curve(
        frames,
        n_used,
        args.res,
        args.extent,
        start_index=start_index,
        past_mode=args.past_mode,
    )
    plot_ms = (time.perf_counter() - t_plot0) * 1000.0
    print(
        f"[timing] scatter={probe['timing_ms']['scatter']:.1f} ms | "
        f"normals={probe['timing_ms']['normals']:.1f} ms | "
        f"plot={plot_ms:.1f} ms"
    )
    plt.show()


if __name__ == "__main__":
    main()
