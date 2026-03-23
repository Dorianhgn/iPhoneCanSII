import argparse
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from record_reconstruct import Record3DRecorder, pose_to_matrix


TRIPLANE_RES = 128
SPATIAL_EXTENT = 4.0
N_FRAMES = 10
START_INDEX = 0

NORMAL_ANGLE_DEG = 30.0
CONFIDENCE_MIN = 1


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
        "plane_XY": plane_xy,
        "plane_YZ": plane_yz,
        "plane_ZX": plane_zx,
        "plane_XY_rgb": np.clip(plane_xy_rgb, 0.0, 1.0),
        "plane_YZ_rgb": np.clip(plane_yz_rgb, 0.0, 1.0),
        "plane_ZX_rgb": np.clip(plane_zx_rgb, 0.0, 1.0),
        "occ_XY": plane_xy_occ,
        "occ_YZ": plane_yz_occ,
        "occ_ZX": plane_zx_occ,
        "overlay_XY": overlay_xy,
        "overlay_YZ": overlay_yz,
        "overlay_ZX": overlay_zx,
    }


def render_with_overlay(base_plane, overlay):
    rgb = np.repeat(base_plane[:, :, None], 3, axis=2)
    rgb[overlay] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return rgb


def coverage_percent(occupied):
    return 100.0 * float(np.count_nonzero(occupied)) / float(occupied.size)


def compute_plane_metrics(result):
    metrics = {}
    for suffix in ("XY", "YZ", "ZX"):
        occ = result[f"occ_{suffix}"]
        count = int(np.count_nonzero(occ))
        cov = coverage_percent(occ)
        metrics[suffix] = (count, cov)
    return metrics


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


def accumulate_points(frames, n_frames, start_index=0):
    if start_index < 0:
        start_index = 0
    if start_index >= len(frames):
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
        )

    n = min(n_frames, len(frames) - start_index)
    if n == 0:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
        )

    p0 = frames[start_index]["pose"]
    t0 = pose_to_matrix(
        p0["qx"], p0["qy"], p0["qz"], p0["qw"], p0["tx"], p0["ty"], p0["tz"]
    )
    t0_inv = np.linalg.inv(t0)

    up_world = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    up_frame0 = t0_inv[:3, :3] @ up_world
    up_frame0 /= np.linalg.norm(up_frame0) + 1e-12

    chunks = []
    color_chunks = []
    for i in range(start_index, start_index + n):
        pts, cols = frame_points_in_frame0(frames[i], t0_inv)
        if pts.shape[0] > 0:
            chunks.append(pts)
            color_chunks.append(cols)

    if not chunks:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
            up_frame0.astype(np.float32),
        )

    return np.vstack(chunks), np.vstack(color_chunks), up_frame0.astype(np.float32)


def compute_steep_mask(points, vertical_axis):
    if points.shape[0] < 8:
        return np.zeros(points.shape[0], dtype=bool)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.08, max_nn=30)
    )
    normals = np.asarray(pcd.normals)
    if normals.shape[0] == 0:
        return np.zeros(points.shape[0], dtype=bool)

    vertical = vertical_axis.astype(np.float64)
    vertical /= np.linalg.norm(vertical) + 1e-12

    cos_abs = np.abs(normals @ vertical)
    threshold = np.cos(np.deg2rad(NORMAL_ANGLE_DEG))
    return cos_abs < threshold


def draw_triplanes(axes, result, res, render_mode="mono_red"):
    ordered = [
        ("plane_XY", "occ_XY", "overlay_XY", "plane_XY (Top: Z max)"),
        ("plane_YZ", "occ_YZ", "overlay_YZ", "plane_YZ (Side: X min)"),
        ("plane_ZX", "occ_ZX", "overlay_ZX", "plane_ZX (Front: Y max)"),
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


def density_series(frames, max_n, res, extent, start_index=0):
    if start_index < 0:
        start_index = 0
    if start_index >= len(frames):
        return [], []

    nmax = min(max_n, len(frames) - start_index)
    xs = list(range(1, nmax + 1))
    ys = []

    for n in xs:
        pts, cols, vertical = accumulate_points(frames, n, start_index=start_index)
        if pts.shape[0] == 0:
            ys.append(0.0)
            continue

        steep = compute_steep_mask(pts, vertical)
        tri = project_triplanes(pts, cols, steep, res=res, extent=extent)

        cov_xy = coverage_percent(tri["occ_XY"])
        cov_yz = coverage_percent(tri["occ_YZ"])
        cov_zx = coverage_percent(tri["occ_ZX"])
        ys.append((cov_xy + cov_yz + cov_zx) / 3.0)

    return xs, ys


def draw_density_curve(ax, xs, ys, n_frames, start_index):
    ax.clear()
    ax.plot(xs, ys, marker="o", linewidth=2)
    ax.set_xlabel("N_FRAMES accumulees")
    ax.set_ylabel("Densite pixels non-nuls (%)")
    ax.set_title(
        f"Densite moyenne des Tri-planes vs N_FRAMES (start={start_index}, window={n_frames})"
    )
    ax.grid(True, alpha=0.3)


def density_curve(frames, max_n, res, extent, start_index=0):
    xs, ys = density_series(frames, max_n, res, extent, start_index=start_index)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    draw_density_curve(ax, xs, ys, max_n, start_index)
    plt.tight_layout()


def compute_probe(frames, start_index, n_frames, res, extent):
    points, colors, vertical = accumulate_points(frames, n_frames, start_index=start_index)
    if points.shape[0] == 0:
        return None

    steep_mask = compute_steep_mask(points, vertical)
    tri = project_triplanes(points, colors, steep_mask, res=res, extent=extent)
    return {
        "tri": tri,
        "metrics": compute_plane_metrics(tri),
        "n_points": int(points.shape[0]),
    }


def interactive_viewer(frames, start_index, n_frames, res, extent):
    state = {
        "start": max(0, min(start_index, len(frames) - 1)),
        "n": max(1, n_frames),
        "render_mode": "mono_red",
    }

    fig_planes, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig_curve, ax_curve = plt.subplots(1, 1, figsize=(8, 5))

    def redraw():
        probe = compute_probe(frames, state["start"], state["n"], res, extent)
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

        xs, ys = density_series(frames, state["n"], res, extent, start_index=state["start"])
        draw_density_curve(ax_curve, xs, ys, state["n"], state["start"])
        fig_curve.suptitle(
            "LEFT/RIGHT: start +-1 | SHIFT+LEFT/RIGHT: start +-10 | "
            "UP/DOWN: N +-1 | SHIFT+UP/DOWN: N +-5 | SPACE: color mode"
        )

        fig_planes.canvas.draw_idle()
        fig_curve.canvas.draw_idle()

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
    parser = argparse.ArgumentParser(
        description="Probe tri-plane from a Record3D .npz recording."
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
    args = parser.parse_args()

    frames = Record3DRecorder.load_raw_recording(args.recording)
    if len(frames) == 0:
        print("No frame in recording.")
        return

    max_start = max(0, len(frames) - 1)
    start_index = int(np.clip(args.start_index, 0, max_start))
    n_used = min(args.n_frames, len(frames) - start_index)
    if n_used < args.n_frames:
        print(
            f"Requested {args.n_frames} frames from start={start_index}, "
            f"only {n_used} available in range."
        )

    if not args.no_interactive:
        interactive_viewer(
            frames,
            start_index=start_index,
            n_frames=max(1, n_used),
            res=args.res,
            extent=args.extent,
        )
        return

    probe = compute_probe(
        frames,
        start_index=start_index,
        n_frames=max(1, n_used),
        res=args.res,
        extent=args.extent,
    )
    if probe is None:
        print("No valid 3D points extracted from selected frames.")
        return

    show_triplanes(probe["tri"], args.res)
    density_curve(frames, n_used, args.res, args.extent, start_index=start_index)
    plt.show()


if __name__ == "__main__":
    main()
