"""
Record3D – Enregistrement vidéo + Reconstruction 3D volumétrique (TSDF)
------------------------------------------------------------------------
Ce script enregistre un flux RGBD + poses caméra via Record3D,
puis reconstruit un environnement 3D via TSDF (Truncated Signed Distance
Function), le même principe que KinectFusion et l'app Record3D elle-même.

Pourquoi TSDF et pas accumulation de points ?
  - Une grille volumétrique 3D est maintenue en mémoire
  - Chaque frame RGBD met à jour les voxels qu'elle voit (weighted average)
  - Les zones revisitées ne créent PAS de doublons : les voxels existants
    sont juste re-moyennés → fusion correcte par construction
  - Pas besoin d'ICP a posteriori

Prérequis :
    pip install record3d open3d opencv-python numpy scipy

Usage :
    1. Branche l'iPhone en USB
    2. Record3D app → Settings → "USB Streaming mode" activé
    3. Lance : python record_reconstruct.py
    (optionnel) : python record_reconstruct.py --save-npz
    4. Appuie sur ⏺ dans l'app pour démarrer le flux

Contrôles :
    [ESPACE]  → démarrer / arrêter l'enregistrement
    [Q / ESC] → quitter

Sorties (dans logs/<datetime>/) :
    - reconstructed.ply   : nuage de points avec normales
    - config.txt          : hyperparamètres utilisés
    - performance.log     : métriques de performance
    - scan.npz            : recording brut (si --save-npz activé)
"""

import argparse
import threading
import time
import os
from datetime import datetime
import numpy as np
import cv2
import open3d as o3d
from scipy.spatial.transform import Rotation
from record3d import Record3DStream


# ══════════════════════════════════════════════════════════════════════════════
#  HYPERPARAMÈTRES
# ══════════════════════════════════════════════════════════════════════════════

# ── Mode de reconstruction ────────────────────────────────────────────────────
USE_TSDF  = False    # True  → TSDF volumétrique (qualité, sans doublons)
                    # False → accumulation rapide + voxel downsample (quasi temps-réel)
USE_GPU   = False   # True  → TSDF GPU via Open3D tensor API (CUDA sur Jetson)
                    # False → TSDF CPU (ScalableTSDFVolume)
                    # ⚠  USE_GPU=True nécessite open3d-cuda (pip install open3d-cuda)
                    #    et un device CUDA (Jetson, GPU NVIDIA) ou Metal (mac M-series)
                    #    Sur mac M4 : USE_GPU=False pour l'instant (pas de support CUDA)
                    #    Sur Jetson  : USE_GPU=True avec CUDA:0

# ── Enregistrement ────────────────────────────────────────────────────────────
FRAME_SKIP        = 3       # Ne garder qu'1 frame sur N pendant l'enregistrement.
                            # 1 = toutes (lourd), 3 = bon compromis, 5+ = léger
MIN_TRAVEL_DIST   = 0.01    # Distance min (m) entre 2 frames gardées.
                            # Évite de stocker quand la caméra est immobile.

# ── Mode rapide (USE_TSDF=False) ──────────────────────────────────────────────
SUBSAMPLE         = 2       # Sous-échantillonnage spatial du depth (1=dense, 2=moitié).
VOXEL_SIZE        = 0.01    # Taille du voxel de déduplication (m).
                            # 0.01 = 1cm absorbe le jitter ARKit (~2-5mm) → pas de doublons.
                            # Plus petit = plus de doublons si drift ARKit > voxel.

# ── TSDF (USE_TSDF=True) ─────────────────────────────────────────────────────
TSDF_VOXEL_LENGTH = 0.005   # Résolution de la grille TSDF (m). Paramètre qualité principal.
                            # 0.003=3mm (détaillé, lourd)  0.005=5mm (recommandé)
                            # 0.008=8mm (rapide)  0.01=1cm (léger)
TSDF_SDF_TRUNC    = 0.04    # Troncature SDF (m). Règle : 4-8× TSDF_VOXEL_LENGTH.
TSDF_BLOCK_COUNT  = 50000   # Nb blocs pré-alloués (GPU uniquement).
                            # 50000 ≈ pièce standard. Augmenter pour grandes scènes.

MAX_DEPTH         = 6.0     # Profondeur max (m). 2.0=bureau, 4.0=pièce, 5+=extérieur.
CONFIDENCE_MIN    = 1       # Confidence LiDAR minimum (0=low, 1=medium, 2=high).

# ── Filtrage post-reconstruction ─────────────────────────────────────────────
OUTLIER_NB        = 20      # Nb voisins pour statistical outlier removal.
OUTLIER_STD       = 2.0     # Seuil en écarts-types. Plus petit = plus agressif.

# ── Normales ──────────────────────────────────────────────────────────────────
NORMAL_KNN        = 30      # Nb voisins pour estimation PCA des normales.
NORMAL_RADIUS     = 0.05    # Rayon max (m) pour chercher les voisins.

# ── Preview ───────────────────────────────────────────────────────────────────
PREVIEW_W         = 1280    # Largeur max de la fenêtre preview OpenCV.


# ══════════════════════════════════════════════════════════════════════════════
#  Collecte pour logs
# ══════════════════════════════════════════════════════════════════════════════

HYPERPARAMS = {
    "USE_TSDF":          USE_TSDF,
    "USE_GPU":           USE_GPU,
    "FRAME_SKIP":        FRAME_SKIP,
    "MIN_TRAVEL_DIST":   MIN_TRAVEL_DIST,
    # Mode rapide
    "SUBSAMPLE":         SUBSAMPLE,
    "VOXEL_SIZE":        VOXEL_SIZE,
    # TSDF
    "TSDF_VOXEL_LENGTH": TSDF_VOXEL_LENGTH,
    "TSDF_SDF_TRUNC":    TSDF_SDF_TRUNC,
    "TSDF_BLOCK_COUNT":  TSDF_BLOCK_COUNT,
    # Commun
    "MAX_DEPTH":         MAX_DEPTH,
    "CONFIDENCE_MIN":    CONFIDENCE_MIN,
    "OUTLIER_NB":        OUTLIER_NB,
    "OUTLIER_STD":       OUTLIER_STD,
    "NORMAL_KNN":        NORMAL_KNN,
    "NORMAL_RADIUS":     NORMAL_RADIUS,
}


# ══════════════════════════════════════════════════════════════════════════════
#  UTILITAIRES
# ══════════════════════════════════════════════════════════════════════════════

def pose_to_matrix(qx, qy, qz, qw, tx, ty, tz):
    """
    Convertit un quaternion + translation (pose ARKit) en matrice 4×4 homogène.
    Représente la transformation camera → world.
    """
    R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [tx, ty, tz]
    return T


# ══════════════════════════════════════════════════════════════════════════════
#  CLASSE PRINCIPALE
# ══════════════════════════════════════════════════════════════════════════════

class Record3DRecorder:

    def __init__(self):
        self.session        = None
        self.stream_stopped = threading.Event()
        self.new_frame_evt  = threading.Event()

        # Dernière frame reçue (pour la preview)
        self.latest_rgb        = None
        self.latest_depth      = None
        self.latest_intrinsic  = None
        self.latest_confidence = None
        self.latest_pose       = None
        self._lock             = threading.Lock()

        # État d'enregistrement
        self.recording          = False
        self.recorded_frames    = []
        self.record_start_time  = None
        self.last_pose_position = None
        self.frame_counter      = 0

    # ── Utilitaires ───────────────────────────────────────────────────────────

    def get_intrinsic_mat_from_coeffs(self, coeffs, depth_w, depth_h, rgb_w, rgb_h):
        """Coefficients Record3D (repère RGB) → matrice intrinsèque au repère depth."""
        scale_x = depth_w / rgb_w
        scale_y = depth_h / rgb_h
        return np.array([
            [coeffs.fx * scale_x, 0,                   coeffs.tx * scale_x],
            [0,                   coeffs.fy * scale_y, coeffs.ty * scale_y],
            [0,                   0,                   1                   ]
        ])

    # ── Callbacks Record3D ────────────────────────────────────────────────────

    def on_new_frame(self):
        try:
            rgb    = self.session.get_rgb_frame()
            depth  = self.session.get_depth_frame()
            dH, dW = depth.shape[:2]
            rH, rW = rgb.shape[:2]
            coeffs = self.session.get_intrinsic_mat()
            intrinsic = self.get_intrinsic_mat_from_coeffs(coeffs, dW, dH, rW, rH)
            pose   = self.session.get_camera_pose()

            confidence = None
            if hasattr(self.session, "get_confidence_frame"):
                try:
                    confidence = self.session.get_confidence_frame()
                except Exception:
                    pass

            with self._lock:
                self.latest_rgb        = rgb
                self.latest_depth      = depth
                self.latest_intrinsic  = intrinsic
                self.latest_confidence = confidence
                self.latest_pose       = pose

            # ── Enregistrement ────────────────────────────────────────────
            if self.recording:
                self.frame_counter += 1

                # Frame skip
                if self.frame_counter % FRAME_SKIP != 0:
                    self.new_frame_evt.set()
                    return

                # Distance travel check (évite les doublons quand la caméra est immobile)
                current_pos = np.array([pose.tx, pose.ty, pose.tz])
                if self.last_pose_position is not None:
                    dist = np.linalg.norm(current_pos - self.last_pose_position)
                    if dist < MIN_TRAVEL_DIST:
                        self.new_frame_evt.set()
                        return

                self.last_pose_position = current_pos

                # Resize RGB vers la résolution depth (économie mémoire)
                rgb_resized = cv2.resize(rgb, (dW, dH), interpolation=cv2.INTER_LINEAR)

                self.recorded_frames.append({
                    'rgb':        rgb_resized.copy(),
                    'depth':      depth.copy(),
                    'intrinsic':  intrinsic.copy(),
                    'confidence': confidence.copy() if confidence is not None else None,
                    'pose': {
                        'qx': pose.qx, 'qy': pose.qy, 'qz': pose.qz, 'qw': pose.qw,
                        'tx': pose.tx, 'ty': pose.ty, 'tz': pose.tz,
                    },
                    'timestamp': time.time(),
                })

            self.new_frame_evt.set()

        except Exception as e:
            print(f"[on_new_frame] Erreur : {e}")

    def on_stream_stopped(self):
        print("Stream arrêté.")
        self.stream_stopped.set()
        self.new_frame_evt.set()

    # ── Enregistrement ────────────────────────────────────────────────────────

    def start_recording(self):
        self.recorded_frames    = []
        self.frame_counter      = 0
        self.last_pose_position = None
        self.record_start_time  = time.time()
        self.recording          = True
        print("🔴 Enregistrement démarré — bouge l'iPhone lentement.")

    def stop_recording(self):
        self.recording = False
        record_duration = time.time() - self.record_start_time
        n = len(self.recorded_frames)
        print(f"⏹  Enregistrement arrêté : {n} frames en {record_duration:.1f}s")
        return record_duration

    # ── Reconstruction rapide (accumulation + voxel) ─────────────────────────

    def reconstruct_fast(self, frames):
        """
        Mode rapide (USE_TSDF=False) : déprojection vectorisée + voxel downsample.
        Pas de doublons si VOXEL_SIZE ≥ jitter ARKit (~5-10mm).
        Temps typique : <1s pour 30 frames.
        """
        print(f"\n⚡ Reconstruction rapide de {len(frames)} frames...")
        print(f"   subsample={SUBSAMPLE}  voxel={VOXEL_SIZE*1000:.0f}mm  max_depth={MAX_DEPTH}m")
        t0 = time.time()

        all_points, all_colors, skipped = [], [], 0

        for frame in frames:
            depth      = frame['depth']
            rgb        = frame['rgb']
            K          = frame['intrinsic']
            confidence = frame['confidence']
            H, W = depth.shape

            ys = np.arange(0, H, SUBSAMPLE)
            xs = np.arange(0, W, SUBSAMPLE)
            xv, yv = np.meshgrid(xs, ys)
            xv, yv = xv.flatten(), yv.flatten()

            z    = depth[yv, xv]
            mask = (z > 0) & (z < MAX_DEPTH)
            if confidence is not None:
                mask &= (confidence[yv, xv] >= CONFIDENCE_MIN)

            xv, yv, z = xv[mask], yv[mask], z[mask]
            if len(z) == 0:
                skipped += 1
                continue

            fx, cx, fy, cy = K[0,0], K[0,2], K[1,1], K[1,2]
            # ARKit convention: X right, Y up, Z toward viewer
            # OpenCV/pinhole convention: X right, Y down, Z into scene
            # → flip Y and Z so local_pts match ARKit camera frame
            local_pts = np.stack([(xv-cx)*z/fx, -(yv-cy)*z/fy, -z], axis=-1)

            p = frame['pose']
            T = pose_to_matrix(p['qx'], p['qy'], p['qz'], p['qw'],
                               p['tx'], p['ty'], p['tz'])
            world_pts = (T[:3,:3] @ local_pts.T).T + T[:3,3]

            all_points.append(world_pts)
            all_colors.append(rgb[yv, xv].astype(np.float32) / 255.0)

        stats = {
            'mode': 'fast',
            'n_frames_total':   len(frames),
            'n_frames_skipped': skipped,
            'n_frames_used':    len(frames) - skipped,
        }

        if not all_points:
            print("❌ Aucun point valide.")
            stats.update({'n_raw_points': 0, 'n_final_points': 0, 't_reconstruct': 0})
            return None, stats

        all_points = np.vstack(all_points)
        all_colors = np.vstack(all_colors)
        n_raw = len(all_points)
        print(f"  Points bruts : {n_raw:,}")

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_points)
        pcd.colors = o3d.utility.Vector3dVector(all_colors)

        print(f"  Voxel downsample ({VOXEL_SIZE*1000:.0f}mm)...")
        pcd = pcd.voxel_down_sample(VOXEL_SIZE)
        n_voxel = len(pcd.points)
        print(f"  Après voxel : {n_voxel:,} points ({100*n_voxel/n_raw:.1f}%)")

        print(f"  Outlier removal...")
        pcd, _ = pcd.remove_statistical_outlier(OUTLIER_NB, OUTLIER_STD)
        n_final = len(pcd.points)
        print(f"  Final : {n_final:,} points")

        t_recon = time.time() - t0
        print(f"  ✅ Terminé en {t_recon:.2f}s")
        stats.update({'n_raw_points': n_raw, 'n_voxel_points': n_voxel,
                      'n_final_points': n_final, 't_reconstruct': t_recon})
        return pcd, stats

    # ── Reconstruction TSDF (CPU ou GPU) ──────────────────────────────────────

    def reconstruct(self, frames):
        """
        Reconstruit l'environnement 3D via TSDF volumétrique (ScalableTSDFVolume).

        Principe (identique à KinectFusion / Record3D) :
        - Un volume 3D scalable est initialisé une seule fois
        - Chaque frame RGBD est intégrée via la pose ARKit (camera → world)
        - Les zones revisitées RE-MOYENNENT les voxels existants
          → pas de doublon possible par construction, contrairement à
          l'accumulation de points
        - Extraction finale du nuage de points depuis la grille SDF

        Retourne (pcd, stats_dict) ou (None, stats_dict).
        """
        device_label = "GPU" if USE_GPU else "CPU"
        print(f"\n🔨 Reconstruction TSDF {device_label} de {len(frames)} frames...")
        print(f"   voxel={TSDF_VOXEL_LENGTH*1000:.0f}mm  sdf_trunc={TSDF_SDF_TRUNC*100:.0f}cm  max_depth={MAX_DEPTH}m")
        t0 = time.time()

        skipped      = 0
        n_integrated = 0

        if USE_GPU:
            # ── TSDF GPU : Open3D tensor API (VoxelBlockGrid) ─────────────────
            # Fonctionne avec CUDA (Jetson) ou Metal (Mac M-series si build avec support)
            # pip install open3d-cuda  sur Jetson
            try:
                import open3d.core as o3c
                # Détecte automatiquement CUDA ou Metal
                if o3c.cuda.is_available():
                    device = o3c.Device("CUDA:0")
                    print("  Device : CUDA:0")
                else:
                    device = o3c.Device("CPU:0")
                    print("  ⚠  CUDA non disponible → fallback CPU (installe open3d-cuda)")

                vbg = o3d.t.geometry.VoxelBlockGrid(
                    attr_names   = ('tsdf', 'weight', 'color'),
                    attr_dtypes  = (o3c.float32, o3c.float16, o3c.float16),
                    attr_channels= ((1), (1), (3)),
                    voxel_size   = TSDF_VOXEL_LENGTH,
                    block_resolution = 16,
                    block_count  = TSDF_BLOCK_COUNT,
                    device       = device,
                )

                for i, frame in enumerate(frames):
                    if (i + 1) % 50 == 0 or i == len(frames) - 1:
                        print(f"  Frame {i+1}/{len(frames)} | intégrées : {n_integrated}")

                    depth         = frame['depth'].copy()
                    rgb           = frame['rgb']
                    K             = frame['intrinsic']
                    confidence    = frame['confidence']
                    H, W          = depth.shape

                    if confidence is not None:
                        depth[confidence < CONFIDENCE_MIN] = 0.0

                    fx, cx = K[0,0], K[0,2]
                    fy, cy = K[1,1], K[1,2]

                    depth_t = o3c.Tensor(depth.astype(np.float32), device=device)
                    rgb_t   = o3c.Tensor(rgb.astype(np.uint8),   device=device)
                    intrinsic_t = o3c.Tensor(
                        np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float64)
                    )

                    p = frame['pose']
                    T_cam_to_world = pose_to_matrix(p['qx'], p['qy'], p['qz'], p['qw'],
                                                    p['tx'], p['ty'], p['tz'])
                    T_world_to_cam = np.linalg.inv(T_cam_to_world)
                    extrinsic_t = o3c.Tensor(T_world_to_cam.astype(np.float64))

                    try:
                        frustum_block_coords = vbg.compute_unique_block_coordinates(
                            depth_t, intrinsic_t, extrinsic_t,
                            depth_scale=1.0, depth_max=MAX_DEPTH
                        )
                        vbg.integrate(
                            frustum_block_coords,
                            depth_t, rgb_t,
                            intrinsic_t, intrinsic_t, extrinsic_t,
                            depth_scale=1.0, depth_max=MAX_DEPTH
                        )
                        n_integrated += 1
                    except Exception as e:
                        skipped += 1
                        if skipped <= 3:
                            print(f"  [WARN] Frame {i} ignorée : {e}")

                print("  Extraction depuis VoxelBlockGrid...")
                pcd_t = vbg.extract_point_cloud()
                pcd   = pcd_t.to_legacy()

            except ImportError:
                print("  ❌ open3d.core non disponible. Installe open3d-cuda ou repasse USE_GPU=False.")
                return None, {'mode': 'tsdf_gpu', 'n_frames_total': len(frames),
                              'n_frames_skipped': 0, 'n_frames_used': 0,
                              'n_raw_points': 0, 'n_final_points': 0, 't_reconstruct': 0}

        else:
            # ── TSDF CPU : ScalableTSDFVolume ─────────────────────────────────
            volume = o3d.pipelines.integration.ScalableTSDFVolume(
                voxel_length=TSDF_VOXEL_LENGTH,
                sdf_trunc=TSDF_SDF_TRUNC,
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
            )

            for i, frame in enumerate(frames):
                if (i + 1) % 50 == 0 or i == len(frames) - 1:
                    print(f"  Frame {i+1}/{len(frames)} | intégrées : {n_integrated}")

                depth         = frame['depth'].copy()
                rgb           = frame['rgb']
                K             = frame['intrinsic']
                confidence    = frame['confidence']
                H, W          = depth.shape

                if confidence is not None:
                    depth[confidence < CONFIDENCE_MIN] = 0.0

                fx, cx = K[0,0], K[0,2]
                fy, cy = K[1,1], K[1,2]
                intr  = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)
                rgbd  = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    o3d.geometry.Image(np.ascontiguousarray(rgb.astype(np.uint8))),
                    o3d.geometry.Image(np.ascontiguousarray(depth.astype(np.float32))),
                    depth_scale=1.0, depth_trunc=MAX_DEPTH,
                    convert_rgb_to_intensity=False,
                )

                p = frame['pose']
                T_cam_to_world = pose_to_matrix(p['qx'], p['qy'], p['qz'], p['qw'],
                                                p['tx'], p['ty'], p['tz'])
                # Open3D TSDF expects extrinsic in OpenCV convention (Y↓, Z forward).
                # ARKit pose is in ARKit convention (Y↑, Z toward viewer).
                # R_fix converts from OpenCV cam to ARKit cam (flip Y and Z).
                R_fix = np.diag([1.0, -1.0, -1.0, 1.0])
                T_world_to_cam = np.linalg.inv(T_cam_to_world @ R_fix)

                try:
                    volume.integrate(rgbd, intr, T_world_to_cam)
                    n_integrated += 1
                except Exception as e:
                    skipped += 1
                    if skipped <= 3:
                        print(f"  [WARN] Frame {i} ignorée : {e}")

            print("  Extraction depuis ScalableTSDFVolume...")
            pcd = volume.extract_point_cloud()

        # ── Stats + post-processing communs ───────────────────────────────────
        stats = {
            'mode':             f'tsdf_{"gpu" if USE_GPU else "cpu"}',
            'n_frames_total':   len(frames),
            'n_frames_skipped': skipped,
            'n_frames_used':    n_integrated,
        }

        if pcd is None or len(pcd.points) == 0:
            print("❌ Volume TSDF vide — vérifie les poses ARKit et la depth.")
            stats.update({'n_raw_points': 0, 'n_final_points': 0, 't_reconstruct': 0})
            return None, stats

        n_raw = len(pcd.points)
        print(f"  Points extraits : {n_raw:,}")

        print(f"  Outlier removal (nb={OUTLIER_NB}, std={OUTLIER_STD})...")
        pcd, _ = pcd.remove_statistical_outlier(OUTLIER_NB, OUTLIER_STD)
        n_final = len(pcd.points)
        print(f"  Final : {n_final:,} points")

        t_recon = time.time() - t0
        print(f"  ✅ Reconstruction terminée en {t_recon:.2f}s")
        stats.update({'n_raw_points': n_raw, 'n_final_points': n_final, 't_reconstruct': t_recon})
        return pcd, stats

    # ── Calcul des normales ───────────────────────────────────────────────────

    def compute_normals(self, pcd):
        """Estime les normales par PCA puis oriente vers l'origine."""
        print(f"  Calcul normales (knn={NORMAL_KNN}, radius={NORMAL_RADIUS})...")
        t0 = time.time()
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=NORMAL_RADIUS,
                max_nn=NORMAL_KNN
            )
        )
        pcd.orient_normals_towards_camera_location(np.array([0.0, 0.0, 0.0]))
        t_normals = time.time() - t0
        print(f"  ✅ Normales calculées en {t_normals:.2f}s")
        return pcd, t_normals

    # ── Sauvegarde des logs ───────────────────────────────────────────────────

    def save_logs(self, log_dir, record_duration, stats, t_normals):
        """Écrit config.txt et performance.log dans log_dir."""
        os.makedirs(log_dir, exist_ok=True)

        # config.txt — hyperparamètres
        with open(os.path.join(log_dir, "config.txt"), "w") as f:
            f.write("# Hyperparamètres de la reconstruction\n")
            f.write(f"# Généré le {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            for k, v in HYPERPARAMS.items():
                f.write(f"{k} = {v}\n")

        # performance.log — métriques
        t_recon  = stats.get('t_reconstruct', 0)
        t_total  = t_recon + t_normals
        with open(os.path.join(log_dir, "performance.log"), "w") as f:
            f.write(f"{'='*50}\n")
            f.write(f"  PERFORMANCE LOG\n")
            f.write(f"{'='*50}\n\n")
            f.write(f"Date                    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"--- Enregistrement ---\n")
            f.write(f"Durée enregistrement    : {record_duration:.2f}s\n")
            f.write(f"Frames totales stockées : {stats.get('n_frames_total', 0)}\n")
            f.write(f"Frames utilisées        : {stats.get('n_frames_used', 0)}\n")
            f.write(f"Frames ignorées (erreur): {stats.get('n_frames_skipped', 0)}\n\n")
            mode = stats.get('mode', 'fast')
            if 'tsdf' in mode:
                f.write(f"--- Reconstruction TSDF ({mode}) ---\n")
                f.write(f"Voxel length            : {TSDF_VOXEL_LENGTH*1000:.0f}mm\n")
                f.write(f"SDF trunc               : {TSDF_SDF_TRUNC*100:.0f}cm\n")
                f.write(f"Points extraits (bruts) : {stats.get('n_raw_points', 0):,}\n")
                f.write(f"Points finaux           : {stats.get('n_final_points', 0):,}\n")
            else:
                f.write(f"--- Reconstruction Rapide (accumulation) ---\n")
                f.write(f"Subsample               : {SUBSAMPLE}\n")
                f.write(f"Voxel size              : {VOXEL_SIZE*1000:.0f}mm\n")
                f.write(f"Points bruts            : {stats.get('n_raw_points', 0):,}\n")
                f.write(f"Points après voxel      : {stats.get('n_voxel_points', 0):,}\n")
                f.write(f"Points finaux           : {stats.get('n_final_points', 0):,}\n")
            f.write(f"Temps reconstruction    : {t_recon:.2f}s\n")
            fps_recon = stats.get('n_frames_used', 0) / t_recon if t_recon > 0 else 0
            f.write(f"FPS reconstruction      : {fps_recon:.2f} frames/s\n\n")
            f.write(f"--- Normales ---\n")
            n_pts = stats.get('n_final_points', 0)
            pts_per_sec = n_pts / t_normals if t_normals > 0 else 0
            f.write(f"Temps normales          : {t_normals:.2f}s\n")
            f.write(f"Points/s normales       : {pts_per_sec:,.0f} pts/s\n\n")
            f.write(f"--- Total ---\n")
            f.write(f"Temps total pipeline    : {t_total:.2f}s\n")

        print(f"📁 Logs sauvegardés → {log_dir}/")

    # ── Pipeline complet post-enregistrement ──────────────────────────────────

    def process_recording(self, record_duration, save_npz=False):
        """
        Exécute le pipeline complet :
        reconstruction → normales → sauvegarde PLY + logs.
        """
        if len(self.recorded_frames) < 2:
            print("⚠️  Pas assez de frames enregistrées (min 2).")
            return

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir   = os.path.join("logs", timestamp)
        os.makedirs(log_dir, exist_ok=True)

        if save_npz:
            npz_path = os.path.join(log_dir, "scan.npz")
            self.save_raw_recording(npz_path)

        # 1. Reconstruction (dispatch selon USE_TSDF)
        if USE_TSDF:
            pcd, stats = self.reconstruct(self.recorded_frames)
        else:
            pcd, stats = self.reconstruct_fast(self.recorded_frames)
        if pcd is None:
            return

        # 2. Normales
        pcd, t_normals = self.compute_normals(pcd)

        # 3. Sauvegarde
        ply_path = os.path.join(log_dir, "reconstructed.ply")
        o3d.io.write_point_cloud(ply_path, pcd)
        n_final = stats.get('n_final_points', len(pcd.points))
        print(f"💾 Nuage sauvegardé → {ply_path}  ({n_final:,} points + normales)")

        # 4. Logs
        self.save_logs(log_dir, record_duration, stats, t_normals)

        # Libérer la mémoire des frames enregistrées
        self.recorded_frames = []

        print(f"\n✅ Pipeline terminé. Voir : {log_dir}/")
        print(f"   → Visualiser : python view_ply.py {ply_path}")
    # ── Sauvegarde / chargement d'un recording brut (pour benchmark) ────────────

    def save_raw_recording(self, path):
        """
        Sauvegarde les frames enregistrées dans un fichier .npz pour réutilisation.
        Utile pour lancer des benchmarks reproductibles sans l'iPhone connecté.
        """
        frames = self.recorded_frames
        if not frames:
            print("❌ Aucune frame enregistrée.")
            return

        depths      = np.array([f['depth']     for f in frames], dtype=np.float32)
        rgbs        = np.array([f['rgb']       for f in frames], dtype=np.uint8)
        intrinsics  = np.array([f['intrinsic'] for f in frames], dtype=np.float64)
        has_conf    = frames[0]['confidence'] is not None
        confidences = np.array(
            [f['confidence'] if f['confidence'] is not None
             else np.zeros(frames[0]['depth'].shape, dtype=np.uint8)
             for f in frames], dtype=np.uint8
        )
        poses = np.array(
            [[f['pose']['qx'], f['pose']['qy'], f['pose']['qz'], f['pose']['qw'],
              f['pose']['tx'], f['pose']['ty'], f['pose']['tz']]
             for f in frames], dtype=np.float64
        )

        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        np.savez_compressed(
            path,
            depths=depths, rgbs=rgbs, intrinsics=intrinsics,
            confidences=confidences, has_confidence=np.array([has_conf]),
            poses=poses,
        )
        print(f"💾 Recording brut sauvegardé → {path}  ({len(frames)} frames)")

    @staticmethod
    def load_raw_recording(path):
        """
        Charge un recording sauvegardé (.npz) et retourne une liste de dicts
        au même format que recorded_frames.
        """
        data     = np.load(path, allow_pickle=False)
        n        = len(data['depths'])
        has_conf = bool(data['has_confidence'][0])
        poses    = data['poses']   # shape (N, 7): qx qy qz qw tx ty tz

        frames = []
        for i in range(n):
            frames.append({
                'depth':      data['depths'][i],
                'rgb':        data['rgbs'][i],
                'intrinsic':  data['intrinsics'][i],
                'confidence': data['confidences'][i] if has_conf else None,
                'pose': {
                    'qx': float(poses[i, 0]), 'qy': float(poses[i, 1]),
                    'qz': float(poses[i, 2]), 'qw': float(poses[i, 3]),
                    'tx': float(poses[i, 4]), 'ty': float(poses[i, 5]),
                    'tz': float(poses[i, 6]),
                },
            })
        print(f"\u2705 Recording chargé : {n} frames depuis {path}")
        return frames

    # ── Boucle principale ─────────────────────────────────────────────────────

    def run(self, save_npz=False):
        devices = Record3DStream.get_connected_devices()
        if not devices:
            print("❌ Aucun iPhone détecté via USB.")
            print("   → Vérifie que l'app Record3D est ouverte et que le streaming USB est activé.")
            return

        print(f"✅ Connexion à : product_id={devices[0].product_id}")
        self.session = Record3DStream()
        self.session.on_new_frame      = self.on_new_frame
        self.session.on_stream_stopped = self.on_stream_stopped
        self.session.connect(devices[0])

        print("🎬 Flux prêt. Appuie sur ⏺ dans Record3D pour démarrer.")
        print("   [ESPACE] → démarrer/arrêter l'enregistrement   [Q/ESC] → quitter")

        while not self.stream_stopped.is_set():
            self.new_frame_evt.wait(timeout=0.05)
            self.new_frame_evt.clear()

            with self._lock:
                rgb = self.latest_rgb

            if rgb is None:
                continue

            # ── Preview 2D ────────────────────────────────────────────────
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            h, w = bgr.shape[:2]
            if w > PREVIEW_W:
                bgr = cv2.resize(bgr, (PREVIEW_W, int(PREVIEW_W * h / w)))

            if self.recording:
                n       = len(self.recorded_frames)
                elapsed = time.time() - self.record_start_time
                # Indicateur rouge clignotant
                if int(elapsed * 2) % 2 == 0:
                    cv2.circle(bgr, (30, 25), 10, (0, 0, 255), -1)
                cv2.putText(bgr, f"REC  {elapsed:.1f}s  |  {n} frames",
                            (50, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(bgr, "[ESPACE] Arreter l'enregistrement",
                            (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            else:
                cv2.putText(bgr, "[ESPACE] Enregistrer   [Q/ESC] Quitter",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 100), 2)

            cv2.imshow("Record3D – Enregistrement", bgr)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), 27):
                if self.recording:
                    record_duration = self.stop_recording()
                    cv2.destroyAllWindows()
                    self.process_recording(record_duration, save_npz=save_npz)
                print("Sortie demandée.")
                break

            if key == 32:  # ESPACE
                if not self.recording:
                    self.start_recording()
                else:
                    record_duration = self.stop_recording()
                    cv2.destroyAllWindows()
                    self.process_recording(record_duration, save_npz=save_npz)

                    if not self.stream_stopped.is_set():
                        print("\n   [ESPACE] → nouvel enregistrement   [Q/ESC] → quitter")

        cv2.destroyAllWindows()
        print("👋 Programme terminé.")


# ── Entrée ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record3D capture + reconstruction")
    parser.add_argument(
        "--save-npz",
        action="store_true",
        default=False,
        help="Sauvegarder le recording brut en logs/<datetime>/scan.npz",
    )
    args = parser.parse_args()

    recorder = Record3DRecorder()
    recorder.run(save_npz=args.save_npz)
