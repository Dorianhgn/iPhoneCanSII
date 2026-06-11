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
    pip install record3d open3d opencv-python numpy scipy pyyaml

Usage :
    1. Branche l'iPhone en USB
    2. Record3D app → Settings → "USB Streaming mode" activé
    3. Lance : python record_reconstruct.py
    (optionnel) : python record_reconstruct.py --config config_record_reconstruct.yaml
    (optionnel) : python record_reconstruct.py --from-npz logs/.../scan.npz -c une_config.yaml
    (optionnel) : python record_reconstruct.py --save-npz
    4. Appuie sur ⏺ dans l'app pour démarrer le flux

Contrôles :
    [ESPACE]  → démarrer / arrêter l'enregistrement
    [Q / ESC] → quitter

Sorties (dans logs/<datetime>/) :
    - reconstructed.ply   : nuage de points avec normales
    - config_record_reconstruct.yaml : hyperparamètres utilisés
    - performance.log     : métriques de performance
    - scan.npz            : recording brut (si --save-npz activé)

En mode --from-npz :
    - sortie dans logs/<datetime>/reconstruct_i/ (i auto-incrémenté)
    - pas besoin de réenregistrer avec l'iPhone
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

# record3d n'est nécessaire que pour la capture live (iPhone branché).
# Import paresseux : le bake offline (--from-npz) tourne sans lui.
try:
    from record3d import Record3DStream
except ImportError:
    Record3DStream = None

# Bake spec §4 (segmentation + scene.json). Imports légers (ultralytics chargé
# paresseusement seulement quand on segmente vraiment).
from segmentation import Segmenter, Taxonomy, DEFAULT_TAXONOMY
from scene_export import bake_scene

try:
    import yaml
except ImportError:
    yaml = None


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

# ── Segmentation YOLO (état couleur 'segmentation' + labels, spec §4) ─────────
YOLO_MODEL        = "yolo26s-seg.pt"  # Modèle YOLO-seg (fallback yolov8s-seg.pt).
YOLO_CONF         = 0.4     # Seuil de confiance des détections.
SEG_ENABLED       = False   # Défaut de l'overlay live ('y') ET intent d'enregistrement.
                            # Verrouillé au démarrage de l'enregistrement (guardrail).
SEG_ASSIGN_RADIUS = 0.06    # m : rayon max pour attacher une classe à un point final.

# ── Bake scène (spec §4) ──────────────────────────────────────────────────────
ARROW_ANGLE_DEG_THRESHOLD = 30.0       # Seuil angulaire normale↔verticale (obstacle).
ARROW_ANGLE_AXIS          = [0.0, 1.0, 0.0]  # Axe vertical de référence.
OBSTACLE_DIM      = 0.30    # Atténuation des points NON-obstacle (état obstacles).
MAX_POINTS        = 200000  # Budget de points du nuage baké (< ~200k pour 60 FPS web).
RECENTER          = True    # Recentrer le nuage à l'origine (auto-rotation centrée).

# ── Preview ───────────────────────────────────────────────────────────────────
PREVIEW_W         = 1280    # Largeur max de la fenêtre preview OpenCV.


# ══════════════════════════════════════════════════════════════════════════════
#  Collecte pour logs
# ══════════════════════════════════════════════════════════════════════════════

def get_hyperparams():
    """Retourne les hyperparamètres courants (defaults + éventuels overrides YAML)."""
    return {
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
        # Segmentation + bake §4
        "YOLO_MODEL":        YOLO_MODEL,
        "YOLO_CONF":         YOLO_CONF,
        "SEG_ENABLED":       SEG_ENABLED,
        "SEG_ASSIGN_RADIUS": SEG_ASSIGN_RADIUS,
        "ARROW_ANGLE_DEG_THRESHOLD": ARROW_ANGLE_DEG_THRESHOLD,
        "ARROW_ANGLE_AXIS":  ARROW_ANGLE_AXIS,
        "OBSTACLE_DIM":      OBSTACLE_DIM,
        "MAX_POINTS":        MAX_POINTS,
        "RECENTER":          RECENTER,
    }


def load_hyperparams_from_yaml(config_path):
    """
    Charge les overrides hyperparamètres depuis un YAML.
    Les clés absentes gardent les valeurs par défaut définies dans ce script.
    """
    if not config_path:
        return

    if not os.path.exists(config_path):
        print(f"ℹ️  Fichier config absent ({config_path}) → valeurs par défaut utilisées.")
        return

    if yaml is None:
        print("⚠️  PyYAML non installé. Impossible de lire la config YAML, valeurs par défaut utilisées.")
        print("   → Installe : pip install pyyaml")
        return

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        print(f"⚠️  Le fichier {config_path} ne contient pas un mapping YAML valide. Defaults conservés.")
        return

    valid = set(get_hyperparams().keys())
    unknown = sorted(k for k in data.keys() if k not in valid)
    if unknown:
        print(f"⚠️  Clés YAML ignorées (inconnues) : {', '.join(unknown)}")

    for key in valid:
        if key in data:
            globals()[key] = data[key]

    print(f"✅ Config chargée depuis {config_path}")


def make_reconstruct_output_dir(npz_path):
    """
    Crée un dossier reconstruct_i dans le dossier parent du .npz,
    sans écraser les reconstructions existantes.
    """
    base_dir = os.path.dirname(os.path.abspath(npz_path))
    i = 1
    while True:
        out_dir = os.path.join(base_dir, f"reconstruct_{i}")
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=False)
            return out_dir
        i += 1


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

        # Segmentation YOLO
        self._segmenter         = None          # chargé paresseusement
        self.yolo_enabled       = bool(SEG_ENABLED)  # toggle live ('y')
        self.yolo_locked        = bool(SEG_ENABLED)  # intent verrouillé pdt l'enregistrement
        self._last_overlay      = None          # cache overlay live

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
        # Guardrail : on fige l'intent YOLO au démarrage — il ne peut plus changer
        # pendant l'enregistrement. L'enregistrement est donc "avec" ou "sans" YOLO.
        self.yolo_locked        = bool(self.yolo_enabled)
        self.recording          = True
        state = "AVEC YOLO" if self.yolo_locked else "SANS YOLO"
        print(f"🔴 Enregistrement démarré ({state}, verrouillé) — bouge l'iPhone lentement.")

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

    # ── Segmentation : projette les classes ICANSII sur le nuage final ────────
    def get_segmenter(self):
        """Charge (paresseusement) le Segmenter YOLO partagé."""
        if self._segmenter is None:
            self._segmenter = Segmenter(
                model_path=YOLO_MODEL,
                conf=YOLO_CONF,
                taxonomy=DEFAULT_TAXONOMY,
            )
        return self._segmenter

    def segment_cloud(self, frames, final_points):
        """
        Exécute YOLO-seg sur chaque frame RGB stockée, déprojette les pixels
        classés en 3D (mêmes intrinsèques que la reconstruction), puis attache
        une classe ICANSII à chaque point final par plus-proche-voisin.

        Retourne (class_ids (N,) int16 [-1=aucune], stats_dict).
        L'inférence est 100% offline (spec §4 : aucune inférence côté web).
        """
        from scipy.spatial import cKDTree

        seg = self.get_segmenter()
        print(f"\n🧠 Segmentation offline de {len(frames)} frames (modèle {YOLO_MODEL})...")
        t0 = time.time()

        lab_pts, lab_cls = [], []
        n_inst = 0
        for i, frame in enumerate(frames):
            if (i + 1) % 25 == 0 or i == len(frames) - 1:
                print(f"  Frame {i+1}/{len(frames)} | instances cumulées : {n_inst}")

            rgb        = frame['rgb']
            depth      = frame['depth']
            K          = frame['intrinsic']
            confidence = frame['confidence']
            H, W       = depth.shape

            instances = seg.infer(rgb)
            n_inst += len(instances)
            cmap = seg.class_map(instances, H, W)      # (H,W) int16, -1 = aucune

            ys, xs = np.where(cmap >= 0)
            if len(xs) == 0:
                continue
            z = depth[ys, xs]
            valid = (z > 0) & (z < MAX_DEPTH)
            if confidence is not None:
                valid &= (confidence[ys, xs] >= CONFIDENCE_MIN)
            xs, ys, z = xs[valid], ys[valid], z[valid]
            if len(z) == 0:
                continue
            cls = cmap[ys, xs]

            fx, cx, fy, cy = K[0, 0], K[0, 2], K[1, 1], K[1, 2]
            local = np.stack([(xs - cx) * z / fx, -(ys - cy) * z / fy, -z], axis=-1)
            p = frame['pose']
            T = pose_to_matrix(p['qx'], p['qy'], p['qz'], p['qw'],
                               p['tx'], p['ty'], p['tz'])
            world = (T[:3, :3] @ local.T).T + T[:3, 3]
            lab_pts.append(world)
            lab_cls.append(cls)

        class_ids = np.full(len(final_points), -1, dtype=np.int16)
        stats = {'n_instances': n_inst, 'n_labeled_raw': 0, 'n_assigned': 0,
                 't_segment': time.time() - t0}
        if not lab_pts:
            print("  ⚠️  Aucun objet segmenté → état segmentation vide (gris neutre).")
            return class_ids, stats

        lab_pts = np.vstack(lab_pts)
        lab_cls = np.concatenate(lab_cls).astype(np.int16)
        stats['n_labeled_raw'] = int(len(lab_pts))

        tree = cKDTree(lab_pts)
        d, idx = tree.query(final_points, k=1, distance_upper_bound=SEG_ASSIGN_RADIUS)
        hit = np.isfinite(d)
        class_ids[hit] = lab_cls[idx[hit]]
        stats['n_assigned'] = int(hit.sum())
        stats['t_segment'] = time.time() - t0
        print(f"  ✅ Segmentation : {n_inst} instances, "
              f"{stats['n_assigned']:,}/{len(final_points):,} points classés "
              f"en {stats['t_segment']:.1f}s")
        return class_ids, stats

    # ── Sauvegarde des logs ───────────────────────────────────────────────────

    def save_logs(self, log_dir, record_duration, stats, t_normals):
        """Écrit la config YAML et performance.log dans log_dir."""
        os.makedirs(log_dir, exist_ok=True)

        # config_record_reconstruct.yaml — hyperparamètres effectifs
        config_path = os.path.join(log_dir, "config_record_reconstruct.yaml")
        with open(config_path, "w", encoding="utf-8") as f:
            f.write("# Hyperparamètres de la reconstruction\n")
            f.write(f"# Généré le {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            if yaml is not None:
                yaml.safe_dump(get_hyperparams(), f, sort_keys=False, allow_unicode=True)
            else:
                # Fallback lisible même sans PyYAML.
                for k, v in get_hyperparams().items():
                    f.write(f"{k}: {v}\n")

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

    def process_recording(self, record_duration, save_npz=False, output_dir=None):
        """
        Exécute le pipeline complet :
        reconstruction → normales → sauvegarde PLY + logs.
        """
        if len(self.recorded_frames) < 2:
            print("⚠️  Pas assez de frames enregistrées (min 2).")
            return

        if output_dir is not None:
            log_dir = output_dir
        else:
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

        # 3. Sauvegarde brute (compat) : reconstructed.ply (RGB + normales)
        ply_path = os.path.join(log_dir, "reconstructed.ply")
        o3d.io.write_point_cloud(ply_path, pcd)
        n_final = stats.get('n_final_points', len(pcd.points))
        print(f"💾 Nuage sauvegardé → {ply_path}  ({n_final:,} points + normales)")

        # 4. Segmentation offline (si l'enregistrement était "avec YOLO")
        class_ids = None
        if self.yolo_locked:
            try:
                class_ids, seg_stats = self.segment_cloud(
                    self.recorded_frames, np.asarray(pcd.points)
                )
                stats.update(seg_stats)
            except Exception as e:
                print(f"⚠️  Segmentation échouée ({e}) → état segmentation vide.")
                class_ids = None
        else:
            print("ℹ️  Enregistrement SANS YOLO → pas d'état segmentation baké.")

        # 5. Bake du contrat de données §4 : scene.json + 3 .ply (même ordre)
        bake_stats = bake_scene(
            np.asarray(pcd.points),
            (np.asarray(pcd.colors) * 255.0).astype(np.uint8) if pcd.has_colors()
                else np.full((len(pcd.points), 3), 200, np.uint8),
            normals=np.asarray(pcd.normals) if pcd.has_normals() else None,
            class_ids=class_ids,
            taxonomy=DEFAULT_TAXONOMY,
            angle_axis=ARROW_ANGLE_AXIS,
            angle_threshold_deg=ARROW_ANGLE_DEG_THRESHOLD,
            obstacle_dim=OBSTACLE_DIM,
            max_points=MAX_POINTS,
            recenter=RECENTER,
            out_dir=log_dir,
        )
        stats.update({f"bake_{k}": v for k, v in bake_stats.items()})
        print(f"🎬 Scène bakée (§4) → {bake_stats['scene_dir']}/  "
              f"({bake_stats['n_points']:,} pts, {bake_stats['n_labels']} labels, "
              f"{bake_stats['scene_json_mb']} MB)")

        # 6. Assets §11 : rgb.png + depth.png (frame représentative médiane)
        self.save_preview_assets(log_dir)

        # 7. Logs
        self.save_logs(log_dir, record_duration, stats, t_normals)

        # Libérer la mémoire des frames enregistrées
        self.recorded_frames = []

        print(f"\n✅ Pipeline terminé. Voir : {log_dir}/")
        print(f"   → Visualiser : python view_ply.py {ply_path}")
        print(f"   → Web (§4)   : {log_dir}/scene/scene.json")

    # ── Assets §11 (rgb.png + depth.png) ──────────────────────────────────────
    def save_preview_assets(self, log_dir):
        """Sauve une frame RGB + sa depth colorisée (assets §11, états 2/3)."""
        frames = self.recorded_frames
        if not frames:
            return
        mid = len(frames) // 2
        rgb = frames[mid]['rgb']
        depth = frames[mid]['depth']
        cv2.imwrite(os.path.join(log_dir, "rgb.png"),
                    cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        d = depth.astype(np.float32)
        valid = (d > 0) & (d < MAX_DEPTH)
        dn = np.zeros_like(d)
        if valid.any():
            lo, hi = d[valid].min(), d[valid].max()
            dn[valid] = (d[valid] - lo) / max(hi - lo, 1e-6)
        depth_vis = (dn * 255).astype(np.uint8)
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_TURBO)
        depth_vis[~valid] = 0
        cv2.imwrite(os.path.join(log_dir, "depth.png"), depth_vis)
        print(f"🖼️  Assets §11 → {log_dir}/rgb.png  + depth.png")
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
            # Guardrail : intent YOLO verrouillé de l'enregistrement (with/without).
            yolo_enabled=np.array([bool(self.yolo_locked)]),
        )
        print(f"💾 Recording brut sauvegardé → {path}  ({len(frames)} frames, "
              f"yolo={'on' if self.yolo_locked else 'off'})")

    @staticmethod
    def read_yolo_flag(path):
        """Lit l'intent YOLO stocké dans un .npz (False si absent → compat)."""
        try:
            data = np.load(path, allow_pickle=False)
            if 'yolo_enabled' in data:
                return bool(data['yolo_enabled'][0])
        except Exception:
            pass
        return False

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
        print("   [ESPACE] → enregistrer   [Y] → overlay YOLO (hors enregistrement)   [Q/ESC] → quitter")

        seg_warned = False
        while not self.stream_stopped.is_set():
            self.new_frame_evt.wait(timeout=0.05)
            self.new_frame_evt.clear()

            with self._lock:
                rgb = self.latest_rgb

            if rgb is None:
                continue

            # ── Overlay YOLO live : uniquement HORS enregistrement (guardrail) ──
            # Pendant l'enregistrement, l'intent YOLO est figé : pas d'inférence
            # live (préserve le FPS de capture), juste un indicateur "LOCKED".
            run_overlay = self.yolo_enabled and not self.recording
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if run_overlay:
                try:
                    seg = self.get_segmenter()
                    bgr = seg.overlay(bgr, seg.infer(rgb))
                except Exception as e:
                    if not seg_warned:
                        print(f"⚠️  Overlay YOLO indisponible ({e}) → flux RGB brut.")
                        seg_warned = True

            # ── Preview 2D ────────────────────────────────────────────────
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
                lock_txt = f"YOLO {'ON' if self.yolo_locked else 'OFF'}  [LOCKED]"
                lock_col = (80, 220, 80) if self.yolo_locked else (160, 160, 160)
                cv2.putText(bgr, lock_txt, (10, 95),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, lock_col, 2)
            else:
                cv2.putText(bgr, "[ESPACE] Enregistrer   [Q/ESC] Quitter",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 100), 2)
                yolo_txt = f"YOLO {'ON' if self.yolo_enabled else 'OFF'}   [Y] toggle"
                yolo_col = (80, 220, 80) if self.yolo_enabled else (200, 200, 200)
                cv2.putText(bgr, yolo_txt, (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, yolo_col, 2)

            cv2.imshow("Record3D – Enregistrement", bgr)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), 27):
                if self.recording:
                    record_duration = self.stop_recording()
                    cv2.destroyAllWindows()
                    self.process_recording(record_duration, save_npz=save_npz)
                print("Sortie demandée.")
                break

            if key == ord('y'):
                if self.recording:
                    # Guardrail : interdit de changer l'intent YOLO en cours d'enreg.
                    print("🔒 YOLO verrouillé pendant l'enregistrement (guardrail). Ignoré.")
                else:
                    self.yolo_enabled = not self.yolo_enabled
                    print(f"🧠 Overlay YOLO live : {'ON' if self.yolo_enabled else 'OFF'}")

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
        "-c",
        "--config",
        type=str,
        default="config_record_reconstruct.yaml",
        help="Chemin vers la config YAML (overrides partiels, defaults sinon)",
    )
    parser.add_argument(
        "--from-npz",
        type=str,
        default=None,
        help="Reconstruire depuis un scan.npz existant (sans iPhone)",
    )
    parser.add_argument(
        "--save-npz",
        action="store_true",
        default=False,
        help="Sauvegarder le recording brut en logs/<datetime>/scan.npz",
    )
    parser.add_argument(
        "--yolo", dest="yolo", action="store_true", default=None,
        help="Forcer la segmentation YOLO ON (overlay live + bake segmentation).",
    )
    parser.add_argument(
        "--no-yolo", dest="yolo", action="store_false",
        help="Forcer la segmentation YOLO OFF.",
    )
    parser.add_argument(
        "--max-points", type=int, default=None,
        help=f"Budget de points du nuage baké (défaut: {MAX_POINTS}).",
    )
    args = parser.parse_args()

    load_hyperparams_from_yaml(args.config)
    if args.max_points is not None:
        MAX_POINTS = args.max_points

    recorder = Record3DRecorder()
    # --yolo / --no-yolo surchargent le défaut SEG_ENABLED de l'overlay live.
    if args.yolo is not None:
        recorder.yolo_enabled = bool(args.yolo)
        recorder.yolo_locked  = bool(args.yolo)

    if args.from_npz:
        npz_path = args.from_npz
        if not os.path.exists(npz_path):
            print(f"❌ Fichier NPZ introuvable : {npz_path}")
            raise SystemExit(1)

        print(f"📦 Reconstruction depuis NPZ : {npz_path}")
        recorder.recorded_frames = Record3DRecorder.load_raw_recording(npz_path)
        # Intent YOLO : priorité au flag CLI, sinon celui stocké dans le .npz.
        if args.yolo is None:
            recorder.yolo_locked = Record3DRecorder.read_yolo_flag(npz_path)
        print(f"🧠 Segmentation pour ce bake : "
              f"{'ON' if recorder.yolo_locked else 'OFF'}")
        out_dir = make_reconstruct_output_dir(npz_path)
        print(f"📁 Dossier de sortie : {out_dir}")
        recorder.process_recording(record_duration=0.0, save_npz=False, output_dir=out_dir)
    else:
        if Record3DStream is None:
            print("❌ record3d non installé — capture live impossible sur cette machine.")
            print("   → Branche l'iPhone sur le Mac, ou utilise --from-npz pour le bake offline.")
            raise SystemExit(1)
        recorder.run(save_npz=args.save_npz)
