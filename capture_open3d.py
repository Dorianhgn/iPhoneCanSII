"""
Record3D – Preview 2D (OpenCV) + Nuage de points 3D + Normales (Open3D)
------------------------------------------------------------------------
Basé sur demo-main.py : https://github.com/marek-simonik/record3d

Prérequis :
    pip install record3d open3d opencv-python numpy

Usage :
    1. Branche l'iPhone en USB
    2. Record3D app → Settings → "USB Streaming mode" activé
    3. Lance ce script : python record3d_open3d_v2.py
    4. Appuie sur ⏺ dans l'app pour démarrer le flux

Contrôles fenêtre 3D :
    [N]       → switch nuage de points ↔ normales (flèches)
    Fermer    → retour à la preview 2D
Contrôles preview 2D :
    [ESPACE]  → capturer la frame courante et ouvrir la 3D
    [Q / ESC] → quitter
"""

import threading
import numpy as np
import cv2
import open3d as o3d
from record3d import Record3DStream


# ══════════════════════════════════════════════════════════════════════════════
#  HYPERPARAMÈTRES
# ══════════════════════════════════════════════════════════════════════════════

# ── Nuage de points ───────────────────────────────────────────────────────────
SUBSAMPLE     = 2      # Sous-échantillonnage spatial du depth.
                       # 1 = tous les pixels (très dense, lent)
                       # 2 = 1 pixel sur 2  (recommandé)
                       # 3 = 1 pixel sur 3  (rapide, moins de détails)
                       # ↑ impacte directement la densité des normales aussi

MAX_DEPTH     = np.inf    # Profondeur max en mètres (np.inf = pas de limite)
POINT_SIZE    = 2.0    # Taille des points dans Open3D
PREVIEW_W     = 1280   # Largeur max de la fenêtre preview OpenCV

# ── Normales ──────────────────────────────────────────────────────────────────
NORMAL_KNN    = 30     # Nombre de voisins pour l'estimation PCA de la normale.
                       # ↑ plus grand → normales plus lisses, moins sensibles au bruit
                       # ↓ plus petit → normales plus locales/détaillées, plus bruitées
                       # Valeurs typiques : 10 (détaillé) à 50 (très lisse)

NORMAL_RADIUS = 0.05   # Rayon max (mètres) pour chercher les voisins.
                       # ↑ plus grand → normales plus globales
                       # ↓ plus petit → normales plus fines
                       # Ajuste selon l'échelle : 0.02 petit objet, 0.1-0.2 grande pièce

ARROW_LENGTH  = 0.1   # Longueur des flèches en mètres.
                       # ↑ augmente si la scène est grande ou les objets lointains
                       # Valeurs typiques : 0.01 (très court) à 0.1 (très long)

ARROW_COLOR   = [0.2, 0.8, 1.0]   # Couleur RGB des flèches [0-1]
                                   # Cyan par défaut


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTRUCTION DES FLÈCHES (LineSet)
# ══════════════════════════════════════════════════════════════════════════════

def build_normal_arrows(pcd, length=ARROW_LENGTH, color=ARROW_COLOR):
    """
    Construit un LineSet Open3D représentant les normales comme des segments.
    Chaque segment : point → point + normale * length
    """
    pts  = np.asarray(pcd.points)
    nrms = np.asarray(pcd.normals)

    if len(pts) == 0 or len(nrms) == 0:
        return None

    # Sous-échantillonnage supplémentaire pour lisibilité visuelle
    step    = 2
    pts     = pts[::step]
    nrms    = nrms[::step]

    origins  = pts
    tips     = pts + nrms * length
    n        = len(origins)

    vertices = np.vstack([origins, tips])
    lines    = np.column_stack([np.arange(n), np.arange(n, 2 * n)])

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(vertices)
    ls.lines  = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(
        np.tile(color, (n, 1)).astype(np.float64)
    )
    return ls


# ══════════════════════════════════════════════════════════════════════════════
#  CLASSE PRINCIPALE
# ══════════════════════════════════════════════════════════════════════════════

class Record3DViewer:

    def __init__(self):
        self.session        = None
        self.stream_stopped = threading.Event()
        self.new_frame_evt  = threading.Event()

        self.latest_rgb        = None
        self.latest_depth      = None
        self.latest_intrinsic  = None
        self.latest_confidence = None
        self._lock             = threading.Lock()

    # ── Utilitaires ───────────────────────────────────────────────────────────

    def get_intrinsic_mat_from_coeffs(self, coeffs, depth_w, depth_h, rgb_w, rgb_h):
        """
        Coefficients Record3D exprimés dans le repère RGB → rescale vers depth.
        """
        scale_x = depth_w / rgb_w
        scale_y = depth_h / rgb_h
        return np.array([[coeffs.fx * scale_x,              0,  coeffs.tx * scale_x],
                         [             0,  coeffs.fy * scale_y,  coeffs.ty * scale_y],
                         [             0,              0,                1            ]])

    # ── Callbacks Record3D ────────────────────────────────────────────────────

    def on_new_frame(self):
        try:
            rgb    = self.session.get_rgb_frame()
            depth  = self.session.get_depth_frame()
            dH, dW = depth.shape[:2]
            rH, rW = rgb.shape[:2]
            coeffs = self.session.get_intrinsic_mat()
            intrinsic = self.get_intrinsic_mat_from_coeffs(coeffs, dW, dH, rW, rH)

            if self.latest_depth is None:
                print(f"[DEBUG] Depth : {depth.shape}  RGB : {rgb.shape}")
                print(f"[DEBUG] coeffs bruts : fx={coeffs.fx:.2f} cx={coeffs.tx:.2f} cy={coeffs.ty:.2f}")
                print(f"[DEBUG] intrinsic depth :\n{np.round(intrinsic, 2)}")

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

            self.new_frame_evt.set()

        except Exception as e:
            print(f"[on_new_frame] Erreur : {e}")

    def on_stream_stopped(self):
        print("Stream arrêté.")
        self.stream_stopped.set()
        self.new_frame_evt.set()

    # ── Conversion RGBD → nuage de points ────────────────────────────────────

    def rgbd_to_pointcloud(self, rgb, depth, intrinsic, confidence=None):
        H, W = depth.shape

        if rgb.shape[0] != H or rgb.shape[1] != W:
            rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_LINEAR)

        ys = np.arange(0, H, SUBSAMPLE)
        xs = np.arange(0, W, SUBSAMPLE)
        xv, yv = np.meshgrid(xs, ys)
        xv = xv.flatten()
        yv = yv.flatten()

        z    = depth[yv, xv]
        mask = (z > 0) & (z < MAX_DEPTH)

        if confidence is not None:
            c     = confidence[yv, xv]
            mask &= (c > 0)

        xv, yv, z = xv[mask], yv[mask], z[mask]
        if len(z) == 0:
            return None

        fx = intrinsic[0, 0];  cx = intrinsic[0, 2]
        fy = intrinsic[1, 1];  cy = intrinsic[1, 2]

        X = (xv - cx) * z / fx
        Y = (yv - cy) * z / fy
        Z = z

        points = np.stack([X, Y, Z], axis=-1)
        colors = rgb[yv, xv].astype(np.float32) / 255.0

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd

    # ── Estimation des normales ───────────────────────────────────────────────

    def compute_normals(self, pcd):
        """Estime les normales par PCA sur les voisins locaux puis les oriente."""
        print(f"  Calcul normales : knn={NORMAL_KNN}, radius={NORMAL_RADIUS}…")
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=NORMAL_RADIUS,
                max_nn=NORMAL_KNN
            )
        )
        # Oriente toutes les normales vers la caméra (position [0,0,0])
        pcd.orient_normals_towards_camera_location(np.array([0.0, 0.0, 0.0]))
        print("  ✅ Normales prêtes.")
        return pcd

    # ── Affichage Open3D avec toggle [N] ─────────────────────────────────────

    def show_pointcloud(self, pcd):
        """
        Fenêtre Open3D interactive.
        [N] bascule entre nuage coloré et flèches de normales.
        """
        print("Ouverture Open3D…")
        print("  [N] → switch nuage ↔ normales   [Fermer] → retour preview")

        # Pré-calcul des normales (une seule fois à l'ouverture)
        pcd    = self.compute_normals(pcd)
        arrows = build_normal_arrows(pcd, length=ARROW_LENGTH, color=ARROW_COLOR)

        if arrows is None:
            print("⚠️  Impossible de construire les flèches (nuage vide).")

        state = {"show_normals": False}

        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window("Record3D – [N] normales", width=1280, height=720)

        opt = vis.get_render_option()
        opt.background_color = np.array([0.05, 0.05, 0.05])
        opt.point_size = POINT_SIZE

        vis.add_geometry(pcd)

        def toggle_normals(vis):
            state["show_normals"] = not state["show_normals"]
            if state["show_normals"]:
                vis.remove_geometry(pcd, reset_bounding_box=False)
                if arrows is not None:
                    vis.add_geometry(arrows, reset_bounding_box=False)
                print("  → Normales")
            else:
                if arrows is not None:
                    vis.remove_geometry(arrows, reset_bounding_box=False)
                vis.add_geometry(pcd, reset_bounding_box=False)
                print("  → Nuage de points")

        vis.register_key_callback(78, toggle_normals)   # 78 = touche N
        vis.get_view_control().set_zoom(0.6)
        vis.run()
        vis.destroy_window()
        print("Fenêtre 3D fermée. Preview 2D reprise.")

    # ── Boucle principale ─────────────────────────────────────────────────────

    def run(self):
        devices = Record3DStream.get_connected_devices()
        if not devices:
            print("❌  Aucun iPhone détecté via USB.")
            print("    → Vérifie que l'app Record3D est ouverte et que le streaming USB est activé.")
            return

        print(f"✅  Connexion à : product_id={devices[0].product_id}")
        self.session = Record3DStream()
        self.session.on_new_frame      = self.on_new_frame
        self.session.on_stream_stopped = self.on_stream_stopped
        self.session.connect(devices[0])

        print("🎬  Flux prêt. Appuie sur ⏺ dans Record3D pour démarrer.")
        print("    [ESPACE] → capturer le nuage 3D    [Q / ESC] → quitter")

        while not self.stream_stopped.is_set():

            self.new_frame_evt.wait(timeout=0.05)
            self.new_frame_evt.clear()

            with self._lock:
                rgb        = self.latest_rgb
                depth      = self.latest_depth
                intrinsic  = self.latest_intrinsic
                confidence = self.latest_confidence

            if rgb is None or depth is None:
                continue

            # Preview 2D
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            h, w = bgr.shape[:2]
            if w > PREVIEW_W:
                bgr = cv2.resize(bgr, (PREVIEW_W, int(PREVIEW_W * h / w)))

            cv2.putText(bgr, "[ESPACE] Capturer 3D   [Q/ESC] Quitter",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 100), 2)
            cv2.imshow("Record3D – Preview 2D", bgr)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                print("Sortie demandée.")
                break

            if key == 32:  # ESPACE → capture
                print("📸  Capture en cours…")

                with self._lock:
                    rgb_snap  = self.latest_rgb.copy()        if self.latest_rgb        is not None else None
                    dep_snap  = self.latest_depth.copy()      if self.latest_depth      is not None else None
                    int_snap  = self.latest_intrinsic.copy()  if self.latest_intrinsic  is not None else None
                    conf_snap = self.latest_confidence.copy() if self.latest_confidence is not None else None

                if rgb_snap is None or dep_snap is None or int_snap is None:
                    print("⚠️  Données insuffisantes, réessaie.")
                    continue

                pcd = self.rgbd_to_pointcloud(rgb_snap, dep_snap, int_snap, conf_snap)

                if pcd is None or len(pcd.points) == 0:
                    print("⚠️  Nuage vide.")
                    continue

                print(f"✅  Nuage généré : {len(pcd.points):,} points")
                o3d.io.write_point_cloud("capture.ply", pcd)
                print("💾  Nuage sauvegardé → capture.ply")

                cv2.destroyAllWindows()
                self.show_pointcloud(pcd)

                if not self.stream_stopped.is_set():
                    print("    [ESPACE] → capturer à nouveau   [Q/ESC] → quitter")

        cv2.destroyAllWindows()
        print("👋  Programme terminé.")


# ── Entrée ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    viewer = Record3DViewer()
    viewer.run()