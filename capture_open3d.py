"""
Record3D – Preview 2D (OpenCV) + Nuage de points 3D (Open3D) sur [ESPACE]
--------------------------------------------------------------------------
Basé sur demo-main.py : https://github.com/marek-simonik/record3d

Prérequis :
    pip install record3d open3d opencv-python numpy

Sur Linux :
    sudo apt install libusbmuxd-dev

Usage :
    1. Branche l'iPhone en USB
    2. Record3D app → Settings → "USB Streaming mode" activé
    3. Lance ce script : python record3d_open3d_v2.py
    4. Appuie sur ⏺ dans l'app pour démarrer le flux
    5. [ESPACE]  → capture la frame courante et ouvre le nuage 3D Open3D
       [Q / ESC] → quitte la preview 2D
"""

import threading
import numpy as np
import cv2
import open3d as o3d
from record3d import Record3DStream


# ── Paramètres ────────────────────────────────────────────────────────────────

SUBSAMPLE   = 2      # 1 = tous les pixels, 2 = 1/2, 3 = 1/3 …
MAX_DEPTH   = 5.0    # mètres – points au-delà filtrés
POINT_SIZE  = 2.0    # taille des points dans Open3D
PREVIEW_W   = 1280   # largeur max de la fenêtre preview OpenCV


# ── Classe principale ─────────────────────────────────────────────────────────

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
        Les coefficients Record3D (fx, fy, tx, ty) sont exprimés dans le repère
        de l'image RGB (ex: 720×960). On les rescale à la résolution du depth map
        (ex: 192×256) car c'est sur le depth qu'on fait la rétroprojection.

            scale_x = depth_w / rgb_w   (ex: 192/720 ≈ 0.267)
            scale_y = depth_h / rgb_h   (ex: 256/960 ≈ 0.267)
        """
        scale_x = depth_w / rgb_w
        scale_y = depth_h / rgb_h
        return np.array([[coeffs.fx * scale_x,             0,  coeffs.tx * scale_x],
                         [            0,  coeffs.fy * scale_y,  coeffs.ty * scale_y],
                         [            0,             0,                1            ]])

    # ── Callbacks Record3D ────────────────────────────────────────────────────

    def on_new_frame(self):
        """Appelé dans le thread Record3D à chaque frame RGBD disponible."""
        try:
            rgb       = self.session.get_rgb_frame()
            depth     = self.session.get_depth_frame()
            dH, dW   = depth.shape[:2]
            rH, rW   = rgb.shape[:2]
            coeffs    = self.session.get_intrinsic_mat()
            intrinsic = self.get_intrinsic_mat_from_coeffs(coeffs, dW, dH, rW, rH)

            # Debug à la première frame : affiche les valeurs pour vérification
            if self.latest_depth is None:
                print(f"[DEBUG] Depth shape   : {depth.shape}  (W={dW}, H={dH})")
                print(f"[DEBUG] RGB shape     : {rgb.shape}  (W={rW}, H={rH})")
                print(f"[DEBUG] coeffs bruts  : fx={coeffs.fx:.2f} fy={coeffs.fy:.2f} cx={coeffs.tx:.2f} cy={coeffs.ty:.2f}")
                print(f"[DEBUG] scale x={dW/rW:.4f}  scale y={dH/rH:.4f}")
                print(f"[DEBUG] intrinsic depth px :\n{np.round(intrinsic, 2)}")

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

    # ── Conversion depth → 3D ────────────────────────────────────────────────

    def rgbd_to_pointcloud(self, rgb, depth, intrinsic, confidence=None):
        """
        Rétro-projection RGBD → nuage de points coloré.

        Returns
        -------
        pcd : open3d.geometry.PointCloud
        """
        H, W = depth.shape

        # RGB et depth n'ont pas forcément la même résolution → on aligne RGB sur depth
        if rgb.shape[0] != H or rgb.shape[1] != W:
            rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_LINEAR)

        ys = np.arange(0, H, SUBSAMPLE)
        xs = np.arange(0, W, SUBSAMPLE)
        xv, yv = np.meshgrid(xs, ys)
        xv = xv.flatten()
        yv = yv.flatten()

        z = depth[yv, xv]
        mask = (z > 0) & (z < MAX_DEPTH)

        if confidence is not None:
            c = confidence[yv, xv]
            mask &= (c > 0)

        xv, yv, z = xv[mask], yv[mask], z[mask]

        if len(z) == 0:
            return None

        fx = intrinsic[0, 0]
        fy = intrinsic[1, 1]
        cx = intrinsic[0, 2]
        cy = intrinsic[1, 2]

        X = (xv - cx) * z / fx
        Y = (yv - cy) * z / fy
        Z = z

        points = np.stack([X, Y, Z], axis=-1)
        colors = rgb[yv, xv].astype(np.float32) / 255.0

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd

    # ── Affichage Open3D (fenêtre bloquante) ─────────────────────────────────

    def show_pointcloud(self, pcd):
        """Ouvre une fenêtre Open3D interactive avec le nuage capturé."""
        print("Ouverture Open3D… Ferme la fenêtre pour reprendre la preview.")

        vis = o3d.visualization.Visualizer()
        vis.create_window("Record3D – Point Cloud", width=1280, height=720)

        opt = vis.get_render_option()
        opt.background_color = np.array([0.05, 0.05, 0.05])
        opt.point_size = POINT_SIZE

        vis.add_geometry(pcd)

        # Centrage automatique de la vue
        vis.get_view_control().set_zoom(0.6)
        vis.run()          # bloquant jusqu'à fermeture de la fenêtre
        vis.destroy_window()
        print("Fenêtre 3D fermée. Preview 2D reprise.")

    # ── Boucle principale ─────────────────────────────────────────────────────

    def run(self):
        # -- Connexion à l'appareil --
        devices = Record3DStream.get_connected_devices()
        if not devices:
            print("❌  Aucun iPhone détecté via USB.")
            print("    → Vérifie que l'app Record3D est ouverte et que le streaming USB est activé.")
            return

        print(f"✅  Connexion à : product_id={devices[0].product_id}")
        self.session = Record3DStream()
        self.session.on_new_frame    = self.on_new_frame
        self.session.on_stream_stopped = self.on_stream_stopped
        self.session.connect(devices[0])

        print("🎬  Flux prêt. Appuie sur ⏺ dans Record3D pour démarrer.")
        print("    [ESPACE] → capturer le nuage 3D    [Q / ESC] → quitter")

        while not self.stream_stopped.is_set():

            # -- Attente frame --
            self.new_frame_evt.wait(timeout=0.05)
            self.new_frame_evt.clear()

            with self._lock:
                rgb        = self.latest_rgb
                depth      = self.latest_depth
                intrinsic  = self.latest_intrinsic
                confidence = self.latest_confidence

            if rgb is None or depth is None:
                continue

            # -- Preview 2D --
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            h, w = bgr.shape[:2]
            if w > PREVIEW_W:
                bgr = cv2.resize(bgr, (PREVIEW_W, int(PREVIEW_W * h / w)))

            # Overlay info
            cv2.putText(bgr, "[ESPACE] Capturer 3D   [Q/ESC] Quitter",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 100), 2)

            cv2.imshow("Record3D – Preview 2D", bgr)
            key = cv2.waitKey(1) & 0xFF

            # -- Quitter --
            if key in (ord("q"), 27):
                print("Sortie demandée.")
                break

            # -- Capture 3D sur ESPACE --
            if key == 32:
                print("📸  Capture en cours…")

                with self._lock:
                    rgb_snap       = self.latest_rgb.copy()       if self.latest_rgb        is not None else None
                    depth_snap     = self.latest_depth.copy()     if self.latest_depth      is not None else None
                    intrinsic_snap = self.latest_intrinsic.copy() if self.latest_intrinsic  is not None else None
                    conf_snap      = self.latest_confidence.copy() if self.latest_confidence is not None else None

                if rgb_snap is None or depth_snap is None or intrinsic_snap is None:
                    print("⚠️   Données insuffisantes, réessaie.")
                    continue

                pcd = self.rgbd_to_pointcloud(rgb_snap, depth_snap, intrinsic_snap, conf_snap)

                if pcd is None or len(pcd.points) == 0:
                    print("⚠️   Nuage vide (vérifie la profondeur / confidence).")
                    continue

                print(f"✅  Nuage généré : {len(pcd.points):,} points")

                # -- Sauvegarde optionnelle --
                out_path = "capture.ply"
                o3d.io.write_point_cloud(out_path, pcd)
                print(f"💾  Nuage sauvegardé → {out_path}")

                # Ferme temporairement OpenCV avant Open3D (évite conflits sur macOS/Windows)
                cv2.destroyAllWindows()

                self.show_pointcloud(pcd)

                # Ré-ouvre la preview après la fermeture du viewer 3D
                if not self.stream_stopped.is_set():
                    print("    [ESPACE] → capturer à nouveau   [Q/ESC] → quitter")

        # -- Nettoyage --
        cv2.destroyAllWindows()
        print("👋  Programme terminé.")


# ── Entrée ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    viewer = Record3DViewer()
    viewer.run()