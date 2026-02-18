from record3d import Record3DStream
import cv2
import numpy as np
import open3d as o3d
from threading import Event

class Record3DToOpen3D:
    def __init__(self):
        self.session = None
        self.stream_stopped = Event()
        self.latest_rgb = None
        self.latest_depth = None
        
    def on_stream_stopped(self):
        self.stream_stopped.set()

    def on_new_frame(self):
        self.latest_depth = self.session.get_depth_frame()
        self.latest_rgb = self.session.get_rgb_frame()

    def get_intrinsic_mat_from_coeffs(self, coeffs):
        """Reconstruit la matrice intrinsèque (K) à partir des coefficients record3d."""
        return np.array([[coeffs.fx,         0, coeffs.tx],
                         [        0, coeffs.fy, coeffs.ty],
                         [        0,         0,         1]])

    def capture_and_process(self):
        # 1. Récupérer les données brutes
        rgb = self.latest_rgb.copy()
        depth = self.latest_depth.copy() # En mètres
        
        # 2. Gérer les dimensions (Le FIX de Claude)
        # La depth map est souvent plus petite que l'image RGB.
        # On redimensionne le RGB pour qu'il colle exactement à la depth.
        h, w = depth.shape
        rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # 3. Récupérer et formater la matrice intrinsèque
        coeffs = self.session.get_intrinsic_mat()
        K = self.get_intrinsic_mat_from_coeffs(coeffs)
        
        # Extraction des valeurs pour Open3D
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]
        
        # Création de l'objet caméra Open3D
        o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)

        # 4. Préparation pour Open3D
        o3d_color = o3d.geometry.Image(rgb)
        o3d_depth = o3d.geometry.Image(depth)

        # Création de l'image RGBD
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_color, 
            o3d_depth, 
            depth_scale=1.0, 
            depth_trunc=3.0, 
            convert_rgb_to_intensity=False
        )

        # 5. Projection en Nuage de points
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, 
            o3d_intrinsics
        )

        # Rotation pour l'orientation (souvent inversée sur mobile)
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        print(f"Nuage généré : {len(pcd.points)} points.")

        # 6. Estimation des normales
        print("Estimation des normales 3D...")
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        pcd.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.]))

        return pcd

    def run(self):
        devices = Record3DStream.get_connected_devices()
        if not devices:
            print("Pas d'iPhone détecté via USB (Record3D).")
            return

        print(f"Connexion à {devices[0].product_id}...")
        self.session = Record3DStream()
        self.session.on_new_frame = self.on_new_frame
        self.session.on_stream_stopped = self.on_stream_stopped
        self.session.connect(devices[0])

        print("Flux prêt. Appuie sur [ESPACE] pour capturer la 3D.")

        while not self.stream_stopped.is_set():
            if self.latest_rgb is None or self.latest_depth is None: 
                continue

            # Affichage preview 2D (optionnel)
            # On redimensionne aussi ici juste pour la preview pour que ce soit cohérent
            # (Note : self.latest_rgb est l'original haute def, on l'affiche tel quel ou resize)
            bgr_show = cv2.cvtColor(self.latest_rgb, cv2.COLOR_RGB2BGR)
            # Petit resize pour que ça tienne à l'écran si c'est du 4K
            preview_h, preview_w = bgr_show.shape[:2]
            if preview_w > 1280:
                bgr_show = cv2.resize(bgr_show, (1280, int(1280 * preview_h / preview_w)))
                
            cv2.imshow("Preview 2D - [ESPACE] Capture", bgr_show)
            
            key = cv2.waitKey(1)
            if key == 32: # Espace
                print("Capture en cours...")
                try:
                    pcd = self.capture_and_process()
                    
                    print("Ouverture du viewer 3D...")
                    print("Appuie sur 'N' pour voir les normales.")
                    o3d.visualization.draw_geometries([pcd], 
                                                      window_name="Resultat 3D",
                                                      width=800, height=600,
                                                      point_show_normal=False)
                except Exception as e:
                    print(f"Erreur lors du processing : {e}")
                
            elif key == ord('q'):
                break

        cv2.destroyAllWindows()

if __name__ == '__main__':
    app = Record3DToOpen3D()
    app.run()