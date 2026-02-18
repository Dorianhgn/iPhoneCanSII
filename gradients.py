from record3d import Record3DStream
import cv2
import numpy as np
import matplotlib.pyplot as plt
from threading import Event

class Record3DCaptureApp:
    def __init__(self):
        self.session = None
        self.stream_stopped = Event()
        self.latest_rgb = None
        self.latest_depth = None
        
    def on_stream_stopped(self):
        print('Flux arrêté par l\'appareil.')
        self.stream_stopped.set()

    def on_new_frame(self):
        # Cette fonction est appelée automatiquement à chaque nouvelle frame (30/60 fps)
        # On met à jour nos variables avec les dernières données disponibles
        self.latest_depth = self.session.get_depth_frame()
        self.latest_rgb = self.session.get_rgb_frame()

    def compute_normals(self, depth_img, smoothness=3.0):
        # Convertir en float32 pour les calculs
        Z = depth_img.astype(np.float32)

        # Lissage pour éviter l'effet "bruit de sel" sur les normales
        Z = cv2.GaussianBlur(Z, (5, 5), 0)

        # Calcul des gradients (pentes) selon X et Y
        dz_dx = np.gradient(Z, axis=1)
        dz_dy = np.gradient(Z, axis=0)

        # Construction de la carte de normales
        normal_map = np.zeros((Z.shape[0], Z.shape[1], 3), dtype=np.float32)
        
        # Note : On inverse souvent les axes selon la convention (OpenGL vs OpenCV)
        # Ici on suppose une vue "standard". 
        normal_map[..., 0] = -dz_dx * (1.0 / smoothness)
        normal_map[..., 1] = -dz_dy * (1.0 / smoothness)
        normal_map[..., 2] = 1.0 

        # Normalisation vectorielle
        norm = np.linalg.norm(normal_map, axis=2, keepdims=True)
        normal_map = normal_map / (norm + 1e-6)

        # Mapping vers [0, 255] pour affichage RGB
        # X=R (Rouge), Y=G (Vert), Z=B (Bleu -> face caméra)
        vis_normals = ((normal_map + 1) / 2 * 255).astype(np.uint8)
        
        return vis_normals

    def run(self):
        print("Recherche de l'iPhone en USB...")
        devices = Record3DStream.get_connected_devices()
        if not devices:
            print("Aucun appareil trouvé ! Vérifiez que Record3D est lancé et 'USB Streaming' activé.")
            return

        print(f"Connexion à {devices[0].product_id}...")
        self.session = Record3DStream()
        self.session.on_new_frame = self.on_new_frame
        self.session.on_stream_stopped = self.on_stream_stopped
        self.session.connect(devices[0])

        print("Flux démarré ! Appuyez sur [ESPACE] pour capturer et calculer.")
        
        while not self.stream_stopped.is_set():
            # Attendre d'avoir reçu au moins une frame
            if self.latest_rgb is None or self.latest_depth is None:
                continue

            # Affichage du flux RGB pour viser
            # Record3D envoie du RGB, OpenCV veut du BGR
            rgb_show = cv2.cvtColor(self.latest_rgb, cv2.COLOR_RGB2BGR)
            cv2.imshow("Record3D Live - [ESPACE] pour capturer", rgb_show)

            key = cv2.waitKey(1)
            
            # Si touche ESPACE (32) ou ENTREE (13) est pressée
            if key == 32 or key == 13: 
                print("Capture en cours...")
                
                # On fige les données actuelles
                captured_depth = self.latest_depth.copy()
                captured_rgb = self.latest_rgb.copy()
                
                # Calcul des normales
                print("Calcul des normales...")
                normals = self.compute_normals(captured_depth, smoothness=2.0)
                
                # Affichage Matplotlib pour analyse fine
                plt.figure(figsize=(12, 5))
                
                plt.subplot(1, 3, 1)
                plt.title("RGB")
                plt.imshow(captured_rgb)
                plt.axis('off')
                
                plt.subplot(1, 3, 2)
                plt.title("Profondeur (Brute)")
                plt.imshow(captured_depth, cmap='magma')
                plt.axis('off')
                
                plt.subplot(1, 3, 3)
                plt.title("Normales Estimées")
                plt.imshow(normals)
                plt.axis('off')
                
                print("Fermez la fenêtre du graphique pour reprendre le flux.")
                plt.show()

            # Quitter avec 'q'
            elif key == ord('q'):
                break

        # Nettoyage
        self.session = None # Déconnexion
        cv2.destroyAllWindows()

if __name__ == '__main__':
    app = Record3DCaptureApp()
    app.run()