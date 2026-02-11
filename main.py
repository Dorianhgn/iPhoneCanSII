import cv2
import numpy as np
from record3d import Record3DStream
from threading import Event
from ultralytics import YOLO
import torch

class IOSRobotPerception:
    def __init__(self):
        self.event = Event()
        self.session = None
        self.running = True
        
        # --- CONFIGURATION IA ---
        # Détection automatique de la puce Apple Silicon (M1/M2/M3/M4) pour l'accélération
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        print(f"🚀 Chargement de YOLOv8 sur : {device.upper()}")
        
        # Charge le modèle (télécharge yolov8n.pt tout seul si absent)
        self.model = YOLO("yolov8n.pt") 
        self.model.to(device)

    def on_new_frame(self):
        """
        This method is called from a non-main thread, therefore cannot be used for presenting UI.
        """
        self.event.set()  # Notify the main thread to stop waiting and process new frame.

    def on_stream_stopped(self):
        print('Stream stopped')

    def connect_to_device(self, dev_idx):
        print('🔗 Recherche des appareils connectés...')
        devs = Record3DStream.get_connected_devices()
        print(f'{len(devs)} appareil(s) trouvé(s)')
        for dev in devs:
            print(f'\tID: {dev.product_id}\n\tUDID: {dev.udid}\n')

        if len(devs) <= dev_idx:
            raise RuntimeError(f'Impossible de se connecter à l\'appareil #{dev_idx}, essayez un autre index.')

        dev = devs[dev_idx]
        self.session = Record3DStream()
        self.session.on_new_frame = self.on_new_frame
        self.session.on_stream_stopped = self.on_stream_stopped
        self.session.connect(dev)  # Initiate connection and start capturing

    def get_distance_at_center(self, depth_map, box, rgb_shape):
        """Récupère la distance médiane au centre de la bounding box"""
        x1, y1, x2, y2 = map(int, box)
        
        # Dimensions de l'image RGB et de la depth map
        rgb_h, rgb_w = rgb_shape[:2]
        depth_h, depth_w = depth_map.shape
        
        # Ratio de redimensionnement entre RGB et Depth
        scale_x = depth_w / rgb_w
        scale_y = depth_h / rgb_h
        
        # Convertir les coordonnées RGB vers l'espace Depth
        cx = int((x1 + x2) // 2 * scale_x)
        cy = int((y1 + y2) // 2 * scale_y)
        margin = 5  # Réduit car depth map est plus petite
        
        # Limites sécurisées
        x_min, x_max = max(0, cx - margin), min(depth_w, cx + margin)
        y_min, y_max = max(0, cy - margin), min(depth_h, cy + margin)
        
        roi = depth_map[y_min:y_max, x_min:x_max]
        
        # Filtrer les valeurs invalides (0 ou trop loin)
        valid_pixels = roi[(roi > 0.01) & (roi < 10.0)]
        
        if valid_pixels.size == 0:
            return None
        
        return np.median(valid_pixels)

    def start(self):
        print("🔗 En attente de connexion USB iPhone...")
        print("👉 Ouvre l'app Record3D sur l'iPhone et va dans l'onglet 'USB Streaming'")
        self.connect_to_device(dev_idx=0)  # Connect to the first device
        print("✅ Connecté ! Démarrage du stream...")
        self.loop()

    def loop(self):
        first_frame = True
        while self.running:
            self.event.wait()
            self.event.clear()

            # 1. Récupération des données brutes
            rgb = self.session.get_rgb_frame()
            depth_m = self.session.get_depth_frame()  # Déjà en mètres (tableau 2D de floats)
            
            # Debug première frame pour voir les dimensions
            if first_frame:
                print(f"📐 RGB shape: {rgb.shape}, Depth shape: {depth_m.shape}")
                print(f"📊 Depth min: {depth_m.min():.3f}m, max: {depth_m.max():.3f}m, mean: {depth_m.mean():.3f}m")
                first_frame = False

            # 2. Inférence YOLO (Détection)
            # verbose=False pour ne pas spammer le terminal
            results = self.model(rgb, verbose=False, conf=0.5)

            # 3. Traitement & Affichage
            display_frame = rgb.copy()
            
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Coordonnées
                    coords = box.xyxy[0].tolist()
                    cls_id = int(box.cls[0])
                    label = self.model.names[cls_id]
                    
                    # Calcul Distance (avec mapping RGB -> Depth)
                    dist = self.get_distance_at_center(depth_m, coords, rgb.shape)
                    
                    # Logique Visuelle (Danger)
                    x1, y1, x2, y2 = map(int, coords)
                    
                    if dist and dist < 1.0: # DANGER < 1 mètre
                        color = (0, 0, 255) # Rouge (BGR)
                        text = f"STOP! {label} {dist:.2f}m"
                        thickness = 3
                    else:
                        color = (0, 255, 0) # Vert
                        dist_str = f"{dist:.2f}m" if dist else "?"
                        text = f"{label} {dist_str}"
                        thickness = 2

                    # Dessin
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, thickness)
                    cv2.putText(display_frame, text, (x1, y1 - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # 4. Affichage Fenêtre
            cv2.imshow("iPhone 17 Pro - Robot Eye", display_frame)
            
            # Afficher aussi la depth map pour debug (normalisée pour visualisation)
            depth_vis = cv2.normalize(depth_m, None, 0, 255, cv2.NORM_MINMAX)
            depth_vis = depth_vis.astype(np.uint8)
            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_OCEAN)
            cv2.imshow("Depth Map", depth_vis)

            # Quitter avec 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                self.session.disconnect() # Propre
                cv2.destroyAllWindows()

if __name__ == '__main__':
    app = IOSRobotPerception()
    app.start()