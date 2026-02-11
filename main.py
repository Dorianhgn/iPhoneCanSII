import cv2
import numpy as np
from record3d import Record3DStream
from threading import Event
from ultralytics import YOLO
import torch
from collections import deque
import time
import os
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

class IOSRobotPerception:
    def __init__(self):
        self.event = Event()
        self.session = None
        self.running = True
        
        # --- CONFIGURATION depuis .env ---
        yolo_version = os.getenv('YOLO_VERSION', '8')  # Default: YOLOv8
        tracker_type = os.getenv('TRACKER', 'bytetrack').strip('"')  # Default: bytetrack
        
        # --- CONFIGURATION IA ---
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        
        # Sélection du modèle selon la version
        if yolo_version == '26':
            model_name = "yolo26n.pt"
            print(f"🚀 Chargement de YOLO26 Nano sur : {device.upper()}")
        else:
            model_name = "yolov8n.pt"
            print(f"🚀 Chargement de YOLOv8 Nano sur : {device.upper()}")
        
        # Chargement du modèle avec fallback
        try:
            self.model = YOLO(model_name)
            print(f"✅ Modèle {model_name} chargé avec succès")
        except Exception as e:
            print(f"⚠️ Erreur lors du chargement de {model_name}: {e}")
            if yolo_version == '26':
                print("⚠️ Fallback sur YOLOv8")
                self.model = YOLO("yolov8n.pt")
                model_name = "yolov8n.pt"
            else:
                raise
            
        self.model.to(device)
        
        # Stocker la config
        self.tracker_config = f"{tracker_type}.yaml"
        self.model_name = model_name
        self.tracker_name = tracker_type.upper()
        
        print(f"🎯 Tracker sélectionné : {self.tracker_name}")

        # --- MÉMOIRE & LISSAGE ---
        self.track_history = {}
        self.last_seen_time = {}
        self.history_maxlen = 10
        self.memory_persistence = 2.0  # Secondes avant d'oublier un objet

    def on_new_frame(self):
        self.event.set()

    def on_stream_stopped(self):
        print('Stream stopped')

    def connect_to_device(self, dev_idx):
        print('🔗 Recherche des appareils connectés...')
        devs = Record3DStream.get_connected_devices()
        if not devs:
            print("❌ Aucun appareil trouvé. Vérifie le câble et l'app Record3D.")
            return False
            
        print(f'{len(devs)} appareil(s) trouvé(s)')
        dev = devs[dev_idx]
        self.session = Record3DStream()
        self.session.on_new_frame = self.on_new_frame
        self.session.on_stream_stopped = self.on_stream_stopped
        self.session.connect(dev)
        return True

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
        
        # Zone d'échantillonnage
        margin = 5 
        
        # Limites sécurisées
        x_min, x_max = max(0, cx - margin), min(depth_w, cx + margin)
        y_min, y_max = max(0, cy - margin), min(depth_h, cy + margin)
        
        roi = depth_map[y_min:y_max, x_min:x_max]
        
        # Filtrer les valeurs invalides
        valid_pixels = roi[(roi > 0.01) & (roi < 10.0)]
        
        if valid_pixels.size == 0:
            return None
        
        return np.median(valid_pixels)

    def start(self):
        print("🔗 En attente de connexion USB iPhone...")
        if self.connect_to_device(dev_idx=0):
            print("✅ Connecté ! Démarrage du stream...")
            self.loop()

    def loop(self):
        while self.running:
            self.event.wait()
            self.event.clear()

            # 1. Acquisition
            rgb = self.session.get_rgb_frame()
            depth_m = self.session.get_depth_frame()
            current_time = time.time()

            # 2. Tracking avec le tracker configuré
            # persist=True : Garde la mémoire des objets entre les frames
            results = self.model.track(
                rgb, 
                persist=True, 
                tracker=self.tracker_config,
                verbose=False, 
                conf=0.4,
                iou=0.5
            )

            display_frame = rgb.copy()
            seen_ids = set()

            if results[0].boxes.id is not None:
                # Récupération des tenseurs CPU
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().numpy()
                clss = results[0].boxes.cls.cpu().numpy()

                for box, track_id, cls in zip(boxes, track_ids, clss):
                    seen_ids.add(track_id)
                    self.last_seen_time[track_id] = current_time
                    label = self.model.names[int(cls)]
                    
                    # --- MESURE DISTANCE ---
                    raw_dist = self.get_distance_at_center(depth_m, box, rgb.shape)
                    
                    # --- LISSAGE TEMPOREL ---
                    if raw_dist is not None:
                        if track_id not in self.track_history:
                            self.track_history[track_id] = deque(maxlen=self.history_maxlen)
                        self.track_history[track_id].append(raw_dist)

                    final_dist = None
                    if track_id in self.track_history and len(self.track_history[track_id]) > 0:
                        final_dist = np.mean(self.track_history[track_id])
                    
                    # --- AFFICHAGE ---
                    x1, y1, x2, y2 = map(int, box)
                    
                    if final_dist:
                        # Logique couleur avec effet visuel critique
                        if final_dist < 1.0: 
                            color = (0, 0, 255)  # Rouge - DANGER
                            # Effet visuel X rouge sur toute l'image
                            cv2.line(display_frame, (0, 0), (rgb.shape[1], rgb.shape[0]), (0, 0, 255), 5)
                            cv2.line(display_frame, (rgb.shape[1], 0), (0, rgb.shape[0]), (0, 0, 255), 5)
                            text = f"#{track_id} {label} {final_dist:.2f}m (STOP!)"
                        elif final_dist < 2.5:
                            color = (0, 165, 255)  # Orange - Attention
                            text = f"#{track_id} {label} {final_dist:.2f}m"
                        else: 
                            color = (0, 255, 0)  # Vert - Safe
                            text = f"#{track_id} {label} {final_dist:.2f}m"
                    else:
                        color = (200, 200, 200)  # Gris - Pas de depth
                        text = f"#{track_id} {label}"

                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(display_frame, text, (x1, y1 - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # --- NETTOYAGE MÉMOIRE ---
            # Oublier les objets non vus depuis > memory_persistence secondes
            ids_to_forget = []
            for track_id in list(self.last_seen_time.keys()):
                if current_time - self.last_seen_time[track_id] > self.memory_persistence:
                    ids_to_forget.append(track_id)
            
            for k in ids_to_forget:
                if k in self.track_history:
                    del self.track_history[k]
                if k in self.last_seen_time:
                    del self.last_seen_time[k]

            # Affichage info système
            mode_text = f"Mode: {self.model_name.replace('.pt', '').upper()} + {self.tracker_name}"
            cv2.putText(display_frame, mode_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("iPhone 17 Pro - Robot Vision", display_frame)

            # Afficher aussi la depth map pour debug (normalisée pour visualisation)
            depth_vis = cv2.normalize(depth_m, None, 0, 255, cv2.NORM_MINMAX)
            depth_vis = depth_vis.astype(np.uint8)
            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_OCEAN)
            cv2.imshow("Depth Map", depth_vis)

            # Quitter avec 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                if self.session:
                    self.session.disconnect()
                cv2.destroyAllWindows()

if __name__ == '__main__':
    app = IOSRobotPerception()
    app.start()