"""
segmentation.py — Wrapper YOLO segmentation + classes COCO standard
---------------------------------------------------------------------
Encapsule l'inférence YOLO-seg et mappe **directement** sur les 80 classes COCO,
chacune avec une couleur distincte générée deterministement (roue HSV, golden ratio).

Pas de regroupement en taxonomie personnalisée : chaque objet détecté conserve son
`coco_id` et reçoit une couleur unique. La distinction visuelle suffit pour le pitch.

Deux usages :

1. Aperçu live (opérateur) : `Segmenter.overlay()` dessine les masques colorés
   sur le flux RGB de `record_reconstruct.py` (touche `y`).
2. Bake offline : `Segmenter.class_map()` renvoie, pour une frame RGB, une carte
   de coco_ids par pixel (-1 = aucun). `record_reconstruct.py` projette ces ids
   sur le nuage 3D pour produire l'état couleur `segmentation`.

`ultralytics` / `torch` sont importés **paresseusement** : ce module s'importe
sans eux tant qu'on ne charge pas le modèle.
"""

import colorsys
import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


# ══════════════════════════════════════════════════════════════════════════════
#  80 CLASSES COCO STANDARD
# ══════════════════════════════════════════════════════════════════════════════

COCO_CLASSES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
    20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
    25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
    30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite",
    34: "baseball bat", 35: "baseball glove", 36: "skateboard",
    37: "surfboard", 38: "tennis racket", 39: "bottle", 40: "wine glass",
    41: "cup", 42: "fork", 43: "knife", 44: "spoon", 45: "bowl",
    46: "banana", 47: "apple", 48: "sandwich", 49: "orange", 50: "broccoli",
    51: "carrot", 52: "hot dog", 53: "pizza", 54: "donut", 55: "cake",
    56: "chair", 57: "couch", 58: "potted plant", 59: "bed",
    60: "dining table", 61: "toilet", 62: "tv", 63: "laptop", 64: "mouse",
    65: "remote", 66: "keyboard", 67: "cell phone", 68: "microwave",
    69: "oven", 70: "toaster", 71: "sink", 72: "refrigerator", 73: "book",
    74: "clock", 75: "vase", 76: "scissors", 77: "teddy bear",
    78: "hair drier", 79: "toothbrush",
}

# Gris neutre pour les points non segmentés
NEUTRAL_GRAY = [90, 95, 105]


def _coco_color(coco_id: int) -> list:
    """Couleur RGB 0-255 déterministe pour un coco_id (golden ratio hue spacing)."""
    h = (int(coco_id) * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.75, 0.90)
    return [int(r * 255), int(g * 255), int(b * 255)]


# LUT précalculée (0-79)
_COLOR_LUT = {cid: _coco_color(cid) for cid in COCO_CLASSES}


# ══════════════════════════════════════════════════════════════════════════════
#  TAXONOMY  (interface compatible avec scene_export.bake_scene)
# ══════════════════════════════════════════════════════════════════════════════

class Taxonomy:
    """Lookup COCO id → nom / couleur, sans regroupement."""

    def name(self, coco_id: int) -> str:
        return COCO_CLASSES.get(int(coco_id), f"obj_{coco_id}")

    def color(self, coco_id: int) -> list:
        return _COLOR_LUT.get(int(coco_id), _coco_color(coco_id))

    def color_lut(self, max_id: int = 79) -> np.ndarray:
        """Retourne un tableau (K,3) uint8 : lut[coco_id] = RGB."""
        K = max(max_id + 1, 80)
        lut = np.full((K, 3), 200, dtype=np.uint8)
        for cid in range(K):
            lut[cid] = _coco_color(cid)
        return lut


# Singleton par défaut utilisé par Segmenter
DEFAULT_TAXONOMY = Taxonomy()


# ══════════════════════════════════════════════════════════════════════════════
#  SEGMENTER
# ══════════════════════════════════════════════════════════════════════════════

class Segmenter:
    """
    Wrapper léger autour d'un modèle YOLO-seg Ultralytics.

    Le modèle (torch + ultralytics) n'est chargé qu'au premier appel utile
    (`infer`), pour que l'import du module reste gratuit.
    """

    def __init__(self, model_path="yolo26s-seg.pt", conf=0.4, device=None,
                 taxonomy=None):
        self.model_path = model_path
        self.conf = float(conf)
        self.device = device
        self.taxonomy = taxonomy if isinstance(taxonomy, Taxonomy) else DEFAULT_TAXONOMY
        self.model = None
        self._names = None

    # ── Chargement paresseux ──────────────────────────────────────────────────
    def _ensure_model(self):
        if self.model is not None:
            return
        from ultralytics import YOLO
        try:
            import torch
            if self.device is None:
                if torch.cuda.is_available():
                    self.device = "cuda"
                elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                    self.device = "mps"
                else:
                    self.device = "cpu"
        except Exception:
            self.device = self.device or "cpu"

        print(f"🧠 Chargement YOLO-seg : {self.model_path} (device={self.device})")
        try:
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
        except Exception as e:
            fallback = "yolov8s-seg.pt"
            print(f"⚠️  Échec chargement {self.model_path} ({e}) → fallback {fallback}")
            self.model = YOLO(fallback)
            self.model.to(self.device)
            self.model_path = fallback
        self._names = self.model.names

    # ── Inférence ─────────────────────────────────────────────────────────────
    def infer(self, rgb):
        """
        Lance la segmentation sur une image RGB (H,W,3 uint8).

        Retourne une liste d'instances :
            {coco_id, name, conf, polys: [ (M,2) float array en px image RGB ]}
        """
        self._ensure_model()
        results = self.model(rgb, conf=self.conf, verbose=False)
        out = []
        r = results[0]
        if r.masks is None or r.boxes is None:
            return out
        polys_all = r.masks.xy
        clss = r.boxes.cls.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()
        for poly, coco_id, cf in zip(polys_all, clss, confs):
            if poly is None or len(poly) < 3:
                continue
            out.append({
                "coco_id": int(coco_id),
                "name": COCO_CLASSES.get(int(coco_id), f"obj_{coco_id}"),
                "conf": float(cf),
                "polys": [np.asarray(poly, dtype=np.float32)],
            })
        return out

    # ── Carte de coco_ids par pixel (pour le bake) ────────────────────────────
    def class_map(self, instances, rgb_h, rgb_w):
        """
        Rastérise les instances en une carte (rgb_h, rgb_w) int16 de coco_ids.
        -1 = aucune classe. En cas de chevauchement, la plus haute confiance gagne.
        """
        cmap = np.full((rgb_h, rgb_w), -1, dtype=np.int16)
        if not instances or cv2 is None:
            return cmap
        for inst in sorted(instances, key=lambda d: d["conf"]):
            for poly in inst["polys"]:
                pts = np.round(poly).astype(np.int32).reshape(-1, 1, 2)
                cv2.fillPoly(cmap, [pts], int(inst["coco_id"]))
        return cmap

    # ── Overlay live (aperçu opérateur) ───────────────────────────────────────
    def overlay(self, bgr, instances, alpha=0.45):
        """
        Dessine les masques COCO colorés + libellés sur une image BGR (copie).
        """
        if cv2 is None:
            return bgr
        out = bgr.copy()
        if not instances:
            return out
        overlay_img = out.copy()
        for inst in sorted(instances, key=lambda d: d["conf"]):
            r, g, b = self.taxonomy.color(inst["coco_id"])
            color_bgr = (int(b), int(g), int(r))
            for poly in inst["polys"]:
                pts = np.round(poly).astype(np.int32).reshape(-1, 1, 2)
                cv2.fillPoly(overlay_img, [pts], color_bgr)
        cv2.addWeighted(overlay_img, alpha, out, 1 - alpha, 0, out)
        for inst in instances:
            r, g, b = self.taxonomy.color(inst["coco_id"])
            color_bgr = (int(b), int(g), int(r))
            for poly in inst["polys"]:
                pts = np.round(poly).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(out, [pts], True, color_bgr, 2)
            p0 = np.round(inst["polys"][0][0]).astype(int)
            label = f"{inst['name']} {inst['conf']:.2f}"
            cv2.putText(out, label, (int(p0[0]), int(p0[1]) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2)
        return out
