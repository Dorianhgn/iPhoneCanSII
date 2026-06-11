"""
segmentation.py — Wrapper YOLO segmentation + taxonomie ICANSII
----------------------------------------------------------------
Encapsule l'inférence YOLO-seg (ex. yolo26s-seg) et la projette sur la
**taxonomie 9 classes ICANSII** utilisée par le pitch (cf. spec §4).

Deux usages :

1. Aperçu live (operateur) : `Segmenter.overlay()` dessine les masques colorés
   sur le flux RGB de `record_reconstruct.py` (touche `y`).
2. Bake offline : `Segmenter.class_map()` renvoie, pour une frame RGB, une carte
   de classes ICANSII par pixel (-1 = aucune). `record_reconstruct.py` projette
   ces classes sur le nuage 3D pour produire l'état couleur `segmentation`.

`ultralytics` / `torch` sont importés **paresseusement** : ce module s'importe
sans eux tant qu'on ne charge pas le modèle (utile pour les tests / le bake sans
segmentation).

⚠️  TAXONOMIE — À RELIRE
La taxonomie ci-dessous (DEFAULT_TAXONOMY) est un **défaut raisonnable** pour la
navigation urbaine d'un déficient visuel. Elle n'était définie nulle part dans le
repo ; ajuste les classes / couleurs / mapping COCO selon le vrai référentiel
ICANSII. Surchargeable via la clé YAML `SEG_TAXONOMY`.
"""

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - cv2 attendu dans le pipeline
    cv2 = None


# ══════════════════════════════════════════════════════════════════════════════
#  TAXONOMIE ICANSII (9 classes)  —  mapping depuis les classes COCO de YOLO
# ══════════════════════════════════════════════════════════════════════════════
# id    : identifiant ICANSII stable (utilisé dans scene.json / labels)
# name  : libellé FR affiché (labels flottants de l'état 5)
# color : couleur RGB 0-255 de la classe (état `segmentation`)
# coco  : ids de classes COCO (YOLO) regroupés dans cette classe ICANSII
#
# COCO (rappel) : 0 person · 1 bicycle · 2 car · 3 motorcycle · 5 bus · 6 train
#   7 truck · 9 traffic light · 10 fire hydrant · 11 stop sign · 12 parking meter
#   13 bench · 14 bird · 15 cat · 16 dog · 17 horse · 24 backpack · 25 umbrella
#   26 handbag · 28 suitcase · 56 chair · 57 couch · 58 potted plant · 60 table
DEFAULT_TAXONOMY = [
    {"id": 0, "name": "personne",      "color": [236,  72, 153], "coco": [0]},
    {"id": 1, "name": "vehicule",      "color": [ 59, 130, 246], "coco": [2, 5, 6, 7]},
    {"id": 2, "name": "deux-roues",    "color": [ 56, 189, 248], "coco": [1, 3]},
    {"id": 3, "name": "mobilier",      "color": [168,  85, 247], "coco": [13, 56, 57, 60]},
    {"id": 4, "name": "signalisation", "color": [250, 204,  21], "coco": [9, 10, 11, 12]},
    {"id": 5, "name": "animal",        "color": [ 34, 197,  94], "coco": [14, 15, 16, 17, 18, 19, 20, 21, 22, 23]},
    {"id": 6, "name": "bagage",        "color": [249, 115,  22], "coco": [24, 25, 26, 28]},
    {"id": 7, "name": "vegetation",    "color": [ 22, 101,  52], "coco": [58]},
    {"id": 8, "name": "autre",         "color": [148, 163, 184], "coco": []},
]


class Taxonomy:
    """Convertit DEFAULT_TAXONOMY (ou un override) en tables de lookup rapides."""

    def __init__(self, entries=None, include_other=False):
        self.entries = entries if entries is not None else DEFAULT_TAXONOMY
        self.include_other = include_other

        self.coco_to_id = {}
        self.id_to_name = {}
        self.id_to_color = {}
        self._other_id = None
        for e in self.entries:
            cid = int(e["id"])
            self.id_to_name[cid] = str(e["name"])
            self.id_to_color[cid] = [int(c) for c in e["color"]]
            for coco in e.get("coco", []):
                self.coco_to_id[int(coco)] = cid
            if str(e["name"]).lower() == "autre":
                self._other_id = cid

    def icansii_id(self, coco_id):
        """COCO id -> ICANSII id, ou None si non mappé (et 'autre' désactivé)."""
        cid = self.coco_to_id.get(int(coco_id))
        if cid is not None:
            return cid
        if self.include_other and self._other_id is not None:
            return self._other_id
        return None

    def color(self, icansii_id):
        return self.id_to_color.get(int(icansii_id), [148, 163, 184])

    def name(self, icansii_id):
        return self.id_to_name.get(int(icansii_id), "objet")

    def color_lut(self, max_id=None):
        """Retourne un tableau (K,3) uint8 : color_lut[icansii_id] = RGB."""
        ids = list(self.id_to_color.keys())
        K = (max(ids) + 1) if ids else 1
        if max_id is not None:
            K = max(K, max_id + 1)
        lut = np.full((K, 3), 200, dtype=np.uint8)
        for cid, col in self.id_to_color.items():
            lut[cid] = col
        return lut


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
        self.taxonomy = taxonomy if isinstance(taxonomy, Taxonomy) else Taxonomy(taxonomy)
        self.model = None
        self._names = None

    # ── Chargement paresseux ──────────────────────────────────────────────────
    def _ensure_model(self):
        if self.model is not None:
            return
        from ultralytics import YOLO  # import paresseux
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
            # Fallback : modèle seg standard si le modèle demandé est introuvable.
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
        Les masques sont renvoyés comme polygones en coordonnées image d'origine
        (alignés sur `rgb`), pour rastérisation exacte côté appelant.
        """
        self._ensure_model()
        results = self.model(rgb, conf=self.conf, verbose=False)
        out = []
        r = results[0]
        if r.masks is None or r.boxes is None:
            return out
        polys_all = r.masks.xy  # liste de (M,2) en coords image d'origine
        clss = r.boxes.cls.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()
        for poly, coco_id, cf in zip(polys_all, clss, confs):
            if poly is None or len(poly) < 3:
                continue
            out.append({
                "coco_id": int(coco_id),
                "name": self._names.get(int(coco_id), str(coco_id)) if isinstance(self._names, dict) else str(coco_id),
                "conf": float(cf),
                "polys": [np.asarray(poly, dtype=np.float32)],
            })
        return out

    # ── Carte de classes ICANSII par pixel (pour le bake) ─────────────────────
    def class_map(self, instances, rgb_h, rgb_w):
        """
        Rastérise les instances en une carte (rgb_h, rgb_w) int16 d'ids ICANSII.
        -1 = aucune classe. En cas de chevauchement, la plus haute confiance gagne.
        """
        cmap = np.full((rgb_h, rgb_w), -1, dtype=np.int16)
        if not instances or cv2 is None:
            return cmap
        # Dessiner du moins confiant au plus confiant → le plus confiant écrase.
        for inst in sorted(instances, key=lambda d: d["conf"]):
            icid = self.taxonomy.icansii_id(inst["coco_id"])
            if icid is None:
                continue
            for poly in inst["polys"]:
                pts = np.round(poly).astype(np.int32).reshape(-1, 1, 2)
                cv2.fillPoly(cmap, [pts], int(icid))
        return cmap

    # ── Overlay live (aperçu opérateur) ───────────────────────────────────────
    def overlay(self, bgr, instances, alpha=0.45):
        """
        Dessine les masques ICANSII colorés + libellés sur une image BGR (in-place
        sur une copie). Retourne l'image annotée. `bgr` n'est pas modifié.
        """
        if cv2 is None:
            return bgr
        out = bgr.copy()
        if not instances:
            return out
        overlay = out.copy()
        for inst in sorted(instances, key=lambda d: d["conf"]):
            icid = self.taxonomy.icansii_id(inst["coco_id"])
            if icid is None:
                continue
            r, g, b = self.taxonomy.color(icid)
            color_bgr = (int(b), int(g), int(r))
            for poly in inst["polys"]:
                pts = np.round(poly).astype(np.int32).reshape(-1, 1, 2)
                cv2.fillPoly(overlay, [pts], color_bgr)
        cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, out)
        # Contours + libellés par-dessus
        for inst in instances:
            icid = self.taxonomy.icansii_id(inst["coco_id"])
            if icid is None:
                continue
            r, g, b = self.taxonomy.color(icid)
            color_bgr = (int(b), int(g), int(r))
            for poly in inst["polys"]:
                pts = np.round(poly).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(out, [pts], True, color_bgr, 2)
            p0 = np.round(inst["polys"][0][0]).astype(int)
            label = f"{self.taxonomy.name(icid)} {inst['conf']:.2f}"
            cv2.putText(out, label, (int(p0[0]), int(p0[1]) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2)
        return out
