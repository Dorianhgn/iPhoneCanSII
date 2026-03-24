"""
Visualiseur de nuage de points PLY (Open3D)
--------------------------------------------
Charge un fichier .ply et l'affiche dans une fenêtre Open3D interactive.
Les normales sont chargées depuis le PLY si présentes, sinon recalculées.

Usage :
    python view_ply.py <chemin_vers.ply>
    python view_ply.py logs/2026-02-19_15-30-45/reconstructed.ply

Contrôles :
    [N]       → switch nuage de points ↔ normales (flèches)
    [Fermer]  → quitter
"""

import argparse
import numpy as np
import open3d as o3d

try:
    import yaml
except ImportError:
    yaml = None


# ══════════════════════════════════════════════════════════════════════════════
#  HYPERPARAMÈTRES D'AFFICHAGE
# ══════════════════════════════════════════════════════════════════════════════

POINT_SIZE      = 2.0               # Taille des points
ARROW_LENGTH    = 0.1              # Longueur des flèches de normales (m)
ARROW_COLOR     = [0.2, 0.8, 1.0]  # Couleur des flèches (cyan)
ARROW_COLOR_THRESHOLD = [1.0, 0.0, 0.0]  # Couleur des normales au-dessus du seuil (rouge)
ARROW_ANGLE_DEG_THRESHOLD = 30.0   # Seuil angulaire en degres pour le surlignage
ARROW_ANGLE_AXIS = [0.0, 1.0, 0.0] # Axe de reference pour l'angle (ici Y)
ARROW_STEP      = 2                 # Sous-échantillonnage des flèches (1 sur N)
BG_COLOR        = [0.05, 0.05, 0.05]  # Couleur de fond
FRAME_SIZE      = 0.3               # Taille du repère XYZ en m (X=rouge, Y=vert, Z=bleu)

# ── Normales (utilisé seulement si le PLY n'en contient pas) ──────────────────
NORMAL_KNN      = 30
NORMAL_RADIUS   = 0.05


def _coerce_float(value, key):
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{key} doit être un float (reçu: {value})")


def _coerce_int(value, key, min_value=None):
    try:
        out = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{key} doit être un int (reçu: {value})")

    if min_value is not None and out < min_value:
        raise ValueError(f"{key} doit être >= {min_value} (reçu: {out})")
    return out


def _coerce_axis3(value, key):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{key} doit être une liste de 3 valeurs")
    axis = np.asarray(value, dtype=np.float64)
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        raise ValueError(f"{key} ne peut pas être nul")
    return axis.tolist()


def load_hyperparams_from_yaml(config_path):
    """Charge les réglages d'affichage depuis un YAML compatible reconstruction."""
    if not config_path:
        return

    if yaml is None:
        print("⚠️  PyYAML non installé. --config ignoré (installe: pip install pyyaml)")
        return

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except FileNotFoundError:
        print(f"⚠️  Fichier config introuvable: {config_path} (defaults conservés)")
        return
    except Exception as e:
        print(f"⚠️  Erreur lecture YAML ({config_path}): {e} (defaults conservés)")
        return

    if not isinstance(data, dict):
        print(f"⚠️  YAML invalide (mapping attendu): {config_path} (defaults conservés)")
        return

    mapping = {
        "POINT_SIZE": "POINT_SIZE",
        "ARROW_LENGTH": "ARROW_LENGTH",
        "ARROW_ANGLE_DEG_THRESHOLD": "ARROW_ANGLE_DEG_THRESHOLD",
        "ARROW_ANGLE_AXIS": "ARROW_ANGLE_AXIS",
        "ARROW_STEP": "ARROW_STEP",
        "NORMAL_KNN": "NORMAL_KNN",
        "NORMAL_RADIUS": "NORMAL_RADIUS",
        "BG_COLOR": "BG_COLOR",
        "FRAME_SIZE": "FRAME_SIZE",
    }

    updated = []
    for src_key, dst_key in mapping.items():
        if src_key not in data:
            continue

        raw = data[src_key]
        try:
            if dst_key in {"POINT_SIZE", "ARROW_LENGTH", "NORMAL_RADIUS", "FRAME_SIZE", "ARROW_ANGLE_DEG_THRESHOLD"}:
                value = _coerce_float(raw, src_key)
            elif dst_key in {"NORMAL_KNN", "ARROW_STEP"}:
                value = _coerce_int(raw, src_key, min_value=1)
            elif dst_key == "ARROW_ANGLE_AXIS":
                value = _coerce_axis3(raw, src_key)
            elif dst_key == "BG_COLOR":
                if not isinstance(raw, (list, tuple)) or len(raw) != 3:
                    raise ValueError(f"{src_key} doit être une liste de 3 valeurs")
                value = [float(v) for v in raw]
            else:
                value = raw
        except ValueError as e:
            print(f"⚠️  {e} -> clé ignorée")
            continue

        globals()[dst_key] = value
        updated.append((src_key, dst_key, value))

    if updated:
        print(f"✅ Config chargée depuis {config_path}")
        for src_key, dst_key, value in updated:
            print(f"   {src_key} -> {dst_key} = {value}")
    else:
        print(f"ℹ️  Aucune clé utile trouvée dans {config_path} (defaults conservés)")


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTRUCTION DES FLÈCHES
# ══════════════════════════════════════════════════════════════════════════════

def build_normal_arrows(
    pcd,
    length=ARROW_LENGTH,
    color=ARROW_COLOR,
    step=ARROW_STEP,
    angle_deg_threshold=None,
    color_threshold=ARROW_COLOR_THRESHOLD,
    angle_axis=ARROW_ANGLE_AXIS,
):
    """
    Construit un LineSet Open3D représentant les normales comme des segments.
    Chaque segment : point → point + normale * length
    """
    pts  = np.asarray(pcd.points)
    nrms = np.asarray(pcd.normals)

    if len(pts) == 0 or len(nrms) == 0:
        return None

    pts  = pts[::step]
    nrms = nrms[::step]

    tips = pts + nrms * length
    n    = len(pts)

    vertices = np.vstack([pts, tips])
    lines    = np.column_stack([np.arange(n), np.arange(n, 2 * n)])

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(vertices)
    ls.lines  = o3d.utility.Vector2iVector(lines)

    colors = np.tile(color, (n, 1)).astype(np.float64)
    if angle_deg_threshold is not None:
        axis = np.asarray(angle_axis, dtype=np.float64)
        axis_norm = np.linalg.norm(axis)
        if axis_norm > 0:
            axis = axis / axis_norm
            # Test de proximite angulaire via abs(cos(theta)) pour garder +/- axe.
            # L'utilisateur souhaite garder ce qui n'est *pas* autour de l'axe (angle > seuil)
            # donc on utilise abs(cos(theta)) < cos(seuil).
            cos_threshold = np.cos(np.deg2rad(angle_deg_threshold))
            mask = np.abs(np.sum(nrms * axis, axis=1)) < cos_threshold
            
            # Appliquer le filtrage des points si on est en mode threshold strict (on jette le reste)
            if color_threshold is None:
                # Mode filtrage: on ne garde que les flèches "rouges" (celles au delà du seuil)
                vertices = np.vstack([pts[mask], tips[mask]])
                n_filtered = np.sum(mask)
                lines = np.column_stack([np.arange(n_filtered), np.arange(n_filtered, 2 * n_filtered)])
                colors = np.tile(ARROW_COLOR_THRESHOLD, (n_filtered, 1)).astype(np.float64)
                
                ls.points = o3d.utility.Vector3dVector(vertices)
                ls.lines  = o3d.utility.Vector2iVector(lines)
                ls.colors = o3d.utility.Vector3dVector(colors)
                return ls
            else:
                colors[mask] = np.asarray(color_threshold, dtype=np.float64)

    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls


# ══════════════════════════════════════════════════════════════════════════════
#  VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def view_pointcloud(ply_path):
    print(f"Chargement : {ply_path}")
    pcd = o3d.io.read_point_cloud(ply_path)

    if pcd is None or len(pcd.points) == 0:
        print("❌ Nuage de points vide ou fichier invalide.")
        return

    n_points = len(pcd.points)
    print(f"  {n_points:,} points chargés")

    # Vérifier si les normales sont présentes
    if pcd.has_normals():
        print("  ✅ Normales présentes dans le PLY")
    else:
        print(f"  Normales absentes → calcul (knn={NORMAL_KNN}, radius={NORMAL_RADIUS})...")
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=NORMAL_RADIUS,
                max_nn=NORMAL_KNN
            )
        )
        pcd.orient_normals_towards_camera_location(np.array([0.0, 0.0, 0.0]))
        print("  ✅ Normales calculées")

    # Identifier les points correspondant au seuil
    nrms_full = np.asarray(pcd.normals)
    axis = np.asarray(ARROW_ANGLE_AXIS, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    cos_threshold = np.cos(np.deg2rad(ARROW_ANGLE_DEG_THRESHOLD))
    mask_full = np.abs(np.sum(nrms_full * axis, axis=1)) < cos_threshold
    
    # Créer le sous-nuage de points (filtré)
    pcd_filtered = pcd.select_by_index(np.where(mask_full)[0].tolist())

    # Construire les flèches
    arrows_default = build_normal_arrows(pcd)
    arrows_threshold = build_normal_arrows(
        pcd,
        angle_deg_threshold=ARROW_ANGLE_DEG_THRESHOLD,
        color_threshold=None, # None signifie qu'on va filtrer et ne garder que le rouge
    )

    if arrows_default is None:
        print("⚠️  Impossible de construire les flèches de normales.")

    # ── Fenêtre Open3D ────────────────────────────────────────────────────
    print("\nOuverture Open3D…")
    print("  [N] → switch nuage ↔ normales   [B] → basculer seuil angle   [Fermer] → quitter")

    state = {"show_normals": False, "threshold_mode": False}

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("View PLY – [N] normales", width=1280, height=720)

    opt = vis.get_render_option()
    opt.background_color = np.array(BG_COLOR)
    opt.point_size = POINT_SIZE

    vis.add_geometry(pcd)

    # Repère de coordonnées monde (X=rouge, Y=vert, Z=bleu)
    frame_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=FRAME_SIZE, origin=[0.0, 0.0, 0.0]
    )
    vis.add_geometry(frame_mesh)

    def active_arrows():
        return arrows_threshold if state["threshold_mode"] else arrows_default

    def active_pcd():
        return pcd_filtered if state["threshold_mode"] else pcd

    def toggle_normals(vis):
        state["show_normals"] = not state["show_normals"]
        if state["show_normals"]:
            vis.remove_geometry(active_pcd(), reset_bounding_box=False)
            if active_arrows() is not None:
                vis.add_geometry(active_arrows(), reset_bounding_box=False)
            print("  → Normales")
        else:
            if active_arrows() is not None:
                vis.remove_geometry(active_arrows(), reset_bounding_box=False)
            vis.add_geometry(active_pcd(), reset_bounding_box=False)
            print("  → Nuage de points")

    def toggle_threshold_mode(vis):
        state["threshold_mode"] = not state["threshold_mode"]
        mode = "ON" if state["threshold_mode"] else "OFF"
        print(f"  → Seuil angle ({ARROW_ANGLE_DEG_THRESHOLD}°): {mode}")

        if state["show_normals"]:
            # Remplacement des flèches
            old_arrows = arrows_default if state["threshold_mode"] else arrows_threshold
            new_arrows = active_arrows()
            if old_arrows is not None:
                vis.remove_geometry(old_arrows, reset_bounding_box=False)
            if new_arrows is not None:
                vis.add_geometry(new_arrows, reset_bounding_box=False)
        else:
            # Remplacement du nuage de points
            old_pcd = pcd if state["threshold_mode"] else pcd_filtered
            new_pcd = active_pcd()
            vis.remove_geometry(old_pcd, reset_bounding_box=False)
            vis.add_geometry(new_pcd, reset_bounding_box=False)

        return False

    vis.register_key_callback(78, toggle_normals)  # 78 = touche N
    vis.register_key_callback(66, toggle_threshold_mode)  # 66 = touche B
    vis.get_view_control().set_zoom(0.6)
    vis.run()
    vis.destroy_window()
    print("👋 Fenêtre fermée.")


# ── Entrée ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        help="Chemin vers un YAML de config compatible reconstruction",
    )
    pre_args, _ = pre_parser.parse_known_args()
    load_hyperparams_from_yaml(pre_args.config)

    parser = argparse.ArgumentParser(
        description="Visualiseur de nuage de points PLY avec normales (Open3D)",
        parents=[pre_parser],
    )
    parser.add_argument(
        "ply_path",
        type=str,
        help="Chemin vers le fichier .ply à visualiser"
    )
    parser.add_argument(
        "--point-size", type=float, default=POINT_SIZE,
        help=f"Taille des points (défaut: {POINT_SIZE})"
    )
    parser.add_argument(
        "--arrow-length", type=float, default=ARROW_LENGTH,
        help=f"Longueur des flèches de normales en m (défaut: {ARROW_LENGTH})"
    )
    parser.add_argument(
        "--arrow-angle-threshold", type=float, default=ARROW_ANGLE_DEG_THRESHOLD,
        help=(
            "Seuil angulaire en degres pour surligner certaines normales "
            f"en rouge avec B (defaut: {ARROW_ANGLE_DEG_THRESHOLD})"
        )
    )
    parser.add_argument(
        "--voxel-size", type=float, default=None,
        help="Si spécifié, applique un voxel downsampling avant affichage"
    )

    args = parser.parse_args()

    # Override si arguments CLI fournis
    POINT_SIZE   = args.point_size
    ARROW_LENGTH = args.arrow_length
    ARROW_ANGLE_DEG_THRESHOLD = args.arrow_angle_threshold

    if args.voxel_size is not None:
        print(f"Pré-traitement : voxel downsampling ({args.voxel_size}m)…")
        pcd = o3d.io.read_point_cloud(args.ply_path)
        pcd = pcd.voxel_down_sample(args.voxel_size)
        tmp_path = args.ply_path.replace(".ply", "_downsampled.ply")
        o3d.io.write_point_cloud(tmp_path, pcd)
        view_pointcloud(tmp_path)
    else:
        view_pointcloud(args.ply_path)
