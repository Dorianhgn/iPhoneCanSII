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


# ══════════════════════════════════════════════════════════════════════════════
#  HYPERPARAMÈTRES D'AFFICHAGE
# ══════════════════════════════════════════════════════════════════════════════

POINT_SIZE      = 2.0               # Taille des points
ARROW_LENGTH    = 0.1              # Longueur des flèches de normales (m)
ARROW_COLOR     = [0.2, 0.8, 1.0]  # Couleur des flèches (cyan)
ARROW_STEP      = 2                 # Sous-échantillonnage des flèches (1 sur N)
BG_COLOR        = [0.05, 0.05, 0.05]  # Couleur de fond
FRAME_SIZE      = 0.3               # Taille du repère XYZ en m (X=rouge, Y=vert, Z=bleu)

# ── Normales (utilisé seulement si le PLY n'en contient pas) ──────────────────
NORMAL_KNN      = 30
NORMAL_RADIUS   = 0.05


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTRUCTION DES FLÈCHES
# ══════════════════════════════════════════════════════════════════════════════

def build_normal_arrows(pcd, length=ARROW_LENGTH, color=ARROW_COLOR, step=ARROW_STEP):
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
    ls.colors = o3d.utility.Vector3dVector(
        np.tile(color, (n, 1)).astype(np.float64)
    )
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

    # Construire les flèches
    arrows = build_normal_arrows(pcd)

    if arrows is None:
        print("⚠️  Impossible de construire les flèches de normales.")

    # ── Fenêtre Open3D ────────────────────────────────────────────────────
    print("\nOuverture Open3D…")
    print("  [N] → switch nuage ↔ normales   [Fermer] → quitter")

    state = {"show_normals": False}

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

    vis.register_key_callback(78, toggle_normals)  # 78 = touche N
    vis.get_view_control().set_zoom(0.6)
    vis.run()
    vis.destroy_window()
    print("👋 Fenêtre fermée.")


# ── Entrée ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualiseur de nuage de points PLY avec normales (Open3D)"
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
        "--voxel-size", type=float, default=None,
        help="Si spécifié, applique un voxel downsampling avant affichage"
    )

    args = parser.parse_args()

    # Override si arguments CLI fournis
    POINT_SIZE   = args.point_size
    ARROW_LENGTH = args.arrow_length

    if args.voxel_size is not None:
        print(f"Pré-traitement : voxel downsampling ({args.voxel_size}m)…")
        pcd = o3d.io.read_point_cloud(args.ply_path)
        pcd = pcd.voxel_down_sample(args.voxel_size)
        tmp_path = args.ply_path.replace(".ply", "_downsampled.ply")
        o3d.io.write_point_cloud(tmp_path, pcd)
        view_pointcloud(tmp_path)
    else:
        view_pointcloud(args.ply_path)
