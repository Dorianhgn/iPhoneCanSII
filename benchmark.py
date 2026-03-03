"""
Benchmark de reconstruction 3D – Record3D
------------------------------------------
Compare plusieurs configurations de reconstruction sur le MÊME recording.
Produit un tableau comparatif avec FPS et métriques qualité.

Avantages :
  - Reproductible : un seul scan, N configs testées en boucle
  - Déployable sur Jetson sans iPhone (utiliser recording_path)
  - Génère results.md + results.csv pour archivage

Usage :
    python benchmark.py                           # utilise benchmark_config.json
    python benchmark.py ma_config.json            # config custom
    python benchmark.py config.json --save-ply    # override save_ply
    python benchmark.py config.json --recording recordings/scan.npz

Config JSON → voir benchmark_config.json pour l'exemple complet.

Sorties dans benchmarks/<timestamp>/ :
    results.md      tableau comparatif Markdown
    results.csv     même chose CSV
    <name>/         (si save_ply=true) config.txt + reconstructed.ply
"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import open3d as o3d

# Réutilise le module principal pour la reconstruction
import record_reconstruct as rr
from record_reconstruct import Record3DRecorder


# ══════════════════════════════════════════════════════════════════════════════
#  OVERRIDE DES HYPERPARAMÈTRES
# ══════════════════════════════════════════════════════════════════════════════

OVERRIDABLE_PARAMS = [
    "USE_TSDF", "USE_GPU",
    "FRAME_SKIP", "MIN_TRAVEL_DIST",
    "SUBSAMPLE", "VOXEL_SIZE",
    "TSDF_VOXEL_LENGTH", "TSDF_SDF_TRUNC", "TSDF_BLOCK_COUNT",
    "MAX_DEPTH", "CONFIDENCE_MIN",
    "OUTLIER_NB", "OUTLIER_STD",
    "NORMAL_KNN", "NORMAL_RADIUS",
]

def apply_config(cfg: dict):
    """Monkey-patch les globals du module rr avec les valeurs du dict config."""
    for key in OVERRIDABLE_PARAMS:
        if key in cfg:
            setattr(rr, key, cfg[key])


# ══════════════════════════════════════════════════════════════════════════════
#  FILTRAGE DES FRAMES
# ══════════════════════════════════════════════════════════════════════════════

def filter_frames(frames: list, frame_skip: int, min_travel: float) -> list:
    """
    Réapplique FRAME_SKIP et MIN_TRAVEL_DIST sur la liste brute.
    Permet de simuler différentes valeurs sur le même recording.
    """
    filtered = []
    last_pos = None
    for i, f in enumerate(frames):
        if i % max(frame_skip, 1) != 0:
            continue
        p   = f['pose']
        pos = np.array([p['tx'], p['ty'], p['tz']])
        if last_pos is not None and np.linalg.norm(pos - last_pos) < min_travel:
            continue
        last_pos = pos
        filtered.append(f)
    return filtered


# ══════════════════════════════════════════════════════════════════════════════
#  CAPTURE DEPUIS IPHONE
# ══════════════════════════════════════════════════════════════════════════════

def capture_frames(recording_path_to_save=None) -> list:
    """
    Lance le flux Record3D en mode benchmark (FRAME_SKIP=1 pour tout capturer).
    Retourne la liste complète des frames brutes.
    Sauvegarde dans recording_path_to_save si fourni.
    """
    old_skip  = rr.FRAME_SKIP
    old_dist  = rr.MIN_TRAVEL_DIST
    rr.FRAME_SKIP     = 1        # tout capturer pour avoir le max de données
    rr.MIN_TRAVEL_DIST = 0.001   # distance minimale quasi nulle

    recorder = Record3DRecorder()
    recorder.run()   # boucle interactive : ESPACE → start/stop   Q/ESC → quitter

    rr.FRAME_SKIP      = old_skip
    rr.MIN_TRAVEL_DIST = old_dist

    frames = recorder.recorded_frames

    if recording_path_to_save and frames:
        recorder.save_raw_recording(recording_path_to_save)

    return frames


# ══════════════════════════════════════════════════════════════════════════════
#  EXÉCUTION D'UNE CONFIG
# ══════════════════════════════════════════════════════════════════════════════

def run_one_config(cfg: dict, raw_frames: list, out_dir: str, save_ply: bool) -> dict:
    """
    Exécute reconstruction + normales pour un variant de config.
    Retourne un dict de métriques.
    """
    name = cfg.get('name', 'unnamed')
    print(f"\n{'═'*60}")
    print(f"  Config : {name}")
    print(f"{'═'*60}")

    # Override globals
    apply_config(cfg)

    # Filtrage des frames selon les params de cette config
    frame_skip = cfg.get('FRAME_SKIP',      rr.FRAME_SKIP)
    min_travel = cfg.get('MIN_TRAVEL_DIST', rr.MIN_TRAVEL_DIST)
    frames     = filter_frames(raw_frames, frame_skip, min_travel)
    n_frames   = len(frames)
    print(f"  Frames après filtrage (skip={frame_skip}, travel={min_travel}m) : {n_frames}")

    if n_frames < 2:
        print("  ❌ Pas assez de frames après filtrage.")
        return {
            'name': name, 'mode': '—', 'error': 'not_enough_frames',
            'n_frames_input': len(raw_frames), 'n_frames_used': n_frames,
            'n_points_raw': 0, 'n_points_final': 0,
            't_reconstruct': 0, 'fps_reconstruct': 0,
            't_normals': 0, 'pts_per_sec_normals': 0, 't_total': 0,
        }

    # Reconstruction
    recorder = Record3DRecorder()
    result = recorder.reconstruct(frames) if rr.USE_TSDF else recorder.reconstruct_fast(frames)

    if result is None or result[0] is None:
        print("  ❌ Reconstruction échouée.")
        return {
            'name': name, 'mode': '—', 'error': 'reconstruction_failed',
            'n_frames_input': len(raw_frames), 'n_frames_used': n_frames,
            'n_points_raw': 0, 'n_points_final': 0,
            't_reconstruct': 0, 'fps_reconstruct': 0,
            't_normals': 0, 'pts_per_sec_normals': 0, 't_total': 0,
        }

    pcd, stats      = result
    t_recon         = stats.get('t_reconstruct', 0)
    n_used          = stats.get('n_frames_used', n_frames)
    fps_recon       = n_used / t_recon if t_recon > 0 else 0
    n_pts_raw       = stats.get('n_raw_points', len(pcd.points))
    n_pts_final     = stats.get('n_final_points', len(pcd.points))

    # Normales
    pcd, t_normals      = recorder.compute_normals(pcd)
    n_pts               = len(pcd.points)
    pts_per_sec_normals = n_pts / t_normals if t_normals > 0 else 0

    t_total = t_recon + t_normals

    # Sauvegarde optionnelle
    if save_ply:
        cfg_dir  = os.path.join(out_dir, name)
        os.makedirs(cfg_dir, exist_ok=True)
        ply_path = os.path.join(cfg_dir, "reconstructed.ply")
        o3d.io.write_point_cloud(ply_path, pcd)
        with open(os.path.join(cfg_dir, "config.txt"), "w") as f:
            f.write(f"# Config benchmark : {name}\n")
            for k in OVERRIDABLE_PARAMS:
                f.write(f"{k} = {getattr(rr, k)}\n")
        print(f"  💾 PLY sauvegardé → {ply_path}")

    metrics = {
        'name':                 name,
        'mode':                 stats.get('mode', 'fast'),
        'error':                None,
        'n_frames_input':       len(raw_frames),
        'n_frames_used':        n_used,
        'n_points_raw':         n_pts_raw,
        'n_points_final':       n_pts_final,
        't_reconstruct':        t_recon,
        'fps_reconstruct':      fps_recon,
        't_normals':            t_normals,
        'pts_per_sec_normals':  pts_per_sec_normals,
        't_total':              t_total,
    }

    print(f"\n  ✅  {name}")
    print(f"      Reconstruction : {t_recon:.2f}s  →  {fps_recon:.2f} fps")
    print(f"      Normales       : {t_normals:.2f}s  →  {pts_per_sec_normals:,.0f} pts/s")
    print(f"      Points finaux  : {n_pts_final:,}")
    print(f"      Total          : {t_total:.2f}s")

    return metrics


# ══════════════════════════════════════════════════════════════════════════════
#  FORMATAGE DES RÉSULTATS
# ══════════════════════════════════════════════════════════════════════════════

COLUMNS = [
    # (header,           clé dans metrics,       format)
    ('Config',           'name',                 's'),
    ('Mode',             'mode',                 's'),
    ('Frames',           'n_frames_used',         'd'),
    ('Pts finaux',       'n_points_final',        ',d'),
    ('T recon (s)',      't_reconstruct',          '.2f'),
    ('FPS recon',        'fps_reconstruct',        '.2f'),
    ('T normals (s)',    't_normals',              '.2f'),
    ('Pts/s normals',   'pts_per_sec_normals',    ',.0f'),
    ('T total (s)',      't_total',                '.2f'),
]

def _cell(val, fmt):
    if val is None or val == '' :
        return '—'
    if isinstance(val, float):
        return format(val, fmt)
    if isinstance(val, int):
        return format(val, fmt)
    return str(val)

def format_table(results: list, fmt='md') -> str:
    rows = []
    for r in results:
        if r.get('error'):
            row = [r['name'], r.get('error', 'ERROR')] + ['—'] * (len(COLUMNS) - 2)
        else:
            row = [_cell(r.get(key), f) for _, key, f in COLUMNS]
        rows.append(row)

    headers = [h for h, _, _ in COLUMNS]

    if fmt == 'csv':
        lines = [','.join(headers)]
        for row in rows:
            lines.append(','.join(row))
        return '\n'.join(lines)

    # Markdown
    col_w = [
        max(len(headers[i]), max((len(r[i]) for r in rows), default=0))
        for i in range(len(COLUMNS))
    ]
    def pad(s, w): return s.ljust(w)
    hdr  = '| ' + ' | '.join(pad(h,   col_w[i]) for i, h        in enumerate(headers)) + ' |'
    sep  = '|-' + '-|-'.join('-' * col_w[i]      for i           in range(len(COLUMNS))) + '-|'
    data = ['| ' + ' | '.join(pad(r[i], col_w[i]) for i in range(len(COLUMNS))) + ' |'
            for r in rows]
    return '\n'.join([hdr, sep] + data)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Record3D – compare N configs sur le même recording"
    )
    parser.add_argument(
        'config', nargs='?', default='benchmark_config.json',
        help="Fichier de config JSON (défaut: benchmark_config.json)"
    )
    parser.add_argument(
        '--recording', default=None,
        help="Override recording_path depuis la config"
    )
    parser.add_argument(
        '--save-ply', action='store_true', default=False,
        help="Override save_ply=true depuis la CLI"
    )
    args = parser.parse_args()

    # ── Chargement config ─────────────────────────────────────────────────
    if not os.path.exists(args.config):
        print(f"❌ Config introuvable : {args.config}")
        print("   Génère-en une avec l'exemple benchmark_config.json fourni.")
        sys.exit(1)

    with open(args.config) as f:
        global_cfg = json.load(f)

    recording_path = args.recording or global_cfg.get('recording_path')
    save_ply       = args.save_ply or global_cfg.get('save_ply', False)
    output_dir     = global_cfg.get('output_dir', 'benchmarks')
    variant_cfgs   = global_cfg.get('configs', [])

    if not variant_cfgs:
        print("❌ Aucune config dans 'configs'. Vérifie le fichier JSON.")
        sys.exit(1)

    print(f"📋 Benchmark : {len(variant_cfgs)} config(s) à tester")
    print(f"   save_ply   = {save_ply}")
    print(f"   output_dir = {output_dir}")

    # ── Acquisition des frames ────────────────────────────────────────────
    if recording_path and os.path.exists(recording_path):
        print(f"\n📂 Chargement du recording : {recording_path}")
        raw_frames = Record3DRecorder.load_raw_recording(recording_path)
    else:
        if recording_path:
            print(f"\n📡 recording_path défini mais absent → capture depuis iPhone")
            print(f"   (sera sauvegardé dans : {recording_path})")
            os.makedirs(os.path.dirname(recording_path) or '.', exist_ok=True)
        else:
            print(f"\n📡 Pas de recording_path → capture depuis iPhone")
        print("   [ESPACE] démarrer/arrêter l'enregistrement   [Q/ESC] quitter → lancer le benchmark\n")
        raw_frames = capture_frames(recording_path_to_save=recording_path)

    if len(raw_frames) < 2:
        print("❌ Recording vide ou insuffisant (< 2 frames).")
        sys.exit(1)

    print(f"\n✅ {len(raw_frames)} frames brutes prêtes pour le benchmark")

    # ── Dossier de sortie ─────────────────────────────────────────────────
    ts      = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join(output_dir, ts)
    os.makedirs(out_dir, exist_ok=True)

    # ── Boucle sur les configs ────────────────────────────────────────────
    all_results = []
    for cfg in variant_cfgs:
        metrics = run_one_config(cfg, raw_frames, out_dir, save_ply)
        all_results.append(metrics)

    # ── Affichage + sauvegarde des résultats ──────────────────────────────
    print(f"\n\n{'═'*60}")
    print(f"  RÉSULTATS COMPARATIFS")
    print(f"{'═'*60}\n")

    md_table  = format_table(all_results, fmt='md')
    csv_table = format_table(all_results, fmt='csv')

    print(md_table)

    md_path  = os.path.join(out_dir, 'results.md')
    csv_path = os.path.join(out_dir, 'results.csv')

    with open(md_path, 'w') as f:
        f.write(f"# Benchmark 3D – {ts}\n\n")
        f.write(f"**Recording** : `{recording_path or 'capture live'}`  \n")
        f.write(f"**Frames brutes** : {len(raw_frames)}  \n\n")
        f.write(md_table + '\n')

    with open(csv_path, 'w') as f:
        f.write(csv_table + '\n')

    print(f"\n📁 Résultats → {out_dir}/")
    print(f"   results.md  — tableau Markdown")
    print(f"   results.csv — tableau CSV")
    if save_ply:
        print(f"   <config>/reconstructed.ply — nuages de points")


if __name__ == '__main__':
    main()
