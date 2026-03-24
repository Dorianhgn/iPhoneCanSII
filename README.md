# IOS VISION

**Python Version:** 3.11.14

This repository captures RGBD data with Record3D, reconstructs 3D clouds, and provides a small analysis stack around the resulting data.

Main scripts:

- [record_reconstruct.py](record_reconstruct.py): capture and reconstruct a point cloud
- [view_ply.py](view_ply.py): interactive viewer for `.ply` files
- [triplane_probe.py](triplane_probe.py): tri-plane coverage / density analysis on raw `.npz` recordings

## Quickstart

The recommended workflow is:

1. Capture once and save the raw recording.
2. Re-run reconstruction from the saved `.npz` as many times as needed.
3. Inspect the generated `.ply`.
4. Probe the raw recording with the tri-plane analyzer.

### 1) Capture and save a raw recording

```bash
python record_reconstruct.py --config config_record_reconstruct.yaml --save-npz
```

`--save-npz` is recommended if you want to:

- analyze the capture later,
- compare multiple configs on the same recording,
- re-run the pipeline without reconnecting the iPhone.

### 2) Replay the same recording with a new config

```bash
python record_reconstruct.py --from-npz logs/.../scan.npz -c logs/.../config_record_reconstruct.yaml
```

### 3) View the reconstructed PLY

```bash
python view_ply.py logs/2026-02-19_15-30-45/reconstructed.ply -c logs/.../config_record_reconstruct.yaml
```

`view_ply.py` now accepts `-c/--config` too, so a compatible YAML can drive the display side when it contains the matching keys.

### 4) Probe the raw recording

```bash
python triplane_probe.py logs/2026-03-23_15-03-06/scan.npz -c logs/.../config_record_reconstruct.yaml
```

## Config Model

The goal is to keep one YAML file usable across the pipeline wherever the parameters overlap.

Shared or reusable keys:

- `FRAME_SKIP`: temporal downsampling of frames
- `CONFIDENCE_MIN`: minimum LiDAR confidence kept during reconstruction / probing
- `NORMAL_KNN`: neighbor count for normal estimation
- `NORMAL_RADIUS`: search radius for normal estimation
- `NORMAL_VOXEL_SIZE`: voxel size used by `triplane_probe.py` for normal propagation
- `ARROW_ANGLE_DEG_THRESHOLD`: angular threshold used to highlight steep normals
- `ARROW_ANGLE_AXIS`: axis used as angular reference
- `ARROW_STEP`: subsampling factor for normal arrows / normal analysis

Script-specific keys:

- `record_reconstruct.py`
	- `USE_TSDF`, `USE_GPU`
	- `MIN_TRAVEL_DIST`
	- `SUBSAMPLE`, `VOXEL_SIZE`
	- `TSDF_VOXEL_LENGTH`, `TSDF_SDF_TRUNC`, `TSDF_BLOCK_COUNT`
	- `MAX_DEPTH`
	- `OUTLIER_NB`, `OUTLIER_STD`

- `view_ply.py`
	- `POINT_SIZE`
	- `ARROW_LENGTH`
	- `BG_COLOR`
	- `FRAME_SIZE`

- `triplane_probe.py`
	- `TRIPLANE_RES`
	- `SPATIAL_EXTENT`
	- `N_FRAMES`
	- `START_INDEX`

Important note: `MAX_DEPTH` and `SPATIAL_EXTENT` are intentionally different.

- `MAX_DEPTH` is a capture / reconstruction depth filter.
- `SPATIAL_EXTENT` is the tri-plane analysis window size.

## record_reconstruct.py

This script captures frames and reconstructs a point cloud. It also writes the raw recording when requested.

Example:

```bash
python record_reconstruct.py --config config_record_reconstruct.yaml --save-npz
```

Relevant flags:

- `-c`, `--config`: load a YAML config
- `--from-npz`: replay a saved recording
- `--save-npz`: save the raw recording for later reuse

The top of [record_reconstruct.py](record_reconstruct.py) documents the reconstruction defaults and what each one controls.

### What each major hparam means

- `FRAME_SKIP`: keep one frame every N during capture
- `MIN_TRAVEL_DIST`: minimum camera travel between retained frames
- `SUBSAMPLE`: spatial depth subsampling in fast mode
- `VOXEL_SIZE`: deduplication voxel size in fast mode
- `TSDF_VOXEL_LENGTH`: TSDF grid resolution
- `TSDF_SDF_TRUNC`: TSDF truncation distance
- `TSDF_BLOCK_COUNT`: TSDF preallocation budget
- `MAX_DEPTH`: maximum accepted depth from the sensor
- `CONFIDENCE_MIN`: minimum confidence kept
- `OUTLIER_NB`, `OUTLIER_STD`: post-filtering on reconstructed clouds
- `NORMAL_KNN`, `NORMAL_RADIUS`: normal estimation parameters

## view_ply.py

Interactive PLY viewer with normals overlay.

Example:

```bash
python view_ply.py logs/2026-02-19_15-30-45/reconstructed.ply -c logs/.../config_record_reconstruct.yaml
```

The display-side compatible keys are:

- `POINT_SIZE`
- `ARROW_LENGTH`
- `ARROW_ANGLE_DEG_THRESHOLD`
- `ARROW_ANGLE_AXIS`
- `ARROW_STEP`
- `NORMAL_KNN`
- `NORMAL_RADIUS`
- `BG_COLOR`
- `FRAME_SIZE`

Controls:

- `N`: toggle point cloud / normals
- `B`: toggle angular threshold mode
- close the window to quit

## triplane_probe.py

Tri-plane analysis for raw `.npz` recordings.

It projects the accumulated points onto three orthogonal planes:

- Top view (`ZX`)
- Front view (`XY`)
- Side view (`YZ`)

Example:

```bash
python triplane_probe.py logs/2026-03-23_15-03-06/scan.npz -c logs/.../config_record_reconstruct.yaml
```

Useful flags:

- `--n-frames`: number of frames accumulated around the pivot
- `--start-index`: pivot frame index
- `--res`: tri-plane resolution
- `--extent`: analysis window size in meters
- `--past-mode`: accumulate backward from the pivot instead of forward
- `--skip-normals`: disable normal estimation for speed
- `--no-interactive`: render a static snapshot instead of the interactive window

Controls in interactive mode:

- `LEFT/RIGHT`: move the pivot by 1 frame
- `SHIFT+LEFT/RIGHT`: move the pivot by 10 frames
- `UP/DOWN`: change the accumulated window size by 1
- `SHIFT+UP/DOWN`: change the window size by 5
- `SPACE`: switch between mono overlay and RGB mode

## Suggested workflow

1. Capture a raw scan once with `record_reconstruct.py --save-npz`.
2. Re-run reconstruction from the `.npz` with different YAML configs.
3. Inspect the output with `view_ply.py`.
4. Probe the raw recording with `triplane_probe.py`.

## One-config principle

The intent is for one YAML config to stay reusable across the pipeline wherever possible.

When a parameter is script-specific, it stays local to that script. When a parameter is shared, the same name is used everywhere.