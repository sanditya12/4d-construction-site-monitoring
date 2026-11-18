import copy
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pycolmap
from evo.core import geometry, metrics, sync, trajectory
from evo.core.lie_algebra import se3
from evo.tools import file_interface

# ============================ CONFIG ============================
target_month = "december"                 # may | july | december
cam_name = "cam0"
max_time_diff = 0.01                 # [s] max timestamp mismatch for GT<->SfM matching
min_points3d = 20                   # drop registered images with fewer 3D points (0 = keep all)
trim_fraction = 0.10                 # fraction of worst frames dropped per trimming iteration
trim_iterations = 5                  # number of align -> trim -> refit rounds
heading_step = 10                    # frames used to compute heading direction
skip_start_s = 0

root = Path("/home/kanishka/4d_reconstruction/registration/floor1")
base_dir = root / f"outputs/nov_17/sfm_{target_month}_clean"
sfm_model_dir = base_dir / "sfm"
gt_file_path = root / f"data/ground_truth/{target_month}_gt.txt"
# ================================================================


def pick_model_dir(path: Path) -> Path:
    """If `path` contains numbered sub-models (0/, 1/, ...), use the one with most images."""
    subs = [p for p in path.iterdir() if p.is_dir() and p.name.isdigit()]
    if not subs:
        return path
    best = max(subs, key=lambda p: len(pycolmap.Reconstruction(str(p)).images))
    print(f"Multiple models found, using: {best}")
    return best


def parse_timestamp(image_name: str):
    """Take the trailing digit string of the file stem as the timestamp (ns or s)."""
    m = re.search(r"(\d+(?:\.\d+)?)$", Path(image_name).stem)
    if m is None:
        return None
    ts = float(m.group(1))
    if ts > 1e11:        # looks like nanoseconds
        ts /= 1e9
    return ts


def get_cam_from_world(image):
    cfw = image.cam_from_world
    return cfw() if callable(cfw) else cfw   # pycolmap API differs between versions


def n_points3d(image) -> int:
    """Number of 2D observations that have a triangulated 3D point (version-tolerant)."""
    try:
        v = image.num_points3D
        return int(v() if callable(v) else v)
    except AttributeError:
        return int(sum(p.has_point3D() for p in image.points2D))


def apply_umeyama(traj_est, traj_gt, correct_scale=True, mask=None):
    """Align a copy of traj_est to traj_gt. If mask is given, fit only on those frames."""
    out = copy.deepcopy(traj_est)
    if mask is None:
        r, t, s = out.align(traj_gt, correct_scale=correct_scale)
        return out, r, t, s
    r, t, s = geometry.umeyama_alignment(
        traj_est.positions_xyz[mask].T, traj_gt.positions_xyz[mask].T, correct_scale
    )
    if correct_scale:
        out.scale(s)
    out.transform(se3(r, t))
    return out, r, t, s


def ape_stats(traj_gt, traj_est):
    m = metrics.APE(metrics.PoseRelation.translation_part)
    m.process_data((traj_gt, traj_est))
    return m, {k.value: m.get_statistic(k) for k in (
        metrics.StatisticsType.rmse, metrics.StatisticsType.mean,
        metrics.StatisticsType.median, metrics.StatisticsType.std,
        metrics.StatisticsType.min, metrics.StatisticsType.max)}


def print_stats(title, stats):
    print(f"\n===== {title} =====")
    for k, v in stats.items():
        print(f"{k:>7}: {v:.4f} m")


def headings(xyz, k):
    d = xyz[k:, :2] - xyz[:-k, :2]
    return np.unwrap(np.arctan2(d[:, 1], d[:, 0]))


# ---------------------------------------------------------------
# 1. Load SfM poses (camera-to-world)
# ---------------------------------------------------------------
# ---------------------------------------------------------------
# 1. Load SfM poses and handle unregistered/missing frames
# ---------------------------------------------------------------
print("Loading COLMAP SfM model...")
model_dir = pick_model_dir(sfm_model_dir)
reconstruction = pycolmap.Reconstruction(str(model_dir))
print(f"Total registered images in reconstruction: {len(reconstruction.images)}")

# Build a lookup from image name -> reconstruction image object
reg_images_map = {img.name: img for _, img in reconstruction.images.items()}

# Find all expected images for this camera from your dataset directory or source list
# Assuming your images are stored in a folder like: root / "data/images/cam0" or similar
# Adjust this path to point to your actual image directory containing all 808 frames:
image_dir = root / f"data/images/{cam_name}" 
if not image_dir.exists():
    # Fallback: if folder structure differs, get timestamps from reconstruction sorted & filled,
    # or list all image filenames from your dataset source.
    all_image_names = sorted(reg_images_map.keys())
else:
    all_image_names = sorted([p.name for p in image_dir.glob("*") if cam_name in p.name])

print(f"Total expected frames in dataset for {cam_name}: {len(all_image_names)}")

est_timestamps, est_poses = [], []
n_unregistered, n_low_quality = 0, 0

for name in all_image_names:
    ts = parse_timestamp(name)
    if ts is None:
        continue
        
    if name not in reg_images_map:
        n_unregistered += 1
        # Option A: Append NaN matrix or skip. For interpolation, we can append None temporarily.
        est_timestamps.append(ts)
        est_poses.append(None)
        continue
        
    image = reg_images_map[name]
    if min_points3d > 0 and n_points3d(image) < min_points3d:
        n_low_quality += 1
        est_timestamps.append(ts)
        est_poses.append(None) # Treat low quality as missing to interpolate
        continue
        
    est_timestamps.append(ts)
    P = get_cam_from_world(image).inverse().matrix() # world_from_cam
    if P.shape == (3, 4):
        P = np.vstack([P, [0, 0, 0, 1]])
    est_poses.append(P)

est_timestamps = np.array(est_timestamps)

# Interpolate missing/None poses using neighboring valid poses
valid_mask = [p is not None for p in est_poses]
print(f"Registered & valid: {sum(valid_mask)}, Missing/Dropped: {len(est_poses) - sum(valid_mask)}")

valid_indices = np.where(valid_mask)[0]
all_indices = np.arange(len(est_poses))

# Simple linear interpolation for translation and SLERP/fallback for rotation
# Alternatively, evo's PoseTrajectory3D can handle interpolation if we pass only valid ones first, 
# then interpolate to full timestamps. Let's use evo's built-in interpolation:
valid_timestamps = est_timestamps[valid_mask]
valid_poses = [est_poses[i] for i in valid_indices]

temp_traj = trajectory.PoseTrajectory3D(poses_se3=valid_poses, timestamps=valid_timestamps)
# Create a full trajectory interpolated to all original timestamps
# (evo supports interpolation or you can evaluate directly on the valid subset)
traj_est = temp_traj # or interpolate if your evo version supports it

n_registered = temp_traj.num_poses
print(f"Using {n_registered} valid poses out of {len(all_image_names)} total frames.")

# ---------------------------------------------------------------
# 2. Load GT and associate by timestamp
# ---------------------------------------------------------------
print("\nLoading ground truth trajectory...")
traj_gt = file_interface.read_tum_trajectory_file(str(gt_file_path))
n_gt_total = traj_gt.num_poses

# drop the LiDAR initialisation period from the GT before matching
t0 = traj_gt.timestamps[0]
traj_gt.reduce_to_ids(np.where(traj_gt.timestamps - t0 >= skip_start_s)[0])
n_gt_total = traj_gt.num_poses

traj_est, traj_gt = sync.associate_trajectories(traj_est, traj_gt, max_diff=max_time_diff)

traj_est, traj_gt = sync.associate_trajectories(traj_est, traj_gt, max_diff=max_time_diff)
n_matched = traj_est.num_poses
print(f"Matched {n_matched} pairs | used {cam_name} poses: {n_registered} | GT poses: {n_gt_total}")
print(f"GT coverage by SfM: {100 * n_matched / n_gt_total:.1f}%")
if n_matched < 3:
    raise RuntimeError("Too few matched poses. Increase max_time_diff or check timestamps.")

t_rel = traj_gt.timestamps - traj_gt.timestamps[0]

# ---------------------------------------------------------------
# 3. Sim(3) alignment + APE
# ---------------------------------------------------------------
traj_est_al, R, t, s = apply_umeyama(traj_est, traj_gt, correct_scale=True)
ape_metric, stats = ape_stats(traj_gt, traj_est_al)
print(f"\nSim(3) scale factor: {s:.4f}")
print_stats("APE (translation), Sim(3) alignment on all frames", stats)

err_vec = traj_gt.positions_xyz - traj_est_al.positions_xyz
err_norm = np.linalg.norm(err_vec, axis=1)
print("Per-axis RMSE [x y z] (m):", np.round(np.sqrt((err_vec ** 2).mean(0)), 4))
print(f"XY-only RMSE: {np.sqrt((err_vec[:, :2] ** 2).sum(1).mean()):.4f} m")
print(f"Z-only RMSE:  {np.sqrt((err_vec[:, 2] ** 2).mean()):.4f} m")

# Heading comparison on the UNALIGNED trajectory: flat = rigid rotation, ramp/step = drift / bad join
k = heading_step
if n_matched > k + 1:
    dh = np.rad2deg(np.unwrap(headings(traj_est.positions_xyz, k) - headings(traj_gt.positions_xyz, k)))
    plt.figure(figsize=(8, 4))
    plt.plot(t_rel[:-k], dh)
    plt.xlabel("t [s]"); plt.ylabel("heading difference [deg]")
    plt.title("SfM vs GT heading (unaligned)")
    plt.grid(alpha=0.3)
    heading_path = base_dir / "heading_diff.png"
    plt.savefig(heading_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Heading difference plot saved to: {heading_path}")


# ---------------------------------------------------------------
# 5. RPE (local drift) + rotation APE
# ---------------------------------------------------------------
for rel, unit_label in ((metrics.PoseRelation.translation_part, "m"),
                        (metrics.PoseRelation.rotation_angle_deg, "deg")):
    rpe = metrics.RPE(rel, delta=1, delta_unit=metrics.Unit.frames, all_pairs=False)
    rpe.process_data((traj_gt, traj_est_al))
    print(f"RPE ({rel.value}, 1 frame): "
          f"RMSE {rpe.get_statistic(metrics.StatisticsType.rmse):.4f} {unit_label}, "
          f"median {rpe.get_statistic(metrics.StatisticsType.median):.4f} {unit_label}")

# Orientation error. Constant offset -> camera-frame convention mismatch; growing -> drift.
rot_ape = metrics.APE(metrics.PoseRelation.rotation_angle_deg)
rot_ape.process_data((traj_gt, traj_est_al))
print(f"Rotation APE: RMSE {rot_ape.get_statistic(metrics.StatisticsType.rmse):.2f} deg, "
      f"median {rot_ape.get_statistic(metrics.StatisticsType.median):.2f} deg")

# ---------------------------------------------------------------
# 6. Plots
# ---------------------------------------------------------------
gt_xyz = traj_gt.positions_xyz
es_xyz = traj_est_al.positions_xyz

# 6a. Diagnostics: top-down XY colored by error, Z vs time, error vs time
# 6a. Diagnostics: top-down XY colored by error (fixed scale 0 to 1m), Z vs time, error vs time
fig, ax = plt.subplots(1, 3, figsize=(20, 6))

ax[0].plot(gt_xyz[:, 0], gt_xyz[:, 1], "k--", lw=1, label="GT")
# Added vmin=0 and vmax=1 to fix the colorbar scale
sc = ax[0].scatter(
    es_xyz[:, 0], es_xyz[:, 1], 
    c=err_norm, s=6, cmap="viridis", 
    # vmin=0.0, vmax=1.0, 
    label="SfM (aligned)"
)
plt.colorbar(sc, ax=ax[0], label="APE [m]")
ax[0].set_aspect("equal")
ax[0].set_title("Top-down XY ")
ax[0].legend()
ax[0].set_xlabel("X [m]")
ax[0].set_ylabel("Y [m]")

# Z vs time plot
ax[1].plot(t_rel, gt_xyz[:, 2], "k--", label="GT")
ax[1].plot(t_rel, es_xyz[:, 2], "b", label="SfM (aligned)")
ax[1].set_title("Z vs time")
ax[1].set_xlabel("t [s]")
ax[1].set_ylabel("Z [m]")
ax[1].legend()

# Error vs time plot - fixed Y-axis limit from 0 to 1
ax[2].plot(t_rel, err_norm, label="APE")
ax[2].plot(t_rel, np.abs(err_vec[:, 2]), alpha=0.6, label="|Z error|")
ax[2].axhline(stats["rmse"], color="r", ls=":", label=f"RMSE {stats['rmse']:.2f} m")
ax[2].set_ylim(0.0, 1.0)  # Forces the Y-axis range to 0 to 1 meters
ax[2].set_title("Error vs time ")
ax[2].set_xlabel("t [s]")
ax[2].set_ylabel("[m]")
ax[2].legend()

diag_path = base_dir / "diagnostics.png"
plt.savefig(diag_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"\nDiagnostics plot saved to: {diag_path}")


# 6b. 3D plot with equal aspect (so Z is not visually exaggerated)
fig = plt.figure(figsize=(10, 8))
ax3 = fig.add_subplot(111, projection="3d")
ax3.plot(*gt_xyz.T, "k--", label="Ground Truth")
ax3.plot(*es_xyz.T, "b-", label="Estimated (Aligned)")
allp = np.vstack([gt_xyz, es_xyz])
ax3.set_box_aspect(np.ptp(allp, axis=0))
ax3.set_xlabel("X [m]"); ax3.set_ylabel("Y [m]"); ax3.set_zlabel("Z [m]")
ax3.set_title(f"SfM vs GT (APE RMSE {stats['rmse']:.3f} m)")
ax3.legend()
plot_path = base_dir / "trajectory_alignment_plot.png"
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Trajectory plot saved to: {plot_path}")

# ---------------------------------------------------------------
# 7. Save aligned trajectory (TUM)
# ---------------------------------------------------------------
aligned_output_path = base_dir / "estimated_aligned_trajectory.txt"
file_interface.write_tum_trajectory_file(str(aligned_output_path), traj_est_al)
print(f"Aligned estimated trajectory saved to: {aligned_output_path}")