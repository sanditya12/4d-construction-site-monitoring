"""
Align the query-session SfM map (December) to the reference-session frame (July)
using the poses produced by localization.py, then visualize both maps.

Inputs (all produced by sfm_extrinsics_2.py + localization.py):
    <ref>/sfm                      reference COLMAP model (July)
    <qry>/sfm                      query COLMAP model (December)
    loc_<qry>_in_<ref>/query_poses.txt   (names like  cam0/t3_xxx.jpg)

Outputs (in loc_<qry>_in_<ref>/alignment/):
    alignment.txt / alignment.npz   X_ref = s * R @ X_qry + t   (in the REF frame)
    aligned_map.html                plotly viewer (leveled for display only)
    points3D_aligned.bin            full-res query points in the REF frame

Run: python alignment.py
"""
import pickle
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import pycolmap

# ---------------------------------------------------------------------------
# Paths / settings
# ---------------------------------------------------------------------------
ROOT = Path("/home_domuser2/s84klaks/src/4d-construction-site-monitoring")
REF, QRY = "july", "december"      # reference map / session that was localized

ref_sfm = ROOT / f"data/outputs/sfm/sfm_{REF}_clean/sfm"
qry_sfm = ROOT / f"data/outputs/sfm/sfm_{QRY}_clean/sfm"
loc_dir = ROOT / f"data/outputs/sfm/loc_{QRY}_in_{REF}"
poses_f = loc_dir / "query_poses.txt"
logs_f = Path(str(poses_f) + "_logs.pkl")      # written by hloc localize_sfm
out_dir = loc_dir / "alignment"
out_dir.mkdir(parents=True, exist_ok=True)

WITH_SCALE = True        # False if both sfms use same scale
MIN_INLIERS = 50         # drop PnP poses with fewer inliers (needs the _logs.pkl)
RANSAC_ITERS = 2000
INLIER_FRAC = 0.05       # RANSAC inlier threshold as a fraction of the trajectory extent
MAX_PLOT_PTS = 100_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def qvec_to_rotmat(qw, qx, qy, qz):
    return np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw,     1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw,     2*qy*qz + 2*qx*qw,     1 - 2*qx**2 - 2*qy**2],
    ])


def umeyama(src, dst, with_scale=True):
    """Similarity transform with dst ~= s * R @ src + t."""
    n = src.shape[0]
    mu_src, mu_dst = src.mean(0), dst.mean(0)
    src_c, dst_c = src - mu_src, dst - mu_dst
    var_src = (src_c ** 2).sum() / n
    cov = (dst_c.T @ src_c) / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.diag([1.0, 1.0, np.linalg.det(U @ Vt)])
    R = U @ S @ Vt
    s = (D * np.diag(S)).sum() / var_src if with_scale else 1.0
    t = mu_dst - s * R @ mu_src
    return R, t, s


def apply_sim(R, t, s, pts):
    return s * (pts @ R.T) + t


def ransac_umeyama(src, dst, thresh, iters=2000, with_scale=True, seed=0):
    """Robust alignment: bad cross-season localizations don't corrupt the fit."""
    n = len(src)
    if n < 4:
        raise RuntimeError(f"Need >= 4 matched cameras, got {n}")
    rng = np.random.default_rng(seed)
    best_mask, best_key = None, (-1, np.inf)
    for _ in range(iters):
        idx = rng.choice(n, 4, replace=False)
        try:
            R, t, s = umeyama(src[idx], dst[idx], with_scale)
        except np.linalg.LinAlgError:
            continue
        if not np.isfinite(s) or s <= 0:
            continue
        err = np.linalg.norm(apply_sim(R, t, s, src) - dst, axis=1)
        mask = err < thresh
        key = (int(mask.sum()), -float(err[mask].mean()) if mask.any() else -np.inf)
        if key[0] > best_key[0] or (key[0] == best_key[0] and -key[1] < best_key[1]):
            best_mask, best_key = mask, (key[0], -key[1])
    if best_mask is None or best_mask.sum() < 4:
        raise RuntimeError("RANSAC found no consistent alignment.")
    R, t, s = umeyama(src[best_mask], dst[best_mask], with_scale)  # refit on inliers
    return R, t, s, best_mask


def load_poses(poses_path, logs_path, min_inliers):
    """name (e.g. 'cam0/t3_xxx.jpg') -> (R_cam_from_world, t). Full names are kept
    because cam0 and cam1 share the same basename."""
    inliers = {}
    if logs_path.exists():
        try:
            with open(logs_path, "rb") as f:
                loc = pickle.load(f).get("loc", {})
            inliers = {k: v.get("PnP_ret", {}).get("num_inliers") for k, v in loc.items()}
        except Exception as e:
            print(f"[poses] could not read {logs_path.name}: {e}")
    else:
        print(f"[poses] no {logs_path.name} -> skipping inlier filtering")

    poses, dropped = {}, 0
    with open(poses_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split()
            if len(p) < 8:
                continue
            name = p[0]
            n_in = inliers.get(name)
            if n_in is not None and n_in < min_inliers:
                dropped += 1
                continue
            qw, qx, qy, qz = map(float, p[1:5])
            poses[name] = (qvec_to_rotmat(qw, qx, qy, qz), np.array(list(map(float, p[5:8]))))
    print(f"[poses] kept {len(poses)}, dropped {dropped} (< {min_inliers} inliers)")
    if poses and "/" not in next(iter(poses)):
        raise RuntimeError(
            "Pose names have no 'cam0/'/'cam1/' prefix. Re-run localization.py "
            "(prepend_camera_name=True) so names match the SfM model."
        )
    return poses


# def model_points(model):
#     pts = list(model.points3D.values())
#     xyz = np.array([p.xyz for p in pts], dtype=np.float64).reshape(-1, 3)
#     rgb = np.array([p.color for p in pts], dtype=np.uint8).reshape(-1, 3)
#     return xyz, rgb


from scipy.spatial import cKDTree

def model_points(model, min_track=4, max_err=1.5, k=15, std_ratio=2.0):
    pts = [p for p in model.points3D.values()
           if p.track.length() >= min_track and p.error <= max_err]
    xyz = np.array([p.xyz for p in pts], dtype=np.float64).reshape(-1, 3)
    rgb = np.array([p.color for p in pts], dtype=np.uint8).reshape(-1, 3)

    # statistical outlier removal: drop points far from their neighbours
    d, _ = cKDTree(xyz).query(xyz, k=k + 1)
    md = d[:, 1:].mean(1)
    keep = md < md.mean() + std_ratio * md.std()
    print(f"  kept {keep.sum():,} / {len(model.points3D):,} points")
    return xyz[keep], rgb[keep]

# ----------------------------------

def estimate_up_vector_pca(points):
    c = points - points.mean(0)
    _, vecs = np.linalg.eigh(c.T @ c)
    up = vecs[:, 0]
    return up / np.linalg.norm(up)


def rotation_aligning_vectors(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), np.dot(a, b)
    s = np.linalg.norm(v)
    if s < 1e-8:
        return np.eye(3) if c > 0 else -np.eye(3) + 2 * np.outer(b, b)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / s**2)


def robust_bounds(*arrays, lo=1.0, hi=99.0, pad_frac=0.05):
    allp = np.concatenate(arrays, axis=0)
    lo_xyz, hi_xyz = np.percentile(allp, lo, axis=0), np.percentile(allp, hi, axis=0)
    pad = (hi_xyz - lo_xyz) * pad_frac
    lo_xyz, hi_xyz = lo_xyz - pad, hi_xyz + pad
    span = hi_xyz - lo_xyz
    aspect = dict(x=span[0] / span.max(), y=span[1] / span.max(), z=span[2] / span.max())
    return [lo_xyz[0], hi_xyz[0]], [lo_xyz[1], hi_xyz[1]], [lo_xyz[2], hi_xyz[2]], aspect


def to_rgb(cols_u8):
    return [f"rgb({r},{g},{b})" for r, g, b in cols_u8]


def subsample(xyz, rgb, n, seed=0):
    if len(xyz) <= n:
        return xyz, rgb
    idx = np.random.default_rng(seed).choice(len(xyz), n, replace=False)
    return xyz[idx], rgb[idx]


def write_points3D_binary(path, xyz, rgb_u8):
    """COLMAP points3D.bin with empty tracks (fine for Gaussian-splatting init)."""
    dt = np.dtype([("id", "<u8"), ("xyz", "<f8", 3), ("rgb", "u1", 3),
                   ("err", "<f8"), ("track_len", "<u8")])
    rec = np.zeros(len(xyz), dtype=dt)
    rec["id"] = np.arange(1, len(xyz) + 1)
    rec["xyz"], rec["rgb"] = xyz, rgb_u8
    with open(path, "wb") as f:
        f.write(np.array([len(xyz)], dtype="<u8").tobytes())
        f.write(rec.tobytes())
    print(f"  Wrote {len(xyz):,} points -> {path}")


# ---------------------------------------------------------------------------
# 1. Load poses and models
# ---------------------------------------------------------------------------
print("=== Loading localized poses ===")
poses = load_poses(poses_f, logs_f, MIN_INLIERS)

print("\n=== Loading models ===")
ref_model = pycolmap.Reconstruction(str(ref_sfm))
qry_model = pycolmap.Reconstruction(str(qry_sfm))
print(f"{REF}: {ref_model.num_reg_images()} images, {ref_model.num_points3D()} points")
print(f"{QRY}: {qry_model.num_reg_images()} images, {qry_model.num_points3D()} points")


# ---------------------------------------------------------------------------
# 2. Camera-center correspondences (match on FULL names: cam0/t3_xxx.jpg)
# ---------------------------------------------------------------------------
centers_qry, centers_ref, names = [], [], []
for im in qry_model.images.values():
    if im.name not in poses:
        continue
    R_cw, t_cw = poses[im.name]                       # cam_from_world in REF frame
    centers_qry.append(np.array(im.projection_center()))   # camera center in QRY frame
    centers_ref.append(-R_cw.T @ t_cw)                      # camera center in REF frame
    names.append(im.name)

centers_qry, centers_ref = np.array(centers_qry), np.array(centers_ref)
per_cam = {c: sum(n.startswith(c + "/") for n in names) for c in ("cam0", "cam1")}
print(f"\nMatched {len(names)} cameras {per_cam}")
if len(names) < 4:
    raise RuntimeError("Too few matches between the query SfM model and the localized poses.")


# ---------------------------------------------------------------------------
# 3. Robust similarity alignment QRY -> REF 
# ---------------------------------------------------------------------------
print("\n=== Aligning (RANSAC + Umeyama on camera centers) ===")
extent = np.linalg.norm(np.percentile(centers_ref, 95, 0) - np.percentile(centers_ref, 5, 0))
R_al, t_al, s_al, inl = ransac_umeyama(
    centers_qry, centers_ref, INLIER_FRAC * extent, RANSAC_ITERS, WITH_SCALE
)
err = np.linalg.norm(apply_sim(R_al, t_al, s_al, centers_qry) - centers_ref, axis=1)
print(f"Inliers: {inl.sum()}/{len(inl)}   threshold={INLIER_FRAC * extent:.3f}")
print(f"Inlier error  mean={err[inl].mean():.4f}  max={err[inl].max():.4f}")
print(f"All-camera error  median={np.median(err):.4f}")
print(f"Scale: {s_al:.4f}")


np.savez(out_dir / "alignment.npz", R=R_al, t=t_al, s=np.array(s_al))
with open(out_dir / "alignment.txt", "w") as f:
    f.write(f"# Similarity transform: X_{REF} = s * R @ X_{QRY} + t  (in the {REF} frame)\n\n")
    f.write(f"scale: {s_al:.10f}\n\nrotation:\n")
    for row in R_al:
        f.write("  " + "  ".join(f"{v:.10f}" for v in row) + "\n")
    f.write("\ntranslation:\n  " + "  ".join(f"{v:.10f}" for v in t_al) + "\n")
    f.write(f"\n# Cameras matched: {len(names)}, RANSAC inliers: {int(inl.sum())}\n")
    f.write(f"# Inlier error mean: {err[inl].mean():.6f}  max: {err[inl].max():.6f}\n")
print(f"Saved alignment -> {out_dir}")


print("Rotation matrix:")
print(R_al)

print("Rotation determinant:", np.linalg.det(R_al))

print("Scale:", s_al)

print("Inlier median error:", np.median(err[inl]))
print("Inlier mean error:", np.mean(err[inl]))
print("Inlier max error:", np.max(err[inl]))


# ---------------------------------------------------------------------------
# 4. Align point clouds
# ---------------------------------------------------------------------------
print("\n=== Aligning point clouds ===")
xyz_ref, rgb_ref = model_points(ref_model)
xyz_qry_raw, rgb_qry = model_points(qry_model)
xyz_qry = apply_sim(R_al, t_al, s_al, xyz_qry_raw)
print(f"{REF}: {len(xyz_ref):,} pts   {QRY} aligned: {len(xyz_qry):,} pts")


# ---------------------------------------------------------------------------
# 5. Export aligned query points as points3D.bin (Gaussian Splatting init)
# ---------------------------------------------------------------------------
print("\n=== Exporting aligned points3D.bin ===")
write_points3D_binary(out_dir / "points3D_aligned.bin", xyz_qry, rgb_qry)


# ---------------------------------------------------------------------------
# 6. Visualization (leveled for DISPLAY only; saved transforms stay in the REF frame)
# ---------------------------------------------------------------------------
print("\n=== Building visualization ===")
ref_centers = np.array([im.projection_center() for im in ref_model.images.values()])
up = estimate_up_vector_pca(ref_centers)
if up[1] < 0:
    up = -up
R_level = rotation_aligning_vectors(up, np.array([0.0, 1.0, 0.0]))
print(f"Display tilt correction: {np.degrees(np.arccos(np.clip(up[1], -1, 1))):.2f} deg "
      "(flip the sign check above if the scene is upside-down)")

p_ref, c_ref = subsample(xyz_ref, rgb_ref, MAX_PLOT_PTS)
p_qry, c_qry = subsample(xyz_qry, rgb_qry, MAX_PLOT_PTS)
p_raw, _ = subsample(xyz_qry_raw, rgb_qry, MAX_PLOT_PTS)
p_ref_v, p_qry_v = p_ref @ R_level.T, p_qry @ R_level.T

fig = go.Figure(data=[
    go.Scatter3d(x=p_ref_v[:, 0], y=p_ref_v[:, 1], z=p_ref_v[:, 2], mode="markers",
                 marker=dict(size=1, color=to_rgb(c_ref), opacity=0.6), name=f"{REF} map"),
    go.Scatter3d(x=p_qry_v[:, 0], y=p_qry_v[:, 1], z=p_qry_v[:, 2], mode="markers",
                 marker=dict(size=1, color=to_rgb(c_qry), opacity=0.6),
                 name=f"{QRY} map (aligned)"),
    go.Scatter3d(x=p_raw[:, 0], y=p_raw[:, 1], z=p_raw[:, 2], mode="markers",
                 marker=dict(size=1, color="lime", opacity=0.6),
                 name=f"{QRY} map (raw, unaligned)", visible="legendonly"),
])
xr, yr, zr, aspect = robust_bounds(p_ref_v, p_qry_v)   # raw cloud excluded from bounds
fig.update_layout(
    title=f"{REF} map and {QRY} map aligned to the {REF} frame",
    scene=dict(
        xaxis=dict(title="X", range=xr), yaxis=dict(title="Y (up)", range=yr),
        zaxis=dict(title="Z", range=zr), aspectmode="manual", aspectratio=aspect,
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.1), up=dict(x=0, y=1, z=0),
                    center=dict(x=0, y=0, z=0)),
    ),
    width=1200, height=800, margin=dict(l=0, r=0, t=40, b=0),
)
out_html = out_dir / "aligned_map.html"
fig.write_html(str(out_html))
print(f"Saved to {out_html}")