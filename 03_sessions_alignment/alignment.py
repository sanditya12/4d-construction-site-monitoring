"""
Align t2 SfM to t1 frame using rotational averaging, then visualize both
point clouds together in plotly (opens in browser).

Run: python3 code/align_and_visualize.py
"""

from pathlib import Path
import numpy as np
import pycolmap
import open3d as o3d
import struct
import plotly.graph_objects as go

# --- Paths ---
root       = Path("/home/kanishka/4d_reconstruction/registration")
t1_sfm     = root / "data/kitchen_sfm_t1"
# t1_sfm     = root / "outputs/t1_sfm/sfm"
# t2_sfm     = root / "outputs/t2_sfm/sfm"
t2_sfm     = root / "data/t2_sfm"
t2_poses_f = root / "outputs/t2_localize/t2_poses.txt"
t1_ply     = root / "outputs/out1/t1_sfm/t1_map.ply"
t2_ply     = root / "outputs/out1/t2_sfm/t2_map.ply"
out_html   = root / "outputs/out1/fixed_aligned_map1.html"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def qvec_to_rotmat(qw, qx, qy, qz):
    return np.array([
        [1 - 2*qy**2 - 2*qz**2,  2*qx*qy - 2*qz*qw,  2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw,  1 - 2*qx**2 - 2*qz**2,  2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw,  2*qy*qz + 2*qx*qw,  1 - 2*qx**2 - 2*qy**2],
    ])


def geodesic_distance(R1, R2):
    R_rel     = R1.T @ R2
    cos_angle = np.clip((np.trace(R_rel) - 1) / 2, -1, 1)
    return np.arccos(cos_angle)


# def rotational_average(rotations, weights=None, max_iter=100, tol=1e-8):
#     n       = len(rotations)
#     weights = np.ones(n) / n if weights is None else np.array(weights) / np.sum(weights)
#     R_mean  = rotations[0].copy()

#     for _ in range(max_iter):
#         log_sum = np.zeros((3, 3))
#         for R, w in zip(rotations, weights):
#             log_sum += w * logm(R_mean.T @ R)
#         if np.linalg.norm(log_sum) < tol:
#             break
#         R_mean = R_mean @ expm(log_sum)
#         U, _, Vt = np.linalg.svd(R_mean)
#         R_mean = U @ Vt

#     residuals = np.degrees([geodesic_distance(R_mean, R) for R in rotations])
#     return R_mean, residuals


def umeyama(src, dst):
    """
    Similarity transform (scale, rotation, translation) mapping src -> dst.
    Solves R, t, s jointly via SVD — handles large rotations correctly.
    Returns R (3x3), t (3,), s (scalar)
    """
    n        = src.shape[0]
    mu_src   = src.mean(axis=0)
    mu_dst   = dst.mean(axis=0)
    src_c    = src - mu_src
    dst_c    = dst - mu_dst
    var_src  = (src_c ** 2).sum() / n
    cov      = (dst_c.T @ src_c) / n
    U, D, Vt = np.linalg.svd(cov)
    S        = np.diag([1.0, 1.0, np.linalg.det(U @ Vt)])  
    R        = U @ S @ Vt
    s        = (D * np.diag(S)).sum() / var_src
    t        = mu_dst - s * R @ mu_src
    return R, t, s


def export_ply_from_colmap(sfm_dir: Path, out_ply: Path):
    model  = pycolmap.Reconstruction(str(sfm_dir))
    xyzs   = np.array([p.xyz for p in model.points3D.values()])
    colors = np.array([p.color / 255.0 for p in model.points3D.values()])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyzs)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(str(out_ply), pcd)
    print(f"  Exported {len(xyzs)} points to {out_ply.name}")


def load_ply(ply_path: Path, max_pts=100_000):
    pcd  = o3d.io.read_point_cloud(str(ply_path))
    pts  = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors)
    if len(pts) > max_pts:
        idx  = np.random.choice(len(pts), max_pts, replace=False)
        pts  = pts[idx]
        cols = cols[idx]
    return pts, cols


def to_rgb(cols):
    return ["rgb({},{},{})".format(int(r*255), int(g*255), int(b*255))
            for r, g, b in cols]


def estimate_up_vector_pca(points: np.ndarray) -> np.ndarray:
    """
    Estimate the 'up' direction as the normal to the dominant plane of
    motion/structure, via PCA: the eigenvector of smallest variance.
    """
    centered = points - points.mean(axis=0)
    cov = centered.T @ centered
    eigvals, eigvecs = np.linalg.eigh(cov)   # ascending order
    up = eigvecs[:, 0]                       # smallest-variance direction
    return up / np.linalg.norm(up)


def rotation_aligning_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation matrix R such that R @ a is parallel to b (both unit vectors)."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = np.dot(a, b)
    if s < 1e-8:
        # already parallel (or anti-parallel)
        return np.eye(3) if c > 0 else -np.eye(3) + 2 * np.outer(b, b)
    vx = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / s**2)


def robust_bounds(*point_arrays, lo=1.0, hi=99.0, pad_frac=0.05):
    """
    Bounding box to ignore outliers.
    Returns (xrange, yrange, zrange, aspectratio_dict).
    """
    all_pts = np.concatenate(point_arrays, axis=0)
    lo_xyz  = np.percentile(all_pts, lo, axis=0)
    hi_xyz  = np.percentile(all_pts, hi, axis=0)
    span    = hi_xyz - lo_xyz
    pad     = span * pad_frac
    lo_xyz -= pad
    hi_xyz += pad
    span    = hi_xyz - lo_xyz
    max_span = span.max()
    aspect = dict(x=span[0] / max_span, y=span[1] / max_span, z=span[2] / max_span)
    return (
        [lo_xyz[0], hi_xyz[0]],
        [lo_xyz[1], hi_xyz[1]],
        [lo_xyz[2], hi_xyz[2]],
        aspect,
    )


# ---------------------------------------------------------------------------
# 1. Export .ply files from sfm outputs
# ---------------------------------------------------------------------------
print("=== Exporting point clouds ===")
export_ply_from_colmap(t1_sfm, t1_ply)
export_ply_from_colmap(t2_sfm, t2_ply)


# ---------------------------------------------------------------------------
# 2. Load t2 poses (image basename -> (R, t) in t1 frame)
# ---------------------------------------------------------------------------
print("\n=== Loading t2 poses ===")
t2_poses = {}
with open(t2_poses_f) as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        basename        = parts[0].split("/")[-1]
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz      = map(float, parts[5:8])
        t2_poses[basename] = (qvec_to_rotmat(qw, qx, qy, qz), np.array([tx, ty, tz]))

print(f"Loaded {len(t2_poses)} t2 poses")
print(f"Sample pose keys: {list(t2_poses.keys())[:3]}")


# ---------------------------------------------------------------------------
# 3. Match cameras between t2 SfM and t2 poses
# ---------------------------------------------------------------------------
print("\n=== Matching cameras ===")
t2_model = pycolmap.Reconstruction(str(t2_sfm))
print(f"t2 model: {t2_model.num_reg_images()} images, {t2_model.num_points3D()} points")

centers_t2, centers_t1 = [], []

for im in t2_model.images.values():
    basename = im.name.split("/")[-1]
    if basename not in t2_poses:
        continue
    R_t1, t_t1 = t2_poses[basename]
    centers_t2.append(np.array(im.projection_center()))
    centers_t1.append(-R_t1.T @ t_t1)

centers_t2 = np.array(centers_t2)
centers_t1 = np.array(centers_t1)
print(f"Matched {len(centers_t2)} cameras")

if len(centers_t2) == 0:
    raise RuntimeError(
        "No cameras matched between t2 SfM and t2_poses.txt. "
        "Check that both use the same image basenames."
    )


# ---------------------------------------------------------------------------
# 3b. Estimate 'up' from t1's own camera trajectory and level the scene
# ---------------------------------------------------------------------------

print("\n=== Leveling scene (PCA up-vector from t1 camera centers) ===")
t1_model = pycolmap.Reconstruction(str(t1_sfm))
t1_centers = np.array([im.projection_center() for im in t1_model.images.values()])

up_raw = estimate_up_vector_pca(t1_centers)

if up_raw[1] < 0:
    up_raw = -up_raw

R_level = rotation_aligning_vectors(up_raw, np.array([0.0, 1.0, 0.0]))
tilt_deg = np.degrees(np.arccos(np.clip(np.dot(up_raw, [0, 1, 0]), -1, 1)))
print(f"Estimated tilt: {tilt_deg:.2f} deg -- correcting to align with +Y")
print("If the scene comes out upside-down, flip the sign check above.")

centers_t1 = (R_level @ centers_t1.T).T


# ---------------------------------------------------------------------------
# 5. compute transformation between camera poses
# ---------------------------------------------------------------------------

print("\n=== Computing alignment on camera poses (Umeyama) ===")
R_align, t_align, s_align = umeyama(centers_t2, centers_t1)

aligned = (s_align * (R_align @ centers_t2.T)).T + t_align
errors  = np.linalg.norm(aligned - centers_t1, axis=1)
print(f"Alignment error  mean={errors.mean():.4f}  max={errors.max():.4f}")
print(f"Scale: {s_align:.4f}")

# --- Save alignment transform ---
out_alignment_txt = root / "outputs/out1/alignment.txt"
out_alignment_npz = root / "outputs/out1/alignment.npz"

np.savez(str(out_alignment_npz), R=R_align, t=t_align, s=np.array(s_align))

with open(out_alignment_txt, "w") as f:
    f.write("# Similarity transform: X_t1 = s * R @ X_t2 + t\n\n")
    f.write(f"scale: {s_align:.10f}\n\n")
    f.write("rotation:\n")
    for row in R_align:
        f.write(f"  {row[0]:.10f}  {row[1]:.10f}  {row[2]:.10f}\n")
    f.write("\ntranslation:\n")
    f.write(f"  {t_align[0]:.10f}  {t_align[1]:.10f}  {t_align[2]:.10f}\n")
    f.write(f"\n# Matched cameras: {len(centers_t2)}\n")
    f.write(f"# Mean alignment error: {errors.mean():.6f}\n")
    f.write(f"# Max alignment error:  {errors.max():.6f}\n")

print(f"Saved alignment -> {out_alignment_txt}")
print(f"Saved alignment -> {out_alignment_npz}")

# ---------------------------------------------------------------------------
# 6. Load and align point clouds
# ---------------------------------------------------------------------------



print("\n=== Loading point clouds ===")
pts_t1, cols_t1     = load_ply(t1_ply)
pts_t2_raw, cols_t2 = load_ply(t2_ply)

pts_t2 = (s_align * (R_align @ pts_t2_raw.T)).T + t_align
print(f"t1: {len(pts_t1):,} pts   t2 aligned: {len(pts_t2):,} pts")


pts_t1 = (R_level @ pts_t1.T).T


# ---------------------------------------------------------------------------
# 7. Visualize — t1 pcl, t2 pcl (raw), and t2 pcl (aligned), fixed view
# ---------------------------------------------------------------------------
print("\n=== Building visualization ===")

# Trace order: [0] t1 map, [1] t2 map (aligned), [2] t2 map (raw/unaligned)
fig = go.Figure(data=[
    go.Scatter3d(
        x=pts_t1[:, 0], y=pts_t1[:, 1], z=pts_t1[:, 2],
        mode="markers",
        marker=dict(size=1, color=to_rgb(cols_t1) if len(cols_t1) else "steelblue", opacity=0.6),
        name="t1 map",
    ),
    go.Scatter3d(
        x=pts_t2[:, 0], y=pts_t2[:, 1], z=pts_t2[:, 2],
        mode="markers",
        marker=dict(size=1, color=to_rgb(cols_t2) if len(cols_t2) else "tomato", opacity=0.6),
        name="t2 map (aligned)",
    ),
    go.Scatter3d(
        x=pts_t2_raw[:, 0], y=pts_t2_raw[:, 1], z=pts_t2_raw[:, 2],
        mode="markers",
        marker=dict(size=1, color="lime", opacity=0.6),
        name="t2 map (raw, unaligned)",
        visible="legendonly",
    ),
])


# Apply the outlier mask
xr, yr, zr, aspect = robust_bounds(pts_t1, pts_t2, pts_t2_raw)

camera = dict(
    eye=dict(x=1.5, y=1.5, z=1.1),
    up=dict(x=0, y=1, z=0),
    center=dict(x=0, y=0, z=0),
)

fig.update_layout(
    title="t1 map, t2 map (raw), and t2 map (aligned to t1 frame)",
    scene=dict(
        xaxis=dict(title="X", range=xr),
        yaxis=dict(title="Y (up)", range=yr),
        zaxis=dict(title="Z", range=zr),
        aspectmode="manual",
        aspectratio=aspect,
        camera=camera,
    ),
    width=1200, height=800,
    margin=dict(l=0, r=0, t=40, b=0),
)

fig.write_html(str(out_html))
print(f"\nSaved to {out_html}")
print(f"Open with: xdg-open {out_html}")



print("t1 bbox:", pts_t1.min(axis=0).round(2), pts_t1.max(axis=0).round(2))
print("t2 bbox (raw):", pts_t2_raw.min(axis=0).round(2), pts_t2_raw.max(axis=0).round(2))
print("t2 bbox (aligned):", pts_t2.min(axis=0).round(2), pts_t2.max(axis=0).round(2))


# ---------------------------------------------------------------------------
# 8. Export aligned t2 points as COLMAP points3D.bin (Gaussian Splatting init)
# ---------------------------------------------------------------------------


t2_points3D_bin = root / "outputs/out1/t2_sfm/points3D_aligned.bin"


def write_points3D_binary(path: Path, xyz: np.ndarray, rgb: np.ndarray):
    rgb_u8 = np.clip(rgb * 255.0 if rgb.max() <= 1.0 else rgb, 0, 255).astype(np.uint8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(xyz)))
        for i in range(len(xyz)):
            f.write(struct.pack("<Q", i + 1))               # point3D_id
            f.write(struct.pack("<ddd", *xyz[i]))             # xyz, float64
            f.write(struct.pack("<BBB", *rgb_u8[i]))           # rgb, uint8
            f.write(struct.pack("<d", 0.0))                   # reprojection error
            f.write(struct.pack("<Q", 0))                     # track length (empty)
    print(f"  Wrote {len(xyz):,} points -> {path}")


print("\n=== Exporting aligned t2 points3D.bin (Gaussian Splatting init) ===")

# full resolution t2-pcl transformed through the same alignment computed above.
t2_full_pcd = o3d.io.read_point_cloud(str(t2_ply))
t2_full_xyz_raw = np.asarray(t2_full_pcd.points)
t2_full_rgb     = np.asarray(t2_full_pcd.colors)

t2_full_xyz_aligned = (s_align * (R_align @ t2_full_xyz_raw.T)).T + t_align

write_points3D_binary(t2_points3D_bin, t2_full_xyz_aligned, t2_full_rgb)