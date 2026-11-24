#!/usr/bin/env python3
"""
Align a t2 SfM model to the t1 frame and export a consistent aligned model.

Pipeline
  1. Match t2 SfM cameras to their localized poses in the t1 frame
     (t2_poses.txt: `name qw qx qy qz tx ty tz`, cam_from_world in the t1 frame).
  2. Robust similarity fit (Umeyama + MSAC) on camera centres.
  3. Rotation-consistency check: for every pair, R1 = R2 @ R_align^T must hold.
     Pairs that violate it are dropped from the final refit, and a robust
     rotation average is reported as an independent estimate of R_align.
  4. Apply the similarity to the WHOLE t2 reconstruction (cameras + points)
     via pycolmap, write it as COLMAP binary (sparse/0 layout), and verify
     the result against the closed-form camera transform.
  5. Plotly visualization (levelled for display only; exports stay in t1 frame).

Run: python3 align_and_visualize.py [--help]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

ROOT = Path("/home/kanishka/4d_reconstruction/registration/floor1/outputs/nov_9")


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def qvec_to_rotmat(qw, qx, qy, qz) -> np.ndarray:
    q = np.array([qw, qx, qy, qz], dtype=float)
    q /= np.linalg.norm(q)
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw,     1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw,     2*qy*qz + 2*qx*qw,     1 - 2*qx**2 - 2*qy**2],
    ])


def rotmat_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    """Shepperd's method; only used as a pycolmap-API fallback."""
    tr = np.trace(R)
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        q = [(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s,
             (R[1, 0] - R[0, 1]) / s, 0.25 * s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k])
        q = [0.0, 0.0, 0.0, (R[k, j] - R[j, k]) / s]
        q[i] = 0.25 * s
        q[j] = (R[j, i] + R[i, j]) / s
        q[k] = (R[k, i] + R[i, k]) / s
    q = np.array(q)
    return q / np.linalg.norm(q)


def geodesic_angle(R1: np.ndarray, R2: np.ndarray) -> float:
    """Angle (radians) of the relative rotation R1^T R2."""
    c = np.clip((np.trace(R1.T @ R2) - 1) / 2, -1, 1)
    return float(np.arccos(c))


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
    """Similarity (R, t, s) minimising ||s R src + t - dst||. Needs >= 3 points.
    with_scale=False fixes s = 1 (rigid fit)."""
    n = len(src)
    if n < 3:
        raise ValueError("Umeyama needs at least 3 points")
    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    var_s = (sc ** 2).sum() / n
    if var_s < 1e-12:
        raise ValueError("Degenerate source points")
    U, D, Vt = np.linalg.svd(dc.T @ sc / n)
    S = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    R = U @ S @ Vt
    s = float((D * np.diag(S)).sum() / var_s) if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return R, t, s


def pairwise_scale(src, dst, n=20000, seed=0):
    """Rotation-free scale estimate: ratio of distances between random camera pairs."""
    rng = np.random.default_rng(seed)
    i, j = rng.integers(0, len(src), n), rng.integers(0, len(src), n)
    ds, dd = np.linalg.norm(src[i] - src[j], axis=1), np.linalg.norm(dst[i] - dst[j], axis=1)
    k = ds > 1e-6
    r = dd[k] / ds[k]
    return np.median(r), np.percentile(r, 10), np.percentile(r, 90)


def apply_sim(pts: np.ndarray, R, t, s) -> np.ndarray:
    return s * pts @ R.T + t


def umeyama_ransac(src, dst, thresh, iters=3000, sample_size=6, seed=0, with_scale=True):
    """MSAC-style robust similarity fit. Returns R, t, s, inlier_mask."""
    rng = np.random.default_rng(seed)
    n = len(src)
    if n < sample_size:
        raise RuntimeError(f"Need >= {sample_size} correspondences, got {n}")
    best_score, best_inl = np.inf, None
    for _ in range(iters):
        idx = rng.choice(n, sample_size, replace=False)
        # reject (near-)collinear samples: rotation about the line is unconstrained
        sv = np.linalg.svd(src[idx] - src[idx].mean(0), compute_uv=False)
        if sv[1] < 1e-2 * sv[0]:
            continue
        try:
            R, t, s = umeyama(src[idx], dst[idx], with_scale)
        except ValueError:
            continue
        if not np.isfinite(s) or s <= 0:
            continue
        err = np.linalg.norm(apply_sim(src, R, t, s) - dst, axis=1)
        score = np.minimum(err, thresh).sum()
        if score < best_score:
            best_score, best_inl = score, err < thresh
    if best_inl is None or best_inl.sum() < 4:
        raise RuntimeError("RANSAC found no consistent alignment "
                           "(check name matching / pose convention)")
    inl = best_inl
    for _ in range(10):  # iterative refit on inliers
        R, t, s = umeyama(src[inl], dst[inl], with_scale)
        new_inl = np.linalg.norm(apply_sim(src, R, t, s) - dst, axis=1) < thresh
        if new_inl.sum() < 4:
            raise RuntimeError("Refit collapsed; threshold too tight?")
        if np.array_equal(new_inl, inl):
            break
        inl = new_inl
    return R, t, s, inl


def rotation_average(Rs) -> np.ndarray:
    """Chordal L2 mean of rotations."""
    U, _, Vt = np.linalg.svd(np.sum(Rs, axis=0))
    return U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt


def robust_rotation_average(Rs, iters=5, min_keep_deg=3.0) -> np.ndarray:
    keep = np.ones(len(Rs), bool)
    R = rotation_average(Rs)
    for _ in range(iters):
        ang = np.degrees([geodesic_angle(R, Ri) for Ri in Rs])
        keep = ang <= max(3 * np.median(ang), min_keep_deg)
        if keep.sum() < 3:
            break
        R = rotation_average([Ri for Ri, k in zip(Rs, keep) if k])
    return R


def rotation_aligning_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation R with R @ a = b (unit vectors), including the anti-parallel case."""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v = np.cross(a, b)
    s, c = np.linalg.norm(v), float(a @ b)
    if s < 1e-8:
        if c > 0:
            return np.eye(3)
        # 180 deg about any axis perpendicular to a
        e = np.eye(3)[np.argmin(np.abs(a))]
        axis = np.cross(a, e)
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / s**2)


# ---------------------------------------------------------------------------
# pycolmap / I/O helpers
# ---------------------------------------------------------------------------

def cam_from_world_Rt(im):
    """(R, t) of cam_from_world, tolerant to pycolmap API versions."""
    cfw = im.cam_from_world
    if callable(cfw):
        cfw = cfw()
    rot = cfw.rotation
    R = np.asarray(rot.matrix() if callable(getattr(rot, "matrix", None)) else rot.matrix)
    return R, np.asarray(cfw.translation, dtype=float)


def camera_center(R, t):
    return -R.T @ t


def make_sim3d(R, t, s):
    import pycolmap
    try:
        rot = pycolmap.Rotation3d(R)
    except Exception:
        rot = pycolmap.Rotation3d(rotmat_to_quat_xyzw(R))
    return pycolmap.Sim3d(s, rot, t)


def load_t2_poses(path: Path, rename=None) -> dict:
    """name -> (R, t); cam_from_world in the t1 frame. rename=(old, new) edits the
    start of the basename (e.g. ('t2_', 't1_')) to match the SfM model's names."""
    poses = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split()
            if len(p) < 8:
                continue
            name = p[0]
            if rename:
                d, b = name.rsplit("/", 1) if "/" in name else ("", name)
                if b.startswith(rename[0]):
                    b = rename[1] + b[len(rename[0]):]
                name = f"{d}/{b}" if d else b
            poses[name] = (qvec_to_rotmat(*map(float, p[1:5])),
                           np.array(list(map(float, p[5:8]))))
    return poses


def points_of(rec):
    pts = list(rec.points3D.values())
    xyz = np.array([p.xyz for p in pts], dtype=float)
    rgb = np.array([p.color for p in pts], dtype=np.uint8)
    return xyz, rgb


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def match_cameras(t1_model, t2_model, poses, far_mult=3.0):
    t1_c = np.array([camera_center(*cam_from_world_Rt(im)) for im in t1_model.images.values()])
    t1_cent = t1_c.mean(0)
    t1_rad = np.linalg.norm(t1_c - t1_cent, axis=1).max()

    by_base = {}
    for k in poses:
        by_base.setdefault(k.split("/")[-1], []).append(k)

    names, c1, c2, R1, R2 = [], [], [], [], []
    n_missing = n_bad = n_ambig = 0
    for im in t2_model.images.values():
        key = im.name if im.name in poses else None
        if key is None:
            cands = by_base.get(im.name.split("/")[-1], [])
            if len(cands) == 1:
                key = cands[0]
            elif len(cands) > 1:
                n_ambig += 1
                continue
        if key is None:
            n_missing += 1
            continue
        Rp, tp = poses[key]
        Cp = camera_center(Rp, tp)
        if (not np.all(np.isfinite(Rp)) or not np.all(np.isfinite(tp))
                or abs(np.linalg.det(Rp) - 1) > 1e-2
                or np.linalg.norm(Cp - t1_cent) > far_mult * t1_rad):
            n_bad += 1
            continue
        Rm, tm = cam_from_world_Rt(im)
        names.append(im.name)
        c1.append(Cp); c2.append(camera_center(Rm, tm))
        R1.append(Rp);  R2.append(Rm)

    print(f"  t2 images: {len(t2_model.images)} | poses in file: {len(poses)} | "
          f"matched: {len(names)} | no pose: {n_missing} | ambiguous basename: {n_ambig} "
          f"| rejected (insane): {n_bad}")
    if len(names) < 10:
        raise RuntimeError("Too few matched cameras; check names (--rename-poses OLD NEW)")
    return names, np.array(c1), np.array(c2), np.array(R1), np.array(R2)


def align(names, c1, c2, R1, R2, thresh_frac, rot_thresh_deg, method="ransac", fix_scale=False):
    med, p10, p90 = pairwise_scale(c2, c1)
    print(f"  Pairwise distance ratio t1/t2 (rotation-free scale check): "
          f"median {med:.4f}  (10-90%: {p10:.3f}-{p90:.3f})")
    print(f"  t1 camera extent: {np.ptp(c1, axis=0).round(2)}   t2 extent: {np.ptp(c2, axis=0).round(2)}")
    ws = not fix_scale
    if method == "umeyama":
        # Plain least-squares fit on ALL matched centres: no outlier rejection.
        R, t, s = umeyama(c2, c1, ws)
        inl = np.ones(len(c1), bool)
        print(f"  Plain Umeyama on all {len(c1)} matched cameras  scale={s:.5f}")
    else:
        thresh = thresh_frac * np.linalg.norm(c1 - c1.mean(0), axis=1).max()
        print(f"  RANSAC threshold: {thresh:.4f} (= {thresh_frac:.0%} of t1 camera radius)")
        R, t, s, inl = umeyama_ransac(c2, c1, thresh, with_scale=ws)
        print(f"  Position inliers: {inl.sum()}/{len(c1)}  scale={s:.5f}")

    # Rotation consistency: R1 = R2 @ R^T  =>  R_i := R1^T R2 should equal R_align
    R_i = [a.T @ b for a, b in zip(R1, R2)]
    ang = np.degrees([geodesic_angle(R, Ri) for Ri in R_i])
    rot_ok = ang < rot_thresh_deg
    R_avg = robust_rotation_average(R_i)
    print(f"  Rotation-consistent pairs (<{rot_thresh_deg:g} deg): {rot_ok.sum()}/{len(ang)}; "
          f"median disagreement {np.median(ang):.2f} deg")
    print(f"  Rotational average vs Umeyama R: {np.degrees(geodesic_angle(R, R_avg)):.2f} deg")

    final = inl & rot_ok if method == "ransac" else inl
    if method == "umeyama":
        pass                                   # keep the plain fit as-is
    elif final.sum() >= 10:
        R, t, s = umeyama(c2[final], c1[final], ws)
        print(f"  Refit on {final.sum()} position+rotation inliers  scale={s:.5f}")
    else:
        print("  WARNING: few pairs pass the rotation check; keeping position-only fit. "
              "Check the pose convention in t2_poses.txt (cam_from_world assumed).")
        final = inl

    pos_err = np.linalg.norm(apply_sim(c2, R, t, s) - c1, axis=1)
    rot_err = np.degrees([geodesic_angle(a, b @ R.T) for a, b in zip(R1, R2)])
    print(f"  Final (inliers): pos err mean={pos_err[final].mean():.4f} max={pos_err[final].max():.4f}"
          f" | rot err mean={rot_err[final].mean():.2f} deg")
    print(f"  Final (all matched): pos err median={np.median(pos_err):.4f}, "
          f"rot err median={np.median(rot_err):.2f} deg")
    return R, t, s


def export_aligned_model(t2_model, c2_by_name, R, t, s, out_dir: Path):
    """Transform the whole t2 model (cameras + points) into the t1 frame and write it."""
    t2_model.transform(make_sim3d(R, t, s))

    # Self-check against the closed form: C_new = s R C_old + t
    errs = []
    for name, c_old in c2_by_name.items():
        im = next(i for i in t2_model.images.values() if i.name == name)
        errs.append(np.linalg.norm(camera_center(*cam_from_world_Rt(im)) - (s * R @ c_old + t)))
    if max(errs) > 1e-5 * max(1.0, abs(s)):
        raise RuntimeError(f"pycolmap transform disagrees with closed form (max {max(errs):.2e}); "
                           "check Sim3d/Rotation3d construction for your pycolmap version")
    out_dir.mkdir(parents=True, exist_ok=True)
    t2_model.write(str(out_dir))
    print(f"  Wrote aligned model ({len(t2_model.images)} images, "
          f"{len(t2_model.points3D)} points) -> {out_dir}")


def save_alignment(stem: Path, R, t, s, n_cams):
    """Save X_t1 = s * R @ X_t2 + t as .npz and .txt (stem gets the suffix)."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(stem) + ".npz", R=R, t=t, s=np.array(s))
    with open(str(stem) + ".txt", "w") as f:
        f.write("# Similarity transform: X_t1 = s * R @ X_t2 + t\n")
        f.write(f"scale: {s:.10f}\nrotation:\n")
        for row in R:
            f.write("  " + "  ".join(f"{v:.10f}" for v in row) + "\n")
        f.write("translation:\n  " + "  ".join(f"{v:.10f}" for v in t) + "\n")
        f.write(f"# Matched cameras: {n_cams}\n")
    print(f"  Saved alignment -> {stem}.npz / .txt")


def level_rotation(t1_model):
    """Display-only rotation taking t1's 'up' to +Y."""
    centers, ups = [], []
    for im in t1_model.images.values():
        R, t = cam_from_world_Rt(im)
        centers.append(camera_center(R, t))
        ups.append(-R[1])                      # COLMAP camera +Y points down
    centers = np.array(centers)
    c = centers - centers.mean(0)
    up = np.linalg.eigh(c.T @ c)[1][:, 0]      # normal of dominant plane of motion
    if up @ np.mean(ups, axis=0) < 0:          # sign from the cameras themselves
        up = -up
    print(f"  Estimated tilt: {np.degrees(np.arccos(np.clip(up[1], -1, 1))):.2f} deg")
    return rotation_aligning_vectors(up, np.array([0.0, 1.0, 0.0]))


def robust_bounds(*arrays, lo=1.0, hi=99.0, pad_frac=0.05):
    allp = np.concatenate(arrays)
    a, b = np.percentile(allp, lo, axis=0), np.percentile(allp, hi, axis=0)
    pad = (b - a) * pad_frac
    a, b = a - pad, b + pad
    span = b - a
    return [[a[i], b[i]] for i in range(3)], dict(zip("xyz", span / span.max()))


def visualize(clouds, out_html: Path):
    """clouds: list of (name, xyz, rgb_uint8_or_None, visible)."""
    import plotly.graph_objects as go
    traces = []
    for name, xyz, rgb, visible in clouds:
        color = (["rgb(%d,%d,%d)" % tuple(c) for c in rgb] if rgb is not None else "lime")
        traces.append(go.Scatter3d(
            x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="markers", name=name,
            marker=dict(size=1, color=color, opacity=0.6), visible=visible))
    (xr, yr, zr), aspect = robust_bounds(*[c[1] for c in clouds])
    fig = go.Figure(traces)
    fig.update_layout(
        title="t1 map, t2 map (aligned to t1), t2 map (raw)",
        scene=dict(xaxis=dict(title="X", range=xr), yaxis=dict(title="Y (up)", range=yr),
                   zaxis=dict(title="Z", range=zr), aspectmode="manual", aspectratio=aspect,
                   camera=dict(eye=dict(x=1.5, y=1.5, z=1.1), up=dict(x=0, y=1, z=0))),
        width=1200, height=800, margin=dict(l=0, r=0, t=40, b=0))
    fig.write_html(str(out_html), include_plotlyjs="cdn")
    print(f"  Saved {out_html}")


def subsample(xyz, rgb, n, rng):
    if len(xyz) <= n:
        return xyz, rgb
    idx = rng.choice(len(xyz), n, replace=False)
    return xyz[idx], rgb[idx]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--t1-sfm", type=Path, default=ROOT / "sfm_may_clean/sfm")
    ap.add_argument("--t2-sfm", type=Path, default=ROOT / "sfm_july_clean/sfm")
    ap.add_argument("--t2-poses", type=Path, default=ROOT / "loc_july_in_may/t2_poses.txt")
    ap.add_argument("--out-model", type=Path, default=ROOT / "sfm_july_clean/aligned/sparse/0",
                    help="output COLMAP model dir (cameras/images/points3D .bin)")
    ap.add_argument("--out-html", type=Path, default=ROOT / "aligned_july_may.html")
    ap.add_argument("--rename-poses", nargs=2, metavar=("OLD", "NEW"),
                    help="replace this prefix of pose-file basenames, e.g. t2_ t1_")
    ap.add_argument("--method", choices=["ransac", "umeyama"], default="ransac",
                    help="ransac: robust fit + rotation filter (default); "
                         "umeyama: plain least squares on all matched cameras")
    ap.add_argument("--fix-scale", action="store_true",
                    help="force scale = 1 (both maps are metric because of the calibrated rig)")
    ap.add_argument("--far-mult", type=float, default=3.0,
                    help="drop poses farther than this many t1-radii from the t1 centroid "
                         "(use inf to disable; only matters for --method umeyama)")
    ap.add_argument("--thresh-frac", type=float, default=0.05,
                    help="RANSAC threshold as a fraction of t1 camera radius")
    ap.add_argument("--rot-thresh-deg", type=float, default=5.0)
    ap.add_argument("--max-points", type=int, default=40_000, help="per trace, for the plot")
    ap.add_argument("--no-viz", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    import pycolmap
    rng = np.random.default_rng(0)

    print("=== Loading ===")
    t1 = pycolmap.Reconstruction(str(args.t1_sfm))
    t2 = pycolmap.Reconstruction(str(args.t2_sfm))
    poses = load_t2_poses(args.t2_poses, args.rename_poses)

    print("\n=== Matching cameras ===")
    names, c1, c2, R1, R2 = match_cameras(t1, t2, poses, args.far_mult)

    print("\n=== Alignment ===")
    R, t, s = align(names, c1, c2, R1, R2, args.thresh_frac, args.rot_thresh_deg, args.method, args.fix_scale)
    save_alignment(args.out_model.parent / f"alignment_{args.method}", R, t, s, len(names))

    raw_xyz, raw_rgb = points_of(t2)           # before transform
    t1_xyz, t1_rgb = points_of(t1)
    R_level = level_rotation(t1)

    print("\n=== Exporting aligned t2 model ===")
    export_aligned_model(t2, dict(zip(names, c2)), R, t, s, args.out_model)
    al_xyz, al_rgb = points_of(t2)             # after transform

    if not args.no_viz:
        print("\n=== Visualization ===")
        clouds = []
        for name, xyz, rgb, vis in [("t1 map", t1_xyz, t1_rgb, True),
                                    ("t2 map (aligned)", al_xyz, al_rgb, True),
                                    ("t2 map (raw)", raw_xyz, None, "legendonly")]:
            if rgb is None:
                xyz, _ = subsample(xyz, np.zeros(len(xyz)), args.max_points, rng)
            else:
                xyz, rgb = subsample(xyz, rgb, args.max_points, rng)
            # t1 and aligned t2 live in the t1 frame -> level them together; raw is left as-is
            if name != "t2 map (raw)":
                xyz = xyz @ R_level.T
            clouds.append((name, xyz, rgb, vis))
        visualize(clouds, args.out_html)


if __name__ == "__main__":
    main()