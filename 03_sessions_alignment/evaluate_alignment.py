"""
Evaluate the December->July alignment (alignment.npz from alignment.py) against
ground truth for both sessions.

GT format: TUM trajectory files (t x y z qx qy qz qw), cam0 positions, one per
session, matched to SfM cam0 images by the timestamp in the file name.

IMPORTANT: steps 3-4 assume BOTH GT files are in ONE common frame. If each month's
GT comes from its own run (own origin), set SHARED_GT_FRAME = False: only the
per-map checks (step 1-2) are then meaningful.

Idea
  1. Fit  July SfM frame -> GT frame   (Sim3, from July GT)    => July map error
  2. Fit  Dec  SfM frame -> GT frame   (Sim3, from Dec GT)     => Dec map error
  3. GT-derived Dec->July transform = (1)^-1 o (2); compare with the estimated one
  4. End-to-end: Dec cams -> estimated alignment -> July -> GT frame, compared to
     Dec GT. Dec GT is NOT used to build this path, so it is a held-out test.

Run: python eval_alignment.py
"""
import re
from pathlib import Path

import numpy as np
import pycolmap

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
ROOT = Path("/home_domuser2/s84klaks/src/4d-construction-site-monitoring")
REF, QRY = "july", "december"
cam_name = "cam0"                 # GT is cam0 positions
max_time_diff = 0.01              # [s] max timestamp mismatch for GT<->SfM matching
min_points3d = 20                 # ignore images with fewer 3D points (0 = keep all)
skip_start_s = 0                  # drop this many seconds from the start of the GT
SHARED_GT_FRAME = True            # False -> only evaluate each map vs its own GT
N_PTS = 20_000                    # points used for the map-level comparison

ref_sfm = ROOT / f"data/outputs/sfm/sfm_{REF}_clean/sfm"
qry_sfm = ROOT / f"data/outputs/sfm/sfm_{QRY}_clean/sfm"
align_npz = ROOT / f"data/outputs/sfm/loc_{QRY}_in_{REF}/alignment/alignment.npz"
GT_REF = ROOT / f"data/raw/ground_truth/{REF}_gt.txt"        
GT_QRY = ROOT / f"data/raw/ground_truth/{QRY}_gt.txt"        


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def umeyama(src, dst):
    """dst ~= s * R @ src + t"""
    n = len(src)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    var = (sc ** 2).sum() / n
    U, D, Vt = np.linalg.svd((dc.T @ sc) / n)
    S = np.diag([1.0, 1.0, np.linalg.det(U @ Vt)])
    R = U @ S @ Vt
    s = (D * np.diag(S)).sum() / var
    return R, mu_d - s * R @ mu_s, s


def apply(R, t, s, x):
    return s * (x @ R.T) + t


def angle_deg(R):
    return np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


def stats(label, e, unit="m"):
    e = np.asarray(e)
    print(f"  {label:<36} n={len(e):<5} mean={e.mean():.4f}  median={np.median(e):.4f}  "
          f"p90={np.percentile(e, 90):.4f}  max={e.max():.4f}  rmse={np.sqrt((e**2).mean()):.4f} {unit}")


def parse_timestamp(image_name):
    """Trailing digit string of the file stem (ns or s), e.g. cam0/t3_10172705874000.jpg"""
    m = re.search(r"(\d+(?:\.\d+)?)$", Path(image_name).stem)
    if m is None:
        return None
    ts = float(m.group(1))
    return ts / 1e9 if ts > 1e11 else ts


def n_points3d(image):
    try:
        v = image.num_points3D
        return int(v() if callable(v) else v)
    except AttributeError:
        return int(sum(p.has_point3D() for p in image.points2D))


def load_tum(path):
    d = np.loadtxt(str(path), comments="#")
    t, xyz = d[:, 0], d[:, 1:4]
    keep = t - t[0] >= skip_start_s
    order = np.argsort(t[keep])
    return t[keep][order], xyz[keep][order]


def paired_centers(model, gt_path):
    """SfM cam0 centers matched by timestamp to GT positions."""
    gt_t, gt_xyz = load_tum(gt_path)
    names, sfm_c, gt_c = [], [], []
    for im in model.images.values():
        if not im.name.startswith(cam_name + "/"):
            continue
        if min_points3d > 0 and n_points3d(im) < min_points3d:
            continue
        ts = parse_timestamp(im.name)
        if ts is None:
            continue
        i = np.searchsorted(gt_t, ts)
        cand = [j for j in (i - 1, i) if 0 <= j < len(gt_t)]
        j = min(cand, key=lambda k: abs(gt_t[k] - ts))
        if abs(gt_t[j] - ts) > max_time_diff:
            continue
        names.append(im.name)
        sfm_c.append(np.array(im.projection_center()))
        gt_c.append(gt_xyz[j])
    print(f"  {gt_path.name}: matched {len(names)} / {len(gt_t)} GT poses")
    if len(names) < 4:
        raise RuntimeError(f"Only {len(names)} matches for {gt_path}. Check timestamps / max_time_diff.")
    return np.array(sfm_c), np.array(gt_c)


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
ref_model = pycolmap.Reconstruction(str(ref_sfm))
qry_model = pycolmap.Reconstruction(str(qry_sfm))

print("=== Matching SfM cameras to GT by timestamp ===")
ref_c, ref_gt = paired_centers(ref_model, GT_REF)
qry_c, qry_gt = paired_centers(qry_model, GT_QRY)

print("\n=== GT frame sanity check (are both GT files in one frame?) ===")
for lbl, g in ((REF, ref_gt), (QRY, qry_gt)):
    print(f"  {lbl}: start {np.round(g[0], 2)}  bbox min {np.round(g.min(0), 2)}  max {np.round(g.max(0), 2)}")


# ---------------------------------------------------------------------------
# 1+2. Each map vs its own GT
# ---------------------------------------------------------------------------
R_rg, t_rg, s_rg = umeyama(ref_c, ref_gt)      # July SfM -> GT
R_qg, t_qg, s_qg = umeyama(qry_c, qry_gt)      # Dec  SfM -> GT

print("\n=== Each map vs its own ground truth (best Sim3 fit) ===")
print(f"  {REF}: scale to GT = {s_rg:.4f}     {QRY}: scale to GT = {s_qg:.4f}")
stats(f"{REF} map vs GT", np.linalg.norm(apply(R_rg, t_rg, s_rg, ref_c) - ref_gt, axis=1))
stats(f"{QRY} map vs GT", np.linalg.norm(apply(R_qg, t_qg, s_qg, qry_c) - qry_gt, axis=1))
print("  (These bound what is measurable: the alignment cannot beat map-vs-GT accuracy.)")

if not SHARED_GT_FRAME:
    print("\nSHARED_GT_FRAME = False -> skipping cross-session checks.")
    raise SystemExit

# ---------------------------------------------------------------------------
# 3. GT-derived Dec->July transform vs the estimated one
# ---------------------------------------------------------------------------
a = np.load(align_npz)
R_al, t_al, s_al = a["R"], a["t"], float(a["s"])      # Dec SfM -> July SfM (estimated)

R_gt = R_rg.T @ R_qg
s_gt = s_qg / s_rg
t_gt = (R_rg.T @ (t_qg - t_rg)) / s_rg

print("\n=== Estimated alignment vs GT-derived alignment (Dec -> July) ===")
print(f"  rotation error : {angle_deg(R_al.T @ R_gt):.3f} deg")
print(f"  scale          : est {s_al:.4f}  gt {s_gt:.4f}  ratio {s_al / s_gt:.4f}")

disp_cams = np.linalg.norm(apply(R_al, t_al, s_al, qry_c) - apply(R_gt, t_gt, s_gt, qry_c), axis=1)
stats("Dec camera displacement (est vs gt)", disp_cams)

pts = np.array([p.xyz for p in qry_model.points3D.values()])
if len(pts) > N_PTS:
    pts = pts[np.random.default_rng(0).choice(len(pts), N_PTS, replace=False)]
disp_pts = np.linalg.norm(apply(R_al, t_al, s_al, pts) - apply(R_gt, t_gt, s_gt, pts), axis=1)
stats("Dec map-point displacement", disp_pts)

# ---------------------------------------------------------------------------
# 4. End-to-end, held-out: Dec cams -> est. alignment -> July -> GT, vs Dec GT
# ---------------------------------------------------------------------------
print("\n=== End-to-end (Dec GT never used in this path) ===")
in_gt = apply(R_rg, t_rg, s_rg, apply(R_al, t_al, s_al, qry_c))
e2e = np.linalg.norm(in_gt - qry_gt, axis=1)
oracle = np.linalg.norm(apply(R_qg, t_qg, s_qg, qry_c) - qry_gt, axis=1)
stats("via estimated alignment", e2e)
stats("oracle (Dec fitted to its own GT)", oracle)
print(f"  extra error from alignment: median {np.median(e2e) - np.median(oracle):+.4f} m")


def rig_scale(sfm_dir, baseline=0.040041768):
    m = pycolmap.Reconstruction(str(sfm_dir))
    c = {im.name: np.array(im.projection_center()) for im in m.images.values()}
    d = [np.linalg.norm(c[n] - c[n.replace("cam0/", "cam1/", 1)])
         for n in c if n.startswith("cam0/") and n.replace("cam0/", "cam1/", 1) in c]
    return baseline / np.median(d)        # map units -> metres

s_j, s_d = rig_scale(ref_sfm), rig_scale(qry_sfm)
print("July metric scale", s_j, " Dec metric scale", s_d, " Dec->July expected", s_d / s_j)