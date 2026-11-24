from itertools import combinations
from pathlib import Path
import numpy as np
import pycolmap
from scipy.spatial import KDTree

# ---------------------------------------------------------------------------
# Paths — edit these
# ---------------------------------------------------------------------------
root = Path("/home_domuser2/s84klaks/src/4d-construction-site-monitoring")

REF = "july"                                   # reference session (frame everything is aligned to)
QUERIES = ["december"]                         # sessions aligned to REF (add more if you have them)

ref_sfm = root / f"data/outputs/sfm/sfm_{REF}_clean/sfm"

SESSIONS = {
    q: {
        "sfm_dir":       root / f"data/outputs/sfm/sfm_{q}_clean/sfm",
        "alignment_npz": root / f"data/outputs/sfm/loc_{q}_in_{REF}/alignment/alignment.npz",
    }
    for q in QUERIES
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_points(sfm_dir: Path) -> np.ndarray:
    model = pycolmap.Reconstruction(str(sfm_dir))
    return np.array([p.xyz for p in model.points3D.values()])


def remove_outliers(pts: np.ndarray, k: int = 20, std_ratio: float = 2.0):
    tree = KDTree(pts)
    dists, _ = tree.query(pts, k=k+1)
    mean_dists = dists[:, 1:].mean(axis=1)
    threshold  = mean_dists.mean() + std_ratio * mean_dists.std()
    return pts[mean_dists < threshold]


def chamfer_distance(a: np.ndarray, b: np.ndarray):
    """Returns (cd, cd_a2b, cd_b2a, p95_a2b, p95_b2a)."""
    dists_a2b, _ = KDTree(b).query(a, k=1)
    dists_b2a, _ = KDTree(a).query(b, k=1)
    return (
        dists_a2b.mean() + dists_b2a.mean(),
        dists_a2b.mean(),
        dists_b2a.mean(),
        np.percentile(dists_a2b, 95),
        np.percentile(dists_b2a, 95),
    )


def evaluate(label: str, name_a: str, name_b: str, pts_a: np.ndarray, pts_b: np.ndarray):
    pts_a_c = remove_outliers(pts_a)
    pts_b_c = remove_outliers(pts_b)
    cd, a2b, b2a, p95_a2b, p95_b2a = chamfer_distance(pts_a_c, pts_b_c)
    print(f"  {label}")
    print(f"    points       : {len(pts_a_c):,} vs {len(pts_b_c):,} (after outlier removal)")
    print(f"    {name_a}->{name_b} mean  : {a2b:.4f}    95th: {p95_a2b:.4f}")
    print(f"    {name_b}->{name_a} mean  : {b2a:.4f}    95th: {p95_b2a:.4f}")
    print(f"    Chamfer (sum): {cd:.4f}  {'GOOD' if cd < 0.2 else 'CHECK' if cd < 0.5 else 'POOR'}")
    return cd


# ---------------------------------------------------------------------------
# Load reference once
# ---------------------------------------------------------------------------
print(f"=== Loading {REF} point cloud ===")
pts_ref = load_points(ref_sfm)
print(f"  {REF}: {len(pts_ref):,} points")

results = {}
pts_aligned = {}

# ---------------------------------------------------------------------------
# Evaluate each session
# ---------------------------------------------------------------------------
for session, cfg in SESSIONS.items():
    print(f"\n{'='*60}")
    print(f"Session: {REF} vs {session}")
    print(f"{'='*60}")

    pts_tx_raw = load_points(cfg["sfm_dir"])
    print(f"  {session}: {len(pts_tx_raw):,} points")

    aln   = np.load(str(cfg["alignment_npz"]))
    R_aln = aln["R"]
    t_aln = aln["t"]
    s_aln = float(aln["s"])
    print(f"  Alignment: s={s_aln:.4f}")

    pts_tx_aligned = (s_aln * (R_aln @ pts_tx_raw.T)).T + t_aln

    pts_aligned[session] = pts_tx_aligned

    print("\n  Before alignment:")
    cd_before = evaluate(f"{REF} vs {session} (raw)", REF, session, pts_ref, pts_tx_raw)

    print("\n  After alignment:")
    cd_after = evaluate(f"{REF} vs {session} (aligned)", REF, session, pts_ref, pts_tx_aligned)

    improvement = (cd_before - cd_after) / cd_before * 100
    print(f"\n  Improvement: {cd_before:.4f} -> {cd_after:.4f}  ({improvement:.1f}% reduction)")

    results[session] = {
        "before": cd_before,
        "after":  cd_after,
        "improvement_pct": improvement,
    }


# Between aligned query sessions (only when there are at least two)
pair_cds = {}
for a, b in combinations(pts_aligned, 2):
    pair_cds[(a, b)] = evaluate(f"{a} vs {b} (aligned)", a, b, pts_aligned[a], pts_aligned[b])

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print("Summary")
print(f"{'='*60}")
print(f"  {'Session':<22} {'Before':>10} {'After':>10} {'Improvement':>14}")
print(f"  {'-'*58}")
for session, r in results.items():
    print(f"  {REF + ' vs ' + session:<22} {r['before']:>10.4f} {r['after']:>10.4f} {r['improvement_pct']:>13.1f}%")
for (a, b), cd in pair_cds.items():
    print(f"  {a + ' vs ' + b:<22} {'':>10} {cd:>10.4f}")