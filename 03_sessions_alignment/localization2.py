"""
Localize December (query) images against the July (reference) SfM map.

Naming convention (from the SfM script):
    images_all/cam0/<tag>_<orig>.jpg   ->  h5 / map name:  "cam0/<tag>_<orig>.jpg"
    tags: may=t1, july=t2, december=t3
"""
from pathlib import Path

import h5py
import pycolmap

from hloc import (
    extract_features,
    localize_sfm,
    match_features,
    pairs_from_retrieval,
)
from hloc.utils.io import list_h5_names

# ---------------------------------------------------------------------------
# Configs (single source of truth)
# ---------------------------------------------------------------------------
RETRIEVAL_CONF = extract_features.confs["netvlad"]
FEATURE_CONF = extract_features.confs["superpoint_aachen"]
MATCHER_CONF = match_features.confs["superglue"]
MATCHER_CONF["model"]["weights"] = "indoor"
MATCHER_CONF["model"]["match_threshold"] = 0.3  # same as the SfM script

SESSIONS = {"may": "t1", "july": "t2", "december": "t3"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def nested_names(h5_path):
    """Only 'cam0/..' / 'cam1/..' names. Older flat keys left over from earlier
    runs (hloc appends to existing .h5 files) are ignored."""
    return [n for n in list_h5_names(h5_path) if "/" in n]


def ensure_global_feats(
    session_dir: Path,
    session_tag: str,
    src_image_dirs: dict,       # {"cam0": Path, "cam1": Path} original images (no prefix)
    image_root: Path = None,    # where cam0/ cam1/ symlink tree lives / is rebuilt
    conf=RETRIEVAL_CONF,
) -> Path:
    """NetVLAD descriptors for every image that has local features.
    Regenerated automatically if the existing file has different names."""
    local_h5 = session_dir / f"{FEATURE_CONF['output']}.h5"
    out = session_dir / f"{conf['output']}.h5"
    names = sorted(nested_names(local_h5))

    if out.exists():
        if set(nested_names(out)) == set(names) and set(list_h5_names(out)) == set(names):
            return out
        print(f"[global] {out.name} has stale/mismatched names -> regenerating")
        out.unlink()

    image_root = image_root or session_dir / "images_all"
    prefix = f"{session_tag}_"
    for name in names:
        dst = image_root / name
        if dst.exists():
            continue
        if dst.is_symlink():  # dangling link from another machine
            dst.unlink()
        cam, fname = name.split("/", 1)
        orig = fname[len(prefix):] if fname.startswith(prefix) else fname
        src = Path(src_image_dirs[cam]) / orig
        if not src.exists():
            raise FileNotFoundError(f"{name}: expected original image at {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.symlink_to(src.resolve())

    return extract_features.main(conf, image_root, session_dir, image_list=names)


def merge_h5(src_paths, dst_path: Path):
    """Merge hloc feature files with nested names (cam0/t2_x.jpg)."""
    with h5py.File(dst_path, "a") as dst:
        for sp in src_paths:
            with h5py.File(sp, "r") as src:
                for name in nested_names(sp):
                    if name in dst:
                        continue
                    if "/" in name:
                        dst.require_group(name.rsplit("/", 1)[0])
                    src.copy(src[name], dst, name=name)


def prepare_localization(t1_dir: Path, t2_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    t1_local = t1_dir / f"{FEATURE_CONF['output']}.h5"
    t2_local = t2_dir / f"{FEATURE_CONF['output']}.h5"
    global_t1 = t1_dir / f"{RETRIEVAL_CONF['output']}.h5"
    global_t2 = t2_dir / f"{RETRIEVAL_CONF['output']}.h5"

    merged = out_dir / "feats-merged.h5"
    expected = set(nested_names(t1_local)) | set(nested_names(t2_local))
    if merged.exists() and set(list_h5_names(merged)) != expected:
        print("[merge] stale feats-merged.h5 -> rebuilding")
        merged.unlink()
    if not merged.exists():
        merge_h5([t1_local, t2_local], merged)
    return merged, global_t1, global_t2


def write_queries(t1_sfm: Path, merged_local: Path, out_file: Path, prefix: str):
    """One line per query image: name MODEL W H params... (intrinsics from the map)."""
    rec = pycolmap.Reconstruction(str(t1_sfm))

    cams = {}  # folder -> camera
    for im in rec.images.values():
        cams.setdefault(im.name.split("/", 1)[0], rec.cameras[im.camera_id])
    print("[queries] map cameras:", {
        k: (c.model.name, c.width, c.height, [float(p) for p in c.params[:2]])
        for k, c in cams.items()
    })
    single_cam = next(iter(rec.cameras.values())) if len(rec.cameras) == 1 else None
    if single_cam is not None:
        print("[queries] WARNING: map has ONE camera; all queries get its intrinsics.")

    lines = []
    with h5py.File(merged_local, "r") as f:
        for name in sorted(nested_names(merged_local)):
            folder, leaf = name.split("/", 1)
            if not leaf.startswith(prefix):
                continue
            cam = single_cam or cams[folder]
            iw, ih = map(int, f[name]["image_size"][()])
            if (iw, ih) != (cam.width, cam.height):
                raise ValueError(f"{name}: {iw}x{ih} vs map camera {cam.width}x{cam.height}")
            lines.append(
                f"{name} {cam.model.name} {cam.width} {cam.height} "
                + " ".join(map(str, cam.params))
            )

    if not lines:
        raise RuntimeError(f"0 queries with prefix '{prefix}' in {merged_local}")
    out_file.write_text("\n".join(lines) + "\n")
    print(f"[queries] Wrote {len(lines)} queries to {out_file}")
    return out_file


def check_pairs(pairs_path: Path, merged_local: Path, t1_sfm: Path):
    """Fail fast (seconds) instead of crashing inside the matcher."""
    feat_names = set(list_h5_names(merged_local))
    map_names = {i.name for i in pycolmap.Reconstruction(str(t1_sfm)).images.values()}
    bad_q, bad_db = set(), set()
    for line in open(pairs_path):
        q, d = line.split()
        if q not in feat_names:
            bad_q.add(q)
        if d not in feat_names or d not in map_names:
            bad_db.add(d)
    print("[check] queries missing from feats:", len(bad_q), list(bad_q)[:2])
    print("[check] db missing from feats/map:", len(bad_db), list(bad_db)[:2])
    if bad_q or bad_db:
        raise RuntimeError("Name mismatch between pairs, features and map (see above).")


# ---------------------------------------------------------------------------
# Localization
# ---------------------------------------------------------------------------
def localize(
    t1_sfm: Path,
    out_dir: Path,
    queries_file: Path,
    merged_local: Path,
    global_t1: Path,
    global_t2: Path,
    num_retrieval_matches: int = 10,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs_path = out_dir / "pairs-query-to-ref.txt"
    pairs_from_retrieval.main(
        descriptors=global_t2,
        output=pairs_path,
        num_matched=num_retrieval_matches,
        db_descriptors=global_t1,
    )
    check_pairs(pairs_path, merged_local, t1_sfm)

    matches = match_features.main(
        MATCHER_CONF, pairs_path, merged_local, out_dir,
        matches=out_dir / "matches.h5",
    )

    results_path = out_dir / "query_poses.txt"
    localize_sfm.main(
        reference_sfm=t1_sfm,
        queries=queries_file,
        retrieval=pairs_path,
        features=merged_local,
        matches=matches,
        results=results_path,
        prepend_camera_name=True,
    )
    return results_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    root = Path("/home_domuser2/s84klaks/src/4d-construction-site-monitoring")

    ref, qry = "july", "december"          # reference map / images to localize
    tag_ref, tag_qry = SESSIONS[ref], SESSIONS[qry]

    ref_dir = root / f"data/outputs/sfm/sfm_{ref}_clean"
    qry_dir = root / f"data/outputs/sfm/sfm_{qry}_clean"
    sfm_dir = ref_dir / "sfm"
    loc_dir = root / f"data/outputs/sfm/loc_{qry}_in_{ref}"
    loc_dir.mkdir(parents=True, exist_ok=True)

    def src_dirs(session):
        base = root / f"data/raw/{session}-images/sampled_rotated_frames"
        return {"cam0": base / "cam0", "cam1": base / "cam1"}

    print(f"Reference map: {sfm_dir}")

    # 0. Global descriptors for both sessions (regenerated if stale)
    ensure_global_feats(ref_dir, tag_ref, src_dirs(ref))
    ensure_global_feats(qry_dir, tag_qry, src_dirs(qry))

    # 1. Merge local features (no renaming: sessions already have unique prefixes)
    merged_local, global_ref, global_qry = prepare_localization(ref_dir, qry_dir, loc_dir)

    # 2. Queries file (December images only)
    queries_file = write_queries(
        sfm_dir, merged_local, loc_dir / "queries_with_intrinsics.txt",
        prefix=f"{tag_qry}_",
    )

    # 3. Retrieval -> matching -> PnP
    print(f"\n=== Localizing {qry} images against {ref} map ===")
    results_path = localize(
        t1_sfm=sfm_dir,
        out_dir=loc_dir,
        queries_file=queries_file,
        merged_local=merged_local,
        global_t1=global_ref,
        global_t2=global_qry,
        num_retrieval_matches=10,
    )
    print(f"Poses written to: {results_path}")


if __name__ == "__main__":
    main()