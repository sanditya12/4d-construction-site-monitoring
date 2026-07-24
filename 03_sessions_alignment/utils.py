from pathlib import Path
import pycolmap
import shutil

# import os
# import sys

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# hloc_path = os.path.join(BASE_DIR, "submodule/Hierarchical-Localization")

# if hloc_path not in sys.path:
#     sys.path.insert(0, hloc_path)


from hloc import (
    extract_features,
    match_features,
    pairs_from_retrieval,
    reconstruction,
    localize_sfm,
)


# ---------------------------------------------------------------------------
# Configs for hloc
# ---------------------------------------------------------------------------

RETRIEVAL_CONF = extract_features.confs["netvlad"]
FEATURE_CONF = extract_features.confs["superpoint_aachen"]
MATCHER_CONF = match_features.confs["superglue"]
MATCHER_CONF["model"]["weights"] = "indoor"

# ---------------------------------------------------------------------------
# Localize t2 images against t1 pointcloud
# ---------------------------------------------------------------------------

def localize_t2(
    root: Path,
    t1_images: Path,
    t2_images: Path,
    sfm_dir: Path,
    out_dir: Path,
    queries_file: Path,
    num_retrieval_matches: int = 10,
):

    out_dir.mkdir(parents=True, exist_ok=True)

    # --- symlinks with t1_/t2_ prefix to avoid name collisions ---
    unified = out_dir / "images_all"
    unified.mkdir(exist_ok=True)

    for p in sorted(t1_images.iterdir()):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            dst = unified / f"t1_{p.name}"
            if not dst.exists():
                dst.symlink_to(p.resolve())

    for p in sorted(t2_images.iterdir()):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            dst = unified / f"t2_{p.name}"
            if not dst.exists():
                dst.symlink_to(p.resolve())

    t1_list = sorted([f"t1_{p.name}" for p in t1_images.iterdir()
                      if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    t2_list = sorted([f"t2_{p.name}" for p in t2_images.iterdir()
                      if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    full_list = t1_list + t2_list

    # --- Global descriptors for VPR ---
    global_t1 = extract_features.main(
        RETRIEVAL_CONF, unified, out_dir,
        image_list=t1_list,
        feature_path=out_dir / "global-t1.h5",
    )
    global_t2 = extract_features.main(
        RETRIEVAL_CONF, unified, out_dir,
        image_list=t2_list,
        feature_path=out_dir / "global-t2.h5",
    )

    # --- VPR retrieval ---
    pairs_path = out_dir / "pairs-t2-to-t1.txt"
    pairs_from_retrieval.main(
        descriptors=global_t2,
        output=pairs_path,
        num_matched=num_retrieval_matches,
        db_descriptors=global_t1,
    )

    # --- Fix db-side names to match COLMAP model ---
    # pairs db-side has "t1_0001.png", COLMAP model has "0001.png"
    # so strip the "t1_" prefix to get the COLMAP name
    colmap_names = {img.name for img in pycolmap.Reconstruction(str(sfm_dir)).images.values()}
    fixed_pairs = out_dir / "pairs-t2-to-t1-fixed.txt"
    with open(pairs_path) as fin, open(fixed_pairs, "w") as fout:
        for line in fin:
            q, db = line.strip().split()
            # db is "t1_0001.png" -> strip prefix -> "0001.png"
            db_bare = db[len("t1_"):]
            db_fixed = db_bare if db_bare in colmap_names else db
            fout.write(f"{q} {db_fixed}\n")
    pairs_path = fixed_pairs

    # --- Local features ---
    local_feats = extract_features.main(
        FEATURE_CONF, unified, out_dir,
        image_list=full_list,
    )

    # --- Match features ---
    matches = match_features.main(
        MATCHER_CONF, pairs_path,  
        FEATURE_CONF["output"], out_dir,
        matches=out_dir / "matches.h5",
    )

    # --- PnP localization uses fixed pairs ---
    results_path = out_dir / "t2_poses.txt"
    localize_sfm.main(
        reference_sfm=sfm_dir,
        queries=queries_file,
        retrieval=fixed_pairs,   # bare db names for COLMAP
        features=local_feats,
        matches=matches,
        results=results_path,
    )

    # --- PnP localization ---
    results_path = out_dir / "t2_poses.txt"
    localize_sfm.main(
        reference_sfm=sfm_dir,
        queries=queries_file,
        retrieval=pairs_path,
        features=local_feats,
        matches=matches,
        results=results_path,
    )

    return results_path