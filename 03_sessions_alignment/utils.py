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

try:
    from hloc import pairs_from_sequential
except ImportError:
    pairs_from_sequential = None

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

    # --- Local features ---
    local_feats = extract_features.main(
        FEATURE_CONF, unified, out_dir,
        image_list=full_list,
    )

    # --- Fix pairs for matching (keep t1_ prefix, feature h5 has t1_ keys) ---
    # pairs file already has t2_/t1_ prefixes — use as-is for matching
    matches = match_features.main(
        MATCHER_CONF, pairs_path,  # original pairs with t1_/t2_ prefixes
        FEATURE_CONF["output"], out_dir,
        matches=out_dir / "matches.h5",
    )

    # # --- Fix db-side names for localization only (strip t1_ to match COLMAP) ---
    # colmap_names = {img.name for img in pycolmap.Reconstruction(str(sfm_dir)).images.values()}
    # fixed_pairs = out_dir / "pairs-t2-to-t1-fixed.txt"
    # with open(pairs_path) as fin, open(fixed_pairs, "w") as fout:
    #     for line in fin:
    #         q, db = line.strip().split()
    #         db_bare = db[len("t1_"):]
    #         db_fixed = db_bare if db_bare in colmap_names else db
    #         fout.write(f"{q} {db_fixed}\n")

    # # --- PnP localization uses fixed pairs ---
    # results_path = out_dir / "t2_poses.txt"
    # localize_sfm.main(
    #     reference_sfm=sfm_dir,
    #     queries=queries_file,
    #     retrieval=fixed_pairs,   # bare db names for COLMAP
    #     features=local_feats,
    #     matches=matches,
    #     results=results_path,
    # )

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


# ---------------------------------------------------------------------------
# Build t1 reconstruction (Optional step if SfM reconstructions not available)
# ---------------------------------------------------------------------------
def list_images_relative(root: Path, image_dir: Path, exts=(".jpg", ".jpeg", ".png")):
    """
    Return a sorted list of image paths relative to `root`, for all images
    found in `image_dir`. E.g. if root=project and image_dir=project/t1/images,
    returns ["t1/images/frame_0001.jpg", ...].
    """
    files = [
        p for p in sorted(image_dir.iterdir())
        if p.suffix.lower() in exts
    ]
    return [str(p.relative_to(root)) for p in files]


def build_t1_reconstruction(
    root: Path,
    t1_images: Path,
    out_dir: Path,
    sequential: bool = False,
    seq_window: int = 10,
    num_retrieval_matches: int = 10,
):
    
    out_dir.mkdir(parents=True, exist_ok=True)

    image_list = list_images_relative(root, t1_images)

    # --- Local features ---
    local_feats = extract_features.main(
        FEATURE_CONF, root, out_dir,
        image_list=image_list,
    )

    # --- Pairs ---
    pairs_path = out_dir / "pairs-t1.txt"
    if sequential:
        if pairs_from_sequential is None:
            raise RuntimeError(
                "pairs_from_sequential not available in this hloc version. "
                "Set sequential=False to use retrieval-based pairing instead."
            )
        pairs_from_sequential.main(
            pairs_path,
            image_list=image_list,
            window_size=seq_window,
        )
        global_feats = None
    else:
        global_feats = extract_features.main(
            RETRIEVAL_CONF, root, out_dir,
            image_list=image_list,
        )
        pairs_from_retrieval.main(
            global_feats, pairs_path, num_matched=num_retrieval_matches
        )

    # --- Matches ---
    matches = match_features.main(
        MATCHER_CONF, pairs_path, FEATURE_CONF["output"], out_dir,
        matches=out_dir / "matches-t1.h5",
    )

    # --- Reconstruction ---

    sfm_dir = out_dir / "sfm"
    reconstruction.main(
        sfm_dir=sfm_dir,
        image_dir=root,
        pairs=pairs_path,
        features=local_feats,
        matches=matches,
        image_list=image_list,
        camera_mode=pycolmap.CameraMode.SINGLE,
        image_options={"camera_model": "OPENCV_FISHEYE"},
    )

    return {
        "sfm_dir": sfm_dir,
        "image_list": image_list,
        "global_feats": global_feats,
        "local_feats": local_feats,
        "matches": matches,
    }