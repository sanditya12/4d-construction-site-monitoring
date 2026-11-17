from pathlib import Path
import pycolmap
import shutil
import yaml
import h5py


from hloc import (
    extract_features,
    match_features,
    pairs_from_retrieval,
    reconstruction,
    localize_sfm,
)

import utils

# ---------------------------------------------------------------------------
# Configs from config.yaml file
# ---------------------------------------------------------------------------
# with open('config.yaml', 'r') as file:
#     config = yaml.safe_load(file)

RETRIEVAL_CONF = extract_features.confs["netvlad"]
FEATURE_CONF = extract_features.confs["superpoint_aachen"]
MATCHER_CONF = match_features.confs["superglue"]
MATCHER_CONF["model"]["weights"] = "indoor"

# ---------------------------------------------------------------------------
# Localize t2 images against t1 pointcloud
# ---------------------------------------------------------------------------

def localize_t2(
    t1_sfm: Path,
    out_dir: Path,
    queries_file: Path,
    merged_local: Path,
    global_t1: Path,
    global_t2: Path,
    num_retrieval_matches: int = 10,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- VPR retrieval: July queries -> May db ---
    pairs_path = out_dir / "pairs-t2-to-t1.txt"
    pairs_from_retrieval.main(
        descriptors=global_t2,
        output=pairs_path,
        num_matched=num_retrieval_matches,
        db_descriptors=global_t1,
    )

    # --- Matching ---
    matches = match_features.main(
        MATCHER_CONF, pairs_path, merged_local, out_dir,
        matches=out_dir / "matches.h5",
    )

    # --- Using filtered matches ---

    # matches = out_dir / "matches_filtered.h5"

    # if not matches.exists():
    #     raise FileNotFoundError(
    #         f"Filtered matches file not found: {matches}"
    #     )


    # --- PnP localization ---
    results_path = out_dir / "t2_poses.txt"
    localize_sfm.main(
        reference_sfm=t1_sfm,
        queries=queries_file,
        retrieval=pairs_path,
        features=merged_local,
        matches=matches,
        results=results_path,
    )
    return results_path




def main():
    root = Path("/home/kanishka/4d_reconstruction/registration")

    # --- Existing outputs from build_sfm_mask  ---
    t1, t2 = "may", "july"

    t1_dir = root / f"floor1/outputs/nov_9/sfm_{t1}_clean"  # reference map
    t2_dir = root / f"floor1/outputs/nov_9/sfm_{t2}_clean"   # images to localize
    sfm_dir = t1_dir / "sfm"

    loc_dir = root / f"floor1/outputs/nov_9/loc_{t2}_in_{t1}"
    loc_dir.mkdir(parents=True, exist_ok=True)

    print(f"t1 reconstruction from: {sfm_dir}")

    # 1. Rename t2 h5 keys t1_ -> t2_ and merge with t1 features
    merged_local, global_t1, global_t2 = utils.prepare_t2_localization(
        t1_dir=t1_dir, t2_dir=t2_dir, out_dir=loc_dir,
    )

    # 2. Queries file from the h5 keys, intrinsics from the t1 map
    queries_file = utils.write_queries_from_h5(
        t1_sfm=sfm_dir,
        merged_local=merged_local,
        out_file=loc_dir / "queries_with_intrinsics.txt",
    )

    # 3. Localize
    print("\n=== Localizing t2 images against t1 map ===")
    results_path = localize_t2(
        t1_sfm=sfm_dir,
        out_dir=loc_dir,
        queries_file=queries_file,
        merged_local=merged_local,
        global_t1=global_t1,
        global_t2=global_t2,
        num_retrieval_matches=10,
    )
    print(f"t2 poses written to: {results_path}")


if __name__ == "__main__":
    main()