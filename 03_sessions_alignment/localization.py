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

from pathlib import Path
import pycolmap
from hloc import extract_features

RETRIEVAL_CONF = extract_features.confs["netvlad"]

def ensure_global_feats(
    session_dir: Path,
    model_dir: Path,            # folder with images.bin / cameras.bin / points3D.bin
    session_tag: str,           # prefix used in the names, e.g. "t1" or "t2"
    src_image_dirs: dict,       # {"cam0": Path(...), "cam1": Path(...)} original images
    image_root: Path = None,    # where the cam0/ cam1/ tree lives or should be created
    conf=RETRIEVAL_CONF,
) -> Path:
    out = session_dir / f"{conf['output']}.h5"
    if out.exists():
        return out

    image_root = image_root or session_dir / "images_all"
    model = pycolmap.Reconstruction(str(model_dir))
    names = sorted(im.name for im in model.images.values())

    # rebuild cam0/<tag>_xxx.jpg -> original file symlinks if they're missing
    prefix = f"{session_tag}_"
    for name in names:
        dst = image_root / name
        if dst.exists():                # valid link or file -> keep
            continue
        if dst.is_symlink():            # dangling link from another machine -> replace
            dst.unlink()
        cam, fname = name.split("/", 1)
        orig = fname[len(prefix):] if fname.startswith(prefix) else fname
        src = Path(src_image_dirs[cam]) / orig
        if not src.exists():
            raise FileNotFoundError(f"{name}: expected original image at {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.symlink_to(src.resolve())

    return extract_features.main(conf, image_root, session_dir, image_list=names)

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
    root = Path("/home_domuser2/s84klaks/src/4d-construction-site-monitoring")

    # --- Existing outputs from build_sfm_mask  ---
    t1, t2 = "july", "december"

    t1_dir = root / f"data/outputs/sfm/sfm_{t1}_clean"  # reference map
    t2_dir = root / f"data/outputs/sfm/sfm_{t2}_clean"   # images to localize
    sfm_dir = t1_dir / "sfm"

    loc_dir = root / f"data/outputs/sfm/loc_{t2}_in_{t1}"
    loc_dir.mkdir(parents=True, exist_ok=True)

    print(f"t1 reconstruction from: {sfm_dir}")
    # 0. Make sure both sessions have global descriptors (no-op if already present)
    ensure_global_feats(
        t1_dir, model_dir=t1_dir / "sfm", session_tag=f"t1",
        src_image_dirs={"cam0": root / f"data/raw/{t1}-images/sampled_rotated_frames/cam0",
                        "cam1": root / f"data/raw/{t1}-images/sampled_rotated_frames/cam1"},
    )
    ensure_global_feats(
        t2_dir, model_dir=t2_dir / "sfm", session_tag="t2",
        src_image_dirs={"cam0": root / f"data/raw/{t2}-images/sampled_rotated_frames/cam0",
                        "cam1": root / f"data/raw/{t2}-images/sampled_rotated_frames/cam1"},
    )

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