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
# Helper functions
# ---------------------------------------------------------------------------

def copy_h5_renamed(src_path, dst_path, old_prefix="t1_", new_prefix="t2_"):
    """Copy all entries of an hloc h5 into dst, renaming the key prefix.
    Appends to dst, skips keys that already exist (safe to re-run)."""
    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "a") as dst:
        for name in src.keys():
            new = new_prefix + name[len(old_prefix):] if name.startswith(old_prefix) else name
            if new in dst:
                continue
            src.copy(src[name], dst, name=new)  # keeps datasets + attrs


def prepare_t2_localization(t1_dir: Path, t2_dir: Path, out_dir: Path):
    """t1_dir / t2_dir = out_dir used for build_sfm_mask for May / July."""
    out_dir.mkdir(parents=True, exist_ok=True)

    t1_local  = t1_dir / f"{FEATURE_CONF['output']}.h5"
    t2_local  = t2_dir / f"{FEATURE_CONF['output']}.h5"
    t1_global = t1_dir / f"{RETRIEVAL_CONF['output']}.h5"
    t2_global = t2_dir / f"{RETRIEVAL_CONF['output']}.h5"

    # merged local features: May as-is (t1_), July renamed (t2_)
    merged_local = out_dir / "feats-merged.h5"
    if not merged_local.exists():
        copy_h5_renamed(t1_local, merged_local, "t1_", "t1_")  # plain copy
        copy_h5_renamed(t2_local, merged_local, "t1_", "t2_")

    # July global descriptors renamed; May global used directly
    t2_global_renamed = out_dir / "global-t2.h5"
    if not t2_global_renamed.exists():
        copy_h5_renamed(t2_global, t2_global_renamed, "t1_", "t2_")

    return merged_local, t1_global, t2_global_renamed

def write_queries_from_h5(t1_sfm: Path, merged_local: Path, out_file: Path, prefix: str = "t2_"):
    """Build the queries file from the merged features h5 (no image files needed).
    Every key starting with `prefix` is a query; intrinsics come from the t1 map."""
    rec = pycolmap.Reconstruction(str(t1_sfm))
    cam = next(iter(rec.cameras.values()))  # SINGLE mode -> one shared camera
    model = cam.model.name if hasattr(cam.model, "name") else cam.model_name
    w, h, params = cam.width, cam.height, list(cam.params)
    print(f"[queries] Using intrinsics from map: {model} {w}x{h} {params}")

    lines = []
    with h5py.File(merged_local, "r") as f:
        for name in sorted(f.keys()):
            if not name.startswith(prefix):
                continue
            iw, ih = f[name]["image_size"][()]  # original image size stored by hloc
            if (int(iw), int(ih)) != (w, h):
                raise ValueError(f"{name} is {iw}x{ih}, map camera is {w}x{h}")
            lines.append(f"{name} {model} {w} {h} " + " ".join(map(str, params)))

    out_file.write_text("\n".join(lines) + "\n")
    print(f"[queries] Wrote {len(lines)} queries to {out_file}")
    return out_file


# for sequential matching
def generate_sequential_pairs(
    image_list: list,
    pairs_path: Path,
    window: int = 10,
):
    """
    Generate sequential image pairs from an existing image list.
    """

    pairs = []

    image_list = sorted(image_list)

    for i, image1 in enumerate(image_list):
        for j in range(i + 1, min(i + window + 1, len(image_list))):
            image2 = image_list[j]
            pairs.append((image1, image2))

    pairs_path.parent.mkdir(parents=True, exist_ok=True)

    with open(pairs_path, "w") as f:
        for image1, image2 in pairs:
            f.write(f"{image1} {image2}\n")

    print(f"[sequential] Generated {len(pairs)} pairs (window={window})")

    return pairs

# ---------------------------------------------------------------------------
# SfM reconstruction 
# ---------------------------------------------------------------------------


def build_sfm_mask(
    root: Path,
    images_cam0: Path,
    images_cam1: Path,
    out_dir: Path,
    masks_cam0: Path = None,
    masks_cam1: Path = None,
    sequential: bool = False,
    seq_window: int = 10,
    num_retrieval_matches: int = 10,
    session: str = "t1"
):
    import cv2
    import pycolmap

    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Unified symlink dir with cam0_/cam1_ prefixes ---
    unified = out_dir / "images_all"
    unified.mkdir(exist_ok=True)

    for p in sorted(images_cam0.iterdir()):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            dst = unified / f"{session}_cam0_{p.name}"
            if not dst.exists():
                dst.symlink_to(p.resolve())

    for p in sorted(images_cam1.iterdir()):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            dst = unified / f"{session}_cam1_{p.name}"
            if not dst.exists():
                dst.symlink_to(p.resolve())

    image_list = (
        sorted([f"t1_cam0_{p.name}" for p in images_cam0.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
        + sorted([f"t1_cam1_{p.name}" for p in images_cam1.iterdir()
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    )

    # --- Prepare masks (invert: white=keep -> black=keep for COLMAP) ---
    colmap_masks_dir = None
    if masks_cam0 is not None or masks_cam1 is not None:
        colmap_masks_dir = out_dir / "masks"
        colmap_masks_dir.mkdir(exist_ok=True)

        def write_masks(masks_dir, prefix):
            if masks_dir is None:
                return
            for p in masks_dir.iterdir():
                if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                mask = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                mask_inv = cv2.bitwise_not(mask)
                out_name = colmap_masks_dir / f"{prefix}{p.name}.mask.png"
                cv2.imwrite(str(out_name), mask_inv)

        write_masks(masks_cam0, "cam0_")
        write_masks(masks_cam1, "cam1_")
        print(f"[masks] Prepared {len(list(colmap_masks_dir.iterdir()))} masks")

    # --- Local features ---
    local_feats = extract_features.main(
        FEATURE_CONF, unified, out_dir,
        image_list=image_list,
    )

    # --- Pairs ---
    # pairs_path = out_dir / "pairs-t1.txt"
    # global_feats = None
    # if sequential:
    #     cam0_list = sorted([
    #         f"t1_cam0_{p.name}"
    #         for p in images_cam0.iterdir()
    #         if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    #     ])
    #     cam1_list = sorted([
    #         f"t1_cam1_{p.name}"
    #         for p in images_cam1.iterdir()
    #         if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    #     ])
    #     generate_sequential_pairs(
    #         cam0_list,
    #         pairs_path,
    #         window=seq_window,
    #     )

    #     pairs_cam1_path = out_dir / "pairs-cam1-temp.txt"
    #     generate_sequential_pairs(
    #         cam1_list,
    #         pairs_cam1_path,
    #         window=seq_window,
    #     )

    #     # Combine both cameras' pairs
    #     with open(pairs_path, "a") as f:
    #         with open(pairs_cam1_path, "r") as g:
    #             f.write(g.read())

    #     pairs_cam1_path.unlink()

    #     print(f"[sequential] Combined pairs saved to {pairs_path}")

    # else:
    #     global_feats = extract_features.main(
    #         RETRIEVAL_CONF, unified, out_dir,
    #         image_list=image_list,
    #     )
    #     pairs_from_retrieval.main(
    #         global_feats, pairs_path, num_matched=num_retrieval_matches)

    # # --- Matches ---
    # matches = match_features.main(
    #     MATCHER_CONF, pairs_path, FEATURE_CONF["output"], out_dir,
    #     matches=out_dir / "matches-t1.h5",
    # )

    # PAIRS - SEQUENTIAL + GLOBAL RETRIEVAL
    pairs_path = out_dir / "pairs-t1.txt"
    
    # 1. Sequential pairs for smooth temporal tracking
    cam0_list = sorted([
        f"t1_cam0_{p.name}"
        for p in images_cam0.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ])
    cam1_list = sorted([
        f"t1_cam1_{p.name}"
        for p in images_cam1.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ])
    generate_sequential_pairs(cam0_list, pairs_path, window=seq_window)

    pairs_cam1_path = out_dir / "pairs-cam1-temp.txt"
    generate_sequential_pairs(cam1_list, pairs_cam1_path, window=seq_window)

    with open(pairs_path, "a") as f:
        with open(pairs_cam1_path, "r") as g:
            f.write(g.read())
    pairs_cam1_path.unlink()

    # 2. Global Retrieval (NetVLAD) pairs to bridge loops and reconnect sub-models
    print("[pairs] Extracting NetVLAD global descriptors for loop closure...")
    global_feats = extract_features.main(
        RETRIEVAL_CONF, unified, out_dir,
        image_list=image_list,
    )
    
    retrieval_pairs_path = out_dir / "pairs-retrieval-temp.txt"
    pairs_from_retrieval.main(
        global_feats, retrieval_pairs_path, num_matched=num_retrieval_matches
    )

    # 3. Merge sequential and retrieval pairs, removing duplicate lines
    seen_pairs = set()
    with open(pairs_path, "r") as f:
        for line in f:
            seen_pairs.add(line.strip())

    added_retrieval_count = 0
    with open(retrieval_pairs_path, "r") as f, open(pairs_path, "a") as out_f:
        for line in f:
            line_str = line.strip()
            if line_str and line_str not in seen_pairs:
                seen_pairs.add(line_str)
                out_f.write(line_str + "\n")
                added_retrieval_count += 1

    retrieval_pairs_path.unlink()
    print(f"[pairs] Added {added_retrieval_count} global loop-closure pairs to {pairs_path}")

    # --- Matches ---  
    # matches = match_features.main(
    #     MATCHER_CONF, pairs_path, FEATURE_CONF["output"], out_dir,
    #     matches=out_dir / "matches-t1.h5",
    # )

    matches = out_dir / "matches_filtered.h5"

    # --- Reconstruction ---
    # SINGLE camera mode assumes same intrinsics for both cameras.
    # Use PER_IMAGE if they differ significantly.
    image_options = {"camera_model": "OPENCV_FISHEYE"}
    if colmap_masks_dir is not None:
        image_options["camera_mask_path"] = str(colmap_masks_dir)

    sfm_dir = out_dir / "sfm"
    recon_kwargs = dict(
        sfm_dir=sfm_dir,
        image_dir=unified,
        pairs=pairs_path,
        features=local_feats,
        matches=matches,
        image_list=image_list,
        camera_mode=pycolmap.CameraMode.SINGLE,
        image_options=image_options,
    )
    # if colmap_masks_dir is not None:
    #     recon_kwargs["mask_path"] = colmap_masks_dir

    reconstruction.main(**recon_kwargs)

    return {
        "sfm_dir": sfm_dir,
        "image_list": image_list,
        "global_feats": global_feats,
        "local_feats": local_feats,
        "matches": matches,
    }



def build_sfm_mask_extrinsics(
    root: Path,
    images_cam0: Path,
    images_cam1: Path,
    out_dir: Path,
    masks_cam0: Path = None,
    masks_cam1: Path = None,
    seq_window: int = 10,
    session: str = "t1",
):
    import cv2
    import numpy as np
    import pycolmap

    from hloc import reconstruction as hloc_reconstruction

    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # 1. Organize images into camera folders
    # ---------------------------------------------------------
    unified = out_dir / "images_all"
    cam0_dir = unified / "cam0"
    cam1_dir = unified / "cam1"

    cam0_dir.mkdir(parents=True, exist_ok=True)
    cam1_dir.mkdir(parents=True, exist_ok=True)
    extensions = {".jpg", ".jpeg", ".png"}

    files0 = sorted(
        p for p in images_cam0.iterdir()
        if p.suffix.lower() in extensions
    )
    files1 = sorted(
        p for p in images_cam1.iterdir()
        if p.suffix.lower() in extensions
    )

    names0 = {p.name for p in files0}
    names1 = {p.name for p in files1}

    common_names = sorted(names0 & names1)

    if not common_names:
        raise ValueError("No matching filenames between cam0 and cam1.")

    if names0 != names1:
        print(
            f"[warning] {len(names0 - names1)} cam0 images and "
            f"{len(names1 - names0)} cam1 images have no stereo partner."
        )

    for p in files0:
        dst = cam0_dir / p.name
        if not dst.exists():
            dst.symlink_to(p.resolve())

    for p in files1:
        dst = cam1_dir / p.name
        if not dst.exists():
            dst.symlink_to(p.resolve())

    image_list = (
        [f"cam0/{session}_{p.name}" for p in files0]
        + [f"cam1/{session}_{p.name}" for p in files1]
    )

    # ---------------------------------------------------------
    # 2. Prepare masks
    # ---------------------------------------------------------

    colmap_masks_dir = None

    if masks_cam0 is not None or masks_cam1 is not None:
        colmap_masks_dir = out_dir / "masks"
        colmap_masks_dir.mkdir(exist_ok=True)

        def write_masks(mask_dir, camera_name):
            if mask_dir is None:
                return

            camera_mask_dir = colmap_masks_dir / camera_name
            camera_mask_dir.mkdir(exist_ok=True)

            for p in mask_dir.iterdir():
                if p.suffix.lower() not in extensions:
                    continue

                mask = cv2.imread(
                    str(p), cv2.IMREAD_GRAYSCALE
                )

                if mask is None:
                    continue

                mask_inv = cv2.bitwise_not(mask)

                cv2.imwrite(
                    str(camera_mask_dir / f"{p.name}.png"),
                    mask_inv,
                )

        write_masks(masks_cam0, "cam0")
        write_masks(masks_cam1, "cam1")

    # ---------------------------------------------------------
    # 3. Local features
    # ---------------------------------------------------------

    local_feats = extract_features.main(
        FEATURE_CONF,
        unified,
        out_dir,
        image_list=image_list,
    )

    # ---------------------------------------------------------
    # 4. Sequential pairs ONLY
    # ---------------------------------------------------------

    pairs_path = out_dir / f"pairs-{session}.txt"

    pairs_cam0 = [f"cam0/{p.name}" for p in files0]
    pairs_cam1 = [f"cam1/{p.name}" for p in files1]

    tmp0 = out_dir / "pairs-cam0-temp.txt"
    tmp1 = out_dir / "pairs-cam1-temp.txt"

    generate_sequential_pairs(
        pairs_cam0,
        tmp0,
        window=seq_window,
    )

    generate_sequential_pairs(
        pairs_cam1,
        tmp1,
        window=seq_window,
    )

    # Combine sequential temporal pairs and synchronized stereo pairs.
    pairs = set()

    for pair_file in (tmp0, tmp1):
        with open(pair_file) as f:
            for line in f:
                if line.strip():
                    pairs.add(line.strip())

        pair_file.unlink()

    # Add stereo correspondences for every synchronized frame.
    for name in common_names:
        pairs.add(f"cam0/{name} cam1/{name}")

    with open(pairs_path, "w") as f:
        for pair in sorted(pairs):
            f.write(pair + "\n")

    print(
        f"[pairs] {len(pairs)} sequential + synchronized stereo pairs"
    )

    # ---------------------------------------------------------
    # 5. Feature matching
    # ---------------------------------------------------------

    matches = match_features.main(
        MATCHER_CONF,
        pairs_path,
        FEATURE_CONF["output"],
        out_dir,
        matches=out_dir / f"matches-{session}.h5",
    )

    # ---------------------------------------------------------
    # 6. Rig Configuration
    # ---------------------------------------------------------

    sfm_dir = out_dir / "sfm"
    sfm_dir.mkdir(parents=True, exist_ok=True)

    database_path = sfm_dir / "database.db"

    hloc_reconstruction.create_empty_db(database_path)

    image_options = {
        "camera_model": "OPENCV_FISHEYE",
    }

    if colmap_masks_dir is not None:
        image_options["camera_mask_path"] = str(colmap_masks_dir)

    # IMPORTANT:
    # PER_FOLDER creates one camera calibration for cam0
    # and another for cam1.
    hloc_reconstruction.import_images(
        unified,
        database_path,
        pycolmap.CameraMode.PER_FOLDER,
        image_list=image_list,
        options=image_options,
    )

    image_ids = hloc_reconstruction.get_image_ids(database_path)

    # ---------------------------------------------------------
    # 7. Define calibrated stereo rig
    # ---------------------------------------------------------

    R_cam1_cam0 = np.array([
        [-0.999803492,  0.018300057, -0.007621329],
        [ 0.018294656,  0.999832336,  0.000777732],
        [ 0.007634283,  0.000638150, -0.999970655],
    ])

    t_cam1_cam0 = np.array([
         0.000538979,
        -0.001021010,
        -0.040041768,
    ])

    cam1_from_cam0 = pycolmap.Rigid3d(
        pycolmap.Rotation3d(R_cam1_cam0),
        t_cam1_cam0,
    )

    # cam0 is the reference sensor.
    cam0_config = pycolmap.RigConfigCamera()
    cam0_config.image_prefix = "cam0/"
    cam0_config.ref_sensor = True

    # cam1 is calibrated relative to cam0.
    cam1_config = pycolmap.RigConfigCamera()
    cam1_config.image_prefix = "cam1/"
    cam1_config.cam_from_rig = cam1_from_cam0

    rig_config = pycolmap.RigConfig()
    rig_config.cameras = [
        cam0_config,
        cam1_config,
    ]

    # ---------------------------------------------------------
    # 8. Populate database
    # ---------------------------------------------------------

    with pycolmap.Database.open(database_path) as db:

        # Configure rig BEFORE geometric verification/mapping.
        pycolmap.apply_rig_config(
            [rig_config],
            db,
        )

        print("[rig] Configured stereo rig:")
        print(f"      Reference: cam0")
        print(f"      Secondary: cam1")
        print(f"      Baseline: {np.linalg.norm(t_cam1_cam0):.5f} m")

        # Import local features.
        hloc_reconstruction.import_features(
            image_ids,
            db,
            local_feats,
        )

        # Import matches.
        hloc_reconstruction.import_matches(
            image_ids,
            db,
            pairs_path,
            matches,
        )

    # ---------------------------------------------------------
    # 9. Geometric verification
    # ---------------------------------------------------------

    hloc_reconstruction.estimation_and_geometric_verification(
        database_path,
        pairs_path,
    )

    # ---------------------------------------------------------
    # 10. Rig-aware incremental mapping
    # ---------------------------------------------------------

    mapper_options = {
        "ba_refine_sensor_from_rig": False,
    }

    reconstruction = hloc_reconstruction.run_reconstruction(
        sfm_dir,
        database_path,
        unified,
        options=mapper_options,
    )

    if reconstruction is None:
        raise RuntimeError("COLMAP failed to reconstruct the scene.")

    print("\n[reconstruction]")
    print(reconstruction.summary())

    return {
        "sfm_dir": sfm_dir,
        "image_list": image_list,
        "local_feats": local_feats,
        "matches": matches,
        "pairs": pairs_path,
        "reconstruction": reconstruction,
    }



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
