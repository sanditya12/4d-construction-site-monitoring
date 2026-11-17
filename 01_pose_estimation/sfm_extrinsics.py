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
from hloc import reconstruction as hloc_reconstruction

# CONFIGS

RETRIEVAL_CONF = extract_features.confs["netvlad"]
FEATURE_CONF = extract_features.confs["superpoint_aachen"]
MATCHER_CONF = match_features.confs["superglue"]
MATCHER_CONF["model"]["weights"] = "indoor"


# for sequential matching
def generate_sequential_pairs(
    image_list: list,
    pairs_path: Path,
    window: int = 10,
):
    """
    Generate bidirectional sequential pairs.
    """

    pairs = set()
    image_list = sorted(image_list)

    half_window = window // 2

    for i, image1 in enumerate(image_list):

        start = max(0, i - half_window)
        end = min(len(image_list), i + half_window + 1)

        for j in range(start, end):
            if i == j:
                continue

            image2 = image_list[j]

            pair = tuple(sorted((image1, image2)))
            pairs.add(pair)

    pairs_path.parent.mkdir(parents=True, exist_ok=True)

    with open(pairs_path, "w") as f:
        for image1, image2 in sorted(pairs):
            f.write(f"{image1} {image2}\n")

    print(
        f"[sequential] Generated {len(pairs)} pairs "
        f"(±{half_window} images)"
    )

    return list(pairs)



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


    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # 1. Organize images into camera folders
    # ---------------------------------------------------------
    def tag(p):  # e.g. t1_000123.jpg
        return f"{session}_{p.name}"

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

    # symlink
    for p in files0:
        dst = cam0_dir / tag(p)
        if not dst.exists():
            dst.symlink_to(p.resolve())
    for p in files1:
        dst = cam1_dir / tag(p)
        if not dst.exists():
            dst.symlink_to(p.resolve())

    image_list = (
        [f"cam0/{tag(p)}" for p in files0]
        + [f"cam1/{tag(p)}" for p in files1]
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
    # 2.5. Make masks visible to hloc's extractor inside 'unified'
    # ---------------------------------------------------------
    if colmap_masks_dir is not None and colmap_masks_dir.exists():
        unified_masks = unified / "masks"
        if not unified_masks.exists():
            unified_masks.symlink_to(colmap_masks_dir.resolve(), target_is_directory=True)

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

    pairs_cam0 = [f"cam0/{tag(p)}" for p in files0]
    pairs_cam1 = [f"cam1/{tag(p)}" for p in files1]

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
        pairs.add(f"cam0/{session}_{name} cam1/{session}_{name}")

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

    # --- Using filtered matches ---

    matches = out_dir / "matches_filtered.h5"

    if not matches.exists():
        raise FileNotFoundError(
            f"Filtered matches file not found: {matches}"
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

    # Parameter order for OPENCV_FISHEYE:
    # [fx, fy, cx, cy, k1, k2, k3, k4]

    cam0_params = [
        465.301548,
        465.323037,
        740.954411,
        718.857299,
        0.025800719,
        -0.010909241,
        -0.0016899538,
        0.0001476680,
    ]

    cam1_params = [
        465.436302,
        465.471994,
        737.563603,
        720.259128,
        0.024723386,
        -0.010754674,
        -0.0016168994,
        0.0001367535,
    ]

    # Assign the parameters to the correct camera in the database.
    with pycolmap.Database.open(database_path) as db:

        images = db.read_all_images()

        for camera in db.read_all_cameras():

            # Find which image folders use this camera.
            camera_images = [
                image.name
                for image in images
                if image.camera_id == camera.camera_id
            ]

            folders = {
                name.split("/", 1)[0]
                for name in camera_images
            }

            if folders == {"cam0"}:
                params = cam0_params
                camera_name = "cam0"

            elif folders == {"cam1"}:
                params = cam1_params
                camera_name = "cam1"

            else:
                raise RuntimeError(
                    f"Unexpected camera assignment: "
                    f"camera_id={camera.camera_id}, folders={folders}"
                )

            # Set the calibrated parameters.
            camera.set_params_from_string(
                ",".join(map(str, params))
            )

            # Save the updated camera to the database.
            db.update_camera(camera)

            print(
                f"[calibration] {camera_name}: "
                f"camera_id={camera.camera_id}, "
                f"images={len(camera_images)}, "
                f"params={camera.params}"
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

    with pycolmap.Database.open(database_path) as db:
        geometries = db.read_two_view_geometries()
        matches = db.read_all_matches()

        print("\n========== DATABASE DIAGNOSTICS ==========")

        print("Images:", db.num_images())
        print("Matches:", db.num_matches())

        print("\nGeometry return type:")
        print(type(geometries))

        print("Geometry tuple length:", len(geometries))

        for i, item in enumerate(geometries):
            print(
                f"Geometry item {i}: "
                f"type={type(item)}, "
                f"length={len(item) if hasattr(item, '__len__') else 'N/A'}"
            )

            if isinstance(item, (list, tuple)) and len(item) > 0:
                print("  First element type:", type(item[0]))

        print("\nMatches return type:")
        print(type(matches))

        print("Matches tuple length:", len(matches))

        for i, item in enumerate(matches):
            print(
                f"Match item {i}: "
                f"type={type(item)}, "
                f"length={len(item) if hasattr(item, '__len__') else 'N/A'}"
            )

        print("\n==========================================")
    # ---------------------------------------------------------
    # 10. Rig-aware incremental mapping
    # ---------------------------------------------------------

    mapper_options = {
        "ba_refine_sensor_from_rig": False,
        "ba_refine_focal_length": False,
        "ba_refine_principal_point": False,
        "ba_refine_extra_params": False,
    }

    reconstruction = hloc_reconstruction.run_reconstruction(
        sfm_dir,
        database_path,
        unified,
        verbose= True,
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


###### main ######


def main():
    root = "/home/kanishka/4d_reconstruction/registration"
    root = Path(root)

    target_month = "july"
    sessions= {"may":"t1", "july":"t2", "december":"t3"}

    t1_out = root / f"floor1/outputs/test/sfm_{target_month}_clean"

    print(f"Starting clean SfM reconstruction for [{target_month}] (1:1 frames and masks)...")

    t1_result = build_sfm_mask_extrinsics(
            root=root,
            images_cam0=root / f"floor1/data/test/images/cam0",
            images_cam1=root / f"floor1/data/test/images/cam1",
            out_dir=t1_out,
            masks_cam0=root / f"floor1/data/test/masks/cam0",
            masks_cam1=root / f"floor1/data/test/masks/cam1",
            seq_window=5,
            session= sessions[target_month]
        )
    
    sfm_dir = t1_result["sfm_dir"]
    print(f"\nClean reconstruction successfully saved at: {sfm_dir}")

    # utils.dump_t1_cameras_txt(sfm_dir, t1_out / "sfm_txt")


if __name__ == "__main__":
    main()