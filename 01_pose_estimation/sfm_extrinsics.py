"""
Stereo-rig SfM with hloc + COLMAP.
note: ver without symlinks
"""

import copy
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import cv2
import h5py
import numpy as np
import pycolmap
from hloc import extract_features, match_features
from hloc import reconstruction as hloc_reconstruction

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
FEATURE_CONF = extract_features.confs["superpoint_aachen"]
MATCHER_CONF = copy.deepcopy(match_features.confs["superglue"])
MATCHER_CONF["model"]["weights"] = "indoor"

# OPENCV_FISHEYE parameter order: fx, fy, cx, cy, k1, k2, k3, k4
CAMERA_PARAMS = {
    "cam0": [465.301548, 465.323037, 740.954411, 718.857299,
             0.025800719, -0.010909241, -0.0016899538, 0.0001476680],
    "cam1": [465.436302, 465.471994, 737.563603, 720.259128,
             0.024723386, -0.010754674, -0.0016168994, 0.0001367535],
}

# Calibrated extrinsics: cam1 expressed relative to cam0 (the reference).
R_CAM1_FROM_CAM0 = np.array([
    [-0.999803492,  0.018300057, -0.007621329],
    [ 0.018294656,  0.999832336,  0.000777732],
    [ 0.007634283,  0.000638150, -0.999970655],
])
T_CAM1_FROM_CAM0 = np.array([0.000538979, -0.001021010, -0.040041768])

MAPPER_OPTIONS = {
    "ba_refine_sensor_from_rig": False,
    "ba_refine_focal_length": False,
    "ba_refine_principal_point": False,
    "ba_refine_extra_params": False,
}


@dataclass(frozen=True)
class Layout:
    """Every file a run produces, in one place."""
    out_dir: Path

    @property
    def features(self): return self.out_dir / "features.h5"
    @property
    def pairs(self): return self.out_dir / "pairs.txt"
    @property
    def matches(self): return self.out_dir / "matches.h5"
    @property
    def matches_filtered(self): return self.out_dir / "matches_filtered.h5"
    @property
    def features_masked(self): return self.out_dir / "features_masked.h5"
    @property
    def sfm(self): return self.out_dir / "sfm"
    @property
    def database(self): return self.sfm / "database.db"


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def list_image_files(folder: Path) -> list[str]:
    return sorted(p.name for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def sequential_pairs(names: list[str], window: int) -> set[tuple[str, str]]:
    """Pairs of images within +-window//2 positions of each other."""
    half = window // 2
    pairs = set()
    for i, a in enumerate(names):
        for j in range(max(0, i - half), min(len(names), i + half + 1)):
            if i != j:
                pairs.add(tuple(sorted((a, names[j]))))
    return pairs


def write_pairs(pairs, path: Path) -> None:
    with open(path, "w") as f:
        for a, b in sorted(pairs):
            f.write(f"{a} {b}\n")


def mask_features(features_in: Path, features_out: Path,
                  mask_dirs: dict[str, Path], dilation_px: int = 15,
                  white_is_ignore: bool = False, min_keep: int = 100) -> Path:
 
    stems = {prefix: {q.stem: q for q in d.iterdir() if q.suffix.lower() in IMAGE_EXTS}
             for prefix, d in mask_dirs.items()}
    kernel = None
    if dilation_px > 0:
        k = 2 * dilation_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    n_before = n_after = n_masked_images = 0
    with h5py.File(features_in, "r") as fin, h5py.File(features_out, "w") as fout:
        names = []

        def collect(name, obj):
            if isinstance(obj, h5py.Group) and "keypoints" in obj:
                names.append(name)

        fin.visititems(collect)

        for name in names:
            g_in = fin[name]
            kp = g_in["keypoints"][()]
            keep = np.ones(len(kp), dtype=bool)

            prefix, _, filename = name.rpartition("/")
            src = stems.get(prefix, {}).get(Path(filename).stem)
            if src is not None:
                mask = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    ignore = ((mask > 127) if white_is_ignore else (mask <= 127)).astype(np.uint8)
                    if kernel is not None:
                        ignore = cv2.dilate(ignore, kernel)
                    w, h = (int(v) for v in g_in["image_size"][()])
                    if ignore.shape != (h, w):
                        ignore = cv2.resize(ignore, (w, h), interpolation=cv2.INTER_NEAREST)
                    xi = np.clip(np.round(kp[:, 0]).astype(int), 0, w - 1)
                    yi = np.clip(np.round(kp[:, 1]).astype(int), 0, h - 1)
                    keep = ignore[yi, xi] == 0
                    n_masked_images += 1

            g_out = fout.create_group(name)
            for k_, v in g_in.attrs.items():
                g_out.attrs[k_] = v
            for key, ds in g_in.items():
                data = ds[()]
                if key in ("keypoints", "scores"):
                    data = data[keep]
                elif key == "descriptors":
                    data = data[:, keep]
                out_ds = g_out.create_dataset(key, data=data)
                for k_, v in ds.attrs.items():
                    out_ds.attrs[k_] = v

            n_before += len(kp)
            n_after += int(keep.sum())
            if keep.sum() < min_keep:
                print(f"[mask][warning] {name}: only {int(keep.sum())} keypoints left")

    print(f"[mask] {n_masked_images}/{len(names)} images masked; "
          f"keypoints {n_before} -> {n_after} ({100 * (1 - n_after / max(n_before, 1)):.1f}% removed)")
    return features_out


def assign_intrinsics(database_path: Path, folder_params: dict[str, list[float]]) -> None:
    """Set calibrated parameters on each camera, identified by image folder."""
    with pycolmap.Database.open(database_path) as db:
        images = db.read_all_images()
        for camera in db.read_all_cameras():
            names = [im.name for im in images if im.camera_id == camera.camera_id]
            folders = {Path(n).parent.as_posix() for n in names}
            if len(folders) != 1 or next(iter(folders)) not in folder_params:
                raise RuntimeError(
                    f"camera_id={camera.camera_id} spans unexpected folders: {folders}"
                )
            folder = next(iter(folders))
            camera.set_params_from_string(",".join(map(str, folder_params[folder])))
            db.update_camera(camera)
            print(f"[calibration] {folder}: camera_id={camera.camera_id}, "
                  f"images={len(names)}, params={camera.params}")


def make_rig_config(prefix0: str, prefix1: str) -> pycolmap.RigConfig:
    cam0 = pycolmap.RigConfigCamera()
    cam0.image_prefix = f"{prefix0}/"
    cam0.ref_sensor = True

    cam1 = pycolmap.RigConfigCamera()
    cam1.image_prefix = f"{prefix1}/"
    cam1.cam_from_rig = pycolmap.Rigid3d(
        pycolmap.Rotation3d(R_CAM1_FROM_CAM0), T_CAM1_FROM_CAM0
    )

    rig = pycolmap.RigConfig()
    rig.cameras = [cam0, cam1]
    return rig


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------
def build_stereo_sfm(
    cam0_dir: Path,
    cam1_dir: Path,
    out_dir: Path,
    masks_cam0: Optional[Path] = None,
    masks_cam1: Optional[Path] = None,
    image_root: Optional[Path] = None,
    seq_window: int = 10,
    mask_dilation_px: int = 0,
    mask_white_is_ignore: bool = True,
    match_filter: Optional[Callable[..., Path]] = None,
    overwrite: bool = False,
):
    """
    masks_cam0/1: folders of source masks. Keypoints inside the masks (grown
                  by mask_dilation_px) are removed BEFORE matching.
    match_filter: optional callable(matches_path=, features_path=, output_path=)
                  returning the filtered matches path, e.g.
                  filter_matches.filter_matches.
    overwrite:    delete out_dir first for a guaranteed clean run.
    """
    cam0_dir, cam1_dir, out_dir = Path(cam0_dir).resolve(), Path(cam1_dir).resolve(), Path(out_dir)

    # image_root : hloc/COLMAP image directory.
    image_root = Path(image_root).resolve() if image_root else Path(
        os.path.commonpath([cam0_dir, cam1_dir])
    )
    prefix0 = cam0_dir.relative_to(image_root).as_posix()
    prefix1 = cam1_dir.relative_to(image_root).as_posix()
    if prefix0 == "." or prefix1 == "." or prefix0 == prefix1:
        raise ValueError(f"cam folders must be distinct subfolders of image_root: {prefix0}, {prefix1}")

    if overwrite and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = Layout(out_dir)

    # 1. Image names ------------------------------------------------------
    files0, files1 = list_image_files(cam0_dir), list_image_files(cam1_dir)
    common = sorted(set(files0) & set(files1))
    if not common:
        raise ValueError("No matching filenames between cam0 and cam1.")
    if len(common) != len(files0) or len(common) != len(files1):
        print(f"[warning] {len(files0) - len(common)} cam0 / {len(files1) - len(common)} "
              f"cam1 images have no stereo partner.")

    names0 = [f"{prefix0}/{n}" for n in files0]
    names1 = [f"{prefix1}/{n}" for n in files1]
    image_list = names0 + names1
    print(f"[images] root={image_root}  {prefix0}: {len(names0)}  {prefix1}: {len(names1)}")

    # 2. Local features ---------------------------------------------------
    local_feats = extract_features.main(
        FEATURE_CONF, image_root, out_dir,
        image_list=image_list, feature_path=paths.features,
    )

    # 3. Remove masked keypoints before matching --------------------------
    mask_dirs = {pre: d for pre, d in ((prefix0, masks_cam0), (prefix1, masks_cam1))
                 if d is not None}
    if mask_dirs:
        local_feats = mask_features(
            paths.features, paths.features_masked,
            {pre: Path(d) for pre, d in mask_dirs.items()},
            dilation_px=mask_dilation_px, white_is_ignore=mask_white_is_ignore,
        )
        # Keypoint indices changed, so matches from earlier runs are invalid.
        paths.matches.unlink(missing_ok=True)

    # 4. Pairs: sequential --------------------------------------------- --
    pairs = sequential_pairs(names0, seq_window) | sequential_pairs(names1, seq_window)
    pairs |= {(f"{prefix0}/{n}", f"{prefix1}/{n}") for n in common}
    write_pairs(pairs, paths.pairs)
    print(f"[pairs] {len(pairs)} total")

    # 5. Matching (+ optional filtering) ----------------------------------
    match_path = match_features.main(
        MATCHER_CONF, paths.pairs, local_feats, matches=paths.matches,
    )
    if match_filter is not None:
        print("Filtering matches...")
        match_path = match_filter(
            matches_path=match_path,
            features_path=local_feats,
            output_path=paths.matches_filtered,
        )

    # 6. COLMAP database + rig --------------------------------------------
    paths.sfm.mkdir(parents=True, exist_ok=True)
    hloc_reconstruction.create_empty_db(paths.database)

    # PER_FOLDER -> one camera per image folder (cam0, cam1).
    hloc_reconstruction.import_images(
        image_root, paths.database, pycolmap.CameraMode.PER_FOLDER,
        image_list=image_list, options={"camera_model": "OPENCV_FISHEYE"},
    )
    assign_intrinsics(paths.database, {prefix0: CAMERA_PARAMS["cam0"],
                                       prefix1: CAMERA_PARAMS["cam1"]})
    image_ids = hloc_reconstruction.get_image_ids(paths.database)

    with pycolmap.Database.open(paths.database) as db:
        # Rig must be configured before geometric verification / mapping.
        pycolmap.apply_rig_config([make_rig_config(prefix0, prefix1)], db)
        print(f"[rig] ref={prefix0}  second={prefix1}  "
              f"baseline={np.linalg.norm(T_CAM1_FROM_CAM0):.5f} m")
        hloc_reconstruction.import_features(image_ids, db, local_feats)
        hloc_reconstruction.import_matches(image_ids, db, paths.pairs, match_path)

    # 7. Geometric verification + mapping ---------------------------------
    hloc_reconstruction.estimation_and_geometric_verification(paths.database, paths.pairs)

    model = hloc_reconstruction.run_reconstruction(
        paths.sfm, paths.database, image_root, verbose=True, options=MAPPER_OPTIONS,
    )
    if model is None:
        raise RuntimeError("COLMAP failed to reconstruct the scene.")
    print("\n[reconstruction]")
    print(model.summary())

    return {
        "paths": paths,
        "image_root": image_root,
        "image_list": image_list,
        "reconstruction": model,
    }


# --------------------------------------------------------------------------
def main():
    # root: Project folder
    root = Path("/home/kanishka/4d_reconstruction/registration")
    session = "may"

    data = root / "floor1/data/subsampled/july" 
    out_dir = root / "floor1/outputs/nov_15" / f"sfm_{session}"

    result = build_stereo_sfm(
        cam0_dir=data / "sampled_rotated_frames/cam0",
        cam1_dir=data / "sampled_rotated_frames/cam1",
        out_dir=out_dir,
        masks_cam0=data / "masks_sam3_clean/cam0",
        masks_cam1=data / "masks_sam3_clean/cam1",
        seq_window=20,
        overwrite=True,
    )
    print(f"\nSaved at: {result['paths'].sfm}")


if __name__ == "__main__":
    main()