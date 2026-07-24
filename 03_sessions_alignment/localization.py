from pathlib import Path
import utils


def main():
    # root = "/home/kanishka/4d_reconstruction/registration"
    # root = Path(root)
    root = Path(__file__).resolve().parents[1]

    t1_images = root / "data/kitchen_v1/kitchen_t1/frames/camera_0"
    t2_images = root / "data/kitchen_v1/kitchen_t2/frames/camera_0"

    t1_out = root / "data/outputs/t1_sfm"
    t2_out = root / "data/outputs/t2_localize"

    # t1_out = root / "outputs/out1/t1_sfm"
    # t2_out = root / "outputs/out1/t2_localize"


    queries_file = root / "outputs/queries_with_intrinsics.txt"

    # using t1 sfm reconstruction
    sfm_dir = root / "data/kitchen_sfm_t1"
    print(f"t1 reconstruction from: {sfm_dir}")

    # t1_result = utils.build_t1_reconstruction(
    #     root=root,
    #     t1_images=t1_images,
    #     out_dir=t1_out,
    #     sequential=False,       
    #     seq_window=10,
    #     num_retrieval_matches=10,
    # )
    # sfm_dir = t1_result["sfm_dir"]

    utils.dump_t1_cameras_txt(sfm_dir, t1_out / "sfm_txt")

    

    # ------------------------------------------------------------------
    # Localize t2 images against t1 map
    # ------------------------------------------------------------------
    # camera parameters
    fx, fy, cx, cy = 1031.98, 1032.67, 1920.0, 1920.0
    k1, k2, k3, k4 = 0.0348231, -0.0017807, 0.00474112, -0.00197457
    w, h = 3840, 3840

    with open( queries_file, "w") as f:
        for img in sorted((t2_images).iterdir()):
            f.write(f"t2_{img.name} OPENCV_FISHEYE {w} {h} {fx} {fy} {cx} {cy} {k1} {k2} {k3} {k4}\n")
            # f.write(f"{img.name} OPENCV_FISHEYE {w} {h} {fx} {fy} {cx} {cy} {k1} {k2} {k3} {k4}\n")
            # f.write(f"data/kitchen_v1/kitchen_t2/frames/camera_0/{img.name} OPENCV_FISHEYE {w} {h} {fx} {fy} {cx} {cy} {k1} {k2} {k3} {k4}\n")

    if not queries_file.exists():
        print(
            f"\n!! {queries_file} not found.\n"
            "Create it before running localization (see file docstring).\n"
        )
        return

    print("\nLocalizing t2 images against t1 map ===")
    results_path = utils.localize_t2(
        root=root,
        t1_images=t1_images,
        t2_images=t2_images,
        sfm_dir=sfm_dir,
        out_dir=t2_out,
        queries_file=queries_file,
        num_retrieval_matches=10,
    )
    print(f"t2 poses written to: {results_path}")


if __name__ == "__main__":
    main()