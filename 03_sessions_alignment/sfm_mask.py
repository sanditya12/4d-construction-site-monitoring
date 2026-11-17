from pathlib import Path
# import utils_old as utils
import utils



def main():
    root = "/home/kanishka/4d_reconstruction/registration"
    root = Path(root)

    target_month = "july"
    sessions= {"may":"t1", "july":"t2", "december":"t3"}

    # t1_out = root / f"floor1/outputs/subsampled_seq_vpr/sfm_{target_month}_clean"
    t1_out = Path("/home/kanishka/4d_reconstruction/registration/floor1/outputs/nov_14/sfm_july_clean")

    print(f"Starting clean SfM reconstruction for [{target_month}] (1:1 frames and masks)...")

    t1_result = utils.build_sfm_mask(
        root=root,
        images_cam0=root / f"floor1/data/subsampled/{target_month}/sampled_rotated_frames/cam0",
        images_cam1=root / f"floor1/data/subsampled/{target_month}/sampled_rotated_frames/cam1",
        out_dir=t1_out,
        masks_cam0=root / f"floor1/data/subsampled/{target_month}/masks_sam3_clean/cam0",
        masks_cam1=root / f"floor1/data/subsampled/{target_month}/masks_sam3_clean/cam1",
        seq_window=20,
        session= sessions[target_month]
    )
    
    sfm_dir = t1_result["sfm_dir"]
    print(f"\nClean reconstruction successfully saved at: {sfm_dir}")

    # utils.dump_t1_cameras_txt(sfm_dir, t1_out / "sfm_txt")


if __name__ == "__main__":
    main()