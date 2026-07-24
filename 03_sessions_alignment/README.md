# 03. Multisession Alignment

### Objective:

Align two COLMAP reconstructions captured at different points in time using image-based matching and Hierarchical-Localization (hloc) by Paul-Edouard Sarlin et al.

Given two sets of raw images and their corresponding SfM models, the pipeline recovers a similarity transform (R, t, s) that maps the second session's coordinate frame into the first.

### Usage

**Step 1 — Localize t2 images against the t1 map**
 
Edit the paths and intrinsics at the top of `localization.py`, then run:
```bash
python code/localization.py
```
 
This produces `outputs/t2_localize/t2_poses.txt` containing the estimated pose of each t2 image in the t1 coordinate frame.
 
**Step 2 — Compute the similarity transform**
 
```bash
python code/alignment.py
```
 
This reads the t2 poses and the t1 COLMAP model and outputs the similarity transform (R, t, s) that maps the t2 SfM frame into the t1 frame, saved to `outputs/alignment.npz` and `outputs/alignment.txt`.
 
**Intrinsics**
 
Set your camera intrinsics in `localization.py`:
```python
fx, fy, cx, cy = 1031.98, 1032.67, 1920.0, 1920.0
k1, k2, k3, k4 = 0.0348231, -0.0017807, 0.00474112, -0.00197457
w, h = 3840, 3840
```
 
The camera model is `OPENCV_FISHEYE`. To read the intrinsics estimated by COLMAP during t1 reconstruction, run:
```python
utils.dump_t1_cameras_txt(sfm_dir, output_dir)
```
 
---
### Output
 
`alignment.txt` contains the similarity transform mapping t2 frame → t1 frame:
```
scale: 1.0012345
rotation:
   0.9998  -0.0123   0.0045
   0.0124   0.9997  -0.0021
  -0.0042   0.0022   0.9999
translation:
   0.1234   -0.0456   0.2345
```
 
`alignment.npz` contains the same as numpy arrays (`R`, `t`, `s`, `inlier_mask`).
 
To apply the transform to a point X in the t2 frame:
```python
X_t1 = s * R @ X_t2 + t
```
 
---
