# Training

Run training with:

```bash
python train.py \
  --config-name apps/colmap_3dgut.yaml \
  path=data/transformed_t2 \
  out_dir=runs \
  experiment_name=transformed_t2 \
  compute_extra_metrics=False \
  export_ply.enabled=True \
  export_ply.path=runs/kitchen_opt0/tra_t2.ply \
  n_iterations=10000
```

## Expected Dataset Structure

The `path` argument should point to a directory with the following structure:

```text
transformed_t2/
├── images/
│   ├── camera_0/
│   └── camera_1/
├── masks/
│   ├── camera_0/
│   └── camera_1/
├── sparse/
│   └── 0/
│       ├── cameras.bin
│       ├── images.bin
│       └── points3D.bin
└── render_mask.png
```

### Directory Contents

* `images/` — Original/raw input images, organized by camera.
* `masks/` — Optional/dynamic masks, organized by camera.
* `sparse/` — COLMAP reconstruction output.
* `sparse/0/` — COLMAP binary model containing:

  * `cameras.bin`
  * `images.bin`
  * `points3D.bin`
* `render_mask.png` — Mask used for rendering.

---

# Rendering a Custom Pose

After training, render an image from a custom camera pose using:

```bash
python render_custom_pose.py \
  --checkpoint runs/kitchen_t1/frames-2506_120748/ckpt_last.pt \
  --tx 0.5 \
  --ty 0.2 \
  --tz 1.5 \
  --ry 90 \
  --name view01.png \
  --intrinsic_id 1 \
  --out_dir my_render
```

## Arguments

| Argument         | Description                                            |
| ---------------- | ------------------------------------------------------ |
| `--checkpoint`   | Path to the trained model checkpoint (`ckpt_last.pt`). |
| `--tx`           | Translation along the X-axis.                          |
| `--ty`           | Translation along the Y-axis.                          |
| `--tz`           | Translation along the Z-axis.                          |
| `--ry`           | Rotation around the Y-axis, in degrees.                |
| `--name`         | Filename of the rendered image.                        |
| `--intrinsic_id` | Selects which camera intrinsic parameters to use.      |
| `--out_dir`      | Directory where the rendered image is saved.           |

### Selecting a Camera

Use `--intrinsic_id` to select the camera whose intrinsic parameters are used for rendering.

For example:

```bash
--intrinsic_id 1
```

selects camera/intrinsic ID `1`.
