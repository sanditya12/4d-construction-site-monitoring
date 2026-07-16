import os
import cv2
import struct
import numpy as np
from plyfile import PlyData, PlyElement

# ==========================================
# 1. CONFIGURATION
# ==========================================
# Ensure your trained splat is in the project folder
INPUT_SPLAT = "dataset/kitchen_t2/splat/GS_mask.ply" 
OUTPUT_SPLAT = "result/splat/encoded_splat_t2.ply"

MASK_DIR = "dataset/kitchen_t2/masks/"
SPARSE_DIR = "dataset/kitchen_t2/sparse/0/"

MIN_VOTES = 10  # The splat must be inside the white mask in at least 3 photos

# ==========================================
# 2. COLMAP BINARY PARSERS
# ==========================================
def read_cameras_binary(path):
    cameras = {}
    with open(path, "rb") as fid:
        num_cameras = struct.unpack("<Q", fid.read(8))[0]
        for _ in range(num_cameras):
            camera_properties = struct.unpack("<iiQQ", fid.read(24))
            camera_id, model_id, width, height = camera_properties
            
            # Identify exactly how many parameters COLMAP saved for this lens
            if model_id == 1: num_params = 3   # SIMPLE_PINHOLE
            elif model_id == 2: num_params = 4 # PINHOLE
            elif model_id == 3: num_params = 4 # SIMPLE_RADIAL
            elif model_id == 4: num_params = 5 # RADIAL
            elif model_id == 5: num_params = 8 # OPENCV
            elif model_id == 6: num_params = 8 # OPENCV_FISHEYE
            else:
                raise ValueError(f"Camera model {model_id} not supported.")
                
            # Safely read the exact number of bytes needed
            params = struct.unpack("<" + "d" * num_params, fid.read(8 * num_params))
            
            # Map the specific parameters to the focal length and optical center
            if model_id in [1, 3]: # Models with a single shared focal length
                fx, fy = params[0], params[0]
                cx, cy = params[1], params[2]
            elif model_id in [2, 4, 5, 6]: # Models with independent X and Y focal lengths
                fx, fy = params[0], params[1]
                cx, cy = params[2], params[3]
                
            cameras[camera_id] = {
                "w": width, "h": height, 
                "fx": fx, "fy": fy, 
                "cx": cx, "cy": cy,
                "model_id": model_id
            }
    return cameras

def read_images_binary(path):
    images = {}
    with open(path, "rb") as fid:
        num_reg_images = struct.unpack("<Q", fid.read(8))[0]
        for _ in range(num_reg_images):
            binary_image_properties = struct.unpack("<idddddddi", fid.read(64))
            image_id = binary_image_properties[0]
            qw, qx, qy, qz = binary_image_properties[1:5]
            tx, ty, tz = binary_image_properties[5:8]
            camera_id = binary_image_properties[8]
            
            name = ""
            current_char = struct.unpack("<c", fid.read(1))[0]
            while current_char != b"\x00":
                name += current_char.decode("utf-8")
                current_char = struct.unpack("<c", fid.read(1))[0]
                
            num_points2D = struct.unpack("<Q", fid.read(8))[0]
            fid.read(24 * num_points2D) # Skip 2D points data
            
            # Quaternion to Rotation Matrix
            R = np.array([
                [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
                [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
                [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
            ])
            T = np.array([tx, ty, tz])
            images[name] = {"cam_id": camera_id, "R": R, "T": T}
    return images

# ==========================================
# 3. LOAD DATA & PROJECT
# ==========================================
print("Loading pristine Gaussian Splat... (This preserves all original colors)")
plydata = PlyData.read(INPUT_SPLAT)
vertex_data = plydata['vertex'].data

# Extract X, Y, Z for projection
pts = np.vstack([vertex_data['x'], vertex_data['y'], vertex_data['z']]).T
num_splats = pts.shape[0]

print("Parsing COLMAP camera matrices...")
cameras = read_cameras_binary(os.path.join(SPARSE_DIR, "cameras.bin"))
images = read_images_binary(os.path.join(SPARSE_DIR, "images.bin"))

votes = np.zeros(num_splats, dtype=np.uint8)
processed = 0

print("Projecting splats against 131 2D masks...")
for img_name, img_data in images.items():
    # The mask should share the exact filename as the original image
    mask_path = os.path.join(MASK_DIR, img_name)
    if not os.path.exists(mask_path):
        # Swap extension to .png if necessary
        base_name = os.path.splitext(img_name)[0]
        mask_path = os.path.join(MASK_DIR, base_name + ".png")
        if not os.path.exists(mask_path):
            continue

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None: continue

    cam = cameras[img_data["cam_id"]]
    R, T = img_data["R"], img_data["T"]

    # Transform to camera space
    pts_cam = (R @ pts.T).T + T
    z = pts_cam[:, 2]
    
    # Ignore splats behind the camera
    valid_depth = z > 0.01 

    # Project to 2D pixels
    u = np.round((pts_cam[:, 0] * cam["fx"] / z) + cam["cx"]).astype(int)
    v = np.round((pts_cam[:, 1] * cam["fy"] / z) + cam["cy"]).astype(int)
    
    # Ensure pixels are within image bounds
    valid_u = (u >= 0) & (u < cam["w"])
    valid_v = (v >= 0) & (v < cam["h"])
    
    valid = valid_depth & valid_u & valid_v
    valid_indices = np.where(valid)[0]
    
    # Check if the mask pixel is white (> 127)
    hits = mask[v[valid], u[valid]] > 127
    
    # Tally votes
    votes[valid_indices[hits]] += 1
    processed += 1

print(f"Processed {processed} masks. {np.sum(votes >= MIN_VOTES)} splats passed the threshold.")

# ==========================================
# 4. INJECT METADATA & SAVE
# ==========================================
print("Injecting 'isolated_object' metadata column...")
winners = (votes >= MIN_VOTES).astype(np.uint8)

# Create a new data schema that appends our custom column
old_descr = vertex_data.dtype.descr
new_descr = old_descr + [('isolated_object', 'u1')]

# Build the new array structure
new_vertex_data = np.empty(num_splats, dtype=new_descr)

# Copy the original 62+ columns perfectly
for name in vertex_data.dtype.names:
    new_vertex_data[name] = vertex_data[name]

# Insert the binary mask data
new_vertex_data['isolated_object'] = winners

print(f"Saving final file to {OUTPUT_SPLAT}...")
new_element = PlyElement.describe(new_vertex_data, 'vertex')
PlyData([new_element], text=False).write(OUTPUT_SPLAT)

print("✅ Complete! Your splat file now has embedded semantic metadata.")