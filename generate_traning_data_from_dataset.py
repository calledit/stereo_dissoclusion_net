import os
import cv2
import zlib
import zipfile
import numpy as np

import sys
sys.path.append("C:\\Users\\calle\\projects\\metric_depth_video_toolbox")
import depth_frames_helper
import depth_map_tools
import video_da3
import unik3d_video
from contextlib import redirect_stdout
import torch
from skimage.metrics import structural_similarity as ssim
import random
dataset_folder = "unlabeled2017.zip"
#dataset_folder = "dataset_imgs"
training_output_folder = "training_data"+os.sep

depth_resolution = 504#252 #we need to pick a resolution that aligns with the depth model so we can skip rescaling after the depth model da3 supports resolutions like 252, 504 ...
MODEL_maxOUTPUT_depth = 100.0#max 100m cut the rest of it wont make much paralx so it does not matter
save_resolution = 254

da3model = None
unk3dmodel = None


def rename(from_path, to_path):
    if os.path.exists(to_path) and os.path.exists(from_path):
        os.remove(to_path)
    os.rename(from_path, to_path)

# ---------------------------------------------------------
# convert_to_training_data
# ---------------------------------------------------------

def downsample_mask_maxpool(mask, factor=2):
    H, W = mask.shape
    
    # pad to even
    pad_h = (factor - H % factor) % factor
    pad_w = (factor - W % factor) % factor

    mask_padded = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="edge")

    # reshape into blocks and OR
    new_h = mask_padded.shape[0] // factor
    new_w = mask_padded.shape[1] // factor

    return mask_padded.reshape(new_h, factor, new_w, factor).any(axis=(1, 3))

mesh = None
def convert_to_training_data(img_path, from_zip=False, zip_ref=None):
    global mesh
    """
    Loads an image from either:
      - a filesystem path (from_zip=False)
      - or from inside a zip file (from_zip=True, zip_ref must be a ZipFile)
    
    Variable naming convention asumes you staart with a left image and generate a right one  but that is just naming convention it can be flipped using do_right = False
    
    """
    
    name_only = os.path.splitext(os.path.basename(img_path))[0]
    crc_hex = f"{zlib.crc32(name_only.encode()):08x}"  # 8 hex chars
    subfolder = crc_hex[0]
    img_output_folder = training_output_folder + subfolder + os.sep
    
    meta_output = img_output_folder + name_only + '.txt'
    img_output = img_output_folder + name_only + '.png'
    img_output_depth = img_output + "_depth.png"
    img_output_gen = img_output + "_gen.png"
    img_output_normal_gen = img_output + "_gen_normal.png"
    img_output_depth_gen = img_output + "_gen_depth.png"
    
    
    tmp_file_str = "_tmp.png"
    
    #if os.path.exists(img_output) and os.path.exists(img_output_depth) and os.path.exists(img_output_gen) and os.path.exists(img_output_normal_gen) and os.path.exists(img_output_depth_gen):
    if os.path.exists(meta_output):
        print(f"file already done: {img_output}")
        return #has already been converted
        
    
    if from_zip:
        # Read raw bytes from the zip file
        data = zip_ref.read(img_path)
        img_array = np.frombuffer(data, np.uint8)
        img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    else:
        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    
    
    

    if img_bgr is None:
        print(f"[WARN] Could not load image: {img_path}")
        return None
    os.makedirs(os.path.dirname(img_output), exist_ok=True)
    
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    #Dont need to see what the depth model is doing so pipe it to null
    with open(os.devnull, "w") as f, redirect_stdout(f):
        depth_out = da3model.inference(
            [img_rgb],
            process_res = depth_resolution
        )
        
    cam_matrix = depth_out.intrinsics[0]
    depth = depth_out.depth[0]
    
    #100 is the max depth
    depth = depth.clip(0, 100)
    
    H, W = depth.shape
    down_scaled_org_image = cv2.resize(img_rgb, (W, H)).astype(np.uint8)
    
    #compare with unik3d
    rgb_torch = torch.from_numpy(down_scaled_org_image).permute(2, 0, 1)
    unik_ored = unk3dmodel.infer(rgb_torch)
    unik_depth = unik_ored["depth"].squeeze().cpu().numpy()
    
    unik_depth = unik_depth.clip(0, 100)
    
    fx, fy = unik3d_video.estimate_focal_lengths(unik_ored['points'], W, H)

    #Get generic matrix that we can fill in
    unik3d_cam = depth_map_tools.compute_camera_matrix(90, None, W, H)
    
    #Fill in real focal lengths
    unik3d_cam[0][0] = fx
    unik3d_cam[1][1] = fy
    
    
    
    norm_d = depth_frames_helper.normalize_depth(depth)
    unik_norm_d = depth_frames_helper.normalize_depth(unik_depth)
    
    ssim_val = ssim(norm_d, unik_norm_d, data_range=1.0)
    
    
    #Calculate final output size
    scale = save_resolution / min(H, W)
    output_W = int(W * scale)
    output_H = int(H * scale)
    
    
    
    
    #For debuging some general info about the data
    fovx, _ = depth_map_tools.fov_from_camera_matrix(cam_matrix)
    unik_fovx, _ = depth_map_tools.fov_from_camera_matrix(unik3d_cam)
    data = [img_output, "ssim_val:", ssim_val, "xfov:", fovx, "unik_fovx:", unik_fovx, "min_depth:", depth.min(), "mean_depth:", depth.mean(), "max_depth:", depth.max(), "min_unik_depth:" , unik_depth.min(), "mean_unik_depth:", unik_depth.mean(), "max_unik_depth:", unik_depth.max()]
    def to_str(val):
        if isinstance(val, (int, float, np.floating)):  # round numbers
            return f"{val:.2f}"
        return str(val)

    str_desc = " ".join(to_str(v) for v in data)
    print(str_desc)
    
    
    #tested and 0.72 semas like a decent cutoff, there is still some jank but gets rid of the worst cases
    if ssim_val < 0.72:
        print("ignoring image: depth models dont agree enogh about depth")
        with open(meta_output, "w") as file:
            file.write("ignored: "+str_desc)
        return
    
    # Randomly select left or right conversion
    do_right = random.random() < 0.5
    ipd_baseline = 0.063  # 63 mm = standard human IPD
    
    
    near = 0.1
    far = 110.0
    
    
    
    # Create a mesh from the data
    mesh, _ = depth_map_tools.mesh_from_depth_and_rgb(depth, down_scaled_org_image, cam_matrix, mask_for_right = do_right, stero_paralax_shift = ipd_baseline, set_paralaxed_to_color = None)
    
    
    # Camera setup (origin, looking forward)
    if do_right:
        eye    = np.array([ipd_baseline, 0, 0], dtype=np.float32)
    else:
        eye    = np.array([-ipd_baseline, 0, 0], dtype=np.float32)
    target = eye + np.array([0, 0, -1], dtype=np.float32)
    up     = np.array([0, 1, 0], dtype=np.float32)

    view = depth_map_tools.gl_look_at(eye, target, up) 
    proj = depth_map_tools.open_gl_projection_from_camera_matrix(cam_matrix, near, far)
    model = np.eye(4, dtype=np.float32)
    mvp = proj @ view @ model
    
    
    # Render to a bigger surface to make sure that all pixels from the original image get renderd (or the visible ones) if we only render at orgiginal resolution then some pixels gets overwriten by their neighbors do to how the GPU does stuff
    generated_right_image, generated_right_depth, normals, first_morph_ids = depth_map_tools.gl_render(mesh, mvp, W*2, H*2, near, far, bg_color = [1.0, 1.0, 0.0])
    
    
    background1 = generated_right_depth > 101
    background1 = downsample_mask_maxpool(background1, factor=2)
    #Make sure that background pixels are not cliped away
    generated_right_depth = generated_right_depth.clip(0, 105)
    
    # Downscale after render
    generated_right_depth = cv2.resize(generated_right_depth, (W, H), interpolation=cv2.INTER_NEAREST)
    generated_right_image = cv2.resize(generated_right_image, (W, H)) 
    
    # Create a new mesh from the output of the first render a morphed mesh morped to the direction of choise the
    mesh_morpehed, paralax_mask2 = depth_map_tools.mesh_from_depth_and_rgb(generated_right_depth, generated_right_image, cam_matrix, mask_for_right = not do_right, stero_paralax_shift = ipd_baseline, set_paralaxed_to_color = None, parallax_ignore_mask = background1)
    
    #The ids of the pixels that have been calculated to need infill after paralax are marked
    id_paralax_map = (np.arange(H * W, dtype=np.uint32) + 1).reshape(H, W)
    
    
    
    ids_with_paralax = id_paralax_map[paralax_mask2]
    #hole_mask1 = np.isin(first_morph_ids, ids_with_paralax)
    
        
    
    # Generate a new projection matric to render the morphed mesh from the original camera position where the pixels that we callculated to probably have paralaxy will be more visible
    if not do_right:
        eye    = np.array([ipd_baseline, 0, 0], dtype=np.float32)
    else:
        eye    = np.array([-ipd_baseline, 0, 0], dtype=np.float32)
    target = eye + np.array([0, 0, -1], dtype=np.float32)
    view = depth_map_tools.gl_look_at(eye, target, up)
    mvp_left = proj @ view @ model
    
    #render
    image_generated_left, depth_generated_left, normals_generated_left, ids_duble_morp = depth_map_tools.gl_render(mesh_morpehed, mvp_left, output_W, output_H, near, far, bg_color = [0.0, 1.0, 0.0])
    
    
    #resacle to specified save res
    out_org_rgb = cv2.resize(down_scaled_org_image, (output_W, output_H))
    save_depth = cv2.resize(depth, (output_W, output_H), interpolation=cv2.INTER_NEAREST)
    
    #Any ids that was marked as having a need for paralax infill are used to make a mask 
    hole_mask = np.isin(ids_duble_morp, ids_with_paralax)
    
    
    
    #If a pixel has no id it means it is a background pixel
    #background  = ids_duble_morp == 0
    
    background = depth_generated_left > 109
    
    #Find the pixels that was lost in the first morph
    background_from_first_morph = (depth_generated_left > 103) & ~background # we cliped it to 105
    
    
    
    # Create the generated out image
    out_generated = image_generated_left.copy()
    #Set any background pixels to be black
    out_generated[background] = 0
    #set any pixels that was lost in the first morpg wo their orgiginal value
    out_generated[background_from_first_morph] = out_org_rgb[background_from_first_morph]
    
    #Create a special normal image that will be used to fill in the normals of pixels that was not renderd
    xnormal = depth_map_tools.generate_normal_bg_image(output_W, output_H)
    
    #Fix the normal map so that it can be exported
    normals_left_masked = normals_generated_left.copy()
    normals_left_masked[~hole_mask] = 0
    normals_left_masked[background] = xnormal[background]
    normals_left_masked = (normals_left_masked*255).astype(np.uint8)
    
    
    depth_left_masked = depth_generated_left.copy()
    #Set any background pixels to 0 depth
    depth_left_masked[background] = 0
    #set any pixels that was lost in the first morpg wo their orgiginal value
    depth_left_masked[background_from_first_morph] = save_depth[background_from_first_morph]
    
    
    
    # Encode depth for saving to file
    encoded_depth = depth_frames_helper.encode_depth_as_uint32(save_depth, MODEL_maxOUTPUT_depth)
    org_bgr24bit = depth_frames_helper.encode_data_as_BGR(encoded_depth, save_depth.shape[1], save_depth.shape[0], bit16 = True)
    
    encoded_depth = depth_frames_helper.encode_depth_as_uint32(depth_left_masked, MODEL_maxOUTPUT_depth)
    bgr24bit = depth_frames_helper.encode_data_as_BGR(encoded_depth, depth_left_masked.shape[1], depth_left_masked.shape[0], bit16 = True)
    

    print("save:", img_output)
    
    #First we write to tmp files
    cv2.imwrite(img_output+tmp_file_str, cv2.cvtColor(out_org_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(img_output_depth+tmp_file_str, org_bgr24bit)
    cv2.imwrite(img_output_depth_gen+tmp_file_str, bgr24bit)
    cv2.imwrite(img_output_normal_gen+tmp_file_str, cv2.cvtColor(normals_left_masked, cv2.COLOR_RGB2BGR))
    cv2.imwrite(img_output_gen+tmp_file_str, cv2.cvtColor(out_generated, cv2.COLOR_RGB2BGR))
    
    
    
    #rename the temp files when they are done
    rename(img_output+tmp_file_str, img_output)
    rename(img_output_depth+tmp_file_str, img_output_depth)
    rename(img_output_depth_gen+tmp_file_str, img_output_depth_gen)
    rename(img_output_normal_gen+tmp_file_str, img_output_normal_gen)
    rename(img_output_gen+tmp_file_str, img_output_gen)
    
    with open(meta_output, "w") as file:
        file.write("OK: "+str_desc)


    return img_rgb


# ---------------------------------------------------------
# process_dataset_folder
# ---------------------------------------------------------
def process_dataset_folder(dataset_path):
    """
    Walks through the dataset folder and processes all images
    by calling convert_to_training_data(img_path).
    """
    # Allowed image extensions
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")

    if dataset_path.lower().endswith(".zip"):
        print(f"Opening ZIP file: {dataset_path}")

        with zipfile.ZipFile(dataset_path, 'r') as z:
            # Filter: only files under unlabeled2017/
            image_names = [
                n for n in z.namelist()
                if n.startswith("unlabeled2017/") and n.lower().endswith(exts)
            ]

            print(f"Found {len(image_names)} images inside zip under unlabeled2017/")

            for idx, name in enumerate(image_names):
                print(f"[{idx+1}/{len(image_names)}] Processing: {name}")
                img = convert_to_training_data(name, from_zip=True, zip_ref=z)

        return

    # Case B: Folder
    if os.path.isdir(dataset_path):
        print(f"Walking folder: {dataset_path}")

        image_paths = []
        for root, _, files in os.walk(dataset_path):
            for fname in files:
                if fname.lower().endswith(exts):
                    image_paths.append(os.path.join(root, fname))

        print(f"Found {len(image_paths)} images in folder.")

        for idx, img_path in enumerate(image_paths):
            print(f"[{idx+1}/{len(image_paths)}] Processing: {img_path}")
            img = convert_to_training_data(img_path, from_zip=False)

        return

    print("ERROR: Dataset path is neither folder nor .zip file.")


# ---------------------------------------------------------
# RUN
# ---------------------------------------------------------
if __name__ == "__main__":
    os.makedirs(training_output_folder, exist_ok=True)
    print("loading depth model")
    da3model = video_da3.load_model()
    unk3dmodel = unik3d_video.load_model()
    process_dataset_folder(dataset_folder)
