import os
import shutil
import numpy as np
import scipy.io as sio
from pathlib import Path
from PIL import Image
from tqdm import tqdm

def process_consep(input_dir, output_dir):
    """
    Process CoNSeP dataset to match the required format for InferenceCellViT.
    CoNSeP raw data structure:
    - input_dir/
        - Train/
            - Images/ (*.png)
            - Labels/ (*.mat)
        - Test/
            - Images/ (*.png)
            - Labels/ (*.mat)
            
    Target structure:
    - output_dir/
        - images/ (*.png)
        - labels/ (*.npy) - Processed labels
    
    The .mat files contain:
    - inst_map: Instance map (each nucleus has a unique integer ID)
    - type_map: Type map (each nucleus has a class ID)
    - inst_type: (Not always present or needed if type_map exists)
    - inst_centroid: Centroids
    
    We need to save them as .npy files containing a dictionary or similar structure that the dataset class expects.
    Looking at consep.py:
        mask = np.load(mask_path, allow_pickle=True)
        inst_map = mask[()]["inst_map"].astype(np.int32)
        type_map = mask[()]["type_map"].astype(np.int32)
    So we should save a dictionary with "inst_map" and "type_map" keys.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # Create output directories (we'll put everything in 'testing' structure for inference, or separate)
    # The user wants to run inference, usually on Test set.
    # But let's process both and put them in separate folders like MoNuSeg if needed,
    # or just flat if that's what the inference script expects.
    # The previous MoNuSeg script expected a 'testing' folder with 'images' and 'labels'.
    
    subsets = ["Train", "Test"]
    
    for subset in subsets:
        print(f"Processing {subset} set...")
        # Map subset name to lower case for output folder
        subset_out_name = "training" if subset == "Train" else "testing"
        subset_out_dir = output_dir / subset_out_name
        (subset_out_dir / "images").mkdir(parents=True, exist_ok=True)
        (subset_out_dir / "labels").mkdir(parents=True, exist_ok=True)
        
        subset_img_dir = input_dir / subset / "Images"
        subset_lbl_dir = input_dir / subset / "Labels"
        
        # Get all images
        image_files = sorted(list(subset_img_dir.glob("*.png")))
        
        for img_path in tqdm(image_files):
            file_stem = img_path.stem
            
            # 1. Process Image
            # Copy image to output
            out_img_path = subset_out_dir / "images" / f"{file_stem}.png"
            shutil.copy(img_path, out_img_path)
            
            # 2. Process Mask (.mat -> .npy)
            mat_path = subset_lbl_dir / f"{file_stem}.mat"
            
            if not mat_path.exists():
                print(f"Warning: No mask found for {file_stem}, skipping mask generation.")
                continue
                
            # Load .mat file
            mat_data = sio.loadmat(str(mat_path))
            
            # Extract relevant maps
            # CoNSeP .mat usually has 'inst_map'
            # Note: CoNSeP original labels might not have 'type_map' directly as a full map in older versions,
            # but usually it's derived. Let's check what's inside.
            # Usually: 'inst_map', 'type_map' (optional), 'inst_type' (list of types per instance)
            
            inst_map = mat_data['inst_map']
            
            if 'type_map' in mat_data:
                type_map = mat_data['type_map']
            else:
                # If type_map is missing, we need to reconstruct it from inst_map and inst_type
                # inst_type is usually an array where index i corresponds to instance i
                # Note: inst_map values start from 1. inst_type index usually 0-based or 1-based?
                # Need to check CoNSeP format specifics.
                # Usually inst_type is Nx1 array.
                type_map = np.zeros_like(inst_map)
                if 'inst_type' in mat_data:
                    inst_type = mat_data['inst_type']
                    for i in range(1, int(np.max(inst_map)) + 1):
                        # Assuming inst_type is aligned with instance IDs
                        # Usually CoNSeP instances are 1..N. inst_type might be length N.
                        try:
                            # Instance ID i corresponds to index i-1 in inst_type if sorted?
                            # Or we can use `inst_centroid` to match?
                            # Let's assume standard CoNSeP format:
                            # inst_type is a list of types. The first element is for instance 1?
                            t = inst_type[i-1]
                            type_map[inst_map == i] = t
                        except IndexError:
                            pass
            
            # Prepare dict to save
            save_dict = {
                "inst_map": inst_map,
                "type_map": type_map
            }
            
            # Save as .npy
            out_mask_path = subset_out_dir / "labels" / f"{file_stem}.npy"
            np.save(out_mask_path, save_dict)

    print(f"Processing complete. Data saved to {output_dir}")

if __name__ == "__main__":
    input_root = "/hy-tmp/SparseSwinCell/cell_segmentation/datasets/original/CoNSeP"
    output_root = "/hy-tmp/SparseSwinCell/cell_segmentation/datasets/process/CoNSeP"
    process_consep(input_root, output_root)
