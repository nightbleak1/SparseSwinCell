import os
import numpy as np
from PIL import Image
import argparse
import shutil
from pathlib import Path

def convert_processed_data(input_dir, output_dir):
    """
    Convert processed PanNuke data to the format expected by the training pipeline.
    
    Args:
        input_dir (str): Path to the processed PanNuke data directory
        output_dir (str): Path to the output directory
    """
    # Create output directory structure
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process each fold
    for fold in ['fold0', 'fold1', 'fold2']:
        fold_path = Path(input_dir) / fold
        if not fold_path.exists():
            print(f"Warning: {fold_path} does not exist, skipping...")
            continue
            
        print(f"Processing {fold}...")
        
        # Create fold directory in output
        fold_output = output_path / fold
        fold_output.mkdir(exist_ok=True)
        
        # Create Images and Masks directories
        images_dir = fold_output / "Images"
        masks_dir = fold_output / "Masks"
        images_dir.mkdir(exist_ok=True)
        masks_dir.mkdir(exist_ok=True)
        
        # Process images
        src_images_dir = fold_path / "images"
        src_labels_dir = fold_path / "labels"
        
        if not src_images_dir.exists() or not src_labels_dir.exists():
            print(f"Warning: Images or labels directory missing in {fold}, skipping...")
            continue
            
        # Get all image files
        image_files = list(src_images_dir.glob("*.png"))
        print(f"Found {len(image_files)} images in {fold}")
        
        # Copy images and convert labels
        for img_file in image_files:
            # Copy image
            dst_img_path = images_dir / img_file.name
            shutil.copy2(img_file, dst_img_path)
            
            # Convert label from .npy to .npy (ensure it's in the right format)
            label_name = img_file.stem + ".npy"
            src_label_path = src_labels_dir / label_name
            dst_label_path = masks_dir / label_name
            
            if src_label_path.exists():
                # Load and save label to ensure format consistency
                label_data = np.load(src_label_path)
                # Ensure the label is in the correct format (H, W, 6)
                if len(label_data.shape) == 3 and label_data.shape[2] != 6:
                    # If it's not in the expected format, we might need to process it
                    print(f"Warning: Label {label_name} has shape {label_data.shape}, expected (H, W, 6)")
                
                np.save(dst_label_path, label_data)
            else:
                print(f"Warning: Label file {src_label_path} not found")
        
        print(f"Completed processing {fold}")
    
    print("Data conversion completed!")

def create_dataset_splits(output_dir):
    """
    Create the dataset splits file as expected by the training pipeline.
    """
    output_path = Path(output_dir)
    splits_file = output_path / "pannuke_splits.npy"
    
    # For simplicity, we'll create a basic split
    # In a real scenario, you might want to load the actual splits
    splits = {
        'train': [0],  # Using fold0 for training
        'val': [1],    # Using fold1 for validation
        'test': [2]    # Using fold2 for testing
    }
    
    np.save(splits_file, splits)
    print(f"Created dataset splits file at {splits_file}")

def main():
    parser = argparse.ArgumentParser(description="Convert processed PanNuke data to training format")
    parser.add_argument("--input_dir", type=str, required=True, 
                        help="Path to the processed PanNuke data directory")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Path to the output directory")
    
    args = parser.parse_args()
    
    # Convert the data
    convert_processed_data(args.input_dir, args.output_dir)
    
    # Create dataset splits
    create_dataset_splits(args.output_dir)
    
    print(f"Data conversion completed successfully!")
    print(f"Output directory: {args.output_dir}")

if __name__ == "__main__":
    main()