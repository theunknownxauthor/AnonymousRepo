import os
import numpy as np
import rasterio
import argparse


    
def preprocess_raster(input_data):
    # Reload processed data for model input
    with rasterio.open(input_data) as src:
        input_data = src.read()
        profile = src.profile
        input_data = np.moveaxis(input_data, 0, -1)
   
    return input_data, profile    

def get_sorted_tif_paths(input_folder):
    input_paths = sorted([
        os.path.join(input_folder, fname)
        for fname in os.listdir(input_folder)
        if fname.endswith(".tif")
    ])

    return input_paths

def threshold_image(input_data, THRESHOLD, output_raster):
    input_data, profile = preprocess_raster(input_data)

    # Apply thresholding: pixels >= THRESHOLD become 1, others become 0
    binary_mask = (input_data >= THRESHOLD).astype(np.float32)

    profile.update(count=1, dtype=rasterio.float32)

    with rasterio.open(output_raster, 'w', **profile) as dst:
        # Write the thresholded mask as a single band
        dst.write(binary_mask[:, :, 0], 1)  # Assuming you're thresholding the first band

    print(f"THRESHOLD raster saved to {output_raster}.")

def threshold_images(input_data_paths, THRESHOLD):
    output_dir = f"output_threshold_{THRESHOLD}"
    os.makedirs(output_dir, exist_ok=True)




    for file_idx, input_path in enumerate(input_data_paths):
        print(f"Building detection THRESHOLD in the file: {input_path}")

        # Build output path and checkpoint file name
        file_name = os.path.basename(input_path)
        output_path = os.path.join(output_dir, file_name)
        base, _ = os.path.splitext(output_path)

        # Run THRESHOLD
        threshold_image(input_path, THRESHOLD,output_path)

# ========== ARGPARSE ==========
parser = argparse.ArgumentParser(description="Train TriFusion model with configurable parameters.")
parser.add_argument("--path", type=str, default="output", help="path")
parser.add_argument("--threshold", type=float, default=0.9, help="threshold")

args = parser.parse_args()



# ========== CONFIGURATION ==========
PATH = args.path
THRESHOLD = args.threshold


# ========== FOLDER PATHS ==========
DATA_DIR = "."  # base folder, or replace with absolute path

path_label_dir = os.path.join(DATA_DIR, PATH)


# ========== LOAD PATHS ==========
print ("Load Paths...")
input_paths = get_sorted_tif_paths(path_label_dir)

threshold_images(input_paths, THRESHOLD )

