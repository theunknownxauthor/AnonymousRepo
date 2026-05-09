import os
import numpy as np
import tensorflow as tf
import rasterio
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from TriFusion_Gate_Atrous_Gate import create_TriFusion
import TriFusion_Gate_Atrous_Gate as TriF
import pandas as pd
from tensorflow.keras import backend as K

import argparse
import random


# ========== HELPER FUNCTIONS ==========
# Save indices to CSV files
def save_indices_to_csv(indices, filename):
    df = pd.DataFrame(indices, columns=["file_idx", "i", "j"])
    df.to_csv(filename, index=False)
    
def get_sorted_tif_paths(input_folder, label_folder):
    input_paths = sorted([
        os.path.join(input_folder, fname)
        for fname in os.listdir(input_folder)
        if fname.endswith(".tif")
    ])
    label_paths = sorted([
        os.path.join(label_folder, fname)
        for fname in os.listdir(label_folder)
        if fname.endswith(".tif")
    ])
    assert len(input_paths) == len(label_paths), f"Mismatch in {input_folder} and {label_folder}"
    return input_paths, label_paths

def generate_patch_indices(input_data_paths, patch_size_global,index_from, index_to):
    building_indices = []
    nonbuilding_indices = []
    half_patch = patch_size_global // 2
    limit=178500
    for file_idx, input_path in enumerate(input_data_paths):
        if not((index_from<=file_idx) and (file_idx<=index_to)):
            continue
        with rasterio.open(input_path) as src:
            H, W = src.height, src.width

        last_label = None
        
        for i in range(half_patch, H - half_patch):
            for j in range(half_patch, W - half_patch):
                with rasterio.open(input_path) as src_tgt:
                    label = src_tgt.read(1, window=rasterio.windows.Window(j, i, 1, 1))[0, 0]

                if label == 255.0:
                    building_indices.append((file_idx, i, j))
                elif label == 0.0:
                    nonbuilding_indices.append((file_idx, i, j))

                print(f"\rIndices: ({file_idx},{i},{j}), Collected: {len(building_indices)} building, {len(nonbuilding_indices)} non-building", end='')
                
    # Calculate balanced length
    min_class_len = min(len(building_indices), len(nonbuilding_indices))

    # Randomly sample from both
    building_sample = random.sample(building_indices, min_class_len)
    nonbuilding_sample = random.sample(nonbuilding_indices, min_class_len)

    # print(f"Final balanced set: {len(balanced_indices)} samples (each class: {min_class_len})")
    return building_sample, nonbuilding_sample

# ========== ARGPARSE ==========
parser = argparse.ArgumentParser(description="Train TriFusion model with configurable parameters.")
parser.add_argument("--p1", type=int, default=0, help="Batch size")
parser.add_argument("--p2", type=int, default=20, help="Training")
parser.add_argument("--first", type=int, default=0, help="Training")
args = parser.parse_args()

# ========== CONFIGURATION ==========
index_from = args.p1
index_to = args.p2
first=args.first

# ========== FILES ==========
config=f"Option_{index_from}_{index_to}_first:{first}"


print(f"config:{config}")

# ========== FOLDER PATHS ==========
DATA_DIR = "."  # base folder, or replace with absolute path

train_dem_dir = os.path.join(DATA_DIR, "train_dem")
train_label_dir = os.path.join(DATA_DIR, "train_labels")

val_dem_dir = os.path.join(DATA_DIR, "val_dem")
val_label_dir = os.path.join(DATA_DIR, "val_labels")

test_dem_dir = os.path.join(DATA_DIR, "test_dem")
test_label_dir = os.path.join(DATA_DIR, "test_labels")


# ========== LOAD PATHS ==========
print ("Load Paths...")
print ("Load train_dem_dir...")
train_input_paths, train_target_paths = get_sorted_tif_paths(train_dem_dir, train_label_dir)
print ("Load val_dem_dir...")
val_input_paths, val_target_paths = get_sorted_tif_paths(val_dem_dir, val_label_dir)
print ("Load test_dem_dir...")
test_input_paths, test_target_paths = get_sorted_tif_paths(test_dem_dir, test_label_dir)

print ("Generate Indices...")
if(first==0):
    print ("Generate Train Indices")
    train_1_indices, train_0_indices = generate_patch_indices(train_target_paths, 35,index_from, index_to)

    print("Saving index files...")
    save_indices_to_csv(train_1_indices, f"train_indices_1_{index_from}_{index_to}.csv")
    save_indices_to_csv(train_0_indices, f"train_indices_0_{index_from}_{index_to}.csv")

if(first==1):

    print ("Generate Val Indices")
    val_1_indices, val_0_indices = generate_patch_indices(val_target_paths, 35,index_from, index_to)
    print ("Generate Test Indices")
    test_1_indices, test_0_indices = generate_patch_indices(test_target_paths, 35,index_from, index_to)

    save_indices_to_csv(val_1_indices, f"val_indices_1_{index_from}_{index_to}.csv")
    save_indices_to_csv(val_0_indices, f"val_indices_0_{index_from}_{index_to}.csv")
    save_indices_to_csv(test_1_indices, f"test_indices_1_{index_from}_{index_to}.csv")
    save_indices_to_csv(test_0_indices, f"test_indices_0_{index_from}_{index_to}.csv")

   