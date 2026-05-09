import os
import numpy as np
import tensorflow as tf
import rasterio
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from TriFusion_Gate_Atrous_Gate import create_TriFusion
import TriFusion_Gate_Atrous_Gate as TriF
from tensorflow.keras import backend as K

import argparse

    
def preprocess_raster(input_data, output_raster, bands):

    # Reload processed data for model input
    with rasterio.open(input_data) as src:
        input_data = src.read()
        profile = src.profile
        input_data = np.moveaxis(input_data, 0, -1)

    print(f"Sanity check: min={input_data.min()}, max={input_data.max()}, has_nan={np.isnan(input_data).any()}")
   
    
    return input_data, profile    

def get_sorted_tif_paths(input_folder):
    input_paths = sorted([
        os.path.join(input_folder, fname)
        for fname in os.listdir(input_folder)
        if fname.endswith(".tif")
    ])

    return input_paths
 
def predict_image(model, input_data, patch_size_global, patch_size, bands, batch_size_p, output_raster, checkpoint_file):

    input_data, profile = preprocess_raster(input_data, "nothing.tif", bands)
    half_patch_size_global = patch_size_global // 2
    half_patch_size = patch_size // 2
    _, height, width = input_data.shape[2], input_data.shape[0], input_data.shape[1]
    input_data_reshaped = input_data.reshape((height, width, bands))

    # Check if the output raster file exists
    if os.path.exists(output_raster):
        print("Load the existing predicted building map")
        with rasterio.open(output_raster) as src:
            predicted_building = src.read(1)
    else:
        print("Initialize a zero array for the predicted building map")
        predicted_building = np.zeros((height, width))

    # Check if a checkpoint file exists to resume from the last position
    if os.path.exists(checkpoint_file):
        print("Checkpoint file exists to resume from the last position")
        with open(checkpoint_file, 'r+') as f:
            last_i, last_j = map(int, f.read().strip().split(','))
    else:
        print("Checkpoint file not existing to resume from the last position")
        last_i, last_j = half_patch_size_global, half_patch_size_global  # Start from the first valid position

    # Initialize lists to hold the batch data
    batch_X_g = []
    batch_X_c = []
    batch_X_p = []
    batch_positions = []
    max_pred = 0

    # Loop over the entire input image, resuming from the last position    
    collected="No batch collected"
    nb_collected=0
    profile.update(
    dtype=rasterio.float32,
    count=1,
    nodata=None,  # or 0.0 if you want to specify nodata
    compress='lzw'  # optional: helps compress output
    )
    for i in range(last_i, height - half_patch_size_global):

        for j in range(last_j if i == last_i else half_patch_size_global, width - half_patch_size_global):
            
            # Extract the patch centered at (i, j)
            patch = input_data_reshaped[i - half_patch_size:i + half_patch_size + 1, j - half_patch_size:j + half_patch_size + 1, :]
            patch_global = input_data_reshaped[i - half_patch_size_global:i + half_patch_size_global + 1, j - half_patch_size_global:j + half_patch_size_global + 1, :]
            
            X_c = patch[:, :, -bands:]
            X_g = patch_global[:, :, -bands:]
            X_p = patch[half_patch_size, half_patch_size, :]

            # Append the data to the batch lists

            batch_positions.append((i, j))
            min_values = np.array([0.0, 0.0, 0.0, -20.0, 0.0])
            max_values = np.array([255.0, 255.0, 255.0, 194.0, 53.0])
            # Normalize local context
            X_c = (X_c - min_values) / (max_values - min_values + 1e-8)  # Add small epsilon for stability

            # Normalize global context
            X_g = (X_g - min_values) / (max_values - min_values + 1e-8)
            
            # Normalize pixel context
            X_p = (X_p - min_values) / (max_values - min_values + 1e-8)
            batch_X_c.append(X_c)
            batch_X_g.append(X_g)
            batch_X_p.append(X_p)

            
            length=len(batch_X_g)
            if (len(batch_X_g) == batch_size_p) or ((j == (width - half_patch_size_global - 1)) and (i == (height - half_patch_size_global - 1))):
                samples = batch_size_p
                if len(batch_X_g) != batch_size_p:
                    samples = len(batch_X_g)
                nb_collected=nb_collected+1    
                ok= f"Yes {i},{j}"
                collected=f"last batch collected at {i},{j}"
                
                with open('test.txt', 'a') as f: 
                    f.write(f"Yes {i},{j}\n")
                batch_X_c_np = np.array(batch_X_c).reshape(samples, patch_size       , patch_size,        bands)
                batch_X_g_np = np.array(batch_X_g).reshape(samples, patch_size_global, patch_size_global, bands)
                batch_X_p_np = np.array(batch_X_p).reshape(samples, 1                , 1,                 bands)
                # Predict the population density for the batch


                dummy_y_true = np.zeros((samples, 1), dtype=np.float32)
                _, batch_predicted_building = model.predict([batch_X_c_np, batch_X_g_np, batch_X_p_np, dummy_y_true ])
                
                # Assign predictions to the correct positions in the output array
                for (i_pos, j_pos), pred in zip(batch_positions, batch_predicted_building):
                    predicted_building[i_pos, j_pos] = pred
                    print(f"pred at ({i_pos},{j_pos}):{pred}")
                    if (pred > max_pred):
                        max_pred = pred

        
                # Clear the batch lists
                batch_X_c.clear()
                batch_X_p.clear()
                batch_X_g.clear()
                batch_positions.clear()

                # Save the predicted population density to the output raster file
                profile.update(count=1)  # Update profile to single band
                with rasterio.open(output_raster, 'w', **profile) as dst:
                    dst.write(predicted_building.astype(rasterio.float32), 1)
            else:
                ok =f"No {i},{j}"
                # Create the line to print and write
            line = (
                f"Batch collected: {ok} | nb_collected: {nb_collected} | {collected} | length: {length} ")
            print("\r" + line, end='')  # still print it live
            # Save the current position to the checkpoint file
            with open(checkpoint_file, 'w') as f:
                f.write(f"{i},{j}")
            # tf.keras.backend.clear_session()
            

    # Save the predicted image
    profile.update(count=1)  # Update profile to single band
    with rasterio.open(output_raster, 'w', **profile) as dst:
        dst.write(predicted_building.astype(rasterio.float32), 1)

    print(f"Predicted raster saved to {output_raster}.")

def predict_test_images(model, input_data_paths, patch_size_global, patch_size, bands):
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    processed_file = "file_processed.txt"
    processed_paths = set()

    # Load previously processed paths if the file exists
    if os.path.exists(processed_file):
        with open(processed_file, "r") as f:
            processed_paths = set(line.strip() for line in f)


    for file_idx, input_path in enumerate(input_data_paths):
        if input_path in processed_paths:
            print(f"Skipping already processed file: {input_path}")
            continue

        print(f"Building detection in the file: {input_path}")

        # Build output path and checkpoint file name
        file_name = os.path.basename(input_path)
        output_path = os.path.join(output_dir, file_name)
        base, _ = os.path.splitext(output_path)
        checkpoint_file = base + ".txt"

        # Run prediction
        predict_image(model, input_path, patch_size_global, patch_size, bands, 5000, output_path, checkpoint_file)

        # Append processed file path
        with open(processed_file, "a") as f:
            f.write(input_path + "\n")
# ========== ARGPARSE ==========
parser = argparse.ArgumentParser(description="Train TriFusion model with configurable parameters.")
parser.add_argument("--model_option", type=str, default="GAG", help="Model Option")
parser.add_argument("--patch_size_global", type=int, default=35, help="Global patch size")
parser.add_argument("--latent_dim", type=int, default=50, help="Latent dimension")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
parser.add_argument("--training", type=int, default=1, help="Training")
parser.add_argument("--dataset", type=int, default=0, help="Dataset")
args = parser.parse_args()



# ========== CONFIGURATION ==========
PATCH_SIZE = 11
PATCH_SIZE_GLOBAL = args.patch_size_global
LATENT_DIM = args.latent_dim
BANDS = 5
BANDS_CONTEXT = 5
BATCH_SIZE = args.batch_size
MODEL_OPTION = args.model_option
EPOCHS = 100
TRAINING = args.training
DATASET = args.dataset


if(MODEL_OPTION=="GAG"):
    atrous=1
else:
    atrous=0
# ========== FILES ==========
weights=f"weight.h5"

# ========== FOLDER PATHS ==========
DATA_DIR = "."  # base folder, or replace with absolute path

test_dem_dir = os.path.join(DATA_DIR, "test_updated")


# ========== LOAD PATHS ==========

print ("Load test_updated...")
test_input_paths = get_sorted_tif_paths(test_dem_dir)

# ========== MODEL ==========
model = create_TriFusion(
    patch_size=PATCH_SIZE,
    patch_size_global=PATCH_SIZE_GLOBAL,
    latent_dim=LATENT_DIM,
    bands=BANDS_CONTEXT,
    atrous=atrous
)

# Load the best weights if they exist
noweights=0
try:
	model.load_weights(weights) 
    
except:
	print("No weights found.")
	noweights=1


if(noweights==0):
	predict_test_images(model, test_input_paths, PATCH_SIZE_GLOBAL, PATCH_SIZE, BANDS)

