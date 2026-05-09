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
import gc



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
    # building_sample = random.sample(building_indices, min_class_len)
    # nonbuilding_sample = random.sample(nonbuilding_indices, min_class_len)

    # Combine and shuffle
    balanced_indices = building_indices + nonbuilding_indices
    # random.shuffle(balanced_indices)

    # print(f"Final balanced set: {len(balanced_indices)} samples (each class: {min_class_len})")
    return balanced_indices
    
def preprocess_raster(input_data, output_raster, bands):

    # Reload processed data for model input
    with rasterio.open(input_data) as src:
        input_data = src.read()
        profile = src.profile
        input_data = np.moveaxis(input_data, 0, -1)

    print(f"Sanity check: min={input_data.min()}, max={input_data.max()}, has_nan={np.isnan(input_data).any()}")
    
    if(bands==3):
        input_data = np.delete(input_data, [3, 4], axis=0)
        profile.update(count=bands, nodata=0.0)  
    
    if(bands==4):
        input_data = np.delete(input_data, [4], axis=0)
        profile.update(count=bands, nodata=0.0)

    return input_data, profile    
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
            batch_X_c.append(X_c)
            batch_X_g.append(X_g)
            batch_X_p.append(X_p)
            batch_positions.append((i, j))

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
                _, batch_predicted_pop = model.predict([batch_X_c_np, batch_X_g_np, batch_X_p_np, dummy_y_true ])
                
                # Assign predictions to the correct positions in the output array
                for (i_pos, j_pos), pred in zip(batch_positions, batch_predicted_pop):
                    predicted_building[i_pos, j_pos] = pred
                    print("Sample pred:", pred)
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
def load_indices_from_csv(filename):
    df = pd.read_csv(filename)
    return [tuple(row) for row in df.to_numpy()]
def get_trainable_params(model):
    return int(np.sum([K.count_params(w) for w in model.trainable_weights]))

# ========== ARGPARSE ==========
parser = argparse.ArgumentParser(description="Train TriFusion model with configurable parameters.")
parser.add_argument("--model_option", type=str, default="GAG", help="Model Option")
parser.add_argument("--patch_size_global", type=int, default=35, help="Global patch size")
parser.add_argument("--latent_dim", type=int, default=50, help="Latent dimension")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
parser.add_argument("--training", type=int, default=1, help="Training")
parser.add_argument("--dataset", type=int, default=0, help="Dataset")
parser.add_argument("--bands", type=int, default=5, help="Dataset")
args = parser.parse_args()



# ========== CONFIGURATION ==========
PATCH_SIZE = 11
PATCH_SIZE_GLOBAL = args.patch_size_global
LATENT_DIM = args.latent_dim
BANDS = args.bands
BATCH_SIZE = args.batch_size
MODEL_OPTION = args.model_option
EPOCHS = 100
TRAINING = args.training
DATASET = args.dataset

# ========== FILES ==========
weights=f"TriFusion_Option_{MODEL_OPTION}_Bands_{BANDS}_Global_{PATCH_SIZE_GLOBAL}_Laten_{LATENT_DIM}_Laten_{BATCH_SIZE}.h5"


print(f"Configuration:{weights}")

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

# ========== GENERATE INDICES ==========

print("Loading index files from CSV...")
train_indices_0 = load_indices_from_csv("train_indices_0.csv")
train_indices_1 = load_indices_from_csv("train_indices_1.csv")
val_indices_0 = load_indices_from_csv("val_indices_0.csv")
val_indices_1 = load_indices_from_csv("val_indices_1.csv")
test_indices_0 = load_indices_from_csv("test_indices_0.csv")
test_indices_1 = load_indices_from_csv("test_indices_1.csv")
 
random.shuffle(train_indices_0)
random.shuffle(val_indices_0)
random.shuffle(test_indices_0)  

random.shuffle(train_indices_1)
random.shuffle(val_indices_1)
random.shuffle(test_indices_1) 
if(MODEL_OPTION=="GAG"):
    atrous=1
else:
    atrous=0
# ========== MODEL ==========
model = create_TriFusion(
    patch_size=PATCH_SIZE,
    patch_size_global=PATCH_SIZE_GLOBAL,
    latent_dim=LATENT_DIM,
    bands=BANDS,
    atrous=atrous
)
trainable_params = get_trainable_params(model)
print(f"Total number of trainable parameters in the model: {trainable_params}")    

import random
max_samples = 10*trainable_params
if len(train_indices_0) > (max_samples//2):
    print(f"Reducing training indices_0 from {len(train_indices_0)//2} to {max_samples//2} based on 10×param rule.")
    random.seed(42)  # for reproducibility
    train_indices_0 = random.sample(train_indices_0, max_samples//2)
else:
    print(f"Training indices ({len(train_indices_0)}) already within rule-of-thumb limit ({max_samples//2}).")

if len(train_indices_1) > (max_samples//2):
    print(f"Reducing training indices_1 from {len(train_indices_1)//2} to {max_samples//2} based on 10×param rule.")
    random.seed(42)  # for reproducibility
    train_indices_1 = random.sample(train_indices_1, max_samples//2)
else:
    print(f"Training indices ({len(train_indices_1)}) already within rule-of-thumb limit ({max_samples//2}).")

# --- Compute D, val and test limits ---
D_total = int(max_samples / 0.7)
max_val = max_test = int(((D_total) - max_samples) / 2)

print(f"Reducing val indices from {len(val_indices_0)//2}  to {max_val} based on 70-15-15 split.")

# --- Reduce val and test indices ---
if len(val_indices_0) > (max_val//2):
    random.seed(42)
    val_indices_0 = random.sample(val_indices_0, max_val)

if len(val_indices_1) > (max_val//2):
    random.seed(42)
    val_indices_1 = random.sample(val_indices_1, max_val)

print(f"Reducing test indices from {len(test_indices_0)//2}  to {max_test} based on 70-15-15 split.")

if len(test_indices_0) > (max_test//2):
    random.seed(42)
    test_indices_0 = random.sample(test_indices_0, max_test)
    
if len(test_indices_1) > (max_test//2):
    random.seed(42)
    test_indices_1 = random.sample(test_indices_1, max_test)
    
train_indices=train_indices_0+train_indices_1
val_indices=val_indices_0+val_indices_1
test_indices=test_indices_0+test_indices_1
random.shuffle(train_indices)
random.shuffle(val_indices)
random.shuffle(test_indices)


# ========== DATA GENERATORS ==========
train_gen = TriF.DataGenerator(train_indices, train_input_paths, train_target_paths,
                          patch_size=PATCH_SIZE, patch_size_global=PATCH_SIZE_GLOBAL,
                          bands=BANDS, bands_context=BANDS, batch_size=BATCH_SIZE)


val_gen = TriF.DataGenerator(val_indices, val_input_paths, val_target_paths,
                        patch_size=PATCH_SIZE, patch_size_global=PATCH_SIZE_GLOBAL,
                        bands=BANDS, bands_context=BANDS, batch_size=BATCH_SIZE)

test_gen = TriF.DataGenerator(test_indices, test_input_paths, test_target_paths,
                         patch_size=PATCH_SIZE, patch_size_global=PATCH_SIZE_GLOBAL,
                         bands=BANDS, bands_context=BANDS, batch_size=BATCH_SIZE)

del train_indices_0, train_indices_1
del val_indices_0, val_indices_1
del test_indices_0, test_indices_1
del train_indices, val_indices, test_indices
gc.collect()

# Load the best weights if they exist
try:
	model.load_weights(weights) 
except:
	print("No best weights found. Starting training from scratch.")

# ========== CALLBACKS ==========
checkpoint_cb = ModelCheckpoint(weights, monitor="val_loss", save_best_only=True)
earlystop_cb = EarlyStopping(monitor="val_loss", patience=100, restore_best_weights=True)



# ========== TRAINING ==========
if(TRAINING==1):

	model.fit(train_gen,
			  validation_data=val_gen,
			  epochs=EPOCHS,
			  callbacks=[checkpoint_cb, earlystop_cb])
			
	# ========== SAVE FINAL MODEL ==========
	model.save(weights)

	# Optional: Evaluate on test set after training
	test_results = model.evaluate(test_gen)
	print(f"Test Loss: {test_results}")


