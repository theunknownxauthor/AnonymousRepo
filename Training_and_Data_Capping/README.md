
# Training and Data Capping Module  

---

---

## DEM and Slope Integration Required


> This training pipeline assumes that **DEM and Slope bands have been successfully added to all dataset tiles before running any script**.
>
> The DEM and Slope preprocessing procedure is fully explained in the folder:
>
> 👉 [`ArcGIS_Pro_3/`](../ArcGIS_Pro_3/)
>
> Please follow that pipeline carefully before launching training.
>
> The preprocessed tiles must be stored as:
>
> - `train_dem/` → training tiles  
> - `val_dem/` → validation tiles  
> - `test_dem/` → test tiles  
>
> Each tile must already contain the correct number of bands:
>
> - `--bands 5` → RGB + DEM + Slope  
> - `--bands 3` → RGB only  
>
> If DEM and Slope are not correctly integrated and stored in these folders, the training pipeline may fail or produce incorrect results.

---
---
## 📦 Requirements

- Python 3.8+
- TensorFlow
- numpy
- rasterio
Install dependencies:

    pip install -r requirements.txt

Create `requirements.txt`:

    tensorflow==2.15.0
    numpy
    rasterio

---
## TriFusionBD – Principled Class Balancing and Training

This folder contains the Python implementation of the **Principled Data Capping and Class-Balancing Mechanism** and the full training pipeline of the TriFusionBD model.

This module implements:

1. Pixel index generation
2. Balanced sampling (building / non-building)
3. Parameter-aware 10×M data capping
4. 70–15–15 split control
5. Model definition (TriFusion)
6. Training
7. Full-image inference

---

# Folder Contents

- `create_csv_files.py`
- `Test_TriFusion_Gate_Atrous_Gate.py`
- `TriFusion_Gate_Atrous_Gate.py`

---

# 1️⃣ Step 1 – Pixel Index Generation  
File: `create_csv_files.py`

## Purpose

Generate pixel-level indices for:
- Building class (label = 255)
- Non-building class (label = 0)

These indices define valid patch centers for training.

---

## Workflow

For each raster in:

train_dem/
val_dem/
test_dem/


The script:

1. Iterates over every valid pixel (respecting patch borders)
2. Extracts label value at (i, j)
3. Stores:
   - (file_idx, i, j)

Output CSV files:
train_indices_1_.csv
train_indices_0_.csv
val_indices_1_.csv
val_indices_0_.csv
test_indices_1_.csv
test_indices_0_.csv


Each CSV contains:

| file_idx | i | j |
|----------|---|---|

---

## Class Balancing Strategy

After collecting indices:

- Compute minimum class size
- Randomly sample equal number from both classes

This ensures strict **class balance** before model training.

---

# 2️⃣ Step 2 – Model Definition  
File: `TriFusion_Gate_Atrous_Gate.py`

This file implements the full TriFusionBD architecture.

## Architecture Overview

TriFusionBD consists of:

### 1. Probabilistic Local Branch (VAE)
- Operates on local patches
- Learns latent representation (z)
- Includes reconstruction loss
- Includes KL divergence loss

### 2. Deterministic Global Branch (Atrous)
- Multi-rate dilated convolutions
- Captures large receptive fields
- Global average pooling

### 3. Pixel Branch
- Uses center pixel features
- Normalized using predefined min/max values

### 4. Fusion
- Concatenation of:
  - Pixel context
  - VAE latent vector
  - Atrous global features (if enabled)
- Final convolution layers
- Sigmoid building probability output

---

## Loss Function

The custom loss combines:

- Reconstruction Loss (MSE)
- KL Divergence
- Binary Cross-Entropy (prediction loss)

Total loss:

L = λ1 * Recon + λ2 * KL + λ3 * BCE

λ values:
- λ1 = 1
- λ2 = 0.01
- λ3 = 1

---

## Data Generator

The custom `DataGenerator`:

- Reads raster patches using Rasterio
- Extracts:
  - Local patch
  - Global patch
  - Center pixel
- Applies normalization
- Returns:
  - Model inputs
  - Ground truth labels

Normalization ranges are explicitly defined for:
- RGB
- DEM
- Slope

This guarantees stable training.

---

# 3️⃣ Step 3 – Parameter-Aware Data Capping  
File: `Test_TriFusion_Gate_Atrous_Gate.py`

This script implements the **10×M rule**, one of the core methodological contributions.

---

## 10×M Rule

Let:

M = number of trainable parameters in the model

Maximum training samples:

D_max = 10 × M

Rationale:
- Prevents overfitting
- Controls variance
- Avoids excessive imbalance
- Ensures dataset size scales with model capacity

---

## Implementation

1. Compute number of trainable parameters:
trainable_params = get_trainable_params(model)
2. Compute cap:
max_samples = 10 * trainable_params

3. Limit each class:
max_samples / 2 per class

4. Apply random sampling (seed = 42 for reproducibility)

---

## 70–15–15 Split

After determining D_max:

- 70% → Training
- 15% → Validation
- 15% → Testing

Validation and test sets are capped proportionally:

D_total = max_samples / 0.7
max_val = max_test = (D_total - max_samples) / 2

This ensures:

- No data leakage
- Controlled dataset size 

---

# 4️⃣ Training Procedure

## Model Configuration

Configurable parameters:

- patch_size
- patch_size_global
- latent_dim
- bands (3 / 4 / 5)
- batch_size
- atrous (on/off)

Weights saved as:
model.fit(
train_gen,
validation_data=val_gen,
epochs=100,
callbacks=[checkpoint_cb, earlystop_cb]
)

Includes:

- ModelCheckpoint
- EarlyStopping
---

## ▶ Run Training from Command Line

Training is executed using:


Test_TriFusion_Gate_Atrous_Gate.py

### Example 1 — Train with 5 Bands (RGB + DEM + Slope)

```bash
python3 Test_TriFusion_Gate_Atrous_Gate.py \
  --patch_size_global 35 \
  --latent_dim 50 \
  --batch_size 16 \
  --training 1 \
  --bands 5
```
### Example 2 — — Train with 3 Bands (RGB Only)

```bash

python3 Test_TriFusion_Gate_Atrous_Gate.py \
  --patch_size_global 35 \
  --latent_dim 50 \
  --batch_size 16 \
  --training 1 \
  --bands 3
```

▶ Output Weights

Weights are automatically saved as:
TriFusion_Option_GAG_Bands_<BANDS>_Global_<PATCH_SIZE_GLOBAL>_Laten_<LATENT_DIM>_Laten_<BATCH_SIZE>.h5

### Example:
TriFusion_Option_GAG_Bands_5_Global_35_Laten_50_Laten_16.h5

The best validation model is saved using ModelCheckpoint.


---

# 5️⃣  Full-Image Inference

The script includes:predict_test_images()

This:

- Slides across entire image
- Uses batch inference
- Saves predicted raster
- Supports checkpoint resume
- Writes GeoTIFF output

Output:output/*.tif

---

# Reproducibility Guarantees

✔ Fixed random seed (42)  
✔ Explicit normalization ranges  
✔ Explicit parameter-based capping  
✔ Explicit 70–15–15 split  
✔ No implicit sampling  
✔ Deterministic CSV index storage  

All preprocessing scripts are provided in:

- `ArcGIS_Pro_3/`
- `Google_Earth_Engine/`

---

# Relation to the Manuscript

This module implements:

- The **Principled Data Capping Mechanism (10×M)**
- Class-balanced patch sampling
- Pixel-level probabilistic–detinistic fusion training
- Terrain-aware building detection

The training strategy directly supports the claims made in:

“TriFusionBD: A probabilistic–deterministic deep fusion network for building extraction using satellite imagery and auxiliary geospatial data”

---

# Summary

This folder contains the complete and reproducible implementation of:

- Balanced pixel sampling
- Parameter-aware data capping
- TriFusionBD architecture
- Training pipeline
- Full-image inference










