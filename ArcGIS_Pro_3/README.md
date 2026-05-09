# The DEM and Slope Preprocessing Module

---

##  Important – Full Preprocessing Dependency Chain

> This ArcGIS preprocessing stage is part of a multi-step pipeline and **cannot be executed independently**.
>
> The correct order is:
>
> ```
> ArcGIS (generate mosaic shapefile)
>        ↓
> Google Earth Engine (compute DEM & Slope)
>        ↓
> ArcGIS (project, resample, and add DEM/Slope bands to RGB tiles)
>        ↓
> Training_and_Data_Capping
> ```
>
> Step details:
>
> 1️⃣ In **ArcGIS**, generate the mosaic shapefile (`mosaic_extent.shp`).  
> 2️⃣ Upload this shapefile to the Google Earth Engine workflow described in  👉 [`Google_Earth_Engine/`](../Google_Earth_Engine/)
>
> This stage generates:
> - `dem_mosaic_extent_project.tif`
> - `slope_mosaic_extent_project2.tif`
>
> 3️⃣ Return to **ArcGIS** and use these files to:
>    - Reproject and resample to match RGB resolution
>    - Add DEM and Slope bands to each RGB tile
>    - Save outputs into:
>       - `train_dem/`
>       - `val_dem/`
>       - `test_dem/`
>
> Skipping the GEE stage or breaking this order will result in missing or misaligned DEM/Slope layers and incorrect training data.

---
---

This folder documents the complete ArcGIS Pro 3 preprocessing workflow used in the TriFusionBD study for building extraction using satellite imagery and auxiliary topographic data.

All preprocessing steps were implemented using ArcPy (Python API for ArcGIS Pro 3) to ensure full reproducibility.

The objective of this pipeline is to:

1. Mosaic all satellite tiles (train/val/test) into a single raster.
2. Generate a shapefile representing the global spatial extent.
3. Use that shapefile in Google Earth Engine (GEE) to extract DEM and slope layers.
4. Reproject and resample DEM and slope data to match the imagery grid.
5. Add DEM and slope as additional bands to each satellite tile.
6. Create final augmented train/validation/test datasets.

---

# Folder Contents

- mosaic.py  
- shp_from_mosaic_all.py  
- adding_dem_slope.py  

---

# Software Requirements

- ArcGIS Pro 3.x  
- Spatial Analyst Extension  
- ArcPy (included with ArcGIS Pro)  

To enable the Spatial Analyst extension in Python:

```python
arcpy.CheckOutExtension("Spatial")
```

---

# Step 1 – Create Global Mosaic  
File: mosaic.py

Purpose:  
Combine all TIFF satellite images from train, validation, and test folders into a single raster.

Input Directory Structure:

Massa/
   ├── train/
   ├── val/
   ├── test/

The script:

- Iterates through all `.tiff` files.
- Collects raster paths.
- Uses arcpy.management.MosaicToNewRaster.
- Output pixel type: 32_BIT_FLOAT
- Number of bands: 3 (RGB imagery)
- Mosaic method: LAST
- Colormap mode: MATCH

Output:

mosaic_all.tif

This mosaic is used only to determine the full spatial extent of the dataset.

---

# Step 2 – Create Extent Shapefile  
File: shp_from_mosaic_all.py

Purpose:  
Generate a polygon shapefile representing the full spatial coverage of the imagery dataset.

Input:
- mosaic_all.tif

Output:
- mosaic_extent.shp  
- Final shapefile stored in the folder: 👉 [**`Final Extent Shapefile/`**](https://github.com/theunknownxauthor/TriFusionBD/tree/main/ArcGIS_Pro_3/Final%20Extent%20Shapefile)
Method:

- Extract raster extent using arcpy.Describe.
- Build polygon from corner coordinates.
- Preserve spatial reference.
- Export as shapefile.

This shapefile is uploaded to Google Earth Engine (GEE) to:

- Clip SRTM DEM
- Compute slope
- Ensure spatial alignment between imagery and terrain data

---

# Step 3 – Google Earth Engine Stage

The shapefile `mosaic_extent.shp` is uploaded to Google Earth Engine.

Data source:
- SRTM DEM (USGS/SRTMGL1_003)
- Native resolution: ~30 meters
- Elevation unit: meters above sea level

Processing in GEE:
- DEM clipped to dataset extent
- Slope computed using ee.Terrain.slope()
- Outputs exported as GeoTIFF

Generated files:
- dem_mosaic_extent_project.tif
- slope_mosaic_extent_project2.tif

> [!IMPORTANT]
> The native spatial resolution of SRTM DEM data is approximately 30 m, whereas the satellite imagery used in this study has a spatial resolution of approximately 1 m.  
> Consequently, proper projection, alignment, and resampling of the DEM and derived slope layers are mandatory prior to band fusion to ensure spatial consistency and prevent misregistration artifacts.
>
> The corresponding Google Earth Engine (GEE) implementation required to generate these datasets is available in:
>
> 👉 [`Google_Earth_Engine/`](../Google_Earth_Engine/)
>

# Step 4 – Add DEM and Slope Bands  
File: adding_dem_slope.py

Purpose:  
Integrate DEM and slope as additional channels into each satellite tile.

For each image in:

train/
val/
test/

The script:

1. Loads the satellite raster.
2. Extracts corresponding DEM and slope using ExtractByMask.
3. Ensures matching spatial extent.
4. Composites:
   - Original RGB bands
   - DEM band
   - Slope band
5. Saves new 5-band GeoTIFF.

Output Structure:

Massa/
   ├── train_dem/
   ├── val_dem/
   ├── test_dem/

Each output raster contains:

Band 1–3 → RGB  
Band 4 → DEM  
Band 5 → Slope  

---

# Projection and Resampling

All terrain layers are projected to match the CRS of the Massachusetts imagery dataset.

Resampling method:
- Bilinear interpolation (appropriate for continuous elevation data)

Design decision:
DEM and slope are treated as low-frequency topographic context rather than fine-scale elevation detail. This prevents misinterpretation of resampled terrain data at 1 m resolution.

This choice is explicitly discussed in the manuscript to address resolution mismatch and spatial autocorrelation concerns.

---

# Final Dataset Characteristics

Each output raster:

- GeoTIFF format
- 5 bands
- Identical CRS
- Identical spatial resolution
- Pixel-aligned with original imagery

These datasets are used directly in TriFusionBD training and evaluation.

---

# Reproducibility Instructions

To reproduce the full preprocessing pipeline:

1. Run mosaic.py  
2. Run shp_from_mosaic_all.py  
3. Upload mosaic_extent.shp to Google Earth Engine and export DEM and slope  
4. Run adding_dem_slope.py  

All parameters are explicitly defined in the scripts.  
No manual GIS operations were performed outside these scripts.

---

# Relation to the TriFusionBD Paper

This ArcGIS Pro 3 pipeline ensures:

- Strict spatial consistency  
- Controlled terrain-data integration  
- Reproducible preprocessing  
- Proper DEM/slope alignment  

The resulting 5-band datasets (RGB + DEM + Slope) are used in the probabilistic–deterministic fusion network described in the TriFusionBD manuscript to improve building extraction performance, particularly in topographically complex regions.
