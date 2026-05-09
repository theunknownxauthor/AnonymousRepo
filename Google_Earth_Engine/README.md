# Google Earth Engine – DEM and Slope Preprocessing

This folder contains the Google Earth Engine (GEE) script used in the TriFusionBD study for generating Digital Elevation Model (DEM) and slope layers for Massachusetts (USA).

File included:
- `gee_dem_slop.js` – Script for DEM extraction, slope computation, and export.

---

## 1. Data Sources

### Administrative Boundary
- **Dataset:** User-uploaded Massachusetts shapefile  
- **Platform:** Google Earth Engine (Assets)  
- **Asset Type:** ee.FeatureCollection  
- **Example Asset Path:**  
  `users/<your_username>/massachusetts_shapefile`

The official Massachusetts state boundary shapefile was uploaded manually to Google Earth Engine and used as the spatial constraint for clipping DEM and slope layers.

⚠️ Replace `<your_username>` and `massachusetts_shapefile` in the script with your actual GEE asset path.

---

### Digital Elevation Model (DEM)

- **Dataset:** SRTM Global 1 Arc-Second  
- **Asset ID:** `USGS/SRTMGL1_003`  
- **Native spatial resolution:** ~30 meters  
- **Elevation unit:** meters above sea level  

Official dataset documentation:  
https://developers.google.com/earth-engine/datasets/catalog/USGS_SRTMGL1_003

---

## 2. Processing Workflow

The GEE script performs the following steps:

1. Load the uploaded Massachusetts shapefile from GEE Assets.
2. Load the SRTM DEM dataset (`USGS/SRTMGL1_003`).
3. Clip the DEM to the Massachusetts boundary geometry.
4. Compute slope using `ee.Terrain.slope()`:
   - Slope unit: **degrees**
   - Range: 0° (flat terrain) to 90° (vertical)
   - Computed using 4-connected neighbors
5. Visualize DEM and slope layers in the GEE Map interface.
6. Export DEM and slope layers to Google Drive as GeoTIFF files.

No additional filtering or terrain corrections were applied.

---

## 3. Export Parameters

### DEM Export

- Scale: 30 meters (native SRTM resolution)
- Projection: EPSG:26986 (NAD83 / Massachusetts Mainland State Plane)
- File format: GeoTIFF
- maxPixels: 1e13

---

### Slope Export

- Derived from SRTM DEM
- Export scale: 30 meters (recommended to preserve native resolution)
- File format: GeoTIFF
- maxPixels: 1e13

---

### Important Note on Resolution Alignment

SRTM native resolution is 30 m.

When integrating DEM and slope layers with 1 m satellite imagery in TriFusionBD:

- Bilinear interpolation was used during resampling.
- DEM is treated as low-frequency topographic context.
- It is **not** interpreted as fine-scale elevation detail.

All final resampling and spatial alignment were performed locally during the TriFusionBD training pipeline.

---

