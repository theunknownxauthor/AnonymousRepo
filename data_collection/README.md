# Data Collection

This directory contains the Google Earth Engine (GEE) JavaScript scripts used to collect the raw geospatial datasets required by the GeoKRL framework.

Each script exports one specific geospatial layer for all Regions of Interest (ROIs) used in the GeoKG benchmark. Every exported ROI covers an area of **50 km × 50 km** and is stored individually in Google Drive.

---

## Folder Contents

| File | Description |
|------|-------------|
| `rgb.js` | Exports Sentinel-2 RGB imagery. |
| `NDVI.js` | Computes and exports annual NDVI composites. |
| `LC.js` | Exports ESA WorldCover land-cover maps. |
| `NTL.js` | Exports nighttime light imagery. |
| `Climate.js` | Exports ERA5-Land annual median temperature. |
| `Biomass.js` | Exports above-ground biomass. |
| `dem_slope.js` | Exports Digital Elevation Model (DEM) and terrain slope. |
| `shape_file.js` | Generates ROI shapefiles used for extracting World Settlement Footprint (WSF 2019) and WorldPop 2022 datasets outside Google Earth Engine. |

---

## Region of Interest (ROI)

Each country is represented by a **single-character identifier** and contains **two independent Regions of Interest (ROIs)**.

Every ROI is defined by:

- center longitude
- center latitude
- projected coordinate reference system (UTM)
- fixed size of **50 km × 50 km**

For example,

```javascript
exportClimate("g",1,13.40,52.52,"EPSG:32633");
```

exports the first German ROI.

---

## Dataset Split

The ROI locations are fixed throughout the entire GeoKRL framework and correspond to the benchmark splits used during pretraining and downstream evaluation.

### Training Countries

| Code | Country |
|------|---------|
| a | Australia |
| b | Brazil |
| c | China |
| g | Germany |
| i | India |
| p | Poland |
| u | United States |
| v | Vietnam |

### Validation Countries

| Code | Country |
|------|---------|
| f | France |
| m | Mexico |
| r | Argentina |

### Testing Countries

| Code | Country |
|------|---------|
| d | Canada |
| k | Kenya |
| n | Indonesia |
| t | Tunisia |

Each country contains two ROIs (`roi_1` and `roi_2`).

---

## Export Procedure

All scripts follow the same workflow:

1. Define the ROI from its center coordinates.
2. Construct a 50 km × 50 km square in the appropriate projected CRS.
3. Load the corresponding Earth Engine dataset.
4. Compute the desired annual composite (when required).
5. Clip the raster to the ROI.
6. Export the raster to Google Drive.

The only exception is **`shape_file.js`**, which exports ROI polygons instead of raster data.

---

## ROI Shapefiles

The generated shapefiles are **not used directly by GeoKRL**.

Instead, they serve as extraction masks for datasets that are processed outside Google Earth Engine:

- **World Settlement Footprint (WSF 2019)** (building footprints)
- **WorldPop 2022** (population density)

The extracted rasters are subsequently aligned with the remaining GeoKG layers during the preprocessing stage.

---

## Output Organization

Each script exports its results into a dedicated Google Drive folder.

Example:

```
rgb_a/
    roi_1.tif
    roi_2.tif

climate_g/
    roi_1.tif
    roi_2.tif

ntl_u/
    roi_1.tif
    roi_2.tif
```

This directory organization is preserved during the preprocessing stage and is expected by the GeoKRL training pipeline.

---

## Notes

- All ROIs use projected UTM coordinate systems to preserve metric distances.
- Every exported raster corresponds to exactly one country and one ROI.
- The same ROIs are used consistently across all geospatial layers to ensure perfect spatial alignment after preprocessing.
- The benchmark split (training, validation, and testing countries) remains fixed for all experiments.
