# Data Preprocessing

This directory contains the Python scripts used to transform the raw geospatial data collected from Google Earth Engine and external datasets into the standardized **GeoKG** benchmark used by GeoKRL.

All preprocessing scripts were developed using **ArcPy (ArcGIS Pro)** and ensure that every raster layer shares the same coordinate system, spatial resolution, spatial extent, and pixel alignment.

The complete preprocessing workflow is illustrated below.

```
Raw Data
│
├── Google Earth Engine exports
│      (RGB, DEM, NDVI, Climate, NTL, Land Cover, Biomass, ...)
│
├── WorldPop (Population)
│
└── World Settlement Footprint 2019 (Building)
        │
        ▼
ROI Extraction
        │
        ▼
Projection & Spatial Alignment
        │
        ▼
Global Normalization Statistics
        │
        ▼
GeoKG Dataset
```

---

# Folder Contents

| Script | Description |
|----------|-------------|
| `extract_roi_from_worldpop_files.py` | Extracts population rasters for every ROI from the WorldPop country rasters. |
| `extract_roi_from_wsf_tiles.py` | Extracts building footprint rasters from the World Settlement Footprint (WSF2019) tiles. |
| `ProjectRasterSnapRaster_same_as_rgb_files.py` | Reprojects, resamples, and aligns every modality to the RGB reference grid. |
| `compute_global_normalization_statistics.py` | Computes global minimum and maximum values for every modality and generates the normalization file used during model training. |

---

# Processing Pipeline

The preprocessing pipeline consists of four consecutive stages.

## 1. ROI Extraction

The ROI shapefiles generated during the data collection stage are used to crop datasets that cannot be exported directly from Google Earth Engine.

### Population

WorldPop provides one raster per country rather than individual ROI exports.

`extract_roi_from_worldpop_files.py` performs the following operations:

- identifies the appropriate WorldPop raster for each country;
- clips the raster using the corresponding ROI shapefile;
- projects the clipped raster to the ROI coordinate reference system;
- saves one population raster for every country and ROI.

---

### Building Footprints

Building footprints are obtained from the **World Settlement Footprint 2019 (WSF2019)** dataset.

Because WSF is distributed as multiple raster tiles, `extract_roi_from_wsf_tiles.py` automatically:

- identifies all tiles intersecting each ROI;
- mosaics the intersecting tiles;
- clips the mosaic using the ROI boundary;
- projects the output into the ROI coordinate system.

The resulting raster contains the building footprint corresponding to the same geographic region as the remaining modalities.

---

## 2. Spatial Alignment

Different geospatial datasets exhibit different:

- coordinate systems,
- spatial resolutions,
- raster origins,
- pixel grids.

To enable pixel-wise multimodal learning, every modality must be perfectly aligned.

`ProjectRasterSnapRaster_same_as_rgb_files.py` performs the following operations for every raster:

- projects the raster into the RGB coordinate system;
- resamples it to a spatial resolution of **10 m**;
- snaps every pixel to the RGB grid;
- clips the raster to the exact RGB spatial extent;
- produces a standardized **5000 × 5000** raster.

Continuous variables (e.g., DEM, NDVI, Climate, Population, Biomass) are resampled using **bilinear interpolation**, whereas categorical layers (e.g., Land Cover and Building) are resampled using **nearest-neighbor interpolation** to preserve class labels.

After this stage, every modality shares exactly the same:

- coordinate reference system;
- raster dimensions;
- pixel size;
- raster extent;
- pixel alignment.

This guarantees that a pixel located at position *(i, j)* corresponds to the same geographic location across all modalities.

---

## 3. Global Normalization Statistics

The GeoKRL framework applies min-max normalization independently to every modality.

`compute_global_normalization_statistics.py` scans every raster contained in the GeoKG dataset and computes the global minimum and maximum values.

For RGB imagery, statistics are computed independently for:

- Red
- Green
- Blue

For every other modality, one global minimum and maximum are computed across the entire dataset.

The resulting statistics are stored in:

```
GeoKG/
└── normalization_statistics.csv
```

which is automatically loaded during model training.

---

# Output

After preprocessing, the standardized dataset has the following structure:

```
GeoKG/

├── rgb/
├── dem/
├── slope/
├── ndvi/
├── ntl/
├── climate/
├── biomass/
├── building/
├── population/
├── lc/
└── normalization_statistics.csv
```

Each modality contains one folder per country and two aligned ROIs.

Example:

```
GeoKG/

└── rgb/
    └── rgb_g/
        ├── roi_1.tif
        └── roi_2.tif
```

Every raster inside GeoKG has:

- identical dimensions (5000 × 5000 pixels);
- identical spatial resolution (10 m where applicable);
- identical coordinate reference system;
- identical raster extent;
- perfectly aligned pixels across all modalities.

This standardized representation enables efficient multimodal sampling during GeoKRL backbone pretraining.
