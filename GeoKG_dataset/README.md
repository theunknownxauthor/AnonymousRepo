# GeoKG Dataset

GeoKG is a large-scale multimodal geospatial dataset developed for self-supervised representation learning and downstream geospatial prediction. The dataset accompanies the **GeoKRL** framework and contains co-registered raster layers extracted over multiple countries using a unified acquisition and preprocessing pipeline.

The complete dataset was generated using the scripts provided in the `data_collection/` and `data_preprocessing/` directories of this repository.

---

# Dataset Download

The complete GeoKG dataset is distributed in two compressed parts.

| File | Download |
|------|----------|
| GeoKG Part 1 | https://drive.google.com/file/d/18FeNggI7lcVv-CRhRg5W2W0FrmJMZaEy/view?usp=drive_link |
| GeoKG Part 2 | https://drive.google.com/file/d/1bET66fWxY3idqiD4gmxt8rdt5YGEwa1s/view?usp=drive_link |

> **Note**
>
> After downloading both parts, extract them into a single directory named **GeoKG**.

---

# Dataset Structure

```
GeoKG/
│
├── biomass/
│
├── building/
│
├── climate/
│
├── dem/
│
├── lc/
│
├── ndvi/
│
├── ntl/
│
├── population/
│
├── rgb/
│
├── slope/
│
└── normalization_statistics.csv
```

Each modality contains one folder per country.

Example

```
rgb/
│
├── rgb_a/
│   ├── roi_1.tif
│   └── roi_2.tif
│
├── rgb_b/
│   ├── roi_1.tif
│   └── roi_2.tif
│
...
```

The same organization is used for every modality.

---

# Dataset Organization

GeoKG contains **15 countries**, each represented by **two non-overlapping Regions of Interest (ROIs)** of **50 km × 50 km**.

Each ROI is stored as a co-registered raster stack with a spatial resolution of **10 m**, resulting in raster dimensions of **5000 × 5000 pixels** after preprocessing.

Every pixel can therefore be uniquely identified by

```
(country,
 ROI,
 row,
 column)
```

across every modality.

---

# Modalities

The dataset contains the following raster layers.

| Modality | Description | Resolution |
|-----------|-------------|------------|
| RGB | Sentinel-2 RGB imagery | 10 m |
| DEM | Digital Elevation Model | 10 m |
| Slope | Terrain slope derived from DEM | 10 m |
| NDVI | Normalized Difference Vegetation Index | 10 m |
| NTL | Nighttime Lights | 10 m |
| Climate | ERA5-Land annual median temperature | 10 m |
| Biomass | Above-ground biomass | 10 m |
| Building | World Settlement Footprint 2019 | 10 m |
| Population | WorldPop 2022 | 10 m |
| LC | ESA WorldCover land-cover map | 10 m |

All layers are projected onto the same coordinate reference system as the corresponding Sentinel-2 RGB image and perfectly aligned pixel-by-pixel.

---

# Country Split

The same country split is used for all experiments.

## Training Countries

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

---

## Validation Countries

| Code | Country |
|------|---------|
| f | France |
| m | Mexico |
| r | Argentina |

---

## Test Countries

| Code | Country |
|------|---------|
| d | Canada |
| k | Kenya |
| n | Indonesia |
| t | Tunisia |

This country-level split guarantees complete geographic separation between training, validation, and testing.

---

# Normalization Statistics

The file

```
normalization_statistics.csv
```

contains the global minimum and maximum values computed across the entire GeoKG dataset for every modality.

Example

| Layer | Minimum | Maximum |
|--------|---------:|---------:|
| Biomass | 0.0 | 526.0 |
| Building | 0.0 | 255.0 |
| Climate | 276.72472417199 | 300.43162423153 |
| DEM | -113.0 | 3082.0 |
| Land Cover | 0.0 | 8.0 |
| NDVI | -1.0 | 1.0 |
| Nighttime Lights | 0.0 | 287.17218017578 |
| Population | 0.0 | 981.05548095703 |
| RGB Red | 0.0 | 17416.0 |
| RGB Green | 0.0 | 18296.0 |
| RGB Blue | 0.0 | 18424.0 |

These statistics are automatically loaded by the GeoKRL framework during training to perform min-max normalization.

---

# Dataset Size

| Property | Value |
|----------|------:|
| Countries | 15 |
| ROIs | 30 |
| Modalities | 10 |
| ROI size | 50 km × 50 km |
| Spatial resolution | 10 m |
| Raster size | 5000 × 5000 pixels |
| Total dataset size | **≈ 20 GB** |

---

# Reproducibility

The complete GeoKG dataset can be regenerated from scratch using the scripts provided in this repository.

1. Download the raw datasets using the Google Earth Engine scripts in

```
data_collection/
```

2. Execute the ArcPy preprocessing pipeline in

```
data_preprocessing/
```

3. The resulting dataset will match the directory structure described above and can be directly used by the GeoKRL training framework.

---

# Citation

If you use the GeoKG dataset in your research, please cite:

```bibtex
Citation will be added after publication.
```

---

# License

GeoKG is released for research purposes. Users should also comply with the licenses of the original data providers (Sentinel-2, ERA5-Land, WorldPop, ESA WorldCover, World Settlement Footprint, and other referenced datasets).
