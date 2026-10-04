import os
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = os.path.join(PROJECT_ROOT, "GeoKG")
NORMALIZATION_CSV = os.path.join(DATASET_ROOT, "normalization_statistics.csv")
TRAIN_COUNTRIES = ["a", "b", "c", "g","i", "p", "u", "v"]
VALIDATION_COUNTRIES = ["d", "m", "r"]
TEST_COUNTRIES = ["f", "k", "n", "t"]
ROI_NAMES = ["roi_1","roi_2"]
COUNTRIES = {   "a": "Australia",
                "b": "Brazil",
                "c": "China",
                "d": "Canada",
                "f": "France",
                "g": "Germany",
                "i": "India",
                "k": "Kenya",
                "m": "Mexico",
                "n": "Indonesia",
                "p": "Poland",
                "r": "Argentina",
                "t": "Tunisia",
                "u": "United States",
                "v": "Vietnam"}
                
RGB = "rgb"
DEM = "dem"
SLOPE = "slope"
NDVI = "ndvi"
LAND_COVER = "lc"
NTL = "ntl"
CLIMATE = "climate"

ANCILLARY_LAYERS = [    DEM,
                        SLOPE,
                        NDVI,
                        LAND_COVER,
                        NTL,
                        CLIMATE
                        ]

TASK_POPULATION = "population"
TASK_BIOMASS = "biomass"
TASK_BUILDING = "building"

TASKS = [   TASK_POPULATION,
            TASK_BIOMASS,
            TASK_BUILDING] 

TASK_MODALITIES = { TASK_POPULATION: [  NTL,
                                        LAND_COVER,
                                        SLOPE],

                    TASK_BIOMASS: [ DEM,
                                    NDVI,
                                    CLIMATE
                                    ],

                    TASK_BUILDING: [    NTL,
                                        SLOPE,
                                        NDVI] }
RASTER_WIDTH = 5000
RASTER_HEIGHT = 5000
RASTER_RESOLUTION = 10
RGB_BANDS = 3
SELECTED_MODALITIES = 3
PIXEL_BANDS = RGB_BANDS + SELECTED_MODALITIES
CONTEXT_BANDS = RGB_BANDS + SELECTED_MODALITIES
MASK_DIM = 7
RANDOM_SEED = 42

# ==========================================================
# Output
# ==========================================================
CHECKPOINT_DIR = "auto_train_checkpoints"
MODEL_DIR = "auto_train_models"
LOG_DIR = "auto_train_logs"
CSV_INDEX_DIR = "auto_train_csv_indexes"
RESULT_DIR = "auto_train_results"

TASK_CHECKPOINT_DIR = "task_auto_train_checkpoints"
TASK_MODEL_DIR = "task_auto_train_models"
TASK_LOG_DIR = "task_auto_train_logs"
TASK_CSV_INDEX_DIR = "task_auto_train_csv_indexes"
TASK_RESULT_DIR = "task_auto_train_results"


ROI_GPS = {
    "a": { "roi_1": (-33.81, 150.90), "roi_2": (-27.56, 151.95),},
    "b": { "roi_1": (-2.44, -54.70), "roi_2": (-11.86, -55.50),},
    "c": { "roi_1": (31.30, 120.62), "roi_2": (23.13, 113.26), },
    "d": { "roi_1": (53.55, -113.49), "roi_2": (52.13, -106.67), },
    "f": { "roi_1": (43.61, 3.88), "roi_2": (43.60, 1.44), },
    "g": { "roi_1": (52.52, 13.40), "roi_2": (49.01, 8.40), },
    "i": { "roi_1": (18.52, 73.86), "roi_2": (30.90, 75.85), },
    "k": { "roi_1": (-1.29, 36.82), "roi_2": (-0.30, 36.08), },
    "m": { "roi_1": (20.59, -100.39), "roi_2": (20.67, -103.35),},
    "n": { "roi_1": (-7.80, 110.37), "roi_2": (-6.91, 107.61), },
    "p": { "roi_1": (52.23, 21.01), "roi_2": (52.41, 16.93), },
    "r": { "roi_1": (-32.95, -60.67),"roi_2": (-31.42, -64.19), },
    "t": { "roi_1": (35.83, 10.64), "roi_2": (35.68, 10.10), },
    "u": { "roi_1": (38.58, -121.49), "roi_2": (35.23, -80.84), },
    "v": { "roi_1": (21.03, 105.85), "roi_2": (10.05, 105.75), }, }
