import arcpy
from arcpy import env
from arcpy.sa import *
import os
from arcpy.management import CompositeBands

# Enable Spatial Analyst extension
arcpy.CheckOutExtension("Spatial")

# Set workspace and input paths
base_folder = r"M:\proposed\code\Data Collection\Massa"
slope_path = r"M:\proposed\code\Data Collection\Massa\slope_mosaic_extent_project2.tif"
dem_path = r"M:\proposed\code\Data Collection\Massa\dem_mosaic_extent_project.tif"

# Load DEM and Slope data
srtm_dem = Raster(dem_path)
srtm_slope = Raster(slope_path)

# Function to add DEM and slope bands
def add_dem_slope_bands(image_path, output_path):
    print(f"Processing: {image_path}")

    # Get raster properties
    raster = Raster(image_path)
    extent = raster.extent
    cell_size = raster.meanCellWidth

    extracted_dem = ExtractByMask(srtm_dem, raster)
    extracted_slope = ExtractByMask(srtm_slope, raster)

    # Composite the bands (original + DEM + slope) and save directly
    CompositeBands([raster, extracted_dem, extracted_slope], output_path)
    print(f"Saved: {output_path}")

# Iterate through test, train, and val folders
for folder in ['test', 'train', 'val']:
    folder_path = os.path.join(base_folder, folder)
    output_folder = os.path.join(base_folder, f"{folder}_dem")
    os.makedirs(output_folder, exist_ok=True)
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.tiff'):
            image_path = os.path.join(folder_path, filename)
            output_name = os.path.splitext(filename)[0] + ".tif"
            output_path = os.path.join(output_folder, output_name)
            add_dem_slope_bands(image_path, output_path)


