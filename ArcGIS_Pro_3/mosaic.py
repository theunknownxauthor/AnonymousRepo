import arcpy
import os

# Define root path
root_dataset = r"M:\proposed\code\Data Collection\Massa"
folders = ["train", "test", "val"]
output_mosaic = os.path.join(root_dataset, "mosaic_all.tif")

# List to store all image paths
raster_list = []

# Gather all .tif files from train, test, and val
for folder in folders:
    folder_path = os.path.join(root_dataset, folder)
    for file in os.listdir(folder_path):
        if file.endswith(".tiff"):
            raster_list.append(os.path.join(folder_path, file))

if raster_list:
    print(f"Found {len(raster_list)} images. Creating mosaic...")

    # Set workspace and environment
    arcpy.env.workspace = root_dataset
    arcpy.env.overwriteOutput = True

    # Use Mosaic To New Raster
    arcpy.management.MosaicToNewRaster(
        input_rasters=raster_list,
        output_location=root_dataset,
        raster_dataset_name_with_extension="mosaic_all.tif",
        coordinate_system_for_the_raster="",  # Can specify EPSG or leave blank
        pixel_type="32_BIT_FLOAT",
        number_of_bands=3,  # Change this if your images have more bands
        mosaic_method="LAST",
        mosaic_colormap_mode="MATCH"
    )

    print(f"Mosaic created: {output_mosaic}")
else:
    print("No TIFF images found.")
