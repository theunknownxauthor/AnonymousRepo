import arcpy
import os

# Paths
root_dataset = r"M:\proposed\code\Data Collection\Massa"
mosaic_path = os.path.join(root_dataset, "mosaic_all.tif")
shapefile_path = os.path.join(root_dataset, "mosaic_extent.shp")

# Set environment
arcpy.env.workspace = root_dataset
arcpy.env.overwriteOutput = True

# Describe the raster to get extent
desc = arcpy.Describe(mosaic_path)
extent = desc.extent
spatial_ref = desc.spatialReference

# Create a polygon from the extent
array = arcpy.Array([
    arcpy.Point(extent.XMin, extent.YMin),
    arcpy.Point(extent.XMin, extent.YMax),
    arcpy.Point(extent.XMax, extent.YMax),
    arcpy.Point(extent.XMax, extent.YMin),
    arcpy.Point(extent.XMin, extent.YMin)  # Close the polygon
])
polygon = arcpy.Polygon(array, spatial_ref)

# Save as shapefile
arcpy.CopyFeatures_management(polygon, shapefile_path)

print(f"Shapefile created: {shapefile_path}")
