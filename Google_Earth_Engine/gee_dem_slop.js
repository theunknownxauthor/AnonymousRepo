// Load your Massachusetts shapefile (replace with your asset path)
var massachusetts = ee.FeatureCollection("..../massachusetts_shapefile");

// Load the SRTM DEM dataset
var srtm = ee.Image("USGS/SRTMGL1_003");

// Clip the DEM to the Massachusetts boundary
var demMassachusetts = srtm.clip(massachusetts);

// Display boundary
Map.centerObject(massachusetts, 7);
Map.addLayer(massachusetts, {}, 'Massachusetts Boundary');

// Display DEM
Map.addLayer(demMassachusetts, 
  {min: 0, max: 3000, palette: ['blue', 'green', 'yellow', 'brown', 'white']}, 
  'SRTM DEM Massachusetts'
);

// Calculate slope
var slopeMassachusetts = ee.Terrain.slope(demMassachusetts);

// Display slope
Map.addLayer(slopeMassachusetts, 
  {min: 0, max: 60, palette: ['00FFFF', '008000', 'FFFF00', 'FFA500', 'FF0000']}, 
  'Slope Massachusetts'
);

// Export DEM
Export.image.toDrive({
  image: demMassachusetts,
  description: 'SRTM_DEM_Massachusetts',
  scale: 30,
  region: massachusetts.geometry(),
  fileFormat: 'GeoTIFF',
  maxPixels: 1e13
});

// Export Slope
Export.image.toDrive({
  image: slopeMassachusetts,
  description: 'SRTM_Slope_Massachusetts',
  scale: 30,   // ⚠️ Use 30 to match SRTM resolution
  region: massachusetts.geometry(),
  fileFormat: 'GeoTIFF',
  maxPixels: 1e13
});
