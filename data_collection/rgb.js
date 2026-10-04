//----------------------------------------------------------
// Sentinel-2 RGB Exporter (10 m)
//----------------------------------------------------------

// Date range
var START = '2025-01-01';
var END   = '2025-12-31';

// Half-size of ROI (50 km x 50 km)
var HALF = 25000;

//----------------------------------------------------------
// Cloud masking using SCL
//----------------------------------------------------------

function maskS2(image) {

    var scl = image.select('SCL');

    var mask =
        scl.neq(3)   // cloud shadow
        .and(scl.neq(8))   // cloud medium probability
        .and(scl.neq(9))   // cloud high probability
        .and(scl.neq(10))  // cirrus
        .and(scl.neq(11)); // snow

    return image.updateMask(mask);
}

//----------------------------------------------------------

function exportRGB(letter, roi, lon, lat, crs){

    var proj = ee.Projection(crs);

    var pt = ee.Geometry.Point([lon, lat]).transform(proj, 1);

    var xy = ee.List(pt.coordinates());

    var x = ee.Number(xy.get(0));
    var y = ee.Number(xy.get(1));

    //------------------------------------------------------
    // Build 50 km × 50 km ROI
    //------------------------------------------------------

    var rect = ee.Geometry.Rectangle(
        [
            x.subtract(HALF),
            y.subtract(HALF),
            x.add(HALF),
            y.add(HALF)
        ],
        proj,
        false
    );

    //------------------------------------------------------
    // Sentinel-2 Surface Reflectance
    //------------------------------------------------------

    var rgb = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(rect)
        .filterDate(START, END)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
        .map(maskS2)
        .median()
        .select(['B4','B3','B2']); // RGB

    //------------------------------------------------------
    // Export
    //------------------------------------------------------

    Export.image.toDrive({

        image: rgb.clip(rect),

        description: letter + "_rgb_roi_" + roi,

        folder: "rgb_" + letter,

        fileNamePrefix: "roi_" + roi,

        region: rect,

        crs: crs,

        scale: 10,

        maxPixels: 1e13

    });
}

//========================================================
// TRAINING
//========================================================

// USA
exportRGB("u",1,-121.49,38.58,"EPSG:32610");
exportRGB("u",2,-80.84,35.23,"EPSG:32617");

// Germany
exportRGB("g",1,13.40,52.52,"EPSG:32633");
exportRGB("g",2,8.40,49.01,"EPSG:32632");

// China
exportRGB("c",1,120.62,31.30,"EPSG:32651");
exportRGB("c",2,113.26,23.13,"EPSG:32649");

// India
exportRGB("i",1,73.86,18.52,"EPSG:32643");
exportRGB("i",2,75.85,30.90,"EPSG:32643");

// Brazil
exportRGB("b",1,-54.70,-2.44,"EPSG:32721");
exportRGB("b",2,-55.50,-11.86,"EPSG:32721");

// Australia
exportRGB("a",1,150.90,-33.81,"EPSG:32756");
exportRGB("a",2,151.95,-27.56,"EPSG:32755");

// Poland
exportRGB("p",1,21.01,52.23,"EPSG:32634");
exportRGB("p",2,16.93,52.41,"EPSG:32633");

// Vietnam
exportRGB("v",1,105.85,21.03,"EPSG:32648");
exportRGB("v",2,105.75,10.05,"EPSG:32648");

//========================================================
// VALIDATION
//========================================================

// Argentina
exportRGB("r",1,-60.67,-32.95,"EPSG:32721");
exportRGB("r",2,-64.19,-31.42,"EPSG:32720");

// France
exportRGB("f",1,3.88,43.61,"EPSG:32631");
exportRGB("f",2,1.44,43.60,"EPSG:32631");

// Mexico
exportRGB("m",1,-100.39,20.59,"EPSG:32614");
exportRGB("m",2,-103.35,20.67,"EPSG:32613");

//========================================================
// TESTING
//========================================================

// Tunisia
exportRGB("t",1,10.64,35.83,"EPSG:32632");
exportRGB("t",2,10.10,35.68,"EPSG:32632");

// Kenya
exportRGB("k",1,36.82,-1.29,"EPSG:32737");
exportRGB("k",2,36.08,-0.30,"EPSG:32736");

// Indonesia
exportRGB("n",1,110.37,-7.80,"EPSG:32749");
exportRGB("n",2,107.61,-6.91,"EPSG:32748");

// Canada
exportRGB("d",1,-113.49,53.55,"EPSG:32612");
exportRGB("d",2,-106.67,52.13,"EPSG:32613");