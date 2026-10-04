//----------------------------------------------------------
// Sentinel-2 NDVI Exporter (10 m)
// Temporal aggregation: median
//----------------------------------------------------------

var START = '2025-01-01';
var END   = '2025-12-31';

var HALF = 25000;

//----------------------------------------------------------
// Cloud masking
//----------------------------------------------------------

function maskS2(image) {

    var scl = image.select('SCL');

    var mask =
        scl.neq(3)   // cloud shadow
        .and(scl.neq(8))
        .and(scl.neq(9))
        .and(scl.neq(10))
        .and(scl.neq(11));

    return image.updateMask(mask);
}

//----------------------------------------------------------

function exportNDVI(letter, roi, lon, lat, crs){

    var proj = ee.Projection(crs);

    var pt = ee.Geometry.Point([lon, lat]).transform(proj, 1);

    var xy = ee.List(pt.coordinates());

    var x = ee.Number(xy.get(0));
    var y = ee.Number(xy.get(1));

    //------------------------------------------------------
    // ROI
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
    // Sentinel-2 NDVI
    //------------------------------------------------------

    var ndvi = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(rect)
        .filterDate(START, END)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
        .map(maskS2)
        .map(function(img){

            return img.normalizedDifference(
                ['B8','B4']
            ).rename('NDVI');

        })
        .median();

    //------------------------------------------------------
    // Export
    //------------------------------------------------------

    Export.image.toDrive({

        image: ndvi.clip(rect),

        description: letter + "_ndvi_roi_" + roi,

        folder: "ndvi_" + letter,

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
exportNDVI("u",1,-121.49,38.58,"EPSG:32610");
exportNDVI("u",2,-80.84,35.23,"EPSG:32617");

// Germany
exportNDVI("g",1,13.40,52.52,"EPSG:32633");
exportNDVI("g",2,8.40,49.01,"EPSG:32632");

// China
exportNDVI("c",1,120.62,31.30,"EPSG:32651");
exportNDVI("c",2,113.26,23.13,"EPSG:32649");

// India
exportNDVI("i",1,73.86,18.52,"EPSG:32643");
exportNDVI("i",2,75.85,30.90,"EPSG:32643");

// Brazil
exportNDVI("b",1,-54.70,-2.44,"EPSG:32721");
exportNDVI("b",2,-55.50,-11.86,"EPSG:32721");

// Australia
exportNDVI("a",1,150.90,-33.81,"EPSG:32756");
exportNDVI("a",2,151.95,-27.56,"EPSG:32755");

// Poland
exportNDVI("p",1,21.01,52.23,"EPSG:32634");
exportNDVI("p",2,16.93,52.41,"EPSG:32633");

// Vietnam
exportNDVI("v",1,105.85,21.03,"EPSG:32648");
exportNDVI("v",2,105.75,10.05,"EPSG:32648");

//========================================================
// VALIDATION
//========================================================

// Argentina
exportNDVI("r",1,-60.67,-32.95,"EPSG:32721");
exportNDVI("r",2,-64.19,-31.42,"EPSG:32720");

// France
exportNDVI("f",1,3.88,43.61,"EPSG:32631");
exportNDVI("f",2,1.44,43.60,"EPSG:32631");

// Mexico
exportNDVI("m",1,-100.39,20.59,"EPSG:32614");
exportNDVI("m",2,-103.35,20.67,"EPSG:32613");

//========================================================
// TESTING
//========================================================

// Tunisia
exportNDVI("t",1,10.64,35.83,"EPSG:32632");
exportNDVI("t",2,10.10,35.68,"EPSG:32632");

// Kenya
exportNDVI("k",1,36.82,-1.29,"EPSG:32737");
exportNDVI("k",2,36.08,-0.30,"EPSG:32736");

// Indonesia
exportNDVI("n",1,110.37,-7.80,"EPSG:32749");
exportNDVI("n",2,107.61,-6.91,"EPSG:32748");

// Canada
exportNDVI("d",1,-113.49,53.55,"EPSG:32612");
exportNDVI("d",2,-106.67,52.13,"EPSG:32613");