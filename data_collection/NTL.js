//----------------------------------------------------------
// VIIRS Nighttime Lights Exporter
// Band: avg_rad
// Temporal aggregation: median
//----------------------------------------------------------

var START = '2023-01-01';
var END   = '2023-12-31';

var HALF = 25000;

//----------------------------------------------------------

function exportNTL(letter, roi, lon, lat, crs){

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
    // VIIRS Monthly NTL
    //------------------------------------------------------

    var ntl = ee.ImageCollection(
            'NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG'
        )
        .filterBounds(rect)
        .filterDate(START, END)
        .select('avg_rad')
        .median();

    //------------------------------------------------------
    // Export
    //------------------------------------------------------

    Export.image.toDrive({

        image: ntl.clip(rect),

        description: letter + "_ntl_roi_" + roi,

        folder: "ntl_" + letter,

        fileNamePrefix: "roi_" + roi,

        region: rect,

        crs: crs,

        scale: 463.83,      // native VIIRS resolution

        maxPixels: 1e13

    });
}

//========================================================
// TRAINING
//========================================================

// USA
exportNTL("u",1,-121.49,38.58,"EPSG:32610");
exportNTL("u",2,-80.84,35.23,"EPSG:32617");

// Germany
exportNTL("g",1,13.40,52.52,"EPSG:32633");
exportNTL("g",2,8.40,49.01,"EPSG:32632");

// China
exportNTL("c",1,120.62,31.30,"EPSG:32651");
exportNTL("c",2,113.26,23.13,"EPSG:32649");

// India
exportNTL("i",1,73.86,18.52,"EPSG:32643");
exportNTL("i",2,75.85,30.90,"EPSG:32643");

// Brazil
exportNTL("b",1,-54.70,-2.44,"EPSG:32721");
exportNTL("b",2,-55.50,-11.86,"EPSG:32721");

// Australia
exportNTL("a",1,150.90,-33.81,"EPSG:32756");
exportNTL("a",2,151.95,-27.56,"EPSG:32755");

// Poland
exportNTL("p",1,21.01,52.23,"EPSG:32634");
exportNTL("p",2,16.93,52.41,"EPSG:32633");

// Vietnam
exportNTL("v",1,105.85,21.03,"EPSG:32648");
exportNTL("v",2,105.75,10.05,"EPSG:32648");

//========================================================
// VALIDATION
//========================================================

// Argentina
exportNTL("r",1,-60.67,-32.95,"EPSG:32721");
exportNTL("r",2,-64.19,-31.42,"EPSG:32720");

// France
exportNTL("f",1,3.88,43.61,"EPSG:32631");
exportNTL("f",2,1.44,43.60,"EPSG:32631");

// Mexico
exportNTL("m",1,-100.39,20.59,"EPSG:32614");
exportNTL("m",2,-103.35,20.67,"EPSG:32613");

//========================================================
// TESTING
//========================================================

// Tunisia
exportNTL("t",1,10.64,35.83,"EPSG:32632");
exportNTL("t",2,10.10,35.68,"EPSG:32632");

// Kenya
exportNTL("k",1,36.82,-1.29,"EPSG:32737");
exportNTL("k",2,36.08,-0.30,"EPSG:32736");

// Indonesia
exportNTL("n",1,110.37,-7.80,"EPSG:32749");
exportNTL("n",2,107.61,-6.91,"EPSG:32748");

// Canada
exportNTL("d",1,-113.49,53.55,"EPSG:32612");
exportNTL("d",2,-106.67,52.13,"EPSG:32613");