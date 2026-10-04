//----------------------------------------------------------
// ESA CCI Above Ground Biomass
// Year: 2022
// Band: agb
//----------------------------------------------------------

var BIOMASS =
    ee.Image('ESA/CCI/Above_Ground_Biomass/V6_0/2022')
      .select('agb');

var HALF = 25000;

//----------------------------------------------------------

function exportAGB(letter, roi, lon, lat, crs){

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
    // Export
    //------------------------------------------------------

    Export.image.toDrive({

        image: BIOMASS.clip(rect),

        description: letter + "_agb_roi_" + roi,

        folder: "agb_" + letter,

        fileNamePrefix: "roi_" + roi,

        region: rect,

        crs: crs,

        scale: 100,     // native ESA CCI AGB resolution

        maxPixels: 1e13

    });
}

//========================================================
// TRAINING
//========================================================

// USA
exportAGB("u",1,-121.49,38.58,"EPSG:32610");
exportAGB("u",2,-80.84,35.23,"EPSG:32617");

// Germany
exportAGB("g",1,13.40,52.52,"EPSG:32633");
exportAGB("g",2,8.40,49.01,"EPSG:32632");

// China
exportAGB("c",1,120.62,31.30,"EPSG:32651");
exportAGB("c",2,113.26,23.13,"EPSG:32649");

// India
exportAGB("i",1,73.86,18.52,"EPSG:32643");
exportAGB("i",2,75.85,30.90,"EPSG:32643");

// Brazil
exportAGB("b",1,-54.70,-2.44,"EPSG:32721");
exportAGB("b",2,-55.50,-11.86,"EPSG:32721");

// Australia
exportAGB("a",1,150.90,-33.81,"EPSG:32756");
exportAGB("a",2,151.95,-27.56,"EPSG:32755");

// Poland
exportAGB("p",1,21.01,52.23,"EPSG:32634");
exportAGB("p",2,16.93,52.41,"EPSG:32633");

// Vietnam
exportAGB("v",1,105.85,21.03,"EPSG:32648");
exportAGB("v",2,105.75,10.05,"EPSG:32648");

//========================================================
// VALIDATION
//========================================================

// Argentina
exportAGB("r",1,-60.67,-32.95,"EPSG:32721");
exportAGB("r",2,-64.19,-31.42,"EPSG:32720");

// France
exportAGB("f",1,3.88,43.61,"EPSG:32631");
exportAGB("f",2,1.44,43.60,"EPSG:32631");

// Mexico
exportAGB("m",1,-100.39,20.59,"EPSG:32614");
exportAGB("m",2,-103.35,20.67,"EPSG:32613");

//========================================================
// TESTING
//========================================================

// Tunisia
exportAGB("t",1,10.64,35.83,"EPSG:32632");
exportAGB("t",2,10.10,35.68,"EPSG:32632");

// Kenya
exportAGB("k",1,36.82,-1.29,"EPSG:32737");
exportAGB("k",2,36.08,-0.30,"EPSG:32736");

// Indonesia
exportAGB("n",1,110.37,-7.80,"EPSG:32749");
exportAGB("n",2,107.61,-6.91,"EPSG:32748");

// Canada
exportAGB("d",1,-113.49,53.55,"EPSG:32612");
exportAGB("d",2,-106.67,52.13,"EPSG:32613");