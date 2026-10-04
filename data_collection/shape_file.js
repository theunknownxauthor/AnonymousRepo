//--------------------------------------------------------
// 50 km × 50 km ROI Generator
//--------------------------------------------------------

var HALF_SIZE = 25000;   // 25 km

//--------------------------------------------------------

function exportROI(letter, roi, lon, lat, crs){

  var proj = ee.Projection(crs);

  var pt = ee.Geometry.Point([lon,lat]).transform(proj,1);

  var xy = ee.List(pt.coordinates());

  var x = ee.Number(xy.get(0));
  var y = ee.Number(xy.get(1));

  var rect = ee.Geometry.Rectangle(
      [
        x.subtract(HALF_SIZE),
        y.subtract(HALF_SIZE),
        x.add(HALF_SIZE),
        y.add(HALF_SIZE)
      ],
      proj,
      false
  );

  Export.table.toDrive({
    collection: ee.FeatureCollection([
      ee.Feature(rect)
    ]),
    description: letter+"_roi_"+roi,
    folder: "shape_file_"+letter,
    fileNamePrefix: "roi_"+roi,
    fileFormat: "SHP"
  });

}

//========================================================
// TRAINING
//========================================================

// USA
exportROI("u",1,-121.49,38.58,"EPSG:32610");
exportROI("u",2,-80.84,35.23,"EPSG:32617");

// Germany
exportROI("g",1,13.40,52.52,"EPSG:32633");
exportROI("g",2,8.40,49.01,"EPSG:32632");

// China
exportROI("c",1,120.62,31.30,"EPSG:32651");
exportROI("c",2,113.26,23.13,"EPSG:32649");

// India
exportROI("i",1,73.86,18.52,"EPSG:32643");
exportROI("i",2,75.85,30.90,"EPSG:32643");

// Brazil
exportROI("b",1,-54.70,-2.44,"EPSG:32721");
exportROI("b",2,-55.50,-11.86,"EPSG:32721");

// Australia
exportROI("a",1,150.90,-33.81,"EPSG:32756");
exportROI("a",2,151.95,-27.56,"EPSG:32755");

// Poland
exportROI("p",1,21.01,52.23,"EPSG:32634");
exportROI("p",2,16.93,52.41,"EPSG:32633");

// Vietnam
exportROI("v",1,105.85,21.03,"EPSG:32648");
exportROI("v",2,105.75,10.05,"EPSG:32648");

//========================================================
// VALIDATION
//========================================================

// Argentina
exportROI("r",1,-60.67,-32.95,"EPSG:32721");
exportROI("r",2,-64.19,-31.42,"EPSG:32720");

// France
exportROI("f",1,3.88,43.61,"EPSG:32631");
exportROI("f",2,1.44,43.60,"EPSG:32631");

// Mexico
exportROI("m",1,-100.39,20.59,"EPSG:32614");
exportROI("m",2,-103.35,20.67,"EPSG:32613");

//========================================================
// TESTING
//========================================================

// Tunisia
exportROI("t",1,10.64,35.83,"EPSG:32632");
exportROI("t",2,10.10,35.68,"EPSG:32632");

// Kenya
exportROI("k",1,36.82,-1.29,"EPSG:32737");
exportROI("k",2,36.08,-0.30,"EPSG:32736");

// Indonesia
exportROI("n",1,110.37,-7.80,"EPSG:32749");
exportROI("n",2,107.61,-6.91,"EPSG:32748");

// Canada
exportROI("d",1,-113.49,53.55,"EPSG:32612");
exportROI("d",2,-106.67,52.13,"EPSG:32613");