from datetime import datetime, timedelta
from itertools import product

import geopandas
import matplotlib
import pandas
import requests
from icecream import ic
from shapely.geometry import box
from shapely.ops import polygonize_full

uk_boundaries_reference_url = "https://github.com/wmgeolab/geoBoundaries/blob/main/releaseData/gbOpen/GBR/ADM0/geoBoundaries-GBR-ADM0_simplified.geojson"
uk_json = "geoBoundaries-GBR-ADM0_simplified.geojson"

uk_bbox = [0, 0, 700000, 1300000]


def get_uk():
    uk_boundaries_url = "https://raw.githubusercontent.com/wmgeolab/geoBoundaries/main/releaseData/gbOpen/GBR/ADM0/geoBoundaries-GBR-ADM0_simplified.geojson"
    uk_gdf = geopandas.read_file(uk_boundaries_url)
    return uk_gdf


def create_grid_of_chips(chip_px_width: int):
    # Hardcode the assumption that one pixel==10 metres
    # chipe_dimension is the real-world chip size in metres
    chip_dimension = chip_px_width * 10

    # First create a flat dataframe with attributes
    # - chip_id
    # - x_lower_left
    # - y_lower_left
    x_range = range(uk_bbox[0], uk_bbox[2], chip_dimension)
    y_range = range(uk_bbox[1], uk_bbox[3], chip_dimension)

    ids = []
    xs = []
    ys = []

    for x, y in product(x_range, y_range):
        ids.append(get_chip_id(x, y, chip_dimension))
        xs.append(x)
        ys.append(y)

    df = pandas.DataFrame.from_dict(
        {"chip_id": ids, "x_lower_left": xs, "y_lower_left": ys}
    )

    # Now create the geometeries for each cell and convert to a geodataframe
    def _get_box(row):
        x = row["x_lower_left"]
        y = row["y_lower_left"]
        return box(x, y, x + chip_dimension, y + chip_dimension)

    df["geometry"] = df.apply(_get_box, axis=1)
    gdf = geopandas.GeoDataFrame(df, geometry="geometry", crs="EPSG:27700")

    return gdf


def get_chip_id(x, y, chip_dimension):
    """
    Create a chip_id which is unique irrespective of chip_size and meaningful
    """
    return "{:0>6}_{:0>7}_{:0>3}".format(x, y, chip_dimension)


def coastline_to_coastpoly(path_to_coastline=None):
    if path_to_coastline is None:
        path_to_coastline = "zip://strtgi_essh_gb.zip!strtgi_essh_gb/data/coastline.shp"

    coastline = geopandas.read_file(path_to_coastline)

    all_lines = coastline["geometry"].to_list()
    ic(len(all_lines))
    result, dangles, cuts, invalids = polygonize_full(all_lines)
    ic(len(result.geoms))
    ic(len(result.geoms))
    ic(len(dangles.geoms))
    ic(len(cuts.geoms))
    ic(len(invalids.geoms))

    ic(result)
    ic(dangles)
    ic(cuts)
    ic(invalids)

    polys_gdf = geopandas.GeoDataFrame(
        {"geometry": result.geoms, "id": True}, crs="EPSG:27700"
    )
    ic(polys_gdf)

    polys_gdf.to_file("coast_polys.gpkg", layer="all_polys")
    polys_gdf = polys_gdf.dissolve()
    polys_gdf.to_file("coast_polys.gpkg", layer="polys_disolved")

    # gdf['inside'] = gdf['geometry'].apply(lambda shp: shp.intersects(gdfBuffer.dissolve('LAND').iloc[0]['geometry']))

    return polys_gdf


if __name__ == "__main__":
    polys_gdf = coastline_to_coastpoly()
    # chip_px_width = 32
    options = {
        16: "chips_16",
        32: "chips_32",
        32 * 16: "images_0512",
        32 * 32: "images_1024",
    }
    options = {32 * 32: "images_1024", 32 * 16: "images_0512"}

    all_grids = {}

    for chip_px_width, layer_name in options.items():
        start_time = datetime.now()

        chips_gdf = create_grid_of_chips(chip_px_width)

        stage1_time = datetime.now()
        ic(stage1_time - start_time)
        ic(len(chips_gdf))

        coast = polys_gdf["geometry"][0]
        ic(type(coast))
        chips_gdf["on_land"] = chips_gdf["geometry"].intersects(coast)
        all_grids[layer_name] = chips_gdf

        # layer_name = f"chips_{chip_px_width:0>2}"
        chips_gdf.to_parquet(f"{layer_name}.parquet")

        stage2_time = datetime.now()
        ic(stage2_time - stage1_time)

        chips_gdf.to_file("chips.gpkg", layer=layer_name)
        stage3_time = datetime.now()
        ic(stage3_time - stage2_time)
        ic(stage3_time - start_time)
