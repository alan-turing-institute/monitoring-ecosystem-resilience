import re
from itertools import product
from typing import List
from unittest import result

import geopandas
import numpy as np
import pandas
from icecream import ic
from osgeo import gdal, osr
from shapely.geometry import Polygon, box
from shapely.ops import polygonize_full

chip_search = re.compile("(?P<x>\d{6})_(?P<y>\d{7})_(?P<d>\d+)")


def create_grid_of_chips(
    chip_px_width: int, pixel_scale: int, target_crs: str, bounding_box: List[int]
) -> geopandas.GeoDataFrame:
    """
    @param chip_px_width The number of pixels along each edge fo the a chip/image
    @param pixel_scale The real-world size of one pixel in the units of the target_crs
    @param target_crs The target CRS
    """
    # Hardcode the assumption that one pixel==10 metres
    # chip_dimension is the real-world chip size in metres
    chip_dimension = chip_px_width * pixel_scale

    # First create a flat dataframe with attributes
    # - chip_id
    # - x_lower_left
    # - y_lower_left
    x_range = range(bounding_box[0], bounding_box[2], chip_dimension)
    y_range = range(bounding_box[1], bounding_box[3], chip_dimension)

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
    gdf = geopandas.GeoDataFrame(df, geometry="geometry", crs=target_crs)

    return gdf


def get_chip_id(x: int, y: int, chip_dimension: int) -> str:
    """
    Create a chip_id which is both meaningful and unique irrespective of chip_size.
    """
    return "{:0>6}_{:0>7}_{:0>3}".format(x, y, chip_dimension)


def get_image_id_from_chip(chip_id: str, chip_dimension: int) -> str:
    # result = re.match("(?P<x>\d{6})_(?P<y>\d{7})_(?P<d>\d+)", chip_id)
    result = chip_search.match(chip_id)
    x_source = int(result.group("x"))
    y_source = int(result.group("y"))
    d_source = int(result.group("d"))

    if d_source > chip_dimension:
        # ic(d_source, chip_dimension)
        raise ValueError(
            "Cannot derive chip_id of a chip which is smaller than the source id"
        )

    return get_parent_image_id(
        child_x=x_source, child_y=y_source, parent_chip_dimension=chip_dimension
    )


def get_parent_image_id(child_x: int, child_y: int, parent_chip_dimension: int) -> str:

    x_dest = child_x - (child_x % parent_chip_dimension)
    y_dest = child_y - (child_y % parent_chip_dimension)

    return get_chip_id(x_dest, y_dest, parent_chip_dimension)


def coastline_to_poly(path_to_coastline: str, crs: str) -> Polygon:

    if path_to_coastline is None:
        path_to_coastline = "zip://strtgi_essh_gb.zip!strtgi_essh_gb/data/coastline.shp"

    coastline = geopandas.read_file(path_to_coastline)

    all_lines = coastline["geometry"].to_list()
    result, dangles, cuts, invalids = polygonize_full(all_lines)

    # Check that there where no errors
    assert len(dangles.geoms) == 0
    assert len(cuts.geoms) == 0
    assert len(invalids.geoms) == 0

    polys_gdf = geopandas.GeoDataFrame({"geometry": result.geoms, "id": True}, crs=crs)

    polys_gdf = polys_gdf.dissolve()
    coast = polys_gdf["geometry"][0]

    return coast


def create_raster_of_chips(
    chip_px_width: int,
    pixel_scale: int,
    target_crs: str,
    bounding_box: List[int],
    base_output_filename: str,
):
    """
    Using https://stackoverflow.com/a/33950009
    """
    #  Choose some Geographic Transform
    # uk_bbox = [0, 0, 700000, 1300000]
    # chip_px_width = 32
    chip_dimension = chip_px_width * pixel_scale

    # First create a flat dataframe with attributes
    # - chip_id
    # - x_lower_left
    # - y_lower_left
    x_range = range(bounding_box[0], bounding_box[2], chip_dimension)
    y_range = range(bounding_box[1], bounding_box[3], chip_dimension)
    image_size = (len(y_range), len(x_range))
    ic(image_size)

    #  Create Each Channel
    pixels = np.zeros((image_size), dtype=np.uint8)

    # set geotransform
    nx = image_size[1]
    ny = image_size[0]

    #  Set the Pixel Data
    for x, y in product(range(0, nx), range(0, ny)):
        # Create a checker board effect
        if bool(x % 2) == bool(y % 2):
            pixels[y, x] = 255
        else:
            pixels[y, x] = 0

    # Create the transformation
    # The raster should be aligned at the top-left of the image
    # Whilst the grid is aligned at the bottom-left
    xmin, ymin, xmax, ymax = bounding_box
    top_left = ymax + chip_dimension - (ymax % chip_dimension)
    geotransform = (xmin, chip_dimension, 0, top_left, 0, -chip_dimension)

    # Get the EPSG number as an int
    result = re.match("EPSG:(?P<code>\d+)", target_crs)
    epsg_code = int(result.group("code"))
    ic(epsg_code)

    # create the 3-band raster file
    dst_ds = gdal.GetDriverByName("GTiff").Create(
        f"{base_output_filename}.tiff", nx, ny, 1, gdal.GDT_Byte
    )

    dst_ds.SetGeoTransform(geotransform)  # specify coords
    srs = osr.SpatialReference()  # establish encoding
    srs.ImportFromEPSG(epsg_code)  # WGS84 lat/long
    dst_ds.SetProjection(srs.ExportToWkt())  # export coords to file
    dst_ds.GetRasterBand(1).WriteArray(pixels)  # write r-band to the raster
    dst_ds.FlushCache()  # write to disk
    dst_ds = None
