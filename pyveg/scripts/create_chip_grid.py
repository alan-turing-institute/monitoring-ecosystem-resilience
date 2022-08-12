from datetime import datetime

import fiona
from icecream import ic

from pyveg.src.grid_creator import (
    coastline_to_poly,
    create_grid_of_chips,
    create_raster_of_chips,
    get_parent_image_id,
)

# A collection of config options
config = {
    "bounding_box": [0, 0, 700000, 1300000],
    "target_crs": "EPSG:27700",
    "path_to_coastline": "zip://../chips/strtgi_essh_gb.zip!strtgi_essh_gb/data/coastline.shp",
    "output_path": "./data",
    "pixel_scale": 10,
}


"""
The `output_options` dict should be specified in the format
<chip_width_in_pixels> : <base_output_filename>

One output file name will be created for each key/values pair
"""
# Production values
output_options = {
    32: "chips_32",
    32 * 16: "images_0512",
    32 * 32: "images_1024",
}

# Testing values
# output_options = {
#     32 * 16: "images_0512",
#     32 * 32: "images_1024",
# }


def get_larger_chip_id_func(chip_dimension: int):
    def apply_larger_chip_id(row):
        return get_parent_image_id(
            child_x=row["x_lower_left"],
            child_y=row["y_lower_left"],
            parent_chip_dimension=chip_dimension,
        )

    return apply_larger_chip_id


if __name__ == "__main__":

    try:
        coast = coastline_to_poly(
            path_to_coastline=config["path_to_coastline"], crs=config["target_crs"]
        )
    except fiona.errors.DriverError:
        ic("warning: coastline data not found")
        coast = None

    for chip_px_width, layer_name in output_options.items():
        start_time = datetime.now()
        ic(start_time)

        # First create the raster
        create_raster_of_chips(
            chip_px_width=chip_px_width,
            pixel_scale=config["pixel_scale"],
            target_crs=config["target_crs"],
            bounding_box=config["bounding_box"],
            base_output_filename=layer_name,
        )

        # Next create the grid as polygons
        chips_gdf = create_grid_of_chips(
            chip_px_width=chip_px_width,
            pixel_scale=config["pixel_scale"],
            target_crs=config["target_crs"],
            bounding_box=config["bounding_box"],
        )

        stage1_time = datetime.now()
        ic(stage1_time - start_time)

        # Add `on_land` column if the coast data is available
        if coast:
            chips_gdf["on_land"] = chips_gdf["geometry"].intersects(coast)

        stage2_time = datetime.now()
        ic(stage2_time - stage1_time)

        # # Now add cross-references to image ids (for different size images)
        # for larger_chip_width, larger_name in output_options.items():
        #     ic(larger_chip_width, chip_px_width)
        #     if larger_chip_width > chip_px_width:
        #         get_larger_chip_id = get_larger_chip_id_func(
        #             larger_chip_width * config["pixel_scale"]
        #         )
        #         chips_gdf[f"{larger_name}_id"] = chips_gdf.apply(
        #             get_larger_chip_id, axis=1
        #         )

        # ic(chips_gdf.head(10))

        # Write to geoparquet file
        chips_gdf.to_parquet(f"{layer_name}.parquet")

        stage3_time = datetime.now()
        ic(stage3_time - stage2_time)

        # Write to GeoPackage DBN (useful for display in GIS)
        # chips_gdf.to_file("chips.gpkg", layer=layer_name)
        stage4_time = datetime.now()
        ic(stage4_time - stage3_time)
        ic(stage4_time - start_time)
