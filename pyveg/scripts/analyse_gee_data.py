#!/usr/bin/env python

"""
Scripts to process the output of the GEE images and json files with network centrality measures.

The json outputs are turned into a dataframe and the values of a particular metric a plotted
as a function of time.

Finally a GIF file is produced with all of the network metric images, as well as the original 10km x 10km dowloaded images.
"""

import argparse
import os

from pyveg.src.data_analysis_utils import (
    variable_read_json_to_dataframe,
    create_lat_long_metric_figures,
    make_time_series,
    convert_to_geopandas,
    coarse_dataframe,
    drop_outliers_and_smooth
)

from pyveg.src.plotting import plot_time_series, plot_smoothed_time_series

from pyveg.src.image_utils import (
    create_gif_from_images
)


def main():
    """
    use command line arguments to choose images.
    """
    parser = argparse.ArgumentParser(description="process json files with network centrality measures from from GEE images")
    parser.add_argument("--input_dir",help="results directory from `download_gee_data` script, containing `results_summary.json`")
    
    print('-'*35)
    print('Running analyse_gee_data.py')
    print('-'*35)

    # parse args
    args = parser.parse_args()
    input_dir = args.input_dir

    # put output plots in the results dir
    output_dir = os.path.join(input_dir, 'analysis')

    # check input file exists
    json_summary_path = os.path.join(input_dir, 'results_summary.json')
    if not os.path.exists(json_summary_path):
        raise FileNotFoundError(f'Could not find file "{json_summary_path}".')

    # make output subdir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # read all json files in the directory and produce a dataframe
    print(f"Reading results from '{json_summary_path}'...")
    dfs = variable_read_json_to_dataframe(json_summary_path)

    # from the dataframe, produce network metric figure for each avalaible date
    print('Creating spatial plots...')
    for collection_name, df in dfs.items():
        if collection_name == 'COPERNICUS/S2' or 'LANDSAT' in collection_name:
            data_df_geo = convert_to_geopandas(df)
            data_df_geo_coarse = coarse_dataframe(data_df_geo, 2)
            create_lat_long_metric_figures(data_df_geo_coarse, 'offset50', output_dir)


    # time series analysis and plotting 
    # ------------------------------------------------
    # convert to time series
    time_series_dfs = make_time_series(dfs)

    # make the time series plot
    print('Plotting time series...')
    plot_time_series(time_series_dfs, output_dir)

    # drop outliers and smooth results
    dfs_smooth = drop_outliers_and_smooth(dfs)

    # make a smoothed time series plot
    print('Plotting smoothed time series...')
    plot_smoothed_time_series(dfs, output_dir)
    # ------------------------------------------------

    print('Done!')



if __name__ == "__main__":
    main()
