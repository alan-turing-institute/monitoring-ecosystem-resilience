import json
import pandas as pd
import os
from os.path import isfile, join

import geopandas as gpd
from shapely.geometry import Point
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt
import numpy as np

def process_json_metrics_to_dataframe(directory_path):

    """
    Read JSON files produced by GEE and get output metrics into a dataframe
    """

    # find all json files in a given directory
    list_json_files = [f for f in os.listdir(directory_path) if (isfile(join(directory_path, f)) and f.endswith(".json"))]


    # in case the directory do not contain json files
    if len(list_json_files) == 0:
        raise RuntimeError("No json files in " + directory_path )

    else:

        metrics = []

        #loop by each json file and get metrics
        for file_json in list_json_files:
            try:
                with open(os.path.join(directory_path , file_json)) as f:
                    d = json.load(f)
                    metrics.append(d)
            except:
                print('Issue with file', os.path.join(directory_path , file_json))
                continue

        if len(metrics) != 0:
            # turn metrics into dataframe
            data_df = pd.DataFrame.from_dict(metrics)

            data_df["date"]= data_df["date"].astype('datetime64[ns]')

            # Add a year variable

            data_df.sort_values(by=['date'], inplace=True, ascending=True)

        else:
            raise RuntimeError('Faulty json files in' + directory_path)


    return data_df


def create_network_figures(data_df, metric, output_dir, output_name):

    """
    From input dataframe with processed network metrics create figure for each date avalaible using geopandas.


    """
    if set(['date','longitude','latitude',metric]).issubset(data_df.columns):


        # turn lat, long into geopandas
        data_df['geometry'] = [Point(xy) for xy in zip(data_df.latitude, data_df.longitude)]

        crs = {'init': 'epsg:4326'}
        data_geo_pd = gpd.GeoDataFrame(data_df, crs=crs, geometry=data_df['geometry'])

        # get min and max values observed in the data to create a range
        vmin = min(data_df[metric])
        vmax = max(data_df[metric])

        # get all dates avalaibles
        list_of_dates = np.unique(data_geo_pd['date'])

        for date in list_of_dates:

            # create figure and axes for Matplotlib
            fig, ax = plt.subplots(1, figsize=(6, 6))

            data_geo_pd[data_geo_pd['date'] == date].plot(marker='o', ax=ax, alpha=.5, markersize=100, column=metric, \
                                                          figsize=(10, 10), linewidth=0.8, edgecolor='0.8', cmap='Reds')

            # ridiculous step
            date_str = pd.to_datetime(str(date)).strftime('%Y-%m-%d')

            # create a date annotation on the figure
            ax.annotate(date_str, xy=(0.15, 0.08), xycoords='figure fraction',
                        horizontalalignment='left', verticalalignment='top',
                        fontsize=25)

            # Create colorbar as a legend
            sm = plt.cm.ScalarMappable(cmap='Reds', norm=plt.Normalize(vmin=vmin, vmax=vmax))
            sm._A = []
            fig.colorbar(sm)

            # create output directoriy
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # this saves the figure as a high-res png in the output path.
            filepath = os.path.join(output_dir, output_name + '_network_values' + date_str + '.png')
            fig.savefig(filepath, dpi=300)
    else:
        raise RuntimeError("Expected variables not present in input dataframe")




