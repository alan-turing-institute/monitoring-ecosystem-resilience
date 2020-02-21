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

        data_df['abs_metric'] = data_df[metric]*-1

        # turn lat, long into geopandas
        data_df['geometry'] = [Point(xy) for xy in zip(data_df.latitude, data_df.longitude)]

        crs = {'init': 'epsg:4326'}
        data_geo_pd = gpd.GeoDataFrame(data_df, crs=crs, geometry=data_df['geometry'])

        # get min and max values observed in the data to create a range
        vmin = 0
        vmax = 1000

        # get all dates avalaibles
        list_of_dates = np.unique(data_geo_pd['date'])

        for date in list_of_dates:

            # create figure and axes for Matplotlib
            fig, ax = plt.subplots(1, figsize=(6, 6))

            import matplotlib.cm as cm

            cmap = cm.summer

            data_geo_pd[data_geo_pd['date'] == date].plot(marker='s', ax=ax, alpha=.5, markersize=100, column='abs_metric', \
                                                          figsize=(10, 10), linewidth=0.8, edgecolor='0.8', cmap=cmap)

            # ridiculous step
            date_str = pd.to_datetime(str(date)).strftime('%Y-%m-%d')

            # create a date annotation on the figure
            ax.annotate(date_str, xy=(0.15, 0.08), xycoords='figure fraction',
                        horizontalalignment='left', verticalalignment='top',
                        fontsize=25)

            # Create colorbar as a legend

            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))


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




def create_network_time_series(data_df, metric, output_dir, output_name):


    # get data for each unique pair of lat, long
    unique_coords = data_df[['latitude','longitude',metric,'date']].groupby(['latitude','longitude'])

    fig, ax = plt.subplots(1, figsize=(12, 6))

    count = 0

    # get time series
    for name, group in unique_coords:

        if group.shape[0]>1:

            plt.plot(group['date'], group['offset50'],label = str(round(name[0],2))+","+str(round(name[1],2)))
            count = count +1

        if count > 10:

            plt.title('Network Centrality Measure')
            plt.ylabel('offset50');
            plt.legend();
            ax.tick_params(axis='x', rotation=45)
            plt.ylim((-1200, 0))
            fig.savefig(os.path.join(output_dir,output_name+"_"+str(round(name[0],2))+","+str(round(name[1],2))+"_time_series.png"), dpi=100)
            plt.clf()

            count = 0




def create_general_network_time_series(data_df, metric):


    unique_dates = data_df[['latitude','longitude',metric,'date']].groupby(['date'])[metric].agg(['mean', 'std'])

    fig, ax = plt.subplots(2, figsize=(12, 6))
    ax[0].plot(unique_dates.index, unique_dates['mean'])
    ax[0].set_ylim((-800,-400))
    ax[1].plot(unique_dates.index, unique_dates['std'])
    fig.savefig('test.png', dpi=100)
