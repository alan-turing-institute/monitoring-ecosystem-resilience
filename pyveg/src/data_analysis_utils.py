import json
import pandas as pd
import os
from os.path import isfile, join

import geopandas as gpd
from shapely.geometry import Point
import matplotlib
#matplotlib.use('PS')
import matplotlib.pyplot as plt
import numpy as np



def read_json_to_dataframe(filename):

    #filename = '/Users/svanstroud/work/ds4sd/monitoring-ecosystem-resilience/output/RUN1__2020-03-04_14-38-04/results_summary.json'
    
    # output dataframe
    df = pd.DataFrame(columns=['date', 'lat', 'long'])
    
    # json read
    data = None
    with open(filename) as json_file:
        data = json.load(json_file)

    # index
    i = 0

    # loop over collections
    for collection_name, coll_results in data.items():

        # for vegetation
        if coll_results['type'] == 'vegetation':

            # loop over time series
            for time_point in coll_results['time-series-data'].values():

                # check we have data
                if time_point is None:
                    continue
                
                # for each space point
                for space_point in time_point.values():
                    date = space_point['date']
                    lat = space_point['latitude']
                    long = space_point['longitude']

                    matched_indices = df.index[(df['date'] == date) & (df['lat'] == lat) & (df['long'] == long)].tolist()

                    if len(matched_indices) == 0:
                        df.loc[i, 'date'] = space_point['date']
                        df.loc[i, 'lat'] = space_point['latitude']
                        df.loc[i, 'long'] = space_point['longitude']
                        df.loc[i, f'{collection_name}_offset50'] = space_point['offset50']
                        i += 1

                    elif len(matched_indices) == 1:
                        index = matched_indices[0]
                        df.loc[index, f'{collection_name}_offset50'] = space_point['offset50']

                    else:
                        raise RuntimeError

        # for vegetation
        elif coll_results['type'] == 'weather':

            # loop over time series
            for date, values in coll_results['time-series-data'].items():

                # check we have data
                if values is None:
                    continue

                matched_indices = df.index[(df['date'] == date)]

                for metric, value in values.items():
                    df.loc[matched_indices, metric] = value
    
    # turn lat, long into geopandas
    df['geometry'] = [Point(xy) for xy in zip(df.lat, df.long)]
    crs = {'init': 'epsg:4326'}
    data_geo_pd = gpd.GeoDataFrame(df, crs=crs, geometry=df['geometry'])

    return data_geo_pd


def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


def make_time_series():


    filename = '/Users/svanstroud/work/ds4sd/monitoring-ecosystem-resilience/output/RUN1__2020-03-04_14-38-04/results_summary.json'

    df = read_json_to_dataframe(filename)

    df = df.groupby('date').mean()

    return df


def plot_time_series():
    #
    """
    turn into two functions
    one just process the dataframe into a time series (may be still df)
    the other does the plotting
    """


    filename = '/Users/svanstroud/work/ds4sd/monitoring-ecosystem-resilience/pyveg/output/RUN1__2020-03-04_15-41-47/results_summary.json'

    df = read_json_to_dataframe(filename)

    df = df.groupby('date').mean()

    xs = df.index
    ys1 = df['COPERNICUS/S2_offset50']
    ys2 = df['total_precipitation']
    ys3 = df['mean_2m_air_temperature'] - 273.15
    ys4 = df['LANDSAT/LC08/C01/T1_SR_offset50']




    fig, ax1 = plt.subplots()
    fig.subplots_adjust(right=0.75)

    color = 'tab:green'
    ax1.set_xlabel('date')
    ax1.set_ylabel('COPERNICUS', color=color)
    ax1.plot(xs, ys1, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('precipitation', color=color)  # we already handled the x-label with ax1
    ax2.plot(xs, ys2, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("axes", 1.2))
    make_patch_spines_invisible(ax3)
    ax3.spines["right"].set_visible(True)

    color = 'tab:red'
    ax3.set_ylabel('temp', color=color)  # we already handled the x-label with ax1
    ax3.plot(xs, ys3, color=color)
    ax3.tick_params(axis='y', labelcolor=color)



    ax4 = ax1.twinx()
    
    ax4.spines["left"].set_position(("axes", -0.2))
    make_patch_spines_invisible(ax4)
    ax4.spines["left"].set_visible(True)

    color = 'tab:purple'
    ax4.set_ylabel('landsat', color=color)  # we already handled the x-label with ax1
    ax4.yaxis.tick_left()
    ax4.plot(xs, ys4, color=color)
    ax4.tick_params(axis='y', labelcolor=color)
    


    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()




plot_time_series()

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




