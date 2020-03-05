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
    """
    Read a json file and convert the result to a Geopandas DataFrame.

    Parameters
    ----------
    filename : str
        Full path to input json file.

    Returns
    ----------
    DataFrame
        The saved results in a DataFrame format.
    """
    # check file exists
    if not os.path.exists(filename):
        raise FileNotFoundError

    # json read
    data = None
    with open(filename) as json_file:
        data = json.load(json_file)

    # start with empty output dataframes
    veg_df = pd.DataFrame(columns=['date', 'lat', 'long'])
    weather_df = pd.DataFrame(columns=['date'])
    
    # dataframe index
    i = 0

    # first loop over collections and put vegetation results into one 
    # dataframe
    for collection_name, coll_results in data.items():

        # skip non-vegetation data
        if coll_results['type'] != 'vegetation':
            continue

        # loop over time series
        for time_point in coll_results['time-series-data'].values():

            # check we have data for this time point
            if time_point is None:
                continue
            
            # for each space point
            for space_point in time_point.values():

                # get coordinates
                date = space_point['date']
                lat = space_point['latitude']
                long = space_point['longitude']

                # find other indices in the dataframe which match the date and coordinates
                match_criteria = (veg_df['date'] == date) & (veg_df['lat'] == lat) & (veg_df['long'] == long)
                matched_indices = veg_df.index[match_criteria].tolist()

                # if there is no matching entry
                if len(matched_indices) == 0:

                    # add a new entry to the dataframe
                    veg_df.loc[i, 'date'] = space_point['date']
                    veg_df.loc[i, 'lat'] = space_point['latitude']
                    veg_df.loc[i, 'long'] = space_point['longitude']
                    veg_df.loc[i, f'{collection_name}_offset50'] = space_point['offset50']

                    # increment dataframe index
                    i += 1

                # if we find a row that matches the date and coordinates
                elif len(matched_indices) == 1:

                    # get the index of the matched row
                    index = matched_indices[0]

                    # add information in a new column
                    veg_df.loc[index, f'{collection_name}_offset50'] = space_point['offset50']

                else:
                    raise RuntimeError('Error when building DataFrame, check input json.')

    # next, loop again and put weather data into another dataframe
    # reset dataframe index
    i = 0

    # loop over collections
    for collection_name, coll_results in data.items():

        # skip vegetation data
        if coll_results['type'] == 'vegetation':
            continue
        
        # loop over time series
        for date, values in coll_results['time-series-data'].items():

            # check we have data
            if values is None:
                continue
            
            # check if this we already have a row with this date
            matched_indices = weather_df.index[(weather_df['date'] == date)]

            # if there is no matching entry
            if len(matched_indices) == 0:

                # loop over weather data and add to the same date
                for metric, value in values.items():
                    weather_df.loc[i, 'date'] = date
                    weather_df.loc[i, metric] = value
                
                i += 1

            # if we find a row that matches the date and coordinates
            elif len(matched_indices) == 1:

                # get the index of the matched row
                index = matched_indices[0]
                
                # loop over weather data and add to the same date
                for metric, value in values.items():

                    # add information in a new column
                    weather_df.loc[index, metric] = value

            else:
                raise RuntimeError('Error when building DataFrame, check input json.')

    # combine dataframes in a missing value friendly way
    df = pd.merge(veg_df, weather_df, on='date', how='outer')

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




#plot_time_series()

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




