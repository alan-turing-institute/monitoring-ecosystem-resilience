import json
import pandas as pd
import os
from os.path import isfile, join
import datetime

import geopandas as gpd
from shapely.geometry import Point
import matplotlib
#matplotlib.use('PS')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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




def variable_read_json_to_dataframe(filename):
    """
    Read a json file and convert the result to Geopandas DataFrame(s).

    Parameters
    ----------
    filename : str
        Full path to input json file.

    Returns
    ----------
    dict
        A dict of the saved results in a DataFrame format. Keys are 
        names of collections and the values are DataFrame of results
        for that collection.
    """
    # check file exists
    if not os.path.exists(filename):
        raise FileNotFoundError

    # json read
    json_file = open(filename)
    data = json.load(json_file)

    # start with empty output dataframes
    dfs = {}

    # loop over collections and make a DataFrame from the results of each
    for collection_name, coll_results in data.items():

        df = pd.DataFrame()
        rows_list = []

        # loop over time series
        for date, time_point in coll_results['time-series-data'].items():\

            # check we have data for this time point
            if time_point is None  or time_point == {}:
                continue

            # if we are looking at veg data, loop over space points
            if isinstance(list(time_point.values())[0], dict):
                for space_point in time_point.values():
                    print(space_point)
                    rows_list.append(space_point)
            
            # otherwise, just add the row
            else:
                # the key of each object in the time series is the date, and data
                # for this date should be the values. Here we just add the date 
                # as a value to enable us to add the whole row in one go later.
                time_point['date'] = date

                rows_list.append(time_point)
        
        print(rows_list)
        # make a DataFrame and add it to the dict of DataFrames
        df = pd.DataFrame(rows_list)
        dfs[collection_name] = df

    return dfs


def convert_to_geopandas(df):
    """
    Given a pandas DatFrame with `lat` and `long` columns, convert 
    to geopandas DataFrame.

    Parameters
    ----------
    df : DataFrame
        Pandas DatFrame with `lat` and `long` columns.

    Returns
    ----------
    geopandas DataFrame
    """
    df['geometry'] = [Point(xy) for xy in zip(df.lat, df.long)]
    crs = {'init': 'epsg:4326'}
    df = gpd.GeoDataFrame(df, crs=crs, geometry=df['geometry'])


def make_time_series(dfs):
    """
    Given a DataFrame which may contian many rows per time point (corresponding
    to the network centrality values of different sub-locations), collapse this
    into a time series by calculating the mean and std of the different sub-
    locations at each date.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame read by `read_json_to_dataframe`.

    Returns
    ----------
    DataFrame
        The time-series results averaged over sub-locations.
    """

    # loop over collections
    for col_name, df in dfs.items():
        
        # if vegetation data
        if col_name == 'COPERNICUS/S2' or  'LANDSAT' in col_name:

            # group by date to collapse all network centrality measurements
            groups = df.groupby('date')

            # get summaries
            means = groups.mean()
            stds = groups.std()

            # rename columns
            stds = stds.rename(columns={'offset50': 'offset50_std'})

            # merge
            stds = stds[['offset50_std']]
            df = pd.merge(means, stds, on='date', how='inner')
            dfs[col_name] = df

    return dfs


def plot_time_series(dfs, output_dir):
    #
    """
    Given a DataFrame where each row corresponds to a different time point
    (constructed with `make_time_series`) plot the time series.

    Parameters
    ----------
    df : DataFrame
         The time-series results averaged over sub-locations.
    """

    # auxiliary function to help with plotting
    def make_patch_spines_invisible(ax):
        ax.set_frame_on(True)
        ax.patch.set_visible(False)
        for sp in ax.spines.values():
            sp.set_visible(False)

    s2 = 'COPERNICUS/S2'
    l8 = 'LANDSAT/LC08/C01/T1_SR'

    # prepare data
    cop_means = dfs[s2]['offset50']
    cop_stds = dfs[s2]['offset50_std']
    cop_dates = dfs[s2].index
    cop_xs = [datetime.datetime.strptime(d,'%Y-%m-%d').date() for d in cop_dates]

    l8_means = dfs[l8]['offset50']
    l8_stds = dfs[l8]['offset50_std']
    l8_dates = dfs[l8].index
    l8_xs = [datetime.datetime.strptime(d,'%Y-%m-%d').date() for d in l8_dates]

    precip = dfs['ECMWF/ERA5/MONTHLY']['total_precipitation'] * 1000 # convert to mm
    temp = dfs['ECMWF/ERA5/MONTHLY']['mean_2m_air_temperature'] - 273.15 # convert to Celcius
    weather_dates = dfs['ECMWF/ERA5/MONTHLY']['date']
    w_xs = [datetime.datetime.strptime(d,'%Y-%m-%d').date() for d in weather_dates]

    # setup plot
    fig, ax1 = plt.subplots(figsize=(13,5))
    fig.subplots_adjust(right=0.9)
    
    # set up x axis to handle dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    ax1.set_xlabel('Time')

    # add copernicus
    color = 'tab:green'
    ax1.set_ylabel('Copernicus Offset50', color=color)
    ax1.plot(cop_xs, cop_means, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    plt.fill_between(cop_xs, cop_means-cop_stds, cop_means+cop_stds, 
                     facecolor='green', alpha=0.1)

    # add precip
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('Precipitation [??]', color=color)  # we already handled the x-label with ax1
    ax2.plot(w_xs, precip, color=color, alpha=0.5)
    ax2.tick_params(axis='y', labelcolor=color)

    # add temp
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("axes", 1.075))
    make_patch_spines_invisible(ax3)
    ax3.spines["right"].set_visible(True)
    color = 'tab:red'
    ax3.set_ylabel('Mean Temperature [$^\circ$C]', color=color)  # we already handled the x-label with ax1
    ax3.plot(w_xs, temp, color=color, alpha=0.2)
    ax3.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    # save the plot
    output_filename = 'time-series.png'
    plt.savefig(os.path.join(output_dir, output_filename), dpi=100)

    # add l8
    ax4 = ax1.twinx()
    ax4.spines["left"].set_position(("axes", -0.1))
    make_patch_spines_invisible(ax4)
    ax4.spines["left"].set_visible(True)
    color = 'tab:purple'
    ax4.set_ylabel('landsat', color=color)  # we already handled the x-label with ax1
    ax4.yaxis.tick_left()
    ax4.plot(l8_xs, l8_means, color=color)
    ax4.tick_params(axis='y', labelcolor=color)
    plt.fill_between(l8_xs, l8_means-l8_stds, l8_means+l8_stds, 
                     facecolor='purple', alpha=0.05)

    # save the plot
    output_filename = 'time-series-full.png'
    plt.savefig(os.path.join(output_dir, output_filename), dpi=100)

    
    # ------------------------------------------------
    # setup plot
    fig, ax1 = plt.subplots(figsize=(13,5))
    fig.subplots_adjust(right=0.9)
    
    # set up x axis to handle dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    ax1.set_xlabel('Time')

    # add copernicus
    color = 'tab:green'
    ax1.set_ylabel('Copernicus Offset50', color=color)
    ax1.plot(cop_xs, cop_means, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    plt.fill_between(cop_xs, cop_means-cop_stds, cop_means+cop_stds, 
                     facecolor='green', alpha=0.2)

    # add l8
    ax4 = ax1.twinx()
    color = 'tab:purple'
    ax4.set_ylabel('landsat', color=color)  # we already handled the x-label with ax1
    #ax4.yaxis.tick_left()
    ax4.plot(l8_xs, l8_means, color=color)
    ax4.tick_params(axis='y', labelcolor=color)
    plt.fill_between(l8_xs, l8_means-l8_stds, l8_means+l8_stds, 
                     facecolor='purple', alpha=0.2)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    # save the plot
    output_filename = 'time-series-offsets-only.png'
    plt.savefig(os.path.join(output_dir, output_filename), dpi=100)




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




