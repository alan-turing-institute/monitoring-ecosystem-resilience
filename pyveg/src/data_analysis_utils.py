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
import matplotlib.cm as cm


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


def create_lat_long_metric_figures(data_df, metric,output_dir):

    """
    From input dataframe with processed network metrics create figure for each date avalaible using geopandas.


    """
    if set(['date',metric]).issubset(data_df.columns):

        data_df['abs_metric'] = data_df[metric]*-1

        # get min and max values observed in the data to create a range

        vmin = 0
        vmax = 1000

        # get all dates avalaibles

        list_of_dates = np.unique(data_df['date'])

        for date in list_of_dates:

            print (date)
            try:
                # create figure and axes for Matplotlib
                fig, ax = plt.subplots(1, figsize=(6, 6))


                cmap = cm.summer


                data_df[data_df['date'] == date].plot(marker='s', ax=ax, alpha=.5, markersize=100, column='abs_metric', \
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
                filepath = os.path.join(output_dir, 'network_values' + date_str + '.png')
                fig.savefig(filepath, dpi=300)

            except:
                print ('Something')
    else:
        raise RuntimeError("Expected variables not present in input dataframe")


def coarse_dataframe(data_df, side_square):
    """

    Coarse the granularity of a dataframe by grouping lat,long points that are close to each other in a square of L = size_square
    :param data_df:  Input dataframe
    :param side_square: Side of the square
    :return: a coarser dataframe
    """

    # initialise the categories
    data_df['category'] = -1

    category = 0
    for n in range(data_df.shape[0]):

        # only process lat,long point that do not have a category
        if data_df['category'].iloc[n] == -1:

            # find the closest points and sort
            distances = [(i, pt.distance(data_df['geometry'].iloc[n]),data_df['category'].iloc[n]) for i, pt in enumerate(data_df['geometry'])]
            distances.sort(key=lambda x: x[1])

            # only use point that do not have a category assigned
            distances = [i for i in distances if i[2] == -1]

            # find the max distance observed in the given squared
            max_dist = np.max(np.unique([dist[1] for dist in distances])[:2 * side_square])

            # find the indexes of point in the dataframe that are within that max distance
            indexes = [dist[0] for dist in distances if (dist[1] <= max_dist)]

            # assing them all to the same categorty
            data_df['category'].iloc[indexes] = str(category)

            category = category + 1

    # create unique categories per date
    data_df['category'] =  data_df['category'].str.cat(data_df['date'],sep="_")

    data_df2 = data_df.dissolve(by=['category','date'], aggfunc='mean')

    # reassing the date because we are losing it
    date = [i[1] for i in data_df2.index]
    data_df2['date']= date

    return data_df2







if __name__ == "__main__":
    data_df = read_json_to_dataframe('/Users/crangelsmith/PycharmProjects/monitoring-ecosystem-resilience/results_summary')
    #data_df["datetime"] = data_df["date"].astype('datetime64[ns]')
    data_df['lat'] = data_df['lat'].astype('float64', copy=False)
    data_df['long'] = data_df['long'].astype('float64', copy=False)

    #data_df["date"] = data_df["date"].astype('datetime64[ns]')

    data_df = coarse_dataframe(data_df,3)

    print (data_df.columns)
    create_lat_long_metric_figures(data_df, 'LANDSAT/LC08/C01/T1_SR_offset50','.')