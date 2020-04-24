"""
This module consists of methods to process downloaded GEE data. The starting
point is a json file written out at the end of the downloading step. This 
module cleans, resamples, and reformats the data to make it ready for analysis.

"""

import json
import math
import os

import numpy as np
import pandas as pd

from statsmodels.nonparametric.smoothers_lowess import lowess


def read_json_to_dataframes(filename):
    """
    Read a json file and convert the result to a dict of DataFrame.

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
        raise FileNotFoundError(f'Could not find file "{os.path.abspath(filename)}".')

    # json read
    json_file = open(filename)
    data = json.load(json_file)

    # start with empty output dataframes
    dfs = {}

    # loop over collections and make a DataFrame from the results of each
    for collection_name, coll_results in data.items():

        rows_list = []

        # loop over time series
        for date, time_point in coll_results['time-series-data'].items():

            # check we have data for this time point
            if time_point is None or time_point == {}:
                
                # add Null row if data is missing at this time point    
                rows_list.append({'date': date})

            # if we are looking at veg data, loop over space points
            elif isinstance(list(time_point)[0], dict):
                for space_point in time_point:
                    rows_list.append(space_point)

            # otherwise, just add the row
            else:
                # the key of each object in the time series is the date, and data
                # for this date should be the values. Here we just add the date
                # as a value to enable us to add the whole row in one go later.
                time_point['date'] = date

                rows_list.append(time_point)

        # make a DataFrame and add it to the dict of DataFrames
        df = pd.DataFrame(rows_list)
        df = df.drop(columns=['slope', 'offset', 'mean', 'std'], errors='ignore')
        df = df.sort_values(by='date')
        assert( df.empty == False )
        dfs[collection_name] = df

    return dfs


def make_time_series(dfs):
    """
    Given a dictionary of DataFrames which may contian many rows per time point (corresponding
    to the network centrality values of different sub-locations), collapse this
    into a time series by calculating the mean and std of the different sub-
    locations at each date.

    Parameters
    ----------
    dfs : dict of DataFrame
        Input DataFrame read by `read_json_to_dataframes`.

    Returns
    ----------
    DataFrame
        The time-series results averaged over sub-locations.
    """

    # the time series dataframe
    ts_df = pd.DataFrame(columns=['date'])

    # loop over collections
    for col_name, df in dfs.items():

        #  if vegetation data
        if 'COPERNICUS/S2' in col_name or 'LANDSAT' in col_name:

            # group by date to collapse all network centrality measurements
            groups = df.groupby('date')

            # get summaries
            means = groups.mean()
            stds = groups.std()

            # rename columns
            if 'COPERNICUS/S2' in col_name:
                s = 'S2_'
            elif 'LANDSAT' in col_name:
                s = 'L' + col_name.split('/LC0')[1][0] + '_'
            else: 
                s = col_name + '_'
            means = means.rename(columns={c: s + c + '_mean' for c in means.columns})
            stds = stds.rename(columns={c: s + c + '_std' for c in stds.columns})

            # merge
            df = pd.merge(means, stds, on='date', how='inner')
            ts_df = pd.merge_ordered(ts_df, df, on='date', how='outer')

        # add climate data if availible
        elif 'ECMWF/ERA5/MONTHLY' == col_name:
            df = df.set_index('date')
            ts_df = pd.merge_ordered(ts_df, df, on='date', how='outer')

    # remove unneeded columns
    ts_df = ts_df.loc[:,~ts_df.columns.str.contains('latitude_std', case=False)]     
    ts_df = ts_df.loc[:,~ts_df.columns.str.contains('longitude_std', case=False)]

    assert( ts_df.empty == False )
    return ts_df


def resample_time_series(df, col_name="offset50", period="D"):
    """
    Resample and interpolate a time series dataframe so we have one row
    per day (useful for FFT)

    Parameters
    ----------
    df: DataFrame
        Dataframe with date as index
    col_name: string,
        Identifying the column we will pull out
    period: string
        Period for resampling
    Returns
    -------
    new_series: pandas Series with datetime index, and one column, one row per day
    """
    series = df[col_name]
    # just in case the index isn't already datetime type
    series.index = pd.to_datetime(series.index)

    # resample to get one row per day
    rseries = series.resample(period).mean()
    new_series = rseries.interpolate()

    return new_series


def drop_veg_outliers(dfs, column='offset50', sigmas=3.0):
    """
    Loop over vegetation DataFrames and drop points in the
    time series that a significantly far away from the mean
    of the time series. Such points are assumed to be unphysical.

    Parameters
    ----------
    dfs : dict of DataFrame
        Time series data for multiple sub-image locations.
    column : str
        Name of the column to drop outliers on.
    sigmas : float
        Number of standard deviations a data point has to be
        from the mean to be labelled as an outlier and dropped.

    Returns
    ----------
    dict of DataFrame
        Time series data for multiple sub-image locations with
        some values in `column` potentially set to NaN.
    """

    # set to None data points that are far from the mean, these are
    # assumed to be unphysical

    # loop over collections
    for col_name, veg_df in dfs.items():

        #  if vegetation data
        if 'COPERNICUS/S2' in col_name or 'LANDSAT' in col_name:

            # group by (lat, long)
            d = {}
            for name, group in veg_df.groupby(['latitude', 'longitude']):
                d[name] = group

            # for each sub-image
            for key, df_ in d.items():
                # calcualte residuals to the mean
                res = (df_[column] - df_[column].mean()).abs()

                # determine which are outliers
                outlier = res > df_[column].std() * sigmas

                # set to None
                df_.loc[outlier, column] = None

                # replace the df
                d[key] = df_

            # reconstruct the DataFrame
            df = list(d.values())[0]
            for df_ in list(d.values())[1:]:
                df = df.append(df_)

            # replace value in df
            dfs[col_name] = df

    return dfs


def smooth_veg_data(dfs, column='offset50', n=4):
    """
    Loop over vegetation DataFrames and perform LOESS smoothing
    on the time series of each sub-image.

    Parameters
    ----------
    dfs : dict of DataFrame
        Time series data for multiple sub-image locations.
    column : str
        Name of the column to drop outliers and smooth.
    n : int
        Number of neighbouring point to use in smoothing
    Returns
    ----------
    dict of DataFrame
        Time series data for multiple sub-image locations with
        new column for smoothed data and ci.
    """

    # create a new dataframe to avoid overwriting input
    dfs = dfs.copy()

    # loop over collections
    for col_name, df in dfs.items():

        #  if vegetation data
        if 'COPERNICUS/S2' in col_name or 'LANDSAT' in col_name:
            # remove outliers and smooth
            df = smooth_all_sub_images(df, column=column, n=n)

            # calculate ci
            #df = get_confidence_intervals(df, column=column)

            # replace DataFrame
            dfs[col_name] = df

    return dfs


def smooth_subimage(df, column='offset50', n=4, it=3):
    """
    Perform LOWESS (Locally Weighted Scatterplot Smoothing) on the time
    series of a single sub-image.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame containing the time series for a single
        sub-image.
    column : string, optional
        Name of the column in df to smooth.
    n : int, optional
        Size of smoothing window.
    it : int, optional
        Number of iterations of LOESS smoothing to perform.

    Returns
    ----------
    DataFrame
        The time-series DataFrame with a new column containing the
        smoothed results.
    """

    # add a new column of datetime objects
    df['datetime'] = pd.to_datetime(df['date'], format='%Y/%m/%d')

    # extract data
    xs = df['datetime']
    ys = df[column]

    # num_days_per_timepoint = (xs.iloc[1] - xs.iloc[0]).days
    frac_data = min(n / len(ys), 1.0)

    # perform smoothing
    smoothed_y = lowess(ys, xs, is_sorted=True, return_sorted=False, frac=frac_data, it=it)

    # add to df
    df[column + '_smooth'] = smoothed_y

    return df


def smooth_all_sub_images(df, column='offset50', n=4, it=3):
    """
    Perform LOWESS (Locally Weighted Scatterplot Smoothing) on the time
    series of a set of sub-images.

    Parameters
    ----------
    df : DataFrame
        DataFrame containing time series results for all sub-images,
        with multiple rows per time point and (lat,long) point.
    column : string, optional
        Name of the column in df to smooth.
    n : int, optional
        Size of smoothing window.
    it : int, optional
        Number of iterations of LOESS smoothing to perform.

    Returns
    ----------
    Dataframe
        DataFrame of results with a new column containing a
        LOESS smoothed version of the column `column`.
    """

    # group by (lat, long)
    d = {}
    for name, group in df.groupby(['latitude', 'longitude']):
        d[name] = group

    # for each sub-image
    for key, df_ in d.items():
        # perform smoothing
        d[key] = smooth_subimage(df_, column=column, n=n, it=it)

    # reconstruct the DataFrame
    df = list(d.values())[0]
    for df_ in list(d.values())[1:]:
        df = df.append(df_)

    return df


def remove_seasonality(df, lag, period='M'):
    """
    Loop over time series DataFrames and remove
    time series seasonality.

    Parameters
    ----------
    df : DataFrame
        Time series data for multiple sub-image locations.
    lag : float
        Periodicity to remove

    period: string
        Type of periodicitty (day, month, year)

    Returns
    ----------
    dict of DataFrame
        Time series data for multiple sub-image with
        seasonality removed
    """

    # set to None data points that are far from the mean, these are
    # assumed to be unphysical

    # loop over collections

    df_resampled = pd.DataFrame()

    for col in df.columns:

        if col == 'latitude' or col == 'longitude':
            df_resampled[col] = df[col].iloc[0]
            continue

        if col == 'date' or col == 'datetime':
            df_resampled[col] = df_resampled.index
            continue

        if col == 'feature_vec':
            continue

        series_resampled = resample_time_series(df, col, period)

        df_resampled[col] = series_resampled.diff(lag)

    df_resampled.dropna(inplace=True)

    return df_resampled


def store_feature_vectors(dfs, output_dir):
    """
    Write out all feature vector information to a csv file, to be read
    later by the feature vector plotting script.

    Parameters
    ----------
    dfs : dict of DataFrame
        Time series data for multiple sub-image locations.
    output_dir : str
        Path to directory to save the csv.
    """

    # loop over collections
    for col_name, veg_df in dfs.items():

        #  if vegetation data
        if 'COPERNICUS/S2' in col_name or 'LANDSAT' in col_name:

            # check the feature vectors are availible
            if 'feature_vec' not in veg_df.columns:
                print('Could not find feature vectors.')
                continue
            
            # sort by date
            veg_df = veg_df.sort_values(by='date')

            # create a df to store feature vectors
            df = pd.DataFrame()

            # add feature vectors to dataframe
            df = pd.DataFrame(value for value in veg_df.feature_vec)

            # rename percentile columns
            df = df.rename(columns={n: f'{(n+1)*5}th_percentile' for n in df.columns})

            # reindex
            df.index = veg_df.index

            # add information
            df.insert(0, 'date', veg_df['date'])
            df.insert(1, 'latitude', veg_df['latitude'])
            df.insert(2, 'longitude', veg_df['longitude'])

            # save csv
            if col_name == 'COPERNICUS/S2':
                s = 'S2'
            elif 'LANDSAT' in col_name:
                s = 'L' + col_name.split('/LC0')[1][0] + '_'
            else:
                s = col_name
            
            filename = os.path.join(output_dir, s+'_feature_vectors.csv')
            df.to_csv(filename, index=False)
            

def preprocess_data(input_dir, drop_outliers=True, fill_missing=True, resample=True, smoothing=True):
    """
    This function reads and process data downloaded by GEE. Processing
    can be configured by the function arguments. Processed data is written
    to csv.

    Parameters
    ----------
    input_dir : str
        Path to the directory created during a GEE download job.
    drop_outliers : bool, optional
        Remove outliers in sub-image time series.
    fill_missing : bool, optional
        Fill missing points in the time series.
    resample : bool, optional
        Resample the time series using linear interpolation.
    smooth : bool, optional
        Smooth the time series using LOESS smoothing.

    Returns
    ----------
    str
        Path to the csv file containing processed data.   
    """

    # put output plots in the results dir
    output_dir = os.path.join(input_dir, 'processed_data')

    # check input file exists
    json_summary_path = os.path.join(input_dir, 'results_summary.json')
    if not os.path.exists(json_summary_path):
        raise FileNotFoundError(f'Could not find file "{os.path.abspath(json_summary_path)}".')

    # make output subdir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # read all json files in the directory and produce a dataframe
    print(f"Reading results from '{os.path.abspath(json_summary_path)}'...")

    # read json file to dataframes
    dfs = read_json_to_dataframes(json_summary_path)

    print('Preprocessing data...')

    # remove outliers from the time series
    if drop_outliers:
        print('- Dropping vegetation outliers...')
        dfs = drop_veg_outliers(dfs, sigmas=3)

    # LOESS smoothing on sub-image time series
    if smoothing:
        print('- Smoothing vegetation time series...')
        dfs = smooth_veg_data(dfs, n=4)

    # store feature vectors
    print('Saving feature vectors...')
    store_feature_vectors(dfs, output_dir)
    
    # average over sub-images
    ts_df = make_time_series(dfs)

    # save as csv
    ts_filename = os.path.join(output_dir, 'time_series.csv')
    print(f'Saving time series to "{ts_filename}".')
    ts_df.to_csv(ts_filename, index=False)
    
    print('Data preprocessing complete.\n')
    
    return ts_filename, dfs # for now return `dfs` for compatibility 
