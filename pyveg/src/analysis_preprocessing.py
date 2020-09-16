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
import ewstools

from statsmodels.nonparametric.smoothers_lowess import lowess

from pyveg.src.data_analysis_utils import write_to_json

from pyveg.src.date_utils import get_time_diff

try:
    from pyveg.src import azure_utils
except:
    print("Unable to import azure_utils")


def read_results_summary(input_location,
                         input_filename="results_summary.json",
                         input_location_type="local"):
    """
    Read the results_summary.json, either from local storage or from Azure blob storage.

    Parameters
    ==========
    input_location: str, directory or container with results_summary.json in
    input_filename: str, name of json file, default is "results_summary.json"
    input_location_type: str: 'local' or 'azure'

    Returns
    =======
    json_data: dict, the contents of results_summary.json
    """

    if input_location_type == "local":
        json_filepath = os.path.join(input_location, input_filename)
        if not os.path.exists(json_filepath):
            raise FileNotFoundError("Unable to find {}".format(json_filepath))
        json_data = json.load(open(json_filepath))
        return json_data
    elif input_location_type == "azure":
        subdirs = azure_utils.list_directory(input_location, input_location)
        print("Found subdirs {}".format(subdirs))
        for subdir in subdirs:
            print("looking at subdir {}".format(subdir))
            if "combine" in subdir:
                files = azure_utils.list_directory(input_location+"/"+subdir,
                                                   input_location)
                if input_filename in files:
                    return azure_utils.read_json(input_location+"/"+subdir+"/"+input_filename,
                                                 input_location)
                else:
                    raise RuntimeError("No {} found in {}".format(input_filename, subdir))
        return {}
    else:
        raise RuntimeError("input_location_type needs to be either 'local' or 'azure'")



def read_label_json(input_label_json):
    """
    Read JSON output of ImageLabeller, and return a list of coords to be masked out

    Parameters
    ----------
    input_label_json : str, path to json file

    Returns
    ----------
    mask: list
        A list of coordinate tuples (long,lat) to be masked out.
    """

    if not input_label_json:
        return []
    if not os.path.exists(input_label_json):
        print("WARNING! Could not find file {} - will not mask out any sub-images".format(input_label_json))
        return []
    df = pd.DataFrame.from_records(json.load(open(input_label_json)))
    top_labels = df.groupby(["longitude","latitude"]).category.max().to_dict()
    coords_to_mask = [k for k,v in top_labels.items() if v=="Not patterned vegetation"]
    return coords_to_mask


def mask_space_point(space_point, mask_list):
    """
    See if a space point is in a list of coords that are to be masked (because
    they're not patterned vegetation.

    Parameters
    ==========
    space_point: dict, include keys "latitude" and "longitude"
    mask_list: list of tuples (long,lat)

    Returns
    =======
    True if space point is to be masked out, False otherwise
    """
    if (not mask_list) or len(mask_list)==0 :
        return False
    if not ("latitude" in space_point.keys() and "longitude" in space_point.keys()):
        return False
    for coords in mask_list:
        # comparing floats - need to be careful.  Allow +/- 0.002 precision
        # in case of rounding
        if abs(space_point["latitude"] - coords[1]) < 0.02 \
           and abs(space_point["longitude" ] - coords[0]) < 0.02:
            return True
    return False


def read_json_to_dataframes(data, mask_list=None):
    """
    convert json data to a dict of DataFrame.
    Parameters
    ----------
    data : dict, json data output from run_pyveg_pipeline
    mask_list: list of tuples of floats.  Coordinates (long,lat) to be masked out.

    Returns
    ----------
    dict
        A dict of the saved results in a DataFrame format. Keys are
        names of collections and the values are DataFrame of results
        for that collection.
    """

    # start with empty output dataframes
    dfs = {}

    # loop over collections and make a DataFrame from the results of each
    for collection_name, coll_results in data.items():

        rows_list = []

        if "time-series-data" in coll_results.keys():

            # loop over time series
            for date, time_point in coll_results["time-series-data"].items():

                # check we have data for this time point
                if time_point is None or time_point == {} or time_point == []:
                    # add Null row if data is missing at this time point
                    rows_list.append({"date": date})

                # if we are looking at veg data, loop over space points
                elif isinstance(list(time_point)[0], dict):
                    for space_point in time_point:
                        if not mask_space_point(space_point, mask_list):
                            if 'ndvi' in space_point.keys():
                                space_point['ndvi'] = space_point['ndvi'] * (2.0/255.0) - 1
                            if 'ndvi_veg' in space_point.keys():
                                space_point['ndvi_veg'] = space_point['ndvi_veg'] * (2.0/255.0) - 1
                            rows_list.append(space_point)

                # otherwise, just add the row
                else:
                    # the key of each object in the time series is the date, and data
                    # for this date should be the values. Here we just add the date
                    # as a value to enable us to add the whole row in one go later.
                    time_point["date"] = date
                    rows_list.append(time_point)

            # make a DataFrame and add it to the dict of DataFrames
            df = pd.DataFrame(rows_list)
            df = df.drop(columns=["slope", "offset", "mean", "std"], errors="ignore")
            df = df.sort_values(by="date")
            assert df.empty == False
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
    ts_list: list of DataFrames
        The time-series results averaged over sub-locations.
        First entry will be main dataframe of vegetation and weather.
        Second one (if present) will be historical weather.
    """

    # the time series dataframe
    ts_df = pd.DataFrame(columns=["date"])

    veg_satellite_prefix = ""

    # loop over collections
    for col_name, df in dfs.items():

        #  if vegetation data
        if "COPERNICUS/S2" in col_name or "LANDSAT" in col_name:

            # group by date to collapse all network centrality measurements
            groups = df.groupby("date")

            # get summaries
            means = groups.mean()
            stds = groups.std()

            # rename columns
            if "COPERNICUS/S2" in col_name:
                s = "S2_"
                veg_satellite_prefix = s
            elif "LANDSAT" in col_name:
                s = "L" + col_name.split("/")[1][-1] + "_"
            else:
                s = col_name + "_"
                veg_satellite_prefix = s

            means = means.rename(columns={c: s + c + "_mean" for c in means.columns})
            stds = stds.rename(columns={c: s + c + "_std" for c in stds.columns})

            # merge
            df = pd.merge(means, stds, on="date", how="inner")
            ts_df = pd.merge_ordered(ts_df, df, on="date", how="outer")

        # add climate data if availible
        elif "ECMWF/ERA5/" in col_name:
            df = df.set_index("date")
            ts_df = pd.merge_ordered(ts_df, df, on="date", how="outer")

    # remove unneeded columns
    ts_df = ts_df.loc[:, ~ts_df.columns.str.contains("latitude_std", case=False)]
    ts_df = ts_df.loc[:, ~ts_df.columns.str.contains("longitude_std", case=False)]

    assert ts_df.empty == False
    ts_list = []

    # if there is a big (>10yr) gap between the start of veg and weather time-series,
    # we want to make a separate historic time-series.

    veg_col_name = [col for col in ts_df.columns if col.startswith(veg_satellite_prefix)][0]

    earliest_date = ts_df.iloc[0]["date"]
    earliest_veg_date = ts_df[ts_df[veg_col_name].notna()].iloc[0]["date"]
    if get_time_diff(earliest_veg_date,earliest_date) > 10:
        ts_df_historic = ts_df[ts_df["date"] < earliest_veg_date][["date","mean_2m_air_temperature","total_precipitation"]]
        ts_df = ts_df[ts_df["date"] >= earliest_veg_date]
        ts_list.append(ts_df)
        ts_list.append(ts_df_historic)
    else :
        ts_list.append(ts_df)

    return ts_list


def resample_time_series(series, period="MS"):
    """
    Resample and interpolate a time series dataframe so we have one row
    per time period (useful for FFT)

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
    Series:
        pandas Series with datetime index, and one column, one row per day
    """

    # give the series a date index if the DataFrame is not index by date already
    #   if df.index.name != 'date':
    #    series.index = df.date

    # just in case the index isn't already datetime type
    series.index = pd.to_datetime(series.index)

    # resample to get one row per time period
    rseries = series.resample(period).mean()
    new_series = rseries.interpolate()

    return new_series


def resample_dataframe(df, columns, period="MS"):
    """
    Resample and interpolate a time series dataframe so we have one row
    per time period.

    Parameters
    ----------
    df: DataFrame
        Dataframe with date as index.
    columns: list
        List of column names to resample. Should contain numeric data.
    period: string
        Period for resampling.

    Returns
    -------
    DataFrame:
        DataFrame with resample time series in `columns`.
    """

    # new empty df to deal with length mismatches after resampling
    df_out = pd.DataFrame()

    # for each column to resample
    for column in columns:

        # resample the column
        series = df.set_index("date")[column]
        df_out[column] = resample_time_series(series, period=period)

    # generate a clean index
    df_out = df_out.reset_index()

    return df_out


def resample_data(dfs, period="MS"):
    """
    Resample vegetation and rainfall DataFrames. Vegetation
    DataFrames are resampled at the sub-image level.

    Parameters
    ----------
    dfs : dict of DataFrame
        Time series data for multiple sub-image locations.
    period: string
        Period for resampling.

    Returns
    ----------
    dict of DataFrame
        Resampled data.
    """

    # loop over collections
    for col_name, df in dfs.items():

        #  if vegetation data
        if "COPERNICUS/S2" in col_name or "LANDSAT" in col_name:

            # specify veg columns to resample
            columns = [c for c in df.columns if "offset50" in c]

            # group by (lat, long)
            d = {}
            for name, group in df.groupby(["latitude", "longitude"]):
                d[name] = group

            # for each sub-image
            for key, df_ in d.items():

                # resample
                df_ = resample_dataframe(df_, columns, period=period)

                # replace df
                d[key] = df_

            # reconstruct the DataFrame
            df = list(d.values())[0]
            for df_ in list(d.values())[1:]:
                df = df.append(df_)

            # replace collection
            dfs[col_name] = df

        else:
            # assume ERA5 data
            columns = ["total_precipitation", "mean_2m_air_temperature"]

            # resample
            df_ = resample_dataframe(df_, columns, period=period)

            # replace df
            d[key] = df_

    return dfs


def drop_veg_outliers(dfs, column="offset50", sigmas=3.0):
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

    # loop over collections
    for col_name, veg_df in dfs.items():

        #  if vegetation data
        if "COPERNICUS/S2" in col_name or "LANDSAT" in col_name:

            # group by (lat, long)
            d = {}
            for name, group in veg_df.groupby(["latitude", "longitude"]):
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


def smooth_veg_data(dfs, column="offset50", n=4):
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
        if "COPERNICUS/S2" in col_name or "LANDSAT" in col_name:

            # remove outliers and smooth
            df = smooth_all_sub_images(df, column=column, n=n)

            # calculate ci
            # df = get_confidence_intervals(df, column=column)

            # replace DataFrame
            dfs[col_name] = df

    return dfs


def smooth_subimage(df, column="offset50", n=4, it=3):
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
    df.dropna(inplace=True)

    # add a new column of datetime objects
    df["datetime"] = pd.to_datetime(df["date"], format="%Y/%m/%d")

    # extract data
    xs = df["datetime"]
    ys = df[column]

    # num_days_per_timepoint = (xs.iloc[1] - xs.iloc[0]).days
    frac_data = min(n / len(ys), 1.0)

    # perform smoothing
    smoothed_y = lowess(
        ys, xs, is_sorted=True, return_sorted=False, frac=frac_data, it=it
    )

    # add to df
    df[column + "_smooth"] = smoothed_y
    df[column + "_smooth_res"] = ys - smoothed_y

    return df


def smooth_all_sub_images(df, column="offset50", n=4, it=3):
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
    for name, group in df.groupby(["latitude", "longitude"]):
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
        if "COPERNICUS/S2" in col_name or "LANDSAT" in col_name:

            # check the feature vectors are availible
            if "feature_vec" not in veg_df.columns:
                print("Could not find feature vectors.")
                continue

            # sort by date
            veg_df = veg_df.sort_values(by="date").dropna()

            # create a df to store feature vectors
            df = pd.DataFrame()
            [
                print(value)
                for value in veg_df.feature_vec
                if not isinstance(value, list)
            ]
            # add feature vectors to dataframe
            df = pd.DataFrame(value for value in veg_df.feature_vec)

            # rename percentile columns
            df = df.rename(columns={n: f"{(n+1)*5}th_percentile" for n in df.columns})

            # reindex
            df.index = veg_df.index

            # add information
            df.insert(0, "date", veg_df["date"])
            df.insert(1, "latitude", veg_df["latitude"])
            df.insert(2, "longitude", veg_df["longitude"])

            # save csv
            if col_name == "COPERNICUS/S2":
                s = "S2"
            elif "LANDSAT" in col_name:
                s = "L" + col_name.split("/")[1][-1] + "_"
            else:
                s = col_name

            filename = os.path.join(output_dir, s + "_feature_vectors.csv")
            df.to_csv(filename, index=False)


def fill_veg_gaps(dfs, missing):
    """
    Loop through sub-image time series and replace any gaps with mean
    value of the same month in other years.

    Parameters
    ----------
    dfs : dict of DataFrame
        Time series data for multiple sub-image locations.

    missing : dict of array
        Missing time points where no sub-images were analyse for
        each veg dataframe in `dfs`.
    """

    # loop over collections
    for col_name, veg_df in dfs.items():

        #  if vegetation data
        if "COPERNICUS/S2" in col_name or "LANDSAT" in col_name:

            # group by (lat, long)
            d = {}
            for name, group in veg_df.groupby(["latitude", "longitude"]):
                d[name] = group

            # for each sub-image
            for key, df_ in d.items():

                # get lat, long of this sub-image
                lats = df_.latitude.drop_duplicates().values
                longs = df_.longitude.drop_duplicates().values
                assert len(lats) == 1
                assert len(longs) == 1
                lat = lats[0]
                long = longs[0]

                # construct missing rows
                missing_rows = [pd.Series({"date": date}) for date in missing[col_name]]

                if len(missing_rows) == 0:
                    continue

                # add back in missing values if necessary
                df_ = df_.append(missing_rows, ignore_index=True).sort_values(by="date")

                # make a new 'month' column
                df_["month"] = df_.date.str.split("-").str[1]

                # group by month and get monthly means
                monthly_means = df_.groupby("month").mean().offset50

                # loop through dataframe
                for index, row in df_.iterrows():

                    # fill missing months with mean value
                    if pd.isnull(row.offset50):
                        this_month = row.month
                        df_.loc[index, "offset50"] = monthly_means.loc[this_month]
                        df_.loc[index, "latitude"] = lat
                        df_.loc[index, "longitude"] = long
                        df_.loc[index, "feature_vec"] = np.NaN

                # drop month column and replace old df
                df_ = df_.drop(columns="month")
                d[key] = df_

            # reconstruct the DataFrame
            df = list(d.values())[0]
            for df_ in list(d.values())[1:]:
                df = df.append(df_)

            dfs[col_name] = df

    return dfs


def get_missing_time_points(dfs):
    """
    Find missing time points for each vegetation dataframe in `dfs`,
    and return a dict, with the same key as in `dfs`, but with values
    corresponding to missing dates.

    Parameters
    ----------
    dfs : dict of DataFrame
        Time series data for multiple sub-image locations.

    Returns
    ----------
    dict
        Missing time points for each vegetation df.
    """

    # determine missing vegetation time points
    missing_points = {}

    # loop over collections
    for col_name, veg_df in dfs.items():

        #  if vegetation data
        if "COPERNICUS/S2" in col_name or "LANDSAT" in col_name:

            # get the start of the vegetation time series
            veg_start_date = veg_df.dropna().index[0]

            # remove leading NaNs
            veg_df = veg_df.loc[veg_start_date:]

            # store missing time points
            missing_points[col_name] = veg_df.drop_duplicates(
                subset="date", keep=False
            ).date.values

    return missing_points


def detrend_df(df, period="MS"):
    """
    Remove seasonality from a DataFrame containing the time series
    for a single sub-image.

    Parameters
    ----------
    df : DataFrame
        Time series data for a single sub-image location.
    period : str, optional
    `   Resample time series to this frequency and then infer
        lag to use for deseasonalizing.

    Returns
    ----------
    DataFrame
        Input with seasonality removed from time series columns.
    """

    # infer lag from period, we need at least 2 years for diferenciation to work
    if period == "MS":
        lag = 12
    else:
        raise ValueError('Periods other than "MS" are not well supported yet!')

    # new empty df to deal with length mismatches after resampling
    df_out = pd.DataFrame()

    # resample time series (in case not done already)
    columns = [
        c
        for c in df.columns
        if any([s in c for s in ["offset50", "precipitation", "temperature", "ndvi"]])
    ]

    df_out = resample_dataframe(df, columns, period=period)

    # detrend veg and climate columns
    for col in columns:
        df_out[col] = df_out[col].diff(lag)

    # need to keep this info for smoothing later
    try:
        df_out["latitude"] = df["latitude"].iloc[0]
        df_out["longitude"] = df["longitude"].iloc[0]
    except:
        pass

    return df_out


def detrend_data(dfs, period="MS"):
    """
    Loop over each sub image time series DataFrames and remove
    time series seasonality by subtracting the previous year.
    Remove seasonality from precipitation data in the same way.

    Parameters
    ----------
    dfs : dict of DataFrame
        Time series data for multiple sub-image locations.
    period : str, optional
    `   Resample time series to this frequency and then infer
        lag to use for deseasonalizing.

    Returns
    ----------
    dict of DataFrame
        Time series data for multiple sub-image with
        seasonality removed.

    """

    # don't overwrite input
    dfs = dfs.copy()

    for col_name, df in dfs.items():

        #  if vegetation data
        if "COPERNICUS/S2" in col_name or "LANDSAT" in col_name:

            # group by (lat, long)
            d = {}
            for name, group in df.groupby(["latitude", "longitude"], as_index=False):
                d[name] = group

            # for each sub-image
            for key, df_ in d.items():
                d[key] = detrend_df(df_, period)

            # reconstruct the DataFrame
            df = list(d.values())[0]
            for df_ in list(d.values())[1:]:
                df = df.append(df_)

            df.dropna(inplace=True)

            dfs[col_name] = df

        else:
            # remove seasonality for weather data, this is a simpler time series

            dfs[col_name] = detrend_df(dfs[col_name], period)
            df.dropna(inplace=True)

    return dfs


def preprocess_data(
        input_json,
        output_basedir,
        input_label_json=None,
        drop_outliers=True,
        fill_missing=True,
        resample=True,
        smoothing=True,
        detrend=True,
        n_smooth=4,
        period="MS",
):
    """
    This function reads and process data downloaded by GEE. Processing
    can be configured by the function arguments. Processed data is
    written to csv.

    Parameters
    ----------
    input_json : dict
       JSON data created during a GEE download job.
    output_basedir : str,
       Directory where time-series csv will be put.
    input_label_json: str,
        JSON file that is the output of ImageLabeller.  If specified, will be used
        to "mask off" sub-images that are labelled as "Not patterned vegetation".
    drop_outliers : bool, optional
        Remove outliers in sub-image time series.
    fill_missing : bool, optional
        Fill missing points in the time series.
    resample : bool, optional
        Resample the time series using linear interpolation.
    smoothing : bool, optional
        Smooth the time series using LOESS smoothing.
    detrend : bool, optional
        Remove seasonal component by subtracting previous year.
    n_smooth : int, optional
        Number of time points to use for the smoothing window size.
    period : str, optional
        Pandas DateOffset string describing sampling frequency.

    Returns
    ----------
    output_dir: str
        Path to the csv file containing processed data.
    defs: dict
        Dictionary of dataframes.
    """

    # put output plots in the results dir
    output_dir = os.path.join(output_basedir, "processed_data")


    # make output subdir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # if we're given a json file from ImageLabeller output to mask out non-patterned
    # coordinates, read it here
    if input_label_json:
        mask_list = read_label_json(input_label_json)
    else:
        mask_list=None

    # read dict from json file to dataframes
    dfs = read_json_to_dataframes(input_json, mask_list)

    # keep track of time points where data is missing (by default pandas
    # groupby operations, which is used haveily in this module, drop NaNs)
    missing = get_missing_time_points(dfs)
    missing_json = {k: list(v) for k, v in missing.items()}
    write_to_json(os.path.join(output_dir, "missing_dates.json"), missing_json)

    print("\nPreprocessing data...")
    print("-" * 21)

    # remove outliers from the time series
    if drop_outliers:
        print("- Dropping vegetation outliers...")
        dfs = drop_veg_outliers(dfs, sigmas=3)

    # use the same month in different years to fill gaps
    if fill_missing:
        print("- Fill gaps in sub-image time series...")
        dfs = fill_veg_gaps(dfs, missing)

    # LOESS smoothing on sub-image time series
    if smoothing:
        print("- Smoothing vegetation time series...")
        dfs = smooth_veg_data(dfs, n=n_smooth)

    # store feature vectors before averaging over sub-images
    print("- Saving feature vectors...")
    store_feature_vectors(dfs, output_dir)

    # average over sub-images
    ts_list = make_time_series(dfs)
    ts_df = ts_list[0]
    if len(ts_list) > 1 :
        ts_historic = ts_list[1]
    else :
        ts_historic = pd.DataFrame()

    # resample the averaged time series using linear interpolation
    if resample:
        print("- Resampling time series...")
        columns = [
            c
            for c in ts_df.columns
            if any([s in c for s in ["offset50", "precipitation", "temperature"]])
        ]
        ts_df = resample_dataframe(ts_df, columns, period=period)

    #  save as csv
    ts_filename = os.path.join(output_dir, "time_series.csv")
    print(f'- Saving time series to "{ts_filename}".')
    ts_df.to_csv(ts_filename, index=False)
    if not ts_historic.empty :
        ts_filename = os.path.join(output_dir, "time_series_historic.csv")
        print(f'- Saving time series to "{ts_filename}".')
        ts_historic.to_csv(ts_filename, index=False)

    # additionally save resampled & detrended time series
    # this detrending option (one year seasonality substraction) only works in monthly data that has at least 2 years of data
    if detrend and ts_df.shape[0]>24 and period=='MS':
        print("- Detrending time series...")

        # remove seasonality from sub-image time series
        dfs_detrended = detrend_data(dfs, period=period)

        print("- Smoothing vegetation time series after removing seasonality...")
        dfs_detrended_smooth = smooth_veg_data(dfs_detrended, n=12)

        # combine over sub-images
        ts_df_detrended_smooth = make_time_series(dfs_detrended_smooth)[0]

        # save output
        ts_filename_detrended = os.path.join(output_dir, "time_series_detrended.csv")
        print(f'- Saving detrended time series to "{ts_filename_detrended}".')
        ts_df_detrended_smooth.to_csv(ts_filename_detrended, index=False)

    return output_dir, dfs  #  for now return `dfs` for spatial plot compatibility


def save_ts_summary_stats(ts_dirname, output_dir, metadata):
        """
        Given a time series DataFrames (constructed with `make_time_series`),
        give summary statistics of all the avalaible time series.

        Parameters
        ----------
        ts_dirname : str
              Directory where the time series are saved.

        output_dir : str
            Directory to save the plots in.

        metadata: dict
            Dictionary with metadata from location

        """

        # read processed data

        # get filenames of preprocessed data time series
        ts_filenames = [f for f in os.listdir(ts_dirname) if "time_series" in f]

        # we should get one seasonal time series and a detrended one
        ts_df_detrended = pd.DataFrame()
        ts_df_historic = pd.DataFrame()
        for filename in ts_filenames:
            if "detrended" in filename:
                ts_df_detrended = pd.read_csv(os.path.join(ts_dirname,filename))
            elif "historic" in filename:
                ts_df_historic = pd.read_csv(os.path.join(ts_dirname,filename))
            else:
                ts_df = pd.read_csv(os.path.join(ts_dirname,filename))


        def get_ts_summary_stats(series):
            ''' Function that gets the summary stats of the time series and returns a dictionary'''
            stats_dict = {}

            stats_dict['min'] = series.min()
            stats_dict['max'] = series.max()
            stats_dict['mean'] = series.mean()
            stats_dict['median'] = series.median()
            stats_dict['std'] = series.std()

            return stats_dict

        # calculate summary statistics for each relevant time series
        ts_dict_list = []
        # only look at relevant time series (offset50, ndvi and precipitation)
        if not ts_df_historic.empty :
            column_dict = get_ts_summary_stats(ts_df_historic["total_precipitation"])
            column_dict["ts_id"] = "total_precipitation_historic"
            for key in metadata:
                column_dict[key] = metadata[key]
            ts_dict_list.append(column_dict)


        column_names = [c for c in ts_df.columns if 'offset50_mean' in c or
                        'ndvi_mean' in c or
                        'total_precipitation' in c]


        for column in column_names:

            print(f'Calculating summary stats for "{column}"...')

            column_dict = get_ts_summary_stats(ts_df[column])
            column_dict['ts_id'] = column
            for key in metadata:
                column_dict[key] = metadata[key]

            # We want the AR1 and Standard deviation of the detreded timeseries for the summary stats
            if ts_df_detrended.empty==False:
                ews_dic_veg = ewstools.core.ews_compute(ts_df_detrended[column].dropna(),
                                                        roll_window=0.99 ,
                                                        smooth='Gaussian',
                                                        lag_times=[1],
                                                        ews= ["var", "ac"],
                                                        band_width=6)

                EWSmetrics_df = ews_dic_veg['EWS metrics']
                column_dict["Lag-1 AC (0.99 rolling window)"] = EWSmetrics_df["Lag-1 AC"].iloc[-1]
                column_dict["Variance (0.99 rolling window)"] = EWSmetrics_df["Variance"].iloc[-1]

                ews_dic_veg_50 = ewstools.core.ews_compute(ts_df_detrended[column].dropna(),
                                                        roll_window=0.5,
                                                        smooth='Gaussian',
                                                        lag_times=[1],
                                                        ews=["var", "ac"],
                                                        band_width=6)

                Kendall_tau_50 = ews_dic_veg_50['Kendall tau']
                column_dict["Kendall tau Lag-1 AC (0.5 rolling window)"] = Kendall_tau_50["Lag-1 AC"].iloc[-1]
                column_dict["Kendall tau Variance (0.5 rolling window)"] = Kendall_tau_50["Variance"].iloc[-1]



            ts_dict_list.append(column_dict)


        string_name = "_".join([str(metadata[key]) for key in metadata])
        string_name = string_name.replace("/","_")
        # turn the list of dictionary to dataframe and save it
        ts_df_summary = pd.DataFrame(ts_dict_list)

        #save both name specific and generic (might be useful inside the analysis later)
        ts_df_summary.to_csv(os.path.join(output_dir, "time_series_summary_stats.csv"))
        ts_df_summary.to_csv(os.path.join(output_dir, "time_series_summary_stats_"+string_name+".csv"))
