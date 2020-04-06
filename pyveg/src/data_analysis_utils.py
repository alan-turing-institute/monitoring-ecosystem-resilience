"""
Data analysis code including functions to read the .json results file, 
and functions analyse and plot the data.
"""

# system imports
import json
import os
from os.path import isfile, join
import math
import datetime

# data
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# plotting
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm

# stats
from scipy.fftpack import fft
from scipy.stats import sem, t
from statsmodels.nonparametric.smoothers_lowess import lowess


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

    # first loop over collections and put vegetation results into one
    # dataframe
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
    df['geometry'] = [Point(xy) for xy in zip(df.latitude, df.longitude)]
    crs = {'init': 'epsg:4326'}
    df = gpd.GeoDataFrame(df, crs=crs, geometry=df['geometry'])


    return df


def make_time_series(dfs):
    """
    Given a dictionary of DataFrames which may contian many rows per time point (corresponding
    to the network centrality values of different sub-locations), collapse this
    into a time series by calculating the mean and std of the different sub-
    locations at each date.

    Parameters
    ----------
    dfs : dict of DataFrame
        Input DataFrame read by `variable_read_json_to_dataframe`.

    Returns
    ----------
    DataFrame
        The time-series results averaged over sub-locations.
    """

    # loop over collections
    for col_name, df in dfs.items():

        # if vegetation data
        if 'COPERNICUS/S2' in col_name or 'LANDSAT' in col_name:

            # group by date to collapse all network centrality measurements
            groups = df.groupby('date')

            # get summaries
            means = groups.mean()
            stds = groups.std()

            # rename columns
            means = means.rename(columns={s: s+'_mean' for s in means.columns})
            stds = stds.rename(columns={s: s+'_std' for s in stds.columns})

            # merge
            df = pd.merge(means, stds, on='date', how='inner')
            dfs[col_name] = df

        else: # assume weather data
            df = df.set_index('date')
            dfs[col_name] = df

    return dfs


def get_veg_time_series(dfs):
    df_out = pd.DataFrame(columns=['date'])
    for collection_name, df in dfs.items():
        if collection_name == 'COPERNICUS/S2' or 'LANDSAT' in collection_name:
            df = df[[col for col in df.columns if 'offset50' in col]]
            df_out = pd.merge(df, df_out, on='date', how='outer')
    return df_out


def get_weather_time_series(dfs):

    df_ERA5 = None
    df_NASA = None

    for collection_name, df in dfs.items():
        if collection_name == 'ECMWF/ERA5/MONTHLY':
            df_ERA5 = df
            df_ERA5['total_precipitation'] *= 1e3 # convert to mm
            df_ERA5['mean_2m_air_temperature'] -= 273.15 # convert to Celcius
            df_ERA5 = df_ERA5.rename(columns={'total_precipitation': 'ERA5_precipitation',
                                    'mean_2m_air_temperature': 'ERA5_temperature'})

        elif collection_name == 'NASA/GPM_L3/IMERG_V06':
            df_NASA = df
            df_NASA = df_NASA.rename(columns={'precipitationCal': 'NASA_precipitation'})

    # if we have both satellites
    if df_ERA5 is not None and df_NASA is not None:
        # combine precipitation and get error
        df = pd.merge(df_ERA5, df_NASA, on='date', how='inner')
        df['precipitation_mean'] = df[['ERA5_precipitation', 'NASA_precipitation']].mean(axis=1)
        df['precipitation_std'] = df[['ERA5_precipitation', 'NASA_precipitation']].std(axis=1)
        return df.drop(columns=['ERA5_precipitation', 'NASA_precipitation'])

    # if we only have ERA5
    elif df_ERA5 is not None:
        return df_ERA5

    # if we only have NASA
    elif df_NASA is not None:
        return df_NASA


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
    remove_outliers : bool, optional
        Remove datapoints >3 standard deviations
        from the mean before smoothing.

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

    #num_days_per_timepoint = (xs.iloc[1] - xs.iloc[0]).days
    frac_data = min(n / len(ys), 1.0)

    # perform smoothing
    smoothed_y = lowess(ys, xs, is_sorted=True, return_sorted=False, frac=frac_data, it=it)

    # add to df
    df[column+'_smooth'] = smoothed_y
    
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
    remove_outliers : bool, optional
        Remove datapoints >3 standard deviations
        from the mean before smoothing.

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
    for df in d.values():

        # perform smoothing
        df = smooth_subimage(df, column=column, n=n, it=it)

    # reconstruct the DataFrame
    df = list(d.values())[0]
    for df_ in list(d.values())[1:]:
        df = df.append(df_)
        
    return df

def calculate_ci(data, ci_level=0.99):
    """
    Calculate the confidence interval on the mean for a set of data.

    Parameters
    ----------
    data : Series
        Series of data to calculate the confidence interval of the mean.
    ci_level : float, optional
        Size of the confidence interval to calculate

    Returns
    ----------
    float
        Confidence interval value where the CI is [mu - h, mu + h],
        where mu is the mean.

    """
    
    # remove NaNs
    ys = data.dropna().values

    # calculate CI
    n = len(ys)
    std_err = sem(ys)
    h = std_err * t.ppf((1 + ci_level) / 2, n - 1)
    
    return h


def get_confidence_intervals(df, column, ci_level=0.99):
    """
    Calculate the confidence interval at each time point of a 
    DataFrame containing data for a large image.

    Parameters
    ----------
    df : DataFrame
        Time series data for multiple sub-image locations.
    column : str
        Name of the column to calculate the CI of.
    ci_level : float, optional
        Size of the confidence interval to calculate

    Returns
    ----------
    DataFrame
        Time series data for multiple sub-image locations with
        added column for the ci.
    """
    
    # group all the data at each date
    d = {}
    for name, group in df.groupby(['date']):
        d[name] = group
    
    # for each timepoint, calculate the CI
    for df in d.values():
        df['ci'] = calculate_ci(df[column], ci_level=ci_level)
    
    # merge results
    df = list(d.values())[0]
    for df_ in list(d.values())[1:]:
        df = df.append(df_)

    return df


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
    for col_name, df in dfs.items():

        # if vegetation data
        if 'COPERNICUS/S2' in col_name or 'LANDSAT' in col_name:

            # calcualte residuals to the mean
            res = (df[column] - df[column].mean()).abs()

            # determine which are outliers
            outlier = res > df[column].std()*sigmas
            #outlier = res > 300

            # set to None
            df.loc[outlier, column] = None

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

    Returns
    ----------
    dict of DataFrame
        Time series data for multiple sub-image locations with 
        new column for smoothed data and ci.
    """

    # loop over collections
    for col_name, df in dfs.items():

        # if vegetation data
        if 'COPERNICUS/S2' in col_name or 'LANDSAT' in col_name:

            # remove outliers and smooth
            df = smooth_all_sub_images(df, column=column, n=n)

            # calculate ci
            df = get_confidence_intervals(df, column=column)

            # replace DataFrame
            dfs[col_name] = df

    return dfs


def create_lat_long_metric_figures(data_df, metric, output_dir):

    """
    From input data-frame with processed network metrics create 2D gird figure for each date available using Geopandas.

    :param data_df -- input dataframe
    :param metric -- variable to plot
    :param output_dir -- directory to save the figures

    :return:
    """

    if set(['date',metric]).issubset(data_df.columns):

        # get min and max values observed in the data to create a range

        vmin = min(data_df[metric])
        vmax = max(data_df[metric])

        # get all dates available
        list_of_dates = np.unique(data_df['date'])

        for date in list_of_dates:

            if (data_df[data_df['date'] == date][metric].isnull().values.any()):
                print('Problem with date ' + pd.to_datetime(str(date)).strftime('%Y-%m-%d') + ' nan entries found.')
                continue

            network_figure(data_df,date,metric,vmin,vmax,output_dir)

    else:
        raise RuntimeError("Expected variables not present in input dataframe")


def coarse_dataframe(data_df_all, side_square):
    """

    Coarse the granularity of a dataframe by grouping lat,long points that are close to each other in a square of L = size_square
    :param data_df:  Input dataframe
    :param side_square: Side of the square
    :return: a coarser dataframe
    """

    # initialise the categories

    data_df_all['category'] = -1

    # do calculations on the first date, then extrapolate to the rest
    data_df = data_df_all[data_df_all['date']==np.unique(data_df_all['date'])[0]]

    data_df = data_df.sort_values(by=['latitude', 'longitude'])

    n_grids = int(math.sqrt(data_df.shape[0]))


    category = 0

    for n in range(data_df.shape[0]):

        # only process lat,long point that do not have a category
        if data_df.loc[n,'category'] == -1:

            # get the side_square^2 nearest indexes to the point.
            indexes = []
            for i in range(side_square):
                for j in range(side_square):

                    if n + n_grids*i + j < n_grids*n_grids and data_df['category'].iloc[n + n_grids*i + j]==-1:
                            indexes.append(n + n_grids*i + j)

            # assing them all to the same categorty
            data_df.loc[indexes,'category'] = str(category)

            # get the geometry points of that catery
            cat_geometry = data_df[data_df['category']==str(category)]['geometry']

            # get indexes of each point belonging to the category
            indexes_all = []
            for point in cat_geometry:
                indexes_all.append(data_df_all[data_df_all['geometry'] == point].index.tolist())

            indexes_all_flat = [item for sublist in indexes_all for item in sublist]

            data_df_all.loc[indexes_all_flat,'category'] = str(category)

            category = category + 1



    data_df_all['category'] =  (data_df_all['category'].astype(str)).str.cat(data_df_all['date'],sep="_")

    data_df_all = data_df_all.dissolve(by=['category','date'], aggfunc='mean')

    # re-assing the date because we are losing it
    data_df_all['date']= [i[1] for i in data_df_all.index]

    data_df_all['category'] =  [i[0] for i in data_df_all.index]


    return data_df_all


def network_figure(data_df, date, metric, vmin, vmax, output_dir):
    '''

    Make 2D heatmap plot with network centrality measures

    :param data_df: input dataframe
    :param date: date to be plot
    :param metric: which metric is going to be plot
    :param vmin: colorbar minimum values
    :param vmax: colorbar max values
    :param output_dir: dir where to save the plots
    :return:
    '''


    fig, ax = plt.subplots(1, figsize=(6, 6))

    cmap = cm.coolwarm
    data_df[data_df['date'] == date].plot(marker='s', ax=ax, alpha=.5, markersize=100, column=metric, \
                                          figsize=(10, 10), linewidth=0.8, edgecolor='0.8', cmap=cmap)

    # from datetime type to a string
    date_str = pd.to_datetime(str(date)).strftime('%Y-%m-%d')

    # create a date annotation on the figure
    ax.annotate(date_str, xy=(0.15, 0.08), xycoords='figure fraction',
                horizontalalignment='left', verticalalignment='top',
                fontsize=25)

    # Create colorbar as a legend
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm._A = []
    fig.colorbar(sm)

    # create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    metric_output_name = metric.replace("/", "_")

    # this saves the figure as a high-res png in the output path.
    filepath = os.path.join(output_dir, metric_output_name + '_network_2D_grid_' + date_str + '.png')
    fig.savefig(filepath, dpi=200)

    plt.close(fig)




def resample_time_series(df, col_name="offset50"):
    """
    Resample and interpolate a time series dataframe so we have one row
    per day (useful for FFT)

    Parameters
    ----------
    df: DataFrame with date as index
    col_name: string, identifying the column we will pull out

    Returns
    -------
    new_series: pandas Series with datetime index, and one column, one row per day
    """
    series = df[col_name]
    # just in case the index isn't already datetime type
    series.index = pd.to_datetime(series.index)

    # resample to get one row per day
    rseries = series.resample("D")
    new_series = rseries.interpolate()

    return new_series



def fft_series(time_series):
    """
    Perform Fast Fourier Transform on an input series (assume one row per day).

    Parameters
    ----------
    time_series: a pandas Series with one row per day, and datetime index (which we'll ignore)

    Returns
    -------
    xvals, yvals: np.arrays of frequencies (1/day) and strengths in frequency space.
                  Ready to be plotted directly in a matplotlib plot.
    """

    ts = list(time_series)
    # Number of points
    N = len(ts)
    # Sample spacing (days)
    T = 1.0
    fourier = fft(ts)
    # x-axis values
    xvals = np.linspace(0.,1.0/(20*T), N//20)
    yvals = 2.0/N * np.abs(fourier[0:N//20])
    return xvals, yvals


def write_slimmed_csv(dfs, output_dir):

    for collection_name, veg_df in dfs.items():
        if collection_name == 'COPERNICUS/S2' or 'LANDSAT' in collection_name:

            df_summary = dfs['ECMWF/ERA5/MONTHLY']
            df_summary.loc[veg_df.index, 'offset50_mean'] = veg_df['offset50_mean']
            df_summary.loc[veg_df.index, 'offset50_std'] = veg_df['offset50_std']
            df_summary.loc[veg_df.index, 'offset50_smooth_mean'] = veg_df['offset50_smooth_mean']
            df_summary.loc[veg_df.index, 'offset50_smooth_std'] = veg_df['offset50_smooth_std']

            summary_csv_filename = os.path.join(output_dir, collection_name.replace('/', '-')+'_time_series.csv')

            print(f"\nWriting '{summary_csv_filename}'...")
            df_summary.to_csv(summary_csv_filename)


def get_AR1_parameter_estimate(ys):
    """
    Fit an AR(1) model to the time series data and return
    the associated parameter of the model.

    Parameters
    ----------
    ys: array
        Input time series data.

    Returns
    -------
    float
        The parameter value of the AR(1) model..
    """
    
    if len(ys) < 5:
        return np.NaN

    from statsmodels.tsa.ar_model import AutoReg
    
    # more sophisticated models to consider:
    #from statsmodels.tsa.statespace.sarimax import SARIMAX
    #from statsmodels.tsa.arima_model import ARMA

    # create the AR(1) model
    model = AutoReg(ys, lags=1)

    # fit
    model = model.fit()
    
    # get the single parameter value
    parameter = model.params[1]

    return parameter


def get_kendell_tau(ys):
    """
    Kendall's tau gives information about the trend of the time series.
    It is just a rank correlation test with one variable being time 
    (or the vector 1 to the length of the time series), and the other 
    variable being the data itself. A tau value of 1 means that the 
    time series is always increasing, whereas -1 mean always decreasing,
    and 0 signifies no overall trend.

    Parameters
    ----------
    ys: array
        Input time series data.

    Returns
    -------
    float
        The value of tau.
    float
        The p value of the rank correlation test.
    """

    from scipy.stats import kendalltau

    # calculate Kendall tau
    tau, p = kendalltau(range(len(ys)), ys)

    return tau, p