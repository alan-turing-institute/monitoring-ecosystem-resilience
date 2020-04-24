"""
Data analysis code including functions to read the .json results file,
and functions analyse and plot the data.

"""

import json
import math
import os

import numpy as np
import pandas as pd

from shapely.geometry import Point
import geopandas as gpd

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from scipy.fftpack import fft
from scipy.stats import sem, t
from statsmodels.tsa.seasonal import STL


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


def create_lat_long_metric_figures(geodf, metric, output_dir):
    """
    From input data-frame with processed network metrics create 2D gird figure for each date available using Geopandas.

    Parameters
    ----------
    geodf:  GeoDataframe
        Input dataframe
    metric: string
        Variable to plot
    output_dir: string
        Directory to save the figures

     Returns
    ----------

    """

    if {'date', metric}.issubset(geodf.columns):

        # get min and max values observed in the data to create a range

        vmin = min(geodf[metric])
        vmax = max(geodf[metric])

        # get all dates available
        list_of_dates = np.unique(geodf['date'])

        for date in list_of_dates:

            if geodf[geodf['date'] == date][metric].isnull().values.any():
                print('Problem with date ' + pd.to_datetime(str(date)).strftime('%Y-%m-%d') + ' nan entries found.')
                continue
            else:
                print('Saving network figure for date ' + pd.to_datetime(str(date)).strftime('%Y-%m-%d'))
                network_figure(geodf, date, metric, vmin, vmax, output_dir)

    else:
        raise RuntimeError("Expected variables not present in input dataframe")


def coarse_dataframe(geodf, side_square):
    """
    Coarse the granularity of a dataframe by grouping lat,long points
    that are close to each other in a square of L = size_square

    Parameters
    ----------
    geodf:  Dataframe
        Input dataframe
    side_square: integer
        Side of the square


    Returns
    ----------
     A  dataframe
        A coarser dataframe

    """

    # initialise the categories

    geodf['category'] = -1

    # do calculations on the first date, then extrapolate to the rest
    data_df = geodf[geodf['date'] == np.unique(geodf['date'])[0]]

    data_df = data_df.sort_values(by=['longitude', 'latitude'])

    n_grids = int(math.sqrt(data_df.shape[0]))

    category = 0

    for n in range(data_df.shape[0]):

        # only process lat,long point that do not have a category
        if data_df['category'].iloc[n] == -1:

            # get the side_square^2 nearest indexes to the point.
            indexes = []
            for i in range(side_square):
                for j in range(side_square):

                    if n + n_grids * i + j < n_grids * n_grids and data_df['category'].iloc[n + n_grids * i + j] == -1:
                        indexes.append(n + n_grids * i + j)

            # assing them all to the same categorty
            data_df['category'].iloc[indexes] = str(category)

            # get the geometry points of that catery
            cat_geometry = data_df[data_df['category'] == str(category)]['geometry']

            # get indexes of each point belonging to the category
            indexes_all = []
            for point in cat_geometry:
                indexes_all.append(geodf[geodf['geometry'] == point].index.tolist())

            indexes_all_flat = [item for sublist in indexes_all for item in sublist]

            geodf['category'].iloc[indexes_all_flat] = str(category)

            category = category + 1

    geodf['category'] = (geodf['category'].astype(str)).str.cat(geodf['date'], sep="_")

    geodf = geodf.dissolve(by=['category', 'date'], aggfunc='mean')

    # re-assing the date because we are losing it
    geodf['date'] = [i[1] for i in geodf.index]

    geodf['category'] = [i[0] for i in geodf.index]

    return geodf


def network_figure(df, date, metric, vmin, vmax, output_dir):
    """

    Make 2D heatmap plot with network centrality measures

    Parameters
    ----------
    df:  Dataframe
        Input dataframe
    date: String
        Date to be plot
    metric: string
        Which metric is going to be plot
    vmin: int
        Colorbar minimum values
    vmax: int
        Colorbar max values
    output_dir: string
        Directory where to save the plots

    Returns
    ----------
    """

    fig, ax = plt.subplots(1, figsize=(6, 6))

    cmap = matplotlib.cm.get_cmap('coolwarm')

    df[df['date'] == date].plot(marker='s', ax=ax, alpha=.5, markersize=100, column=metric,
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
    xvals = np.linspace(0., 1.0 / (20 * T), N // 20)
    yvals = 2.0 / N * np.abs(fourier[0:N // 20])
    return xvals, yvals


def write_slimmed_csv(dfs, output_dir, filename_suffix=''):
    for collection_name, veg_df in dfs.items():
        if collection_name == 'COPERNICUS/S2' or 'LANDSAT' in collection_name:
            df_summary = dfs['ECMWF/ERA5/MONTHLY']
            df_summary.loc[veg_df.index, 'offset50_mean'] = veg_df['offset50_mean']
            df_summary.loc[veg_df.index, 'offset50_std'] = veg_df['offset50_std']
            df_summary.loc[veg_df.index, 'offset50_smooth_mean'] = veg_df['offset50_smooth_mean']
            df_summary.loc[veg_df.index, 'offset50_smooth_std'] = veg_df['offset50_smooth_std']

            summary_csv_filename = os.path.join(output_dir, collection_name.replace('/',
                                                                                    '-') + '_time_series' + filename_suffix + '.csv')

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
    float
        The parameter standard error
    """

    ys = ys.dropna()

    if len(ys) < 5:
        return np.NaN, np.NaN

    from statsmodels.tsa.ar_model import AutoReg

    # more sophisticated models to consider:
    # from statsmodels.tsa.statespace.sarimax import SARIMAX
    # from statsmodels.tsa.arima_model import ARMA

    # create the AR(1) model
    model = AutoReg(ys, lags=1)

    # fit
    model = model.fit()

    # get the single parameter value
    parameter = model.params[1]
    se = model.bse[1]

    return parameter, se


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


def write_to_json(filename, out_dict):
    """
    Create or append the contents of `out_dict` 
    to json file `filename`.

    Parameters
    ----------
    filename: array
        Output json filename.
    out_dict: dict
        Information to save.
    """

    # if file doesn't exist
    if not os.path.exists(filename):
        # make enclosing dir if needed
        output_dir = os.path.dirname(filename)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        #  write new json file
        with open(filename, 'w') as json_file:
            json.dump(out_dict, json_file, indent=2)

    # file exists
    else:
        # json read
        data = None
        with open(filename, 'r') as json_file:
            data = json.load(json_file)

        # update dict   
        for k, v in out_dict.items():
            data[k] = v

        # json write
        with open(filename, 'w') as json_file:
            json.dump(data, json_file, indent=2)


def remove_seasonality_all_sub_images(dfs, lag, period):
    """
    Loop over each sub image time series DataFrames and remove
    time series seasonality.

    Parameters
    ----------
    dfs : dict of DataFrame
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
    for col_name, df in dfs.items():

        #  if vegetation data
        if 'COPERNICUS/S2' in col_name or 'LANDSAT' in col_name:

            # group by (lat, long)
            d = {}
            for name, group in df.groupby(['latitude', 'longitude']):
                d[name] = group
                # for each sub-image
            for key, df_ in d.items():
                df_new = df_.set_index('date')

                uns_df = remove_seasonality(df_new.copy(), lag, period)

                d[key] = uns_df

            # reconstruct the DataFrame
            df = list(d.values())[0]
            for df_ in list(d.values())[1:]:
                df = df.append(df_)

            dfs[col_name] = df

        else:

            # remove seasonality for weather data, this is a simpler time series
            df = dfs[col_name]
            df_new = df.set_index('date')
            uns_df = remove_seasonality(df_new, lag, period)

            uns_df['date'] = uns_df.index
            dfs[col_name] = uns_df

    return dfs


def remove_seasonality_combined(dfs, lag, period='M'):
    """
    Loop over time series DataFrames and remove
    time series seasonality.

    Parameters
    ----------
    dfs : dict of DataFrame
        Time series data for multiple sub-image locations.
    lag : float
        Periodicity to remove

    period: string
        Type of periodicity (day, month, year)

    Returns
    ----------
    dict of DataFrame
        Time series data with
        seasonality removed
    """

    # loop over collections

    for collection_name, df in dfs.items():

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

        dfs[collection_name] = df_resampled

    return dfs


def stl_decomposition(ts_df, period=12):
    stl = STL(ts_df, period, robust=True)

    res = stl.fit()

    return res


def get_max_lagged_cor(dirname, veg_prefix):
    """
    Convenience function which returns the maximum correlation as a 
    function of lag (using a file saved earlier).

    Parameters
    ----------
    dirname : str
        Path to the `analysis/` directory of the current analysis job.

    veg_prefix : str
        Compact representation of the satellite collection name used to
        obtain vegetation data.

    Returns
    ----------
    tuple
        Max correlation, and lag, for smoothed and unsmoothed vegetation time
        series.
    """
    
    # construct path to lagged correlations file
    filename = os.path.join(dirname, 'correlations', 'lagged_correlations.json')
    
     # check file exists
    if not os.path.exists(filename):
        raise FileNotFoundError(f'Could not find file "{os.path.abspath(filename)}".')

    # read file
    json_file = open(filename)
    lagged_cor = json.load(json_file)

    # calculate max corr
    lagged_cor = {k: np.array(v[:5]) for k, v in lagged_cor.items() if veg_prefix in k}
    lagged_cor = {k: (np.max(v), np.argmax(v)) for k, v in lagged_cor.items()}

    max_corr_unsmoothed = lagged_cor[veg_prefix + '_offset50_mean_lagged_correlation']
    max_corr_smooth = lagged_cor[veg_prefix + '_offset50_smooth_mean_lagged_correlation']

    return max_corr_smooth, max_corr_unsmoothed
