"""
Data analysis code including functions to read the .json results file,
and functions analyse and plot the data.

"""

import json
import math
import os
import datetime

import numpy as np
import pandas as pd

from shapely.geometry import Point
import geopandas as gpd

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from scipy.fftpack import fft
from scipy.stats import sem, t
from statsmodels.tsa.seasonal import STL
import ewstools


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

    cmap = plt.cm.get_cmap('coolwarm')

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
            df_summary = dfs['ECMWF/ERA5/DAILY']
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

    if len(ys) < 4:
        print('Time series too short to reliably calculate AR1')
        return np.NaN, np.NaN

    from statsmodels.tsa.ar_model import AutoReg

    # more sophisticated models to consider:
    # from statsmodels.tsa.statespace.sarimax import SARIMAX
    # from statsmodels.tsa.arima_model import ARMA

    # create and fit the AR(1) model
    if pd.infer_freq(ys.index) is not None:
        # explicitly add frequency to index to prevent warnings
        ys.index = pd.DatetimeIndex(ys.index, freq=pd.infer_freq(ys.index))
        model = AutoReg(ys, lags=1, missing='drop').fit() # currently warning
    else:
        # remove index
        model = AutoReg(ys.values, lags=1, missing='drop').fit() # currently warning

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


def stl_decomposition(series, period=12):
    """
    Run STL decomposition on a pandas Series object.

    Parameters
    ----------
    series : Series object
        The observations to be deseasonalised.
    period : int (optional)
        Length of the seasonal period in observations.
    """

    stl = STL(series, period, robust=True)
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

    if veg_prefix + '_offset50_mean_lagged_correlation' in lagged_cor.keys():
        max_corr_unsmoothed = lagged_cor[veg_prefix + '_offset50_mean_lagged_correlation']
    else:
        max_corr_unsmoothed = (np.NaN, np.NaN)
    if veg_prefix + '_offset50_smooth_mean_lagged_correlation' in lagged_cor.keys():
        max_corr_smooth = lagged_cor[veg_prefix + '_offset50_smooth_mean_lagged_correlation']
    else:
        max_corr_smooth = (np.NaN, np.NaN)

    return max_corr_smooth, max_corr_unsmoothed


def variance_moving_average_time_series(series, length=1):
    """
    Calculate a variance time series using a moving average

    Parameters
    ----------
    series : pandas Series
        Time series observations.
    length : int
        Length of the moving window in number of observations.

    Returns
    -------
    pandas Series: 
        pandas Series with datetime index, and one column, one row per date.
    """
    
    # just in case the index isn't already datetime type
    series.index = pd.to_datetime(series.index)

    variance = series.rolling(length).var()

    variance.name = series.name+"_var"

    return variance


def ar1_moving_average_time_series(series, length=1):
    """
    Calculate an AR1 time series using a moving average
    
    Parameters
    ----------
    series : pandas Series
        Time series observations.
    length : int
        Length of the moving window in number of observations.
    
    Returns
    -------
    pandas Series: 
        pandas Series with datetime index, and one column, one row per date
    """

    # just in case the index isn't already datetime type
    series.index = pd.to_datetime(series.index)

    ar1 = []
    ar1_se = []
    index = []

    for i in range(len(series) - length ):
        #print(series[i:(length  + i)])
        param, se = get_AR1_parameter_estimate(series[i:(length  + i)])
        ar1.append(param)
        ar1_se.append(se)
        index.append(series.index[length  + i])

    ar1_name = series.name+"_ar1"
    ar1_se_name = series.name+"_ar1_se"

    ar1_df = pd.DataFrame()
    ar1_df[ar1_name] = pd.Series(ar1)
    ar1_df[ar1_se_name] = pd.Series(ar1_se)
    ar1_df.index = index

    return ar1_df


def get_ar1_var_timeseries_df(series, window_size=0.5):
    """
    Given a time series calculate AR1 and variance using
    a moving window. Put the two resulting time series into
    a new DataFrame and return the result.

    Parameters
    ----------
    series : pandas Series
        Time series observations.
    window_size: float (optional)
        Size of the moving window as a fraction of the time series length.

    Returns
    ----------
    DataFrame
        The AR1 and variance results in a time series dataframe.
    """

    # drop null values
    series = series.dropna()

    # calculate the length in number of time points of the moving window
    length = round(len(series) * window_size)

    # calculate the ar1 and variance
    ar1_df = ar1_moving_average_time_series(series, length)
    variance = variance_moving_average_time_series(series, length)

    # merge results
    ar1_var_df = pd.merge(variance, ar1_df, left_index=True, right_index=True)

    return ar1_var_df


def moving_window_analysis(df, output_dir, window_size=0.5):
    """
    Run moving window AR1 and variance calculations for several
    input time series time series.

    Parameters
    ----------
    df : DataFrame
        Input time series DataFrame containing several time series.
    output_dir : str
        Path output plotting directory.
    window_size: float (optional)
        Size of the moving window as a fraction of the time series length.

    Returns
    ----------
    DataFrame
        AR1 and variance time-series for each of the input time series.
    """

    # new output dataframe
    mwa_df = pd.DataFrame()

    # loop through columns
    for column in df.columns:

        # run moving window analysis veg and precip columns
        if ( ('offset50' in column or 'ndvi' in column) and 'mean' in column or 
             'total_precipitation' in column ):
            
            # reindex time series using data
            time_series = df.set_index('date')[column]

            # compute AR1 and variance time series
            df_ = get_ar1_var_timeseries_df(time_series, window_size)
            mwa_df = mwa_df.join(df_, how='outer')


    # use date as a column, and reset index
    mwa_df.index.name = 'date'
    mwa_df = mwa_df.reset_index()

    return mwa_df


def get_datetime_xs(df):
    
    try:
        xs = [datetime.datetime.strptime(d, '%Y-%m-%d').date() for d in df.date]
    except:
        # if the time series has been resampled the index is a TimeStamp object
        xs = [datetime.datetime.strptime(d._date_repr, '%Y-%m-%d').date() for d in df.date]

    return xs

def early_warnings_sensitivity_analysis(series,
                                        indicators=['var','ac'],
                                        winsizerange = [0.10, 0.8],
                                        incrwinsize = 0.10,
                                        smooth = "Gaussian",
                                        bandwidthrange = [0.05, 1.],
                                        spanrange = [0.05, 1.1],
                                        incrbandwidth = 0.2,
                                        incrspanrange = 0.1):

    '''

    Function to estimate the sensitivity of the early warnings analysis to the smoothing and windowsize used. The function
    returns a dataframe that contains the Kendall tau rank correlation estimates for the rolling window sizes (winsize variable)
    and bandwidths or span sizes depending on the de-trending (smooth variable).

    This function is inspired in the sensitivity_ews.R function from Vasilis Dakos, Leo Lahti in the early-warnings-R package
    https://github.com/earlywarningtoolbox/earlywarnings-R.

    Parameters
    ----------
    series : pandas Series
        Time series observations.
    indicators: list of strings
        The statistics (leading indicator) selected for which the sensitivity analysis is perfomed.
    winsizerange: list of float
        Range of the rolling window sizes expressed as ratio of the timeseries length (must be numeric between 0 and 1). Default is 0.25 - 0.75.
    incrwinsize: float
        Increments the rolling window size (must be numeric between 0 and 1). Default is 0.25.
    smooth: string
        Type of detrending. It can be {'Gaussian', 'Lowess', 'None'}.
    bandwidthrange: list of float
        Range of the bandwidth used for the Gaussian kernel when gaussian filtering is selected. It is expressed as percentage of the timeseries length (must be numeric between 0 and 100). Default is 5\% - 100\%.
    spanrange: list of float
        Parameter that controls the degree of Lowess smoothing (numeric between 0 and 1). Default is 0.05 - 1.
    incrbandwidth: float
        Size to increment the bandwidth used for the Gaussian kernel when gaussian filtering is applied. It is expressed as percentage of the timeseries length (must be numeric between 0 and 1). Default is 0.2.
    incrspanrange: float
        Size to increment the the span used for the Lowess smoothing

    Returns
    --------
    DataFrame:
        A dataframe that contains the Kendall tau rank correlation estimates for the rolling window sizes (winsize variable)
     and bandwidths or span sizes depending on the de-trending (smooth variable).


    '''

    results_kendal_tau = []
    for winsize in np.arange(winsizerange[0],winsizerange[1]+0.01,incrwinsize):

        winsize = round(winsize,3)
        if smooth == "Gaussian":

            for bw in np.arange(bandwidthrange[0], bandwidthrange[1]+0.01, incrbandwidth):

                bw = round(bw, 3)
                ews_dic_veg = ewstools.core.ews_compute(series.dropna(),
                                                        roll_window=winsize,
                                                        smooth=smooth,
                                                        lag_times=[1, 2],
                                                        ews=indicators,
                                                        band_width=bw)

                result = ews_dic_veg['Kendall tau']
                result['smooth'] = bw
                result['winsize'] = winsize

                results_kendal_tau.append(result)


        elif smooth =="Lowess":

            for span in np.arange(spanrange[0], spanrange[1]+0.01, incrspanrange):

                span = round(span,2)
                ews_dic_veg = ewstools.core.ews_compute(series.dropna(),
                                                        roll_window=winsize,
                                                        smooth=smooth,
                                                        lag_times=[1, 2],
                                                        ews=indicators,
                                                        span=span)

                result = ews_dic_veg['Kendall tau']
                result['smooth'] = bw
                result['winsize'] = winsize

                results_kendal_tau.append(result)

        else:

            ews_dic_veg = ewstools.core.ews_compute(series.dropna(),
                                                    roll_window=winsize,
                                                    smooth='None',
                                                    lag_times=[1, 2],
                                                    ews=indicators)

            result = ews_dic_veg['Kendall tau']
            result['smooth'] = 0
            result['winsize'] = winsize

            results_kendal_tau.append(result)

    sensitivity_df = pd.concat(results_kendal_tau)

    return sensitivity_df


def early_warnings_null_hypothesis(series,
                                   roll_window=0.4,
                                   smooth='Lowess',
                                   span=0.1,
                                   band_width=0.2,
                                   indicators=['var', 'ac'],
                                   lag_times=[1],
                                   n_simulations=1000):

    ews_dic = ewstools.core.ews_compute(series,
                                        roll_window=roll_window,
                                        smooth=smooth,
                                        span=span,
                                        band_width=band_width,
                                        ews=indicators,
                                        lag_times=lag_times)

    from statsmodels.tsa.arima_model import ARIMA
    from statsmodels.tsa.arima_process import ArmaProcess

    # Use the short_series EWS if smooth='None'. Otherwise use reiduals.
    eval_series = ews_dic['EWS metrics']['Residuals']

    # Fit ARMA model based on AIC
    aic_max = 10000

    for i in range(0, 2):
        for j in range(0, 2):

            model = ARIMA(eval_series, order=(i, j, 0))
            model_fit = model.fit()
            aic = model_fit.aic

            print("AR", "MA", "AIC")
            print(i, j, aic)

            if aic < aic_max:
                aic_max = aic
                result = model_fit

    def compute_indicators(series):

        ''' Rolling window indicators computation based on the ewstools.core.ews_compute function from
        ewstools
        '''

        df_ews = pd.DataFrame()
        # Compute the rolling window size (integer value)
        rw_size = int(np.floor(roll_window * series.shape[0]))

        # ------------ Compute temporal EWS---------------#

        # Compute standard deviation as a Series and add to the DataFrame
        if 'sd' in indicators:
            roll_sd = series.rolling(window=rw_size).std()
            df_ews['Standard deviation'] = roll_sd

        # Compute variance as a Series and add to the DataFrame
        if 'var' in indicators:
            roll_var = series.rolling(window=rw_size).var()
            df_ews['Variance'] = roll_var

        # Compute autocorrelation for each lag in lag_times and add to the DataFrame
        if 'ac' in indicators:
            for i in range(len(lag_times)):
                roll_ac = series.rolling(window=rw_size).apply(
                    func=lambda x: pd.Series(x).autocorr(lag=lag_times[i]),
                    raw=True)
                df_ews['Lag-' + str(lag_times[i]) + ' AC'] = roll_ac

        # Compute Coefficient of Variation (C.V) and add to the DataFrame
        if 'cv' in indicators:
            # mean of raw_series
            roll_mean = series.rolling(window=rw_size).mean()
            # standard deviation of residuals
            roll_std = series.rolling(window=rw_size).std()
            # coefficient of variation
            roll_cv = roll_std.divide(roll_mean)
            df_ews['Coefficient of variation'] = roll_cv

        # Compute skewness and add to the DataFrame
        if 'skew' in indicators:
            roll_skew = series.rolling(window=rw_size).skew()
            df_ews['Skewness'] = roll_skew

        # Compute Kurtosis and add to DataFrame
        if 'kurt' in indicators:
            roll_kurt = series.rolling(window=rw_size).kurt()
            df_ews['Kurtosis'] = roll_kurt

        # ------------Compute Kendall tau coefficients----------------#

        ''' In this section we compute the kendall correlation coefficients for each EWS
            with respect to time. Values close to one indicate high correlation (i.e. EWS
            increasing with time), values close to zero indicate no significant correlation,
            and values close to negative one indicate high negative correlation (i.e. EWS
            decreasing with time).'''

        # Put time values as their own series for correlation computation
        time_vals = pd.Series(df_ews.index, index=df_ews.index)

        # List of EWS that can be used for Kendall tau computation
        ktau_metrics = ['Variance', 'Standard deviation', 'Skewness', 'Kurtosis', 'Coefficient of variation', 'Smax',
                        'Smax/Var', 'Smax/Mean'] + ['Lag-' + str(i) + ' AC' for i in lag_times]
        # Find intersection with this list and EWS computed
        ews_list = df_ews.columns.values.tolist()
        ktau_metrics = list(set(ews_list) & set(ktau_metrics))

        # Find Kendall tau for each EWS and store in a DataFrame
        dic_ktau = {x: df_ews[x].corr(time_vals, method='kendall') for x in ktau_metrics}  # temporary dictionary
        df_ktau = pd.DataFrame(dic_ktau, index=[0])  # DataFrame (easier for concatenation purposes)

        # -------------Organise final output and return--------------#

        # Ouptut a dictionary containing EWS DataFrame, power spectra DataFrame, and Kendall tau values
        output_dic = {'EWS metrics': df_ews, 'Kendall tau': df_ktau}

        return output_dic

    process = ArmaProcess.from_estimation(result)

    # run simulations on best fitted ARIMA process and get values
    kendall_tau = []
    for i in range(n_simulations):
        ts = process.generate_sample(len(eval_series))

        kendall_tau.append(compute_indicators(pd.Series(ts))['Kendall tau'])

    surrogates_kendall_tau_df = pd.concat(kendall_tau)
    surrogates_kendall_tau_df['true_data'] = False

    # get results for true data
    data_kendall_tau_df = compute_indicators(eval_series)['Kendall tau']
    data_kendall_tau_df['true_data'] = True

    # return dataframe with both surrogates and true data
    kendall_tau_df = pd.concat([data_kendall_tau_df,surrogates_kendall_tau_df])

    return kendall_tau_df
