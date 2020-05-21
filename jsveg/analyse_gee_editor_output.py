#!/usr/bin/env python

"""
Using the `time_series.js` file in this repository, and the GEE editor
(https://code.earthengine.google.com/), You can download a precipitation
and NDVI time series for a given location quickly. This script let's you 
run the `pyveg` analysis on the downloaded result.

"""


import os
import argparse
import datetime

import pandas as pd

from pyveg.src.analysis_preprocessing import resample_dataframe, detrend_df
from pyveg.scripts.analyse_gee_data import run_time_series_analysis, run_early_warnings_resilience_analysis

def convert_gee_date(gee_date_series):
    """
    Given a pandas Series containing dates of the format "Jan 1, 2000"
    (as downloaded from the GEE Editor), convert to "%Y-%m-%d".

    Parameters
    ----------
    gee_date_series : pandas Series
        Series of dates with format "Jan 1, 2000".

    Returns
    ----------
    pandas Series :
        Series of dates in format "%Y-%m-%d".
    """

    dates = [datetime.datetime.strptime(d, '%b %d, %Y').strftime('%Y-%m-%d') for d in gee_date_series]

    return dates
    

def convert_gee_outputs(input_dir):
    """
    Given a path to a directory, read csv files in the directory
    and convert them to a format on which the pyveg analysis can
    be run.

    Parameters
    ----------
    input_dir : str
        Path to the csv data download from the GEE Editor.

    Returns
    ----------
    str :
        Path to single csv file that can be read by the pyveg analysis 
        code.
    """

    print('Converting GEE Editor outputs...')

    # check the input folder exists
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f'Could not find dir "{input_dir}"! Exiting.')

    # get a list of relevant files
    files = [f for f in os.listdir(input_dir) 
                     if 'ee-chart' in f and '.csv' in f]

    # check some relevant files exist
    if len(files) == 0:
        raise FileNotFoundError((f'Could not find any compatible csv '
                                  'files in "{input_dir}"! Exiting.'))

    # data downloaded from the GEE Editor has this column in it
    gee_default_date_colname = 'system:time_start'

    # create output df
    out_df = pd.DataFrame(columns=['date'])

    # loop through time series csv
    for f in files:

        # read the file
        df = pd.read_csv(os.path.join(input_dir, f))

        # convert date column
        df[gee_default_date_colname] = convert_gee_date(df[gee_default_date_colname])
        df = df.rename(columns={gee_default_date_colname: 'date'})
        df = df.rename(columns={'NDVI': 'L7_ndvi_mean', 'EVI': 'L7_ndvi_mean'})

        # resample to monthly data
        df = resample_dataframe(df, [c for c in df.columns if 'date' not in c])

        # join with out_df
        out_df = pd.merge(out_df, df, how='outer')

    detrended_df = detrend_df(out_df)

    # save the df
    out_filename = os.path.join(input_dir, 'time_series.csv')
    out_filename_detrended = os.path.join(input_dir, 'time_series_detrended.csv')
    out_df.to_csv(out_filename, index=False)
    detrended_df.to_csv(out_filename_detrended, index=False)

    return out_filename, out_filename_detrended


def main():
    """
    CLI interface.
    """
    parser = argparse.ArgumentParser(description="Run pyveg analysis on data download from the GEE Editor.")
    parser.add_argument("--input_dir", help="Directory containing files corresponding to different downloaded timeseries.")

    print('-' * 36)
    print('Running analyse_gee_editor_output.py')
    print('-' * 36)

    # parse args
    args = parser.parse_args()
    input_dir = args.input_dir

    # put output plots in the results dir
    output_dir = os.path.join(input_dir, 'analysis')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # convert GEE Editor files
    ts_filenames = convert_gee_outputs(input_dir)

    # for each time series
    for filename in ts_filenames:
        
        ts_file = filename
        print(f'\n* Analysing "{ts_file}"...')
        print('.'*50)

        # run the standard or detrended analysis
        if 'detrended' in filename:
            output_subdir = os.path.join(output_dir, 'detrended')
            run_time_series_analysis(ts_file, output_subdir, detrended=True)

            ews_subdir = os.path.join(output_dir, 'resiliance/deseasonalised')
            run_early_warnings_resilience_analysis(ts_file, ews_subdir)

        else:
            output_subdir = output_dir
            run_time_series_analysis(ts_file, output_subdir)

            ews_subdir = os.path.join(output_dir, 'resiliance/seasonal')
            run_early_warnings_resilience_analysis(ts_file, ews_subdir)
            

if __name__ == "__main__":
    main()
