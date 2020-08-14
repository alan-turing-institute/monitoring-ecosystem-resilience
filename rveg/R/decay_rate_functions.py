#	Functions to calculate the decay rate in a time series
#	Input (for now) is a single time series, default option is monthly
#	Looks at the mean annual time series to calculate rate on (from max to min)
#
#	Author: Chris A. Boulton
#	Date: 11th August 2020
#	email: c.a.boulton@exeter.ac.uk
#

import pandas as pd
import numpy as np

def mean_annual_ts(x, resolution=12):
    # calculates the mean annual time series of time series x
    # resolution is the number of time points per year in the time series
    # infills missing data by linear interpolation
    # note that the first or last value will not be infilled if missing
    missing_inds = which(is_na(x))
    if length(missing_inds) > 0:
        for i in range(1, length(missing_inds)):
            x[i] = mean(make_tuple(x[i - 1], x[i + 1]))
    mean_cycle = rep(NA, resolution)
    for i in range(1, resolution):
        mean_cycle[i] = mean(x[seq(i, length(x), resolution)], na_rm=T)
    return mean_cycle


def decay_rate(x, resolution=12, method='basic'):
    # takes a time series as an input and uses mean_annual_ts()
    # then calculates the decay rate between the max and min values
    # choice of 'basic' or 'adjusted' for method:
    #   basic - uses the raw values
    # adjusted - substracts the minimum value and adds 1 to time series before
    # calculation
    annual_cycle = mean_annual_ts(x, resolution)
    if method == 'basic':
        ts = annual_cycle
    elif method == 'adjusted':
        ts = annual_cycle - min(annual_cycle) + 1
    else:
        ts = NA  # causes fail if method is not specified properly
    max_ind = which_max(ts)
    min_ind = which_min(ts)
    if min_ind < max_ind:
        # this ensures the length of time for decay is correct below
        min_ind = min_ind + resolution
    dr = log(min(ts) / max(ts)) / (min_ind - max_ind)
    return dr


def exp_model_fit(x, resolution=12, method='basic'):
	#takes a time series as an input and uses mean_annual_ts()
	#then fits a linear regression model on the time series from max to min value
	#choice of 'basic' or 'adjusted' for method:
	#	basic - uses the raw values
	#	adjusted - substracts the minimum value and adds 1 to time series before calculation
	#####	NB: offset50 values are negative by default and so have to use adjusted method #####
	annual_cycle = mean_annual_ts(x, resolution)
	if method == 'basic':
		ts = annual_cycle
	elif method == 'adjusted':
		ts = annual_cycle - min(annual_cycle) + 1
	else:
		ts = NA		#causes fail if method is not specified properly
	max_ind = which.max(ts)
	min_ind = which.min(ts)
	#in most cases we find the minimum value is earlier in the year
	#so the below crosses Dec/Jan if this is the case
	#otherwise remains within a single year cycle
	if min_ind < max_ind:
		exp_ts = ts[c(max_ind:resolution,1:min_ind)]
	else:
		exp_ts <- ts[max_ind:min_ind]
	exp_mod = lm(log(exp_ts) ~ c(1:length(exp_ts)))
	return exp_mod
	#this will return the output of the model, from which the exponential decay can be extracted











