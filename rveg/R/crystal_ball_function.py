#	Function to fit a crystal ball pdf to a time series
#	Input (for now) is a single time series, default option is monthly
#	Looks at the mean annual time series which is reversed such that the tail is
#	on the left, and then normalises as the crystal ball function is a pdf
#
#	Author: Chris A. Boulton
#	Translated from R to Python by Jesse F. Abrams & Chris A. Boulton
#	Date: 14th August 2020
#	email: c.a.boulton@exeter.ac.uk
#

import numpy as np
import scipy as sp
import pandas as pd
import scipy.optimize as sco
from scipy.stats import norm

def mean_annual_ts(x, resolution=12):
    # calculates the mean annual time series of time series x
    # resolution is the number of time points per year in the time series
    # infills missing data by linear interpolation
    # note that the first or last value will not be infilled if missing
    missing_inds = np.where(np.isnan(x))[0]
    if len(missing_inds) > 0:
        for i in range(len(missing_inds)):
            print(i)
            x[missing_inds[i]] = np.mean([x[missing_inds[i] - 1], x[missing_inds[i] + 1]])
    mean_cycle = np.repeat(np.nan, resolution, axis=0)
    for i in range(resolution):
        mean_cycle[i] = np.nanmean(x[range(i,len(x),12)])
    return mean_cycle

def reverse_normalise_ts(x):
	#takes the time series as an input (assumed to be from mean_annual_ts
	#sorts such that the lowest point is at the end of the series
	#reverses the order and the normalises (adding min(x) and dividing by sum(x))
	#this puts the lowest point at the start
	min_ind = np.where(x == np.min(x))[0][0]
	arrangex = np.append(x[(min_ind+1):len(x)],x[0:(min_ind+1)])
	revx = arrangex[::-1]
	normx = (revx-np.min(revx))/sum(revx-np.min(revx))
	return normx

##A,B,C,D and N are all sub-functions in crystal ball
##erf (error function) is a common function also used
def erf(x):
	output = 2 * norm.cdf(x * np.sqrt(2)) - 1
	return output

def A(alpha,n):
	output = ((n/np.abs(alpha))**n)*np.exp((-np.abs(alpha)**2)/2)
	return output

def B(alpha,n):
	output = n/np.abs(alpha) - np.abs(alpha)
	return output

def N(sigma,C,D):
	output = 1/(sigma*(C+D))
	return output

def C(alpha,n):
	output = (n/np.abs(alpha))*(1/(n-1))*np.exp((-np.abs(alpha)**2)/2)
	return output

def D(alpha):
	output = np.sqrt(np.pi/2)*(1+erf(np.abs(alpha)/np.sqrt(2)))
	return output

def cball(x,alpha,n,xbar,sigma):
	#uses the functions above to create an output from input x
	#works for single values but assumed x will be a time series
	fx = np.repeat(np.nan, len(x), axis=0)
	for i in range(len(x)):
		if (((x[i]-xbar)/sigma) > -alpha):
			fx[i] = N(sigma,C(alpha,n),D(alpha))*np.exp((-(x[i]-xbar)**2)/(2*sigma**2))
		if (((x[i]-xbar)/sigma) <= -alpha):
			fx[i] = N(sigma,C(alpha,n),D(alpha))*A(alpha,n)*(B(alpha,n)-(x[i]-xbar)/sigma)**(-n)
	return fx

def err_func(params, ts):
	model_output = cball(range(1,len(ts)+1),params[0], params[1], params[2], params[3])
	residuals = []
	for i in range(0,len(ts)):
		r = model_output[i] - ts[i]
		residuals.append(r)
	return residuals

def cball_parfit(p0, timeseries):
	mean_ts = mean_annual_ts(timeseries)
	ts = reverse_normalise_ts(mean_ts)
	p1, success = sco.leastsq(err_func, p0, args=ts)
	return p1, success

#intial parameters in order alpha, n, xbar, sigma
p0 = [1.5, 150.0, 8.0, 2.0]







