#	Function to fit a crystal ball pdf to a time series
#	Input (for now) is a single time series, default option is monthly
#	Looks at the mean annual time series which is reversed such that the tail is
#	on the left, and then normalises as the crystal ball function is a pdf
#
#	Author: Chris A. Boulton
#	Date: 14th August 2020
#	email: c.a.boulton@exeter.ac.uk
#

import pandas as pd
import numpy as np

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
        mean_cycle[i] = np.nanmean(x[np.linspace(start=i, stop=len(x)-1, num=resolution,dtype=int)])
    return mean_cycle

def reverse_normalise_ts(x):
	#takes the time series as an input (assumed to be from mean_annual_ts
	#sorts such that the lowest point is at the end of the series
	#reverses the order and the normalises (adding min(x) and dividing by sum(x))
	#this puts the lowest point at the start
	min_ind = np.where(x == np.min(x))[0][0]
	arrangex = np.append(x[(min_ind+1):len(x)],x[0:min_ind])
	revx = rev(arrangex)
	normx = (revx-np.min(revx))/sum(revx-np.min(revx))
	return normx

##A,B,C,D and N are all sub-functions in crystal ball
##erf (error function) is a common function also used
def erf(x):
	output = 2 * pnorm(x * sqrt(2)) - 1
	return output

def A(alpha,n):
	output = ((n/abs(alpha))^n)*exp((-abs(alpha)^2)/2)
	return output

def B(alpha,n):
	output = n/abs(alpha) - abs(alpha)
	return output

def N(sigma,C,D):
	output = 1/(sigma*(C+D))
	return output

def C(alpha,n):
	output = (n/abs(alpha))*(1/(n-1))*exp((-abs(alpha)^2)/2)
	return output

def D(alpha):
	output = sqrt(pi/2)*(1+erf(abs(alpha)/sqrt(2)))
	return output

def cball(x,alpha,n,xbar,sigma):
	#uses the functions above to create an output from input x
	#works for single values but assumed x will be a time series
	fx = rep(NA, length(x))
	for (i in 1:length(x)):
		if (((x[i]-xbar)/sigma) > -alpha):
			fx[i] = N(sigma,C(alpha,n),D(alpha,n))*exp((-(x[i]-xbar)^2)/(2*sigma^2))
		if (((x[i]-xbar)/sigma) <= -alpha):
			fx[i] = N(sigma,C(alpha,n),D(alpha,n))*A(alpha,n)*(B(alpha,n)-(x[i]-xbar)/sigma)^(-n)
	return fx


##use nonlinear least squares to estimate parameters alpha, n, xbar and sigma
##cball fails when using nls() so use nlsLM() from package below
##will fit this using a full time series

if (!require("minpack.lm")) install.packages("minpack.lm")
library(minpack.lm)

def cball_parfit(x, alphastart=1.5, nstart=150, xbarstart=8, sigmastart=2, allparams=FALSE):
	#x is original offset50 time series
	# #there are specified options for the inital values in the nls()
	#these can be user defined to
	mean_ts = mean_annual_ts(x)
	norm_ts = reverse_normalise_ts(mean_ts)
	cball_fit = nlsLM(norm_ts ~ cball(1:length(norm_ts),alpha,n,xbar,sigma), start=list(alpha=alphastart, n=nstart, xbar=xbarstart, sigma=sigmastart))

	#return alpha and n as these are most important as default
	#can specify with allparams=TRUE to get all parameters out
	if (allparams==FALSE):
		return coef(cball_fit)[1:2]
	if (allparams==TRUE):
		return coef(cball_fit)
