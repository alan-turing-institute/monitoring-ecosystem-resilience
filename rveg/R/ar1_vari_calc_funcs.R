#functions assume that there is a time series 'x' to calculate the signals on
#wl='half' is the default such that a window length equal to half the length of
#the time series is used.
#I have also included a smoothing function option for detrending but this is
#off by default.
#This should not be confused with the annual cycle I was removing in the plot,
#this is more after removing a drift perhaps, and will also be used in robustness
#testing when we get to that.

ar1_ts_calc <- function(x, wl='half', bw='none') {
	l <- length(x)
	if (wl == 'half') {
		wl <- floor(l/2)
				}
	if (is.numeric(bw)) {
		#ksmooth has an x and a y output, the x output would be c(1:l)
		x_d <- x - ksmooth(c(1:l), x, bandwidth=bw, x.points=c(1:l))$y
	} else {
		x_d <- x
		}
	ar1 <- rep(NA, (l-wl))
	for (z in 1:(l-wl)) {
		#The '-1' in the indexing ensures the window is the correct length (wl long)
		#aic=FALSE forces an output, even if there's not a significant ar1 value
		#order.max=1 means it only looks for ar1 and not ar2/3/4 etc.
		#The ar value from the output is what we want
		ar1[z] <- ar.ols(x_d[z:(z+wl-1)], aic=FALSE, order.max=1)$ar
				}
	return(ar1)
						}


vari_ts_calc <- function(x, wl='half', bw='none') {
	l <- length(x)
	if (wl == 'half') {
		wl <- floor(l/2)
				}
	if (is.numeric(bw)) {
		#ksmooth has an x and a y output, the x output would be c(1:l)
		x_d <- x - ksmooth(c(1:l), x, bandwidth=bw, x.points=c(1:l))$y
	} else {
		x_d <- x
		}
	ar1 <- rep(NA, (l-wl))
	for (z in 1:(l-wl)) {
		#Some people use standard deviation rather than variance so this is a personal choice
		vari[z] <- var(x_d[z:(z+wl-1)], na.rm=T)
				}
	return(vari)
						}


#Kendall tau gives us the tendency of the time series, it's just a rank correlation test
#with one variable being time (or the vector 1 to the length of the time series)
#1 means always increasing, -1 always decreasing and 0 no overal trend
#(for example the images I showed at the end of my plots on Wednesday showed variance
#decreases everywhere)

kend_tau <- function(x) {
	tau <- cor.test(c(1:length(x)), x, method='kendall', na.rm=T)$estimate
	return(tau)
				}






	