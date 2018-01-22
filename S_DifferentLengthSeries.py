''' This script illustrates estimation on time series with different len.
 In particular, it performs a comparison between the estimators obtained applying the
 Maximum likelihood with Flexible Probabilities algorithm for series of different
 len (DLFP) and the MLFP estimators on the truncated series (both under Student t assumption).
 The estimation is performed on the daily changes of the [1, 2, 3, 5, 7, 8, 10] years swap
 rates. The figure prints the results relative to the 2 and 5 years
 rates
'''

## For details, see [here](/lab/redirect.php?permalink=ExerDiffLength_copy(1)).
## Prepare the environment

import os
import os.path as path
import sys

sys.path.append(path.join(path.dirname(path.dirname(path.abspath('.'))), 'Functions'))

import numpy as np
from numpy import arange, array, zeros, percentile, diff, exp, sum as npsum

from scipy.io import loadmat

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, legend, xlim, ylim, scatter, ylabel, \
    xlabel
import matplotlib.dates as mdates

plt.style.use('seaborn')
# %config InlineBackend.figure_format = 'svg'

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from intersect_matlab import intersect
from ARPM_utils import save_plot, struct_to_dict, date_mtop
from MaxLikelihoodFPLocDispT import MaxLikelihoodFPLocDispT
from RollPrices2YieldToMat import RollPrices2YieldToMat
from PlotTwoDimEllipsoid import PlotTwoDimEllipsoid
from DiffLengthMLFP import DiffLengthMLFP
from ColorCodedFP import ColorCodedFP
import pdb; pdb.set_trace()  # breakpoint 0b1e6fd3 //


## Upload dataset

try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_SwapCurve'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_SwapCurve'), squeeze_me=True)

DF_Rolling = struct_to_dict(db['DF_Rolling'])

dates = DF_Rolling.Dates

## Compute the swap rates daily changes and select the last 700 available observations

# times to maturity (in years)
tau = array([1, 2, 3, 5, 7, 8, 10])

# zero rates from rolling pricing
y,_ = RollPrices2YieldToMat(DF_Rolling.TimeToMat, DF_Rolling.Prices)

# select zero rates
_, _, tauIndices = intersect(tau, DF_Rolling.TimeToMat)
y_tau = y[tauIndices, :]

# daily changes (last 700 obs available)
i_ = len(tau)
t_ = 700

dy = diff(y_tau, 1, axis=1)
dy = dy[:, -t_:]
dates = dates[-t_:]

## Maximum likelihood with Flex. Probs. - complete series

nu = 4
lam = 0.002
flex_prob = exp((-lam * arange(t_, 1 + -1, -1)))
flex_prob = flex_prob / npsum(flex_prob)
flex_prob = flex_prob.reshape(1,-1)
tol = 10 ** -6
mu_all, s2_all, _ = MaxLikelihoodFPLocDispT(dy, flex_prob.reshape(1,-1), nu, tol, 1)
mu_all = mu_all.reshape(-1,1)

epsi_25 = dy[[1, 3], :]
mu_all_25 = mu_all[[1, 3]]
s2_all_25 = s2_all[np.ix_([1, 3], [1, 3])]

## Series of different len: drop the first 300 observations from the 2yr and 5yr series

r = 300
epsi = dy
epsi[[1, 3], :r] = np.NaN

## Maximum likelihood with Flex. Probs. - different len

mu_DLFP, s2_DLFP = DiffLengthMLFP(epsi, flex_prob.reshape(1,-1), nu, tol)
mu_DLFP_25 = mu_DLFP.reshape(-1,1)[[1, 3]]
s2_DLFP_25 = s2_DLFP[np.ix_([1, 3],[1, 3])]

## Maximum likelihood with Flex. Probs. - truncated series

flex_prob_trunc = flex_prob[[0],r:] / sum(flex_prob[0,r:])
mu_trunc, s2_trunc, _ = MaxLikelihoodFPLocDispT(epsi[:, r:], flex_prob_trunc, 4, 10 ** -6, 1)
mu_trunc_25 = mu_trunc.reshape(-1,1)[[1, 3]]
s2_trunc_25 = s2_trunc[np.ix_([1, 3], [1, 3])]

## Figure

blue = 'b'
orange = [0.95, 0.35, 0]
green = [0, 0.7, 0.3]
# scatter colormap and colors
CM, C = ColorCodedFP(flex_prob, GreyRange=arange(0.25,0.91,0.01), Cmin=0, Cmax=1, ValueRange=[1, 0])
figure()
ax = plt.subplot2grid((4,1),(0,0),rowspan=3)
scatter(epsi_25[0], epsi_25[1], 20, c=C, marker='o',cmap=CM)
plt.axis([percentile(epsi_25[0], 5), percentile(epsi_25[0], 95),percentile(epsi_25[1], 5), percentile(epsi_25[1], 95)])
xlabel('2yr rate daily changes')
ylabel('5yr rate daily changes')
# Ellipsoids
ell1 = PlotTwoDimEllipsoid(mu_DLFP_25, s2_DLFP_25, 1, [], [], orange, 2, fig=plt.gcf())
ell = PlotTwoDimEllipsoid(mu_all_25, s2_all_25, 1, [], [], blue, 2, fig=plt.gcf())
ell2 = PlotTwoDimEllipsoid(mu_trunc_25, s2_trunc_25, 1, [], [], green, 2, fig=plt.gcf())
dr = plot(epsi_25[0,:r], epsi_25[1, :r],markersize=5,marker='o',markerfacecolor=[.9, .7, .7],linestyle='')  # Dropped observations
# leg
leg = legend(['MLFP - different len', 'MLFP - complete series','MLFP - truncated series','Dropped observations'])

# bottom plot: highlight missing observations in the dataset as white spots
ax = plt.subplot2grid((4,1),(3,0))
myFmt = mdates.DateFormatter('%d-%b-%y')
dates_dt = [date_mtop(i) for i in dates[49:t_:200]]
na = zeros((i_, t_))
na[5:8, :r] = 1  # na=1: not-available data (2y and 5y series are placed as last two entries)
ax.imshow(abs(na-1), extent=[mdates.date2num(dates_dt[0]),mdates.date2num(dates_dt[-1]),0, 8], aspect='auto')
ax.set_xticks(dates_dt)
ax.set_yticks([1,3])
ax.xaxis.set_major_formatter(myFmt)
xlim([min(dates_dt), max(dates_dt)])
ylim([0, i_])
ax.set_yticklabels([' 2yr', ' 5yr'])
ax.grid(False)
plt.tight_layout();
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
