"""
This module is for fitting randomly missing values in
our data, using EM algorithm with flexible probabilities
under student t assumption with fixed degree of freedom nu

Reference:
Liu & Rubin 1995, "ML estimation of the t distribution using EM and
its extensions, ECM and ECME"
Meucci's bootcamp materials

Yu Mu
Jan. 23th, 2018
"""
import numpy as np
from scipy.io import loadmat
import os
import os.path
from CONFIG import GLOBAL_DB
from ARPM_utils import struct_to_dict
from RollPrices2YieldToMat import RollPrices2YieldToMat
from intersect_matlab import intersect
import EMalgorithmFP


class Fit_Randomly_Missing_Values(object):

    def __init__(self, data, FP, nu, tol):
        """
        Parameters
        ----------
        data: stock returns; (numStocks, numDays)
        FP: flexible probabilities; (numDays)
        nu: tail parameter; float
        tol: tolerance of convergence criterio; float
        """
        self._data = data
        self._FP = FP
        self._nu = nu
        self._tol = tol
        self._numStocks, self._numDays = self._data.shape

    def FP_mean_cov(self, FP, data):
        """
        calculate sample mean and covariance subject to flexible probability
        Returns
        -------
        mu: flexible sample mean; (numStocks, 1)
        sigma2: flexible sample covariance; (numStocks, numStocks)
        """
        numStocks, numDays = data.shape
        mu = data.dot(FP)[:, np.newaxis]
        centered_data = data - mu
        sigma2 = (centered_data*FP).dot(centered_data.T)
        sigma2 = (sigma2 + sigma2.T)/2                     # eliminate numerical error and make covariance symmetric
        return mu, sigma2

    def run_EM(self):
        I = np.isnan(self._data)
        # get truncated data with no missing values by removing the columns containing missing values
        trunc_data = self._data[:, np.sum(I, axis=0)==0]

        FP_adjusted = self._FP[np.sum(I, axis=0)==0]
        FP_adjusted = FP_adjusted/np.sum(FP_adjusted)
        FP_mu, FP_sigma2 = self.FP_mean_cov(FP_adjusted, trunc_data)
        weights = np.ones(self._numDays)
        Error = np.ones(2)*10**6               # two tolerance to check convergence of mu and sigma2
        # start main loop
        j = 0
        gamma = {}
        while any(Error>tol):
            j += 1
            eps = np.zeros([self._numStocks, self._numDays])
            for t in range(self._numDays):
                gamma[t] = np.zeros([self._numStocks, self._numStocks])

                na = []
                for i in range(self._numStocks):
                    if np.isnan(self._data[i, t]):
                        na = np.r_[na, i]

                a = np.arange(self._numStocks)
                if isinstance(na, np.ndarray):
                    if na.size > 0:
                        mask = np.ones(a.shape, dtype=bool)
                        na = list(map(int, na))
                        mask[na] = False
                        a = a[mask]
                A = self._numStocks - len(na)
                eps[a, t] = self._data[a, t]
                eps[na, t] = self._data[na, t]
                # step 1:
                # update weights
                import pdb; pdb.set_trace()  # breakpoint f0bdd6b8 //
                inv_sigma2 = np.linalg.inv(FP_sigma2[np.ix_(a, a, [j-1])].squeeze())
        import pdb; pdb.set_trace()  # breakpoint 9785518a //
        print "done"


if __name__ == '__main__':

    db = loadmat(os.path.join(GLOBAL_DB, 'db_SwapCurve.mat'), squeeze_me=True)

    DF_Rolling = struct_to_dict(db['DF_Rolling'])

    dates = DF_Rolling.Dates

    tau = np.array([1, 2, 3, 5, 7, 8, 10])

    y, _ = RollPrices2YieldToMat(DF_Rolling.TimeToMat, DF_Rolling.Prices)

    # select zero rates
    _, _, tauIndices = intersect(tau, DF_Rolling.TimeToMat)
    y_tau = y[tauIndices, :]

    # daily changes (last 700 obs available)
    i_ = len(tau)
    t_ = 700

    dy = np.diff(y_tau, 1, axis=1)
    dy = dy[:, -t_:]
    dates = dates[-t_:]
    dy[[1, 3], 30:150] = np.NaN

    # Maximum likelihood with Flex. Probs. - complete series
    nu = 4
    lam = 0.002
    flex_prob = np.exp((-lam * np.arange(t_, 1 + -1, -1)))
    flex_prob = flex_prob / sum(flex_prob)
    flex_prob = flex_prob.reshape(1, -1)
    tol = 10 ** -6
    import pdb; pdb.set_trace()  # breakpoint 6c54608e //

    # inst = Fit_Randomly_Missing_Values(dy, flex_prob[0, :], nu, tol)
    # inst.run_EM()
    mu, sigma2 = EMalgorithmFP.EMalgorithmFP(dy, flex_prob, nu, tol)
    import pdb; pdb.set_trace()  # breakpoint 4377c5fd //
    print "done"
