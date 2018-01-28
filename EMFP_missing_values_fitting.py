"""
This module is for implementing
Expectation maximization with flexible probabilities for missing values under
multivariate t distribution

Yu Mu

Jan. 22th, 2018
"""

import numpy as np
from SmartInverse import SmartInverse


def EM_FP_missing_values_fitting(data, FP, nu, tol, maxiter=10**5):
    """
    Parameters
    ----------
    data: stock returns; (numStocks, numDays)
    FP: flexible probabilities; (numDays)
    nu: tail parameter; float
    tol: tolerance for convergence criterion; float
    maxiter: maximum iteration
    """

    # initialization
    numStocks, numDays = data.shape
    mu_MLFP = np.zeros([numStocks, 1])
    sigma2_MLFP = np.zeros([numStocks, numStocks, 1])

    # calculate sufficient stats with flexible probability
    mu_MLFP = data.dot(FP[:, np.newaxis])
    centered_data = data - mu_MLFP
    sigma2_MLFP[:, :, 0] = centered_data.dot(np.diag(FP)).dot(centered_data.T)

    # run algorithm
    error = [10**6, 10**6]
    k = 0
    while sum(error>tol) >= 1 and k < maxiter:
        k += 1
        # update weights


    print "done"
