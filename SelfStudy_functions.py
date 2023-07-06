"""
Title: A causal econometrics study
Author: Jonas Huwyler 
"""

# import modules
import sys
import pandas as pd
import numpy as np
import statsmodels.api as sm
from numpy.random import multivariate_normal
from numpy.random import binomial
from numpy.random import normal


# ----------------------------------------------------------------------------#
# Data Generating Processes
# ----------------------------------------------------------------------------#

# DGP 1
def dgp1(n, b, m1, m2, cov1, cov2, g):
    """
    First data generating process.

    Input
    ----------
    n:    Sample size
    b:    True beta vector
    m1:   Means of confounders
    m2:   Means of covariates
    cov1: Covariance matrix of confounders
    cov2: Covariance matrix of covariates
    g:    Coefficients through which confounders influence treatment

    Output
    -------
    x: Matrix of dependent variables
    y: Independent variable

    """
    X  = multivariate_normal(m1, cov1, n)          # confounder matrix
    V  = multivariate_normal(m2, cov2, n)          # covariate matrix
    pr = 1/(1 + np.exp(-(X @ g)))                  # treatment probability influened by confounders
    T  = np.asmatrix(binomial(1, pr, n)).T         # binary treatment variable
    I  = np.asmatrix(np.ones(n)).T                 # intercept
    x  = np.concatenate([I, X**2, V, T], axis = 1) # concatenate all independent variables to matrix
    y  = (x @ b) + normal(0, 1, n)                 # data generating process with standard-normal noise
    x  = pd.DataFrame(x)                           # convert independent variable to dataframe
    y  = pd.DataFrame(y)                           # convert ependent variable to dataframe                           
    y  = y.iloc[0]                                 # makes furthre use of the y-variable more convenient
    return (x, y)

# DGP 2
def dgp2(n, p, b, m1, m2, cov1, cov2):
    """
    Second data generating process.

    Input
    ----------
    n:    Sample size
    b:    True beta vector
    m1:   Means of confounders
    m2:   Means of covariates
    cov1: Covariance matrix of confounders
    cov2: Covariance matrix of covariates
    p:    Exogenously fixed treatment probability

    Output
    -------
    x: Matrix of dependent variables
    y: True independent variable
    """
    X = multivariate_normal(m1, cov1, n)       # confounder matrix
    V = multivariate_normal(m2, cov2, n)       # covariate matrix
    T = np.asmatrix(binomial(1, p, n)).T       # binary treatment variable with fixed p
    I = np.asmatrix(np.ones(n)).T              # intercept
    x = np.concatenate([I, X, V, T], axis = 1) # concatenate all independent variables to matrix
    y = (x @ b) + normal(0, 1, n)              # data generating process with standard-normal noise
    x = pd.DataFrame(x)                        # convert independent variable to dataframe
    y = pd.DataFrame(y)                        # convert dependent variable to dataframe
    y = y.iloc[0]                              # makes furthre use of the y-variable more convenient
    return (x, y)

# DGP 3
def dgp3(n, b_3, m1, m2, cov1, cov2, g):
    """
    Third data generating process.

    Input
    ----------
    n:    Sample size
    b_3:  True beta vector
    m1:   Means of confounders
    m2:   Means of covariates
    cov1: Covariance matrix of confounders
    cov2: Covariance matrix of covariates
    g:    Coefficients through which confounders influence treatment

    Output
    -------
    x: Matrix of dependent variables
    y: True independent variable

    """
    X  = multivariate_normal(m1, cov1, n)    # confounder matrix
    V  = multivariate_normal(m2, cov2, n)    # covariate matrix
    pr = 1/(1 + np.exp(-(X @ g)))            # treatment probability influened by confounders
    T  = np.asmatrix(binomial(1, pr, n)).T   # binary treatment variable
    I  = np.asmatrix(np.ones(n)).T           # intercept
    x  = np.concatenate([I, V, T], axis = 1) # concatenate all independent variables to matrix
    y  = (x @ b_3) + normal(0, 1, n)         # data generating process with standard-normal noise
    x  = pd.DataFrame(x)                     # convert in dependent variable to dataframe
    y  = pd.DataFrame(y)                     # convert dependent variable to dataframe
    y  = y.iloc[0]                           # makes furthre use of the y-variable more convenient
    return (x, y)


# ----------------------------------------------------------------------------#
# OLS
# ----------------------------------------------------------------------------#

# OLS Betas
def betas_ols(y, x):
    """
    Returns the beta estimates of an OLS regression.

    Input
    ----------
    y: True independent variable
    x: Matrix of dependent variables

    Output
    -------
    betas: Estimated betas of the OLS regression
    """    
    betas = np.linalg.pinv(x.T @ x) @ x.T @ y  # formula for beta coefficients
    return(betas)

# OLS Simulation
def ols_sim(t, m, n, p, b, b_3, m1, m2, cov1, cov2, g):
    """
    Simulation for the OLS estimator.

    Input
    ----------
    t:    The true ATE from the DGP
    m:    Number of iterations in monte carlo
    n:    Sample size
    p:    Fixed probability (required for DGP2)
    b:    True beta vector (for DGP1 and DGP2)
    b_3:  True beta vector (for DGP3)
    m1:   Means of confounders
    m2:   Means of covariates
    cov1: Covariance matrix of confounders
    cov2: Covariance matrix of covariates
    g:    Coefficients through which confounders influence treatment

    Output
    -------
    result: Stores bias, variance and MSE
    """
    ate1 = np.empty(m+1) # empty vector to store ATE's
    ate2 = np.empty(m+1) # empty vector to store ATE's
    ate3 = np.empty(m+1) # empty vector to store ATE's
    
    for i in range(m): # Monte Carlo Simulation
        
        (x1, y1) = dgp1(n, b, m1, m2, cov1, cov2, g)      # DGP1
        (x2, y2) = dgp2(n+1000, p, b, m1, m2, cov1, cov2) # DGP2
        (x3, y3) = dgp3(n, b_3, m1, m2, cov1, cov2, g)    # DGP3
        
        b1 = betas_ols(y1, x1) # estimate betas of DGP1
        b2 = betas_ols(y2, x2) # estimate betas of DGP2
        b3 = betas_ols(y3, x3) # estimate betas of DGP3
        
        ypred1 = x1 @ b1 # predict y values for DGP1
        ypred2 = x2 @ b2 # predict y values for DGP2
        ypred3 = x3 @ b3 # predict y values for DGP3
    
        # ATE estimation by mean differences
        ypred1_0 = ypred1[x1.loc[:, 5] == 0] # prepare control group from DGP1
        ypred2_0 = ypred2[x2.loc[:, 5] == 0] # prepare control group from DGP2
        ypred3_0 = ypred3[x3.loc[:, 3] == 0] # prepare control group from DGP3
        
        ypred1_1 = ypred1[x1.loc[:, 5] == 1] # prepare treatment group from DGP1
        ypred2_1 = ypred2[x2.loc[:, 5] == 1] # prepare treatment group from DGP2
        ypred3_1 = ypred3[x3.loc[:, 3] == 1] # prepare treatment group from DGP3
        
        ate1_i  = ypred1_1.mean() - ypred1_0.mean() # ATE of DGP1
        ate1[i] = ate1_i                            # store results
        ate2_i  = ypred2_1.mean() - ypred2_0.mean() # ATE of DGP2
        ate2[i] = ate2_i                            # store results
        ate3_i  = ypred3_1.mean() - ypred3_0.mean() # ATE of DGP3
        ate3[i] = ate3_i                            # store results        
        
    bias1 = abs(ate1.mean() - t) # bias of ATE estimator for DGP1
    bias2 = abs(ate2.mean() - t) # bias of ATE estimator for DGP2
    bias3 = abs(ate3.mean() - t) # bias of ATE estimator for DGP3
    
    var1  = np.mean((ate1 - ate1.mean())**2) # variance of ATE estimator for DGP1
    var2  = np.mean((ate2 - ate2.mean())**2) # variance of ATE estimator for DGP2
    var3  = np.mean((ate3 - ate3.mean())**2) # variance of ATE estimator for DGP3
    
    mse1  = np.mean((ate1 - t)**2) # mse of ATE estimator for DGP1
    mse2  = np.mean((ate2 - t)**2) # mse of ATE estimator for DGP2
    mse3  = np.mean((ate3 - t)**2) # mse of ATE estimator for DGP3
    
    # store results in list of lists
    result = [[bias1, var1, mse1],
              [bias2, var2, mse2],
              [bias3, var3, mse3]]
    
    return(result)


# ----------------------------------------------------------------------------#
# Inverse Probability Weighting (IPW)
# ----------------------------------------------------------------------------#

# IPW  ATE
def ipw_ate(x, y, d): # (in the manner of PC2 tutorial DA II)
    """
    ATE estimation function for IPW.

    Input
    ----------
    x: Matrix of dependent variables
    y: Array of independent variable
    d: Array of the treatment dummy
        
    Output
    -------
    ate: Stores bias, variance and MSE.
    """
    ps  = sm.Logit(d, x).fit(disp=0).predict()             # formula to calculate the propensity score
    ate = np.mean((d * y) / ps - ((1 - d) * y) / (1 - ps)) # formula to calculat the ATE
    
    return ate
    
# IPW Simulation    
def ipw_sim(t, m, n, p, b, b_3, m1, m2, cov1, cov2, g):
    """
    Simulation for the IPW estimator.
    
    Input
    ----------
    t:    The true ATE from the DGP
    m:    Number of iterations in monte carlo
    n:    Sample size
    p:    Fixed probability (required for DGP2)
    b:    True beta vector (for DGP1 and DGP2)
    b_3:  True beta vector (for DGP3)
    m1:   Means of confounders
    m2:   Means of covariates
    cov1: Covariance matrix of confounders
    cov2: Covariance matrix of covariates
    g:    Coefficients through which confounders influence treatment

    Output
    -------
    result: Stores bias, variance and MSE
    """
    ate1 = np.empty(m+1) # empty vector to store ATE's
    ate2 = np.empty(m+1) # empty vector to store ATE's
    ate3 = np.empty(m+1) # empty vector to store ATE's
    
    for i in range(m): # Monte Carlo Simulation
        
        (x1, y1) = dgp1(n, b, m1, m2, cov1, cov2, g)      # DGP1
        (x2, y2) = dgp2(n+1000, p, b, m1, m2, cov1, cov2) # DGP2
        (x3, y3) = dgp3(n, b_3, m1, m2, cov1, cov2, g)    # DGP3
        
        ate1[i] = ipw_ate(x1.iloc[:, 0:4], y1, x1.iloc[:, 5]) # ATE of DGP1
        ate2[i] = ipw_ate(x2.iloc[:, 0:4], y2, x2.iloc[:, 5]) # ATE of DGP2
        ate3[i] = ipw_ate(x3.iloc[:, 0:2], y3, x3.iloc[:, 3]) # ATE of DGP3
        
    bias1 = abs(ate1.mean() - t) # bias of ATE estimator for DGP1
    bias2 = abs(ate2.mean() - t) # bias of ATE estimator for DGP2
    bias3 = abs(ate3.mean() - t) # bias of ATE estimator for DGP3
    
    var1  = np.mean((ate1 - ate1.mean())**2)   # variance of ATE estimator for DGP1
    var2  = np.mean((ate2 - ate2.mean())**2)   # variance of ATE estimator for DGP2
    var3  = np.mean((ate3 - ate3.mean())**2)   # variance of ATE estimator for DGP3
    
    mse1  = np.mean((ate1 - t)**2)     # mse of ATE estimator for DGP1
    mse2  = np.mean((ate2 - t)**2)     # mse of ATE estimator for DGP2
    mse3  = np.mean((ate3 - t)**2)     # mse of ATE estimator for DGP3
    
    # store results in list
    result = [[bias1, var1, mse1],
              [bias2, var2, mse2],
              [bias3, var3, mse3]]
    
    return(result)

# ----------------------------------------------------------------------------#
# ----------------------------------------------------------------------------#

# define an Output class for simultaneous console - file output
class Output():
    """Output class for simultaneous console/file output."""

    def __init__(self, path, name):

        self.terminal = sys.stdout
        self.output = open(path + name + ".txt", "w")

    def write(self, message):
        """Write both into terminal and file."""
        self.terminal.write(message)
        self.output.write(message)

    def flush(self):
        """Python 3 compatibility."""
