"""
Title: A causal econometrics study
Author: Jonas Huwyler 
"""

# import modules
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


# set working directory
PATH = '/Users/jones/Desktop/FS21/Data Analytics II/Self-Study/My_selfstudy/Jonas_Huwyler_16-610-958_SelfStudy'
sys.path.append(PATH)

# load own functions
import SelfStudy_functions as pc

# define the name for the output file
OUTPUT_NAME = 'PC_output'

# save the console output in parallel to a txt file
orig_stdout = sys.stdout
sys.stdout = pc.Output(path=PATH, name=OUTPUT_NAME)

# set the seed for replicability
np.random.seed(123)

# ----------------------------------------------------------------------------#
# Initializing Parameters
# ----------------------------------------------------------------------------#

# Data Generating Process Parameters
b   = [1.0, 0.3, 0.6, 0.6, 0.7, 1.8] # true beta coefficients (intercept + confounders + covariates + treatment dummy)
b_3 = [1.0, 0.6, 0.7, 1.8]           # true beta coefficients for DGP3
g   = [-0.35, -0.30]                 # coefficients for confounders influencing the treatment probability
t   = b[5]                           # true treatment effect 
p   = 0.01                           # fixed treatment probability at 1% (for DGP2)
       
m1 = [1.0, 0.5] # confounder means
m2 = [0.5, 0.3] # covariate means

cov1 = [[1.0, 0.1], 
        [0.3, 1.0]] # confounder covariance matrix
cov2 = [[1.0, 0.3], 
        [0.4, 1.0]] # covariates covariance matrix

# Simulation parameters
m     = 100                      # number of monte carlo simulations
i     = -1                       # setting first index negativ to start with index 0
start = 40                       # first sample size
end   = 540                      # last sample size
step  = 20                       # step size
rows  = np.int((end-start)/step) # rows of empty vectors
samp  = 1000                     # sample adjustment for DGP2


# ----------------------------------------------------------------------------#
# Simulation 
# ----------------------------------------------------------------------------#

# OLS Simulation
sim1_ols = np.zeros((rows, 3)) # empty matrices to store results
sim2_ols = np.zeros((rows, 3)) # empty matrices to store results
sim3_ols = np.zeros((rows, 3)) # empty matrices to store results

for n in range(start, end, step): # loop through sample sizes
    
    i = i+1 # run index
    
    sim1_ols[i] = pc.ols_sim(t, m, n, p, b, b_3, m1, m2, cov1, cov2, g)[0] # calling simulation function for DGP1
    sim2_ols[i] = pc.ols_sim(t, m, n, p, b, b_3, m1, m2, cov1, cov2, g)[1] # calling simulation function for DGP2
    sim3_ols[i] = pc.ols_sim(t, m, n, p, b, b_3, m1, m2, cov1, cov2, g)[2] # calling simulation function for DGP3

# convert to dataframes
ols1 = pd.DataFrame(sim1_ols, columns = ["Bias", "Variance", "MSE"]) # store results in dataframe for DGP1
ols2 = pd.DataFrame(sim2_ols, columns = ["Bias", "Variance", "MSE"]) # store results in dataframe for DGP2
ols3 = pd.DataFrame(sim3_ols, columns = ["Bias", "Variance", "MSE"]) # store results in dataframe for DGP3


# IPW Simulation
sim1_ipw = np.zeros((rows, 3)) # empty matrices to store results
sim2_ipw = np.zeros((rows, 3)) # empty matrices to store results
sim3_ipw = np.zeros((rows, 3)) # empty matrices to store results

i = -1 # reset index to -1

for n in range(start, end, step): # loop through sample sizes 
    
    i = i+1 # run index
    
    # calling simulations
    sim1_ipw[i] = pc.ipw_sim(t, m, n, p, b, b_3, m1, m2, cov1, cov2, g)[0] # calling simulation function for DGP1
    sim2_ipw[i] = pc.ipw_sim(t, m, n, p, b, b_3, m1, m2, cov1, cov2, g)[1] # calling simulation function for DGP2
    sim3_ipw[i] = pc.ipw_sim(t, m, n, p, b, b_3, m1, m2, cov1, cov2, g)[2] # calling simulation function for DGP3
    
# convert to dataframes
ipw1 = pd.DataFrame(sim1_ipw, columns = ["Bias", "Variance", "MSE"]) # store results in dataframe for DGP1
ipw2 = pd.DataFrame(sim2_ipw, columns = ["Bias", "Variance", "MSE"]) # store results in dataframe for DGP2
ipw3 = pd.DataFrame(sim3_ipw, columns = ["Bias", "Variance", "MSE"]) # store results in dataframe for DGP3


# ----------------------------------------------------------------------------#
# Visualization
# ----------------------------------------------------------------------------#

# plotting the performance measures per estimator 
ols1.plot(title = "DGP1: OLS" , marker='o', linewidth=1, markersize=2) # OLS estimate for DGP1
ols2.plot(title = "DGP2: OLS" , marker='o', linewidth=1, markersize=2) # OLS estimate for DGP2
ols3.plot(title = "DGP3: OLS" , marker='o', linewidth=1, markersize=2) # OLS estimate for DGP3

ipw1.plot(title = "DGP1: IPW" , marker='o', linewidth=1, markersize=2) # IPW estimate for DGP1
ipw2.plot(title = "DGP2: IPW" , marker='o', linewidth=1, markersize=2) # IPW estimate for DGP2
ipw3.plot(title = "DGP3: IPW" , marker='o', linewidth=1, markersize=2) # IPW estimate for DGP3

# plots to compare sinlge performance measures among estimators
BIA1 = pd.concat([ols1["Bias"], ipw1["Bias"]], axis=1, keys = ["OLS Bias", "IPW Bias"]) # data frame to support plots
BIA2 = pd.concat([ols2["Bias"], ipw2["Bias"]], axis=1, keys = ["OLS Bias", "IPW Bias"]) # data frame to support plots
BIA3 = pd.concat([ols3["Bias"], ipw3["Bias"]], axis=1, keys = ["OLS Bias", "IPW Bias"]) # data frame to support plots

MSE1 = pd.concat([ols1["MSE"], ipw1["MSE"]], axis=1, keys = ["OLS MSE", "IPW MSE"]) # data frame to support plots
MSE2 = pd.concat([ols2["MSE"], ipw2["MSE"]], axis=1, keys = ["OLS MSE", "IPW MSE"]) # data frame to support plots
MSE3 = pd.concat([ols3["MSE"], ipw3["MSE"]], axis=1, keys = ["OLS MSE", "IPW MSE"]) # data frame to support plots

VAR1 = pd.concat([ols1["Variance"], ipw1["Variance"]], axis=1, keys = ["OLS Variance", "IPW Variance"]) # data frame to support plots
VAR2 = pd.concat([ols2["Variance"], ipw2["Variance"]], axis=1, keys = ["OLS Variance", "IPW Variance"]) # data frame to support plots
VAR3 = pd.concat([ols3["Variance"], ipw3["Variance"]], axis=1, keys = ["OLS Variance", "IPW Variance"]) # data frame to support plots

# set plot style
matplotlib.style.use('seaborn-notebook')
plt.figure() #indicate that a new plot begins

# DGP1 performance measures 
# Bias
BIA1["OLS Bias"].plot(legend=True)                    # plot OLS bias
BIA1["IPW Bias"].plot(legend=True)                    # plot IPW bias
plt.xlabel('Sample size')                             # set x-label
plt.ylabel('Bias')                                    # set y-label
plt.xticks(list(range(1, rows+1, 2)), 
           list(range(start+step, end+step, step*2))) # set correct x-ticks
plt.title('DGP1: Bias')                               # set plot title
plt.show()                                            # show plot

# Variance
VAR1["OLS Variance"].plot(legend=True)                # plot OLS variance
VAR1["IPW Variance"].plot(legend=True)                # plot IPW variance
plt.xlabel('Sample size')                             # set x-label
plt.ylabel('Variance')                                # set y-label
plt.xticks(list(range(1, rows+1, 2)), 
           list(range(start+step, end+step, step*2))) # set correct x-ticks
plt.title('DGP1: Variance')                           # set plot title
plt.show()                                            # show plot

# MSE
MSE1["OLS MSE"].plot(legend=True)                     # plot OLS MSE
MSE1["IPW MSE"].plot(legend=True)                     # plot IPW MSE
plt.xlabel('Sample size')                             # set x-label
plt.ylabel('MSE')                                     # set y-label
plt.xticks(list(range(1, rows+1, 2)), 
           list(range(start+step, end+step, step*2))) # set correct x-ticks
plt.title('DGP1: MSE')                                # set plot title
plt.show()                                            # show plot

# DGP2 performance measures 
# Bias
BIA2["OLS Bias"].plot(legend=True)                              # plot OLS bias
BIA2["IPW Bias"].plot(legend=True)                              # plot IPW bias
plt.xlabel('Sample size')                                       # set x-label
plt.ylabel('Bias')                                              # set y-label
plt.xticks(list(range(1, rows+1, 2)), 
           list(range(start+step+1000, end+step+1000, step*2))) # set correct x-ticks
plt.title('DGP2: Bias')                                         # set plot title
plt.show()                                                      # show plot

# Variance
VAR2["OLS Variance"].plot(legend=True)                          # plot OLS variance
VAR2["IPW Variance"].plot(legend=True)                          # plot IPW variance
plt.xlabel('Sample size')                                       # set x-label
plt.ylabel('Variance')                                          # set y-label
plt.xticks(list(range(1, rows+1, 2)), 
           list(range(start+step+1000, end+step+1000, step*2))) # set correct x-ticks
plt.title('DGP2: Variance')                                     # set plot title
plt.show()                                                      # show plot

# MSE
MSE2["OLS MSE"].plot(legend=True)                               # plot OLS MSE
MSE2["IPW MSE"].plot(legend=True)                               # plot IPW MSE
plt.xlabel('Sample size')                                       # set x-label
plt.ylabel('MSE')                                               # set y-label
plt.xticks(list(range(1, rows+1, 2)), 
           list(range(start+step+1000, end+step+1000, step*2))) # set correct x-ticks
plt.title('DGP2: MSE')                                          # set plot title
plt.show()                                                      # show plot

# DGP3 performance measures 
# Bias
BIA3["OLS Bias"].plot(legend=True)                    # plot OLS MSE
BIA3["IPW Bias"].plot(legend=True)                    # plot IPW MSE
plt.xlabel('Sample size')                             # set x-label
plt.ylabel('Bias')                                    # set y-label
plt.xticks(list(range(1, rows+1, 2)), 
           list(range(start+step, end+step, step*2))) # set correct x-ticks
plt.title('DGP3: Bias')                               # set plot title
plt.show()                                            # show plot

# Variance
VAR3["OLS Variance"].plot(legend=True)                # plot OLS variance
VAR3["IPW Variance"].plot(legend=True)                # plot IPW variance
plt.xlabel('Sample size')                             # set x-label
plt.ylabel('Variance')                                # set y-label
plt.xticks(list(range(1, rows+1, 2)), 
           list(range(start+step, end+step, step*2))) # set correct x-ticks
plt.title('DGP3: Variance')                           # set plot title
plt.show()                                            # show plot

# MSE
MSE3["OLS MSE"].plot(legend=True)                     # plot OLS MSE
MSE3["IPW MSE"].plot(legend=True)                     # plot IPW MSE
plt.xlabel('Sample size')                             # set x-label
plt.ylabel('MSE')                                     # set y-label
plt.xticks(list(range(1, rows+1, 2)), 
           list(range(start+step, end+step, step*2))) # set correct x-ticks
plt.title('DGP3: MSE')                                # set plot title
plt.show()                                            # show plot

# close the output file
sys.stdout.output.close()
sys.stdout = orig_stdout

# end of the Self-Study #
