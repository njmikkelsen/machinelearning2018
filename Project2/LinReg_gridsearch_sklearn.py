"""
This is a copy of 'LinReg_gridsearch.py', but implemented with sci-kit learn.
"""

import os
from IsingData import *
from misclib import *
from plotlib import LinRegPlot_MSE_R2
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

"""
This program performs a grid-search for categorising the regularization parameter's effect on
linear regression analyses of the 1-dimensional Ising data.
The MSE and R2 scores are plotted as functions of the regularization parameter lambda and
stored under './output/LinReg_MSE_sklearn.png' and './output/LinReg_R2_sklearn.png'
"""

# program parameters
N = 40     # dimensionality of spin system
J = 1      # coupling strength
M = 10000  # number of samples
r = 0.3    # training portion of samples

log_lmbda_min = -3   # log(min regularization)
log_lmbda_max = 5    # log(max regularization)
N_lmbda       = 100  # number of regularization parameters

# setup data & regularization
data  = IsingData.one_dim(N,J,M)
Lmbda = np.logspace(log_lmbda_min,log_lmbda_max,N_lmbda)
data.split(r)

# setup Regressor objects
OLS   = linear_model.LinearRegression(fit_intercept=False,normalize=False)
Ridge = linear_model.Ridge(           fit_intercept=False,normalize=False)
Lasso = linear_model.Lasso(           fit_intercept=False,normalize=False,max_iter=10000)

Ridge.MSE_train = np.zeros(N_lmbda)
Ridge.MSE_test  = np.zeros(N_lmbda)
Ridge.R2_train  = np.zeros(N_lmbda)
Ridge.R2_test   = np.zeros(N_lmbda)

Lasso.MSE_train = np.zeros(N_lmbda)
Lasso.MSE_test  = np.zeros(N_lmbda)
Lasso.R2_train  = np.zeros(N_lmbda)
Lasso.R2_test   = np.zeros(N_lmbda)

# Ordinary Least Squares
OLS.fit(data.X_train,data.Y_train)
predict_train = OLS.predict(data.X_train).flatten()
predict_test  = OLS.predict(data.X_test).flatten()
 
OLS.MSE_train = MSE(data.Y_train.flatten(),predict_train)
OLS.MSE_test  = MSE(data.Y_test.flatten(), predict_test)
OLS.R2_train  = R2( data.Y_train.flatten(),predict_train)
OLS.R2_test   = R2( data.Y_test.flatten(), predict_test)

# Ridge & Lasso
progress = progress_bar(2*N_lmbda)
for i,lmbda in enumerate(Lmbda):
  for LinReg in [Ridge,Lasso]:
    LinReg.set_params(alpha=lmbda)
    LinReg.fit(data.X_train,data.Y_train)
    predict_train = LinReg.predict(data.X_train)
    predict_test  = LinReg.predict(data.X_test)
    
    LinReg.MSE_train[i] = MSE(data.Y_train.flatten(),predict_train)
    LinReg.MSE_test[i]  = MSE(data.Y_test.flatten(), predict_test)
    LinReg.R2_train[i]  = R2( data.Y_train.flatten(),predict_train)
    LinReg.R2_test[i]   = R2( data.Y_test.flatten(), predict_test)
    
    progress.update()

# plot results
(fig1,ax1),(fig2,ax2) = LinRegPlot_MSE_R2(Lmbda,OLS,Ridge,Lasso)
if not os.path.isdir("./output/LinReg"): os.mkdir("./output/LinReg")
fig1.savefig("./output/LinReg/gridsearch_MSE_sklearn.png")
fig2.savefig("./output/LinReg/gridsearch_R2_sklearn.png")
plt.show()


