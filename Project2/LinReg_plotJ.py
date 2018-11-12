import sys,os
from IsingData import *
from LinearRegression import *
from misclib import *
from plotlib import LinRegPlot_Jmatrix
import numpy as np
import matplotlib.pyplot as plt

# verify output folder
if not os.path.isdir("./output/"):       os.mkdir("./output")
if not os.path.isdir("./output/LinReg"): os.mkdir("./output/LinReg")

"""
This program plots the resulting J-matrix from an OLS, Ridge or Lasso regression with regularization lambda.
"""

# program parameters
N          = 40      # dimensionality of spin system
J          = 1       # coupling strength
M          = 10000   # number of samples
r          = 0.3     # training portion of samples
plot_OLS   = False   # whether to plot OLS   coefficient
plot_Ridge = False   # whether to plot Ridge coefficient
plot_Lasso = True    # whether to plot Lasso coefficient
lmbda_R    = 1.0e4   # Ridge regularization parameter strength
lmbda_L    = 8.5e-2  # Lasso regularization parameter strength
lmbda_acc  = 1       # floating point accuracy of lambda in matplotlib title
cmap       = 0       # 0: seismic, 1: tab20
idx_save   = 4       # index for saved figure

# failsafe
if plot_OLS == plot_Ridge == plot_Lasso == False:
  print("Error: at least one of 'plot_OLS', 'plot_Ridge' or 'plot_Lasso' must be True")
  sys.exit(1)

# setup data
data = IsingData.one_dim(N,J,M)
data.split(r)

# precomputed SVD of training data matrix
if plot_OLS or plot_Ridge:
  U,s,Vh = SVD(data.X_train)

# preparation
J_matrix  = IsingData.one_dim.J_vec_to_matrix
plot      = [plot_OLS,plot_Ridge,plot_Lasso]
lmbda     = [0,lmbda_R,lmbda_L]
cmaps     = ["seismic","tab20"]
cmap_args = dict(vmin=-1.,vmax=1.,cmap=cmaps[cmap])
Regressor = [OLSRegressor,RidgeRegressor,LassoRegressor]
regressor = ["OLS","Ridge","Lasso"]

# run and plot
for i in range(3):
  if plot[i]:
    # setup regressor
    LinReg = Regressor[i](std=False)
    if i in [0,1]: LinReg.enter_SVD(U,s,Vh)
    # learn J
    LinReg.fit(data.X_train,data.Y_train,alpha=lmbda[i])
    LinReg.J = J_matrix(LinReg.coeff)
    # plot J matrix
    fig = LinRegPlot_Jmatrix(LinReg.J,i,regressor[i],cmap_args,lmbda[i],lmbda_acc)

    plt.savefig("./output/LinReg/Jmatrix_{:s}{:d}.png".format(regressor[i],idx_save))
    plt.show()

