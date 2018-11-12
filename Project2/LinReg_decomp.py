import os,shutil
from IsingData import *
from LinearRegression import *
from plotlib import LinRegPlot_BiasVarDecomposition
import numpy as np
import matplotlib.pyplot as plt

# verify output folder
if not os.path.isdir("./output/"):       os.mkdir("./output")
if not os.path.isdir("./output/LinReg"): os.mkdir("./output/LinReg")

"""
This program analyses the 1-dimensional Ising data using either of the linear regression methods:
OLS, Ridge or Lasso with various regularization strengths.
The bias and variance of the models are estimated using the Bootstrap resampling technique
and plotted as functions of the regularization parameter lambda.
"""

# program parameters
N             = 40     # dimensionality of spin system
J             = 1      # coupling strength
M             = 10000  # number of samples
r             = 0.3    # training portion of samples
log_lmbda_min = -3.0   # log(min regularization)
log_lmbda_max = +5.0   # log(max regularization)
N_lmbda       = 15     # number of regularization parameters
B             = 30     # number of bootstrap cycles
regressor     = 2      # 0: OLS, 1: Ridge, 2: Lasso

# setup data & regularization
data = IsingData.one_dim(N,J,M)
data.split(r)
if regressor == 0: Lmbda = [0]
else:              Lmbda = np.logspace(log_lmbda_min,log_lmbda_max,N_lmbda)

# setup regressor object
if   regressor == 0: LinReg = OLSRegressor(  std=False)
elif regressor == 1: LinReg = RidgeRegressor(std=False)
elif regressor == 2: LinReg = LassoRegressor(std=False)

# precompute SVD
if regressor < 2: svd_train,svd_test = SVD(data.X_train),SVD(data.X_test)

# setup Bootstrap object
Bootstrap = BootstrapResampler(B)

# prepare arrays
LinReg.MSE_train   = np.zeros(len(Lmbda))
LinReg.Bias2_train = np.zeros(len(Lmbda))
LinReg.Var_train   = np.zeros(len(Lmbda))

LinReg.MSE_test   = np.zeros(len(Lmbda))
LinReg.Bias2_test = np.zeros(len(Lmbda))
LinReg.Var_test   = np.zeros(len(Lmbda))

# run Bootstrap
progress = progress_bar(2*len(Lmbda))
for i,lmbda in enumerate(Lmbda):
  # run training set
  if regressor < 2: LinReg.enter_SVD(*svd_train)
  LinReg.MSE_train[i],LinReg.Bias2_train[i],LinReg.Var_train[i] = Bootstrap.run_resampling(LinReg,data.X_train,data.Y_train,lmbda)
  progress.update()
  
  # run test set
  if regressor < 2: LinReg.enter_SVD(*svd_test)
  LinReg.MSE_test[i],LinReg.Bias2_test[i],LinReg.Var_test[i] = Bootstrap.run_resampling(LinReg,data.X_test,data.Y_test,lmbda)
  progress.update()

# print results to file
np.savetxt("./output/LinReg/BiasVarDecomp_{:s}.dat".format(LinReg.regressor_type),
           np.c_[LinReg.MSE_train,LinReg.Bias2_train,LinReg.Var_train,LinReg.MSE_test,LinReg.Bias2_test,LinReg.Var_test],
           fmt="%+8.6e | %+8.6e | %+8.6e | %+8.6e | %+8.6e | %+8.6e",
           header="               training set                 |                   test set\n" + \
                  "    MSE     |     Bias^2    |      Var      |      MSE      |     Bias^2    |      Var")

# display results
if regressor == 0:
  print("training set:")
  print("  MSE    =",LinReg.MSE_train[0])
  print("  Bias^2 =",LinReg.Bias2_train[0])
  print("  Var    =",LinReg.Var_train[0])
  print("\ntest set:")
  print("  MSE    =",LinReg.MSE_test[0])
  print("  Bias^2 =",LinReg.Bias2_test[0])
  print("  Var    =",LinReg.Var_test[0])
else:
  fig1,ax1 = LinRegPlot_BiasVarDecomposition(LinReg,Lmbda,loglog=False)
  fig2,ax2 = LinRegPlot_BiasVarDecomposition(LinReg,Lmbda,loglog=True)

  fig1.savefig("./output/LinReg/BiasVarDecomp_{:s}_linlog.png".format(LinReg.regressor_type))
  fig2.savefig("./output/LinReg/BiasVarDecomp_{:s}_loglog.png".format(LinReg.regressor_type))

  plt.show()


