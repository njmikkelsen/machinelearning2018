import os
import numpy as np
import matplotlib.pyplot as plt
from IsingData import *
from misclib import *
from plotlib import NNRegPlot_Hyper
from NeuralNetwork import *

# verify folders
if not os.path.isdir("./output"):                os.mkdir("./output")
if not os.path.isdir("./output/NNReg"):          os.mkdir("./output/NNReg")
if not os.path.isdir("./output/NNReg/networks"): os.mkdir("./output/NNReg/networks")

"""
Program for analysing the regression neural network with respect to hyperparameters
learning rate and regularization.
"""

# program parameters
N             = 40        # dimensionality of spin system
J             = 1         # coupling strength
M             = 10000     # number of samples
r             = 0.3       # training portion of samples
config        = "hyper1"  # name of network config file
save_name     = "logit"   # matplotlib savefig name
save_idx      = 3         # matplotlib savefig index
wide_plot     = True      # matplotlib wide plot
log_eta_min   = -2.0      # log(min regularization)
log_eta_max   = -1.0      # log(max regularization)
N_eta         = 8         # number of regularization parameters
log_lmbda_min = -4        # log(min regularization)
log_lmbda_max = -2        # log(max regularization)
N_lmbda       = 12        # number of regularization parameters

# setup data, regularization & learning rate
data = IsingData.one_dim(N,J,M)
data.split(r)
data.transpose()
data.Y_train,data.Y_test = data.Y_train[:,None].T,data.Y_test[:,None].T

Lmbda = np.logspace(log_lmbda_min,log_lmbda_max,N_lmbda)
Eta   = np.logspace(log_eta_min,  log_eta_max,  N_eta)

# load neural network config
activation,N_nodes,sigma_w,b0,eta0,alpha,epochs,N_batch,lmbda = np.loadtxt("./output/NNReg/networks/{:s}.dat".format(config),unpack=True)
activation,N_nodes,epochs,N_batch = ["logit","tanh"][int(activation)],int(N_nodes),int(epochs),int(N_batch)

# setup network
Regressor = MLPRegressor(data.X_train.shape[0],network_dir="./output/NNReg/networks/")
Regressor.add_layer(N_nodes,activation,std_weights=sigma_w,const_bias=b0)
Regressor.add_layer(1,"identity",hidden=False,std_weights=sigma_w,const_bias=b0)

# setup arrays
Y_train  = data.Y_train.flatten()
Y_test   = data.Y_test.flatten()
R2_train = np.zeros((N_eta,N_lmbda))
R2_test  = np.zeros((N_eta,N_lmbda))

# train network
for i,eta in enumerate(Eta):
  print("eta = {:4.3e}   ({:d}/{:d})".format(eta,i+1,N_eta))
  for j,lmbda in enumerate(Lmbda):
    # prepare
    Regressor.init_network()
    Regressor.add_penalty(lmbda)
    # train
    Regressor.train(data.X_train,data.Y_train,epochs,N_batch,eta,0)
    # evaluate
    y_train       = Regressor.predict(data.X_train).flatten()
    y_test        = Regressor.predict(data.X_test).flatten()
    R2_train[i,j] = R2(Y_train,y_train)
    R2_test[i,j]  = R2(Y_test, y_test)
    
# plot results
fig = NNRegPlot_Hyper1(0,R2_train,R2_test,Lmbda,Eta,wide=wide_plot)

plt.savefig("./output/NNReg/R2Scores_{:s}_{:d}.png".format(save_name,save_idx))
plt.show()


