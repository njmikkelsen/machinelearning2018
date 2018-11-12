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
config        = "hyper2"  # name of network config file
save_name     = "logit"   # matplotlib savefig name
save_idx      = 4         # matplotlib savefig index
wide_plot     = True      # matplotlib wide plot
epochs_min    = 10        # minimum "number of epochs"
epochs_max    = 30        # maximum "number of epochs"
N_epochs      = 5         # number of "number of epochs"
alpha_min     = 1e-4      # minimum falloff parameter
alpha_max     = 1e+0      # maximum falloff parameter
N_alpha       = 5        # number of falloff parameters

# setup data, regularization & learning rate
data = IsingData.one_dim(N,J,M)
data.split(r)
data.transpose()
data.Y_train,data.Y_test = data.Y_train[:,None].T,data.Y_test[:,None].T

Epochs = np.linspace(epochs_min,epochs_max,N_epochs,dtype=np.int8)
Alpha  = np.linspace(alpha_min, alpha_max, N_alpha)

# load neural network config
activation,N_nodes,sigma_w,b0,eta,alpha0,epochs0,N_batch,lmbda = np.loadtxt("./output/NNReg/networks/{:s}.dat".format(config),unpack=True)
activation,N_nodes,N_batch = ["logit","tanh"][int(activation)],int(N_nodes),int(N_batch)

# setup network
Regressor = MLPRegressor(data.X_train.shape[0],network_dir="./output/NNReg/networks/")
Regressor.add_layer(N_nodes,activation,std_weights=sigma_w,const_bias=b0)
Regressor.add_layer(1,"identity",hidden=False,std_weights=sigma_w,const_bias=b0)

# setup arrays
Y_train  = data.Y_train.flatten()
Y_test   = data.Y_test.flatten()
R2_train = np.zeros((N_epochs,N_alpha))
R2_test  = np.zeros((N_epochs,N_alpha))

# train network
for i,epochs in enumerate(Epochs):
  print("epochs = {:d}   ({:d}/{:d})".format(epochs,i+1,N_epochs))
  for j,alpha in enumerate(Alpha):
    # prepare
    Regressor.init_network()
    Regressor.add_penalty(lmbda)
    # train
    Regressor.train(data.X_train,data.Y_train,epochs,N_batch,eta,alpha)
    # evaluate
    y_train       = Regressor.predict(data.X_train).flatten()
    y_test        = Regressor.predict(data.X_test).flatten()
    R2_train[i,j] = R2(Y_train,y_train)
    R2_test[i,j]  = R2(Y_test, y_test)
    
# plot results
fig = NNRegPlot_Hyper(1,R2_train,R2_test,Epochs,Alpha)

plt.savefig("./output/NNReg/R2Scores_{:s}_{:d}.png".format(save_name,save_idx))
plt.show()


