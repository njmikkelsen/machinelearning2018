import os
import numpy as np
import matplotlib.pyplot as plt
from IsingData import *
from misclib import *
from plotlib import NNRegPlot_CostEvolution
from NeuralNetwork import *

# verify folders
if not os.path.isdir("./output"):                os.mkdir("./output")
if not os.path.isdir("./output/NNReg"):          os.mkdir("./output/NNReg")
if not os.path.isdir("./output/NNReg/networks"): os.mkdir("./output/NNReg/networks")

"""
Program for training a neural network, based on parameters in config .dat file.
The network contains a single hiddel layer and is built for regression analysis
of 1-dimensional Ising data.
"""

# program parameters
N            = 40        # dimensionality of spin system
J            = 1         # coupling strength
M            = 10000     # number of samples
r            = 0.3       # training portion of samples
config       = "tanh1"   # name of network config file
save_network = False     # whether to save trained network
loglog       = True      # loglog axes of cost function evolution plot

# setup data
data = IsingData.one_dim(N,J,M)
data.split(r)
data.transpose()
data.Y_train,data.Y_test = data.Y_train[:,None].T,data.Y_test[:,None].T

# load neural network config
activation,N_nodes,sigma_w,b0,eta0,alpha,epochs,N_batch,lmbda = \
np.loadtxt("./output/NNReg/networks/{:s}.dat".format(config),unpack=True)
activation,N_nodes,epochs,N_batch = ["logit","tanh"][int(activation)],int(N_nodes),int(epochs),int(N_batch)

# setup network
Regressor = MLPRegressor(data.X_train.shape[0],network_dir="./output/NNReg/networks/")
Regressor.add_layer(N_nodes,activation,std_weights=sigma_w,const_bias=b0)
Regressor.add_layer(1,"identity",hidden=False,std_weights=sigma_w,const_bias=b0)
Regressor.init_network()
if lmbda > 0: Regressor.add_penalty(lmbda)

# train network
Regressor.train(data.X_train,data.Y_train,epochs,N_batch,eta0,alpha,track_cost=[data.X_test,data.Y_test])

# evaluate network
Y_train = data.Y_train.flatten()
Y_test  = data.Y_test.flatten()
y_train = Regressor.predict(data.X_train).flatten()
y_test  = Regressor.predict(data.X_test).flatten()

R2_train = R2(Y_train,y_train)
R2_test  = R2(Y_test, y_test)

print("R2 train =",R2_train)
print("R2 test  =",R2_test)

# write test results to file
np.savetxt("./output/NNReg/{:s}_results.dat".format(config),np.c_[Y_test,y_test],fmt="%+12.10e  |  %+12.10e", 
           header="\nNetwork '{:s}' prediction results.  R2 = {:e}\n\n     True              Prediction".format(config,R2_test))

# plot
fig = NNRegPlot_CostEvolution(Regressor,loglog)
fig.savefig("./output/NNReg/CostEvolution_{:s}.png".format(config))
plt.show()

# store network
if save_network:
  Regressor.save_network(config)


