import sys,os
import numpy as np
import matplotlib.pyplot as plt
from IsingData import *
from misclib import *
from NeuralNetwork import *

# verify folders
if not os.path.isdir("./output"):                  os.mkdir("./output")
if not os.path.isdir("./output/NNClass"):          os.mkdir("./output/NNClass")
if not os.path.isdir("./output/NNClass/networks"): os.mkdir("./output/NNClass/networks")

"""
Program for training a neural network, based on parameters in config .dat file.
The network contains a single hiddel layer and is built for regression analysis
of 1-dimensional Ising data.
"""

# program parameters
r            = 0.3       # training portion of samples
config       = "test"    # name of network config file
save_network = True      # whether to save trained network

# setup data
data = IsingData.two_dim(data_config='noncrit')
data.transpose()
data.split(r)
data.make_hot()

# load neural network config
N_nodes,sigma_w,b0,eta,alpha,epochs,N_batch,lmbda = np.loadtxt("./output/NNClass/networks/{:s}.dat".format(config),unpack=True)
N_nodes,epochs,N_batch = int(N_nodes),int(epochs),int(N_batch)

# setup network
Classifier = MLPClassifier(data.X_train.shape[0],network_dir="./output/NNClass/networks/")
Classifier.add_layer(N_nodes,"tanh",           hidden=True, std_weights=sigma_w,const_bias=b0)
Classifier.add_layer(2,      "softmax_in_loss",hidden=False,std_weights=sigma_w,const_bias=b0)
Classifier.init_network()
if lmbda > 0: Classifier.add_penalty(lmbda)

# train network
Classifier.train(data.X_train,data.Y_train,epochs,N_batch,eta,alpha,track_cost=[data.X_test,data.Y_test],one_hot=True)

# evaluate network
plt.plot(np.arange(len(Classifier.accuracy_train)),Classifier.accuracy_train,marker='.',label="train")
plt.plot(np.arange(len(Classifier.accuracy_test)), Classifier.accuracy_test, marker='.',label="test")
plt.title("Evolution of accuracy score for classification neural network")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.legend()
plt.savefig("./output/NNClass/Accuracy_Evolution.png")
plt.show()

# store network
if save_network:
  Classifier.save_network(config)


