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
r        = 0.3       # training portion of samples
network  = "test"    # name of network config file

# setup data
data = IsingData.two_dim(data_config='critical')
data.transpose()
data.make_hot()

# setup neural network
Classifier = MLPClassifier.load_network("./output/NNClass/networks/{:s}.pkl".format(network))

# evaluate classifier with respect to 'critical' data set
prediction = Classifier.predict(data.X)
Accuracy   = accuracy(np.argmax(data.T,axis=0),prediction)
print("accuracy(critical) = {:f}".format(Accuracy))


