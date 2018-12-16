import sys,os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

"""
Program for analysing how neural networks and the pulsar data set behave
with respect to hyperparameter:
  - train_size | train-to-test ratio in division of data into training and test sets

NOTE: This program loads results produced by the program 'NNgridsearch3_visualize.py'.
  
On the structure of the neural networks:
  All hidden layers have the same number of nodes.
  All hidden layers have the same activation function.

On the performance metrics:
  The performance of the networks will be judged according to 3 metrics:
    1) final accuracy score
    2) required number of training epochs
    3) training time

On the program's structure:
  The program both creates and saves output plots based loaded data.

Program parameters:
  L               | number of layers
  N               | number of nodes per layer
  gamma0          | initial learning rate
  eta             | strength of learning momentum
  tag             | extra save tag used in np.save/np.load of results
  production_path | path for produced results
"""

# data parameters
load_production = True
tag             = "1"
L               = 5
N               = 20
gamma0          = 0.001000
eta             = 0.900000

# load data
[[train_size],[Accuracy,Epochs,Timing]] = np.load("../results/NN/npy/gridsearch3_{:d}_{:d}_{:f}_{:f}_{:s}.npy".format(L,N,gamma0,eta,tag))

# plot results as images
labels = ["logistic","tanh","relu"]
cs     = ["r","b","g"]

fig,axes = plt.subplots(nrows=3,ncols=1,sharex=True,figsize=(5,9.5))

for i in range(3):
  axes[0].plot(train_size,Accuracy[i],label=labels[i],c=cs[i])
  axes[0].plot(train_size,Epochs[i],  label=labels[i],c=cs[i])
  axes[0].plot(train_size,Timing[i],  label=labels[i],c=cs[i])

fig.suptitle("Neural network w/ {:s} activation".format(activation),fontsize=18)

axes[2].set_xlabel("train-to-test set ratio",fontsize=18)
axes[0].set_ylabel("accuray",                fontsize=18)
axes[1].set_ylabel("epochs",                 fontsize=18)
axes[2].set_ylabel("timing",                 fontsize=18)

plt.tight_layout()
fig.subplots_adjust(top=0.92)

plt.savefig("../results/NN/img/gridsearch3_{:d}_{:d}_{:s}{:s}.png".format(L,N,activation,tag))
plt.show()



