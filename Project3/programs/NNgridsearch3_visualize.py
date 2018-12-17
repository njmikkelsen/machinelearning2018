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
tag = sys.argv[1]
L   = int(sys.argv[2])
N   = int(sys.argv[3])

# load data
[[train_size],[Accuracy,Epochs,Timing]] = np.load("../results/NN/npy/gridsearch3_{:d}_{:d}_{:s}.npy".format(L,N,tag))

# plot thresholds
epochs_max = 5000
timing_max = 100

Epochs[Epochs > epochs_max] = None
Timing[Timing > timing_max] = None

# plot results as images
labels = ["logistic","tanh","relu"]
cs     = ["r","b","g"]  

fig,axes = plt.subplots(nrows=3,ncols=1,sharex=True,figsize=(5,9.5))

for i in range(3):
  axes[0].plot(100*train_size,Accuracy[:,i],label=labels[i],c=cs[i])
  axes[1].plot(100*train_size,Epochs[:,i],  label=labels[i],c=cs[i])
  axes[2].plot(100*train_size,Timing[:,i],  label=labels[i],c=cs[i])

axes[0].legend(loc='best')
axes[1].legend(loc='best')
axes[2].legend(loc='best')

#fig.suptitle("{:d} layers w/ {:d} nodes per layer".format(L,N),x=0.5,y=0.99,fontsize=18)
#fig.suptitle("simple networks".format(L,N),x=0.5,y=0.99,fontsize=18)
fig.suptitle("complex networks".format(L,N),x=0.5,y=0.99,fontsize=18)

axes[2].set_xlabel("training set size [%]",fontsize=18)
axes[0].set_ylabel("accuracy scores",      fontsize=18)
axes[1].set_ylabel("number of epochs",     fontsize=18)
axes[2].set_ylabel("training times [sec]", fontsize=18)

axes[2].set_xlim([10,90])

plt.tight_layout()
fig.subplots_adjust(top=0.94)

plt.savefig("../results/NN/img/gridsearch3_{:d}_{:d}_{:s}.png".format(L,N,tag))
plt.show()


